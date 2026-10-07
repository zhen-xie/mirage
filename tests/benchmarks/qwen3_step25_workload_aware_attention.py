"""Validate workload-aware MPK attention selection across batch sizes."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
BATCHES = (1, 8, 32)
S_IN, S_OUT, MAX_SEQ = 1024, 128, 1280
COMPARE = 10
EXPECTED_AUTO = {
    1: ("split-kv", 128, 10),
    8: ("split-kv", 640, 2),
    32: ("default", None, None),
}
FIELDS = (
    "batch_size", "mode", "status", "resolved_attention", "chunk_size",
    "num_chunks", "base_tasks", "target_tasks", "target_splits",
    "minimum_first10_matches", "passing_requests", "invalid_token_count",
    "incomplete_requests", "prefill_ms", "decode_ms", "decode_step_ms",
    "decode_tokens_per_second", "speedup_vs_fixed_128", "cache_status",
    "kernel_prepare_ms", "output_path", "log_path", "reason",
)


def terminate(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(command, output, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return None, "timeout"
    if code:
        return None, f"exit code {code}"
    if not output.is_file():
        return None, "missing output JSON"
    return json.loads(output.read_text(encoding="utf-8")), ""


def command(args, output, batch, mode=None, cache=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-seq-length", str(MAX_SEQ),
        "--max-new-tokens", str(S_OUT), "--page-size", str(MAX_SEQ),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if mode is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto" if mode == "auto" else "split-kv",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-auto-attention-target-tasks", str(args.target_tasks),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", "fifo",
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
    return cmd


def write(args, rows):
    fixed = {
        row["batch_size"]: row["decode_ms"] for row in rows
        if row["mode"] == "fixed_128" and row["status"] == "completed"
    }
    for row in rows:
        baseline = fixed.get(row["batch_size"])
        row["speedup_vs_fixed_128"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None
        )
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = len(rows) == 2 * len(BATCHES) and all(
        row["status"] == "completed" for row in rows)
    summary = {
        "step": 25,
        "phase": "workload_aware_attention",
        "status": "passed" if passed else "failed",
        "model": args.model,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ,
        "batch_sizes": list(BATCHES),
        "target_attention_tasks": args.target_tasks,
        "minimum_split_chunk_size": 128,
        "warmup_runs": 1,
        "measured_runs": 1,
        "correctness_gate": (
            "Every request matches Torch for the first 10 tokens, contains "
            "no invalid token, and completes all 128 output tokens."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.output_dir / "torch.json"
    reference_log = args.output_dir / "torch.log"
    print("Running Torch correctness reference...", flush=True)
    reference, error = run(
        command(args, reference_path, 1), reference_path, reference_log,
        args.timeout)
    if reference is None:
        raise RuntimeError(f"Torch reference failed: {error}; see {reference_log}")
    expected_tokens = reference["token_ids"][:COMPARE]

    rows = []
    failures = 0
    for batch in BATCHES:
        for mode in ("fixed_128", "auto"):
            case_dir = args.output_dir / f"b{batch}_{mode}"
            cache = case_dir / "cache"
            case_dir.mkdir(exist_ok=True)
            cache.mkdir(exist_ok=True)
            warm_output = case_dir / "warmup.json"
            output = case_dir / "tokens.json"
            warm_log = case_dir / "warmup.log"
            log = case_dir / "run.log"
            print(f"Warmup/compile B={batch} mode={mode}...", flush=True)
            warm, error = run(
                command(args, warm_output, batch, mode, cache), warm_output,
                warm_log, args.timeout)
            data = None
            failure_log = warm_log
            if warm is not None:
                print(f"Measuring B={batch} mode={mode}...", flush=True)
                data, error = run(
                    command(args, output, batch, mode, cache), output, log,
                    args.timeout)
                failure_log = log

            reasons = [error] if error else []
            matches = []
            invalid = incomplete = None
            if data is not None:
                tokens = data.get("token_ids_by_request", [])
                lengths = data.get("generate_lengths_by_request", [])
                invalid_counts = data.get("invalid_token_counts_by_request", [])
                matches = [
                    sum(a == b for a, b in zip(expected_tokens, request[:COMPARE]))
                    for request in tokens
                ]
                invalid = (
                    sum(invalid_counts) if len(invalid_counts) == batch else None)
                incomplete = (
                    sum(length != S_OUT for length in lengths)
                    if len(lengths) == batch else None)
                if len(tokens) != batch or not matches or min(matches) != COMPARE:
                    reasons.append(
                        f"minimum first-10={min(matches) if matches else None}")
                if invalid != 0:
                    reasons.append(f"invalid tokens={invalid}")
                if incomplete != 0:
                    reasons.append(f"incomplete requests={incomplete}")

                if mode == "fixed_128":
                    wanted = ("split-kv", 128, 10)
                else:
                    wanted = EXPECTED_AUTO[batch]
                actual = (
                    data.get("mpk_attention"),
                    data.get("mpk_split_kv_chunk_size"),
                    data.get("mpk_split_kv_num_chunks"),
                )
                if actual != wanted:
                    reasons.append(f"attention plan={actual}, expected={wanted}")
                if mode == "auto" and data.get(
                        "mpk_auto_attention_target_tasks") != args.target_tasks:
                    reasons.append("wrong auto target task metadata")

            decode_ms = data.get("decode_time_ms") if data else None
            steps = data.get("decode_steps") if data else None
            row = {
                "batch_size": batch,
                "mode": mode,
                "status": "failed" if reasons else "completed",
                "resolved_attention": data.get("mpk_attention") if data else None,
                "chunk_size": data.get("mpk_split_kv_chunk_size") if data else None,
                "num_chunks": data.get("mpk_split_kv_num_chunks") if data else None,
                "base_tasks": data.get("mpk_auto_attention_base_tasks") if data else None,
                "target_tasks": data.get("mpk_auto_attention_target_tasks") if data else None,
                "target_splits": data.get("mpk_auto_attention_target_splits") if data else None,
                "minimum_first10_matches": min(matches) if matches else None,
                "passing_requests": sum(value == COMPARE for value in matches),
                "invalid_token_count": invalid,
                "incomplete_requests": incomplete,
                "prefill_ms": data.get("prefill_time_ms") if data else None,
                "decode_ms": decode_ms,
                "decode_step_ms": data.get("decode_step_time_ms") if data else None,
                "decode_tokens_per_second": (
                    1000.0 * batch * steps / decode_ms
                    if decode_ms and steps else None),
                "speedup_vs_fixed_128": None,
                "cache_status": data.get("mpk_kernel_cache_status") if data else None,
                "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
                "output_path": str(output),
                "log_path": str(failure_log),
                "reason": "; ".join(reasons),
            }
            rows.append(row)
            failures += bool(reasons)
            write(args, rows)
            print(
                f"B={batch} mode={mode}: "
                f"{'PASS' if not reasons else 'FAIL'}; "
                f"attention={row['resolved_attention']}; "
                f"chunk={row['chunk_size']}; first-10="
                f"{row['minimum_first10_matches']}; "
                f"decode={row['decode_step_ms']} ms/step; "
                f"throughput={row['decode_tokens_per_second']}",
                flush=True,
            )

    write(args, rows)
    print(
        f"Step 25 workload-aware attention completed with "
        f"{failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
