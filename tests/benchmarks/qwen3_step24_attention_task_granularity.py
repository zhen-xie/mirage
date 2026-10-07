"""Measure MPK attention task granularity at B32 long context."""

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
BATCH, S_IN, S_OUT, MAX_SEQ = 32, 1024, 128, 1280
COMPARE = 10
MODES = (
    ("default", None),
    ("split_2", 640),
    ("split_4", 320),
    ("split_5", 256),
)
FIELDS = (
    "mode", "chunk_size", "compiled_splits", "status",
    "minimum_first10_matches", "passing_requests", "invalid_token_count",
    "incomplete_requests", "resolved_attention", "prefill_ms", "decode_ms",
    "decode_step_ms", "decode_tokens_per_second", "speedup_vs_split_5",
    "kernel_cache_status", "kernel_prepare_ms", "output_path", "log_path",
    "reason",
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


def command(args, output, batch, cache=None, chunk=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-seq-length", str(MAX_SEQ),
        "--max-new-tokens", str(S_OUT), "--page-size", str(MAX_SEQ),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if cache is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "default" if chunk is None else "split-kv",
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", "fifo",
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
        if chunk is not None:
            cmd += ["--mpk-split-kv-chunk-size", str(chunk)]
    return cmd


def write(args, rows):
    baseline = next(
        (r["decode_ms"] for r in rows
         if r["mode"] == "split_5" and r["status"] == "completed"), None)
    for row in rows:
        row["speedup_vs_split_5"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None)
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = (len(rows) == len(MODES) and
              all(r["status"] == "completed" for r in rows))
    completed = [r for r in rows if r["status"] == "completed"]
    best = min(completed, key=lambda r: r["decode_step_ms"]) if completed else None
    summary = {
        "step": 24,
        "phase": "attention_task_granularity",
        "status": "passed" if passed else "failed",
        "model": args.model,
        "batch_size": BATCH,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ,
        "num_workers_target_note": (
            "B32 already provides batch*KV-head parallelism; this ablation "
            "tests whether extra KV splits and merge work are beneficial."
        ),
        "warmup_runs": 1,
        "measured_runs": 1,
        "correctness_gate": (
            "Every request matches Torch for the first 10 generated tokens, "
            "has no invalid token, and generates all 128 requested tokens."
        ),
        "best_mode": best["mode"] if best else None,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.output_dir / "torch.json"
    reference_log = args.output_dir / "torch.log"
    print("Running Torch correctness reference...", flush=True)
    reference, error = run(
        command(args, reference_path, 1), reference_path,
        reference_log, args.timeout)
    if reference is None:
        raise RuntimeError(f"Torch reference failed: {error}; see {reference_log}")
    expected = reference["token_ids"][:COMPARE]

    rows, failures = [], 0
    for mode, chunk in MODES:
        case_dir = args.output_dir / mode
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_output, output = case_dir / "warmup.json", case_dir / "tokens.json"
        warm_log, log = case_dir / "warmup.log", case_dir / "run.log"
        print(f"Warmup/compile mode={mode}...", flush=True)
        warm, error = run(
            command(args, warm_output, BATCH, cache, chunk), warm_output,
            warm_log, args.timeout)
        data, failure_log = None, warm_log
        if warm is not None:
            print(f"Measuring mode={mode}...", flush=True)
            data, error = run(
                command(args, output, BATCH, cache, chunk), output, log,
                args.timeout)
            failure_log = log
        reasons = [error] if error else []
        matches = []
        invalid = incomplete = None
        if data is not None:
            tokens = data.get("token_ids_by_request", [])
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            matches = [sum(a == b for a, b in zip(expected, t[:COMPARE]))
                       for t in tokens]
            invalid = sum(invalid_counts) if len(invalid_counts) == BATCH else None
            incomplete = (sum(n != S_OUT for n in lengths)
                          if len(lengths) == BATCH else None)
            expected_attention = "default" if chunk is None else "split-kv"
            if data.get("mpk_attention") != expected_attention:
                reasons.append(f"attention={data.get('mpk_attention')}")
            if len(tokens) != BATCH or not matches or min(matches) != COMPARE:
                reasons.append(f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
        decode_ms = data.get("decode_time_ms") if data else None
        steps = data.get("decode_steps") if data else None
        row = {
            "mode": mode,
            "chunk_size": chunk,
            "compiled_splits": 1 if chunk is None else MAX_SEQ // chunk,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": min(matches) if matches else None,
            "passing_requests": sum(m == COMPARE for m in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "prefill_ms": data.get("prefill_time_ms") if data else None,
            "decode_ms": decode_ms,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "decode_tokens_per_second": (
                1000.0 * BATCH * steps / decode_ms if decode_ms and steps else None),
            "speedup_vs_split_5": None,
            "kernel_cache_status": data.get("mpk_kernel_cache_status") if data else None,
            "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
            "output_path": str(output), "log_path": str(failure_log),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        write(args, rows)
        print(
            f"mode={mode}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['decode_tokens_per_second']}", flush=True)
    write(args, rows)
    print(f"Step 24 task-granularity ablation completed with {failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
