"""Compare two- and three-stage Hopper MPK attention KV pipelines."""

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
BATCH, S_IN, S_OUT, MAX_SEQ, COMPARE = 32, 1024, 128, 1280, 10
STAGES = (2, 3)
FIELDS = (
    "pipeline_stages", "status", "resolved_attention",
    "minimum_first10_matches", "minimum_full_matches_vs_stage2",
    "passing_requests", "invalid_token_count", "incomplete_requests",
    "prefill_ms", "decode_ms", "decode_step_ms",
    "decode_tokens_per_second", "speedup_vs_stage2", "cache_status",
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


def command(args, output, stages=None, cache=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-seq-length", str(MAX_SEQ),
        "--max-new-tokens", str(S_OUT), "--page-size", str(MAX_SEQ),
        "--max-num-pages", str(BATCH if stages is not None else 1),
        "--max-num-batched-requests", str(BATCH if stages is not None else 1),
        "--max-num-batched-tokens", str(BATCH if stages is not None else 8),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if stages is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-auto-attention-target-tasks", str(args.target_tasks),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", "fifo",
            "--mpk-attention-kv-pipeline-stages", str(stages),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    return cmd


def token_batches(data):
    return data.get("token_ids_by_request", [])


def write(args, rows):
    baseline = next((
        row["decode_ms"] for row in rows
        if row["pipeline_stages"] == 2 and row["status"] == "completed"
    ), None)
    for row in rows:
        row["speedup_vs_stage2"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None)
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = len(rows) == len(STAGES) and all(
        row["status"] == "completed" for row in rows)
    summary = {
        "step": 28,
        "phase": "attention_kv_pipeline_depth",
        "status": "passed" if passed else "failed",
        "model": args.model,
        "batch_size": BATCH,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ,
        "pipeline_stages": list(STAGES),
        "warmup_runs": 1,
        "measured_runs": 1,
        "correctness_gate": (
            "Every request matches Torch for the first 10 tokens, matches "
            "the two-stage control for all 128 tokens, contains no invalid "
            "token, and completes all output tokens."
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
        command(args, reference_path), reference_path, reference_log,
        args.timeout)
    if reference is None:
        raise RuntimeError(
            f"Torch reference failed: {error}; see {reference_log}")
    expected = reference["token_ids"][:COMPARE]

    rows = []
    stage2_tokens = None
    failures = 0
    for stages in STAGES:
        case_dir = args.output_dir / f"stages_{stages}"
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_output = case_dir / "warmup.json"
        output = case_dir / "tokens.json"
        warm_log = case_dir / "warmup.log"
        log = case_dir / "run.log"
        print(f"Warmup/compile pipeline stages={stages}...", flush=True)
        warm, error = run(
            command(args, warm_output, stages, cache), warm_output,
            warm_log, args.timeout)
        data = None
        failure_log = warm_log
        if warm is not None:
            print(f"Measuring pipeline stages={stages}...", flush=True)
            data, error = run(
                command(args, output, stages, cache), output, log,
                args.timeout)
            failure_log = log

        reasons = [error] if error else []
        matches = []
        full = invalid = incomplete = None
        if data is not None:
            tokens = token_batches(data)
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            matches = [
                sum(a == b for a, b in zip(expected, value[:COMPARE]))
                for value in tokens
            ]
            invalid = sum(invalid_counts) if len(invalid_counts) == BATCH else None
            incomplete = (
                sum(value != S_OUT for value in lengths)
                if len(lengths) == BATCH else None)
            if len(tokens) != BATCH or min(matches, default=-1) != COMPARE:
                reasons.append(
                    f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            if data.get("mpk_attention") != "default":
                reasons.append(f"attention={data.get('mpk_attention')}")
            if data.get("mpk_attention_kv_pipeline_stages") != stages:
                reasons.append("wrong pipeline-stage metadata")
            if stages == 2:
                stage2_tokens = tokens
                full = S_OUT
            elif stage2_tokens is not None and len(tokens) == BATCH:
                full = min(
                    sum(a == b for a, b in zip(base, value))
                    for base, value in zip(stage2_tokens, tokens))
                if full != S_OUT:
                    reasons.append(f"minimum full matches={full}/{S_OUT}")

        decode_ms = data.get("decode_time_ms") if data else None
        steps = data.get("decode_steps") if data else None
        row = {
            "pipeline_stages": stages,
            "status": "failed" if reasons else "completed",
            "resolved_attention": data.get("mpk_attention") if data else None,
            "minimum_first10_matches": min(matches) if matches else None,
            "minimum_full_matches_vs_stage2": full,
            "passing_requests": sum(value == COMPARE for value in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "prefill_ms": data.get("prefill_time_ms") if data else None,
            "decode_ms": decode_ms,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "decode_tokens_per_second": (
                1000.0 * BATCH * steps / decode_ms
                if decode_ms and steps else None),
            "speedup_vs_stage2": None,
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
            f"stages={stages}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"full={full}/{S_OUT}; decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['decode_tokens_per_second']}", flush=True)

    write(args, rows)
    print(
        f"Step 28 attention pipeline completed with {failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
