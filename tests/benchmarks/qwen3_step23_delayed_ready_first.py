"""Compare FIFO with delayed ready-first MPK worker-local scheduling."""

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
BATCH, S_IN, S_OUT, MAX_SEQ, CHUNK = 32, 1024, 128, 1280, 256
POLICIES = ("fifo", "delayed-ready-first")
COMPARE = 10
FIELDS = (
    "policy", "status", "minimum_first10_matches", "passing_requests",
    "invalid_token_count", "incomplete_requests", "prefill_ms", "decode_ms",
    "decode_step_ms", "decode_tokens_per_second", "speedup_vs_fifo",
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


def command(args, output, batch, cache=None, policy="fifo"):
    result = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-seq-length", str(MAX_SEQ),
        "--max-new-tokens", str(S_OUT), "--page-size", str(MAX_SEQ),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if cache is not None:
        result += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "split-kv",
            "--mpk-split-kv-chunk-size", str(CHUNK),
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", policy,
            "--mpk-ready-first-spin-iters", str(args.spin_iters),
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
    return result


def write(args, rows):
    baseline = next((r["decode_ms"] for r in rows
                     if r["policy"] == "fifo" and
                     r["status"] == "completed"), None)
    for row in rows:
        row["speedup_vs_fifo"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None)
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = (len(rows) == len(POLICIES) and
              all(r["status"] == "completed" for r in rows))
    completed = [r for r in rows if r["status"] == "completed"]
    best = min(completed, key=lambda r: r["decode_step_ms"]) if completed else None
    summary = {
        "step": 23,
        "phase": "delayed_ready_first_ablation",
        "status": "passed" if passed else "failed",
        "model": args.model,
        "batch_size": BATCH,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ,
        "split_kv_chunk_size": CHUNK,
        "ready_first_spin_iters": args.spin_iters,
        "warmup_runs": 1,
        "measured_runs": 1,
        "correctness_gate": "All 32 requests match Torch for the first 10 tokens, contain no invalid tokens, and generate 128 tokens.",
        "best_policy": best["policy"] if best else None,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--spin-iters", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_path, ref_log = args.output_dir / "torch.json", args.output_dir / "torch.log"
    print("Running Torch correctness reference...", flush=True)
    reference, error = run(
        command(args, ref_path, 1), ref_path, ref_log, args.timeout)
    if reference is None:
        raise RuntimeError(f"Torch reference failed: {error}; see {ref_log}")
    expected = reference["token_ids"][:COMPARE]

    rows, failures = [], 0
    for policy in POLICIES:
        case_dir = args.output_dir / policy.replace("-", "_")
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_output, output = case_dir / "warmup.json", case_dir / "tokens.json"
        warm_log, log = case_dir / "warmup.log", case_dir / "run.log"
        print(f"Warmup/compile policy={policy}...", flush=True)
        warm, error = run(
            command(args, warm_output, BATCH, cache, policy), warm_output,
            warm_log, args.timeout)
        data, failure_log = None, warm_log
        if warm is not None:
            print(f"Measuring policy={policy}...", flush=True)
            data, error = run(
                command(args, output, BATCH, cache, policy), output, log,
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
            if len(tokens) != BATCH or not matches or min(matches) != COMPARE:
                reasons.append(f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            if data.get("mpk_worker_policy") != policy:
                reasons.append(
                    f"reported worker policy={data.get('mpk_worker_policy')}")
        decode_ms = data.get("decode_time_ms") if data else None
        steps = data.get("decode_steps") if data else None
        row = {
            "policy": policy,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": min(matches) if matches else None,
            "passing_requests": sum(m == COMPARE for m in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "prefill_ms": data.get("prefill_time_ms") if data else None,
            "decode_ms": decode_ms,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "decode_tokens_per_second": (
                1000.0 * BATCH * steps / decode_ms if decode_ms and steps else None),
            "speedup_vs_fifo": None,
            "kernel_cache_status": data.get("mpk_kernel_cache_status") if data else None,
            "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
            "output_path": str(output), "log_path": str(failure_log),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        write(args, rows)
        print(
            f"policy={policy}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['decode_tokens_per_second']}", flush=True)
    write(args, rows)
    print(f"Step 23 delayed ready-first ablation completed with {failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
