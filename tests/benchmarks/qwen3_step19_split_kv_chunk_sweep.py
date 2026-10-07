"""Sweep MPK split-KV chunk sizes for the B32 long-context case."""

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
BATCH = 32
S_IN = 1024
S_OUT = 128
MAX_SEQ = 1280
COMPARE_TOKENS = 10
FIELDS = (
    "chunk_size", "status", "minimum_first10_matches", "passing_requests",
    "invalid_token_count", "incomplete_requests", "resolved_attention",
    "prefill_ms", "decode_ms", "decode_step_ms", "decode_tokens_per_second",
    "speedup_vs_chunk128", "kernel_cache_status", "kernel_prepare_ms",
    "output_path", "log_path", "reason",
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
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
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


def command(args, output, cache=None, chunk=128, batch=BATCH):
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
            "--mpk-split-kv-chunk-size", str(chunk),
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1",
            "--normal-prefill-cuda-graph",
        ]
    return result


def write(args, rows):
    baseline = next(
        (row["decode_ms"] for row in rows if row["chunk_size"] == 128 and row["status"] == "completed"),
        None,
    )
    for row in rows:
        row["speedup_vs_chunk128"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None
        )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    complete = len(rows) == len(args.chunk_sizes)
    passed = complete and all(row["status"] == "completed" for row in rows)
    valid = [row for row in rows if row["status"] == "completed"]
    best = min(valid, key=lambda row: row["decode_step_ms"]) if valid else None
    summary = {
        "step": 19, "phase": "split_kv_chunk_sweep",
        "status": "passed" if passed else "failed",
        "model": args.model, "batch_size": BATCH,
        "s_in": S_IN, "s_out": S_OUT, "max_seq_length": MAX_SEQ,
        "warmup_runs": 1, "measured_runs": 1,
        "correctness_gate": (
            "Every request matches the Torch reference for its first 10 generated tokens, "
            "has no invalid token, and generates all 128 requested tokens."
        ),
        "best_chunk_size": best["chunk_size"] if best else None,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for chunk in args.chunk_sizes:
        if chunk <= 0 or MAX_SEQ % chunk:
            parser.error(f"chunk size {chunk} must divide max sequence length {MAX_SEQ}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "cache"
    cache_root.mkdir(exist_ok=True)

    reference_output = args.output_dir / "torch.json"
    reference_log = args.output_dir / "torch.log"
    if reference_output.is_file():
        reference = json.loads(reference_output.read_text(encoding="utf-8"))
    else:
        print("Running Torch correctness reference...", flush=True)
        reference, error = run(
            command(args, reference_output, batch=1), reference_output,
            reference_log, args.timeout,
        )
        if reference is None:
            raise RuntimeError(f"Torch reference failed: {error}; see {reference_log}")
    expected = reference["token_ids"][:COMPARE_TOKENS]

    summary_path = args.output_dir / "summary.json"
    rows = []
    if summary_path.is_file():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [row for row in old.get("rows", []) if row.get("status") == "completed"]
        print(f"Resuming with {len(rows)} completed chunk size(s).", flush=True)
    completed = {int(row["chunk_size"]) for row in rows}

    failures = 0
    for chunk in args.chunk_sizes:
        if chunk in completed:
            print(f"Skipping completed chunk={chunk}.", flush=True)
            continue
        case_dir = args.output_dir / f"chunk_{chunk}"
        case_dir.mkdir(exist_ok=True)
        cache = cache_root / f"chunk_{chunk}"
        warm_output = case_dir / "warmup.json"
        warm_log = case_dir / "warmup.log"
        output = case_dir / "tokens.json"
        log = case_dir / "run.log"
        print(f"Warmup/compile chunk={chunk}...", flush=True)
        warm, error = run(
            command(args, warm_output, cache, chunk), warm_output,
            warm_log, args.timeout,
        )
        data = None
        failure_log = warm_log
        if warm is not None:
            print(f"Measuring chunk={chunk}...", flush=True)
            data, error = run(
                command(args, output, cache, chunk), output, log, args.timeout,
            )
            failure_log = log
        reasons = [error] if error else []
        matches = []
        invalid = incomplete = None
        if data is not None:
            tokens = data.get("token_ids_by_request", [])
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            matches = [
                sum(a == b for a, b in zip(expected, actual[:COMPARE_TOKENS]))
                for actual in tokens
            ]
            invalid = sum(invalid_counts) if len(invalid_counts) == BATCH else None
            incomplete = sum(length != S_OUT for length in lengths) if len(lengths) == BATCH else None
            if len(tokens) != BATCH or not matches or min(matches) != COMPARE_TOKENS:
                reasons.append(f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
        decode_ms = data.get("decode_time_ms") if data else None
        steps = data.get("decode_steps") if data else None
        row = {
            "chunk_size": chunk,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": min(matches) if matches else None,
            "passing_requests": sum(value == COMPARE_TOKENS for value in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "prefill_ms": data.get("prefill_time_ms") if data else None,
            "decode_ms": decode_ms,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "decode_tokens_per_second": (
                1000.0 * BATCH * steps / decode_ms if decode_ms and steps else None
            ),
            "speedup_vs_chunk128": None,
            "kernel_cache_status": data.get("mpk_kernel_cache_status") if data else None,
            "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
            "output_path": str(output), "log_path": str(failure_log),
            "reason": "; ".join(reasons),
        }
        rows = [item for item in rows if int(item["chunk_size"]) != chunk]
        rows.append(row)
        rows.sort(key=lambda item: int(item["chunk_size"]))
        failures += bool(reasons)
        write(args, rows)
        print(
            f"chunk={chunk}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['decode_tokens_per_second']}",
            flush=True,
        )
    write(args, rows)
    print(f"Step 19 split-KV chunk sweep completed with {failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
