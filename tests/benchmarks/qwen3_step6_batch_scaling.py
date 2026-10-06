"""Measure adaptive MPK batch scaling with first-10 token validation."""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
COMPARE_TOKENS = 10
S_IN = 1024
S_OUT = 128
MAX_SEQ_LENGTH = 1152
CSV_FIELDS = (
    "batch_size", "status", "minimum_first10_matches", "passing_requests",
    "invalid_token_count", "incomplete_requests", "resolved_attention",
    "prefill_time_ms", "decode_time_ms", "decode_step_time_ms",
    "decode_tokens_per_second", "throughput_scale_vs_b1",
    "throughput_efficiency_vs_b1", "cache_status", "kernel_prepare_time_ms",
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


def execute(args, name, batch_size, mpk):
    output = args.output_dir / f"{name}.json"
    log = args.output_dir / f"{name}.log"
    command = [
        sys.executable, str(DEMO),
        "--model", args.model,
        "--input-length", str(S_IN),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--max-new-tokens", str(S_OUT),
        "--page-size", str(MAX_SEQ_LENGTH),
        "--max-num-pages", str(batch_size),
        "--max-num-batched-requests", str(batch_size),
        "--max-num-batched-tokens", str(max(8, batch_size)),
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mpk:
        cache = args.cache_dir / f"b{batch_size}_seq{MAX_SEQ_LENGTH}"
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    print(
        f"Running {name}: B={batch_size} S_IN={S_IN} S_OUT={S_OUT}",
        flush=True,
    )
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return {
                "status": "timeout", "reason": "timeout",
                "output": output, "log": log,
            }
    if returncode:
        return {
            "status": "runtime_failed", "reason": f"exit code {returncode}",
            "output": output, "log": log,
        }
    if not output.is_file():
        return {
            "status": "missing_output", "reason": "missing JSON",
            "output": output, "log": log,
        }
    return {
        "status": "completed", "data": json.loads(output.read_text()),
        "output": output, "log": log,
    }


def validate(result, reference_tokens, batch_size):
    if result["status"] != "completed":
        return result["status"], result.get("reason", result["status"]), {}
    data = result["data"]
    tokens_by_request = data.get("token_ids_by_request", [])
    lengths = data.get("generate_lengths_by_request", [])
    invalid_counts = data.get("invalid_token_counts_by_request", [])
    errors = []
    if len(tokens_by_request) != batch_size:
        errors.append("wrong token batch size")
        tokens_by_request = []
    if len(lengths) != batch_size:
        errors.append("wrong generation-length batch size")
        lengths = []
    if len(invalid_counts) != batch_size:
        errors.append("wrong invalid-token batch size")
        invalid_counts = []
    matches = [
        sum(a == b for a, b in zip(reference_tokens[:10], tokens[:10]))
        for tokens in tokens_by_request
    ]
    minimum_matches = min(matches) if matches else None
    passing_requests = sum(value == COMPARE_TOKENS for value in matches)
    invalid_total = sum(invalid_counts)
    incomplete_requests = sum(value != S_OUT for value in lengths)
    if minimum_matches != COMPARE_TOKENS:
        errors.append(f"minimum first-10={minimum_matches}/10")
    if passing_requests != batch_size:
        errors.append(f"passing requests={passing_requests}/{batch_size}")
    if invalid_total:
        errors.append(f"invalid tokens={invalid_total}")
    if incomplete_requests:
        errors.append(f"incomplete requests={incomplete_requests}")
    if data.get("mpk_attention_requested") != "auto":
        errors.append(f"requested attention={data.get('mpk_attention_requested')!r}")
    if data.get("mpk_attention") != "split-kv":
        errors.append(f"resolved attention={data.get('mpk_attention')!r}")
    details = {
        "minimum_matches": minimum_matches,
        "passing_requests": passing_requests,
        "invalid_total": invalid_total,
        "incomplete_requests": incomplete_requests,
    }
    return (
        "failed" if errors else "completed",
        "; ".join(errors),
        details,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in args.batch_sizes):
        parser.error("--batch-sizes values must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    reference = execute(args, "reference_torch", 1, mpk=False)
    if reference["status"] != "completed":
        raise RuntimeError(
            f"Torch reference failed: {reference.get('reason')}; "
            f"see {reference['log']}"
        )
    reference_data = reference["data"]
    if (
        reference_data.get("generate_length") != S_OUT
        or reference_data.get("invalid_token_count") != 0
    ):
        raise ValueError("Torch reference is incomplete or invalid")
    reference_tokens = reference_data["token_ids"]

    rows = []
    all_passed = True
    for batch_size in args.batch_sizes:
        result = execute(args, f"batch_{batch_size}", batch_size, mpk=True)
        status, reason, details = validate(
            result, reference_tokens, batch_size
        )
        data = result.get("data", {})
        decode_ms = data.get("decode_time_ms")
        decode_steps = data.get("decode_steps")
        throughput = None
        if (
            isinstance(decode_ms, (int, float)) and decode_ms > 0
            and isinstance(decode_steps, int) and decode_steps > 0
        ):
            throughput = 1000.0 * batch_size * decode_steps / decode_ms
        row = {
            "batch_size": batch_size,
            "status": status,
            "minimum_first10_matches": details.get("minimum_matches"),
            "passing_requests": details.get("passing_requests"),
            "invalid_token_count": details.get("invalid_total"),
            "incomplete_requests": details.get("incomplete_requests"),
            "resolved_attention": data.get("mpk_attention"),
            "prefill_time_ms": data.get("prefill_time_ms"),
            "decode_time_ms": decode_ms,
            "decode_step_time_ms": data.get("decode_step_time_ms"),
            "decode_tokens_per_second": throughput,
            "throughput_scale_vs_b1": None,
            "throughput_efficiency_vs_b1": None,
            "cache_status": data.get("mpk_kernel_cache_status"),
            "kernel_prepare_time_ms": data.get("mpk_kernel_prepare_time_ms"),
            "output_path": str(result["output"]),
            "log_path": str(result["log"]),
            "reason": reason,
        }
        rows.append(row)
        all_passed &= status == "completed"
        print(
            f"B={batch_size}: {'PASS' if status == 'completed' else 'FAIL'}; "
            f"minimum first-10={details.get('minimum_matches')}; "
            f"passing={details.get('passing_requests')}/{batch_size}; "
            f"invalid={details.get('invalid_total')}; "
            f"incomplete={details.get('incomplete_requests')}; "
            f"decode throughput={throughput}",
            flush=True,
        )

    b1_row = next((row for row in rows if row["batch_size"] == 1), None)
    b1_throughput = b1_row["decode_tokens_per_second"] if b1_row else None
    if isinstance(b1_throughput, (int, float)) and b1_throughput > 0:
        for row in rows:
            throughput = row["decode_tokens_per_second"]
            if isinstance(throughput, (int, float)):
                scale = throughput / b1_throughput
                row["throughput_scale_vs_b1"] = scale
                row["throughput_efficiency_vs_b1"] = (
                    scale / row["batch_size"]
                )

    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 6,
        "status": "passed" if all_passed else "failed",
        "model": args.model,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ_LENGTH,
        "batch_sizes": args.batch_sizes,
        "compare_tokens": COMPARE_TOKENS,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(
        f"Step 6 batch scaling validation: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
