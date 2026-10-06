"""Validate automatic MPK attention selection and token correctness."""

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
CASES = (
    ("short", 2, 240, 16, 256, "default"),
    ("long_context", 8, 1024, 128, 1152, "split-kv"),
    ("long_generation", 2, 128, 1024, 1152, "split-kv"),
)
CSV_FIELDS = (
    "case", "batch_size", "s_in", "s_out", "max_seq_length",
    "expected_attention", "resolved_attention", "status",
    "minimum_first10_matches", "passing_requests", "invalid_token_count",
    "incomplete_requests", "prefill_time_ms", "decode_time_ms",
    "decode_step_time_ms", "cache_status", "kernel_prepare_time_ms",
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


def execute(args, name, batch_size, s_in, s_out, max_seq_length, mpk):
    suffix = "mpk" if mpk else "torch"
    output = args.output_dir / f"{name}_{suffix}.json"
    log = args.output_dir / f"{name}_{suffix}.log"
    command = [
        sys.executable, str(DEMO),
        "--model", args.model,
        "--input-length", str(s_in),
        "--max-seq-length", str(max_seq_length),
        "--max-new-tokens", str(s_out),
        "--page-size", str(max_seq_length),
        "--max-num-pages", str(batch_size),
        "--max-num-batched-requests", str(batch_size),
        "--max-num-batched-tokens", str(max(8, batch_size)),
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mpk:
        cache = args.cache_dir / f"{name}_b{batch_size}_seq{max_seq_length}"
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    print(
        f"Running case={name} backend={suffix} B={batch_size} "
        f"S_IN={s_in} S_OUT={s_out}",
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
            return {"status": "timeout", "output": output, "log": log}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    rows = []
    all_passed = True
    for name, batch_size, s_in, s_out, max_seq_length, expected in CASES:
        reference = execute(
            args, name, 1, s_in, s_out, max_seq_length, mpk=False
        )
        result = execute(
            args, name, batch_size, s_in, s_out, max_seq_length, mpk=True
        )
        errors = []
        details = {}
        data = result.get("data", {})
        if reference["status"] != "completed":
            errors.append(f"reference status={reference['status']}")
        if result["status"] != "completed":
            errors.append(f"MPK status={result['status']}")
        if not errors:
            reference_data = reference["data"]
            expected_tokens = reference_data.get("token_ids", [])
            tokens_by_request = data.get("token_ids_by_request", [])
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            if reference_data.get("generate_length") != s_out:
                errors.append("incomplete reference generation")
            if reference_data.get("invalid_token_count") != 0:
                errors.append("invalid reference token")
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
                sum(a == b for a, b in zip(expected_tokens[:10], tokens[:10]))
                for tokens in tokens_by_request
            ]
            details["minimum_matches"] = min(matches) if matches else None
            details["passing_requests"] = sum(
                value == COMPARE_TOKENS for value in matches
            )
            details["invalid_total"] = sum(invalid_counts)
            details["incomplete_requests"] = sum(
                value != s_out for value in lengths
            )
            if details["minimum_matches"] != COMPARE_TOKENS:
                errors.append(
                    f"minimum first-10={details['minimum_matches']}/10"
                )
            if details["invalid_total"]:
                errors.append(f"invalid tokens={details['invalid_total']}")
            if details["incomplete_requests"]:
                errors.append(
                    f"incomplete requests={details['incomplete_requests']}"
                )
            if data.get("mpk_attention_requested") != "auto":
                errors.append(
                    f"requested={data.get('mpk_attention_requested')!r}"
                )
            if data.get("mpk_attention") != expected:
                errors.append(f"resolved={data.get('mpk_attention')!r}")
            if expected == "split-kv":
                if data.get("mpk_split_kv_chunk_size") != 128:
                    errors.append("wrong split-KV chunk size")
                expected_chunks = (max_seq_length + 127) // 128
                if data.get("mpk_split_kv_num_chunks") != expected_chunks:
                    errors.append("wrong split-KV chunk count")

        status = "failed" if errors else "completed"
        all_passed &= status == "completed"
        row = {
            "case": name, "batch_size": batch_size, "s_in": s_in,
            "s_out": s_out, "max_seq_length": max_seq_length,
            "expected_attention": expected,
            "resolved_attention": data.get("mpk_attention"),
            "status": status,
            "minimum_first10_matches": details.get("minimum_matches"),
            "passing_requests": details.get("passing_requests"),
            "invalid_token_count": details.get("invalid_total"),
            "incomplete_requests": details.get("incomplete_requests"),
            "prefill_time_ms": data.get("prefill_time_ms"),
            "decode_time_ms": data.get("decode_time_ms"),
            "decode_step_time_ms": data.get("decode_step_time_ms"),
            "cache_status": data.get("mpk_kernel_cache_status"),
            "kernel_prepare_time_ms": data.get("mpk_kernel_prepare_time_ms"),
            "output_path": str(result["output"]),
            "log_path": str(result["log"]),
            "reason": "; ".join(errors),
        }
        rows.append(row)
        print(
            f"{name}: {'PASS' if status == 'completed' else 'FAIL'}; "
            f"attention={data.get('mpk_attention')}; "
            f"minimum first-10={details.get('minimum_matches')}; "
            f"passing={details.get('passing_requests')}/{batch_size}; "
            f"invalid={details.get('invalid_total')}; "
            f"incomplete={details.get('incomplete_requests')}",
            flush=True,
        )

    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 5,
        "status": "passed" if all_passed else "failed",
        "model": args.model,
        "threshold": args.threshold,
        "compare_tokens": COMPARE_TOKENS,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(
        f"Step 5 adaptive attention validation: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
