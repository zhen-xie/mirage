"""Validate split-KV decode-only capacity and multi-request correctness."""

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
REFERENCE_SHAPES = {
    # End exactly at max_seq_length while exercising a partially filled
    # second 128-token split-KV chunk.
    "short_tail": (240, 16, 256),
    "long_context": (1024, 128, 1152),
    "long_generation": (128, 1024, 1152),
}
CASES = (
    ("capacity_b2", "short_tail", 2),
    ("capacity_b8", "short_tail", 8),
    ("capacity_b32", "short_tail", 32),
    ("capacity_b128", "short_tail", 128),
    ("long_context_b8", "long_context", 8),
    ("long_generation_b2", "long_generation", 2),
)
CSV_FIELDS = (
    "case", "batch_size", "s_in", "s_out", "max_seq_length", "status",
    "minimum_first10_matches", "passing_requests", "invalid_token_count",
    "incomplete_requests", "prefill_time_ms", "decode_time_ms",
    "decode_step_time_ms", "decode_tokens_per_second", "cache_status",
    "kernel_prepare_time_ms", "split_kv_chunk_size", "split_kv_num_chunks",
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


def execute(args, name, s_in, s_out, max_seq_length, batch_size):
    output = args.output_dir / f"{name}.json"
    log = args.output_dir / f"{name}.log"
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
    if batch_size > 1:
        cache = args.cache_dir / f"b{batch_size}_seq{max_seq_length}"
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-kernel-cache-dir", str(cache),
        ]
        if args.attention_mode == "split-kv":
            command += [
                "--split-kv-cache", "--mpk-split-kv-chunk-size", "128",
            ]
    print(
        f"Running {name}: B={batch_size} S_IN={s_in} S_OUT={s_out}",
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
            return {"status": "timeout", "reason": "timeout", "output": output, "log": log}
    if returncode:
        return {
            "status": "runtime_failed", "reason": f"exit code {returncode}",
            "output": output, "log": log,
        }
    if not output.is_file():
        return {"status": "missing_output", "reason": "missing JSON", "output": output, "log": log}
    return {
        "status": "completed", "data": json.loads(output.read_text()),
        "output": output, "log": log,
    }


def validate_case(
    result, reference, batch_size, s_out, max_seq_length, attention_mode
):
    if result["status"] != "completed":
        return result["status"], result.get("reason"), {}
    data = result["data"]
    tokens_by_request = data.get("token_ids_by_request")
    lengths = data.get("generate_lengths_by_request")
    invalid_counts = data.get("invalid_token_counts_by_request")
    errors = []
    if not isinstance(tokens_by_request, list) or len(tokens_by_request) != batch_size:
        errors.append("token_ids_by_request has the wrong batch size")
        tokens_by_request = []
    if not isinstance(lengths, list) or len(lengths) != batch_size:
        errors.append("generate_lengths_by_request has the wrong batch size")
        lengths = []
    if not isinstance(invalid_counts, list) or len(invalid_counts) != batch_size:
        errors.append("invalid_token_counts_by_request has the wrong batch size")
        invalid_counts = []
    matches = [
        sum(a == b for a, b in zip(reference[:10], tokens[:10]))
        for tokens in tokens_by_request
    ]
    minimum_matches = min(matches) if matches else None
    passing_requests = sum(value == COMPARE_TOKENS for value in matches)
    incomplete_requests = sum(value != s_out for value in lengths)
    invalid_total = sum(invalid_counts)
    if minimum_matches != COMPARE_TOKENS:
        errors.append(f"minimum first-10 matches={minimum_matches}/10")
    if incomplete_requests:
        errors.append(f"incomplete_requests={incomplete_requests}")
    if invalid_total:
        errors.append(f"invalid_token_count={invalid_total}")
    expected_attention = attention_mode
    if data.get("mpk_attention") != expected_attention:
        errors.append(f"attention={data.get('mpk_attention')!r}")
    if expected_attention == "split-kv":
        if data.get("mpk_split_kv_chunk_size") != 128:
            errors.append(f"chunk_size={data.get('mpk_split_kv_chunk_size')}")
        if data.get("mpk_split_kv_num_chunks") != max_seq_length // 128:
            errors.append(f"num_chunks={data.get('mpk_split_kv_num_chunks')}")
    if data.get("mpk_kernel_cache_status") != "miss_compiled":
        errors.append(f"cache_status={data.get('mpk_kernel_cache_status')!r}")
    details = {
        "minimum_matches": minimum_matches,
        "passing_requests": passing_requests,
        "incomplete_requests": incomplete_requests,
        "invalid_total": invalid_total,
    }
    return (
        "failed" if errors else "completed",
        "; ".join(errors) if errors else None,
        details,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attention-mode", choices=("default", "split-kv"),
        default="split-kv",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case[0] for case in CASES],
        help="Run only the selected Step 4 cases.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    selected_cases = (
        [case for case in CASES if case[0] in args.cases]
        if args.cases else list(CASES)
    )
    required_shapes = {case[1] for case in selected_cases}
    references = {}
    all_passed = True
    for name, (s_in, s_out, max_seq_length) in REFERENCE_SHAPES.items():
        if name not in required_shapes:
            continue
        result = execute(
            args, f"reference_{name}", s_in, s_out, max_seq_length, 1
        )
        if result["status"] != "completed":
            raise RuntimeError(
                f"Reference {name} failed: {result.get('reason')}; "
                f"see {result['log']}"
            )
        data = result["data"]
        if data.get("generate_length") != s_out or data.get("invalid_token_count") != 0:
            raise ValueError(f"Reference {name} is incomplete or invalid")
        references[name] = data["token_ids"]

    rows = []
    for case, shape, batch_size in selected_cases:
        s_in, s_out, max_seq_length = REFERENCE_SHAPES[shape]
        result = execute(
            args, case, s_in, s_out, max_seq_length, batch_size
        )
        status, reason, details = validate_case(
            result, references[shape], batch_size, s_out, max_seq_length,
            args.attention_mode,
        )
        data = result.get("data", {})
        decode_ms = data.get("decode_time_ms")
        decode_steps = data.get("decode_steps")
        throughput = (
            1000.0 * batch_size * decode_steps / decode_ms
            if isinstance(decode_ms, (int, float)) and decode_ms > 0
            and isinstance(decode_steps, int) else None
        )
        row = {
            "case": case, "batch_size": batch_size, "s_in": s_in,
            "s_out": s_out, "max_seq_length": max_seq_length,
            "status": status,
            "minimum_first10_matches": details.get("minimum_matches"),
            "passing_requests": details.get("passing_requests"),
            "invalid_token_count": details.get("invalid_total"),
            "incomplete_requests": details.get("incomplete_requests"),
            "prefill_time_ms": data.get("prefill_time_ms"),
            "decode_time_ms": decode_ms,
            "decode_step_time_ms": data.get("decode_step_time_ms"),
            "decode_tokens_per_second": throughput,
            "cache_status": data.get("mpk_kernel_cache_status"),
            "kernel_prepare_time_ms": data.get("mpk_kernel_prepare_time_ms"),
            "split_kv_chunk_size": data.get("mpk_split_kv_chunk_size"),
            "split_kv_num_chunks": data.get("mpk_split_kv_num_chunks"),
            "output_path": str(result["output"]),
            "log_path": str(result["log"]), "reason": reason,
        }
        rows.append(row)
        all_passed &= status == "completed"
        print(
            f"{case}: {'PASS' if status == 'completed' else 'FAIL'}; "
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
        "step": 4,
        "status": "passed" if all_passed else "failed",
        "model": args.model,
        "compare_tokens": COMPARE_TOKENS,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Step 4 split-KV batch validation: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
