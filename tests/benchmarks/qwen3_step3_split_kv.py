"""Validate fixed 128-token MPK split-KV attention against Step 2."""

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
CASES = (
    ("short", 128, 128),
    ("long_context", 1024, 128),
    ("long_generation", 128, 1024),
)
BACKENDS = ("torch", "mpk_default", "mpk_split_kv_128")
COMPARE_TOKENS = 10
CSV_FIELDS = (
    "case", "backend", "s_in", "s_out", "status",
    "first10_matches_vs_torch", "generate_length", "invalid_token_count",
    "prefill_time_ms", "decode_time_ms", "decode_steps",
    "decode_step_time_ms", "decode_tokens_per_second",
    "split_speedup_vs_default", "cache_status", "kernel_prepare_time_ms",
    "attention", "split_kv_chunk_size", "split_kv_num_chunks",
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


def run_backend(args, case, s_in, s_out, backend):
    case_dir = args.output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / f"{backend}.json"
    log = case_dir / f"{backend}.log"
    max_seq_length = s_in + s_out
    command = [
        sys.executable, str(DEMO),
        "--model", args.model,
        "--input-length", str(s_in),
        "--max-seq-length", str(max_seq_length),
        "--max-new-tokens", str(s_out),
        "--page-size", str(max_seq_length),
        "--max-num-pages", "1",
        "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8",
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if backend != "torch":
        cache = args.cache_dir / f"seq_{max_seq_length}" / backend
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-kernel-cache-dir", str(cache),
        ]
        if backend == "mpk_split_kv_128":
            command += [
                "--split-kv-cache",
                "--mpk-split-kv-chunk-size", "128",
            ]
    print(
        f"Running case={case} backend={backend} "
        f"B=1 S_IN={s_in} S_OUT={s_out}",
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
        "status": "completed",
        "data": json.loads(output.read_text()),
        "output": output,
        "log": log,
    }


def validate(result, backend, s_out, reference, expected_cache_status):
    if result["status"] != "completed":
        return result["status"], result.get("reason"), None
    data = result["data"]
    errors = []
    tokens = data.get("token_ids", [])
    matches = sum(a == b for a, b in zip(reference[:10], tokens[:10]))
    if len(tokens) < COMPARE_TOKENS or matches != COMPARE_TOKENS:
        errors.append(f"first-10 matches={matches}/10")
    if data.get("generate_length") != s_out:
        errors.append(f"generate_length={data.get('generate_length')} != {s_out}")
    if data.get("invalid_token_count") != 0:
        errors.append(f"invalid_token_count={data.get('invalid_token_count')}")
    if data.get("decode_steps") != s_out - 1:
        errors.append(f"decode_steps={data.get('decode_steps')} != {s_out - 1}")
    if backend != "torch":
        if data.get("mpk_kernel_cache_status") != expected_cache_status:
            errors.append(
                f"cache_status={data.get('mpk_kernel_cache_status')!r} "
                f"!= {expected_cache_status!r}"
            )
        expected_attention = (
            "split-kv" if backend == "mpk_split_kv_128" else "default"
        )
        if data.get("mpk_attention") != expected_attention:
            errors.append(f"attention={data.get('mpk_attention')!r}")
    if backend == "mpk_split_kv_128":
        if data.get("mpk_split_kv_chunk_size") != 128:
            errors.append(
                f"chunk_size={data.get('mpk_split_kv_chunk_size')} != 128"
            )
        expected_chunks = (data.get("prompt_length", 0) + s_out) // 128
        if data.get("mpk_split_kv_num_chunks") != expected_chunks:
            errors.append(
                f"num_chunks={data.get('mpk_split_kv_num_chunks')} "
                f"!= {expected_chunks}"
            )
    return (
        "failed" if errors else "completed",
        "; ".join(errors) if errors else None,
        matches,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    rows = []
    all_passed = True
    seen_cache_keys = set()
    summaries = {}
    for case, s_in, s_out in CASES:
        results = {}
        for backend in BACKENDS:
            results[backend] = run_backend(args, case, s_in, s_out, backend)
        torch_data = results["torch"].get("data", {})
        reference = torch_data.get("token_ids", [])
        case_rows = {}
        for backend in BACKENDS:
            expected_cache_status = None
            if backend != "torch":
                key = (s_in + s_out, backend)
                expected_cache_status = (
                    "hit" if key in seen_cache_keys else "miss_compiled"
                )
                seen_cache_keys.add(key)
            status, reason, matches = validate(
                results[backend], backend, s_out, reference,
                expected_cache_status,
            )
            data = results[backend].get("data", {})
            decode_ms = data.get("decode_time_ms")
            decode_steps = data.get("decode_steps")
            throughput = (
                1000.0 * decode_steps / decode_ms
                if isinstance(decode_ms, (int, float)) and decode_ms > 0
                and isinstance(decode_steps, int) else None
            )
            row = {
                "case": case, "backend": backend, "s_in": s_in,
                "s_out": s_out, "status": status,
                "first10_matches_vs_torch": matches,
                "generate_length": data.get("generate_length"),
                "invalid_token_count": data.get("invalid_token_count"),
                "prefill_time_ms": data.get("prefill_time_ms"),
                "decode_time_ms": decode_ms,
                "decode_steps": decode_steps,
                "decode_step_time_ms": data.get("decode_step_time_ms"),
                "decode_tokens_per_second": throughput,
                "split_speedup_vs_default": None,
                "cache_status": data.get("mpk_kernel_cache_status"),
                "kernel_prepare_time_ms": data.get("mpk_kernel_prepare_time_ms"),
                "attention": data.get("mpk_attention"),
                "split_kv_chunk_size": data.get("mpk_split_kv_chunk_size"),
                "split_kv_num_chunks": data.get("mpk_split_kv_num_chunks"),
                "output_path": str(results[backend]["output"]),
                "log_path": str(results[backend]["log"]),
                "reason": reason,
            }
            rows.append(row)
            case_rows[backend] = row
            all_passed &= status == "completed"
        default_ms = case_rows["mpk_default"]["decode_time_ms"]
        split_ms = case_rows["mpk_split_kv_128"]["decode_time_ms"]
        if default_ms and split_ms:
            case_rows["mpk_split_kv_128"]["split_speedup_vs_default"] = (
                default_ms / split_ms
            )
        summaries[case] = case_rows
        print(
            f"{case}: {'PASS' if all(row['status'] == 'completed' for row in case_rows.values()) else 'FAIL'}; "
            f"split speedup={case_rows['mpk_split_kv_128']['split_speedup_vs_default']}",
            flush=True,
        )

    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "step": 3,
        "status": "passed" if all_passed else "failed",
        "model": args.model,
        "split_kv_chunk_size": 128,
        "cases": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Step 3 split-KV validation: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
