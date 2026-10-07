"""Validate adaptive MPK decode-only across dense Qwen3 model sizes."""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
COMPARE_TOKENS = 10
BATCH_SIZE = 1
S_IN = 128
S_OUT = 128
MAX_SEQ_LENGTH = 256
CSV_FIELDS = (
    "model", "status", "first10_matches", "invalid_token_count",
    "incomplete_requests", "resolved_attention", "prefill_time_ms",
    "decode_time_ms", "decode_step_time_ms", "decode_tokens_per_second",
    "cache_status", "kernel_prepare_time_ms", "torch_output_path",
    "mpk_output_path", "torch_log_path", "mpk_log_path", "reason",
    "prefill_stage_profile_ms",
)


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


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


def execute(args, model, backend, mpk=None):
    if mpk is None:
        mpk = backend == "mpk"
    model_name = safe_name(model)
    output = args.output_dir / f"{model_name}_{backend}.json"
    log = args.output_dir / f"{model_name}_{backend}.log"
    command = [
        sys.executable, str(DEMO),
        "--model", model,
        "--input-length", str(S_IN),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--max-new-tokens", str(S_OUT),
        "--page-size", str(MAX_SEQ_LENGTH),
        "--max-num-pages", "1",
        "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8",
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mpk:
        cache = args.cache_dir / model_name
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-kernel-cache-dir", str(cache),
        ]
        if args.profile_prefill_stages:
            command.append("--profile-prefill-stages")
    print(
        f"Running model={model} backend={backend} B={BATCH_SIZE} "
        f"S_IN={S_IN} S_OUT={S_OUT}",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--profile-prefill-stages", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    rows = []
    all_passed = True
    for model in args.models:
        torch_result = execute(args, model, "torch")
        for warmup_index in range(args.warmup_runs):
            warmup_result = execute(
                args, model, f"mpk_warmup_{warmup_index + 1}", mpk=True
            )
            if warmup_result["status"] != "completed":
                raise RuntimeError(
                    f"MPK warmup failed for {model}: "
                    f"{warmup_result.get('reason')}; see {warmup_result['log']}"
                )
        mpk_result = execute(args, model, "mpk")
        errors = []
        torch_data = torch_result.get("data", {})
        mpk_data = mpk_result.get("data", {})
        if torch_result["status"] != "completed":
            errors.append(
                f"Torch status={torch_result['status']}: "
                f"{torch_result.get('reason', '')}"
            )
        if mpk_result["status"] != "completed":
            errors.append(
                f"MPK status={mpk_result['status']}: "
                f"{mpk_result.get('reason', '')}"
            )

        matches = None
        invalid_total = None
        incomplete_requests = None
        if not errors:
            torch_tokens = torch_data.get("token_ids", [])
            mpk_tokens_by_request = mpk_data.get("token_ids_by_request", [])
            lengths = mpk_data.get("generate_lengths_by_request", [])
            invalid_counts = mpk_data.get(
                "invalid_token_counts_by_request", []
            )
            if torch_data.get("generate_length") != S_OUT:
                errors.append("incomplete Torch output")
            if torch_data.get("invalid_token_count") != 0:
                errors.append("invalid Torch token")
            if len(mpk_tokens_by_request) != BATCH_SIZE:
                errors.append("wrong MPK token batch size")
            if len(lengths) != BATCH_SIZE:
                errors.append("wrong MPK generation-length batch size")
            if len(invalid_counts) != BATCH_SIZE:
                errors.append("wrong MPK invalid-token batch size")
            if mpk_tokens_by_request:
                matches = sum(
                    a == b
                    for a, b in zip(
                        torch_tokens[:COMPARE_TOKENS],
                        mpk_tokens_by_request[0][:COMPARE_TOKENS],
                    )
                )
            invalid_total = sum(invalid_counts)
            incomplete_requests = sum(value != S_OUT for value in lengths)
            if matches != COMPARE_TOKENS:
                errors.append(f"first-10 matches={matches}/10")
            if invalid_total:
                errors.append(f"invalid tokens={invalid_total}")
            if incomplete_requests:
                errors.append(f"incomplete requests={incomplete_requests}")
            if mpk_data.get("mpk_attention_requested") != "auto":
                errors.append("adaptive attention was not requested")
            if mpk_data.get("mpk_attention") != "default":
                errors.append(
                    f"resolved attention={mpk_data.get('mpk_attention')!r}"
                )

        status = "failed" if errors else "completed"
        all_passed &= status == "completed"
        decode_ms = mpk_data.get("decode_time_ms")
        decode_steps = mpk_data.get("decode_steps")
        throughput = None
        if (
            isinstance(decode_ms, (int, float)) and decode_ms > 0
            and isinstance(decode_steps, int) and decode_steps > 0
        ):
            throughput = 1000.0 * decode_steps / decode_ms
        row = {
            "model": model,
            "status": status,
            "first10_matches": matches,
            "invalid_token_count": invalid_total,
            "incomplete_requests": incomplete_requests,
            "resolved_attention": mpk_data.get("mpk_attention"),
            "prefill_time_ms": mpk_data.get("prefill_time_ms"),
            "decode_time_ms": decode_ms,
            "decode_step_time_ms": mpk_data.get("decode_step_time_ms"),
            "decode_tokens_per_second": throughput,
            "cache_status": mpk_data.get("mpk_kernel_cache_status"),
            "kernel_prepare_time_ms": mpk_data.get(
                "mpk_kernel_prepare_time_ms"
            ),
            "torch_output_path": str(torch_result["output"]),
            "mpk_output_path": str(mpk_result["output"]),
            "torch_log_path": str(torch_result["log"]),
            "mpk_log_path": str(mpk_result["log"]),
            "reason": "; ".join(errors),
            "prefill_stage_profile_ms": json.dumps(
                mpk_data.get("prefill_stage_profile_ms")
            ),
        }
        rows.append(row)
        print(
            f"{model}: {'PASS' if status == 'completed' else 'FAIL'}; "
            f"attention={mpk_data.get('mpk_attention')}; "
            f"first-10={matches}; invalid={invalid_total}; "
            f"incomplete={incomplete_requests}; "
            f"decode throughput={throughput}",
            flush=True,
        )

    with (args.output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 7,
        "status": "passed" if all_passed else "failed",
        "models": args.models,
        "batch_size": BATCH_SIZE,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ_LENGTH,
        "compare_tokens": COMPARE_TOKENS,
        "warmup_runs": args.warmup_runs,
        "measured_runs": 1,
        "profile_prefill_stages": args.profile_prefill_stages,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(
        f"Step 7 multi-model smoke validation: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
