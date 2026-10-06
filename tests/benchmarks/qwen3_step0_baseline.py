"""Run and validate the Step 0 Torch and MPK-always Qwen3 baselines."""

import argparse
import csv
import json
import os
import platform
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
CASES = (
    ("short", 128, 128),
    ("long_context", 1024, 128),
    ("long_generation", 128, 1024),
)
BACKENDS = ("torch", "mpk_always")
COMPARE_TOKENS = 10
CSV_FIELDS = (
    "case", "backend", "model", "batch_size", "s_in", "s_out",
    "prompt_length", "generate_length", "requested_generate_length",
    "saved_token_count", "first10_matches_vs_torch", "first_mismatch",
    "invalid_token_count", "total_time_ms", "latency_ms_per_token",
    "status", "output_path", "log_path",
)


def run_text(command):
    try:
        return subprocess.run(
            command, cwd=ROOT, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(model):
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    return {
        "created_at_unix": time.time(),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_status": run_text([
            "git", "status", "--short", "--ignore-submodules=all"
        ]),
        "hostname": platform.node(),
        "python": sys.version,
        "model": model,
        "model_commit_hash": getattr(config, "_commit_hash", None),
        "model_config": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", None),
            "vocab_size": config.vocab_size,
            "torch_dtype": str(getattr(config, "torch_dtype", None)),
        },
        "gpu": torch.cuda.get_device_name(0),
        "gpu_properties": str(torch.cuda.get_device_properties(0)),
        "driver": run_text([
            "nvidia-smi", "--query-gpu=driver_version",
            "--format=csv,noheader"
        ]),
        "nvcc": run_text(["nvcc", "--version"]),
        "packages": {
            name: package_version(name)
            for name in (
                "mirage-project", "torch", "transformers", "accelerate",
                "safetensors", "cuda-python",
            )
        },
        "benchmark": {
            "batch_size": 1,
            "dtype": "bfloat16",
            "decoding": "greedy",
            "ignore_eos": True,
            "compare_tokens": COMPARE_TOKENS,
            "cases": [
                {"name": name, "s_in": s_in, "s_out": s_out}
                for name, s_in, s_out in CASES
            ],
        },
    }


def terminate_process_group(process):
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


def run_backend(args, name, s_in, s_out, backend):
    case_dir = args.output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / f"{backend}.json"
    log = case_dir / f"{backend}.log"
    page_size = ((s_in + s_out + 127) // 128) * 128
    command = [
        sys.executable, str(DEMO),
        "--model", args.model,
        "--input-length", str(s_in),
        "--max-seq-length", str(s_in + s_out),
        "--max-new-tokens", str(s_out),
        "--page-size", str(page_size),
        "--max-num-pages", "1",
        "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8",
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if backend == "mpk_always":
        command.append("--use-mirage")
        command += ["--output-dir", str(case_dir / "build")]

    print(
        f"Running case={name} backend={backend} "
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
            terminate_process_group(process)
            return {
                "status": "timeout", "reason": f"Exceeded {args.timeout}s",
                "output": output, "log": log,
            }
    if returncode:
        return {
            "status": "runtime_failed", "reason": f"Exit code {returncode}",
            "output": output, "log": log,
        }
    if not output.is_file():
        return {
            "status": "missing_output", "reason": "Output JSON was not created",
            "output": output, "log": log,
        }
    return {
        "status": "completed", "data": json.loads(output.read_text()),
        "output": output, "log": log,
    }


def validate_result(result, s_in, s_out):
    if result["status"] != "completed":
        return result["status"], result.get("reason"), []
    data = result["data"]
    tokens = data.get("token_ids")
    if not isinstance(tokens, list):
        return "invalid_output", "token_ids is missing or not a list", []
    if data.get("prompt_length") != s_in:
        return "invalid_output", (
            f"prompt length {data.get('prompt_length')} != {s_in}"
        ), tokens
    if data.get("generate_length") != s_out:
        return "incomplete_generation", (
            f"generate length {data.get('generate_length')} != {s_out}"
        ), tokens
    if len(tokens) < min(COMPARE_TOKENS, s_out):
        return "incomplete_saved_tokens", (
            f"saved {len(tokens)} tokens; need {min(COMPARE_TOKENS, s_out)}"
        ), tokens
    vocab_size = data.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        return "invalid_output", "vocab_size is missing", tokens
    invalid_token_count = data.get("invalid_token_count")
    if not isinstance(invalid_token_count, int) or invalid_token_count < 0:
        return "invalid_output", "invalid_token_count is missing", tokens
    if invalid_token_count:
        return "invalid_tokens", (
            f"found {invalid_token_count} invalid token IDs"
        ), tokens
    return "completed", None, tokens


def make_row(name, backend, args, s_in, s_out, result, reference_tokens):
    status, reason, tokens = validate_result(result, s_in, s_out)
    compared = min(COMPARE_TOKENS, s_out)
    matches = None
    first_mismatch = None
    if backend == "torch" and status == "completed":
        matches = compared
    elif status == "completed" and reference_tokens:
        matches = sum(
            expected == actual
            for expected, actual in zip(
                reference_tokens[:compared], tokens[:compared]
            )
        )
        first_mismatch = next(
            (index for index, (expected, actual) in enumerate(zip(
                reference_tokens[:compared], tokens[:compared]
            )) if expected != actual),
            None,
        )
        if matches != compared:
            status = "correctness_failed"
            reason = f"Matched {matches}/{compared}; first mismatch {first_mismatch}"

    data = result.get("data", {})
    invalid_count = data.get("invalid_token_count")
    row = {
        "case": name,
        "backend": backend,
        "model": args.model,
        "batch_size": 1,
        "s_in": s_in,
        "s_out": s_out,
        "prompt_length": data.get("prompt_length"),
        "generate_length": data.get("generate_length"),
        "requested_generate_length": data.get("requested_generate_length"),
        "saved_token_count": len(tokens),
        "first10_matches_vs_torch": matches,
        "first_mismatch": first_mismatch,
        "invalid_token_count": invalid_count,
        "total_time_ms": data.get("total_time_ms"),
        "latency_ms_per_token": data.get("latency_ms_per_token"),
        "status": status,
        "output_path": str(result["output"]),
        "log_path": str(result["log"]),
    }
    return row, reason, tokens


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "qwen3_step0_baseline",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment(args.model)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    case_reports = []
    overall_passed = True
    for name, s_in, s_out in CASES:
        results = {
            backend: run_backend(args, name, s_in, s_out, backend)
            for backend in BACKENDS
        }
        torch_row, torch_reason, reference_tokens = make_row(
            name, "torch", args, s_in, s_out, results["torch"], None
        )
        mpk_row, mpk_reason, _ = make_row(
            name, "mpk_always", args, s_in, s_out,
            results["mpk_always"], reference_tokens,
        )
        rows.extend((torch_row, mpk_row))
        passed = torch_row["status"] == mpk_row["status"] == "completed"
        overall_passed &= passed
        case_reports.append({
            "case": name,
            "batch_size": 1,
            "s_in": s_in,
            "s_out": s_out,
            "torch_status": torch_row["status"],
            "torch_reason": torch_reason,
            "mpk_status": mpk_row["status"],
            "mpk_reason": mpk_reason,
            "first10_matches": mpk_row["first10_matches_vs_torch"],
            "first_mismatch": mpk_row["first_mismatch"],
            "passed": passed,
        })
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}; "
            f"MPK vs Torch first-10="
            f"{mpk_row['first10_matches_vs_torch']}/10",
            flush=True,
        )
        write_csv(args.output_dir / "summary.csv", rows)

    summary = {
        "stage": "mpk-step-00-baseline",
        "reference": "torch",
        "candidate": "mpk_always",
        "required_positional_matches": "10/10",
        "cases": case_reports,
        "status": "passed" if overall_passed else "failed",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {summary_path}")
    print(f"Step 0 validation: {'PASS' if overall_passed else 'FAIL'}")
    raise SystemExit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
