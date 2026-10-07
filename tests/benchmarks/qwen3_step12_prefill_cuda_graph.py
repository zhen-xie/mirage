"""Compare eager and CUDA Graph Normal prefill before MPK decode."""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
MODES = ("eager", "cuda_graph")
FIELDS = (
    "model", "status", "mode", "first10_matches", "prefill_ms",
    "decode_ms", "decode_step_ms", "prefill_speedup_vs_eager",
    "decode_ratio_vs_eager", "invalid_token_count", "incomplete_requests",
    "reason",
)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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


def execute(args, model, name, mode=None):
    model_name = safe_name(model)
    output = args.output_dir / f"{model_name}_{name}.json"
    log = args.output_dir / f"{model_name}_{name}.log"
    command = [
        sys.executable, str(DEMO), "--model", model,
        "--input-length", "128", "--max-seq-length", "256",
        "--max-new-tokens", "128", "--page-size", "256",
        "--max-num-pages", "1", "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8", "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mode is not None:
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-kernel-cache-dir", str(args.cache_dir / model_name),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1",
        ]
        if mode == "cuda_graph":
            command.append("--normal-prefill-cuda-graph")
    print(f"Running model={model} run={name}", flush=True)
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return None, f"timeout; see {log}"
    if code:
        return None, f"exit code {code}; see {log}"
    if not output.is_file():
        return None, f"missing output; see {log}"
    return json.loads(output.read_text(encoding="utf-8")), ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_passed = True
    for model in args.models:
        reference, error = execute(args, model, "torch")
        if reference is None:
            raise RuntimeError(error)
        measured = {
            mode: execute(args, model, mode, mode) for mode in MODES
        }
        model_rows = []
        for mode in MODES:
            data, run_error = measured[mode]
            errors = [run_error] if run_error else []
            matches = invalid = incomplete = None
            if data is not None:
                actual = data.get("token_ids_by_request", [[]])[0]
                expected = reference.get("token_ids", [])
                matches = sum(a == b for a, b in zip(expected[:10], actual[:10]))
                invalid = data.get("invalid_token_count")
                lengths = data.get("generate_lengths_by_request", [])
                incomplete = sum(value != 128 for value in lengths)
                if matches != 10:
                    errors.append(f"first-10={matches}/10")
                if invalid != 0:
                    errors.append(f"invalid tokens={invalid}")
                if incomplete != 0:
                    errors.append(f"incomplete requests={incomplete}")
                expected_graph = mode == "cuda_graph"
                if data.get("normal_prefill_cuda_graph") is not expected_graph:
                    errors.append("wrong recorded CUDA Graph mode")
            row = {
                "model": model,
                "status": "failed" if errors else "completed",
                "mode": mode,
                "first10_matches": matches,
                "prefill_ms": data.get("prefill_time_ms") if data else None,
                "decode_ms": data.get("decode_time_ms") if data else None,
                "decode_step_ms": data.get("decode_step_time_ms") if data else None,
                "prefill_speedup_vs_eager": None,
                "decode_ratio_vs_eager": None,
                "invalid_token_count": invalid,
                "incomplete_requests": incomplete,
                "reason": "; ".join(errors),
            }
            model_rows.append(row)
            all_passed &= not errors
        baseline = model_rows[0]
        for row in model_rows:
            if row["prefill_ms"] and baseline["prefill_ms"]:
                row["prefill_speedup_vs_eager"] = baseline["prefill_ms"] / row["prefill_ms"]
            if row["decode_ms"] and baseline["decode_ms"]:
                row["decode_ratio_vs_eager"] = row["decode_ms"] / baseline["decode_ms"]
            print(
                f"{model} {row['mode']}: "
                f"{'PASS' if row['status'] == 'completed' else 'FAIL'}; "
                f"prefill={row['prefill_ms']} ms; "
                f"speedup={row['prefill_speedup_vs_eager']}; "
                f"first-10={row['first10_matches']}/10",
                flush=True,
            )
        rows.extend(model_rows)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 12,
        "status": "passed" if all_passed else "failed",
        "warmup_runs": 1,
        "measured_runs": 1,
        "batch_size": 1,
        "s_in": 128,
        "s_out": 128,
        "correctness_gate": "First 10 generated tokens equal Torch",
        "rows": rows,
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Step 12 prefill CUDA Graph: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
