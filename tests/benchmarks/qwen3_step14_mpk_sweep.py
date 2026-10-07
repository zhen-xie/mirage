"""Run a resumable representative MPK performance sweep."""

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
DEMO = ROOT / "demo/qwen3/demo.py"
COMPARE_TOKENS = 10
FIELDS = (
    "model", "case", "batch_size", "s_in", "s_out", "status",
    "minimum_first10_matches", "passing_requests", "invalid_token_count",
    "incomplete_requests", "resolved_attention", "prefill_ms", "decode_ms",
    "decode_step_ms", "decode_tokens_per_second", "kernel_cache_status",
    "kernel_prepare_ms", "output_path", "log_path", "reason",
)


def safe(value):
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


def command(args, model, batch, s_in, s_out, output, cache=None):
    max_seq = s_in + s_out
    result = [
        sys.executable, str(DEMO), "--model", model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(s_out), "--page-size", str(max_seq),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if cache is not None:
        result += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1",
            "--normal-prefill-cuda-graph",
        ]
    return result


def write_summary(args, rows):
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = all(row["status"] == "completed" for row in rows)
    expected = len(args.models) * len(args.cases) * len(args.batch_sizes)
    summary = {
        "step": 14,
        "status": "passed" if passed and len(rows) == expected else "incomplete",
        "warmup_runs": 1, "measured_runs": 1,
        "models": args.models, "cases": args.cases,
        "batch_sizes": args.batch_sizes,
        "correctness_gate": "Every MPK request matches Torch for its first 10 generated tokens",
        "timing_note": "The first generated token belongs to prefill; decode contains S_OUT-1 steps.",
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", default=[
        "short:128:128", "long_context:1024:128",
        "long_generation:128:1024",
    ])
    parser.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parsed_cases = []
    for value in args.cases:
        try:
            name, s_in, s_out = value.split(":")
            s_in, s_out = int(s_in), int(s_out)
        except ValueError:
            parser.error(f"invalid case {value!r}; expected NAME:S_IN:S_OUT")
        if min(s_in, s_out) <= 0 or (s_in + s_out) % 64:
            parser.error(f"case {value!r} must be positive and sum to a multiple of 64")
        parsed_cases.append((name, s_in, s_out))
    args.cases = [f"{name}:{s_in}:{s_out}" for name, s_in, s_out in parsed_cases]
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.output_dir / "cache"
    cache_root.mkdir(exist_ok=True)

    summary_path = args.output_dir / "summary.json"
    rows = []
    if summary_path.is_file():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [row for row in old.get("rows", []) if row.get("status") == "completed"]
        print(f"Resuming with {len(rows)} completed result(s)", flush=True)
    completed = {
        (row["model"], row["case"], int(row["batch_size"])) for row in rows
    }

    failed = 0
    total = len(args.models) * len(parsed_cases) * len(args.batch_sizes)
    index = 0
    for model in args.models:
        model_name = safe(model)
        for case_name, s_in, s_out in parsed_cases:
            reference_output = args.output_dir / f"{model_name}_{case_name}_torch.json"
            reference_log = args.output_dir / f"{model_name}_{case_name}_torch.log"
            if reference_output.is_file():
                reference = json.loads(reference_output.read_text(encoding="utf-8"))
            else:
                print(f"Running Torch reference model={model} case={case_name}", flush=True)
                reference, error = run(
                    command(args, model, 1, s_in, s_out, reference_output),
                    reference_output, reference_log, args.timeout,
                )
                if reference is None:
                    raise RuntimeError(f"Torch reference failed: {error}; see {reference_log}")
            expected = reference.get("token_ids", [])
            if len(expected) < min(COMPARE_TOKENS, s_out):
                raise ValueError(f"incomplete Torch reference: {reference_output}")

            for batch in args.batch_sizes:
                index += 1
                key = (model, case_name, batch)
                if key in completed:
                    print(f"[{index}/{total}] Skipping completed {key}", flush=True)
                    continue
                stem = f"{model_name}_{case_name}_b{batch}"
                cache = cache_root / f"{model_name}_b{batch}_seq{s_in + s_out}"
                warm_output = args.output_dir / f"{stem}_warmup.json"
                warm_log = args.output_dir / f"{stem}_warmup.log"
                output = args.output_dir / f"{stem}.json"
                log = args.output_dir / f"{stem}.log"
                print(
                    f"[{index}/{total}] Warmup model={model} case={case_name} "
                    f"B={batch} S_IN={s_in} S_OUT={s_out}", flush=True,
                )
                warm, error = run(
                    command(args, model, batch, s_in, s_out, warm_output, cache),
                    warm_output, warm_log, args.timeout,
                )
                data = None
                if warm is not None:
                    print(f"[{index}/{total}] Measuring {stem}", flush=True)
                    data, error = run(
                        command(args, model, batch, s_in, s_out, output, cache),
                        output, log, args.timeout,
                    )
                errors = [error] if error else []
                matches = []
                invalid = incomplete = None
                if data is not None:
                    tokens = data.get("token_ids_by_request", [])
                    lengths = data.get("generate_lengths_by_request", [])
                    invalid_counts = data.get("invalid_token_counts_by_request", [])
                    matches = [
                        sum(a == b for a, b in zip(expected[:COMPARE_TOKENS], value[:COMPARE_TOKENS]))
                        for value in tokens
                    ]
                    invalid = sum(invalid_counts) if len(invalid_counts) == batch else None
                    incomplete = sum(value != s_out for value in lengths) if len(lengths) == batch else None
                    if len(tokens) != batch or not matches or min(matches) != min(COMPARE_TOKENS, s_out):
                        errors.append(f"minimum first-10={min(matches) if matches else None}")
                    if invalid != 0:
                        errors.append(f"invalid tokens={invalid}")
                    if incomplete != 0:
                        errors.append(f"incomplete requests={incomplete}")
                decode_ms = data.get("decode_time_ms") if data else None
                decode_steps = data.get("decode_steps") if data else None
                throughput = (
                    1000.0 * batch * decode_steps / decode_ms
                    if decode_ms and decode_steps else None
                )
                row = {
                    "model": model, "case": case_name, "batch_size": batch,
                    "s_in": s_in, "s_out": s_out,
                    "status": "failed" if errors else "completed",
                    "minimum_first10_matches": min(matches) if matches else None,
                    "passing_requests": sum(value == min(COMPARE_TOKENS, s_out) for value in matches),
                    "invalid_token_count": invalid,
                    "incomplete_requests": incomplete,
                    "resolved_attention": data.get("mpk_attention") if data else None,
                    "prefill_ms": data.get("prefill_time_ms") if data else None,
                    "decode_ms": decode_ms,
                    "decode_step_ms": data.get("decode_step_time_ms") if data else None,
                    "decode_tokens_per_second": throughput,
                    "kernel_cache_status": data.get("mpk_kernel_cache_status") if data else None,
                    "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
                    "output_path": str(output), "log_path": str(log),
                    "reason": "; ".join(errors),
                }
                rows = [item for item in rows if (item["model"], item["case"], int(item["batch_size"])) != key]
                rows.append(row)
                rows.sort(key=lambda item: (args.models.index(item["model"]), [x[0] for x in parsed_cases].index(item["case"]), int(item["batch_size"])))
                write_summary(args, rows)
                failed += row["status"] != "completed"
                print(
                    f"{stem}: {'PASS' if not errors else 'FAIL'}; "
                    f"first-10={row['minimum_first10_matches']}; "
                    f"decode={row['decode_step_ms']} ms/step; throughput={throughput}",
                    flush=True,
                )
    write_summary(args, rows)
    print(f"Step 14 MPK sweep completed with {failed} failed case(s).")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
