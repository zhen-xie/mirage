"""Profile early, middle, and late MPK decode windows."""

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
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"
S_OUT = 128
COMPARE_TOKENS = 10
WINDOWS = (
    ("early", 1, 9),
    ("middle", 60, 9),
    ("late", 119, 9),
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


def run(command, log, timeout):
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return "timeout"
    return "" if code == 0 else f"exit code {code}"


def command(args, batch, s_in, max_seq, output, cache=None, trace=None, window=None):
    result = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(S_OUT), "--page-size", str(max_seq),
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
            "--normal-prefill-attention", "sdpa",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    if trace is not None:
        _, start, steps = window
        result += [
            "--profiling", "--trace-name", str(trace),
            "--profiler-buffer-entries-per-block", str(args.profiler_entries_per_block),
            "--profiler-decode-start-step", str(start),
            "--profiler-decode-num-steps", str(steps),
        ]
    return result


def tokens_by_request(data):
    return data.get("token_ids_by_request") or [data.get("token_ids", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument("--step14-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = (
        ("short", 1, 128, 256),
        ("short", 32, 128, 256),
        ("long_context", 32, 1024, 1152),
    )
    step14_data = json.loads(args.step14_summary.read_text(encoding="utf-8"))
    step14 = {
        (row["model"], row["case"], int(row["batch_size"])): row
        for row in step14_data["rows"] if row.get("status") == "completed"
    }
    model_name = safe(args.model)
    references = {}
    controls = {}
    rows = []
    failed = 0

    for case_name, batch, s_in, max_seq in cases:
        if s_in not in references:
            output = args.output_dir / f"torch_in{s_in}.json"
            log = args.output_dir / f"torch_in{s_in}.log"
            print(f"Running Torch reference S_IN={s_in}...", flush=True)
            error = run(command(args, 1, s_in, max_seq, output), log, args.timeout)
            if error:
                raise RuntimeError(f"Torch reference failed: {error}; see {log}")
            references[s_in] = json.loads(output.read_text(encoding="utf-8"))["token_ids"]

        case_stem = f"{model_name}_{case_name}_b{batch}"
        control_output = args.output_dir / f"{case_stem}_unprofiled.json"
        control_log = args.output_dir / f"{case_stem}_unprofiled.log"
        control_cache = args.output_dir / "cache" / f"{case_stem}_unprofiled"
        print(f"Running unprofiled control {case_stem}...", flush=True)
        error = run(
            command(args, batch, s_in, max_seq, control_output, control_cache),
            control_log, args.timeout,
        )
        if error:
            raise RuntimeError(f"MPK control failed: {error}; see {control_log}")
        control = json.loads(control_output.read_text(encoding="utf-8"))
        controls[(case_name, batch)] = tokens_by_request(control)
        reference = references[s_in]
        control_first10 = min(
            sum(a == b for a, b in zip(reference[:COMPARE_TOKENS], value[:COMPARE_TOKENS]))
            for value in controls[(case_name, batch)]
        )
        if control_first10 != COMPARE_TOKENS:
            raise ValueError(f"Unprofiled control first-10 failed for {case_stem}")

        for window_name, start, steps in WINDOWS:
            stem = f"{case_stem}_{window_name}"
            case_dir = args.output_dir / stem
            case_dir.mkdir(exist_ok=True)
            output = case_dir / "tokens.json"
            log = case_dir / "run.log"
            trace = case_dir / "mpk_profile"
            cache = args.output_dir / "cache" / stem
            print(
                f"Profiling {case_stem} window={window_name} "
                f"steps={start}-{start + steps - 1}...",
                flush=True,
            )
            error = run(
                command(
                    args, batch, s_in, max_seq, output, cache, trace,
                    (window_name, start, steps),
                ),
                log, args.timeout,
            )
            reasons = [error] if error else []
            data = profile = None
            first10 = full_matches = invalid = incomplete = None
            if not error:
                data = json.loads(output.read_text(encoding="utf-8"))
                actual = tokens_by_request(data)
                expected_batch = controls[(case_name, batch)]
                first10_values = [
                    sum(a == b for a, b in zip(reference[:10], value[:10]))
                    for value in actual
                ]
                first10 = min(first10_values) if first10_values else None
                full_values = [
                    sum(a == b for a, b in zip(expected, value))
                    for expected, value in zip(expected_batch, actual)
                ]
                full_matches = min(full_values) if full_values else None
                lengths = data.get("generate_lengths_by_request", [])
                invalid_counts = data.get("invalid_token_counts_by_request", [])
                invalid = sum(invalid_counts) if len(invalid_counts) == batch else None
                incomplete = sum(value != S_OUT for value in lengths) if len(lengths) == batch else None
                if len(actual) != batch or first10 != COMPARE_TOKENS:
                    reasons.append(f"minimum first-10={first10}")
                if full_matches != S_OUT:
                    reasons.append(f"minimum full matches={full_matches}/{S_OUT}")
                if invalid != 0:
                    reasons.append(f"invalid tokens={invalid}")
                if incomplete != 0:
                    reasons.append(f"incomplete requests={incomplete}")
                summary_dir = case_dir / "summary"
                result = subprocess.run([
                    sys.executable, str(SUMMARIZER), str(trace) + ".csv",
                    "--output-dir", str(summary_dir),
                ], cwd=ROOT)
                if result.returncode:
                    reasons.append("profile summary failed")
                else:
                    profile = json.loads(
                        (summary_dir / "profile_summary.json").read_text(encoding="utf-8")
                    )

            shares = {
                item["category"]: item["worker_time_share"]
                for item in (profile or {}).get("categories", [])
            }
            baseline = step14.get((args.model, case_name, batch), {})
            row = {
                "model": args.model, "case": case_name, "batch_size": batch,
                "s_in": s_in, "s_out": S_OUT,
                "window": window_name, "window_start": start,
                "window_steps": steps,
                "status": "failed" if reasons else "completed",
                "minimum_first10_matches": first10,
                "minimum_full_matches": full_matches,
                "invalid_token_count": invalid,
                "incomplete_requests": incomplete,
                "resolved_attention": data.get("mpk_attention") if data else None,
                "step14_decode_step_ms": baseline.get("decode_step_ms"),
                "linear_worker_share": shares.get("linear"),
                "attention_worker_share": shares.get("attention"),
                "norm_worker_share": shares.get("norm"),
                "activation_worker_share": shares.get("activation"),
                "sampling_worker_share": shares.get("sampling"),
                "profile_paired_events": (profile or {}).get("paired_events"),
                "case_dir": str(case_dir), "reason": "; ".join(reasons),
            }
            rows.append(row)
            failed += bool(reasons)
            print(
                f"{stem}: {'PASS' if not reasons else 'FAIL'}; "
                f"first10={first10}; full={full_matches}/{S_OUT}; "
                f"linear={100 * (shares.get('linear') or 0):.1f}%; "
                f"attention={100 * (shares.get('attention') or 0):.1f}%",
                flush=True,
            )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 15, "phase": "mpk_window_profile",
        "status": "passed" if not failed else "failed",
        "windows": [
            {"name": name, "start": start, "steps": steps}
            for name, start, steps in WINDOWS
        ],
        "correctness_gate": (
            "Every request matches Torch for the first 10 tokens and its "
            "unprofiled MPK control for all 128 tokens"
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Step 15 windowed MPK profile completed with {failed} failed window(s).")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
