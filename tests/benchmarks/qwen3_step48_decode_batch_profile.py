"""Profile Qwen3 decode across the B=4..8 transition."""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from qwen3_step15_mpk_window_profile import (
    COMPARE_TOKENS,
    WINDOWS,
    command,
    run,
    safe,
    tokens_by_request,
)
from qwen3_step18_mpk_concurrency import analyze


ROOT = Path(__file__).resolve().parents[2]
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"
S_IN = 1024
S_OUT = 128
MAX_SEQ = 1152


def dependency_wait_metrics(profile_csv):
    with profile_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    waits = [row for row in rows if row["task_type_name"] == "TASK_GET_EVENT"]
    compute = [row for row in rows if row["task_type_name"] != "TASK_GET_EVENT"]
    wait_ns = sum(int(row["duration_ns"]) for row in waits)
    compute_ns = sum(int(row["duration_ns"]) for row in compute)
    by_block = defaultdict(lambda: [0, 0])
    for row in waits:
        by_block[int(row["block_idx"])][0] += int(row["duration_ns"])
    for row in compute:
        by_block[int(row["block_idx"])][1] += int(row["duration_ns"])
    shares = sorted(
        wait / (wait + work) if wait + work else 0.0
        for wait, work in by_block.values()
    )
    percentile = lambda fraction: (
        shares[round((len(shares) - 1) * fraction)] if shares else None
    )
    return {
        "dependency_wait_events": len(waits),
        "dependency_wait_worker_ms": wait_ns / 1e6,
        "compute_worker_ms": compute_ns / 1e6,
        "dependency_wait_share": (
            wait_ns / (wait_ns + compute_ns) if wait_ns + compute_ns else None
        ),
        "dependency_wait_block_p10": percentile(0.1),
        "dependency_wait_block_p50": percentile(0.5),
        "dependency_wait_block_p90": percentile(0.9),
    }


def find_step47_row(data, model, batch):
    matches = [
        row for row in data.get("rows", [])
        if row.get("model") == model
        and row.get("case") == "long_context"
        and int(row.get("batch_size", -1)) == batch
    ]
    return matches[0] if len(matches) == 1 else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument("--step47-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step47 = json.loads(args.step47_comparison.read_text(encoding="utf-8"))

    reference_path = args.output_dir / "torch.json"
    print("Running Torch correctness reference...", flush=True)
    error = run(
        command(args, 1, S_IN, MAX_SEQ, reference_path),
        args.output_dir / "torch.log", args.timeout,
    )
    if error:
        raise RuntimeError(f"Torch reference failed: {error}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["token_ids"]

    rows = []
    failed = 0
    for batch in args.batch_sizes:
        stem = f"{safe(args.model)}_long_context_b{batch}"
        control_path = args.output_dir / f"{stem}_control.json"
        cache = args.output_dir / "cache" / stem
        print(f"Running unprofiled decode control B={batch}...", flush=True)
        error = run(
            command(args, batch, S_IN, MAX_SEQ, control_path, cache),
            args.output_dir / f"{stem}_control.log", args.timeout,
        )
        if error:
            raise RuntimeError(f"B={batch} control failed: {error}")
        control = json.loads(control_path.read_text(encoding="utf-8"))
        control_tokens = tokens_by_request(control)
        first10 = min(
            sum(a == b for a, b in zip(reference[:COMPARE_TOKENS], tokens[:COMPARE_TOKENS]))
            for tokens in control_tokens
        )
        if first10 != COMPARE_TOKENS:
            raise ValueError(f"B={batch} control correctness failed: first-10={first10}")
        baseline = find_step47_row(step47, args.model, batch)

        for window in WINDOWS:
            name, start, steps = window
            case_dir = args.output_dir / f"{stem}_{name}"
            case_dir.mkdir(exist_ok=True)
            output = case_dir / "tokens.json"
            trace = case_dir / "mpk_profile"
            profile_cache = args.output_dir / "cache" / f"{stem}_{name}"
            print(f"Profiling B={batch} {name} steps={start}-{start + steps - 1}...", flush=True)
            error = run(
                command(
                    args, batch, S_IN, MAX_SEQ, output,
                    profile_cache, trace, window,
                ),
                case_dir / "run.log", args.timeout,
            )
            reasons = [error] if error else []
            profile_summary = {}
            concurrency = {}
            waits = {}
            full_matches = invalid = incomplete = None
            if not error:
                data = json.loads(output.read_text(encoding="utf-8"))
                actual = tokens_by_request(data)
                full_matches = min(
                    sum(a == b for a, b in zip(expected, observed))
                    for expected, observed in zip(control_tokens, actual)
                ) if len(actual) == batch else None
                invalid_counts = data.get("invalid_token_counts_by_request", [])
                lengths = data.get("generate_lengths_by_request", [])
                invalid = sum(invalid_counts) if len(invalid_counts) == batch else None
                incomplete = sum(length != S_OUT for length in lengths) if len(lengths) == batch else None
                if full_matches != S_OUT:
                    reasons.append(f"full matches={full_matches}/{S_OUT}")
                if invalid != 0 or incomplete != 0:
                    reasons.append(f"invalid={invalid}, incomplete={incomplete}")
                summary_dir = case_dir / "summary"
                rc = subprocess.run([
                    sys.executable, str(SUMMARIZER), str(trace) + ".csv",
                    "--output-dir", str(summary_dir),
                ], cwd=ROOT).returncode
                if rc:
                    reasons.append("profile summary failed")
                else:
                    profile_summary = json.loads(
                        (summary_dir / "profile_summary.json").read_text(encoding="utf-8")
                    )
                    concurrency = analyze(Path(str(trace) + ".csv"))
                    waits = dependency_wait_metrics(Path(str(trace) + ".csv"))

            shares = {
                item["category"]: item["worker_time_share"]
                for item in profile_summary.get("categories", [])
            }
            categories = {
                item["category"]: item
                for item in concurrency.get("categories", [])
            }
            row = {
                "model": args.model, "case": "long_context", "batch_size": batch,
                "window": name, "window_start": start, "window_steps": steps,
                "status": "failed" if reasons else "passed",
                "minimum_full_matches": full_matches,
                "invalid_token_count": invalid, "incomplete_requests": incomplete,
                "control_decode_step_ms": control.get("decode_step_ms"),
                "step47_mpk_decode_step_ms": baseline.get("mpk_decode_step_ms_median"),
                "step47_sglang_decode_step_ms": baseline.get("sglang_decode_step_ms_median"),
                "step47_decode_mpk_over_sglang": baseline.get("decode_step_ms_mpk_over_sglang"),
                "profile_span_ms": concurrency.get("span_ms"),
                "average_concurrent_tasks": concurrency.get("average_concurrent_tasks"),
                "global_idle_fraction": concurrency.get("global_idle_fraction"),
                "block_utilization_p50": concurrency.get("block_utilization_p50"),
                "linear_worker_share": shares.get("linear"),
                "attention_worker_share": shares.get("attention"),
                "linear_average_concurrency": categories.get("linear", {}).get("average_concurrency"),
                "attention_average_concurrency": categories.get("attention", {}).get("average_concurrency"),
                **waits,
                "case_dir": str(case_dir), "reason": "; ".join(reasons),
            }
            rows.append(row)
            failed += bool(reasons)
            print(
                f"B={batch} {name:6s}: {'PASS' if not reasons else 'FAIL'}; "
                f"wall ratio={row['step47_decode_mpk_over_sglang']}; "
                f"linear/attention={100*(shares.get('linear') or 0):.1f}/"
                f"{100*(shares.get('attention') or 0):.1f}%; "
                f"dependency wait={100*(waits.get('dependency_wait_share') or 0):.1f}%",
                flush=True,
            )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 48, "phase": "decode_batch_transition_profile",
        "status": "passed" if not failed else "failed",
        "model": args.model, "batch_sizes": args.batch_sizes,
        "s_in": S_IN, "s_out": S_OUT,
        "performance_scope": "decode only; prefill latency is intentionally excluded",
        "windows": [
            {"name": name, "start": start, "steps": steps}
            for name, start, steps in WINDOWS
        ],
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Step 48 decode batch profile: {summary['status'].upper()}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
