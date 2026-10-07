"""Compare windowed MPK and SGLang profiles with end-to-end Step 14 timing."""

import argparse
import csv
import json
from pathlib import Path


WINDOW_ORDER = {"early": 0, "middle": 1, "late": 2}


def key(row):
    return row["model"], row["case"], int(row["batch_size"]), row["window"]


def timing_key(row):
    return row["model"], row["case"], int(row["batch_size"])


def mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def percent(value):
    return "--" if value is None else f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-summary", type=Path, required=True)
    parser.add_argument("--sglang-summary", type=Path, required=True)
    parser.add_argument("--step14-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpk_data = json.loads(args.mpk_summary.read_text(encoding="utf-8"))
    sg_data = json.loads(args.sglang_summary.read_text(encoding="utf-8"))
    timing_data = json.loads(args.step14_comparison.read_text(encoding="utf-8"))
    mpk = {key(row): row for row in mpk_data["rows"]}
    sglang = {key(row): row for row in sg_data["rows"]}
    timings = {timing_key(row): row for row in timing_data["rows"]}

    expected = set(mpk)
    errors = []
    if expected != set(sglang):
        errors.append(
            f"profile key mismatch: MPK-only={sorted(expected - set(sglang))}; "
            f"SGLang-only={sorted(set(sglang) - expected)}"
        )
    rows = []
    for item_key in sorted(expected & set(sglang), key=lambda item: (item[:3], WINDOW_ORDER[item[3]])):
        mpk_row = mpk[item_key]
        sg_row = sglang[item_key]
        model, case, batch, window = item_key
        timing = timings.get((model, case, batch), {})
        reasons = []
        if mpk_row.get("status") != "completed":
            reasons.append("MPK profile failed")
        if sg_row.get("status") != "completed":
            reasons.append("SGLang profile failed")
        if not timing:
            reasons.append("missing Step 14 timing")
        row = {
            "model": model, "case": case, "batch_size": batch,
            "window": window,
            "mpk_decode_step_ms": timing.get("mpk_decode_step_ms"),
            "sglang_decode_step_ms": timing.get("sglang_decode_step_ms"),
            "decode_step_mpk_over_sglang": timing.get("decode_step_mpk_over_sglang"),
            "mpk_linear_worker_share": mpk_row.get("linear_worker_share"),
            "mpk_attention_worker_share": mpk_row.get("attention_worker_share"),
            "sglang_linear_kernel_share": sg_row.get("linear_kernel_share"),
            "sglang_attention_kernel_share": sg_row.get("attention_kernel_share"),
            "sglang_cuda_kernel_time_ms_9_steps": sg_row.get("cuda_kernel_time_ms"),
            "status": "failed" if reasons else "completed",
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        errors.extend(reasons)
        print(
            f"{case:12s} B={batch:2d} {window:6s}: "
            f"wall MPK/SGLang={row['decode_step_mpk_over_sglang']:.3f}x; "
            f"MPK linear/attention={percent(row['mpk_linear_worker_share'])}/"
            f"{percent(row['mpk_attention_worker_share'])}; "
            f"SGLang linear/attention={percent(row['sglang_linear_kernel_share'])}/"
            f"{percent(row['sglang_attention_kernel_share'])}"
        )

    aggregate = []
    groups = sorted({(row["model"], row["case"], row["batch_size"]) for row in rows})
    for model, case, batch in groups:
        group = [
            row for row in rows
            if (row["model"], row["case"], row["batch_size"]) == (model, case, batch)
        ]
        aggregate.append({
            "model": model, "case": case, "batch_size": batch,
            "decode_step_mpk_over_sglang": group[0]["decode_step_mpk_over_sglang"],
            "mean_mpk_linear_worker_share": mean(row["mpk_linear_worker_share"] for row in group),
            "mean_mpk_attention_worker_share": mean(row["mpk_attention_worker_share"] for row in group),
            "mean_sglang_linear_kernel_share": mean(row["sglang_linear_kernel_share"] for row in group),
            "mean_sglang_attention_kernel_share": mean(row["sglang_attention_kernel_share"] for row in group),
            "mean_sglang_cuda_kernel_time_ms_9_steps": mean(
                row["sglang_cuda_kernel_time_ms_9_steps"] for row in group
            ),
        })

    with (args.output_dir / "window_comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "case_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    summary = {
        "step": 16, "phase": "mpk_sglang_profile_comparison",
        "status": "passed" if not errors else "failed",
        "measurement_warning": (
            "MPK shares are summed persistent-worker activity. SGLang shares are summed "
            "CUDA kernel durations. They identify dominant work within each backend but "
            "must not be divided to estimate operator speedup. End-to-end Step 14 decode "
            "ratios are the comparable performance measurement."
        ),
        "sglang_correctness_limitation": sg_data.get("correctness_limitation"),
        "rows": rows, "case_summary": aggregate, "errors": errors,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Step 16 profile comparison: {summary['status'].upper()}")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
