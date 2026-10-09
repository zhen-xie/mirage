"""Compare absolute MPK and SGLang operator time for B=1 decode."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from qwen3_step18_mpk_concurrency import analyze as analyze_mpk
from summarize_qwen3_sglang_profile import category, is_cuda_kernel, read_trace


WINDOW_ORDER = {"early": 0, "middle": 1, "late": 2}
CATEGORIES = ("linear", "attention", "norm", "activation", "sampling", "other")


def milliseconds(value):
    return "--" if value is None else f"{value:.3f}"


def merge_length(intervals):
    if not intervals:
        return 0.0
    start, end = sorted(intervals)[0]
    total = 0.0
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def analyze_sglang(path):
    intervals = defaultdict(list)
    sums = defaultdict(float)
    calls = defaultdict(int)
    all_intervals = []
    for event in read_trace(path):
        if not is_cuda_kernel(event):
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event["dur"])
        if duration <= 0:
            continue
        group = category(str(event.get("name", "<unnamed>")))
        interval = (start, start + duration)
        intervals[group].append(interval)
        all_intervals.append(interval)
        sums[group] += duration
        calls[group] += 1
    if not all_intervals:
        raise ValueError(f"no CUDA kernels in {path}")
    span_us = max(end for _, end in all_intervals) - min(start for start, _ in all_intervals)
    return {
        "span_ms": span_us / 1000.0,
        "active_union_ms": merge_length(all_intervals) / 1000.0,
        "categories": {
            name: {
                "calls": calls[name],
                "summed_ms": sums[name] / 1000.0,
                "active_union_ms": merge_length(intervals[name]) / 1000.0,
            }
            for name in set(intervals)
        },
    }


def mpk_metrics(row):
    profile = analyze_mpk(Path(row["case_dir"]) / "mpk_profile.csv")
    categories = {
        item["category"]: {
            "calls": item["events"],
            "summed_ms": item["activity_ms"],
            "active_union_ms": item["union_coverage"] * profile["span_ms"],
        }
        for item in profile["categories"]
    }
    return {
        "span_ms": profile["span_ms"],
        "active_union_ms": profile["busy_union_ms"],
        "categories": categories,
    }


def sglang_trace(row):
    value = row.get("profile_trace")
    if not value:
        raise ValueError(f"missing SGLang trace for {row.get('window')}")
    return Path(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-summary", type=Path, required=True)
    parser.add_argument("--sglang-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpk_data = json.loads(args.mpk_summary.read_text(encoding="utf-8"))
    sg_data = json.loads(args.sglang_summary.read_text(encoding="utf-8"))
    mpk_rows = {
        row["window"]: row for row in mpk_data["rows"]
        if row["model"] == "Qwen/Qwen3-8B"
        and row["case"] == "long_context"
        and int(row["batch_size"]) == 1
    }
    sg_rows = {
        row["window"]: row for row in sg_data["rows"]
        if row["model"] == "Qwen/Qwen3-8B"
        and row["case"] == "long_context"
        and int(row["batch_size"]) == 1
    }
    expected = set(WINDOW_ORDER)
    errors = []
    if set(mpk_rows) != expected or set(sg_rows) != expected:
        errors.append(
            f"window mismatch: MPK={sorted(mpk_rows)}, SGLang={sorted(sg_rows)}"
        )

    rows = []
    for window in sorted(expected & set(mpk_rows) & set(sg_rows), key=WINDOW_ORDER.get):
        mpk_row = mpk_rows[window]
        sg_row = sg_rows[window]
        reasons = []
        if mpk_row.get("status") not in {"passed", "completed"}:
            reasons.append("MPK profile failed")
        if sg_row.get("status") != "completed":
            reasons.append("SGLang profile failed")
        try:
            mpk = mpk_metrics(mpk_row)
        except Exception as error:
            mpk = {"span_ms": None, "active_union_ms": None, "categories": {}}
            reasons.append(f"MPK {type(error).__name__}: {error}")
        try:
            sglang = analyze_sglang(sglang_trace(sg_row))
        except Exception as error:
            sglang = {"span_ms": None, "active_union_ms": None, "categories": {}}
            reasons.append(f"SGLang {type(error).__name__}: {error}")

        base = {
            "window": window,
            "window_steps": int(mpk_row.get("window_steps", 9)),
            "status": "failed" if reasons else "passed",
            "mpk_window_span_ms": mpk.get("span_ms"),
            "sglang_window_span_ms": sglang.get("span_ms"),
            "mpk_active_union_ms": mpk.get("active_union_ms"),
            "sglang_active_union_ms": sglang.get("active_union_ms"),
            "reason": "; ".join(reasons),
        }
        steps = base["window_steps"]
        for name in CATEGORIES:
            mpk_item = mpk.get("categories", {}).get(name, {})
            sg_item = sglang.get("categories", {}).get(name, {})
            mpk_active = mpk_item.get("active_union_ms", 0.0)
            sg_active = sg_item.get("active_union_ms", 0.0)
            base.update({
                f"mpk_{name}_active_ms_per_step": mpk_active / steps,
                f"sglang_{name}_active_ms_per_step": sg_active / steps,
                f"{name}_active_mpk_over_sglang": (
                    mpk_active / sg_active if sg_active else None
                ),
                f"mpk_{name}_summed_ms_per_step": mpk_item.get("summed_ms", 0.0) / steps,
                f"sglang_{name}_summed_ms_per_step": sg_item.get("summed_ms", 0.0) / steps,
                f"mpk_{name}_calls_per_step": mpk_item.get("calls", 0) / steps,
                f"sglang_{name}_calls_per_step": sg_item.get("calls", 0) / steps,
            })
        rows.append(base)
        errors.extend(reasons)
        print(
            f"{window:6s}: {'PASS' if not reasons else 'FAIL'}; "
            f"span MPK/SGLang={milliseconds(base['mpk_window_span_ms'])}/"
            f"{milliseconds(base['sglang_window_span_ms'])} ms; "
            f"linear={milliseconds(base['mpk_linear_active_ms_per_step'])}/"
            f"{milliseconds(base['sglang_linear_active_ms_per_step'])} ms/step; "
            f"attention={milliseconds(base['mpk_attention_active_ms_per_step'])}/"
            f"{milliseconds(base['sglang_attention_active_ms_per_step'])} ms/step",
            flush=True,
        )

    if len(rows) != 3:
        errors.append(f"expected 3 windows, found {len(rows)}")
    if rows:
        with (args.output_dir / "comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "step": 49, "phase": "b1_absolute_operator_time_comparison",
        "status": "passed" if not errors else "failed",
        "model": "Qwen/Qwen3-8B", "batch_size": 1,
        "s_in": 1024, "s_out": 128,
        "measurement": (
            "active_ms_per_step is the union of device timeline intervals in each "
            "operator category. summed_ms_per_step retains overlapping worker or "
            "kernel work and is not used as wall time. Categories may overlap."
        ),
        "rows": rows, "errors": errors,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Step 49 B=1 operator comparison: {summary['status'].upper()}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
