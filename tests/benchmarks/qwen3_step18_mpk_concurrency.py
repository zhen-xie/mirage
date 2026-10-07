"""Analyze MPK task concurrency and worker balance from windowed profiles."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


MODULUS = 1 << 32


def category(name):
    if "ATTENTION" in name or "TASK_ATTN" in name:
        return "attention"
    if "LINEAR" in name:
        return "linear"
    if "NORM" in name:
        return "norm"
    if "SILU" in name:
        return "activation"
    if "ARGMAX" in name or "SAMPLING" in name:
        return "sampling"
    if "EMBEDDING" in name:
        return "embedding"
    return "other"


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def circular_origin(timestamps):
    values = sorted(set(timestamps))
    if not values:
        raise ValueError("profile has no timestamps")
    gaps = []
    for index, value in enumerate(values):
        following = values[(index + 1) % len(values)]
        gap = (following - value) % MODULUS
        gaps.append((gap, following))
    return max(gaps)[1]


def merge_length(intervals):
    if not intervals:
        return 0
    merged = 0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged += end - start
            start, end = next_start, next_end
    return merged + end - start


def coefficient_of_variation(values):
    if not values:
        return 0.0
    average = sum(values) / len(values)
    if average == 0:
        return 0.0
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / average


def analyze(path):
    with path.open(newline="", encoding="utf-8") as source:
        raw = list(csv.DictReader(source))
    if not raw:
        raise ValueError(f"empty profile: {path}")
    timestamps = [int(row[key]) for row in raw for key in ("begin_ts", "end_ts")]
    origin = circular_origin(timestamps)
    rows = []
    for row in raw:
        duration = int(row["duration_ns"])
        if duration <= 0:
            continue
        begin = (int(row["begin_ts"]) - origin) % MODULUS
        end = begin + duration
        rows.append({
            "task": row["task_type_name"],
            "category": category(row["task_type_name"]),
            "block": int(row["block_idx"]),
            "group": int(row["group_idx"]),
            "begin": begin, "end": end, "duration": duration,
        })
    start = min(row["begin"] for row in rows)
    end = max(row["end"] for row in rows)
    span = end - start
    if span <= 0 or span >= MODULUS // 2:
        raise ValueError(f"invalid reconstructed profile span: {span} ns")

    all_intervals = [(row["begin"], row["end"]) for row in rows]
    busy = merge_length(all_intervals)
    total_activity = sum(row["duration"] for row in rows)
    by_category = defaultdict(list)
    by_task = defaultdict(list)
    by_block = defaultdict(list)
    for row in rows:
        interval = (row["begin"], row["end"])
        by_category[row["category"]].append(interval)
        by_task[row["task"]].append(row["duration"])
        by_block[row["block"]].append(interval)

    category_rows = []
    for name, intervals in sorted(by_category.items()):
        activity = sum(end - begin for begin, end in intervals)
        union = merge_length(intervals)
        category_rows.append({
            "category": name,
            "events": len(intervals),
            "activity_ms": activity / 1e6,
            "activity_share": activity / total_activity,
            "union_coverage": union / span,
            "average_concurrency": activity / span,
        })

    block_utilizations = [merge_length(intervals) / span for intervals in by_block.values()]
    task_rows = []
    for name, durations in sorted(by_task.items(), key=lambda item: sum(item[1]), reverse=True):
        task_rows.append({
            "task": name, "category": category(name), "events": len(durations),
            "activity_ms": sum(durations) / 1e6,
            "mean_us": sum(durations) / len(durations) / 1e3,
            "p90_us": percentile(durations, 0.9) / 1e3,
        })
    return {
        "source": str(path),
        "span_ms": span / 1e6,
        "busy_union_ms": busy / 1e6,
        "global_idle_fraction": 1.0 - busy / span,
        "total_worker_activity_ms": total_activity / 1e6,
        "average_concurrent_tasks": total_activity / span,
        "profiled_blocks": len(by_block),
        "block_utilization_mean": sum(block_utilizations) / len(block_utilizations),
        "block_utilization_p10": percentile(block_utilizations, 0.1),
        "block_utilization_p50": percentile(block_utilizations, 0.5),
        "block_utilization_p90": percentile(block_utilizations, 0.9),
        "block_utilization_cv": coefficient_of_variation(block_utilizations),
        "categories": category_rows,
        "top_tasks": task_rows[:20],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step15-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.step15_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in source["rows"]
        if row["model"] == "Qwen/Qwen3-8B"
        and row["case"] == "long_context"
        and int(row["batch_size"]) == 32
    ]
    errors = []
    results = []
    for row in sorted(selected, key=lambda item: item["window_start"]):
        reasons = []
        if row.get("status") != "completed":
            reasons.append("Step 15 profile failed")
        if row.get("minimum_first10_matches") != 10:
            reasons.append("first-10 correctness failed")
        if row.get("minimum_full_matches") != 128:
            reasons.append("full-output profiler parity failed")
        if row.get("invalid_token_count") != 0 or row.get("incomplete_requests") != 0:
            reasons.append("invalid or incomplete output")
        profile_csv = Path(row["case_dir"]) / "mpk_profile.csv"
        try:
            metrics = analyze(profile_csv)
        except Exception as exc:
            metrics = {}
            reasons.append(f"{type(exc).__name__}: {exc}")
        result = {
            "window": row["window"],
            "window_start": row["window_start"],
            "window_steps": row["window_steps"],
            "status": "failed" if reasons else "completed",
            "first10_matches": row.get("minimum_first10_matches"),
            "full_matches": row.get("minimum_full_matches"),
            **metrics,
            "reason": "; ".join(reasons),
        }
        results.append(result)
        errors.extend(reasons)
        categories = {item["category"]: item for item in metrics.get("categories", [])}
        attention = categories.get("attention", {})
        linear = categories.get("linear", {})
        print(
            f"{row['window']:6s}: {'PASS' if not reasons else 'FAIL'}; "
            f"span={metrics.get('span_ms', 0):.3f} ms; "
            f"avg concurrency={metrics.get('average_concurrent_tasks', 0):.1f}; "
            f"idle={100 * metrics.get('global_idle_fraction', 0):.2f}%; "
            f"block util p10/p50/p90="
            f"{100 * metrics.get('block_utilization_p10', 0):.1f}/"
            f"{100 * metrics.get('block_utilization_p50', 0):.1f}/"
            f"{100 * metrics.get('block_utilization_p90', 0):.1f}%; "
            f"attention concurrency={attention.get('average_concurrency', 0):.1f}; "
            f"linear concurrency={linear.get('average_concurrency', 0):.1f}"
        )

    if len(results) != 3:
        errors.append(f"expected 3 windows, found {len(results)}")
    summary = {
        "step": 18, "phase": "mpk_concurrency_and_balance",
        "status": "passed" if not errors else "failed",
        "model": "Qwen/Qwen3-8B", "batch_size": 32,
        "s_in": 1024, "s_out": 128,
        "measurement_note": (
            "Intervals are reconstructed from device globaltimer timestamps. Category "
            "activity may overlap. average_concurrency is summed task duration divided "
            "by window span; union_coverage is wall-time coverage by at least one task "
            "in that category. Profiler instrumentation changes absolute latency, so "
            "use these metrics for concurrency and balance rather than performance."
        ),
        "windows": results, "errors": errors,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    flat = []
    for result in results:
        for item in result.get("categories", []):
            flat.append({
                "window": result["window"], "span_ms": result.get("span_ms"),
                "global_idle_fraction": result.get("global_idle_fraction"),
                "block_utilization_p10": result.get("block_utilization_p10"),
                "block_utilization_p50": result.get("block_utilization_p50"),
                "block_utilization_p90": result.get("block_utilization_p90"),
                **item,
            })
    if flat:
        with (args.output_dir / "category_concurrency.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print(f"Step 18 MPK concurrency analysis: {summary['status'].upper()}")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
