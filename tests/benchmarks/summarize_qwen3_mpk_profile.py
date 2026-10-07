"""Validate and summarize a Mirage persistent-kernel profiler CSV."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def category(name):
    if "SCHD" in name or name in {"TASK_GET_EVENT", "TASK_GET_NEXT_TASK"}:
        return "scheduler"
    if "ATTENTION" in name or "TASK_ATTN" in name:
        return "attention"
    if "ARGMAX" in name or "SAMPLING" in name:
        return "sampling"
    if "LINEAR" in name:
        return "linear"
    if "NORM" in name:
        return "norm"
    if "SILU" in name:
        return "activation"
    if "EMBEDDING" in name:
        return "embedding"
    return "other"


def percentile(values, fraction):
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-categories", nargs="+", default=[
        "attention", "linear", "norm", "activation", "sampling",
    ])
    args = parser.parse_args()

    with args.profile_csv.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"Profile contains no paired events: {args.profile_csv}")

    by_category = defaultdict(list)
    by_task = defaultdict(list)
    unknown = []
    for row in rows:
        duration = int(row["duration_ns"])
        name = row["task_type_name"]
        by_category[category(name)].append(duration)
        by_task[name].append(duration)
        if name.startswith("UNKNOWN_"):
            unknown.append(name)

    missing = [name for name in args.minimum_categories if not by_category[name]]
    positive = [value for values in by_category.values() for value in values if value > 0]
    status = "passed" if not missing and positive and not unknown else "failed"
    total_worker_ns = sum(positive)

    category_rows = []
    for name, values in sorted(by_category.items()):
        positive_values = [value for value in values if value > 0]
        worker_ns = sum(positive_values)
        category_rows.append({
            "category": name,
            "events": len(values),
            "worker_time_ms": worker_ns / 1e6,
            "worker_time_share": worker_ns / total_worker_ns if total_worker_ns else 0,
            "mean_us": (worker_ns / len(positive_values) / 1e3) if positive_values else 0,
            "p50_us": percentile(positive_values, 0.5) / 1e3,
            "p90_us": percentile(positive_values, 0.9) / 1e3,
        })

    task_rows = []
    for name, values in sorted(by_task.items()):
        positive_values = [value for value in values if value > 0]
        worker_ns = sum(positive_values)
        task_rows.append({
            "task": name,
            "category": category(name),
            "events": len(values),
            "worker_time_ms": worker_ns / 1e6,
            "mean_us": (worker_ns / len(positive_values) / 1e3) if positive_values else 0,
            "p90_us": percentile(positive_values, 0.9) / 1e3,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, fieldnames, values in (
        (args.output_dir / "profile_by_category.csv", category_rows[0].keys(), category_rows),
        (args.output_dir / "profile_by_task.csv", task_rows[0].keys(), task_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(values)

    summary = {
        "status": status,
        "source": str(args.profile_csv),
        "paired_events": len(rows),
        "positive_duration_events": len(positive),
        "missing_categories": missing,
        "unknown_tasks": sorted(set(unknown)),
        "measurement_note": (
            "worker_time sums task durations across parallel CUDA blocks; it is an "
            "activity measure and must not be compared directly with decode wall time. "
            "Scheduler events are excluded because worker and scheduler kernels require "
            "separate profiler buffers."
        ),
        "categories": category_rows,
        "tasks": task_rows,
    }
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Paired events: {len(rows)}")
    for row in category_rows:
        print(
            f"{row['category']:12s} events={row['events']:7d} "
            f"worker_time={row['worker_time_ms']:10.3f} ms "
            f"share={100 * row['worker_time_share']:5.1f}%"
        )
    print(f"Missing categories: {missing}")
    print(f"Unknown tasks: {sorted(set(unknown))}")
    print(f"MPK profiler health: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
