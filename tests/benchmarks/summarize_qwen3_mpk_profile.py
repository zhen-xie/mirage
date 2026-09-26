#!/usr/bin/env python3
"""Summarize an MPK task profile without double-counting it as wall time.

The raw profiler records one duration for every worker task execution.  Those
executions can overlap, so their sum is GPU worker time, not end-to-end
latency.  This tool intentionally reports both execution-duration statistics
and worker-time share; phase wall time continues to come from CUDA events in
the benchmark output.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def task_category(task_name: str) -> str:
    name = task_name.upper()
    if name == "TASK_BEGIN_TASK_GRAPH":
        return "graph_boundary"
    if "PAGED_ATTENTION" in name or "ATTENTION" in name or "TASK_ATTN" in name:
        return "attention"
    if "ARGMAX" in name or "SAMPLING" in name:
        return "sampling"
    if "RMS_NORM" in name:
        return "norm"
    if "SILU" in name:
        return "mlp_activation"
    if "LINEAR" in name or "GEMM" in name:
        return "linear"
    if "EMBEDDING" in name:
        return "embedding"
    if "SCHD" in name or "GET_EVENT" in name or "GET_NEXT_TASK" in name:
        return "scheduler"
    if "ALLREDUCE" in name or "REDUCE" in name or "NVSHMEM" in name:
        return "communication"
    if "TENSOR_INIT" in name or "IDENTITY" in name or "ELEMENTWISE" in name:
        return "elementwise"
    return "other"


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_profile(path: Path):
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            duration_ns = int(row["duration_ns"])
            records.append(
                {
                    **row,
                    "duration_ns": duration_ns,
                    "category": task_category(row["task_type_name"]),
                }
            )
    if not records:
        raise ValueError(f"Profile contains no paired events: {path}")
    return records


def aggregate(records, key_name, decode_steps):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key_name]].append(record["duration_ns"])
    total_ns = sum(record["duration_ns"] for record in records)
    rows = []
    for key, durations in grouped.items():
        worker_ns = sum(durations)
        rows.append(
            {
                key_name: key,
                "task_executions": len(durations),
                "total_worker_time_ms": worker_ns / 1e6,
                "worker_time_share": worker_ns / total_ns if total_ns else 0.0,
                "worker_time_ms_per_decode_step": worker_ns / 1e6 / decode_steps,
                "mean_execution_us": sum(durations) / len(durations) / 1e3,
                "p50_execution_us": percentile(durations, 0.50) / 1e3,
                "p90_execution_us": percentile(durations, 0.90) / 1e3,
                "max_execution_us": max(durations) / 1e3,
            }
        )
    rows.sort(key=lambda row: row["total_worker_time_ms"], reverse=True)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_csv", type=Path)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_steps < 1:
        parser.error("--decode-steps must be positive")

    records = load_profile(args.profile_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_rows = aggregate(records, "category", args.decode_steps)
    task_rows = aggregate(records, "task_type_name", args.decode_steps)
    category_path = args.output_dir / "mpk_profile_by_category.csv"
    task_path = args.output_dir / "mpk_profile_by_task.csv"
    write_csv(category_path, category_rows)
    write_csv(task_path, task_rows)

    summary = {
        "source": str(args.profile_csv),
        "decode_steps": args.decode_steps,
        "raw_event_count": len(records),
        "raw_worker_time_ms": sum(row["duration_ns"] for row in records) / 1e6,
        "category_csv": str(category_path),
        "task_csv": str(task_path),
        "interpretation": (
            "Worker task durations overlap. total_worker_time_ms and its share "
            "measure aggregate worker occupancy, not phase wall-clock latency."
        ),
    }
    summary_path = args.output_dir / "mpk_profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("MPK task profile by category:")
    for row in category_rows:
        print(
            f"  {row['category']:16} "
            f"worker_time={row['total_worker_time_ms']:.3f} ms "
            f"share={row['worker_time_share']:.1%} "
            f"p50={row['p50_execution_us']:.3f} us "
            f"p90={row['p90_execution_us']:.3f} us"
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
