#!/usr/bin/env python3
"""Summarize CUDA kernels from an SGLang/PyTorch Chrome trace."""

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def kernel_category(kernel_name: str) -> str:
    name = kernel_name.lower()
    if any(token in name for token in ("flashinfer", "attention", "paged_kv", "batch_decode")):
        return "attention"
    if any(token in name for token in ("gemm", "cublas", "cutlass", "matmul", "mma")):
        return "gemm"
    if any(token in name for token in ("rmsnorm", "rms_norm", "layer_norm")):
        return "norm"
    if any(token in name for token in ("rope", "rotary")):
        return "rope"
    if any(token in name for token in ("sampling", "argmax", "top_k", "topk")):
        return "sampling"
    if any(token in name for token in ("memcpy", "memset")):
        return "memory_runtime"
    if any(token in name for token in ("elementwise", "vectorized", "reduce", "softmax")):
        return "elementwise"
    return "other"


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_trace(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def cuda_kernel_events(events):
    rows = []
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = str(event.get("cat", "")).lower()
        args = event.get("args") or {}
        # PyTorch traces use cat=kernel.  The fallback handles traces where
        # device kernels are identified by their device/stream metadata.
        is_kernel = "kernel" in category or (
            "stream" in args and any(key in args for key in ("device", "correlation"))
        )
        if not is_kernel:
            continue
        name = str(event.get("name", "unknown"))
        rows.append(
            {
                "kernel_name": name,
                "category": kernel_category(name),
                "duration_us": float(event["dur"]),
            }
        )
    if not rows:
        raise ValueError("No CUDA kernel duration events were found in the trace")
    return rows


def aggregate(records, key_name):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key_name]].append(record["duration_us"])
    total_us = sum(record["duration_us"] for record in records)
    rows = []
    for key, durations in grouped.items():
        duration_us = sum(durations)
        rows.append(
            {
                key_name: key,
                "kernel_launches": len(durations),
                "total_kernel_time_us": duration_us,
                "kernel_time_share": duration_us / total_us,
                "mean_kernel_us": duration_us / len(durations),
                "p50_kernel_us": percentile(durations, 0.50),
                "p90_kernel_us": percentile(durations, 0.90),
                "max_kernel_us": max(durations),
            }
        )
    rows.sort(key=lambda row: row["total_kernel_time_us"], reverse=True)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = cuda_kernel_events(read_trace(args.trace))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_rows = aggregate(records, "category")
    kernel_rows = aggregate(records, "kernel_name")
    category_path = args.output_dir / "sglang_profile_by_category.csv"
    kernel_path = args.output_dir / "sglang_profile_by_kernel.csv"
    write_csv(category_path, category_rows)
    write_csv(kernel_path, kernel_rows)
    summary_path = args.output_dir / "sglang_profile_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": str(args.trace),
                "kernel_event_count": len(records),
                "total_kernel_time_us": sum(r["duration_us"] for r in records),
                "category_csv": str(category_path),
                "kernel_csv": str(kernel_path),
                "interpretation": (
                    "Kernel durations may overlap. Their sum is CUDA kernel work, "
                    "not necessarily decode-step wall-clock latency."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("SGLang CUDA profile by category:")
    for row in category_rows:
        print(
            f"  {row['category']:16} "
            f"kernel_time={row['total_kernel_time_us']:.3f} us "
            f"share={row['kernel_time_share']:.1%} "
            f"launches={row['kernel_launches']}"
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
