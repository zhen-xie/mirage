"""Summarize CUDA kernel activity from an SGLang Torch profiler trace."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


CATEGORY_PATTERNS = {
    "attention": (
        "attention", "flashinfer", "decode", "prefill", "paged", "fmha",
        "flash_fwd", "split_k", "kv_cache", "kvcache", "rope", "rotary",
        "qknorm",
    ),
    "linear": (
        "gemm", "gemv", "matmul", "cutlass", "wgmma", "mma", "cublas",
        "linear", "moe", "grouped_gemm", "nvjet_",
    ),
    "norm": ("rmsnorm", "rms_norm", "layernorm", "layer_norm"),
    "activation": ("silu", "gelu", "swiglu", "activation", "mul_and_silu"),
    "sampling": ("sampling", "topk", "top_k", "argmax", "multinomial"),
    "communication": ("nccl", "allreduce", "all_reduce", "reduce_scatter"),
}


def category(name):
    lower = name.lower()
    for group, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return group
    return "other"


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        data = json.load(source)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def is_cuda_kernel(event):
    if event.get("ph") != "X" or not isinstance(event.get("dur"), (int, float)):
        return False
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", ""))
    # Torch traces contain CUDA runtime/API calls alongside device kernels.
    # API calls can carry stream metadata, so stream presence alone is not a
    # reliable kernel test.
    if name.startswith("cuda") or name.startswith((
        "cuLaunch", "cuMemcpy", "cuMemset", "cuGraph", "cuDevice",
        "cuCtx", "cuEvent", "cuStream",
    )):
        return False
    args = event.get("args") or {}
    device = str(args.get("Device Type", args.get("device_type", ""))).lower()
    return "kernel" in cat or "gpu_kernel" in cat or device in {"1", "cuda", "gpu"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-other-share", type=float, default=0.20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_kernel = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for event in read_trace(args.trace):
        if not is_cuda_kernel(event):
            continue
        name = str(event.get("name", "<unnamed>"))
        by_kernel[name]["calls"] += 1
        by_kernel[name]["duration_us"] += float(event["dur"])
    if not by_kernel:
        raise ValueError(f"No CUDA kernel events found in {args.trace}")

    total_us = sum(item["duration_us"] for item in by_kernel.values())
    kernel_rows = []
    category_totals = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for name, item in by_kernel.items():
        group = category(name)
        category_totals[group]["calls"] += item["calls"]
        category_totals[group]["duration_us"] += item["duration_us"]
        kernel_rows.append({
            "category": group,
            "kernel": name,
            "calls": item["calls"],
            "duration_ms": item["duration_us"] / 1000.0,
            "time_share": item["duration_us"] / total_us,
        })
    kernel_rows.sort(key=lambda row: row["duration_ms"], reverse=True)
    category_rows = [{
        "category": name,
        "calls": item["calls"],
        "duration_ms": item["duration_us"] / 1000.0,
        "time_share": item["duration_us"] / total_us,
    } for name, item in category_totals.items()]
    category_rows.sort(key=lambda row: row["duration_ms"], reverse=True)

    write_csv(args.output_dir / "profile_by_category.csv", category_rows)
    write_csv(args.output_dir / "profile_by_kernel.csv", kernel_rows)
    other_share = next(
        (row["time_share"] for row in category_rows if row["category"] == "other"),
        0.0,
    )
    unclassified = [row for row in kernel_rows if row["category"] == "other"][:25]
    status = "passed" if other_share <= args.maximum_other_share else "classification_incomplete"
    summary = {
        "status": status,
        "source": str(args.trace),
        "cuda_kernel_events": sum(row["calls"] for row in kernel_rows),
        "cuda_kernel_time_ms": total_us / 1000.0,
        "categories": category_rows,
        "top_kernels": kernel_rows[:25],
        "top_unclassified_kernels": unclassified,
        "other_time_share": other_share,
        "maximum_other_share": args.maximum_other_share,
        "classification_note": (
            "Categories are inferred from CUDA kernel names. CUDA kernel durations may "
            "overlap across streams and are not end-to-end wall time."
        ),
    }
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"CUDA kernel events: {summary['cuda_kernel_events']}")
    print(f"CUDA kernel time: {summary['cuda_kernel_time_ms']:.3f} ms")
    for row in category_rows:
        print(
            f"{row['category']:14s} calls={row['calls']:7d} "
            f"kernel_time={row['duration_ms']:10.3f} ms "
            f"share={100 * row['time_share']:5.1f}%"
        )
    if unclassified:
        print("Top unclassified CUDA kernels:")
        for row in unclassified[:15]:
            print(
                f"  {row['duration_ms']:10.3f} ms  calls={row['calls']:7d}  "
                f"{row['kernel']}"
            )
    print(f"Classification status: {status}")
    raise SystemExit(0 if status == "passed" else 2)


if __name__ == "__main__":
    main()
