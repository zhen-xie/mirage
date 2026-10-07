"""Summarize Nsight Systems CUDA kernel and API CSV reports."""

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path


PATTERNS = {
    "attention": (
        "attention", "flashinfer", "paged", "fmha", "flash_fwd", "split_k",
        "kv_cache", "kvcache", "qknorm", "rope", "rotary",
    ),
    "linear": (
        "gemm", "gemv", "matmul", "cutlass", "wgmma", "mma", "cublas",
        "linear", "nvjet_", "grouped_gemm",
    ),
    "norm": ("rmsnorm", "rms_norm", "layernorm", "layer_norm"),
    "activation": ("silu", "gelu", "swiglu", "activation", "mul_and_silu"),
    "sampling": ("sampling", "topk", "top_k", "argmax", "multinomial"),
}


def category(name):
    lower = name.lower()
    if any(value in lower for value in (
        "persistent_kernel", "worker_kernel", "scheduler_kernel",
        "resume_after_prefill_kernel", "prepare_kernel",
    )):
        return "persistent_kernel"
    for group, patterns in PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            return group
    return "other"


def find_column(fieldnames, *needles):
    for field in fieldnames or []:
        normalized = field.lower().replace(" ", "").replace("_", "")
        if all(needle in normalized for needle in needles):
            return field
    return None


def number(value):
    return float(str(value).replace(",", "").strip())


def load_report(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (
            index for index, line in enumerate(lines)
            if "Total Time" in line and "Name" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError(
            f"Nsight CSV header not found in {path}; first lines={lines[:5]}"
        )
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows = list(reader)
    fields = reader.fieldnames
    if not rows:
        raise ValueError(f"Empty Nsight report: {path}")
    name_col = find_column(fields, "name")
    time_col = find_column(fields, "totaltime")
    instances_col = find_column(fields, "instances")
    if not name_col or not time_col:
        raise ValueError(f"Unsupported Nsight columns in {path}: {fields}")
    return [{
        "name": row[name_col],
        "total_time_ns": number(row[time_col]),
        "instances": int(number(row[instances_col])) if instances_col else None,
    } for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mpk", "sglang"), required=True)
    parser.add_argument("--kernel-csv", type=Path, required=True)
    parser.add_argument("--api-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernels = load_report(args.kernel_csv)
    apis = load_report(args.api_csv)

    total_kernel_ns = sum(row["total_time_ns"] for row in kernels)
    categories = defaultdict(lambda: {"total_time_ns": 0.0, "instances": 0})
    for row in kernels:
        group = category(row["name"])
        categories[group]["total_time_ns"] += row["total_time_ns"]
        categories[group]["instances"] += row["instances"] or 0
    category_rows = [{
        "category": group,
        "time_ms": item["total_time_ns"] / 1e6,
        "time_share": item["total_time_ns"] / total_kernel_ns,
        "instances": item["instances"],
    } for group, item in categories.items()]
    category_rows.sort(key=lambda row: row["time_ms"], reverse=True)

    total_api_ns = sum(row["total_time_ns"] for row in apis)
    kernel_sum_ms = total_kernel_ns / 1e6
    effective_kernel_ms = (
        max(row["total_time_ns"] for row in kernels) / 1e6
        if args.backend == "mpk" else kernel_sum_ms
    )
    synchronize_ns = sum(
        row["total_time_ns"] for row in apis
        if row["name"] == "cudaDeviceSynchronize"
    )
    result = {
        "status": "passed",
        "backend": args.backend,
        "kernel_time_ms": effective_kernel_ms,
        "kernel_time_sum_ms": kernel_sum_ms,
        "kernel_time_mode": (
            "maximum concurrent kernel duration"
            if args.backend == "mpk" else "sum of CUDA kernel durations"
        ),
        "kernel_instances": sum(row["instances"] or 0 for row in kernels),
        "cuda_api_time_ms": total_api_ns / 1e6,
        "cuda_device_synchronize_time_ms": synchronize_ns / 1e6,
        "categories": category_rows,
        "top_kernels": sorted(kernels, key=lambda row: row["total_time_ns"], reverse=True)[:25],
        "top_cuda_apis": sorted(apis, key=lambda row: row["total_time_ns"], reverse=True)[:20],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{args.backend}: effective_kernel_time={result['kernel_time_ms']:.3f} ms, "
        f"kernel_sum={result['kernel_time_sum_ms']:.3f} ms, "
        f"kernel_instances={result['kernel_instances']}, "
        f"CUDA_API_time={result['cuda_api_time_ms']:.3f} ms, "
        f"synchronize_time={result['cuda_device_synchronize_time_ms']:.3f} ms"
    )
    for row in category_rows:
        print(
            f"  {row['category']:18s} {row['time_ms']:10.3f} ms "
            f"{100 * row['time_share']:5.1f}% instances={row['instances']}"
        )


if __name__ == "__main__":
    main()
