"""Benchmark FlashInfer batch decode against Torch and MPK profile data."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


BATCH = 32
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
KV_LENGTHS = (128, 512, 1024)
DTYPE = torch.bfloat16


def torch_reference(q, k_pages, v_pages, kv_length):
    k = k_pages.reshape(BATCH, kv_length, KV_HEADS, HEAD_DIM)
    v = v_pages.reshape(BATCH, kv_length, KV_HEADS, HEAD_DIM)
    return F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        enable_gqa=True,
    ).squeeze(2).contiguous()


def plan_wrapper(wrapper, indptr, indices, last_page_len, dtype):
    common = dict(
        pos_encoding_mode="NONE",
        q_data_type=dtype,
    )
    positional = (
        indptr, indices, last_page_len, Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE)
    try:
        wrapper.plan(*positional, **common, kv_data_type=dtype)
    except TypeError:
        wrapper.plan(*positional, **common, data_type=dtype)


def timed_run(wrapper, q, kv, warmup, repeat):
    for _ in range(warmup):
        wrapper.run(q, kv)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        wrapper.run(q, kv)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def load_mpk_rows(step27, step26):
    scaling = json.loads(step27.read_text(encoding="utf-8"))
    profile = json.loads(
        (step26 / "early/summary/profile_summary.json").read_text(
            encoding="utf-8"))
    early = json.loads(
        (step26 / "summary.json").read_text(encoding="utf-8"))
    early_row = next(row for row in early["rows"] if row["window"] == "early")
    if scaling["status"] != "passed" or early_row["status"] != "completed":
        raise ValueError("Step 26/27 MPK correctness prerequisite did not pass")
    if early_row["minimum_first10_matches"] != 10:
        raise ValueError("Step 26 MPK first-10 correctness prerequisite failed")
    attention = next(
        row for row in profile["categories"]
        if row["category"] == "attention")
    by_length = {int(row["s_in"]): row for row in scaling["rows"]}
    by_length[1024] = {
        **by_length[1024],
        "attention_worker_ms": attention["worker_time_ms"],
    }
    return by_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--step27-summary", type=Path, required=True)
    parser.add_argument("--step26-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    mpk = load_mpk_rows(args.step27_summary, args.step26_dir)
    workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    rows = []
    failures = 0

    for kv_length in KV_LENGTHS:
        pages_per_request = math.ceil(kv_length / PAGE_SIZE)
        total_pages = BATCH * pages_per_request
        q = torch.randn(
            BATCH, Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        k = torch.randn(
            total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
            dtype=DTYPE, device="cuda")
        v = torch.randn_like(k)
        indptr = torch.arange(
            0, total_pages + 1, pages_per_request,
            dtype=torch.int32, device="cuda")
        indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
        last_page_len = torch.full(
            (BATCH,), kv_length - (pages_per_request - 1) * PAGE_SIZE,
            dtype=torch.int32, device="cuda")

        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=True)
        plan_wrapper(wrapper, indptr, indices, last_page_len, DTYPE)
        output = wrapper.run(q, (k, v))
        reference = torch_reference(q, k, v, kv_length)
        error = (output.float() - reference.float()).abs()
        max_error = error.max().item()
        mean_error = error.mean().item()
        correct = max_error <= 0.05 and mean_error <= 0.005
        elapsed_ms = timed_run(
            wrapper, q, (k, v), args.warmup, args.repeat)

        mpk_row = mpk[kv_length]
        mpk_mean_us = mpk_row["attention_mean_us"]
        # This is a work-conservation lower bound, not measured wall time:
        # divide summed worker activity by the 128 MPK workers and by the
        # nine profiled decode steps. Each step includes all model layers.
        mpk_lower_bound_ms = (
            mpk_row["attention_worker_ms"] / (128 * 9))
        row = {
            "kv_length": kv_length,
            "batch_size": BATCH,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "flashinfer_ms_per_layer": elapsed_ms,
            "flashinfer_tokens_per_second": BATCH * 1000.0 / elapsed_ms,
            "max_absolute_error_vs_torch": max_error,
            "mean_absolute_error_vs_torch": mean_error,
            "correctness": "passed" if correct else "failed",
            "mpk_attention_mean_task_us": mpk_mean_us,
            "mpk_attention_work_lower_bound_ms_per_decode_step": (
                mpk_lower_bound_ms),
            "note": (
                "FlashInfer is measured wall time for one layer. MPK task "
                "mean and work lower bound come from its in-kernel profiler "
                "and are not direct wall-time measurements."
            ),
        }
        rows.append(row)
        failures += not correct
        print(
            f"KV={kv_length}: {'PASS' if correct else 'FAIL'}; "
            f"FlashInfer={elapsed_ms:.4f} ms/layer; "
            f"max/mean error={max_error:.6f}/{mean_error:.6f}; "
            f"MPK task mean={mpk_mean_us:.3f} us",
            flush=True,
        )

    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 30,
        "phase": "flashinfer_attention_microbenchmark",
        "status": "passed" if not failures else "failed",
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "mpk_correctness_prerequisite": (
            "Step 26/27 passed against the Torch first-10 and full-output "
            "correctness gates."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Step 30 FlashInfer attention: {'PASS' if not failures else 'FAIL'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
