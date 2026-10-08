"""Measure the lower-bound stream handoff cost for an MPK/FlashInfer hybrid."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


BATCH = 32
LAYERS = 36
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
KV_LENGTH = 1024
DTYPE = torch.bfloat16


def plan(wrapper, indptr, indices, last_page_len):
    positional = (
        indptr, indices, last_page_len,
        Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE,
    )
    common = {"pos_encoding_mode": "NONE", "q_data_type": DTYPE}
    try:
        wrapper.plan(*positional, **common, kv_data_type=DTYPE)
    except TypeError:
        wrapper.plan(*positional, **common, data_type=DTYPE)


def measure(callable_, stream, warmup, repeat):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeat):
        callable_()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--step31-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-us", type=float, default=54.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    step31 = json.loads(args.step31_summary.read_text(encoding="utf-8"))
    if step31.get("status") != "passed":
        raise ValueError("Step 31 prerequisite did not pass")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    pages_per_request = math.ceil(KV_LENGTH / PAGE_SIZE)
    total_pages = BATCH * pages_per_request
    q_source = torch.randn(
        BATCH, Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    q_handoff = torch.empty_like(q_source)
    result_handoff = torch.empty_like(q_source)
    k = torch.randn(
        total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)
    indptr = torch.arange(
        0, total_pages + 1, pages_per_request,
        dtype=torch.int32, device="cuda")
    indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full(
        (BATCH,), PAGE_SIZE, dtype=torch.int32, device="cuda")
    workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True)
    plan(wrapper, indptr, indices, last_page_len)

    mpk_stream = torch.cuda.current_stream()
    flashinfer_stream = torch.cuda.Stream()
    q_ready = torch.cuda.Event()
    attention_done = torch.cuda.Event()
    latest_output = [None]

    def same_stream_layer():
        with torch.cuda.stream(mpk_stream):
            q_handoff.copy_(q_source)
            output = wrapper.run(q_handoff, (k, v))
            result_handoff.copy_(output)
            latest_output[0] = output

    def cross_stream_layer():
        with torch.cuda.stream(mpk_stream):
            q_handoff.copy_(q_source)
            q_ready.record(mpk_stream)
        with torch.cuda.stream(flashinfer_stream):
            flashinfer_stream.wait_event(q_ready)
            output = wrapper.run(q_handoff, (k, v))
            attention_done.record(flashinfer_stream)
            latest_output[0] = output
        with torch.cuda.stream(mpk_stream):
            mpk_stream.wait_event(attention_done)
            result_handoff.copy_(output)

    def same_stream_step():
        for _ in range(LAYERS):
            same_stream_layer()

    def cross_stream_step():
        for _ in range(LAYERS):
            cross_stream_layer()

    same_layer_ms = measure(
        same_stream_layer, mpk_stream, args.warmup, args.repeat)
    cross_layer_ms = measure(
        cross_stream_layer, mpk_stream, args.warmup, args.repeat)
    same_step_ms = measure(
        same_stream_step, mpk_stream, args.warmup, args.repeat)
    cross_step_ms = measure(
        cross_stream_step, mpk_stream, args.warmup, args.repeat)

    cross_stream_layer()
    torch.cuda.synchronize()
    dense_k = k.reshape(BATCH, KV_LENGTH, KV_HEADS, HEAD_DIM)
    dense_v = v.reshape(BATCH, KV_LENGTH, KV_HEADS, HEAD_DIM)
    reference = F.scaled_dot_product_attention(
        q_source.unsqueeze(2),
        dense_k.permute(0, 2, 1, 3),
        dense_v.permute(0, 2, 1, 3),
        enable_gqa=True,
    ).squeeze(2).contiguous()
    error = (result_handoff.float() - reference.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    correct = max_error <= 0.05 and mean_error <= 0.005

    single_handoff_us = (cross_layer_ms - same_layer_ms) * 1000.0
    step_handoff_us_per_layer = (
        (cross_step_ms - same_step_ms) * 1000.0 / LAYERS)
    within_budget = step_handoff_us_per_layer <= args.budget_us
    status = "passed" if correct and within_budget else "failed"
    summary = {
        "step": 32,
        "phase": "hybrid_stream_handoff_lower_bound",
        "status": status,
        "model_shape": {
            "batch_size": BATCH,
            "layers": LAYERS,
            "kv_length": KV_LENGTH,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "correctness": {
            "status": "passed" if correct else "failed",
            "max_absolute_error_vs_torch": max_error,
            "mean_absolute_error_vs_torch": mean_error,
        },
        "timing_ms": {
            "same_stream_single_layer": same_layer_ms,
            "cross_stream_single_layer": cross_layer_ms,
            "same_stream_36_layers": same_step_ms,
            "cross_stream_36_layers": cross_step_ms,
        },
        "handoff": {
            "single_layer_incremental_us": single_handoff_us,
            "36_layer_incremental_us_per_layer": step_handoff_us_per_layer,
            "budget_us_per_layer": args.budget_us,
            "within_budget": within_budget,
        },
        "limitations": (
            "This is a CUDA stream/event handoff lower bound. It includes "
            "publishing Q, waiting for FlashInfer, and consuming its output, "
            "but does not yet suspend and resume the MPK persistent scheduler."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Correctness vs Torch: {'PASS' if correct else 'FAIL'}")
    print(f"Same-stream single layer: {same_layer_ms:.4f} ms")
    print(f"Cross-stream single layer: {cross_layer_ms:.4f} ms")
    print(f"Incremental single-layer handoff: {single_handoff_us:.2f} us")
    print(f"Same-stream 36 layers: {same_step_ms:.4f} ms")
    print(f"Cross-stream 36 layers: {cross_step_ms:.4f} ms")
    print(
        "Incremental 36-layer handoff: "
        f"{step_handoff_us_per_layer:.2f} us/layer")
    print(
        f"Handoff budget: {args.budget_us:.2f} us/layer; "
        f"{'PASS' if within_budget else 'FAIL'}")
    print(f"Step 32 Hybrid handoff lower bound: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
