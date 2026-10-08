"""Measure the launch cost and upper bound of a FlashInfer hybrid executor."""

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


def elapsed_ms(callable_, warmup, repeat):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def reference(q, k, v):
    dense_k = k.reshape(BATCH, KV_LENGTH, KV_HEADS, HEAD_DIM)
    dense_v = v.reshape(BATCH, KV_LENGTH, KV_HEADS, HEAD_DIM)
    return F.scaled_dot_product_attention(
        q.unsqueeze(2),
        dense_k.permute(0, 2, 1, 3),
        dense_v.permute(0, 2, 1, 3),
        enable_gqa=True,
    ).squeeze(2).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--step30-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    step30 = json.loads(args.step30_summary.read_text(encoding="utf-8"))
    if step30.get("status") != "passed":
        raise ValueError("Step 30 correctness prerequisite did not pass")
    step30_row = next(
        row for row in step30["rows"] if row["kv_length"] == KV_LENGTH)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    pages_per_request = math.ceil(KV_LENGTH / PAGE_SIZE)
    total_pages = BATCH * pages_per_request
    q = torch.randn(BATCH, Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
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

    eager_workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    eager_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        eager_workspace, kv_layout="NHD", use_tensor_cores=True)
    plan(eager_wrapper, indptr, indices, last_page_len)

    eager_output = eager_wrapper.run(q, (k, v))
    expected = reference(q, k, v)
    error = (eager_output.float() - expected.float()).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    correct = max_error <= 0.05 and mean_error <= 0.005

    def eager_layer():
        eager_wrapper.run(q, (k, v))

    def eager_step():
        for _ in range(LAYERS):
            eager_wrapper.run(q, (k, v))

    single_layer_ms = elapsed_ms(eager_layer, args.warmup, args.repeat)
    eager_step_ms = elapsed_ms(eager_step, args.warmup, args.repeat)

    graph_status = "unsupported"
    graph_reason = None
    graph_step_ms = None
    graph_output_error = None
    try:
        graph_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        graph_indptr = indptr.clone()
        graph_indices = indices.clone()
        graph_last_page_len = last_page_len.clone()
        graph_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            graph_workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=graph_indptr,
            paged_kv_indices_buffer=graph_indices,
            paged_kv_last_page_len_buffer=graph_last_page_len,
            use_tensor_cores=True,
        )
        plan(graph_wrapper, indptr, indices, last_page_len)
        for _ in range(3):
            graph_wrapper.run(q, (k, v))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        captured_outputs = []
        with torch.cuda.graph(graph):
            for _ in range(LAYERS):
                captured_outputs.append(graph_wrapper.run(q, (k, v)))
        graph.replay()
        torch.cuda.synchronize()
        graph_output_error = (
            captured_outputs[-1].float() - expected.float()).abs().max().item()
        graph_step_ms = elapsed_ms(graph.replay, args.warmup, args.repeat)
        graph_status = "passed" if graph_output_error <= 0.05 else "failed"
    except Exception as exc:  # FlashInfer CUDA Graph APIs vary by release.
        graph_reason = f"{type(exc).__name__}: {exc}"

    mpk_attention_ms = step30_row[
        "mpk_attention_work_lower_bound_ms_per_decode_step"]
    projected_eager_saving_ms = mpk_attention_ms - eager_step_ms
    projected_graph_saving_ms = (
        None if graph_step_ms is None else mpk_attention_ms - graph_step_ms)
    launch_amplification = eager_step_ms / (single_layer_ms * LAYERS)
    status = "passed" if correct and graph_status != "failed" else "failed"
    summary = {
        "step": 31,
        "phase": "hybrid_flashinfer_attention_feasibility",
        "status": status,
        "model_shape": {
            "batch_size": BATCH,
            "layers": LAYERS,
            "kv_length": KV_LENGTH,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "correctness": {
            "status": "passed" if correct else "failed",
            "max_absolute_error_vs_torch": max_error,
            "mean_absolute_error_vs_torch": mean_error,
        },
        "timing_ms": {
            "single_flashinfer_layer": single_layer_ms,
            "eager_36_layer_attention_step": eager_step_ms,
            "cuda_graph_36_layer_attention_step": graph_step_ms,
            "mpk_attention_work_lower_bound_step": mpk_attention_ms,
        },
        "analysis": {
            "eager_launch_amplification_vs_36_single_layers": launch_amplification,
            "projected_eager_saving_ms_per_step": projected_eager_saving_ms,
            "projected_cuda_graph_saving_ms_per_step": projected_graph_saving_ms,
            "cuda_graph_status": graph_status,
            "cuda_graph_output_max_error": graph_output_error,
            "cuda_graph_reason": graph_reason,
            "interpretation": (
                "This measures the external FlashInfer attention sequence and "
                "is an upper-bound feasibility test. It does not include MPK "
                "pause/resume, intermediate tensor handoff, or synchronization."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Correctness vs Torch: {'PASS' if correct else 'FAIL'}")
    print(f"Single FlashInfer layer: {single_layer_ms:.4f} ms")
    print(f"36-layer eager sequence: {eager_step_ms:.4f} ms")
    print(f"Eager launch amplification: {launch_amplification:.3f}x")
    if graph_step_ms is None:
        print(f"36-layer CUDA Graph: {graph_status}; {graph_reason}")
    else:
        print(f"36-layer CUDA Graph: {graph_step_ms:.4f} ms; {graph_status}")
    print(f"MPK attention work lower bound: {mpk_attention_ms:.4f} ms/step")
    print(f"Projected eager saving: {projected_eager_saving_ms:.4f} ms/step")
    if projected_graph_saving_ms is not None:
        print(f"Projected graph saving: {projected_graph_saving_ms:.4f} ms/step")
    print(f"Step 31 Hybrid feasibility: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
