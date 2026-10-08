"""Single-layer Hybrid prototype with real MPK linear segments and FlashInfer."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


BATCH = 32
HIDDEN = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
QKV_SIZE = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
PAGE_SIZE = 128
KV_LENGTH = 1024
DTYPE = torch.bfloat16


def make_linear_kernel(x, weight, output, cache_dir, name, tasks):
    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
        use_cutlass_kernel=True,
    )
    kernel = PersistentKernel(**params)
    x_dt = kernel.attach_input(x, name=f"{name}_input")
    weight_dt = kernel.attach_input(weight, name=f"{name}_weight")
    output_dt = kernel.attach_input(output, name=f"{name}_output")
    kernel.linear_layer(
        input=x_dt,
        weight=weight_dt,
        output=output_dt,
        grid_dim=(tasks, 1, 1),
        block_dim=(128, 1, 1),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    kernel.compile(output_dir=str(cache_dir))
    return kernel


def rms_norm(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            * weight.float()).to(DTYPE)


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


def measure(callable_, warmup, repeat):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--step32-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    step32 = json.loads(args.step32_summary.read_text(encoding="utf-8"))
    if step32.get("status") != "passed":
        raise ValueError("Step 32 prerequisite did not pass")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    hidden = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device="cuda") * 0.1
    qkv_weight = (
        torch.randn(QKV_SIZE, HIDDEN, dtype=DTYPE, device="cuda") * 0.01)
    o_weight = (
        torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device="cuda") * 0.01)
    q_weight = torch.randn(Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k_weight = torch.randn(KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    qkv = torch.empty(BATCH, QKV_SIZE, dtype=DTYPE, device="cuda")
    attention_output = torch.empty(
        BATCH, Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    projected = torch.empty(BATCH, HIDDEN, dtype=DTYPE, device="cuda")

    prefix = make_linear_kernel(
        hidden, qkv_weight, qkv,
        args.output_dir / "cache_prefix", "qkv", 96)
    suffix = make_linear_kernel(
        attention_output.view(BATCH, HIDDEN), o_weight, projected,
        args.output_dir / "cache_suffix", "o_proj", 64)

    pages_per_request = math.ceil(KV_LENGTH / PAGE_SIZE)
    total_pages = BATCH * pages_per_request
    k_cache = torch.randn(
        total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device="cuda") * 0.1
    v_cache = torch.randn_like(k_cache) * 0.1
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

    def preprocess_and_attention(qkv_value):
        q = qkv_value[:, :Q_HEADS * HEAD_DIM].view(
            BATCH, Q_HEADS, HEAD_DIM)
        k = qkv_value[:, Q_HEADS * HEAD_DIM:
                      (Q_HEADS + KV_HEADS) * HEAD_DIM].view(
                          BATCH, KV_HEADS, HEAD_DIM)
        v = qkv_value[:, (Q_HEADS + KV_HEADS) * HEAD_DIM:].view(
            BATCH, KV_HEADS, HEAD_DIM)
        q = rms_norm(q, q_weight)
        k = rms_norm(k, k_weight)
        cache_k = k_cache.view(
            BATCH, pages_per_request, PAGE_SIZE, KV_HEADS, HEAD_DIM)
        cache_v = v_cache.view_as(cache_k)
        cache_k[:, -1, -1].copy_(k)
        cache_v[:, -1, -1].copy_(v)
        return wrapper.run(q, (k_cache, v_cache))

    def hybrid_once():
        prefix()
        # The current MPK runtime has no layer boundary event. Full device
        # synchronization is the real synchronization required by this first
        # prototype before FlashInfer can safely consume qkv.
        torch.cuda.synchronize()
        output = preprocess_and_attention(qkv)
        attention_output.copy_(output)
        torch.cuda.synchronize()
        suffix()
        torch.cuda.synchronize()

    def external_once():
        qkv_ref = F.linear(hidden, qkv_weight)
        output = preprocess_and_attention(qkv_ref)
        F.linear(output.reshape(BATCH, HIDDEN), o_weight)

    hybrid_once()
    qkv_reference = F.linear(hidden, qkv_weight)
    reference_attention = preprocess_and_attention(qkv_reference)
    reference = F.linear(reference_attention.reshape(BATCH, HIDDEN), o_weight)
    torch.cuda.synchronize()
    qkv_error = (qkv.float() - qkv_reference.float()).abs()
    attention_error = (
        attention_output.float() - reference_attention.float()).abs()
    output_error = (projected.float() - reference.float()).abs()
    correct = (
        qkv_error.mean().item() <= 0.01
        and attention_error.mean().item() <= 0.005
        and output_error.mean().item() <= 0.05
    )

    external_ms = measure(external_once, args.warmup, args.repeat)
    hybrid_ms = measure(hybrid_once, args.warmup, args.repeat)
    transition_penalty_ms = hybrid_ms - external_ms
    status = "passed" if correct else "failed"
    summary = {
        "step": 33,
        "phase": "single_layer_real_mpk_flashinfer_hybrid",
        "status": status,
        "shape": {
            "batch_size": BATCH,
            "hidden_size": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "kv_length": KV_LENGTH,
        },
        "correctness": {
            "status": "passed" if correct else "failed",
            "qkv_max_error": qkv_error.max().item(),
            "qkv_mean_error": qkv_error.mean().item(),
            "attention_max_error": attention_error.max().item(),
            "attention_mean_error": attention_error.mean().item(),
            "output_max_error": output_error.max().item(),
            "output_mean_error": output_error.mean().item(),
        },
        "timing_ms": {
            "external_torch_flashinfer_layer": external_ms,
            "real_mpk_flashinfer_mpk_layer": hybrid_ms,
            "mpk_segment_and_sync_penalty": transition_penalty_ms,
        },
        "implementation": {
            "prefix": "real MPK QKV linear persistent kernel",
            "attention": "Torch QK norm/KV store plus FlashInfer decode",
            "suffix": "real MPK output projection persistent kernel",
            "synchronization": "cudaDeviceSynchronize at both MPK boundaries",
            "missing_for_full_model": [
                "MPK boundary events without device-wide synchronization",
                "RoPE in the exported pre-attention stage",
                "36-layer decode loop and token-level correctness",
            ],
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"QKV max/mean error: {qkv_error.max().item():.6f}/"
          f"{qkv_error.mean().item():.6f}")
    print(f"Attention max/mean error: {attention_error.max().item():.6f}/"
          f"{attention_error.mean().item():.6f}")
    print(f"Output max/mean error: {output_error.max().item():.6f}/"
          f"{output_error.mean().item():.6f}")
    print(f"External Torch+FlashInfer layer: {external_ms:.4f} ms")
    print(f"Real MPK+FlashInfer+MPK layer: {hybrid_ms:.4f} ms")
    print(f"MPK segment/synchronization penalty: {transition_penalty_ms:.4f} ms")
    print(f"Step 33 single-layer Hybrid: {status.upper()}")
    prefix.finalize()
    suffix.finalize()
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
