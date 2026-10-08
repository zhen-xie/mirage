"""Real Qwen3 decoder-attention Hybrid boundary validation.

Runs one decoder layer's real RMSNorm/QKV and O-projection weights through
finite MPK segments, with Q/K norm, RoPE, paged KV update, and FlashInfer in
between.  The reference uses Torch SDPA with the same cache contents.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


BATCH = 8
KV_LENGTH = 1024
PAGE_SIZE = 128
DTYPE = torch.bfloat16


def rms_norm(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + eps)
            * weight.float()).to(DTYPE)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def make_segment(inputs, operations, cache_dir):
    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        mode="online_notoken",
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
        max_seq_length=BATCH,
        max_num_pages=BATCH,
        page_size=1,
        use_cutlass_kernel=True,
    )
    kernel = PersistentKernel(**params)
    tensors = {
        name: kernel.attach_input(tensor, name=name)
        for name, tensor in inputs.items()
    }
    operations(kernel, tensors)
    cache_dir.mkdir(parents=True, exist_ok=True)
    launcher = cache_dir / (
        f"mpk_launcher_rank0.cpython-{sys.version_info.major}"
        f"{sys.version_info.minor}-x86_64-linux-gnu.so")
    if launcher.is_file():
        print(f"Loading cached MPK segment: {cache_dir}")
        kernel.load_mpk_kernel(output_dir=str(cache_dir))
    else:
        kernel.compile(output_dir=str(cache_dir))
    return kernel


def plan_flashinfer(wrapper, indptr, indices, last_page_len,
                    q_heads, kv_heads, head_dim):
    positional = (indptr, indices, last_page_len,
                  q_heads, kv_heads, head_dim, PAGE_SIZE)
    common = {"pos_encoding_mode": "NONE", "q_data_type": DTYPE}
    try:
        wrapper.plan(*positional, **common, kv_data_type=DTYPE)
    except TypeError:
        wrapper.plan(*positional, **common, data_type=DTYPE)


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map="cuda")
    model.eval()
    config = model.config
    layer = model.model.layers[0]
    attention = layer.self_attn
    hidden_size = config.hidden_size
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden_size // q_heads)
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size

    hidden = torch.randn(
        BATCH, hidden_size, dtype=DTYPE, device="cuda") * 0.1
    normed = torch.empty_like(hidden)
    qkv = torch.empty(BATCH, qkv_size, dtype=DTYPE, device="cuda")
    attention_output = torch.empty(
        BATCH, q_heads, head_dim, dtype=DTYPE, device="cuda")
    projected = torch.empty_like(hidden)
    qkv_weight = torch.cat(
        (attention.q_proj.weight, attention.k_proj.weight,
         attention.v_proj.weight), dim=0).contiguous()

    def build_prefix(kernel, t):
        kernel.rmsnorm_layer(
            t["hidden"], t["input_norm"], t["normed"],
            grid_dim=(BATCH, 1, 1), block_dim=(128, 1, 1))
        kernel.linear_layer(
            t["normed"], t["qkv_weight"], t["qkv"],
            grid_dim=(96, 1, 1), block_dim=(128, 1, 1))

    prefix = make_segment({
        "step34_hidden": hidden,
        "step34_input_norm": layer.input_layernorm.weight,
        "step34_normed": normed,
        "step34_qkv_weight": qkv_weight,
        "step34_qkv": qkv,
    }, lambda kernel, t: build_prefix(kernel, {
        "hidden": t["step34_hidden"],
        "input_norm": t["step34_input_norm"],
        "normed": t["step34_normed"],
        "qkv_weight": t["step34_qkv_weight"],
        "qkv": t["step34_qkv"],
    }), args.output_dir / "cache_prefix")

    def build_suffix(kernel, t):
        kernel.linear_layer(
            t["attention"], t["o_weight"], t["projected"],
            grid_dim=(64, 1, 1), block_dim=(128, 1, 1))

    suffix = make_segment({
        "step34_attention": attention_output.reshape(BATCH, hidden_size),
        "step34_o_weight": attention.o_proj.weight,
        "step34_projected": projected,
    }, lambda kernel, t: build_suffix(kernel, {
        "attention": t["step34_attention"],
        "o_weight": t["step34_o_weight"],
        "projected": t["step34_projected"],
    }), args.output_dir / "cache_suffix")

    pages_per_request = math.ceil(KV_LENGTH / PAGE_SIZE)
    total_pages = BATCH * pages_per_request
    initial_k = torch.randn(
        total_pages, PAGE_SIZE, kv_heads, head_dim,
        dtype=DTYPE, device="cuda") * 0.1
    initial_v = torch.randn_like(initial_k) * 0.1
    k_cache = initial_k.clone()
    v_cache = initial_v.clone()
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
    plan_flashinfer(wrapper, indptr, indices, last_page_len,
                    q_heads, kv_heads, head_dim)

    inv_freq = 1.0 / (
        float(config.rope_theta) **
        (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim))
    freqs = float(KV_LENGTH - 1) * inv_freq
    rope = torch.cat((freqs, freqs), dim=-1)
    cos = rope.cos().to(DTYPE).view(1, 1, head_dim)
    sin = rope.sin().to(DTYPE).view(1, 1, head_dim)
    hybrid_q = torch.empty(
        BATCH, q_heads, head_dim, dtype=DTYPE, device="cuda")

    def preprocess(qkv_value, cache_k, cache_v):
        q = qkv_value[:, :q_size].view(BATCH, q_heads, head_dim)
        k = qkv_value[:, q_size:q_size + kv_size].view(
            BATCH, kv_heads, head_dim)
        v = qkv_value[:, q_size + kv_size:].view(
            BATCH, kv_heads, head_dim)
        q = apply_rope(rms_norm(q, attention.q_norm.weight), cos, sin)
        k = apply_rope(rms_norm(k, attention.k_norm.weight), cos, sin)
        cache_k.view(BATCH, pages_per_request, PAGE_SIZE,
                     kv_heads, head_dim)[:, -1, -1].copy_(k)
        cache_v.view(BATCH, pages_per_request, PAGE_SIZE,
                     kv_heads, head_dim)[:, -1, -1].copy_(v)
        return q, k, v

    def hybrid_once():
        k_cache.copy_(initial_k)
        v_cache.copy_(initial_v)
        prefix()
        torch.cuda.synchronize()
        q, _, _ = preprocess(qkv, k_cache, v_cache)
        hybrid_q.copy_(q)
        attention_output.copy_(wrapper.run(q, (k_cache, v_cache)))
        suffix()
        torch.cuda.synchronize()

    with torch.inference_mode():
        norm_ref = rms_norm(hidden, layer.input_layernorm.weight)
        qkv_ref = F.linear(norm_ref, qkv_weight)
        ref_k = initial_k.clone()
        ref_v = initial_v.clone()
        q_ref, _, _ = preprocess(qkv_ref, ref_k, ref_v)
        k_seq = ref_k.view(BATCH, KV_LENGTH, kv_heads, head_dim)
        v_seq = ref_v.view(BATCH, KV_LENGTH, kv_heads, head_dim)
        reference_attention = F.scaled_dot_product_attention(
            q_ref.unsqueeze(2),
            k_seq.permute(0, 2, 1, 3),
            v_seq.permute(0, 2, 1, 3),
            is_causal=False, enable_gqa=True).squeeze(2)
        # Isolate FlashInfer itself from the MPK-produced QKV.  This uses the
        # Torch QKV projection and an independent cache with the exact same
        # page table as the Hybrid path.
        fi_ref_k = initial_k.clone()
        fi_ref_v = initial_v.clone()
        q_fi_ref, _, _ = preprocess(qkv_ref, fi_ref_k, fi_ref_v)
        flashinfer_reference_attention = wrapper.run(
            q_fi_ref, (fi_ref_k, fi_ref_v)).clone()
        reference = F.linear(
            reference_attention.reshape(BATCH, hidden_size),
            attention.o_proj.weight)

        hybrid_once()
        qkv_error = (qkv.float() - qkv_ref.float()).abs()
        attention_error = (
            attention_output.float() - reference_attention.float()).abs()
        flashinfer_vs_sdpa_error = (
            flashinfer_reference_attention.float()
            - reference_attention.float()).abs()
        hybrid_vs_flashinfer_error = (
            attention_output.float()
            - flashinfer_reference_attention.float()).abs()
        output_error = (projected.float() - reference.float()).abs()
        finite = {
            "hybrid_q": int(torch.isfinite(hybrid_q).sum().item()),
            "sdpa": int(torch.isfinite(reference_attention).sum().item()),
            "flashinfer_reference": int(torch.isfinite(
                flashinfer_reference_attention).sum().item()),
            "hybrid_flashinfer": int(torch.isfinite(
                attention_output).sum().item()),
        }
        total_attention_elements = reference_attention.numel()
        first10_matches = int(torch.isclose(
            projected[0, :10].float(), reference[0, :10].float(),
            atol=0.02, rtol=0.02).sum().item())
        correct = (
            first10_matches == 10
            and attention_error.mean().item() <= 0.005
            and output_error.max().item() <= 0.03
            and output_error.mean().item() <= 0.005
        )
        hybrid_ms = measure(hybrid_once, args.warmup, args.repeat)

    summary = {
        "step": 34,
        "phase": "real_qwen3_layer_hybrid",
        "status": "passed" if correct else "failed",
        "model": args.model,
        "layer": 0,
        "shape": {"batch_size": BATCH, "kv_length": KV_LENGTH,
                  "hidden_size": hidden_size, "q_heads": q_heads,
                  "kv_heads": kv_heads, "head_dim": head_dim},
        "correctness": {
            "first10_output_elements": first10_matches,
            "qkv_max_error": qkv_error.max().item(),
            "qkv_mean_error": qkv_error.mean().item(),
            "attention_max_error": attention_error.max().item(),
            "attention_mean_error": attention_error.mean().item(),
            "flashinfer_vs_sdpa_max_error":
                flashinfer_vs_sdpa_error.max().item(),
            "flashinfer_vs_sdpa_mean_error":
                flashinfer_vs_sdpa_error.mean().item(),
            "hybrid_vs_flashinfer_max_error":
                hybrid_vs_flashinfer_error.max().item(),
            "hybrid_vs_flashinfer_mean_error":
                hybrid_vs_flashinfer_error.mean().item(),
            "finite_attention_elements": finite,
            "total_attention_elements": total_attention_elements,
            "output_max_error": output_error.max().item(),
            "output_mean_error": output_error.mean().item(),
        },
        "hybrid_layer_ms": hybrid_ms,
        "implementation": [
            "real Qwen3 layer-0 weights",
            "finite MPK RMSNorm and QKV segment",
            "Q/K RMSNorm and real-position NeoX RoPE",
            "paged KV update and FlashInfer decode attention",
            "finite MPK output-projection segment",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"First-10 output elements: {first10_matches}/10")
    print(f"QKV max/mean error: {qkv_error.max().item():.6f}/"
          f"{qkv_error.mean().item():.6f}")
    print(f"Attention max/mean error: {attention_error.max().item():.6f}/"
          f"{attention_error.mean().item():.6f}")
    print("Finite attention elements: "
          f"SDPA={finite['sdpa']}/{total_attention_elements}, "
          f"FlashInfer reference={finite['flashinfer_reference']}/"
          f"{total_attention_elements}, Hybrid={finite['hybrid_flashinfer']}/"
          f"{total_attention_elements}")
    print("Finite Hybrid Q elements: "
          f"{finite['hybrid_q']}/{hybrid_q.numel()}")
    print("FlashInfer reference vs SDPA max/mean: "
          f"{flashinfer_vs_sdpa_error.max().item():.6f}/"
          f"{flashinfer_vs_sdpa_error.mean().item():.6f}")
    print("Hybrid vs FlashInfer reference max/mean: "
          f"{hybrid_vs_flashinfer_error.max().item():.6f}/"
          f"{hybrid_vs_flashinfer_error.mean().item():.6f}")
    print(f"Output max/mean error: {output_error.max().item():.6f}/"
          f"{output_error.mean().item():.6f}")
    print(f"Real Qwen3 Hybrid layer: {hybrid_ms:.4f} ms")
    print(f"Step 34 real-layer Hybrid: {summary['status'].upper()}")
    prefix.finalize()
    suffix.finalize()
    del model
    raise SystemExit(0 if correct else 1)


if __name__ == "__main__":
    main()
