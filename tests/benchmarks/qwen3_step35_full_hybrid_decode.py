"""End-to-end Qwen3 decode through MPK/FlashInfer layer boundaries.

The Torch prefill supplies the initial KV cache and first generated token.
Each later token executes every decoder layer as:

  host RMSNorm -> finite MPK QKV -> FlashInfer paged decode ->
  finite MPK output projection/residual -> host MLP

This is a correctness-first executor prototype.  It intentionally measures
the cost of the finite MPK boundaries that a later persistent Hybrid executor
must remove.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


DTYPE = torch.bfloat16


def rms_norm(x, weight, eps=1e-6):
    value = x.float()
    return (value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + eps
    ) * weight.float()).to(DTYPE)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def grid_for_projection(size):
    if size / 96 > 400:
        if size % 256:
            raise ValueError(f"unsupported projection size: {size}")
        return size // 256
    if size % 96 == 0:
        return 96
    if size % 64 == 0:
        return 64
    raise ValueError(f"unsupported projection size: {size}")


def make_segment(inputs, operations, cache_dir, batch):
    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        mode="online_notoken",
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        max_num_batched_tokens=batch,
        max_num_batched_requests=1,
        max_seq_length=batch,
        max_num_pages=batch,
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
        kernel.load_mpk_kernel(output_dir=str(cache_dir))
    else:
        kernel.compile(output_dir=str(cache_dir))
    return kernel


def plan_flashinfer(wrapper, indptr, indices, last_page_len,
                    q_heads, kv_heads, head_dim, page_size):
    positional = (indptr, indices, last_page_len,
                  q_heads, kv_heads, head_dim, page_size)
    common = {"pos_encoding_mode": "NONE", "q_data_type": DTYPE}
    try:
        wrapper.plan(*positional, **common, kv_data_type=DTYPE)
    except TypeError:
        wrapper.plan(*positional, **common, data_type=DTYPE)


def cache_tensors(past_key_values):
    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache,
                        past_key_values.value_cache))
    return list(past_key_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPE, device_map="cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(
        "Give me a short introduction to large language model.",
        return_tensors="pt")["input_ids"][0]
    repeats = math.ceil(args.input_length / encoded.numel())
    prompt = encoded.repeat(repeats)[:args.input_length]
    input_ids = prompt.unsqueeze(0).repeat(args.batch_size, 1).cuda()

    with torch.inference_mode():
        reference_ids = model.generate(
            input_ids,
            max_new_tokens=args.output_length,
            min_new_tokens=args.output_length,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )[:, args.input_length:]
        prefill = model(input_ids, use_cache=True, return_dict=True)

    config = model.config
    batch = args.batch_size
    hidden_size = config.hidden_size
    q_heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden_size // q_heads)
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size
    page_size = 128
    max_kv_length = args.input_length + args.output_length - 1
    pages_per_request = math.ceil(max_kv_length / page_size)
    total_pages = batch * pages_per_request

    hidden_buffers = [torch.empty(
        batch, hidden_size, dtype=DTYPE, device="cuda")
        for _ in range(config.num_hidden_layers + 1)]
    normed_buffers = [torch.empty_like(hidden_buffers[0])
                      for _ in range(config.num_hidden_layers)]
    qkv_buffers = [torch.empty(
        batch, qkv_size, dtype=DTYPE, device="cuda")
        for _ in range(config.num_hidden_layers)]
    attention_buffers = [torch.empty(
        batch, q_heads, head_dim, dtype=DTYPE, device="cuda")
        for _ in range(config.num_hidden_layers)]

    prefixes = []
    suffixes = []
    prefix_cache = args.output_dir / "cache_prefix"
    suffix_cache = args.output_dir / "cache_suffix"
    for layer_idx, layer in enumerate(model.model.layers):
        qkv_weight = torch.cat((
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight), dim=0).contiguous()

        def build_prefix(kernel, t):
            kernel.linear_layer(
                t["normed"], t["qkv_weight"], t["qkv"],
                grid_dim=(grid_for_projection(qkv_size), 1, 1),
                block_dim=(128, 1, 1))

        prefixes.append(make_segment({
            "normed": normed_buffers[layer_idx],
            "qkv_weight": qkv_weight,
            "qkv": qkv_buffers[layer_idx],
        }, build_prefix, prefix_cache, batch))

        attn_residual = torch.empty_like(hidden_buffers[0])

        def build_suffix(kernel, t):
            kernel.linear_with_residual_layer(
                input=t["attention"], weight=t["o_weight"],
                residual=t["hidden"], output=t["attn_residual"],
                grid_dim=(hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1))

        suffixes.append(make_segment({
            "attention": attention_buffers[layer_idx].reshape(batch, hidden_size),
            "o_weight": layer.self_attn.o_proj.weight,
            "hidden": hidden_buffers[layer_idx],
            "attn_residual": attn_residual,
        }, build_suffix, suffix_cache, batch))
        # Keep this tensor alive with the kernel-attached output and use it as
        # the host MLP residual after the finite MPK segment returns.
        suffixes[-1].step35_attn_residual = attn_residual

    kv = cache_tensors(prefill.past_key_values)
    k_cache = torch.zeros(
        config.num_hidden_layers, total_pages, page_size,
        kv_heads, head_dim, dtype=DTYPE, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    for layer_idx, (keys, values) in enumerate(kv):
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        page_view_k = k_cache[layer_idx].view(
            batch, pages_per_request, page_size, kv_heads, head_dim)
        page_view_v = v_cache[layer_idx].view_as(page_view_k)
        page_view_k.view(batch, -1, kv_heads, head_dim)[
            :, :args.input_length].copy_(keys)
        page_view_v.view(batch, -1, kv_heads, head_dim)[
            :, :args.input_length].copy_(values)

    indptr = torch.arange(
        0, total_pages + 1, pages_per_request,
        dtype=torch.int32, device="cuda")
    indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.empty(batch, dtype=torch.int32, device="cuda")
    workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True)

    generated = [prefill.logits[:, -1].argmax(dim=-1)]
    timing = {"prefix_ms": 0.0, "flashinfer_ms": 0.0,
              "suffix_ms": 0.0, "host_mlp_ms": 0.0,
              "decode_ms": 0.0}
    start_total = torch.cuda.Event(enable_timing=True)
    end_total = torch.cuda.Event(enable_timing=True)
    stage_start = torch.cuda.Event(enable_timing=True)
    stage_end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        start_total.record()
        for decode_idx in range(1, args.output_length):
            token = generated[-1]
            hidden_buffers[0].copy_(model.model.embed_tokens(token))
            kv_length = args.input_length + decode_idx
            last_page_len.fill_(((kv_length - 1) % page_size) + 1)
            plan_flashinfer(wrapper, indptr, indices, last_page_len,
                            q_heads, kv_heads, head_dim, page_size)
            inv_freq = 1.0 / (
                float(config.rope_theta) **
                (torch.arange(0, head_dim, 2, device="cuda").float()
                 / head_dim))
            freqs = float(kv_length - 1) * inv_freq
            rope = torch.cat((freqs, freqs), dim=-1)
            cos = rope.cos().to(DTYPE).view(1, 1, head_dim)
            sin = rope.sin().to(DTYPE).view(1, 1, head_dim)

            for layer_idx, layer in enumerate(model.model.layers):
                normed_buffers[layer_idx].copy_(rms_norm(
                    hidden_buffers[layer_idx],
                    layer.input_layernorm.weight))
                stage_start.record()
                prefixes[layer_idx]()
                stage_end.record()
                stage_end.synchronize()
                timing["prefix_ms"] += stage_start.elapsed_time(stage_end)

                qkv = qkv_buffers[layer_idx]
                q = qkv[:, :q_size].view(batch, q_heads, head_dim)
                k = qkv[:, q_size:q_size + kv_size].view(
                    batch, kv_heads, head_dim)
                v = qkv[:, q_size + kv_size:].view(
                    batch, kv_heads, head_dim)
                q = apply_rope(rms_norm(
                    q, layer.self_attn.q_norm.weight), cos, sin)
                k = apply_rope(rms_norm(
                    k, layer.self_attn.k_norm.weight), cos, sin)
                page_k = k_cache[layer_idx].view(
                    batch, -1, kv_heads, head_dim)
                page_v = v_cache[layer_idx].view_as(page_k)
                page_k[:, kv_length - 1].copy_(k)
                page_v[:, kv_length - 1].copy_(v)

                stage_start.record()
                attention_buffers[layer_idx].copy_(wrapper.run(
                    q, (k_cache[layer_idx], v_cache[layer_idx])))
                stage_end.record()
                stage_end.synchronize()
                timing["flashinfer_ms"] += stage_start.elapsed_time(stage_end)

                stage_start.record()
                suffixes[layer_idx]()
                stage_end.record()
                stage_end.synchronize()
                timing["suffix_ms"] += stage_start.elapsed_time(stage_end)

                # The current batched Hopper MPK RMSNorm has a known numeric
                # discrepancy.  Keep norm/MLP on Torch in this first full
                # executor validation; a later step can move them across the
                # boundary independently after its correctness gate passes.
                stage_start.record()
                attn_residual = suffixes[layer_idx].step35_attn_residual
                mlp_input = rms_norm(
                    attn_residual,
                    layer.post_attention_layernorm.weight)
                mlp_output = layer.mlp.down_proj(
                    torch.nn.functional.silu(layer.mlp.gate_proj(mlp_input))
                    * layer.mlp.up_proj(mlp_input))
                hidden_buffers[layer_idx + 1].copy_(
                    attn_residual + mlp_output)
                stage_end.record()
                stage_end.synchronize()
                timing["host_mlp_ms"] += stage_start.elapsed_time(stage_end)

            final_hidden = rms_norm(
                hidden_buffers[-1], model.model.norm.weight)
            logits = model.lm_head(final_hidden)
            generated.append(logits.argmax(dim=-1))
        end_total.record()
        end_total.synchronize()
    timing["decode_ms"] = start_total.elapsed_time(end_total)

    generated_ids = torch.stack(generated, dim=1)
    positional_matches = (generated_ids == reference_ids).sum(dim=1)
    first10 = min(args.output_length, 10)
    first10_matches = (
        generated_ids[:, :first10] == reference_ids[:, :first10]
    ).sum(dim=1)
    minimum_first10 = int(first10_matches.min().item())
    full_matches = int(positional_matches.min().item())
    status = "passed" if minimum_first10 == first10 else "failed"
    steps = max(args.output_length - 1, 1)
    summary = {
        "step": 35,
        "status": status,
        "model": args.model,
        "batch_size": batch,
        "s_in": args.input_length,
        "s_out": args.output_length,
        "minimum_first10_matches": minimum_first10,
        "minimum_full_matches": full_matches,
        "reference_tokens_request0": reference_ids[0].tolist(),
        "hybrid_tokens_request0": generated_ids[0].tolist(),
        "timing_ms": timing,
        "decode_step_ms": timing["decode_ms"] / steps,
        "prefix_ms_per_step": timing["prefix_ms"] / steps,
        "flashinfer_ms_per_step": timing["flashinfer_ms"] / steps,
        "suffix_ms_per_step": timing["suffix_ms"] / steps,
        "host_mlp_ms_per_step": timing["host_mlp_ms"] / steps,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Step 35 full-model Hybrid: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
