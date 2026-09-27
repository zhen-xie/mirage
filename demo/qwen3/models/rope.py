import triton
import triton.language as tl
import torch


@triton.jit
def rotary_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    seq_len,
    num_kv_heads,
    half_dim,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    out_q_stride_b,
    out_q_stride_s,
    out_q_stride_h,
    out_q_stride_d,
    out_k_stride_b,
    out_k_stride_s,
    out_k_stride_h,
    out_k_stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_s * BLOCK_M >= seq_len:
        return

    rm = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    s_mask = rm < seq_len

    q_base = pid_b * q_stride_b + pid_h * q_stride_h
    out_q_base = pid_b * out_q_stride_b + pid_h * out_q_stride_h

    rk_half = tl.arange(0, BLOCK_K // 2)
    rk_half_mask = rk_half < half_dim

    q_offset = q_base + (
        rm[:, None] * q_stride_s + rk_half[None, :] * q_stride_d
    )  # half

    cos_offset = rm[:, None] * cos_stride_s + rk_half[None, :] * cos_stride_d
    sin_offset = rm[:, None] * sin_stride_s + rk_half[None, :] * sin_stride_d

    cos = tl.load(
        cos_ptr + cos_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=1.0
    )
    sin = tl.load(
        sin_ptr + sin_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
    )

    q_first = tl.load(
        q_ptr + q_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
    )
    q_second = tl.load(
        q_ptr + q_offset + half_dim * q_stride_d,
        mask=s_mask[:, None] & rk_half_mask[None, :],
        other=0.0,
    )

    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_second * cos + q_first * sin

    out_q_offset = out_q_base + (
        rm[:, None] * out_q_stride_s + rk_half[None, :] * out_q_stride_d
    )

    tl.store(
        out_q_ptr + out_q_offset,
        q_out_first,
        mask=s_mask[:, None] & rk_half_mask[None, :],
    )
    tl.store(
        out_q_ptr + out_q_offset + half_dim * out_q_stride_d,
        q_out_second,
        mask=s_mask[:, None] & rk_half_mask[None, :],
    )

    if pid_h < num_kv_heads:
        k_base = pid_b * k_stride_b + pid_h * k_stride_h
        k_offset = k_base + (rm[:, None] * k_stride_s + rk_half[None, :] * k_stride_d)

        k_first = tl.load(
            k_ptr + k_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
        )
        k_second = tl.load(
            k_ptr + k_offset + half_dim * k_stride_d,
            mask=s_mask[:, None] & rk_half_mask[None, :],
            other=0.0,
        )
        k_out_first = k_first * cos - k_second * sin
        k_out_second = k_second * cos + k_first * sin
        out_k_base = pid_b * out_k_stride_b + pid_h * out_k_stride_h

        out_k_offset = out_k_base + (
            rm[:, None] * out_k_stride_s + rk_half[None, :] * out_k_stride_d
        )
        tl.store(
            out_k_ptr + out_k_offset,
            k_out_first,
            mask=s_mask[:, None] & rk_half_mask[None, :],
        )
        tl.store(
            out_k_ptr + out_k_offset + half_dim * out_k_stride_d,
            k_out_second,
            mask=s_mask[:, None] & rk_half_mask[None, :],
        )


def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Triton implementation of apply_rotary_pos_emb with 3D grid.

    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim]
        cos/sin: Tensors from rotary embedding [1, seq_len_ro, head_dim]
        position_ids: Optional tensor of position indices [batch, seq_len]
        unsqueeze_dim: Dimension to unsqueeze the cosine/sine tensors

    Returns:
        Tuple of rotated query and key tensors
    """

    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    half_dim = head_dim // 2

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_stride = q.stride()
    k_stride = k.stride()
    cos_stride = cos.stride()
    sin_stride = sin.stride()
    q_out_stride = q_out.stride()
    k_out_stride = k_out.stride()

    BLOCK_K = (
        32
        if head_dim <= 32
        else (64 if head_dim <= 64 else (128 if head_dim <= 128 else 256))
    )
    BLOCK_M = 8 if head_dim <= 128 else 4

    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size, num_heads)
    with torch.cuda.device(q.device):
        rotary_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            seq_len,
            num_kv_heads,
            half_dim,
            q_stride[0],
            q_stride[1],
            q_stride[2],
            q_stride[3],
            k_stride[0],
            k_stride[1],
            k_stride[2],
            k_stride[3],
            cos_stride[1],
            cos_stride[2],
            sin_stride[1],
            sin_stride[2],
            q_out_stride[0],
            q_out_stride[1],
            q_out_stride[2],
            q_out_stride[3],
            k_out_stride[0],
            k_out_stride[1],
            k_out_stride[2],
            k_out_stride[3],
            BLOCK_M,
            BLOCK_K,
        )
    return q_out, k_out


@triton.jit
def decode_rotary_kv_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    key_cache_ptr,
    value_cache_ptr,
    step_ptr,
    num_kv_heads,
    half_dim,
    q_stride_b,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_d,
    cos_stride_d,
    sin_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    kc_stride_b,
    kc_stride_s,
    kc_stride_h,
    kc_stride_d,
    vc_stride_b,
    vc_stride_s,
    vc_stride_h,
    vc_stride_d,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    rk = tl.arange(0, BLOCK_K // 2)
    mask = rk < half_dim

    cos = tl.load(cos_ptr + rk * cos_stride_d, mask=mask, other=1.0)
    sin = tl.load(sin_ptr + rk * sin_stride_d, mask=mask, other=0.0)

    q_offset = (
        pid_b * q_stride_b + pid_h * q_stride_h + rk * q_stride_d
    )
    q_first = tl.load(q_ptr + q_offset, mask=mask, other=0.0)
    q_second = tl.load(
        q_ptr + q_offset + half_dim * q_stride_d,
        mask=mask,
        other=0.0,
    )
    q_out_offset = (
        pid_b * q_out_stride_b
        + pid_h * q_out_stride_h
        + rk * q_out_stride_d
    )
    tl.store(
        q_out_ptr + q_out_offset,
        q_first * cos - q_second * sin,
        mask=mask,
    )
    tl.store(
        q_out_ptr + q_out_offset + half_dim * q_out_stride_d,
        q_second * cos + q_first * sin,
        mask=mask,
    )

    if pid_h < num_kv_heads:
        position = tl.load(step_ptr)
        k_offset = (
            pid_b * k_stride_b + pid_h * k_stride_h + rk * k_stride_d
        )
        k_first = tl.load(k_ptr + k_offset, mask=mask, other=0.0)
        k_second = tl.load(
            k_ptr + k_offset + half_dim * k_stride_d,
            mask=mask,
            other=0.0,
        )
        kc_offset = (
            pid_b * kc_stride_b
            + position * kc_stride_s
            + pid_h * kc_stride_h
            + rk * kc_stride_d
        )
        tl.store(
            key_cache_ptr + kc_offset,
            k_first * cos - k_second * sin,
            mask=mask,
        )
        tl.store(
            key_cache_ptr + kc_offset + half_dim * kc_stride_d,
            k_second * cos + k_first * sin,
            mask=mask,
        )

        v_offset = (
            pid_b * v_stride_b + pid_h * v_stride_h + rk * v_stride_d
        )
        vc_offset = (
            pid_b * vc_stride_b
            + position * vc_stride_s
            + pid_h * vc_stride_h
            + rk * vc_stride_d
        )
        tl.store(
            value_cache_ptr + vc_offset,
            tl.load(v_ptr + v_offset, mask=mask, other=0.0),
            mask=mask,
        )
        tl.store(
            value_cache_ptr + vc_offset + half_dim * vc_stride_d,
            tl.load(
                v_ptr + v_offset + half_dim * v_stride_d,
                mask=mask,
                other=0.0,
            ),
            mask=mask,
        )


def apply_rotary_pos_emb_triton_decode_cache(
    q, k, v, cos, sin, key_cache, value_cache, step
):
    """Apply decode RoPE and append K/V to fixed-address paged caches."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    if seq_len != 1:
        raise ValueError("Fused decode RoPE/cache append requires one token")
    if key_cache.shape[0] < batch_size or value_cache.shape[0] < batch_size:
        raise ValueError("KV cache does not have one page per request")

    q_out = torch.empty_like(q)
    half_dim = head_dim // 2
    block_k = triton.next_power_of_2(head_dim)
    grid = (batch_size, num_heads)

    with torch.cuda.device(q.device):
        decode_rotary_kv_cache_kernel[grid](
            q,
            k,
            v,
            cos,
            sin,
            q_out,
            key_cache,
            value_cache,
            step,
            num_kv_heads,
            half_dim,
            q.stride(0),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(3),
            cos.stride(-1),
            sin.stride(-1),
            q_out.stride(0),
            q_out.stride(2),
            q_out.stride(3),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            BLOCK_K=block_k,
        )
    return q_out
