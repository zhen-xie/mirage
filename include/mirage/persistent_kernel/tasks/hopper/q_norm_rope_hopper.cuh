/* Copyright 2025 CMU
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include "../common/utils.cuh"

namespace kernel {

template <typename T,
          int NUM_QO_PER_KV,
          int HEAD_DIM,
          int QKV_STRIDE,
          int PAGE_SIZE,
          int MAX_TOKENS>
__device__ __forceinline__ void q_norm_rope_preprocess_hopper(
    void const *qkv_ptr,
    void const *q_norm_weight_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    void *output_ptr,
    int const *qo_indptr_buffer_ptr,
    int const *paged_kv_indptr_buffer_ptr,
    int const *paged_kv_last_page_len_buffer_ptr,
    int request_id,
    float eps = 1e-6f) {
  static_assert(NUM_QO_PER_KV <= 4,
                "one consumer warp is assigned to each Q head");
  int const first_token = qo_indptr_buffer_ptr[request_id];
  int const last_token = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = last_token - first_token;
  if (num_tokens <= 0) {
    return;
  }
  int const first_page = paged_kv_indptr_buffer_ptr[request_id];
  int const last_page = paged_kv_indptr_buffer_ptr[request_id + 1];
  int const global_seq_len =
      (last_page - first_page - 1) * PAGE_SIZE +
      paged_kv_last_page_len_buffer_ptr[request_id];

  T const *input = static_cast<T const *>(qkv_ptr) +
                   first_token * QKV_STRIDE;
  T *output = static_cast<T *>(output_ptr) + first_token * QKV_STRIDE;
  T const *weight = static_cast<T const *>(q_norm_weight_ptr);
  T const *cosine = static_cast<T const *>(cos_ptr);
  T const *sine = static_cast<T const *>(sin_ptr);
  int const warp = warp_id();
  int const lane = lane_id();
  unsigned const mask = 0xffffffffu;

  if (warp < NUM_QO_PER_KV) {
    for (int token = 0; token < num_tokens && token < MAX_TOKENS; ++token) {
      int const row_offset = token * QKV_STRIDE + warp * HEAD_DIM;
      float sum = 0.0f;
#pragma unroll
      for (int col = lane; col < HEAD_DIM; col += 32) {
        float const value = (float)input[row_offset + col];
        sum += value * value;
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(mask, sum, offset);
      }
      float const inv_rms = rsqrt(sum / float(HEAD_DIM) + eps);

      // Qwen3 uses full-head RoPE. A lane owns a low/high pair, so neither
      // half can be overwritten before its partner is consumed.
#pragma unroll
      for (int col = lane; col < HEAD_DIM / 2; col += 32) {
        int const high_col = col + HEAD_DIM / 2;
        float const low =
            (float)input[row_offset + col] * inv_rms * (float)weight[col];
        float const high = (float)input[row_offset + high_col] * inv_rms *
                           (float)weight[high_col];
        int const position = global_seq_len - num_tokens + token;
        T const *token_cos = cosine + position * HEAD_DIM;
        T const *token_sin = sine + position * HEAD_DIM;
        output[row_offset + col] =
            (T)(low * (float)token_cos[col] - high * (float)token_sin[col]);
        output[row_offset + high_col] =
            (T)(high * (float)token_cos[high_col] +
                low * (float)token_sin[high_col]);
      }
    }
  }
}

} // namespace kernel
