# Qwen3 MPK decode parity with Optimized Normal

The performance comparison may describe MPK decode as fully aligned only when
every row below has an operator-level GPU correctness test and is marked
equivalent in `tests/benchmarks/qwen3_backend_comparison.py`.

| Optimization objective | Optimized Normal | MPK implementation | Current state |
| --- | --- | --- | --- |
| Launch amortization | CUDA Graph | Persistent kernel | Equivalent method |
| Paged attention | FlashInfer paged attention | Mirage paged attention | Needs algorithm parity |
| KV page size | 128 tokens | 128 tokens | Equivalent |
| Split-KV | FlashInfer split and LSE merge | Mirage fixed-chunk split and merge | Needs split/merge parity |
| QKV projection | One fused QKV projection | One shuffled QKV projection | Equivalent method |
| Q/K norm, RoPE, KV write | Fused decode kernel | Fused in MPK attention task | Equivalent method |
| Attention output and residual | Projection, fused add RMSNorm | Fused projection and residual, then RMSNorm | Equivalent memory fusion |
| RMSNorm | FlashInfer RMSNorm | Mirage Hopper RMSNorm | Equivalent: exact BF16 H100 probe |
| Gate/up projection | One fused gate/up projection | One shuffled gate/up projection | Equivalent method |
| Activation | SiLU and multiply | Mirage SiLU and multiply | Equivalent method |
| Down projection and residual | Projection then residual add | Fused projection and residual | Equivalent method |
| Token selection | Greedy argmax | Partial/reduce argmax | Equivalent tie-break semantics |

## Required gates

1. Compare each stage using identical input tensors, weights, KV page tables,
   positions, and BF16/FP32 accumulation settings.
2. Test context lengths 128 and 1024, including a request that crosses several
   128-token page boundaries.
3. For attention, compare output and LSE before the output projection.  Test
   one and multiple split-KV chunks separately.
4. For RMSNorm, compare standalone input norm, post-attention norm, and final
   norm at the tolerances used by the model correctness suite.
5. Compare logits and argmax at teacher-forced checkpoints.  Autoregressive
   token equality alone is not an operator parity test after the first
   divergence.
6. Permit a benchmark backend named `aligned` only after every manifest item is
   equivalent.  Until then use `mpk_decode_only_page128_split_kv`.

The optimized implementations do not have to share source code.  They must
perform the same operation, preserve the same important fusion or launch
amortization, and pass the operator-level gates above.
