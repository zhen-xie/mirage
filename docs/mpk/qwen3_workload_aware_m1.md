# Qwen3 workload-aware execution: Milestone 1 record

Baseline source commit: `66523b6c033fc4fed9bd47f8e5cb07664980759b`.
The remote baseline files are `tests/benchmarks/baselines/qwen3_normal_baseline.json`
and `qwen3_mpk_baseline.json`.

Environment reported for baseline: NVIDIA H100 80GB HBM3; driver 580.178.04;
CUDA toolkit 13.2; PyTorch 2.14.0+cu130; Transformers 4.57.1;
FlashInfer 0.6.18.post1; Qwen/Qwen3-8B; bfloat16.

| Batch | Input tokens | Output tokens | Normal | MPK always | First 30 tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 128 | 128 | ran | ran | match |
| 1 | 1024 | 128 | ran | ran | match |

The baseline JSON saves 100 generated tokens. The first observed mismatch is at
zero-based output position 43 for input 128 and 95 for input 1024. The agreed
Milestone 1 correctness gate compares the first 30 generated tokens only.
This does not establish correctness for the full generation.

The original demo's reported `latency_ms_per_token` values use different timing
regions for normal and MPK. Do not use them to calculate a speedup. Batch 8 is
not measured because the current normal demo is single-request only.

Validated on the same H100 with 128 and 1024 input tokens, 128 output tokens:

- `--backend normal` and `--backend mpk --mpk-policy always` run.
- `--backend mpk --mpk-policy decode-only` runs normal prefill, then MPK decode.
- `--backend mpk --mpk-policy prefill-only` runs MPK prefill, then normal decode.
- Both mixed modes match the normal baseline for the first 30 generated tokens.
- Legacy `--use-mirage` runs MPK and prints a deprecation notice.
- `--backend normal --mpk-policy always` reports an argument error.
- Phase timings were captured for both mixed modes; these are single, unwarmed
  measurements and are not performance acceptance results.

The mixed backend implementation is currently limited to one GPU, one request,
greedy decoding, and no speculative decoding. MPK decode per-step timing is not
yet exported by the persistent kernel.
