# Qwen3 workload-aware execution: Step 7 correctness record

The second GPU environment is an NVIDIA H100 NVL (95830 MiB), driver
590.48.01, CUDA toolkit 13.3, PyTorch 2.6.0+cu124, Transformers 4.57.1,
FlashInfer 0.7.0, and Apache TVM FFI 0.1.14.post0. The reported remote
commit before the probe changes was `e5447c91e7df795279c748891e2afba503c8761e`.
`TVM_FFI_DISABLE_TORCH_C_DLPACK=1` was required to bypass an incompatible
optional `torch_c_dlpack_ext` shared library when importing FlashInfer.

The prompt was generated on this server to yield exactly 128 input tokens
under the Qwen3 chat template. This is a different prompt from the first GPU
environment, so its results are compared only within this environment.

With 128 input tokens and 128 generated tokens, `normal`, `always`,
`decode-only`, and `prefill-only` all completed. Each MPK mode matched the
normal mode for the first 30 generated tokens. `pytest -q tests/qwen3/`
reported 6 passed and 3 subtests passed with the four output artifacts.
The later agreed correctness gate is at least 20 matching token IDs at the
same positions among the first 30 generated tokens. Full-generation equality
is not claimed.

The two-token diagnostic compared normal decode with MPK decode-only. Both
runs used the same input and first generated token. Captured logits from the
second token position predicted the second generated token in each run.

| Tensor | Max absolute error | Mean absolute error | Cosine similarity |
| --- | ---: | ---: | ---: |
| Logits | 0.4375 | 0.06970496475696564 | 0.9997003674507141 |
| Normalized hidden state | 0.40625 | 0.03687795251607895 | 0.9997891187667847 |

Both runs generated `[151667, 198]`. The measurements are from one prompt
and one decode position. No numeric error acceptance threshold has been set.
The probe should be repeated at other context lengths before treating these
values as representative.

A second two-token diagnostic used a 1024-token prompt on the same server.
Both runs again generated `[151667, 198]`:

| Tensor | Max absolute error | Mean absolute error | Cosine similarity |
| --- | ---: | ---: | ---: |
| Logits | 0.3125 | 0.05400776490569115 | 0.999839723110199 |
| Normalized hidden state | 0.75 | 0.03082827851176262 | 0.999851405620575 |

Both prompts repeat `hello`, so these two cases vary context length but do
not provide diverse text or generated continuations. The numeric probe covers
one decode position per prompt, while the 30-token artifact check covers only
the 128-token prompt on the second server at the time of the probe. A later
1024-input, 128-output normal/decode-only run also matched on the first 30
tokens. No numeric threshold or full-output correctness claim follows from
these observations.

## Step 8 initial timing smoke test

On the H100 NVL, a single B=1, context=128, 16-decode-step run completed
through `tests/benchmarks/qwen3_decode_backend.py`. Decode-only CUDA event
totals were 362.881 ms for normal and 96.988 ms for MPK. These are one sample
per backend, with each backend in a separate cold process. The script's
original `relative_total_range=0` for one sample had no stability meaning;
the script now emits `null` until there are at least two samples. This is
only a smoke test of the measurement path, not a performance conclusion.

A subsequent B=1, context=128 run used 128 decode steps and three separate
processes per backend, alternating backend order. Normal decode totals were
2660.576, 2629.949, and 2636.320 ms (mean 2642.282 ms; range/mean 1.16%).
MPK decode-only totals were 772.649, 779.367, and 779.796 ms (mean 777.270
ms; range/mean 0.92%). The ratio of these decode-only CUDA event means is
3.40. Each process began cold and the persistent MPK kernel provides no
per-step latency distribution. This result supports stability for this one
B=1, context=128 configuration under the current method; it is not a broad
performance boundary or a whole-request speedup.

The same three-process method at B=1, context=1024, and 128 decode steps
gave normal totals of 2643.205, 2638.188, and 2664.494 ms (mean 2648.629
ms; range/mean 0.99%). MPK decode-only totals were 905.400, 904.684, and
899.262 ms (mean 903.115 ms; range/mean 0.68%). The decode CUDA event mean
ratio was 2.93. Both tested B=1 contexts have under 2% range/mean in this
method. These results do not validate B>1 or other context lengths.

## Diverse prompt numerical divergence

A 128-token prompt derived from the Milestone 1 notes produced a different
result from the repeated `hello` prompt. All four modes generated 128 tokens.
`prefill-only` matched normal across the 100 saved tokens. `always` first
diverged at zero-based generated token 35. `decode-only` first diverged at
zero-based generated token 20. This failed the earlier exact-first-30 gate.
The first 20 positions match, so it passes the revised 20-of-30 gate; the
full positional match count will be computed from the saved artifacts.

At the step predicting generated token 20, normal logits ranked token 323 at
31.375 and token 11 at 31.25. Decode-only logits placed both at 31.375 and
selected token 11. Their logit cosine similarity was 0.999928 and normalized
hidden-state cosine similarity was 0.999953; max absolute errors were 0.25
for both tensors. The near tie explains the immediate token difference, but
does not yet identify whether the underlying numeric difference comes from
prefill KV values, MPK decode arithmetic, or both.

At the same step, the `always` probe generated token 323. Its MPK logits
ranked token 323 at 31.625 and token 11 at 31.5, preserving the normal
ordering. Thus the decode-only mismatch is specific to the mixed path in
this case. More evidence is needed to separate prefill KV differences from
resume-state effects.

At the first decode step for this prompt, all tested modes generated token
198. The top-logit gap from normal to the runner-up was 12.0, so these
arithmetic differences did not affect token selection there. Relative to
normal, `always` logits had mean absolute error 0.08029 and cosine 0.999610;
`decode-only` had mean absolute error 0.04999 and cosine 0.999845. The
corresponding normalized hidden-state mean absolute errors were 0.04453 and
0.02921. Thus differences are present from the first decode step, but the
decode-only path is closer to normal on these global metrics at that step.
The `prefill-only` combination was subsequently measured. At the first
decode step its logit mean absolute error was 0.07592 and hidden-state mean
absolute error was 0.04408 relative to normal. At generated token 20 it
still selected token 323, with logits 31.625 for token 323 and 31.5 for
token 11. Thus both combinations that use MPK prefill preserved the normal
argmax at this point, while normal prefill plus MPK decode produced the tie.
This does not yet prove that the resume path is identical to continuous MPK
execution; a split-MPK diagnostic is needed to isolate that factor.

The split-MPK diagnostic has now run at generated token 20. A continuous
`always` run and a run that stopped after MPK prefill then resumed MPK decode
had exactly equal generated tokens, logits, and normalized hidden state.
This rules out an observable resume-path difference for this configuration;
the next diagnostic compares the normal and MPK prefill KV snapshots directly.

The first KV snapshot comparison used the diverse 128-token prompt. Both
caches had shape `[36, 128, 8, 128]`. Key mean absolute error was 0.01374
with max 2.0; value mean absolute error was 0.01884 with max 2.125. Value
cache mean absolute error was largest in layers 34, 33, 35, 32, and 31
(0.0950, 0.0932, 0.0730, 0.0617, and 0.0491). The first comparison script
printed a key cosine above 1 due to fp32 reduction error; that number is
invalid. The script now uses fp64 reduction and reports relative RMSE and
per-layer/position errors. These need to be recomputed from the saved KV
snapshots before interpreting the distribution.

Recomputation with fp64 reductions gave key cosine 0.999970 and relative
RMSE 0.00773, and value cosine 0.999793 and relative RMSE 0.02032. Key
mean absolute error rises from roughly 0.006-0.007 in the first three layers
to 0.015-0.017 in the last five. Value error rises from 0.00005-0.00044 in
the first three layers to 0.049-0.095 in the last five. This pattern is
consistent with differences accumulating across layers, but does not alone
prove the KV values cause the decode-only token divergence. A controlled KV
replacement probe is needed for that causal check.

That controlled replacement has now run at generated token 20. A normal
prefill followed by MPK decode originally selected token 11. Replacing only
its populated KV cache with the saved MPK prefill KV before MPK decode made
the generated tokens, logits, and normalized hidden state exactly equal to
the continuous `always` run; it selected token 323. Together with the
split-MPK equality check, this identifies the different prefill KV values
as the cause of this particular mixed-path divergence. It does not imply the
normal prefill KV values are invalid; both backends use BF16 arithmetic and
their small numeric differences can flip an almost tied argmax.

An fp32 candidate rescore of the BF16 normalized hidden state and BF16
lm-head weights at the divergent step ranked token 323 above token 11 in
both normal and decode-only. Normal candidate scores were 31.319107 vs
31.295866 (gap 0.023241); decode-only scores were 31.410164 vs 31.351900
(gap 0.058264). The decode-only BF16 projection had rounded both scores
to 31.375, and its argmax tie behavior selected token 11. This identifies
the BF16 projection tie as the immediate selection mechanism for this case.
The earlier strict first-30-token check fails, but exact equality is no longer
the acceptance criterion. Resolving ties with extra precision inside the
persistent kernel remains a possible future experiment, with a latency cost
to measure; changing tie order to favor one token ID would not be general.
