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
full positional match count was later measured as 24/30. On the repeated
`hello` prompt, all four modes scored 30/30. The revised pytest suite reported
7 passed and 3 subtests passed on the second server.

An additional positional comparison of the first 50 saved tokens used the
same artifacts. On the repeated `hello` prompt, `always`, `decode-only`, and
`prefill-only` each matched normal at 50/50 positions. On the diverse prompt,
`always` matched at 35/50 (70%; first mismatch at zero-based position 35),
`decode-only` at 29/50 (58%; first mismatch at position 20), and
`prefill-only` at 50/50. These are positional match rates after generation;
an early token difference can change the later continuation. The agreed
acceptance gate remains at least 20 matching positions among the first 30.

## Step 8 batch coverage

The normal path has a lockstep batch implementation: each request uses its
own KV page, batched attention, and its own greedy token output. It requires
one GPU, greedy decoding, `--ignore-eos`, and one KV page per request. On the
H100 NVL, a B=2 run with two distinct 128-token prompts completed 128-token
generation. Both requests matched their B=1 reference at 30/30 positions;
the first 50 positions matched at 43/50 and 50/50. A separate B=1 rerun
matched its existing diverse-prompt reference at 50/50. The B=2 path passes
the agreed correctness gate for these two requests, though more prompts and
batch sizes remain untested.

MPK `decode-only` resume seeds one prefilled KV page and one first generated
token per request. On the H100 NVL, the updated B=1 path matched its prior
output at 50/50 positions. In B=2, the two requests matched B=2 normal at
30/30 and 24/30 positions, respectively; their first-50 counts were 43/50
and 29/50. Both passed the agreed correctness gate.

A B=8 run with eight distinct 128-token prompts completed 128-token generation
for normal and `decode-only`. Against each request's B=1 normal result, B=8
normal matched at least 22/30 positions. Against B=8 normal, B=8
`decode-only` matched at least 20/30 positions, with request 5 exactly at
the threshold. The individual first-50 counts for normal versus B=1 were
43, 50, 50, 50, 50, 22, 50, and 50; for `decode-only` versus B=8 normal
they were 43, 29, 50, 47, 50, 20, 50, and 50. These are correctness smoke
tests, not timing results. The benchmark driver now accepts a JSON list of
distinct equal-length prompts for B>1 and reports aggregate generated tokens
per second over the batch. Remote timing validation is next.

The next batch change permits distinct prompts for `always` and
`prefill-only`. Split MPK prefill reserves every request's KV page so normal
decode can consume it, and `--phase-timing` splits `always` at the prefill
boundary to measure decode separately. On the H100 NVL, B=8 continuous
`always`, split timed `always`, and `prefill-only` all completed with eight
distinct prompts. Their minimum first-30 positional matches against B=8
normal were 24/30, 22/30, and 22/30, respectively. Split and continuous
`always` matched at 50/50 positions for seven requests but only 22/50 for
request 5. Therefore split timing characterizes a two-launch MPK policy,
not an exactly output-equivalent continuous `always` run. The smoke test
reported split `always` prefill/decode at 949.892/795.216 ms and
`prefill-only` at 906.431/2395.238 ms.

The B=8, context=128, 127-decode-step benchmark then completed three cold
processes per mode with `--warmup 0`. Means in milliseconds were:

| Mode | Prefill | Decode | Prefill + decode | Aggregate decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| normal | 388.316 | 2044.670 | 2432.986 | 496.90 |
| split `always` | 902.175 | 795.113 | 1697.288 | 1277.81 |
| `decode-only` | 365.486 | 790.704 | 1156.190 | 1284.93 |
| `prefill-only` | 901.464 | 2387.098 | 3288.562 | 425.62 |

The decode latency range divided by its mean was 1.47%, 0.04%, 1.75%, and
1.63%, respectively. Every repeat had the same first-30 positional match
counts against normal: split `always` and `prefill-only` had minimum 22/30,
and `decode-only` had minimum 20/30. These CUDA event numbers exclude model
load and compilation, and the split `always` numbers must not be presented
as continuous `always` timing.

A B=8, context=1024, 127-decode-step smoke run used eight distinct seed
prompts extended to exactly 1024 input tokens. All eight requests matched
normal at 30/30 first-token positions for each MPK policy. The single-run
prefill/decode CUDA event times in milliseconds were normal 663.929/2050.613,
split `always` 7627.084/909.912, `decode-only` 641.646/925.826, and
`prefill-only` 7695.472/2393.241. MPK prefill is much slower at this
configuration with the current eight-token batch capacity; a three-repeat
measurement is needed before treating the gap as stable. These results do
not include model load or compilation.

The B=8, context=1024 run was repeated three times with the same 127 decode
iterations and `--warmup 0`. Mean CUDA event times in milliseconds were:

| Mode | Prefill | Decode | Prefill + decode | Aggregate decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| normal | 659.384 | 2050.707 | 2710.091 | 495.44 |
| split `always` | 7674.628 | 911.028 | 8585.655 | 1115.22 |
| `decode-only` | 641.577 | 923.176 | 1564.753 | 1100.55 |
| `prefill-only` | 7728.882 | 2375.681 | 10104.563 | 427.67 |

Decode range/mean was 2.17%, 0.20%, 0.25%, and 1.19%, respectively.
Each policy matched normal at all first 30 positions for all eight requests
in every repeat. The normal/decode-only decode mean ratio was 2.22; the
prefill-plus-decode ratio was 1.73. MPK prefill cost remains about 7.7 s
under the current eight-token batch-capacity setting. These runs generated
128 tokens, with the first token produced during prefill; thus they measured
127 decode iterations. The Step 8 acceptance case calls for 128 decode
iterations.

The exact Step 8 B=8, context=1024, 128-decode-iteration case then ran three
cold processes per mode. Mean CUDA event times in milliseconds were:

| Mode | Prefill | Decode | Prefill + decode | Aggregate decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| normal | 663.180 | 2079.036 | 2742.216 | 492.54 |
| split `always` | 7677.418 | 917.935 | 8595.353 | 1115.55 |
| `decode-only` | 643.284 | 919.410 | 1562.695 | 1113.76 |
| `prefill-only` | 7687.961 | 2378.708 | 10066.669 | 430.49 |

The normal/decode-only decode latency ratio was 2.261. Decode range/mean
was 0.68%, 0.16%, 0.72%, and 0.66%, respectively. All eight requests
matched normal at 30/30 positions under each MPK policy in all three
repeats. This meets the Step 8 B=8, context=1024, 128-step correctness and
three-repeat stability check. It remains a fresh-process CUDA event
measurement with `--warmup 0`; no same-process steady-state warmup was
performed. Split `always` timing is a two-launch diagnostic and need not
equal the continuous `always` path.

For the first Step 9 batch pilot at context=1024 and 128 decode iterations,
three cold processes per backend gave B=2 normal/`decode-only` mean decode
latencies of 2062.421/905.803 ms (2.277 ratio), and B=4 means of
2062.725/916.283 ms (2.251 ratio). Every request matched normal at 30/30
first positions in each repeat. Alongside the B=8 result, the three tested
batch sizes all favor MPK decode under this method. A context sweep and
higher batch sizes remain unmeasured.

For a resumable Step 9 pilot, `tests/benchmarks/qwen3_decode_sweep.py`
generates equal-length prompts from eight distinct seeds and measures all
four execution policies for each case: normal, `always`, `decode-only`, and
`prefill-only`. It records `always` twice: split MPK launches for phase
timing and one continuous MPK launch for its actual total duration and
output. `raw_results.csv` has one row per timing mode and case, including
prefill, decode, phase-sum timing, per-repeat values, variability, positional
token matches, speedup versus normal (including continuous always total
speedup), and environment metadata. Each case
keeps its log and complete JSON summary. The sweep now enumerates the requested
three-dimensional grid B={1,2,4,8,16,32,64,128},
S_IN={16,32,64,128,256,512,1024}, and
S_OUT={16,32,64,128,256,512,1024}: 392 configurations and five timing
rows per configuration. S_OUT counts all generated tokens, including the
first token produced during prefill, so decode iterations equal S_OUT−1.
The sweep uses the user-only Qwen3 chat template to make S_IN=16 feasible;
earlier baselines with a system message are separate experiments. It selects
one power-of-two KV page per
request that fits the prompt and decode output, passes the corresponding
page/request/token capacities to the demo, and records memory-skipped or
failed cells explicitly. The sweep also retains timing for a policy that
misses the agreed 20/30 correctness gate and marks its row
`correctness_failed`, so the grid does not silently lose that measurement.
For S_OUT=16, the sweep compares all 16 saved positions and reports that
the 20/30 gate is inapplicable.
B>8 and contexts above 2048 remain experimental
until remote GPU validation; a CSV row for such a cell is not evidence that
the kernel supports it. The CSV is not an advantage map until sufficient
cells complete with stable timing and correctness.
The `--cases B:S_IN:S_OUT` option selects a few edge cases for GPU validation
before the full grid. The three axes can also be set through `B_VALUES`,
`S_IN_VALUES`, and `S_OUT_VALUES` environment variables.
The sweep prints a 392-case progress bar with elapsed time, an approximate
remaining time after the first measured case, the active policy/repeat, and
a 30-second heartbeat during long samples. It also writes the latest state
to `progress.json` beside `raw_results.csv`; rerunning an unchanged sweep
still reuses completed case summaries.
The first 3D smoke run failed all four cases during model construction:
`Qwen3Attention` asserted a fixed KV cache shape with 16 pages of 4096
tokens, while the sweep allocated per-case page counts and sizes. That
assertion now checks the invariant layer, head, and key/value dimensions
without fixing page count or page size. The smoke command should use
`--fail-on-failed-cases` so a CSV containing failed rows does not report a
successful process exit. The next smoke run reached inference: the 16+16
case failed MPK compilation because the 32-token KV page was smaller than
Hopper's 64-token attention tile. The sweep now uses a 64-token page minimum.
The 32+32 case passed; two B=8 cases had policies with only 4/30 positional
matches on at least one request. The sweep accepts `--min-page-size 4096`
to compare the same prompts against the formerly tested page size before
attributing those divergences to a particular backend.
The 64-page 16+16 rerun completed for all policies and matched all 16
generated positions. A 4096-page control of the B=8 cases retained the
4/30 minimum, so a smaller KV page is not the cause of those mismatches.
For B=8, S_IN=128, S_OUT=128, request 5 first diverged at generated token
index 4 for all three MPK policies; requests 1, 2, and 3 also diverged in
one or more policies. For B=8, S_IN=1024, S_OUT=1024, only request 6 in
prefill-only diverged early (index 4). The next check replays those prompts
as B=1 to distinguish batch effects from backend arithmetic on the same
prompt.
The B=1 replays also diverged at index 4 for all three MPK policies on both
prompts, so the mismatch is not confined to B=8. For the 128-token prompt,
B=8 and B=1 agreed within each policy for the first 30 positions; normal
and MPK each followed different paths. For the 1024-token prompt, B=8
normal itself differed from B=1 normal at index 4 while B=8 always and
decode-only matched their respective B=1 outputs. This batch-dependent
normal result needs a logit-margin probe before treating the 4/30 result
as an independent implementation error at every output position.
The index-4 probe found a BF16 tie between token IDs 279 and 773 in the
normal 128-token case (both 28.0), while always ranked 773 at 28.0 versus
279 at 27.875 and prefill-only ranked 773 at 28.125 versus 279 at 28.0.
Logit cosine similarities exceeded 0.9998. For the 1024-token case,
normal BF16 again tied 279 and 773 at 27.75; its fp32 candidate rescore
separated them by only 0.00038. Always and prefill-only ranked 773 above
279 by 0.125 BF16. These are early near-tie trajectory switches, not 26
independent erroneous token computations.
The decode-only probe exposed an additional deterministic argmax issue:
its saved BF16 logits had a different `torch.argmax` winner than the token
selected by MPK. The Ampere/Hopper and Blackwell argmax reductions now
choose the lowest vocabulary index when scores tie, matching PyTorch.
GPU validation is still needed; this change does not remove the separate
normal-versus-MPK logit differences in always or prefill-only.
The H100 rerun of the index-4 decode-only probes validated the tie change.
For both the 128- and 1024-token prompts, saved decode-only BF16 logits
tied IDs 279 and 773 exactly, and MPK selected 279, matching the normal
generated token and `torch.argmax`. The probe comparison now passes for
both cases. Full-length correctness across all policies still needs a new
run, since always and prefill-only retain separate arithmetic differences.
The B=8 full-output rerun after the argmax change produced policy-specific
results. At S_IN=S_OUT=128, all four MPK timing paths had at least one
request below the 20/30 positional gate; request 5 still had 4/30 matches.
At S_IN=S_OUT=1024, split always and decode-only matched all eight requests
at 30/30, while continuous always and prefill-only had request 6 at 4/30.
This confirms the tie fix, but also shows that split always cannot stand in
for continuous always in correctness reporting. The full three-axis sweep
must retain `correctness_failed` rows and exclude them from an MPK advantage
map even when their latency is favorable.

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
