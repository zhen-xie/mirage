# MPK decode-only vs SGLang — partial 34-case comparison

## Scope

- Matched workloads: **34**.
- Covered batches: **[1, 4, 8, 16]**.
- The interrupted run did not cover the full planned B=32 and B=64 range, so this is a partial result.
- Mirage uses `normal` prefill plus MPK decode-only, with one in-process warmup and one recorded repeat.
- SGLang data uses its one-batch benchmark results.
- Ratio is `SGLang latency / MPK latency`; values above 1 mean MPK is faster.

## Overall results

| Phase | Cases | Median ratio | Minimum | Maximum | MPK wins | SGLang wins |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 34 | 0.712 | 0.303 | 0.903 | 0 | 34 |
| Decode | 34 | 0.994 | 0.910 | 1.048 | 14 | 20 |
| Prefill + decode | 34 | 0.969 | 0.850 | 1.016 | 6 | 28 |

The median decode ratio is 0.994, so MPK and SGLang decode latency are effectively tied over this partial set. The median end-to-end ratio is 0.969, corresponding to SGLang being about 3.1% faster at the median, mainly because its prefill is faster.

## Decode ratio by batch size

| Batch | Cases | Median SGLang/MPK | MPK wins | SGLang wins |
|---:|---:|---:|---:|---:|
| 1 | 9 | 0.974 | 0 | 9 |
| 16 | 7 | 0.990 | 2 | 5 |
| 4 | 9 | 1.005 | 5 | 4 |
| 8 | 9 | 1.013 | 7 | 2 |

## Decode ratio by input length

| S_IN | Cases | Median SGLang/MPK | MPK wins | SGLang wins |
|---:|---:|---:|---:|---:|
| 128 | 12 | 0.998 | 6 | 6 |
| 16 | 12 | 1.009 | 7 | 5 |
| 512 | 10 | 0.965 | 1 | 9 |

## Decode ratio by output length

| S_OUT | Cases | Median SGLang/MPK | MPK wins | SGLang wins |
|---:|---:|---:|---:|---:|
| 128 | 12 | 0.992 | 4 | 8 |
| 16 | 11 | 1.017 | 7 | 4 |
| 512 | 11 | 0.978 | 3 | 8 |

## Cases where MPK has the largest decode advantage

| B | S_IN | S_OUT | MPK decode ms | SGLang decode ms | Ratio |
|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 94.08 | 98.60 | 1.048 |
| 8 | 128 | 16 | 98.11 | 102.37 | 1.043 |
| 4 | 16 | 16 | 94.00 | 97.77 | 1.040 |
| 8 | 16 | 128 | 803.94 | 824.97 | 1.026 |
| 4 | 128 | 16 | 97.18 | 99.66 | 1.026 |

## Cases where SGLang has the largest decode advantage

| B | S_IN | S_OUT | MPK decode ms | SGLang decode ms | Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 512 | 3553.98 | 3233.21 | 0.910 |
| 1 | 512 | 128 | 859.34 | 803.03 | 0.934 |
| 4 | 512 | 512 | 3574.09 | 3379.64 | 0.946 |
| 1 | 128 | 512 | 3346.90 | 3218.17 | 0.962 |
| 16 | 512 | 128 | 923.99 | 888.77 | 0.962 |


## Interpretation limits

- This comparison has one recorded repeat, so it measures the observed run and does not provide confidence intervals.
- Six Mirage rows were marked `correctness_failed` because token agreement fell below the configured gate. They contained complete generations without invalid tokens and are retained for the requested performance comparison.
- The two systems use different software stacks. This is a practical system comparison on the same GPU class, rather than an isolated kernel-only comparison.
