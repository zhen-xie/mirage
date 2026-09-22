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
The correctness gate remains the first 30 tokens, not full-generation equality.

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
