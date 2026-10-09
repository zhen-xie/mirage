# MPK vs SGLang, batch size 1

End-to-end decode-latency comparison on one GPU. Reports prefill (TTFT),
per-token decode latency (TPOT) and decode throughput, median over N reps.

## Run

```bash
cd $MIRAGE_HOME
# lock clocks first so reps are comparable
sudo nvidia-smi -i 0 -lgc $(nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits)

python benchmark/mpk_vs_sglang/run_bench.py \
    --model Qwen/Qwen3-8B \
    --input-len 128 --output-len 256 \
    --reps 5 --warmup 1 --gpu 0 \
    --include-torch
```

The driver itself needs no GPU libs — only `conda` on PATH. It shells into
`mirage-mpk` and `sglang-bench` with `conda run`.

Results land in `outputs/bench_<timestamp>/`: `summary.md` (table) and
`raw.json` (every rep).

## What each side runs

| | command | timing source |
|---|---|---|
| MPK | `demo/qwen3/demo.py --use-mirage --mpk-policy decode-only` | CUDA events inside the demo; prefill = Torch prefill, decode = one megakernel launch |
| Torch (optional) | same demo without `--use-mirage` | Triton + FlashInfer kernels |
| SGLang | `sgl_bench.py`, offline `sgl.Engine` with `stream=True` | host timestamps per streamed chunk |

## Fairness notes — read before quoting numbers

1. **`--mpk-policy decode-only` is the honest setting.** The default
   (`always`) runs prefill inside the megakernel too and prints only a
   single total latency, so prefill and decode can't be separated. With
   `decode-only`, MPK's prefill is plain Torch — i.e. the prefill column
   is *not* an MPK result, and only the decode column is the real
   comparison.
2. **Timing bases differ.** MPK decode is measured with CUDA events around
   one kernel launch; SGLang decode is measured from host-side stream
   chunks, which carries a little scheduler/IPC overhead per token. That
   overhead is part of what a real SGLang user pays, but if you want a
   kernel-only number, re-run SGLang under `sglang.bench_one_batch` and
   compare against that instead.
3. **Prompts are synthetic and length-matched, not token-matched.**
   `--input-length` in the demo builds its own synthetic prompt; `sgl_bench.py`
   repeats a single filler token. Same length, different ids. Fine for
   latency, useless for output comparison.
4. **`--ignore-eos` on both sides** so each run decodes exactly
   `output_len` steps.
5. **Greedy on both sides** (temperature 0, MPK sampling off by default).
   Turning on MPK sampling adds its argmax/topk tasks to the megakernel and
   changes the decode cost.
6. **CUDA graphs are on for SGLang by default** — that is SGLang's fair
   best for bs=1. `--sgl-no-cuda-graph` shows how much of its decode cost
   is launch overhead, which is roughly the overhead MPK sets out to erase.
7. **Kernel compile time is excluded** for MPK via `--mpk-kernel-cache-dir`
   plus warmup reps. The first run populates the cache.
8. **Radix cache is disabled** on the SGLang side so repeated identical
   prompts don't skip prefill.
9. Both sides run bf16 on one GPU, `CUDA_VISIBLE_DEVICES` pinned.

## Sweeping

MPK's advantage is largest when the decode step is launch/latency-bound, so
sweep context length rather than just repeating one point:

```bash
for n in 128 512 2048 8192; do
  python benchmark/mpk_vs_sglang/run_bench.py --input-len $n --output-len 128 \
      --outdir outputs/sweep_in$n
done
```

If `--input-len` is large, raise MPK's KV budget (`--max-num-pages` /
`--kv-budget`) in `run_bench.py`'s MPK command — the demo defaults to 16
pages of 4096 tokens.
