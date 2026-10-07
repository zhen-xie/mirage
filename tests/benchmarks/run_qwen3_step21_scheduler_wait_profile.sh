#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step21_scheduler_wait_profile_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
CACHE="$OUTDIR/cache"

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR" "$CACHE"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

printf 'Running syntax checks...\n'
python -m py_compile \
    demo/qwen3/demo.py \
    python/mirage/mpk/persistent_kernel.py \
    python/mirage/mpk/profiler_persistent.py \
    tests/benchmarks/qwen3_step21_scheduler_wait_profile.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
        code=$?
        printf 'Build failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/build.log"
        exit "$code"
    }

printf 'Running Torch correctness reference...\n'
timeout "$TIMEOUT" python demo/qwen3/demo.py \
    --model "$MODEL" --input-length 1024 --max-seq-length 1280 \
    --max-new-tokens 128 --page-size 1280 --max-num-pages 1 \
    --max-num-batched-requests 1 --max-num-batched-tokens 8 \
    --ignore-eos --save-tokens "$OUTDIR/torch.json" \
    > "$OUTDIR/torch.log" 2>&1 || exit $?

printf 'Running profiled MPK decode window...\n'
timeout "$TIMEOUT" python demo/qwen3/demo.py \
    --model "$MODEL" --input-length 1024 --max-seq-length 1280 \
    --max-new-tokens 128 --page-size 1280 --max-num-pages 32 \
    --max-num-batched-requests 32 --max-num-batched-tokens 32 \
    --use-mirage --mpk-policy decode-only \
    --mpk-attention split-kv --mpk-split-kv-chunk-size 256 \
    --mpk-scheduler-policy round-robin \
    --mpk-kernel-cache-dir "$CACHE" \
    --normal-prefill-attention sdpa --prefill-warmup-runs 1 \
    --normal-prefill-cuda-graph --ignore-eos \
    --profiling --profile-scheduler-waits \
    --profiler-buffer-entries-per-block 65536 \
    --profiler-decode-start-step 60 --profiler-decode-num-steps 9 \
    --trace-name "$OUTDIR/mpk_profile" \
    --save-tokens "$OUTDIR/mpk.json" \
    > "$OUTDIR/mpk.log" 2>&1 || {
        code=$?
        printf 'Profiled MPK run failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/mpk.log"
        exit "$code"
    }

printf 'Summarizing scheduler waits...\n'
python tests/benchmarks/qwen3_step21_scheduler_wait_profile.py \
    --torch "$OUTDIR/torch.json" --mpk "$OUTDIR/mpk.json" \
    --profile "$OUTDIR/mpk_profile.csv" --output "$OUTDIR/summary.json"
result=$?
printf 'Step 21 scheduler wait profile exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
