#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step11_steady_prefill_profile"}
PROFILE_DIR="$OUTDIR/raw"
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
STEP8_COMPARISON=${STEP8_COMPARISON:-"$ROOT/results/qwen3_step8_comparison/comparison.json"}
BUILD_LOG="$OUTDIR/build.log"

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR" "$PROFILE_DIR"

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_TOOLKIT"
    export CUDA_PATH="$CUDA_TOOLKIT"
    export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
    export PATH="$CUDA_TOOLKIT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf 'CUDA compiler: %s\n' "$(command -v nvcc)"
nvcc --version | tail -n 1
printf 'Running syntax checks...\n'
python -m py_compile \
    demo/qwen3/demo.py \
    demo/qwen3/models/modeling_qwen3.py \
    tests/benchmarks/qwen3_step7_model_smoke.py \
    tests/benchmarks/qwen3_step9_prefill_profile.py || exit 1

if [[ ! -f "$STEP8_COMPARISON" ]]; then
    printf 'Missing Step 8 comparison: %s\n' "$STEP8_COMPARISON"
    exit 1
fi

printf 'Building and installing Mirage...\n'
if timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$BUILD_LOG" 2>&1
then
    printf 'Build completed.\n'
else
    result=$?
    printf 'Build failed with exit code %s. Last 100 lines:\n' "$result"
    tail -n 100 "$BUILD_LOG"
    exit "$result"
fi

read -r -a models <<< "$MODELS"
printf 'Running steady-state prefill profiling with in-process warmup=1...\n'
python tests/benchmarks/qwen3_step7_model_smoke.py \
    --models "${models[@]}" \
    --warmup-runs 0 \
    --prefill-warmup-runs 1 \
    --profile-prefill-stages \
    --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" \
    --output-dir "$PROFILE_DIR"
result=$?
if [[ "$result" -ne 0 ]]; then
    printf 'Steady-state profile execution failed with exit code %s.\n' "$result"
    exit "$result"
fi

printf 'Summarizing steady-state prefill stages...\n'
python tests/benchmarks/qwen3_step9_prefill_profile.py \
    --step 11 \
    --mpk-summary "$PROFILE_DIR/summary.json" \
    --step8-comparison "$STEP8_COMPARISON" \
    --output-dir "$OUTDIR"
result=$?

printf 'Step 11 runner exited with code %s.\n' "$result"
printf 'Summary JSON: %s\n' "$OUTDIR/prefill_profile.json"
printf 'Summary CSV: %s\n' "$OUTDIR/prefill_profile.csv"
exit "$result"
