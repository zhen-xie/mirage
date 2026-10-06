#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step5_adaptive_attention"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
BUILD_LOG="$OUTDIR/build.log"

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"

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
if ! python -m py_compile \
    demo/qwen3/demo.py \
    demo/qwen3/models/modeling_qwen3.py \
    python/mirage/mpk/persistent_kernel.py \
    tests/benchmarks/qwen3_step5_adaptive_attention.py
then
    printf 'Syntax checks failed.\n'
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

printf 'Running Step 5 adaptive attention validation...\n'
python tests/benchmarks/qwen3_step5_adaptive_attention.py \
    --model "$MODEL" \
    --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" \
    --output-dir "$OUTDIR"
result=$?

printf 'Step 5 runner exited with code %s.\n' "$result"
printf 'Summary JSON: %s\n' "$OUTDIR/summary.json"
printf 'Summary CSV: %s\n' "$OUTDIR/summary.csv"
exit "$result"
