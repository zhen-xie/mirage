#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step35_full_hybrid_decode_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}
BATCH_SIZE=${BATCH_SIZE:-8}
S_IN=${S_IN:-128}
S_OUT=${S_OUT:-10}
TIMEOUT=${TIMEOUT:-7200}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_USE_CUDA_NORM=1
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
    tests/benchmarks/qwen3_step35_full_hybrid_decode.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
        result=$?
        printf 'Build failed with exit code %s.\n' "$result"
        tail -n 100 "$OUTDIR/build.log"
        exit "$result"
    }

printf 'Running Step 35 full-model Hybrid decode...\n'
timeout "$TIMEOUT" python \
    tests/benchmarks/qwen3_step35_full_hybrid_decode.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --input-length "$S_IN" \
    --output-length "$S_OUT" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 35 full-model Hybrid exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
