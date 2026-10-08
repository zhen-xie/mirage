#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step32_hybrid_handoff_v1"}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
BUDGET_US=${BUDGET_US:-54}
STEP31_SUMMARY=${STEP31_SUMMARY:-"$ROOT/results/qwen3_step31_hybrid_attention_feasibility_v1/summary.json"}

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
printf 'Running syntax and prerequisite checks...\n'
python -m py_compile \
    tests/benchmarks/qwen3_step32_hybrid_handoff.py || exit 1
test -f "$STEP31_SUMMARY" || {
    printf 'Missing Step 31 summary: %s\n' "$STEP31_SUMMARY"
    exit 1
}

printf 'Running Step 32 Hybrid stream-handoff lower-bound benchmark...\n'
python tests/benchmarks/qwen3_step32_hybrid_handoff.py \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --budget-us "$BUDGET_US" \
    --step31-summary "$STEP31_SUMMARY" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 32 Hybrid handoff exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
