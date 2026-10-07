#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step15_mpk_operator_profile_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
STEP14_SUMMARY=${STEP14_SUMMARY:-"$ROOT/results/qwen3_step14_mpk/summary.json"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}
CASES=${CASES:-"short_b1 short_b32 long_context_b32"}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT"
    export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
    export PATH="$CUDA_TOOLKIT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf 'CUDA compiler: %s\n' "$(command -v nvcc)"
nvcc --version | tail -n 1
printf 'Running syntax checks...\n'
python -m py_compile \
    demo/qwen3/demo.py \
    python/mirage/mpk/persistent_kernel.py \
    python/mirage/mpk/profiler_persistent.py \
    tests/benchmarks/qwen3_step15_mpk_operator_profile.py \
    tests/benchmarks/summarize_qwen3_mpk_profile.py || exit 1

printf 'Building and installing Mirage...\n'
if timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1; then
    printf 'Build completed.\n'
else
    result=$?
    printf 'Build failed with exit code %s.\n' "$result"
    tail -n 100 "$OUTDIR/build.log"
    exit "$result"
fi

printf 'Running Step 15 MPK operator profiles...\n'
read -r -a cases <<< "$CASES"
python tests/benchmarks/qwen3_step15_mpk_operator_profile.py \
    --model "$MODEL" \
    --step14-summary "$STEP14_SUMMARY" \
    --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" \
    --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
    --cases "${cases[@]}" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 15 MPK operator profile runner exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
