#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step14_mpk"}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
CASES=${CASES:-"short:128:128 long_context:1024:128 long_generation:128:1024"}
BATCH_SIZES=${BATCH_SIZES:-"1 8 32"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}

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
python -m py_compile demo/qwen3/demo.py tests/benchmarks/qwen3_step14_mpk_sweep.py || exit 1
printf 'Building and installing Mirage...\n'
if timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation > "$OUTDIR/build.log" 2>&1; then
    :
else
    result=$?
    printf 'Build failed with exit code %s.\n' "$result"
    tail -n 100 "$OUTDIR/build.log"
    exit "$result"
fi
printf 'Build completed.\n'
read -r -a models <<< "$MODELS"
read -r -a cases <<< "$CASES"
read -r -a batches <<< "$BATCH_SIZES"
python tests/benchmarks/qwen3_step14_mpk_sweep.py \
    --models "${models[@]}" --cases "${cases[@]}" \
    --batch-sizes "${batches[@]}" --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" --output-dir "$OUTDIR"
result=$?
printf 'Step 14 MPK sweep exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
