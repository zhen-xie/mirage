#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step10_flashinfer_prefill"}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
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
python -m py_compile \
    demo/qwen3/demo.py \
    demo/qwen3/models/modeling_qwen3.py \
    tests/benchmarks/qwen3_step10_flashinfer_prefill.py || exit 1

printf 'Checking FlashInfer prefill API...\n'
python - <<'PY' || exit 1
import flashinfer
import torch
assert hasattr(flashinfer, "single_prefill_with_kv_cache")
print("FlashInfer:", getattr(flashinfer, "__version__", "unknown"))
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

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
printf 'Running Step 10 FlashInfer prefill attention ablation...\n'
python tests/benchmarks/qwen3_step10_flashinfer_prefill.py \
    --models "${models[@]}" \
    --timeout "$TIMEOUT" \
    --threshold "$THRESHOLD" \
    --output-dir "$OUTDIR"
result=$?

printf 'Step 10 runner exited with code %s.\n' "$result"
printf 'Summary JSON: %s\n' "$OUTDIR/summary.json"
printf 'Summary CSV: %s\n' "$OUTDIR/summary.csv"
exit "$result"
