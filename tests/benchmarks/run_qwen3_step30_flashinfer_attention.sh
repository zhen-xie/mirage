#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step30_flashinfer_attention_v1"}
WARMUP=${WARMUP:-20}
REPEAT=${REPEAT:-100}
STEP27_SUMMARY=${STEP27_SUMMARY:-"$ROOT/results/qwen3_step27_attention_length_scaling_v1/summary.json"}
STEP26_DIR=${STEP26_DIR:-"$ROOT/results/qwen3_step26_workload_aware_profile_v1"}

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
printf 'Running syntax and FlashInfer API checks...\n'
python -m py_compile \
    tests/benchmarks/qwen3_step30_flashinfer_attention.py || exit 1
python - <<'PY' || exit 1
import flashinfer
import torch
assert hasattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper")
print("FlashInfer:", getattr(flashinfer, "__version__", "unknown"))
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

printf 'Running Step 30 FlashInfer attention microbenchmark...\n'
python tests/benchmarks/qwen3_step30_flashinfer_attention.py \
    --warmup "$WARMUP" --repeat "$REPEAT" \
    --step27-summary "$STEP27_SUMMARY" --step26-dir "$STEP26_DIR" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 30 FlashInfer attention exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
