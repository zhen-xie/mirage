#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step41_workload_aware_tma_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}; TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}; TARGET_TASKS=${TARGET_TASKS:-128}
BUILD=${BUILD:-0}
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONUNBUFFERED=1
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT" CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH" LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOST_CXX=${CUDAHOSTCXX:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}}
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
 export CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"
 export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else
 export NVCC_PREPEND_FLAGS="--threads 8"
fi
python -m py_compile demo/qwen3/demo.py python/mirage/mpk/persistent_kernel.py \
 tests/benchmarks/qwen3_step41_workload_aware_tma.py || exit 1
if [[ "$BUILD" == "1" ]]; then
 timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
  > "$OUTDIR/build.log" 2>&1 || { result=$?; tail -n 100 "$OUTDIR/build.log"; exit "$result"; }
else
 printf 'Reusing the installed editable Mirage build (BUILD=0).\n'
fi
python tests/benchmarks/qwen3_step41_workload_aware_tma.py \
 --model "$MODEL" --timeout "$TIMEOUT" --threshold "$THRESHOLD" \
 --target-tasks "$TARGET_TASKS" --output-dir "$OUTDIR"
result=$?
printf 'Step 41 workload-aware TMA exited with code %s.\nSummary: %s\n' "$result" "$OUTDIR/summary.json"
exit "$result"
