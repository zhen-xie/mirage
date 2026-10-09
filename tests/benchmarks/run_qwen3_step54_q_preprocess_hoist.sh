#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step54_q_preprocess_hoist_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
BUILD=${BUILD:-1}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_USE_CUDA_NORM=1
export PYTHONUNBUFFERED=1

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
    python/mirage/mpk/persistent_kernel.py \
    tests/benchmarks/qwen3_step54_q_preprocess_hoist.py || exit 1

if [[ "$BUILD" == "1" ]]; then
    printf 'Building and installing Mirage...\n'
    if [[ -f build/CMakeCache.txt ]]; then
        mv build/CMakeCache.txt "$OUTDIR/CMakeCache.before-step54.txt"
    fi
    HOST_CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    HOST_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    if [[ ! -x "$HOST_CC" || ! -x "$HOST_CXX" ]]; then
        printf 'Missing Conda host compiler: %s or %s\n' "$HOST_CC" "$HOST_CXX"
        exit 1
    fi
    timeout "$TIMEOUT" env \
        CC="$HOST_CC" CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX" \
        python -m pip install -e . -v --no-build-isolation \
        > "$OUTDIR/build.log" 2>&1
    build_status=$?
    if [[ "$build_status" -ne 0 ]]; then
        printf 'Build failed with exit code %s.\n' "$build_status"
        tail -n 100 "$OUTDIR/build.log"
        exit "$build_status"
    fi
else
    printf 'Skipping editable build (BUILD=%s).\n' "$BUILD"
fi

if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]]; then
    HOST_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else
    export NVCC_PREPEND_FLAGS="--threads 8"
fi

printf 'Running Step 54 split-KV Q preprocess hoist ablation...\n'
python tests/benchmarks/qwen3_step54_q_preprocess_hoist.py \
    --model "$MODEL" --timeout "$TIMEOUT" --output-dir "$OUTDIR"
result=$?
printf 'Step 54 runner exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
