#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step52_b1_attention_internal_profile_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}
BUILD=${BUILD:-0}

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
    tests/benchmarks/qwen3_step37_attention_baseline.py \
    tests/benchmarks/qwen3_step52_b1_attention_internal_profile.py || exit 1

if [[ "$BUILD" == "1" ]]; then
    printf 'Building and installing Mirage...\n'
    # setup.py passes the Z3 include/library paths on each clean configure.
    # A compiler change makes CMake delete its cache and internally rerun
    # without those command-line paths, so preserve the stale cache outside
    # the build tree and start one clean configure with compiler overrides
    # removed from the long-lived interactive shell.
    if [[ -f build/CMakeCache.txt ]]; then
        mv build/CMakeCache.txt "$OUTDIR/CMakeCache.before-step52.txt"
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
        tail -n 120 "$OUTDIR/build.log"
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

candidate="$OUTDIR/candidate"
printf 'Running instrumented B=1 attention case...\n'
python tests/benchmarks/qwen3_step37_attention_baseline.py \
    --model "$MODEL" \
    --batch-sizes "1" \
    --kv-lengths "1024" \
    --warmup 1 --repeat 1 --timeout "$TIMEOUT" \
    --threshold 256 --target-tasks 128 \
    --attention-tma-kv --profile-attention-phases \
    --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
    --skip-flashinfer --output-dir "$candidate"
candidate_status=$?
if [[ "$candidate_status" -ne 0 ]]; then
    printf 'Instrumented attention case failed with code %s.\n' "$candidate_status"
    exit "$candidate_status"
fi

printf 'Summarizing detailed attention phases...\n'
python tests/benchmarks/qwen3_step52_b1_attention_internal_profile.py \
    --candidate-dir "$candidate" --output-dir "$OUTDIR"
result=$?
printf 'Step 52 runner exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
