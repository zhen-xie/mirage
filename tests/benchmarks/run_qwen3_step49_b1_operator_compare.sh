#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step49_b1_operator_compare_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
STEP47_COMPARISON=${STEP47_COMPARISON:-"$ROOT/results/qwen3_step47_batch_1_8_v1/comparison_repeat1/comparison.json"}
STEP47_SGLANG_DIR=${STEP47_SGLANG_DIR:-"$ROOT/results/qwen3_step47_batch_1_8_v1/sglang_repeat1"}
SGLANG_ENV=${SGLANG_ENV:-sglang-bench}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}
BUILD=${BUILD:-1}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONUNBUFFERED=1
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT"
export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOST_CXX=${CUDAHOSTCXX:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}}
HOST_CC=${CC:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}}
HOST_BIN=${CONDA_PREFIX:+$CONDA_PREFIX/bin}
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
  export CC="$HOST_CC" CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"
  export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else
  export NVCC_PREPEND_FLAGS="--threads 8"
fi

python -m py_compile \
  tests/benchmarks/qwen3_step48_decode_batch_profile.py \
  tests/benchmarks/qwen3_step49_b1_operator_compare.py \
  tests/benchmarks/qwen3_step16_sglang_window_profile.py \
  tests/benchmarks/summarize_qwen3_sglang_profile.py || exit 1

if [[ "$BUILD" == "1" ]]; then
  printf 'Building and installing Mirage...\n'
  timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
      result=$?; tail -n 120 "$OUTDIR/build.log"; exit "$result";
    }
fi

printf 'Capturing MPK B=1 decode windows...\n'
python tests/benchmarks/qwen3_step48_decode_batch_profile.py \
  --model "$MODEL" --batch-sizes 1 \
  --step47-comparison "$STEP47_COMPARISON" \
  --timeout "$TIMEOUT" --threshold "$THRESHOLD" \
  --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
  --output-dir "$OUTDIR/mpk"
mpk_result=$?

printf 'Capturing SGLang B=1 decode windows...\n'
SGLANG_PREFIX=$(conda run -n "$SGLANG_ENV" python -c 'import sys; print(sys.prefix)')
conda run --no-capture-output -n "$SGLANG_ENV" \
  env \
  PATH="$SGLANG_PREFIX/bin:$HOST_BIN:$CUDA_TOOLKIT/bin:$PATH" \
  CC="$HOST_CC" CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX" \
  NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8" \
  python tests/benchmarks/qwen3_step16_sglang_window_profile.py \
  --model "$MODEL" --step14-sglang-dir "$STEP47_SGLANG_DIR" \
  --cases long_context_b1 --windows early middle late \
  --timeout "$TIMEOUT" --output-dir "$OUTDIR/sglang"
sglang_result=$?

printf 'Comparing absolute operator times...\n'
python tests/benchmarks/qwen3_step49_b1_operator_compare.py \
  --mpk-summary "$OUTDIR/mpk/summary.json" \
  --sglang-summary "$OUTDIR/sglang/summary.json" \
  --output-dir "$OUTDIR/comparison"
compare_result=$?

printf 'MPK capture exit code: %s\n' "$mpk_result"
printf 'SGLang capture exit code: %s\n' "$sglang_result"
printf 'Step 49 comparison exit code: %s\n' "$compare_result"
printf 'Comparison: %s\n' "$OUTDIR/comparison/comparison.json"
exit "$compare_result"
