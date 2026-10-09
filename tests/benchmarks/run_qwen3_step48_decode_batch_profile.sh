#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step48_decode_batch_profile_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
BATCH_SIZES=${BATCH_SIZES:-"4 5 6 7 8"}
STEP47_COMPARISON=${STEP47_COMPARISON:-"$ROOT/results/qwen3_step47_batch_1_8_v1/comparison_repeat1/comparison.json"}
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
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
  export CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"
  export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else
  export NVCC_PREPEND_FLAGS="--threads 8"
fi

read -r -a batches <<< "$BATCH_SIZES"
python -m py_compile \
  tests/benchmarks/qwen3_step48_decode_batch_profile.py \
  tests/benchmarks/qwen3_step15_mpk_window_profile.py \
  tests/benchmarks/qwen3_step18_mpk_concurrency.py \
  tests/benchmarks/summarize_qwen3_mpk_profile.py || exit 1

if [[ "$BUILD" == "1" ]]; then
  printf 'Building and installing Mirage...\n'
  timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
      result=$?; tail -n 120 "$OUTDIR/build.log"; exit "$result";
    }
fi

printf 'Running Step 48 decode-only B=4..8 transition profiles...\n'
python tests/benchmarks/qwen3_step48_decode_batch_profile.py \
  --model "$MODEL" --batch-sizes "${batches[@]}" \
  --step47-comparison "$STEP47_COMPARISON" \
  --timeout "$TIMEOUT" --threshold "$THRESHOLD" \
  --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
  --output-dir "$OUTDIR"
result=$?
printf 'Step 48 decode batch profile exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
