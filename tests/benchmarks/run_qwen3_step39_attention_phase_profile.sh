#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step39_attention_phase_profile_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}
BATCH_SIZES=${BATCH_SIZES:-"8 32"}
KV_LENGTHS=${KV_LENGTHS:-"128 1024"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
TARGET_TASKS=${TARGET_TASKS:-128}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_USE_CUDA_NORM=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT" CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Running syntax checks...\n'
python -m py_compile demo/qwen3/demo.py python/mirage/mpk/persistent_kernel.py \
  tests/benchmarks/qwen3_step37_attention_baseline.py \
  tests/benchmarks/qwen3_step39_attention_phase_profile.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
  > "$OUTDIR/build.log" 2>&1 || exit $?

printf 'Running instrumented attention cases...\n'
python tests/benchmarks/qwen3_step37_attention_baseline.py \
  --model "$MODEL" --batch-sizes "$BATCH_SIZES" --kv-lengths "$KV_LENGTHS" \
  --warmup 5 --repeat 20 --timeout "$TIMEOUT" --threshold "$THRESHOLD" \
  --target-tasks "$TARGET_TASKS" --profile-attention-phases \
  --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
  --output-dir "$OUTDIR/candidate"
candidate_result=$?

python tests/benchmarks/qwen3_step39_attention_phase_profile.py \
  --candidate-dir "$OUTDIR/candidate" --output-dir "$OUTDIR" \
  --batch-sizes "$BATCH_SIZES" --kv-lengths "$KV_LENGTHS"
summary_result=$?
result=$candidate_result
if [[ "$summary_result" -ne 0 ]]; then result=$summary_result; fi
printf 'Step 39 attention phase profile exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
