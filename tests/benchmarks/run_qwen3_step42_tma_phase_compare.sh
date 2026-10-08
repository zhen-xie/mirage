#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step42_tma_phase_compare_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}; TIMEOUT=${TIMEOUT:-3600}
STEP39_SUMMARY=${STEP39_SUMMARY:-"$ROOT/results/qwen3_step39_attention_phase_profile_v1/summary.json"}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 FLASHINFER_USE_CUDA_NORM=1 PYTHONUNBUFFERED=1
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
python -m py_compile tests/benchmarks/qwen3_step37_attention_baseline.py \
 tests/benchmarks/qwen3_step39_attention_phase_profile.py \
 tests/benchmarks/qwen3_step42_tma_phase_compare.py || exit 1
python tests/benchmarks/qwen3_step37_attention_baseline.py \
 --model "$MODEL" --batch-sizes "8 32" --kv-lengths "1024" \
 --warmup 5 --repeat 20 --timeout "$TIMEOUT" --threshold 256 \
 --target-tasks 128 --attention-tma-kv --profile-attention-phases \
 --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
 --output-dir "$OUTDIR/candidate"
candidate_result=$?
python tests/benchmarks/qwen3_step39_attention_phase_profile.py \
 --candidate-dir "$OUTDIR/candidate" --output-dir "$OUTDIR/phases" \
 --batch-sizes "8 32" --kv-lengths "1024"
phase_result=$?
python tests/benchmarks/qwen3_step42_tma_phase_compare.py \
 --baseline "$STEP39_SUMMARY" --candidate "$OUTDIR/phases/summary.json" \
 --output-dir "$OUTDIR"
compare_result=$?
result=$candidate_result; [[ $phase_result -ne 0 ]] && result=$phase_result
[[ $compare_result -ne 0 ]] && result=$compare_result
printf 'Step 42 TMA phase comparison exited with code %s.\nSummary: %s\n' "$result" "$OUTDIR/summary.json"
exit "$result"
