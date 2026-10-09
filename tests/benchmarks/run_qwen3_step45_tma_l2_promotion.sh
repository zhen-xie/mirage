#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step45_tma_l2_promotion_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}; TIMEOUT=${TIMEOUT:-3600}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}; BUILD=${BUILD:-1}
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONUNBUFFERED=1
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT" CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH" LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOST_CXX=${CUDAHOSTCXX:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}}
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
 export CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"; export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else export NVCC_PREPEND_FLAGS="--threads 8"; fi
python -m py_compile demo/qwen3/demo.py python/mirage/mpk/persistent_kernel.py \
 tests/benchmarks/qwen3_step37_attention_baseline.py \
 tests/benchmarks/qwen3_step39_attention_phase_profile.py \
 tests/benchmarks/qwen3_step45_tma_l2_promotion.py || exit 1
if [[ "$BUILD" == "1" ]]; then
 timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
  > "$OUTDIR/build.log" 2>&1 || { result=$?; tail -n 120 "$OUTDIR/build.log"; exit "$result"; }
fi
result=0
for mode in baseline promoted; do
 extra=(); [[ "$mode" == promoted ]] && extra+=(--attention-tma-l2-promotion)
 printf 'Measuring unprofiled mode=%s...\n' "$mode"
 python tests/benchmarks/qwen3_step37_attention_baseline.py \
  --model "$MODEL" --batch-sizes "8 32" --kv-lengths "1024" --s-out 128 \
  --warmup 1 --repeat 1 --timeout "$TIMEOUT" --threshold 256 --target-tasks 128 \
  --attention-tma-kv --attention-kv-pipeline-stages 2 --skip-flashinfer --skip-mpk-profile \
  "${extra[@]}" --output-dir "$OUTDIR/${mode}_perf"
 rc=$?; [[ $rc -ne 0 ]] && result=$rc
 printf 'Profiling mode=%s...\n' "$mode"
 python tests/benchmarks/qwen3_step37_attention_baseline.py \
  --model "$MODEL" --batch-sizes "8 32" --kv-lengths "1024" --s-out 10 \
  --warmup 1 --repeat 1 --timeout "$TIMEOUT" --threshold 256 --target-tasks 128 \
  --attention-tma-kv --attention-kv-pipeline-stages 2 --skip-flashinfer \
  --profile-attention-phases --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
  "${extra[@]}" --output-dir "$OUTDIR/${mode}_profile"
 rc=$?; [[ $rc -ne 0 ]] && result=$rc
 python tests/benchmarks/qwen3_step39_attention_phase_profile.py \
  --candidate-dir "$OUTDIR/${mode}_profile" --output-dir "$OUTDIR/${mode}_phases" \
  --batch-sizes "8 32" --kv-lengths "1024"
 rc=$?; [[ $rc -ne 0 ]] && result=$rc
done
python tests/benchmarks/qwen3_step45_tma_l2_promotion.py \
 --baseline "$OUTDIR/baseline_perf/summary.json" --promoted "$OUTDIR/promoted_perf/summary.json" \
 --baseline-profile "$OUTDIR/baseline_profile/summary.json" \
 --promoted-profile "$OUTDIR/promoted_profile/summary.json" \
 --baseline-phases "$OUTDIR/baseline_phases/summary.json" \
 --promoted-phases "$OUTDIR/promoted_phases/summary.json" --output-dir "$OUTDIR"
rc=$?; [[ $rc -ne 0 ]] && result=$rc
printf 'Step 45 TMA L2 promotion exited with code %s.\nSummary: %s\n' "$result" "$OUTDIR/summary.json"
exit "$result"
