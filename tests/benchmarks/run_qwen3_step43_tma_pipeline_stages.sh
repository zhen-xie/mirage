#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step43_tma_pipeline_stages_v1"}
MODEL=${MODEL:-Qwen/Qwen3-8B}; TIMEOUT=${TIMEOUT:-3600}
PROFILER_ENTRIES_PER_BLOCK=${PROFILER_ENTRIES_PER_BLOCK:-32768}
BUILD=${BUILD:-0}
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
 tests/benchmarks/qwen3_step43_tma_pipeline_stages.py || exit 1
if [[ "$BUILD" == "1" ]]; then
 timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
  > "$OUTDIR/build.log" 2>&1 || { result=$?; tail -n 100 "$OUTDIR/build.log"; exit "$result"; }
else
 printf 'Reusing the installed editable Mirage build (BUILD=0).\n'
fi

result=0
for stages in 2 3; do
 printf 'Running TMA attention with pipeline stages=%s...\n' "$stages"
 python tests/benchmarks/qwen3_step37_attention_baseline.py \
  --model "$MODEL" --batch-sizes "8 32" --kv-lengths "1024" \
  --warmup 1 --repeat 1 --s-out 128 --timeout "$TIMEOUT" --threshold 256 \
  --target-tasks 128 --attention-tma-kv --profile-attention-phases \
  --attention-kv-pipeline-stages "$stages" --skip-flashinfer \
  --profiler-entries-per-block "$PROFILER_ENTRIES_PER_BLOCK" \
  --output-dir "$OUTDIR/stage${stages}"
 stage_result=$?; [[ $stage_result -ne 0 ]] && result=$stage_result
 python tests/benchmarks/qwen3_step39_attention_phase_profile.py \
  --candidate-dir "$OUTDIR/stage${stages}" \
  --output-dir "$OUTDIR/stage${stages}_phases" \
  --batch-sizes "8 32" --kv-lengths "1024"
 phase_result=$?; [[ $phase_result -ne 0 ]] && result=$phase_result
done

if [[ -f "$OUTDIR/stage2/summary.json" && -f "$OUTDIR/stage3/summary.json" \
   && -f "$OUTDIR/stage2_phases/summary.json" && -f "$OUTDIR/stage3_phases/summary.json" ]]; then
 python tests/benchmarks/qwen3_step43_tma_pipeline_stages.py \
  --stage2 "$OUTDIR/stage2/summary.json" \
  --stage3 "$OUTDIR/stage3/summary.json" \
  --stage2-phases "$OUTDIR/stage2_phases/summary.json" \
  --stage3-phases "$OUTDIR/stage3_phases/summary.json" \
  --output-dir "$OUTDIR"
 compare_result=$?; [[ $compare_result -ne 0 ]] && result=$compare_result
else
 printf 'Step 43 comparison skipped because one or more summaries are missing.\n'
 result=1
fi
printf 'Step 43 TMA pipeline stages exited with code %s.\nSummary: %s\n' "$result" "$OUTDIR/summary.json"
exit "$result"
