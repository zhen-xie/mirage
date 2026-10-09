#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step47_batch_1_8"}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
CASES=${CASES:-"short:128:128 long_context:1024:128 long_generation:128:1024"}
BATCH_SIZES=${BATCH_SIZES:-"1 2 3 4 5 6 7 8"}
REPEATS=${REPEATS:-3}; TIMEOUT=${TIMEOUT:-3600}; THRESHOLD=${THRESHOLD:-256}
SGLANG_ENV=${SGLANG_ENV:-sglang-bench}; BUILD=${BUILD:-1}
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONUNBUFFERED=1
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT" CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH" LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOST_CXX=${CUDAHOSTCXX:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}}
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
 export CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"; export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else export NVCC_PREPEND_FLAGS="--threads 8"; fi
python -m py_compile tests/benchmarks/qwen3_step14_mpk_sweep.py \
 tests/benchmarks/qwen3_step47_batch_1_8_compare.py || exit 1
if [[ "$BUILD" == "1" ]]; then
 timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
  > "$OUTDIR/build.log" 2>&1 || { rc=$?; tail -n 120 "$OUTDIR/build.log"; exit "$rc"; }
fi
read -r -a models <<< "$MODELS"; read -r -a cases <<< "$CASES"; read -r -a batches <<< "$BATCH_SIZES"
result=0
for repeat in $(seq 1 "$REPEATS"); do
 printf '\n=== MPK repeat %s/%s ===\n' "$repeat" "$REPEATS"
 python tests/benchmarks/qwen3_step14_mpk_sweep.py \
  --models "${models[@]}" --cases "${cases[@]}" --batch-sizes "${batches[@]}" \
  --timeout "$TIMEOUT" --threshold "$THRESHOLD" --output-dir "$OUTDIR/mpk_repeat${repeat}"
 rc=$?; [[ $rc -ne 0 ]] && result=$rc
done
for repeat in $(seq 1 "$REPEATS"); do
 printf '\n=== SGLang repeat %s/%s ===\n' "$repeat" "$REPEATS"
 OUTDIR="$OUTDIR/sglang_repeat${repeat}" MODELS="$MODELS" CASES="$CASES" \
 BATCH_SIZES="$BATCH_SIZES" TIMEOUT="$TIMEOUT" CONDA_ENV="$SGLANG_ENV" \
 bash tests/benchmarks/run_qwen3_step14_sglang_sweep.sh
 rc=$?; [[ $rc -ne 0 ]] && result=$rc
done
python tests/benchmarks/qwen3_step47_batch_1_8_compare.py \
 --root-dir "$OUTDIR" --repeats "$REPEATS" --output-dir "$OUTDIR/comparison"
rc=$?; [[ $rc -ne 0 ]] && result=$rc
printf 'Step 47 batch 1-8 comparison exited with code %s.\nComparison: %s\n' \
 "$result" "$OUTDIR/comparison/comparison.json"
exit "$result"
