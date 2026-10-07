#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step24_split_kv_chunk_refinement_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
CHUNK_SIZES=${CHUNK_SIZES:-"128 256 320 640"}
TIMEOUT=${TIMEOUT:-3600}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

printf 'Running syntax checks...\n'
python -m py_compile \
    demo/qwen3/demo.py \
    python/mirage/mpk/persistent_kernel.py \
    tests/benchmarks/qwen3_step24_split_kv_chunk_refinement.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
        code=$?
        printf 'Build failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/build.log"
        exit "$code"
    }

printf 'Running Step 24 split-KV chunk-size refinement...\n'
python tests/benchmarks/qwen3_step24_split_kv_chunk_refinement.py \
    --model "$MODEL" --chunk-sizes $CHUNK_SIZES \
    --timeout "$TIMEOUT" --output-dir "$OUTDIR"
result=$?
printf 'Step 24 split-KV chunk refinement exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
