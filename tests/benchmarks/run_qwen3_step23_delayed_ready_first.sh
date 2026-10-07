#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step23_delayed_ready_first_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
SPIN_ITERS=${SPIN_ITERS:-64}
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
    tests/benchmarks/qwen3_step23_delayed_ready_first.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
        code=$?
        printf 'Build failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/build.log"
        exit "$code"
    }

printf 'Running Step 23 delayed ready-first ablation...\n'
python tests/benchmarks/qwen3_step23_delayed_ready_first.py \
    --model "$MODEL" --spin-iters "$SPIN_ITERS" \
    --timeout "$TIMEOUT" --output-dir "$OUTDIR"
result=$?
printf 'Step 23 delayed ready-first ablation exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
