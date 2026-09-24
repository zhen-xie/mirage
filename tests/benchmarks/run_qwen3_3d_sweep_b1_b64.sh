#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPEAT=${QWEN3_SWEEP_REPEAT:-1}
OUTDIR=${QWEN3_SWEEP_OUTDIR:-$ROOT/results/qwen3_3d_identical_b1_b64_repeat${REPEAT}}
EXIT_CODE_FILE="$OUTDIR/exit_code.txt"

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

if [[ ! "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'QWEN3_SWEEP_REPEAT must be a positive integer, got: %s\n' "$REPEAT"
    exit 2
fi

mkdir -p "$OUTDIR"
cd "$ROOT" || exit 1

printf 'Qwen3 sweep started at %s\n' "$(date --iso-8601=seconds)"
printf 'Repository root: %s\n' "$ROOT"
printf 'Output directory: %s\n' "$OUTDIR"
printf 'Repeats per policy: %s\n' "$REPEAT"
printf 'Process ID: %s\n' "$$"
printf '%s\n' "$$" > "$OUTDIR/worker.pid"
rm -f "$EXIT_CODE_FILE"

printf '\nRunning syntax checks...\n'
python -m py_compile \
    demo/qwen3/demo.py \
    tests/benchmarks/qwen3_decode_backend.py \
    tests/benchmarks/qwen3_decode_sweep.py
result=$?
if [[ "$result" -ne 0 ]]; then
    printf 'Syntax checks failed with exit code %s.\n' "$result"
    printf '%s\n' "$result" > "$EXIT_CODE_FILE"
    exit "$result"
fi

printf 'Syntax checks passed.\n'
printf '\nStarting 252-case sweep in B -> S_IN -> S_OUT order...\n'

python tests/benchmarks/qwen3_decode_sweep.py \
    --batch-sizes 1 2 4 8 16 32 64 \
    --s-in-values 16 32 64 128 256 512 \
    --s-out-values 16 32 64 128 256 512 \
    --batch-prompt-mode identical \
    --max-mpk-batch-size 64 \
    --split-kv-cache-min-batch-size 0 \
    --warmup 0 \
    --repeat "$REPEAT" \
    --timeout 3600 \
    --progress-interval-seconds 300 \
    --fail-on-failed-cases \
    --output-dir "$OUTDIR"

result=$?
printf '%s\n' "$result" > "$EXIT_CODE_FILE"
printf '\nQwen3 sweep finished at %s with exit code %s.\n' \
    "$(date --iso-8601=seconds)" "$result"
printf 'Results: %s\n' "$OUTDIR/raw_results.csv"
printf 'Progress: %s\n' "$OUTDIR/progress.json"

exit "$result"
