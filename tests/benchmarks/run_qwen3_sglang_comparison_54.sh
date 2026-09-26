#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WARMUP=${QWEN3_COMPARISON_WARMUP:-1}
REPEAT=${QWEN3_COMPARISON_REPEAT:-1}
OUTDIR=${QWEN3_COMPARISON_OUTDIR:-$ROOT/results/qwen3_sglang_comparison_54_warmup${WARMUP}_repeat${REPEAT}}
EXIT_CODE_FILE="$OUTDIR/exit_code.txt"

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

if [[ ! "$WARMUP" =~ ^[0-9]+$ ]]; then
    printf 'QWEN3_COMPARISON_WARMUP must be a nonnegative integer, got: %s\n' "$WARMUP"
    exit 2
fi
if [[ ! "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    printf 'QWEN3_COMPARISON_REPEAT must be a positive integer, got: %s\n' "$REPEAT"
    exit 2
fi

mkdir -p "$OUTDIR"
cd "$ROOT" || exit 1

printf 'Qwen3 SGLang comparison sweep started at %s\n' "$(date --iso-8601=seconds)"
printf 'Repository root: %s\n' "$ROOT"
printf 'Output directory: %s\n' "$OUTDIR"
printf 'In-process warmups per recorded sample: %s\n' "$WARMUP"
printf 'Recorded repeats per policy: %s\n' "$REPEAT"
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
printf '\nStarting 54-case sweep in B -> S_IN -> S_OUT order...\n'

python tests/benchmarks/qwen3_decode_sweep.py \
    --batch-sizes 1 4 8 16 32 64 \
    --s-in-values 16 128 512 \
    --s-out-values 16 128 512 \
    --batch-prompt-mode identical \
    --max-mpk-batch-size 64 \
    --split-kv-cache-min-batch-size 0 \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --timeout 3600 \
    --progress-interval-seconds 300 \
    --output-dir "$OUTDIR"

result=$?
printf '%s\n' "$result" > "$EXIT_CODE_FILE"
printf '\nQwen3 comparison sweep finished at %s with exit code %s.\n' \
    "$(date --iso-8601=seconds)" "$result"
printf 'Results: %s\n' "$OUTDIR/raw_results.csv"
printf 'Progress: %s\n' "$OUTDIR/progress.json"

exit "$result"
