#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WARMUP=${QWEN3_COMPARISON_WARMUP:-1}
REPEAT=${QWEN3_COMPARISON_REPEAT:-1}
OUTDIR=${QWEN3_COMPARISON_OUTDIR:-$ROOT/results/qwen3_sglang_decode_only_54_warmup${WARMUP}_repeat${REPEAT}}
EXIT_CODE_FILE="$OUTDIR/exit_code.txt"

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

mkdir -p "$OUTDIR"
cd "$ROOT" || exit 1

printf 'Qwen3 decode-only comparison sweep started at %s\n' "$(date --iso-8601=seconds)"
printf 'Output directory: %s\n' "$OUTDIR"
printf 'In-process warmups: %s\n' "$WARMUP"
printf 'Recorded repeats: %s\n' "$REPEAT"
printf 'Process ID: %s\n' "$$"
printf '%s\n' "$$" > "$OUTDIR/worker.pid"
rm -f "$EXIT_CODE_FILE"

python -m py_compile \
    demo/qwen3/demo.py \
    python/mirage/mpk/persistent_kernel.py \
    tests/benchmarks/qwen3_decode_backend.py \
    tests/benchmarks/qwen3_decode_sweep.py
result=$?
if [[ "$result" -ne 0 ]]; then
    printf '%s\n' "$result" > "$EXIT_CODE_FILE"
    exit "$result"
fi

python tests/benchmarks/qwen3_decode_sweep.py \
    --batch-sizes 1 4 8 16 32 64 \
    --s-in-values 16 128 512 \
    --s-out-values 16 128 512 \
    --policies decode-only \
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
printf 'Qwen3 decode-only sweep finished at %s with exit code %s.\n' \
    "$(date --iso-8601=seconds)" "$result"
printf 'Results: %s/raw_results.csv\n' "$OUTDIR"
exit "$result"
