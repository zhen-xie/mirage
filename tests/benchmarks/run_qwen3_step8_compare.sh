#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MPK_SUMMARY=${MPK_SUMMARY:-"$ROOT/results/qwen3_step8_mpk/summary.json"}
SGLANG_DIR=${SGLANG_DIR:-"$ROOT/results/qwen3_step8_sglang"}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step8_comparison"}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"

printf 'Running syntax checks...\n'
python -m py_compile tests/benchmarks/qwen3_step8_compare_sglang.py || exit 1

if [[ ! -f "$MPK_SUMMARY" ]]; then
    printf 'Missing MPK Step 7 summary: %s\n' "$MPK_SUMMARY"
    exit 1
fi

printf 'Combining MPK and SGLang results...\n'
python tests/benchmarks/qwen3_step8_compare_sglang.py \
    --mpk-summary "$MPK_SUMMARY" \
    --sglang-dir "$SGLANG_DIR" \
    --output-dir "$OUTDIR"
result=$?

printf 'Step 8 comparison exited with code %s.\n' "$result"
printf 'Comparison JSON: %s\n' "$OUTDIR/comparison.json"
printf 'Comparison CSV: %s\n' "$OUTDIR/comparison.csv"
exit "$result"
