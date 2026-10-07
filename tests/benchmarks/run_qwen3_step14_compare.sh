#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MPK_SUMMARY=${MPK_SUMMARY:-"$ROOT/results/qwen3_step14_mpk/summary.json"}
SGLANG_DIR=${SGLANG_DIR:-"$ROOT/results/qwen3_step14_sglang"}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step14_comparison"}
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
python -m py_compile tests/benchmarks/qwen3_step14_compare.py || exit 1
if [[ ! -f "$MPK_SUMMARY" ]]; then
    printf 'Missing MPK summary: %s\n' "$MPK_SUMMARY"
    exit 1
fi
python tests/benchmarks/qwen3_step14_compare.py \
    --mpk-summary "$MPK_SUMMARY" --sglang-dir "$SGLANG_DIR" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 14 comparison exited with code %s.\n' "$result"
printf 'Comparison: %s\n' "$OUTDIR/comparison.json"
exit "$result"
