#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MPK_SUMMARY=${MPK_SUMMARY:-"$ROOT/results/qwen3_step15_mpk_window_profile_v1/summary.json"}
SGLANG_SUMMARY=${SGLANG_SUMMARY:-"$ROOT/results/qwen3_step16_sglang_window_profile_v1/summary.json"}
STEP14_COMPARISON=${STEP14_COMPARISON:-"$ROOT/results/qwen3_step14_comparison/comparison.json"}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step16_profile_comparison"}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
printf 'Running syntax checks...\n'
python -m py_compile tests/benchmarks/qwen3_step16_compare_profiles.py || exit 1

printf 'Comparing MPK and SGLang window profiles...\n'
python tests/benchmarks/qwen3_step16_compare_profiles.py \
    --mpk-summary "$MPK_SUMMARY" \
    --sglang-summary "$SGLANG_SUMMARY" \
    --step14-comparison "$STEP14_COMPARISON" \
    --output-dir "$OUTDIR"
result=$?
printf 'Step 16 profile comparison exited with code %s.\n' "$result"
printf 'Comparison: %s\n' "$OUTDIR/comparison.json"
exit "$result"
