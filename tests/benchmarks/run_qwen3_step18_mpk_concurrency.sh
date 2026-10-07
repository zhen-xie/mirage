#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
STEP15_SUMMARY=${STEP15_SUMMARY:-"$ROOT/results/qwen3_step15_mpk_window_profile_v1/summary.json"}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step18_mpk_concurrency_v1"}

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
printf 'Running syntax checks...\n'
python -m py_compile tests/benchmarks/qwen3_step18_mpk_concurrency.py || exit 1
printf 'Analyzing MPK task concurrency and worker balance...\n'
python tests/benchmarks/qwen3_step18_mpk_concurrency.py \
    --step15-summary "$STEP15_SUMMARY" --output-dir "$OUTDIR"
result=$?
printf 'Step 18 MPK concurrency analysis exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
