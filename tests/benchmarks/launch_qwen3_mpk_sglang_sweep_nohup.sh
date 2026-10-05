#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${MODE:-mpk}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_mpk_sglang_sweep"}
LOG="$OUTDIR/$MODE/nohup.log"
PID_FILE="$OUTDIR/$MODE/launcher.pid"

mkdir -p "$OUTDIR/$MODE"

MODE="$MODE" OUTDIR="$OUTDIR" \
MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B}" \
BATCH_SIZES="${BATCH_SIZES:-1 4 16 64}" \
S_IN_VALUES="${S_IN_VALUES:-128 512 1024}" \
S_OUT_VALUES="${S_OUT_VALUES:-128 512 1024}" \
WARMUP="${WARMUP:-1}" REPEAT="${REPEAT:-1}" TIMEOUT="${TIMEOUT:-7200}" \
nohup bash tests/benchmarks/run_qwen3_mpk_sglang_sweep.sh \
    > "$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
printf 'Started %s sweep.\n' "$MODE"
printf 'PID: %s\n' "$pid"
printf 'Output directory: %s\n' "$OUTDIR"
printf 'Log: %s\n' "$LOG"
printf 'Monitor with: tail -f %q\n' "$LOG"
