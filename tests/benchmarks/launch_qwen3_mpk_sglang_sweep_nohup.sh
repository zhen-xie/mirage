#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${MODE:-mpk}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_mpk_sglang_sweep"}
MODE_DIR="$OUTDIR/$MODE"
LOG="$MODE_DIR/nohup.log"
PID_FILE="$MODE_DIR/launcher.pid"
EXIT_FILE="$MODE_DIR/exit_code.txt"

mkdir -p "$MODE_DIR"
rm -f "$EXIT_FILE"

nohup env \
    MODE="$MODE" OUTDIR="$OUTDIR" \
    MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B}" \
    BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64}" \
    S_IN_VALUES="${S_IN_VALUES:-128 512 1024}" \
    S_OUT_VALUES="${S_OUT_VALUES:-128 512 1024}" \
    WARMUP=1 REPEAT=1 TIMEOUT="${TIMEOUT:-7200}" \
    bash -c 'bash tests/benchmarks/run_qwen3_mpk_sglang_sweep.sh; result=$?; printf "%s\n" "$result" > "$OUTDIR/$MODE/exit_code.txt"; exit "$result"' \
    > "$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
printf 'Started %s sweep.\n' "$MODE"
printf 'PID: %s\n' "$pid"
printf 'Output directory: %s\n' "$OUTDIR"
printf 'Log: %s\n' "$LOG"
printf 'Monitor with: tail -f %q\n' "$LOG"
