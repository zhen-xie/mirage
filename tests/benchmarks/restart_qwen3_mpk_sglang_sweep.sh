#!/usr/bin/env bash

# Stop one sweep process tree, rebuild Mirage when running MPK, and resume the
# same output directory. Completed case summaries are retained and reused.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${MODE:-mpk}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_mpk_sglang_sweep"}
MODE_DIR="$OUTDIR/$MODE"
PID_FILE="$MODE_DIR/launcher.pid"
BUILD_LOG="$MODE_DIR/rebuild.log"

cd "$ROOT" || exit 1
mkdir -p "$MODE_DIR"

declare -a tree_pids=()

collect_tree() {
    local parent=$1
    local child
    while read -r child; do
        [[ -n "$child" ]] || continue
        collect_tree "$child"
    done < <(pgrep -P "$parent" 2>/dev/null || true)
    tree_pids+=("$parent")
}

add_root() {
    local pid=$1
    [[ "$pid" =~ ^[0-9]+$ ]] || return
    kill -0 "$pid" 2>/dev/null || return
    collect_tree "$pid"
}

if [[ -f "$PID_FILE" ]]; then
    add_root "$(cat "$PID_FILE")"
fi

# Recover workers whose launcher PID file is stale. The match is restricted to
# this exact output directory and the two Mirage sweep executables.
while read -r pid; do
    [[ -n "$pid" ]] && add_root "$pid"
done < <(
    {
        pgrep -f "qwen3_decode_sweep.py.*$OUTDIR" 2>/dev/null || true
        pgrep -f "qwen3_decode_backend.py.*$OUTDIR" 2>/dev/null || true
    } | sort -u
)

if (( ${#tree_pids[@]} )); then
    mapfile -t tree_pids < <(printf '%s\n' "${tree_pids[@]}" | awk '!seen[$0]++')
    printf 'Stopping %s process(es): %s\n' "${#tree_pids[@]}" "${tree_pids[*]}"
    kill -TERM "${tree_pids[@]}" 2>/dev/null || true
    for _ in {1..20}; do
        alive=0
        for pid in "${tree_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [[ "$alive" -eq 0 ]] && break
        sleep 1
    done
    for pid in "${tree_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            printf 'Force stopping PID %s.\n' "$pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
else
    printf 'No running process was found for %s.\n' "$MODE_DIR"
fi

rm -f "$PID_FILE"

if [[ "$MODE" == mpk ]]; then
    export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
    export FLASHINFER_USE_CUDA_NORM=1
    export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

    printf 'Running Python syntax checks...\n'
    python -m py_compile \
        demo/qwen3/demo.py \
        python/mirage/mpk/persistent_kernel.py \
        tests/benchmarks/qwen3_decode_backend.py \
        tests/benchmarks/qwen3_decode_sweep.py \
        tests/benchmarks/merge_qwen3_mpk_sglang_model_sweep.py
    result=$?
    if [[ "$result" -ne 0 ]]; then
        printf 'Syntax checks failed with exit code %s.\n' "$result"
        exit "$result"
    fi

    printf 'Building and installing Mirage...\n'
    if python -m pip install -e . -v --no-build-isolation > "$BUILD_LOG" 2>&1; then
        printf 'Build and installation completed.\n'
    else
        result=$?
        printf 'Build failed with exit code %s. Last 100 lines:\n' "$result"
        tail -n 100 "$BUILD_LOG"
        exit "$result"
    fi
fi

printf 'Restarting %s sweep in %s...\n' "$MODE" "$OUTDIR"
MODE="$MODE" OUTDIR="$OUTDIR" \
MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B}" \
BATCH_SIZES="${BATCH_SIZES:-1 4 16 64}" \
S_IN_VALUES="${S_IN_VALUES:-128 512 1024}" \
S_OUT_VALUES="${S_OUT_VALUES:-128 512 1024}" \
WARMUP="${WARMUP:-1}" REPEAT="${REPEAT:-1}" TIMEOUT="${TIMEOUT:-7200}" \
bash tests/benchmarks/launch_qwen3_mpk_sglang_sweep_nohup.sh
