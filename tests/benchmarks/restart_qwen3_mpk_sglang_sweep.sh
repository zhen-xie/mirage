#!/usr/bin/env bash

# Stop the selected sweep process tree, rebuild Mirage for MPK, and restart.

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

if [[ -f "$PID_FILE" ]]; then
    root_pid=$(cat "$PID_FILE")
    if [[ "$root_pid" =~ ^[0-9]+$ ]] && kill -0 "$root_pid" 2>/dev/null; then
        collect_tree "$root_pid"
    fi
fi

if (( ${#tree_pids[@]} )); then
    printf 'Stopping %s process(es): %s\n' "${#tree_pids[@]}" "${tree_pids[*]}"
    kill -TERM "${tree_pids[@]}" 2>/dev/null || true
    for _ in {1..20}; do
        alive=0
        for pid in "${tree_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
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
    printf 'No active launcher found in %s.\n' "$PID_FILE"
fi
rm -f "$PID_FILE"

if [[ "$MODE" == mpk ]]; then
    export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
    export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"
    CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
    if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
        export CUDA_HOME="$CUDA_TOOLKIT"
        export CUDA_PATH="$CUDA_TOOLKIT"
        export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
        export PATH="$CUDA_TOOLKIT/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    printf 'CUDA compiler: %s\n' "$(command -v nvcc)"
    nvcc --version | tail -n 1
    printf 'Running Python syntax checks...\n'
    python -m py_compile \
        demo/qwen3/demo.py \
        tests/benchmarks/qwen3_decode_backend.py \
        tests/benchmarks/qwen3_decode_sweep.py \
        tests/benchmarks/merge_qwen3_mpk_sglang_model_sweep.py || exit $?

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

printf 'Starting %s sweep...\n' "$MODE"
MODE="$MODE" OUTDIR="$OUTDIR" \
MODELS="${MODELS:-Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B}" \
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64}" \
S_IN_VALUES="${S_IN_VALUES:-128 512 1024}" \
S_OUT_VALUES="${S_OUT_VALUES:-128 512 1024}" \
TIMEOUT="${TIMEOUT:-7200}" \
bash tests/benchmarks/launch_qwen3_mpk_sglang_sweep_nohup.sh
