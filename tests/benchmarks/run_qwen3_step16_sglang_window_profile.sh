#!/usr/bin/env bash
set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step16_sglang_window_profile_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
STEP14_SGLANG_DIR=${STEP14_SGLANG_DIR:-"$ROOT/results/qwen3_step14_sglang"}
TIMEOUT=${TIMEOUT:-3600}
CONDA_ENV=${CONDA_ENV:-sglang-bench}
CASES=${CASES:-"short_b1 short_b32 long_context_b32"}
WINDOWS=${WINDOWS:-"early middle late"}

eval "$(conda shell.bash hook)"
if ! conda activate "$CONDA_ENV"; then
    printf 'Failed to activate conda environment: %s\n' "$CONDA_ENV"
    exit 1
fi
set -u
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
export PYTHONUNBUFFERED=1
if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
fi

printf 'Active conda environment: %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf 'Python executable: %s\n' "$(command -v python)"
printf 'Running syntax checks...\n'
python -m py_compile \
    tests/benchmarks/qwen3_step16_sglang_window_profile.py \
    tests/benchmarks/summarize_qwen3_sglang_profile.py || exit 1

printf 'Running Step 16 early/middle/late SGLang profiles...\n'
python tests/benchmarks/qwen3_step16_sglang_window_profile.py \
    --model "$MODEL" --step14-sglang-dir "$STEP14_SGLANG_DIR" \
    --timeout "$TIMEOUT" --output-dir "$OUTDIR" \
    --cases $CASES --windows $WINDOWS
result=$?
printf 'Step 16 SGLang window profile runner exited with code %s.\n' "$result"
printf 'Summary: %s\n' "$OUTDIR/summary.json"
exit "$result"
