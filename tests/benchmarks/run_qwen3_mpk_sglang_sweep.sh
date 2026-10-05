#!/usr/bin/env bash

# Run one side of the MPK/SGLang comparison. Run MODE=mpk from mirage-mpk
# and MODE=sglang from sglang-bench. Both modes use the same model and shape
# lists, so their outputs can be merged without hand-written case mappings.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${MODE:-mpk}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
BATCH_SIZES=${BATCH_SIZES:-"1 4 16 64"}
S_IN_VALUES=${S_IN_VALUES:-"128 512 1024"}
S_OUT_VALUES=${S_OUT_VALUES:-"128 512 1024"}
WARMUP=${WARMUP:-1}
REPEAT=${REPEAT:-1}
TIMEOUT=${TIMEOUT:-7200}
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_mpk_sglang_sweep"}

export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export FLASHINFER_USE_CUDA_NORM=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

# FlashInfer 0.7 emits --compress-mode=size, which CUDA 12.4 does not accept.
# Prefer the cluster CUDA 13.3 toolkit when available while allowing an
# explicit MIRAGE_CUDA_HOME override on other systems.
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
if [[ "$MODE" == mpk && -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_TOOLKIT"
    export CUDA_PATH="$CUDA_TOOLKIT"
    export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
    export PATH="$CUDA_TOOLKIT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

slug_for_model() {
    printf '%s' "$1" | tr '/:' '__'
}

mkdir -p "$OUTDIR/$MODE"
cd "$ROOT" || exit 1

printf 'Mode: %s\n' "$MODE"
printf 'Models: %s\n' "$MODELS"
printf 'Batch sizes: %s\n' "$BATCH_SIZES"
printf 'Input lengths: %s\n' "$S_IN_VALUES"
printf 'Output lengths: %s\n' "$S_OUT_VALUES"
printf 'Warmup: %s; repeat: %s\n' "$WARMUP" "$REPEAT"
if [[ "$MODE" == mpk ]]; then
    printf 'CUDA compiler: %s\n' "$(command -v nvcc)"
    nvcc --version | tail -n 1
fi

failed=0
for model in $MODELS; do
    slug=$(slug_for_model "$model")
    model_dir="$OUTDIR/$MODE/$slug"
    mkdir -p "$model_dir"
    printf '{"model":"%s"}\n' "$model" > "$model_dir/model.json"

    printf '\nRunning %s for %s...\n' "$MODE" "$model"
    if [[ "$MODE" == mpk ]]; then
        if python tests/benchmarks/qwen3_decode_sweep.py \
            --batch-sizes $BATCH_SIZES \
            --s-in-values $S_IN_VALUES \
            --s-out-values $S_OUT_VALUES \
            --policies decode-only \
            --mpk-attention-policy auto \
            --split-kv-cache-min-batch-size 0 \
            --optimized-normal \
            --batch-prompt-mode identical \
            --model "$model" \
            --warmup "$WARMUP" \
            --repeat "$REPEAT" \
            --timeout "$TIMEOUT" \
            --progress-interval-seconds 300 \
            --allow-code-change-resume \
            --output-dir "$model_dir"
        then
            printf 'MPK sweep completed for %s.\n' "$model"
        else
            result=$?
            printf 'MPK sweep failed for %s with exit code %s.\n' "$model" "$result"
            failed=1
        fi
    elif [[ "$MODE" == sglang ]]; then
        result_file="$model_dir/results.jsonl"
        if python -m sglang.benchmark.one_batch \
            --model-path "$model" \
            --tp-size 1 \
            --dtype bfloat16 \
            --batch-size $BATCH_SIZES \
            --input-len $S_IN_VALUES \
            --output-len $S_OUT_VALUES \
            --run-name "mpk_comparison_$slug" \
            --result-filename "$result_file"
        then
            printf 'SGLang sweep completed for %s.\n' "$model"
        else
            result=$?
            printf 'SGLang sweep failed for %s with exit code %s.\n' "$model" "$result"
            failed=1
        fi
    else
        printf 'MODE must be mpk or sglang; received %s.\n' "$MODE"
        exit 2
    fi
done

printf '\nFinished mode %s with %s failed model sweep(s).\n' "$MODE" "$failed"
exit "$failed"
