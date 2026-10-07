#!/usr/bin/env bash
set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step14_sglang"}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
CASES=${CASES:-"short:128:128 long_context:1024:128 long_generation:128:1024"}
BATCH_SIZES=${BATCH_SIZES:-"1 8 32"}
TIMEOUT=${TIMEOUT:-3600}
CONDA_ENV=${CONDA_ENV:-sglang-bench}

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
read -r -a models <<< "$MODELS"
read -r -a cases <<< "$CASES"
read -r -a batches <<< "$BATCH_SIZES"
total=$((${#models[@]} * ${#cases[@]} * ${#batches[@]}))
index=0
failed=0

for model in "${models[@]}"; do
    model_name=${model//\//_}
    for spec in "${cases[@]}"; do
        IFS=: read -r case_name s_in s_out <<< "$spec"
        for batch in "${batches[@]}"; do
            index=$((index + 1))
            stem="${model_name}_${case_name}_b${batch}"
            warmup="$OUTDIR/${stem}_warmup.jsonl"
            warmup_log="$OUTDIR/${stem}_warmup.log"
            result_file="$OUTDIR/${stem}.jsonl"
            log="$OUTDIR/${stem}.log"
            if [[ -s "$result_file" ]]; then
                printf '[%s/%s] Skipping completed %s\n' "$index" "$total" "$stem"
                continue
            fi
            printf '\n[%s/%s] Warmup %s B=%s S_IN=%s S_OUT=%s\n' \
                "$index" "$total" "$model" "$batch" "$s_in" "$s_out"
            rm -f "$warmup"
            if timeout "$TIMEOUT" python -m sglang.benchmark.one_batch \
                --model-path "$model" --tp-size 1 --dtype bfloat16 \
                --batch-size "$batch" --input-len "$s_in" --output-len "$s_out" \
                --run-name "step14_${stem}_warmup" --result-filename "$warmup" \
                > "$warmup_log" 2>&1
            then
                :
            else
                code=$?
                failed=$((failed + 1))
                printf 'Warmup failed with exit code %s: %s\n' "$code" "$stem"
                tail -n 80 "$warmup_log"
                continue
            fi
            printf '[%s/%s] Measuring %s\n' "$index" "$total" "$stem"
            rm -f "$result_file"
            if timeout "$TIMEOUT" python -m sglang.benchmark.one_batch \
                --model-path "$model" --tp-size 1 --dtype bfloat16 \
                --batch-size "$batch" --input-len "$s_in" --output-len "$s_out" \
                --run-name "step14_${stem}" --result-filename "$result_file" \
                > "$log" 2>&1
            then
                tail -n 12 "$log"
            else
                code=$?
                failed=$((failed + 1))
                printf 'Measurement failed with exit code %s: %s\n' "$code" "$stem"
                tail -n 80 "$log"
            fi
        done
    done
done

printf '\nStep 14 SGLang sweep completed with %s failed process(es).\n' "$failed"
printf 'Results: %s\n' "$OUTDIR"
exit "$failed"
