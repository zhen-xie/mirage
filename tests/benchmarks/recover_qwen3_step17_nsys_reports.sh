#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step17_nsys_compare_v1"}
STEP14_MPK=${STEP14_MPK:-"$ROOT/results/qwen3_step14_mpk"}
STEP14_COMPARISON=${STEP14_COMPARISON:-"$ROOT/results/qwen3_step14_comparison/comparison.json"}

cd "$ROOT" || exit 1
python -m py_compile \
    tests/benchmarks/summarize_qwen3_nsys.py \
    tests/benchmarks/qwen3_step17_nsys_compare.py || exit 1

for backend in mpk sglang; do
    prefix="$OUTDIR/$backend/decode"
    if [[ ! -f "$prefix.nsys-rep" ]]; then
        printf 'Missing Nsight report: %s\n' "$prefix.nsys-rep"
        exit 1
    fi
    printf 'Re-exporting %s Nsight reports...\n' "$backend"
    rm -f "$prefix.sqlite"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
        "$prefix.nsys-rep" > "$OUTDIR/$backend/kernels.csv" || exit 1
    nsys stats --force-export=true --report cuda_api_sum --format csv \
        "$prefix.nsys-rep" > "$OUTDIR/$backend/apis.csv" || exit 1
    python tests/benchmarks/summarize_qwen3_nsys.py \
        --backend "$backend" --kernel-csv "$OUTDIR/$backend/kernels.csv" \
        --api-csv "$OUTDIR/$backend/apis.csv" \
        --output "$OUTDIR/$backend/summary.json" || exit 1
done

printf 'Validating recovered matched captures...\n'
python tests/benchmarks/qwen3_step17_nsys_compare.py \
    --mpk-profile "$OUTDIR/mpk/summary.json" \
    --sglang-profile "$OUTDIR/sglang/summary.json" \
    --mpk-output "$OUTDIR/mpk/tokens.json" \
    --torch-reference "$STEP14_MPK/Qwen_Qwen3-8B_long_context_torch.json" \
    --step14-comparison "$STEP14_COMPARISON" \
    --output "$OUTDIR/comparison.json"
result=$?
printf 'Step 17 recovered comparison exited with code %s.\n' "$result"
printf 'Comparison: %s\n' "$OUTDIR/comparison.json"
exit "$result"
