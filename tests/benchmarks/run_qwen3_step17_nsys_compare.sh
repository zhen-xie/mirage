#!/usr/bin/env bash
set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step17_nsys_compare_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
THRESHOLD=${THRESHOLD:-256}
SGLANG_ENV=${SGLANG_ENV:-sglang-bench}
STEP14_MPK=${STEP14_MPK:-"$ROOT/results/qwen3_step14_mpk"}
STEP14_SGLANG=${STEP14_SGLANG:-"$ROOT/results/qwen3_step14_sglang"}
STEP14_COMPARISON=${STEP14_COMPARISON:-"$ROOT/results/qwen3_step14_comparison/comparison.json"}

eval "$(conda shell.bash hook)"
conda activate mirage-mpk || exit 1
set -u
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR/mpk" "$OUTDIR/sglang" "$OUTDIR/cache"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

printf 'Checking tools and syntax...\n'
command -v nsys || exit 1
python -m py_compile \
    demo/qwen3/demo.py \
    tests/benchmarks/summarize_qwen3_nsys.py \
    tests/benchmarks/qwen3_step17_nsys_compare.py || exit 1

printf 'Building and installing Mirage...\n'
timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$OUTDIR/build.log" 2>&1 || {
        code=$?
        printf 'Build failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/build.log"
        exit "$code"
    }

cache="$OUTDIR/cache/qwen3_8b_b32_seq1152"
mpk_output="$OUTDIR/mpk/tokens.json"
mpk_prefix="$OUTDIR/mpk/decode"

mpk_args=(
    --model "$MODEL" --use-mirage --mpk-policy decode-only
    --mpk-attention auto --mpk-auto-split-kv-threshold "$THRESHOLD"
    --mpk-split-kv-chunk-size 128 --normal-prefill-attention sdpa
    --prefill-warmup-runs 1 --normal-prefill-cuda-graph
    --input-length 1024 --max-seq-length 1152 --max-new-tokens 128
    --page-size 1152 --max-num-pages 32 --max-num-batched-requests 32
    --max-num-batched-tokens 32 --ignore-eos
    --mpk-kernel-cache-dir "$cache"
)

printf 'Preparing and correctness-checking MPK kernel cache...\n'
timeout "$TIMEOUT" python demo/qwen3/demo.py "${mpk_args[@]}" \
    --save-tokens "$OUTDIR/mpk/warmup_tokens.json" \
    > "$OUTDIR/mpk/warmup.log" 2>&1 || {
        code=$?
        printf 'MPK preparation failed with exit code %s.\n' "$code"
        tail -n 100 "$OUTDIR/mpk/warmup.log"
        exit "$code"
    }

printf 'Capturing MPK decode with Nsight Systems...\n'
rm -f "$mpk_prefix.nsys-rep" "$mpk_output"
timeout "$TIMEOUT" nsys profile \
    --force-overwrite=true --trace=cuda,nvtx --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$mpk_prefix" \
    python demo/qwen3/demo.py "${mpk_args[@]}" \
    --nsys-decode-capture --save-tokens "$mpk_output" \
    > "$OUTDIR/mpk/run.log" 2>&1 || {
        code=$?
        printf 'MPK Nsight capture failed with exit code %s.\n' "$code"
        tail -n 120 "$OUTDIR/mpk/run.log"
        exit "$code"
    }

printf 'Extracting MPK Nsight reports...\n'
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    "$mpk_prefix.nsys-rep" > "$OUTDIR/mpk/kernels.csv" || exit 1
nsys stats --force-export=true --report cuda_api_sum --format csv \
    "$mpk_prefix.nsys-rep" > "$OUTDIR/mpk/apis.csv" || exit 1
python tests/benchmarks/summarize_qwen3_nsys.py \
    --backend mpk --kernel-csv "$OUTDIR/mpk/kernels.csv" \
    --api-csv "$OUTDIR/mpk/apis.csv" --output "$OUTDIR/mpk/summary.json"

set +u
conda activate "$SGLANG_ENV" || exit 1
set -u
export PYTHONUNBUFFERED=1
if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
fi

sg_prefix="$OUTDIR/sglang/decode"
sg_result="$OUTDIR/sglang/result.jsonl"
printf 'Capturing SGLang decode with Nsight Systems...\n'
rm -f "$sg_prefix.nsys-rep" "$sg_result"
timeout "$TIMEOUT" nsys profile \
    --force-overwrite=true --trace=cuda,nvtx --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$sg_prefix" \
    python -m sglang.benchmark.one_batch \
    --model-path "$MODEL" --tp-size 1 --dtype bfloat16 \
    --batch-size 32 --input-len 1024 --output-len 128 \
    --run-name step17_sglang_nsys --result-filename "$sg_result" \
    --profile --profile-activities CUDA_PROFILER --profile-stage decode \
    --profile-start-step 0 --profile-steps 127 \
    > "$OUTDIR/sglang/run.log" 2>&1 || {
        code=$?
        printf 'SGLang Nsight capture failed with exit code %s.\n' "$code"
        tail -n 120 "$OUTDIR/sglang/run.log"
        exit "$code"
    }

printf 'Extracting SGLang Nsight reports...\n'
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
    "$sg_prefix.nsys-rep" > "$OUTDIR/sglang/kernels.csv" || exit 1
nsys stats --force-export=true --report cuda_api_sum --format csv \
    "$sg_prefix.nsys-rep" > "$OUTDIR/sglang/apis.csv" || exit 1
python tests/benchmarks/summarize_qwen3_nsys.py \
    --backend sglang --kernel-csv "$OUTDIR/sglang/kernels.csv" \
    --api-csv "$OUTDIR/sglang/apis.csv" --output "$OUTDIR/sglang/summary.json"

printf 'Validating matched Nsight captures...\n'
python tests/benchmarks/qwen3_step17_nsys_compare.py \
    --mpk-profile "$OUTDIR/mpk/summary.json" \
    --sglang-profile "$OUTDIR/sglang/summary.json" \
    --mpk-output "$mpk_output" \
    --torch-reference "$STEP14_MPK/Qwen_Qwen3-8B_long_context_torch.json" \
    --step14-comparison "$STEP14_COMPARISON" \
    --output "$OUTDIR/comparison.json"
result=$?
printf 'Step 17 Nsight comparison exited with code %s.\n' "$result"
printf 'Comparison: %s\n' "$OUTDIR/comparison.json"
exit "$result"
