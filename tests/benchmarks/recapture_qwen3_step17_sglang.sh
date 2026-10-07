#!/usr/bin/env bash
set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step17_nsys_compare_v1"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
TIMEOUT=${TIMEOUT:-3600}
CONDA_ENV=${CONDA_ENV:-sglang-bench}

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV" || exit 1
set -u
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR/sglang"
export PYTHONUNBUFFERED=1
if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    export CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
fi

prefix="$OUTDIR/sglang/decode"
result_file="$OUTDIR/sglang/result.jsonl"
rm -f "$prefix.nsys-rep" "$prefix.sqlite" "$result_file"
printf 'Recapturing SGLang with CUDA Graph node tracing...\n'
timeout "$TIMEOUT" nsys profile \
    --force-overwrite=true --trace=cuda,nvtx --sample=none \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output "$prefix" \
    python -m sglang.benchmark.one_batch \
    --model-path "$MODEL" --tp-size 1 --dtype bfloat16 \
    --batch-size 32 --input-len 1024 --output-len 128 \
    --run-name step17_sglang_nsys --result-filename "$result_file" \
    --profile --profile-activities CUDA_PROFILER --profile-stage decode \
    --profile-start-step 0 --profile-steps 127 \
    > "$OUTDIR/sglang/run_graph_nodes.log" 2>&1
code=$?
if [[ "$code" -ne 0 ]]; then
    printf 'SGLang recapture failed with exit code %s.\n' "$code"
    tail -n 120 "$OUTDIR/sglang/run_graph_nodes.log"
    exit "$code"
fi
printf 'SGLang CUDA Graph node capture completed: %s.nsys-rep\n' "$prefix"
