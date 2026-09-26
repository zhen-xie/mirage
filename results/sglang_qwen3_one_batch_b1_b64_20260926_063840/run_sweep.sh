#!/usr/bin/env bash
set -uo pipefail

eval "$(conda shell.bash hook)"

if ! conda activate sglang-bench; then
    printf 'Failed to activate sglang-bench.\n'
    exit 1
fi

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export NVCC_CCBIN="$CXX"
export PYTHONUNBUFFERED=1

outdir="$(cd "$(dirname "$0")" && pwd)"
result_file="$outdir/results.jsonl"
environment_file="$outdir/environment.txt"
status_file="$outdir/status.txt"

printf 'running\n' > "$status_file"

{
    printf 'Start time: '
    date --iso-8601=seconds
    printf 'Host: '
    hostname
    printf 'Working directory: '
    pwd
    printf 'Python: '
    python --version
    printf 'CC: '
    "$CC" --version | head -n 1
    printf 'CXX: '
    "$CXX" --version | head -n 1
    printf 'CUDA:\n'
    nvcc --version
    printf '\nGPU:\n'
    nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit \
        --format=csv,noheader
    printf '\nPackages:\n'
    python - <<'PY'
from importlib import metadata

for package in (
    "sglang",
    "sgl-kernel",
    "torch",
    "flashinfer-python",
    "transformers",
    "triton",
):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY
} > "$environment_file" 2>&1

printf 'Starting SGLang Qwen3 full sweep.\n'
printf 'Output directory: %s\n' "$outdir"
printf 'Total cases: 7 x 6 x 6 = 252\n'
printf 'Loop order: batch size, input length, output length\n'

python -m sglang.benchmark.one_batch \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --dtype bfloat16 \
    --batch-size 1 2 4 8 16 32 64 \
    --input-len 16 32 64 128 256 512 \
    --output-len 16 32 64 128 256 512 \
    --run-name qwen3_mpk_comparison_b1_b64 \
    --result-filename "$result_file"

result=$?

{
    printf 'Exit code: %s\n' "$result"
    printf 'End time: '
    date --iso-8601=seconds
} >> "$environment_file"

if [[ "$result" -eq 0 ]]; then
    printf 'completed\n' > "$status_file"
    printf 'SGLang sweep completed successfully.\n'
else
    printf 'failed:%s\n' "$result" > "$status_file"
    printf 'SGLang sweep failed with exit code %s.\n' "$result"
fi

exit "$result"
