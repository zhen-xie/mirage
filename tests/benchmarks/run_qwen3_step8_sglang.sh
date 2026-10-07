#!/usr/bin/env bash

set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step8_sglang"}
MODELS=${MODELS:-"Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B"}
BATCH_SIZE=${BATCH_SIZE:-1}
S_IN=${S_IN:-128}
S_OUT=${S_OUT:-128}
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
    export CUDAHOSTCXX="$CXX"
    export NVCC_CCBIN="$CXX"
fi

printf 'Active conda environment: %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf 'Python executable: %s\n' "$(command -v python)"
printf 'Shape: B=%s S_IN=%s S_OUT=%s\n' "$BATCH_SIZE" "$S_IN" "$S_OUT"

python - "$OUTDIR/environment.json" <<'PY'
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

packages = {}
for name in ("sglang", "sgl-kernel", "torch", "flashinfer-python", "transformers", "triton"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

import torch
data = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
Path(sys.argv[1]).write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(data, indent=2))
PY

failed=0
index=0
read -r -a models <<< "$MODELS"
total=${#models[@]}

for model in "${models[@]}"; do
    index=$((index + 1))
    safe_name=${model//\//_}
    result_file="$OUTDIR/${safe_name}.jsonl"
    log="$OUTDIR/${safe_name}.log"
    rm -f "$result_file"
    printf '\n[%s/%s] Running SGLang model=%s...\n' "$index" "$total" "$model"
    if timeout "$TIMEOUT" python -m sglang.benchmark.one_batch \
        --model-path "$model" \
        --tp-size 1 \
        --dtype bfloat16 \
        --batch-size "$BATCH_SIZE" \
        --input-len "$S_IN" \
        --output-len "$S_OUT" \
        --run-name "step8_${safe_name}" \
        --result-filename "$result_file" \
        > "$log" 2>&1
    then
        printf 'SGLang model completed: %s\n' "$model"
        tail -n 20 "$log"
    else
        result=$?
        failed=$((failed + 1))
        printf 'SGLang model failed with exit code %s: %s\n' "$result" "$model"
        tail -n 100 "$log"
    fi
done

printf '\nCompleted SGLang runs: %s/%s\n' "$((total - failed))" "$total"
printf 'Results directory: %s\n' "$OUTDIR"
exit "$failed"
