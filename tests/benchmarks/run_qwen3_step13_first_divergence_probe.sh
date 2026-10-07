#!/usr/bin/env bash

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step13_first_divergence_probe"}
MODEL=${MODEL:-"Qwen/Qwen3-4B"}
TIMEOUT=${TIMEOUT:-3600}
BUILD_LOG="$OUTDIR/build.log"

cd "$ROOT" || exit 1
mkdir -p "$OUTDIR" "$OUTDIR/cache"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export PYTHONUNBUFFERED=1
export NVCC_PREPEND_FLAGS="--threads 8${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
if [[ -x "$CUDA_TOOLKIT/bin/nvcc" ]]; then
    export CUDA_HOME="$CUDA_TOOLKIT"
    export CUDA_PATH="$CUDA_TOOLKIT"
    export CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
    export PATH="$CUDA_TOOLKIT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf 'Running syntax checks...\n'
python -m py_compile demo/qwen3/demo.py || exit 1
printf 'Building and installing Mirage...\n'
if timeout "$TIMEOUT" python -m pip install -e . -v --no-build-isolation \
    > "$BUILD_LOG" 2>&1
then
    printf 'Build completed.\n'
else
    result=$?
    printf 'Build failed with exit code %s. Last 100 lines:\n' "$result"
    tail -n 100 "$BUILD_LOG"
    exit "$result"
fi

common=(
    --model "$MODEL"
    --input-length 512
    --max-seq-length 640
    --max-new-tokens 7
    --page-size 640
    --max-num-pages 1
    --max-num-batched-requests 1
    --max-num-batched-tokens 8
    --ignore-eos
    --capture-final-logits-topk 10
)

printf 'Running Torch first-divergence probe...\n'
timeout "$TIMEOUT" python demo/qwen3/demo.py \
    "${common[@]}" \
    --save-tokens "$OUTDIR/torch.json" \
    > "$OUTDIR/torch.log" 2>&1 || {
        result=$?
        tail -n 100 "$OUTDIR/torch.log"
        exit "$result"
    }

printf 'Running MPK first-divergence probe...\n'
timeout "$TIMEOUT" python demo/qwen3/demo.py \
    "${common[@]}" \
    --use-mirage \
    --mpk-policy decode-only \
    --mpk-attention default \
    --mpk-kernel-cache-dir "$OUTDIR/cache" \
    --save-tokens "$OUTDIR/mpk.json" \
    > "$OUTDIR/mpk.log" 2>&1 || {
        result=$?
        tail -n 100 "$OUTDIR/mpk.log"
        exit "$result"
    }

python - "$OUTDIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
torch_data = json.loads((root / "torch.json").read_text())
mpk_data = json.loads((root / "mpk.json").read_text())
torch_tokens = torch_data["token_ids"]
mpk_tokens = mpk_data["token_ids_by_request"][0]

print("Torch tokens:", torch_tokens)
print("MPK tokens:  ", mpk_tokens)
print("Torch final top-k:")
for item in torch_data["final_logits_topk"]:
    print(" ", item)
print("MPK final top-k:")
for item in mpk_data["final_logits_topk"]:
    print(" ", item)

torch_top = torch_data["final_logits_topk"]
mpk_top = mpk_data["final_logits_topk"]
report = {
    "torch_tokens": torch_tokens,
    "mpk_tokens": mpk_tokens,
    "torch_topk": torch_top,
    "mpk_topk": mpk_top,
    "torch_margin": torch_top[0]["logit"] - torch_top[1]["logit"],
    "mpk_margin": mpk_top[0]["logit"] - mpk_top[1]["logit"],
}
(root / "summary.json").write_text(json.dumps(report, indent=2))
print("Torch top-2 margin:", report["torch_margin"])
print("MPK top-2 margin:", report["mpk_margin"])
print("Wrote", root / "summary.json")
PY
