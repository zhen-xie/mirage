#!/usr/bin/env bash
set -o pipefail
set +u

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step16_sglang_profile_probe_v1"}
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

printf 'Active conda environment: %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf 'Python executable: %s\n' "$(command -v python)"

{
    printf 'nsys executable: '
    command -v nsys || true
    if command -v nsys >/dev/null 2>&1; then
        nsys --version || true
        printf '\nnsys profile help filters:\n'
        nsys profile --help 2>&1 | grep -E \
            'capture-range|cuda-graph|trace=|sample=|output=' || true
        printf '\nnsys stats help filters:\n'
        nsys stats --help 2>&1 | grep -E \
            'report|format|output|cuda_gpu_kern|nvtx' || true
    fi
} > "$OUTDIR/nsys.txt" 2>&1

python -m sglang.benchmark.one_batch --help \
    > "$OUTDIR/one_batch_help.txt" 2>&1

python - "$OUTDIR/python_capabilities.json" <<'PY'
import importlib
import inspect
import json
import pkgutil
import sys
from importlib import metadata
from pathlib import Path


def public_names(module):
    return sorted(
        name for name in dir(module)
        if not name.startswith("_") and any(
            word in name.lower()
            for word in ("profil", "trace", "nvtx", "benchmark")
        )
    )


packages = {}
for name in (
    "sglang", "sgl-kernel", "torch", "flashinfer-python",
    "transformers", "triton",
):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None

modules = {}
for name in (
    "sglang",
    "sglang.benchmark.one_batch",
    "sglang.srt",
    "torch.profiler",
):
    try:
        module = importlib.import_module(name)
        modules[name] = {
            "file": getattr(module, "__file__", None),
            "profile_related_names": public_names(module),
        }
    except Exception as error:
        modules[name] = {"error": f"{type(error).__name__}: {error}"}

sglang_modules = []
try:
    import sglang
    for info in pkgutil.walk_packages(sglang.__path__, prefix="sglang."):
        lower = info.name.lower()
        if any(word in lower for word in ("profil", "trace", "benchmark")):
            sglang_modules.append(info.name)
except Exception as error:
    sglang_modules.append(f"ERROR: {type(error).__name__}: {error}")

data = {
    "python": sys.version,
    "packages": packages,
    "modules": modules,
    "candidate_sglang_modules": sorted(sglang_modules),
}
Path(sys.argv[1]).write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps(data, indent=2))
PY

grep -inE 'profil|trace|nvtx|warmup|repeat|input-len|output-len|batch-size' \
    "$OUTDIR/one_batch_help.txt" > "$OUTDIR/one_batch_profile_options.txt" || true

printf '\nSGLang profile capability probe completed.\n'
printf 'nsys summary: %s\n' "$OUTDIR/nsys.txt"
printf 'SGLang CLI filters: %s\n' "$OUTDIR/one_batch_profile_options.txt"
printf 'Python capabilities: %s\n' "$OUTDIR/python_capabilities.json"

printf '\nnsys summary:\n'
cat "$OUTDIR/nsys.txt"
printf '\nSGLang CLI profile-related options:\n'
cat "$OUTDIR/one_batch_profile_options.txt"
