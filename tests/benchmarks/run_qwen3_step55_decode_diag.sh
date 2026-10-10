#!/usr/bin/env bash
# Step 55 diagnosis: isolate which demo option makes MPK decode emit invalid
# tokens at B=1, S_IN=1024. Each variant drops one suspect option.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUTDIR=${OUTDIR:-"$ROOT/results/qwen3_step55_decode_diag"}
MODEL=${MODEL:-Qwen/Qwen3-8B}
S_IN=${S_IN:-1024}; S_OUT=${S_OUT:-128}; TIMEOUT=${TIMEOUT:-1800}
REFERENCE=${REFERENCE:-"$ROOT/results/qwen3_step55_decode_b1_8/mpk_repeat1/Qwen_Qwen3-8B_long_context_torch.json"}
VARIANTS=${VARIANTS:-"full no_prefill_graph no_prefill_warmup default_attention"}
cd "$ROOT" || exit 1; mkdir -p "$OUTDIR"
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONUNBUFFERED=1
CUDA_TOOLKIT=${MIRAGE_CUDA_HOME:-/opt/ohpc/pub/apps/cuda/13.3}
export CUDA_HOME="$CUDA_TOOLKIT" CUDA_PATH="$CUDA_TOOLKIT" CUDACXX="$CUDA_TOOLKIT/bin/nvcc"
export PATH="$CUDA_TOOLKIT/bin:$PATH" LD_LIBRARY_PATH="$CUDA_TOOLKIT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOST_CXX=${CUDAHOSTCXX:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++}}
if [[ -n "$HOST_CXX" && -x "$HOST_CXX" ]]; then
 export CXX="$HOST_CXX" CUDAHOSTCXX="$HOST_CXX"; export NVCC_PREPEND_FLAGS="-ccbin $HOST_CXX --threads 8"
else export NVCC_PREPEND_FLAGS="--threads 8"; fi
MAX_SEQ=$((S_IN + S_OUT))
base=(python demo/qwen3/demo.py --model "$MODEL"
 --input-length "$S_IN" --max-seq-length "$MAX_SEQ" --max-new-tokens "$S_OUT"
 --page-size "$MAX_SEQ" --max-num-pages 1 --max-num-batched-requests 1
 --max-num-batched-tokens 8 --ignore-eos
 --use-mirage --mpk-policy decode-only --normal-prefill-attention sdpa)
split=(--mpk-attention auto --mpk-auto-split-kv-threshold 256 --mpk-split-kv-chunk-size 128)
read -r -a variants <<< "$VARIANTS"
for variant in "${variants[@]}"; do
 case "$variant" in
  full) extra=("${split[@]}" --prefill-warmup-runs 1 --normal-prefill-cuda-graph); cache=auto ;;
  no_prefill_graph) extra=("${split[@]}" --prefill-warmup-runs 1); cache=auto ;;
  no_prefill_warmup) extra=("${split[@]}" --prefill-warmup-runs 0); cache=auto ;;
  default_attention) extra=(--mpk-attention default --prefill-warmup-runs 0); cache=default ;;
  *) printf 'Unknown variant %s\n' "$variant"; continue ;;
 esac
 printf '\n=== variant %s ===\n' "$variant"
 timeout "$TIMEOUT" "${base[@]}" "${extra[@]}" \
  --mpk-kernel-cache-dir "$OUTDIR/cache_$cache" \
  --save-tokens "$OUTDIR/$variant.json" > "$OUTDIR/$variant.log" 2>&1
 printf 'variant %s exit code %s\n' "$variant" "$?"
 grep -E "Prompt length|Error|error:" "$OUTDIR/$variant.log" | tail -n 3
done
python - "$OUTDIR" "$REFERENCE" "$S_OUT" $VARIANTS <<'PY'
import json, sys
from pathlib import Path
outdir, ref_path, s_out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
ref = json.loads(ref_path.read_text())["token_ids"] if ref_path.is_file() else []
rows = []
for variant in sys.argv[4:]:
    path = outdir / f"{variant}.json"
    if not path.is_file():
        rows.append({"variant": variant, "status": "no_json"}); continue
    d = json.loads(path.read_text())
    ids = d.get("token_ids", [])
    first_bad = next((i for i, t in enumerate(ids) if t < 0 or t >= d.get("vocab_size", 1 << 30)), None)
    match = next((i for i, (a, b) in enumerate(zip(ref, ids)) if a != b), min(len(ref), len(ids)))
    row = {"variant": variant, "generate_length": d.get("generate_length"),
           "invalid_token_count": d.get("invalid_token_count"),
           "first_invalid_index": first_bad, "prefix_match_vs_torch": match,
           "decode_step_ms": d.get("decode_step_time_ms"), "first16": ids[:16]}
    row["status"] = ("passed" if row["generate_length"] == s_out
                     and row["invalid_token_count"] == 0 and match >= min(10, s_out)
                     else "failed")
    rows.append(row)
print("torch first16:", ref[:16])
for row in rows:
    print(json.dumps(row))
(outdir / "summary.json").write_text(json.dumps({"step": 55, "phase": "decode_diag", "rows": rows}, indent=2) + "\n")
PY
