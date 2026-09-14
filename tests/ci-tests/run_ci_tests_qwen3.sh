#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

echo "MIRAGE_HOME=${MIRAGE_HOME}"
CORRECTNESS_FAILURES=0

write_summary_row() {
  local mode="$1"
  local batch="$2"
  local input_length="$3"
  local output_length="$4"
  local torch_output="$5"
  local mpk_output="$6"
  local correctness="$7"

  # pytest has completed successfully when this is called, so every row in
  # the summary represents a passed length/prefix correctness check.
  python - "$RESULTS_FILE" "$mode" "$batch" "$input_length" "$output_length" \
    "$torch_output" "$mpk_output" "$correctness" <<'PY'
import csv
import json
import sys

(summary_path, mode, batch, sin, sout, torch_path, mpk_path, correctness) = sys.argv[1:]
with open(torch_path, encoding="utf-8") as f:
    torch = json.load(f)
with open(mpk_path, encoding="utf-8") as f:
    mpk = json.load(f)

torch_latency = float(torch["latency_ms_per_token"])
mpk_latency = float(mpk["latency_ms_per_token"])
batch_size = int(batch)
torch_tokens = torch.get("token_ids", [])
mpk_tokens = mpk.get("token_ids", [])
if torch["generate_length"] != mpk["generate_length"]:
    detail = f"length:{torch['generate_length']}!={mpk['generate_length']}"
else:
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(torch_tokens[:50], mpk_tokens[:50])) if a != b),
        None,
    )
    detail = "match" if mismatch is None else f"token:{mismatch}"
row = [
    mode, batch_size, sin, sout,
    torch["generate_length"], mpk["generate_length"],
    f"{torch_latency:.6f}", f"{mpk_latency:.6f}",
    f"{torch_latency / mpk_latency:.4f}",
    f"{batch_size * 1000.0 / torch_latency:.3f}",
    f"{batch_size * 1000.0 / mpk_latency:.3f}",
    correctness, detail,
]
with open(summary_path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(row)
PY
}

run_correctness_test() {
  local mode="$1"
  local batch="$2"
  local input_length="$3"
  local output_length="$4"
  local torch_output="$5"
  local mpk_output="$6"

  echo "Comparing outputs..."
  if TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
      pytest -q "$ROOT/tests/ci-tests/test_inference_output.py"; then
    write_summary_row "$mode" "$batch" "$input_length" "$output_length" \
      "$torch_output" "$mpk_output" "PASS"
  else
    CORRECTNESS_FAILURES=$((CORRECTNESS_FAILURES + 1))
    write_summary_row "$mode" "$batch" "$input_length" "$output_length" \
      "$torch_output" "$mpk_output" "FAIL"
    echo "Recorded correctness failure; continuing with the remaining matrix points."
  fi
}

run_default() {
  local batch="${1:-1}"
  local point_dir="$ROOT/outputs/qwen3_batch/b${batch}"
  local torch_output="$point_dir/torch_output.json"
  local mpk_output="$point_dir/mpk_output.json"
  local batch_args=(
    --max-num-batched-requests "$batch"
    --max-num-batched-tokens "$batch"
  )

  mkdir -p "$point_dir"
  echo ""
  echo "===== B=${batch} (default prompt and EOS stopping) ====="
  echo "Running Torch baseline..."
  python "$ROOT/demo/qwen3/demo.py" "${batch_args[@]}" \
    --save-tokens "$torch_output"

  echo "Running MPK..."
  python "$ROOT/demo/qwen3/demo.py" --use-mirage "${batch_args[@]}" \
    --save-tokens "$mpk_output"

  run_correctness_test "default_eos" "$batch" "" "" "$torch_output" "$mpk_output"

  echo "Performance comparison..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    python "$ROOT/tests/ci-tests/perf_comparison.py"
}

run_point() {
  local batch="$1"
  local input_length="$2"
  local output_length="$3"
  local total_length=$((input_length + output_length))
  local point_dir="$ROOT/outputs/qwen3_grid/b${batch}_in${input_length}_out${output_length}"
  local torch_output="$point_dir/torch_output.json"
  local mpk_output="$point_dir/mpk_output.json"
  local common_args=(
    --max-num-batched-requests "$batch"
    --max-num-batched-tokens "$batch"
    --input-length "$input_length"
    --max-new-tokens "$output_length"
    --max-seq-length "$total_length"
    --ignore-eos
  )

  mkdir -p "$point_dir"
  echo ""
  echo "===== B=${batch}, S_in=${input_length}, S_out=${output_length} ====="
  echo "Running Torch baseline..."
  python "$ROOT/demo/qwen3/demo.py" "${common_args[@]}" \
    --save-tokens "$torch_output"

  echo "Running MPK..."
  python "$ROOT/demo/qwen3/demo.py" --use-mirage "${common_args[@]}" \
    --save-tokens "$mpk_output"

  run_correctness_test "fixed_length" "$batch" "$input_length" "$output_length" "$torch_output" "$mpk_output"

  echo "Performance comparison..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    python "$ROOT/tests/ci-tests/perf_comparison.py"
}

# Set all three variables to whitespace-separated values to run a benchmark
# matrix.  With none set, preserve the original single Qwen3 CI workflow.
if [[ -z "${S_IN_VALUES:-}" && -z "${S_OUT_VALUES:-}" ]]; then
  RESULTS_FILE="${RESULTS_FILE:-$ROOT/outputs/qwen3_batch/summary.csv}"
  mkdir -p "$(dirname "$RESULTS_FILE")"
  printf '%s\n' 'mode,batch_size,input_length,output_length,torch_generate_length,mpk_generate_length,torch_e2e_ms_per_output_token_incl_prefill,mpk_e2e_ms_per_output_token_incl_prefill,speedup,torch_aggregate_output_tokens_per_s,mpk_aggregate_output_tokens_per_s,correctness,correctness_detail' > "$RESULTS_FILE"
  echo "Summary file: $RESULTS_FILE"
  for batch in ${B_VALUES:-1}; do
    run_default "$batch"
  done
elif [[ -z "${S_IN_VALUES:-}" || -z "${S_OUT_VALUES:-}" ]]; then
  echo "Set both S_IN_VALUES and S_OUT_VALUES when running a grid." >&2
  exit 2
else
  RESULTS_FILE="${RESULTS_FILE:-$ROOT/outputs/qwen3_grid/summary.csv}"
  mkdir -p "$(dirname "$RESULTS_FILE")"
  printf '%s\n' 'mode,batch_size,input_length,output_length,torch_generate_length,mpk_generate_length,torch_e2e_ms_per_output_token_incl_prefill,mpk_e2e_ms_per_output_token_incl_prefill,speedup,torch_aggregate_output_tokens_per_s,mpk_aggregate_output_tokens_per_s,correctness,correctness_detail' > "$RESULTS_FILE"
  echo "Summary file: $RESULTS_FILE"
  for batch in ${B_VALUES:-1}; do
    for input_length in ${S_IN_VALUES}; do
      for output_length in ${S_OUT_VALUES}; do
        run_point "$batch" "$input_length" "$output_length"
      done
    done
  done
fi

echo ""
echo "Completed summary: $RESULTS_FILE"
if (( CORRECTNESS_FAILURES > 0 )); then
  echo "Correctness failed at $CORRECTNESS_FAILURES matrix point(s); see $RESULTS_FILE." >&2
  exit 1
fi
