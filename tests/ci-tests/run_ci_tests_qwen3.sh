#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

echo "MIRAGE_HOME=${MIRAGE_HOME}"
CORRECTNESS_FAILURES=0
TEMP_OUTPUT_DIRS=()
SUMMARY_HEADER='mode,batch_size,input_length,output_length,torch_generate_length,mpk_generate_length,torch_batch_step_ms_incl_prefill,mpk_batch_step_ms_incl_prefill,torch_aggregate_ms_per_output_token,mpk_aggregate_ms_per_output_token,speedup,torch_aggregate_output_tokens_per_s,mpk_aggregate_output_tokens_per_s,correctness,correctness_detail'

cleanup_output_dir() {
  local temp_dir="$1"
  rm -f -- "$temp_dir/torch_output.json" "$temp_dir/mpk_output.json"
  rmdir -- "$temp_dir" 2>/dev/null || true
}

cleanup_temp_outputs() {
  local temp_dir
  for temp_dir in "${TEMP_OUTPUT_DIRS[@]}"; do
    cleanup_output_dir "$temp_dir"
  done
}
trap cleanup_temp_outputs EXIT

prepare_results_file() {
  mkdir -p "$(dirname "$RESULTS_FILE")"
  if [[ "${RESUME:-1}" == "1" && -s "$RESULTS_FILE" ]]; then
    local existing_header
    existing_header="$(head -n 1 "$RESULTS_FILE" | tr -d '\r')"
    if [[ "$existing_header" != "$SUMMARY_HEADER" ]]; then
      echo "Existing summary has an incompatible format: $RESULTS_FILE" >&2
      echo "Run with RESUME=0 to start a new sweep." >&2
      exit 2
    fi
    echo "Resuming from summary: $RESULTS_FILE"
  else
    printf '%s\n' "$SUMMARY_HEADER" > "$RESULTS_FILE"
    echo "Starting new summary: $RESULTS_FILE"
  fi
}

summary_has_point() {
  local mode="$1"
  local batch="$2"
  local input_length="$3"
  local output_length="$4"
  awk -F, -v mode="$mode" -v batch="$batch" -v sin="$input_length" -v sout="$output_length" \
    'NR > 1 && $1 == mode && $2 == batch && $3 == sin && $4 == sout { found=1; exit } END { exit !found }' \
    "$RESULTS_FILE"
}

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
torch_step = float(torch["batch_step_latency_ms"])
mpk_step = float(mpk["batch_step_latency_ms"])
batch_size = int(batch)
torch_tokens = torch.get("token_ids", [])
mpk_tokens = mpk.get("token_ids", [])
if torch["generate_length"] != mpk["generate_length"]:
    detail = f"length:{torch['generate_length']}!={mpk['generate_length']}"
else:
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(torch_tokens[:10], mpk_tokens[:10])) if a != b),
        None,
    )
    detail = "match" if mismatch is None else f"token:{mismatch}"
row = [
    mode, batch_size, sin, sout,
    torch["generate_length"], mpk["generate_length"],
    f"{torch_step:.6f}", f"{mpk_step:.6f}",
    f"{torch_latency:.6f}", f"{mpk_latency:.6f}",
    f"{torch_latency / mpk_latency:.4f}",
    f"{1000.0 / torch_latency:.3f}",
    f"{1000.0 / mpk_latency:.3f}",
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
  if summary_has_point "default_eos" "$batch" "" ""; then
    echo "Skipping completed point: B=${batch} (default prompt and EOS stopping)"
    return 0
  fi
  local point_dir
  point_dir="$(mktemp -d "${TMPDIR:-/tmp}/mirage-qwen3.XXXXXX")"
  TEMP_OUTPUT_DIRS+=("$point_dir")
  local torch_output="$point_dir/torch_output.json"
  local mpk_output="$point_dir/mpk_output.json"
  local batch_args=(
    --max-num-batched-requests "$batch"
    --max-num-batched-tokens "$batch"
  )

  echo ""
  echo "===== B=${batch} (default prompt and EOS stopping) ====="
  echo "Running Torch baseline..."
  python "$ROOT/demo/qwen3/demo.py" "${batch_args[@]}" \
    --save-tokens "$torch_output" --quiet-token-save

  echo "Running MPK..."
  python "$ROOT/demo/qwen3/demo.py" --use-mirage "${batch_args[@]}" \
    --save-tokens "$mpk_output" --quiet-token-save

  run_correctness_test "default_eos" "$batch" "" "" "$torch_output" "$mpk_output"

  echo "Performance comparison..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    python "$ROOT/tests/ci-tests/perf_comparison.py"
  cleanup_output_dir "$point_dir"
}

run_point() {
  local batch="$1"
  local input_length="$2"
  local output_length="$3"
  if summary_has_point "fixed_length" "$batch" "$input_length" "$output_length"; then
    echo "Skipping completed point: B=${batch}, S_in=${input_length}, S_out=${output_length}"
    return 0
  fi
  local total_length=$((input_length + output_length))
  # The Hopper paged-attention kernel requires page size to be a multiple of
  # its tile size.  One page per request also matches the Torch reference
  # cache mapping, which uses request index as page index.
  local auto_page_size=$(( ((total_length + 127) / 128) * 128 ))
  local page_size="${PAGE_SIZE:-$auto_page_size}"
  local max_num_pages="${MAX_NUM_PAGES:-$batch}"
  # Hopper linear_swapAB supports at most 16 active tokens in one kernel
  # invocation.  The persistent scheduler can still serve more requests by
  # processing them in groups of up to this token budget.
  local auto_batched_tokens="$batch"
  if (( auto_batched_tokens > 16 )); then
    auto_batched_tokens=16
  fi
  local max_batched_tokens="${MAX_BATCHED_TOKENS:-$auto_batched_tokens}"
  if (( page_size < total_length || page_size % 128 != 0 )); then
    echo "PAGE_SIZE must be >= S_in + S_out and divisible by 128; got $page_size for total length $total_length." >&2
    return 2
  fi
  if (( max_num_pages < batch )); then
    echo "MAX_NUM_PAGES must be >= batch size; got $max_num_pages for B=$batch." >&2
    return 2
  fi
  if (( max_batched_tokens < 1 || max_batched_tokens > 16 )); then
    echo "MAX_BATCHED_TOKENS must be in [1, 16] for the Hopper Qwen3 kernels; got $max_batched_tokens." >&2
    return 2
  fi
  local point_dir
  point_dir="$(mktemp -d "${TMPDIR:-/tmp}/mirage-qwen3.XXXXXX")"
  TEMP_OUTPUT_DIRS+=("$point_dir")
  local torch_output="$point_dir/torch_output.json"
  local mpk_output="$point_dir/mpk_output.json"
  local common_args=(
    --max-num-batched-requests "$batch"
    --max-num-batched-tokens "$max_batched_tokens"
    --max-num-pages "$max_num_pages"
    --page-size "$page_size"
    --input-length "$input_length"
    --max-new-tokens "$output_length"
    --max-seq-length "$total_length"
    --ignore-eos
  )

  echo ""
  echo "===== B=${batch}, S_in=${input_length}, S_out=${output_length}, active_tokens=${max_batched_tokens}, page_size=${page_size}, pages=${max_num_pages} ====="
  echo "Running Torch baseline..."
  python "$ROOT/demo/qwen3/demo.py" "${common_args[@]}" \
    --save-tokens "$torch_output" --quiet-token-save

  echo "Running MPK..."
  python "$ROOT/demo/qwen3/demo.py" --use-mirage "${common_args[@]}" \
    --save-tokens "$mpk_output" --quiet-token-save

  run_correctness_test "fixed_length" "$batch" "$input_length" "$output_length" "$torch_output" "$mpk_output"

  echo "Performance comparison..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    python "$ROOT/tests/ci-tests/perf_comparison.py"
  cleanup_output_dir "$point_dir"
}

# Set all three variables to whitespace-separated values to run a benchmark
# matrix.  With none set, preserve the original single Qwen3 CI workflow.
if [[ -z "${S_IN_VALUES:-}" && -z "${S_OUT_VALUES:-}" ]]; then
  RESULTS_FILE="${RESULTS_FILE:-$ROOT/outputs/qwen3_batch/summary.csv}"
  prepare_results_file
  for batch in ${B_VALUES:-1}; do
    run_default "$batch"
  done
elif [[ -z "${S_IN_VALUES:-}" || -z "${S_OUT_VALUES:-}" ]]; then
  echo "Set both S_IN_VALUES and S_OUT_VALUES when running a grid." >&2
  exit 2
else
  RESULTS_FILE="${RESULTS_FILE:-$ROOT/outputs/qwen3_grid/summary.csv}"
  prepare_results_file
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
TOTAL_CORRECTNESS_FAILURES="$(awk -F, 'NR > 1 && $14 == "FAIL" { count++ } END { print count + 0 }' "$RESULTS_FILE")"
if (( TOTAL_CORRECTNESS_FAILURES > 0 )); then
  echo "Correctness failed at $TOTAL_CORRECTNESS_FAILURES matrix point(s); see $RESULTS_FILE." >&2
  exit 1
fi
