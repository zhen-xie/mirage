#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

echo "MIRAGE_HOME=${MIRAGE_HOME}"

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

  echo "Comparing outputs..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    pytest -q "$ROOT/tests/ci-tests/test_inference_output.py"

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

  echo "Comparing outputs..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    pytest -q "$ROOT/tests/ci-tests/test_inference_output.py"

  echo "Performance comparison..."
  TORCH_OUTPUT="$torch_output" MPK_OUTPUT="$mpk_output" \
    python "$ROOT/tests/ci-tests/perf_comparison.py"
}

# Set all three variables to whitespace-separated values to run a benchmark
# matrix.  With none set, preserve the original single Qwen3 CI workflow.
if [[ -z "${S_IN_VALUES:-}" && -z "${S_OUT_VALUES:-}" ]]; then
  for batch in ${B_VALUES:-1}; do
    run_default "$batch"
  done
elif [[ -z "${S_IN_VALUES:-}" || -z "${S_OUT_VALUES:-}" ]]; then
  echo "Set both S_IN_VALUES and S_OUT_VALUES when running a grid." >&2
  exit 2
else
  for batch in ${B_VALUES:-1}; do
    for input_length in ${S_IN_VALUES}; do
      for output_length in ${S_OUT_VALUES}; do
        run_point "$batch" "$input_length" "$output_length"
      done
    done
  done
fi
