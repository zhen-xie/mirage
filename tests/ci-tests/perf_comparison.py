"""Print Torch vs MPK latency comparison (informational, never fails)."""

import json
import os
import sys

DEFAULT_OUTPUT_DIR = os.path.join("outputs", "qwen3")
TORCH_OUTPUT = os.environ.get(
    "TORCH_OUTPUT", os.path.join(DEFAULT_OUTPUT_DIR, "torch_output.json")
)
MPK_OUTPUT = os.environ.get(
    "MPK_OUTPUT", os.path.join(DEFAULT_OUTPUT_DIR, "mpk_output.json")
)
HYBRID_OUTPUT = os.environ.get("HYBRID_OUTPUT")
MPK_LABEL = os.environ.get("MPK_LABEL", "MPK")


def _load_meta(path: str):
    if not os.path.exists(path):
        print(f"Missing output file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    torch_meta = _load_meta(TORCH_OUTPUT)
    mpk_meta = _load_meta(MPK_OUTPUT)
    if torch_meta is None or mpk_meta is None:
        return

    torch_lat = torch_meta.get("latency_ms_per_token")
    mpk_lat = mpk_meta.get("latency_ms_per_token")
    torch_step = torch_meta.get("batch_step_latency_ms")
    mpk_step = mpk_meta.get("batch_step_latency_ms")
    torch_throughput = torch_meta.get("aggregate_throughput_tokens_per_s")
    mpk_throughput = mpk_meta.get("aggregate_throughput_tokens_per_s")
    torch_len = torch_meta.get("generate_length", "?")
    mpk_len = mpk_meta.get("generate_length", "?")

    if torch_lat is None or mpk_lat is None:
        print("latency_ms_per_token missing in output JSON, skipping comparison")
        return

    speedup = torch_lat / mpk_lat if mpk_lat > 0 else float("inf")

    print("")
    print("==================== Performance Comparison ====================")
    print(f"  Torch:  {torch_step:.3f} ms/batch step, {torch_lat:.3f} ms/token, {torch_throughput:.3f} tokens/s  (generated {torch_len} tokens/request)")
    print(
        f"  {MPK_LABEL}: {mpk_step:.3f} ms/batch step, "
        f"{mpk_lat:.3f} ms/token, {mpk_throughput:.3f} tokens/s  "
        f"(generated {mpk_len} tokens/request), Speedup vs Torch: {speedup:.2f}x"
    )
    if HYBRID_OUTPUT:
        hybrid_meta = _load_meta(HYBRID_OUTPUT)
        if hybrid_meta is not None:
            hybrid_lat = hybrid_meta.get("latency_ms_per_token")
            hybrid_step = hybrid_meta.get("batch_step_latency_ms")
            hybrid_throughput = hybrid_meta.get("aggregate_throughput_tokens_per_s")
            hybrid_len = hybrid_meta.get("generate_length", "?")
            if hybrid_lat is not None:
                hybrid_speedup = (
                    torch_lat / hybrid_lat if hybrid_lat > 0 else float("inf")
                )
                print(
                    f"  Torch prefill + MPK decode: {hybrid_step:.3f} ms/batch step, "
                    f"{hybrid_lat:.3f} ms/token, {hybrid_throughput:.3f} tokens/s  "
                    f"(generated {hybrid_len} tokens/request), "
                    f"Speedup vs Torch: {hybrid_speedup:.2f}x"
                )
    print("===============================================================")


if __name__ == "__main__":
    sys.exit(main())
