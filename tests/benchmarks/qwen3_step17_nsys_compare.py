"""Validate and compare matched Nsight Systems decode captures."""

import argparse
import json
from pathlib import Path


def first10(reference, data):
    expected = reference["token_ids"][:10]
    requests = data.get("token_ids_by_request") or [data.get("token_ids", [])]
    return min(sum(a == b for a, b in zip(expected, actual[:10])) for actual in requests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-profile", type=Path, required=True)
    parser.add_argument("--sglang-profile", type=Path, required=True)
    parser.add_argument("--mpk-output", type=Path, required=True)
    parser.add_argument("--torch-reference", type=Path, required=True)
    parser.add_argument("--step14-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mpk = json.loads(args.mpk_profile.read_text(encoding="utf-8"))
    sglang = json.loads(args.sglang_profile.read_text(encoding="utf-8"))
    mpk_output = json.loads(args.mpk_output.read_text(encoding="utf-8"))
    reference = json.loads(args.torch_reference.read_text(encoding="utf-8"))
    step14 = json.loads(args.step14_comparison.read_text(encoding="utf-8"))
    timing = next(
        row for row in step14["rows"]
        if row["model"] == "Qwen/Qwen3-8B"
        and row["case"] == "long_context"
        and int(row["batch_size"]) == 32
    )
    matches = first10(reference, mpk_output)
    invalid = sum(mpk_output.get("invalid_token_counts_by_request", []))
    lengths = mpk_output.get("generate_lengths_by_request", [])
    incomplete = sum(length != 128 for length in lengths)
    errors = []
    if matches != 10:
        errors.append(f"MPK first-10={matches}/10")
    if invalid:
        errors.append(f"MPK invalid tokens={invalid}")
    if len(lengths) != 32 or incomplete:
        errors.append(f"MPK incomplete requests={incomplete}; count={len(lengths)}")
    mpk_categories = {row["category"] for row in mpk.get("categories", [])}
    if "persistent_kernel" not in mpk_categories:
        errors.append("MPK capture is missing persistent worker/scheduler kernels")
    if not sglang.get("kernel_instances"):
        errors.append("SGLang capture contains no kernels")

    kernel_ratio = mpk["kernel_time_ms"] / sglang["kernel_time_ms"]
    sync_ratio = (
        mpk["cuda_device_synchronize_time_ms"]
        / sglang["cuda_device_synchronize_time_ms"]
    )
    result = {
        "step": 17,
        "status": "passed" if not errors else "failed",
        "model": "Qwen/Qwen3-8B", "batch_size": 32,
        "s_in": 1024, "s_out": 128,
        "mpk_first10_matches": matches,
        "mpk_invalid_token_count": invalid,
        "mpk_incomplete_requests": incomplete,
        "step14_decode_step_mpk_over_sglang": timing["decode_step_mpk_over_sglang"],
        "mpk_kernel_time_ms": mpk["kernel_time_ms"],
        "sglang_kernel_time_ms": sglang["kernel_time_ms"],
        "nsys_kernel_time_mpk_over_sglang": kernel_ratio,
        "nsys_synchronize_wait_mpk_over_sglang": sync_ratio,
        "mpk_kernel_instances": mpk["kernel_instances"],
        "sglang_kernel_instances": sglang["kernel_instances"],
        "mpk_cuda_api_time_ms": mpk["cuda_api_time_ms"],
        "sglang_cuda_api_time_ms": sglang["cuda_api_time_ms"],
        "mpk_categories": mpk["categories"],
        "sglang_categories": sglang["categories"],
        "interpretation": (
            "MPK appears as one persistent kernel, so Nsight cannot split its internal "
            "operators. SGLang launches ordinary kernels. Kernel-time ratio is directly "
            "comparable for the captured decode range; category comparison uses Step 15 "
            "for MPK internals and Nsight kernel names for SGLang."
        ),
        "errors": errors,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"MPK first-10: {matches}/10; invalid={invalid}; incomplete={incomplete}")
    print(f"Step 14 wall-time MPK/SGLang: {timing['decode_step_mpk_over_sglang']:.3f}x")
    print(f"Nsight kernel-time MPK/SGLang: {kernel_ratio:.3f}x")
    print(f"Nsight synchronize-wait MPK/SGLang: {sync_ratio:.3f}x")
    print(f"MPK kernels: {mpk['kernel_instances']}; SGLang kernels: {sglang['kernel_instances']}")
    print(f"Step 17 Nsight comparison: {result['status'].upper()}")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
