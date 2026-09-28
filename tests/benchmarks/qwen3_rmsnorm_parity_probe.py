"""Compare the standalone MPK Hopper RMSNorm with Optimized Normal.

The Optimized Normal reference is FlashInfer RMSNorm when available and the
model's explicit FP32-accumulation formula otherwise.  This probe uses the
Qwen3-8B hidden size and BF16 storage used by decode.
"""

import argparse
import json
import os
from pathlib import Path

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--mean-atol", type=float, default=0.001)
    parser.add_argument(
        "--allow-torch-reference",
        action="store_true",
        help="Use the FP32 formula only when FlashInfer cannot be loaded",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_rmsnorm_parity_probe"),
    )
    return parser.parse_args()


def torch_reference(value, weight, eps):
    fp32 = value.float()
    variance = fp32.square().mean(dim=-1, keepdim=True)
    return (fp32 * torch.rsqrt(variance + eps) * weight.float()).to(value.dtype)


def flashinfer_reference(value, weight, eps, allow_torch_reference):
    try:
        import flashinfer

        output = flashinfer.norm.rmsnorm(value, weight, eps=eps)
        return output, "flashinfer"
    except Exception as error:
        if not allow_torch_reference:
            raise RuntimeError(
                "FlashInfer RMSNorm is required for this parity gate"
            ) from error
        print(f"FlashInfer RMSNorm unavailable: {error}")
        return torch_reference(value, weight, eps), "torch_fp32_formula"


def main():
    args = parse_args()
    if args.batch_size < 1 or args.hidden_size < 1:
        raise ValueError("batch-size and hidden-size must be positive")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    value = torch.randn(
        args.batch_size, args.hidden_size, device=device, dtype=dtype
    )
    weight = torch.randn(args.hidden_size, device=device, dtype=dtype)
    output = torch.full_like(value, float("nan"))
    reference, reference_backend = flashinfer_reference(
        value, weight, args.eps, args.allow_torch_reference
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=args.batch_size,
        max_num_batched_requests=args.batch_size,
    )
    kernel = PersistentKernel(**params)
    value_dt = kernel.attach_input(value, name="rmsnorm_input")
    weight_dt = kernel.attach_input(weight, name="rmsnorm_weight")
    output_dt = kernel.attach_input(output, name="rmsnorm_output")
    kernel.rmsnorm_layer(
        input=value_dt,
        weight=weight_dt,
        output=output_dt,
        grid_dim=(args.batch_size, 1, 1),
        block_dim=(128, 1, 1),
        eps=args.eps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Compiling isolated MPK RMSNorm kernel...")
    kernel.compile(output_dir=str(args.output_dir))
    print("Running isolated MPK RMSNorm kernel...")
    kernel()
    torch.cuda.synchronize()

    difference = (output.float() - reference.float()).abs()
    report = {
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "dtype": str(dtype),
        "eps": args.eps,
        "reference_backend": reference_backend,
        "flashinfer_use_cuda_norm": os.environ.get("FLASHINFER_USE_CUDA_NORM"),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item(),
    }
    report["passed"] = (
        report["max_absolute_error"] <= args.atol
        and report["mean_absolute_error"] <= args.mean_atol
    )
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    kernel.finalize()
    if not report["passed"]:
        raise SystemExit(1)
    print("MPK RMSNorm parity probe: PASS")


if __name__ == "__main__":
    main()
