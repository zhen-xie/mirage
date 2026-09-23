"""Isolate the Hopper large-batch MPK linear kernel from Qwen inference."""

import argparse
from pathlib import Path

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--output-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_linear_b32_probe"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 16 or args.batch_size > 128 or args.batch_size % 8:
        raise ValueError("batch size must be 8-token aligned and in (16, 128]")
    if args.output_size % 64:
        raise ValueError("output size must be divisible by 64")

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    inputs = torch.randn(
        args.batch_size, args.hidden_size, device=device, dtype=dtype
    ) * 0.1
    weights = torch.randn(
        args.output_size, args.hidden_size, device=device, dtype=dtype
    ) * 0.01
    outputs = torch.full(
        (args.batch_size, args.output_size),
        float("nan"),
        device=device,
        dtype=dtype,
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
    if not 90 <= kernel.target_cc < 100:
        raise RuntimeError(f"This probe requires Hopper, got cc={kernel.target_cc}")

    input_dt = kernel.attach_input(inputs, name="probe_input")
    weight_dt = kernel.attach_input(weights, name="probe_weight")
    output_dt = kernel.attach_input(outputs, name="probe_output")
    kernel.linear_layer(
        input=input_dt,
        weight=weight_dt,
        output=output_dt,
        grid_dim=(args.output_size // 64, 1, 1),
        block_dim=(128, 1, 1),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Compiling isolated large-batch linear kernel...")
    kernel.compile(output_dir=str(args.output_dir))
    print("Running isolated large-batch linear kernel...")
    kernel()
    torch.cuda.synchronize()

    reference = inputs.float() @ weights.float().T
    actual = outputs.float()
    absolute_error = (actual - reference).abs()
    row_max = absolute_error.amax(dim=1)
    row_mean = absolute_error.mean(dim=1)
    finite = bool(torch.isfinite(actual).all().item())
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    ).item()

    print(f"finite output: {finite}")
    print(f"maximum absolute error: {absolute_error.max().item():.6f}")
    print(f"mean absolute error: {absolute_error.mean().item():.6f}")
    print(f"cosine similarity: {cosine:.8f}")
    print("per-request maximum absolute error:")
    print([round(value, 6) for value in row_max.tolist()])
    print("per-request mean absolute error:")
    print([round(value, 6) for value in row_mean.tolist()])

    kernel.finalize()
    if not finite or cosine < 0.99 or absolute_error.mean().item() > 0.1:
        raise SystemExit(1)
    print("Large-batch linear probe: PASS")


if __name__ == "__main__":
    main()
