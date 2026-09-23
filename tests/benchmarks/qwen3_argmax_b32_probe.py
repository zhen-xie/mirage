"""Isolate the batched MPK argmax path used by Qwen greedy decoding."""

import argparse
from pathlib import Path

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--padded-vocab-size", type=int, default=153600)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_argmax_b32_probe"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    if args.padded_vocab_size % num_workers:
        raise ValueError(
            f"padded vocab {args.padded_vocab_size} is not divisible by "
            f"num_workers={num_workers}"
        )

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

    torch.manual_seed(29)
    logits = torch.randn(
        args.batch_size, args.padded_vocab_size, device=device, dtype=dtype
    )
    # Give every request a known, distinct winner. Padded positions receive a
    # larger value and must still be excluded by the real-vocabulary bound.
    expected = (
        torch.arange(args.batch_size, device=device, dtype=torch.int64) * 4099
        + 137
    ) % args.vocab_size
    rows = torch.arange(args.batch_size, device=device)
    logits[rows, expected] = 50
    logits[:, args.vocab_size :] = 60

    partial_values = torch.full(
        (args.batch_size, num_workers),
        float("nan"),
        device=device,
        dtype=dtype,
    )
    partial_indices = torch.full(
        (args.batch_size, num_workers),
        -1,
        device=device,
        dtype=torch.int64,
    )
    output = torch.full(
        (args.batch_size, 1), -1, device=device, dtype=torch.int64
    )

    logits_dt = kernel.attach_input(logits, name="probe_logits")
    partial_values_dt = kernel.attach_input(
        partial_values, name="probe_partial_values"
    )
    partial_indices_dt = kernel.attach_input(
        partial_indices, name="probe_partial_indices"
    )
    output_dt = kernel.attach_input(output, name="probe_output")

    kernel.argmax_partial_layer(
        input=logits_dt,
        output=(partial_values_dt, partial_indices_dt),
        grid_dim=(num_workers, 1, 1),
        block_dim=(128, 1, 1),
        vocab_size=args.vocab_size,
    )
    kernel.argmax_reduce_layer(
        input=(partial_values_dt, partial_indices_dt),
        output=output_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"num_workers: {num_workers}")
    print("Compiling isolated B=32 argmax kernel...")
    kernel.compile(output_dir=str(args.output_dir))
    print("Running isolated B=32 argmax kernel...")
    kernel()
    torch.cuda.synchronize()

    actual = output[:, 0]
    matches = actual == expected
    print(f"matching requests: {matches.sum().item()}/{args.batch_size}")
    print("expected token ids:")
    print(expected.tolist())
    print("actual token ids:")
    print(actual.tolist())
    if not bool(matches.all().item()):
        bad_rows = torch.nonzero(~matches, as_tuple=False).flatten().tolist()
        print("mismatching request ids:")
        print(bad_rows)
        kernel.finalize()
        raise SystemExit(1)

    kernel.finalize()
    print("Batched argmax probe: PASS")


if __name__ == "__main__":
    main()
