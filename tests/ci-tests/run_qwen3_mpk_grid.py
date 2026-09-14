"""Run a fixed-shape Qwen3 MPK performance grid.

Each point measures a batch of equal-length synthetic prompts.  ``S_in`` and
``S_out`` are enforced by setting the prompt length and total sequence length,
with EOS disabled.  Compilation and model-load time are not part of the CUDA
event timing reported by ``run_batch_perf.py``.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POINT_RUNNER = ROOT / "tests" / "ci-tests" / "run_batch_perf.py"


def _values(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a non-empty list of positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=_values, default=[1, 2, 4, 8, 16])
    parser.add_argument("--input-lengths", type=_values, default=[16, 32, 64, 128, 256])
    parser.add_argument("--output-lengths", type=_values, default=[16, 32, 64, 128, 256])
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="outputs/qwen3_grid")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points = list(itertools.product(args.batches, args.input_lengths, args.output_lengths))
    print(f"Running {len(points)} MPK points; results will be written to {output_dir}")

    failures: list[tuple[int, int, int]] = []
    for index, (batch, s_in, s_out) in enumerate(points, start=1):
        print(f"\n===== [{index}/{len(points)}] B={batch}, S_in={s_in}, S_out={s_out} =====", flush=True)
        command = [
            sys.executable,
            str(POINT_RUNNER),
            "--model", args.model,
            "--max-num-batched-requests", str(batch),
            "--max-num-batched-tokens", str(batch),
            "--prompt-length", str(s_in),
            "--max-seq-length", str(s_in + s_out),
            "--ignore-eos",
            "--output-dir", str(output_dir),
        ]
        if args.dry_run:
            print(" ".join(command))
            continue
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            failures.append((batch, s_in, s_out))

    if failures:
        print(f"\nFAILED points: {failures}")
        return 1
    print("\nMPK grid completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
