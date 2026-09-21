"""Archive the unmodified Qwen3 demo's normal and MPK baseline outputs.

Run from the repository root after producing normal_{128,1024}.json and
mpk_{128,1024}.json under tests/benchmarks/baselines/.
"""

import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


BASE = Path("tests/benchmarks/baselines")
INPUT_LENGTHS = (128, 1024)
TOKENS_TO_COMPARE = 30


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment():
    import torch

    gpu = command(
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    )
    return {
        "git_commit": command("git", "rev-parse", "HEAD"),
        "gpu": gpu,
        "cuda_toolkit": command("nvcc", "--version"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "flashinfer": package_version("flashinfer-python"),
        "model": "Qwen/Qwen3-8B",
        "dtype": "bfloat16",
    }


def read_case(backend, length):
    path = BASE / f"{backend}_{length}.json"
    data = json.loads(path.read_text())
    if data["prompt_length"] != length or data["generate_length"] != 128:
        raise ValueError(f"Unexpected prompt or generation length in {path}")
    if len(data["token_ids"]) < TOKENS_TO_COMPARE:
        raise ValueError(f"Fewer than {TOKENS_TO_COMPARE} saved tokens in {path}")
    return data


def main():
    records = {backend: [] for backend in ("normal", "mpk")}
    for length in INPUT_LENGTHS:
        normal = read_case("normal", length)
        mpk = read_case("mpk", length)
        first_mismatch = next(
            (i for i, pair in enumerate(zip(normal["token_ids"], mpk["token_ids"]))
             if pair[0] != pair[1]),
            None,
        )
        matches_30 = normal["token_ids"][:TOKENS_TO_COMPARE] == mpk["token_ids"][:TOKENS_TO_COMPARE]
        for backend, data in (("normal", normal), ("mpk", mpk)):
            records[backend].append({
                "batch_size": 1,
                "input_tokens": length,
                "output_tokens": data["generate_length"],
                "greedy": True,
                "ignore_eos": True,
                "saved_generated_tokens": len(data["token_ids"]),
                "first_30_tokens_match": matches_30,
                "first_mismatch_in_saved_tokens_zero_based": first_mismatch,
                "reported_latency_ms_per_token": data["latency_ms_per_token"],
                "source_json": f"{backend}_{length}.json",
                "source_log": f"{backend}_{length}.log",
            })
        print(f"input={length}: first 30 match={matches_30}; saved-token first mismatch={first_mismatch}")
        if not matches_30:
            raise ValueError(f"First {TOKENS_TO_COMPARE} tokens differ for input={length}")

    meta = environment()
    for backend, cases in records.items():
        output = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "environment": meta,
            "correctness_scope": "First 30 generated tokens only",
            "latency_note": (
                "Original demo-reported latency; normal and MPK use different "
                "timing regions. Do not calculate a speedup from these values."
            ),
            "batch_8_status": "Not measured: current normal demo is single-request only",
            "cases": cases,
        }
        path = BASE / f"qwen3_{backend}_baseline.json"
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
