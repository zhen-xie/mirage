"""Validate Step 2 MPK kernel cache correctness and reuse."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
S_IN = 128
S_OUT = 128
COMPARE_TOKENS = 10


def terminate(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(args, name, use_mirage=False):
    output = args.output_dir / f"{name}.json"
    log = args.output_dir / f"{name}.log"
    command = [
        sys.executable, str(DEMO),
        "--model", args.model,
        "--input-length", str(S_IN),
        "--max-seq-length", str(S_IN + S_OUT),
        "--max-new-tokens", str(S_OUT),
        "--page-size", str(S_IN + S_OUT),
        "--max-num-pages", "1",
        "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8",
        "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if use_mirage:
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-kernel-cache-dir", str(args.cache_dir),
        ]
    print(f"Running {name}...", flush=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            raise RuntimeError(f"{name} exceeded {args.timeout}s; see {log}")
    wall_time = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"{name} exited {returncode}; see {log}")
    if not output.is_file():
        raise RuntimeError(f"{name} did not create {output}")
    return json.loads(output.read_text()), wall_time


def validate(data, name, reference, expected_cache_status=None):
    errors = []
    tokens = data.get("token_ids", [])
    if data.get("generate_length") != S_OUT:
        errors.append(f"generate_length={data.get('generate_length')} != {S_OUT}")
    if data.get("invalid_token_count") != 0:
        errors.append(f"invalid_token_count={data.get('invalid_token_count')}")
    matches = sum(a == b for a, b in zip(reference[:10], tokens[:10]))
    if len(tokens) < COMPARE_TOKENS or matches != COMPARE_TOKENS:
        errors.append(f"first-10 matches={matches}/10")
    if expected_cache_status is not None:
        actual = data.get("mpk_kernel_cache_status")
        if actual != expected_cache_status:
            errors.append(
                f"cache status={actual!r}, expected={expected_cache_status!r}"
            )
        prepare_ms = data.get("mpk_kernel_prepare_time_ms")
        if not isinstance(prepare_ms, (int, float)) or prepare_ms < 0:
            errors.append("mpk_kernel_prepare_time_ms is missing or invalid")
    if errors:
        raise ValueError(f"{name}: " + "; ".join(errors))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.output_dir / "cache"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)

    torch_data, torch_wall = run(args, "torch")
    reference = torch_data.get("token_ids", [])
    validate(torch_data, "torch", reference)

    compiled, compiled_wall = run(args, "mpk_compile", use_mirage=True)
    compile_matches = validate(
        compiled, "mpk_compile", reference, "miss_compiled"
    )
    loaded, loaded_wall = run(args, "mpk_load", use_mirage=True)
    load_matches = validate(loaded, "mpk_load", reference, "hit")

    expected_artifacts = (
        "kernel_metadata_rank0.json",
        "task_graph_rank0.json",
        "test_rank0.cu",
    )
    missing = [name for name in expected_artifacts if not (args.cache_dir / name).is_file()]
    launchers = list(args.cache_dir.glob("mpk_launcher_rank0*.so"))
    if missing or not launchers:
        raise ValueError(
            f"Incomplete cache: missing={missing}, launchers={len(launchers)}"
        )

    summary = {
        "step": 2,
        "model": args.model,
        "batch_size": 1,
        "s_in": S_IN,
        "s_out": S_OUT,
        "status": "passed",
        "first10_matches": {
            "mpk_compile_vs_torch": compile_matches,
            "mpk_load_vs_torch": load_matches,
        },
        "wall_time_seconds": {
            "torch": torch_wall,
            "mpk_compile": compiled_wall,
            "mpk_load": loaded_wall,
        },
        "kernel_prepare_time_ms": {
            "mpk_compile": compiled["mpk_kernel_prepare_time_ms"],
            "mpk_load": loaded["mpk_kernel_prepare_time_ms"],
        },
        "cache_dir": str(args.cache_dir),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Step 2 kernel cache validation: PASS")


if __name__ == "__main__":
    main()
