"""Run a short correctness-gated MPK profiler health check."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"


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


def run(command, log, timeout):
    with log.open("w", encoding="utf-8") as destination:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=destination, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            raise RuntimeError(f"timeout; see {log}")
    if code:
        raise RuntimeError(f"exit code {code}; see {log}")


def command(args, output, mpk_mode=None):
    result = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", "128", "--max-seq-length", "256",
        "--max-new-tokens", "16", "--page-size", "256",
        "--max-num-pages", "1", "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8", "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mpk_mode is not None:
        result += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "default", "--normal-prefill-attention", "sdpa",
            "--mpk-kernel-cache-dir", str(args.output_dir / f"cache_{mpk_mode}"),
        ]
        if mpk_mode == "profiled":
            result += [
                "--profiling", "--trace-name", str(args.output_dir / "mpk_profile"),
            ]
    return result


def compare(expected, actual):
    expected = expected[:10]
    actual = actual[:10]
    matches = sum(a == b for a, b in zip(expected, actual))
    first_mismatch = next(
        (index for index, (left, right) in enumerate(zip(expected, actual)) if left != right),
        None,
    )
    if len(expected) != len(actual):
        first_mismatch = min(len(expected), len(actual))
    return matches, first_mismatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch_output = args.output_dir / "torch.json"
    plain_output = args.output_dir / "mpk_unprofiled.json"
    mpk_output = args.output_dir / "mpk.json"
    print("Running Torch correctness reference...", flush=True)
    run(command(args, torch_output), args.output_dir / "torch.log", args.timeout)
    print("Running unprofiled MPK decode-only control...", flush=True)
    run(
        command(args, plain_output, "unprofiled"),
        args.output_dir / "mpk_unprofiled.log", args.timeout,
    )
    print("Running profiled MPK decode-only...", flush=True)
    run(
        command(args, mpk_output, "profiled"),
        args.output_dir / "mpk.log", args.timeout,
    )

    torch_data = json.loads(torch_output.read_text(encoding="utf-8"))
    plain_data = json.loads(plain_output.read_text(encoding="utf-8"))
    mpk_data = json.loads(mpk_output.read_text(encoding="utf-8"))
    expected = torch_data["token_ids"]
    plain_tokens = plain_data["token_ids"]
    actual = mpk_data["token_ids"]
    plain_matches, plain_first_mismatch = compare(expected, plain_tokens)
    matches, first_mismatch = compare(expected, actual)
    profile_vs_plain, profile_plain_first_mismatch = compare(plain_tokens, actual)
    print(f"Torch first 10:          {expected[:10]}")
    print(f"Unprofiled MPK first 10: {plain_tokens[:10]}")
    print(f"Profiled MPK first 10:   {actual[:10]}")
    print(
        f"Unprofiled MPK vs Torch: {plain_matches}/10; "
        f"first mismatch={plain_first_mismatch}"
    )
    print(
        f"Profiled MPK vs Torch: {matches}/10; first mismatch={first_mismatch}"
    )
    print(
        f"Profiled vs unprofiled MPK: {profile_vs_plain}/10; "
        f"first mismatch={profile_plain_first_mismatch}"
    )
    if plain_matches != 10:
        raise ValueError(
            f"unprofiled MPK control failed: first-10={plain_matches}/10"
        )
    if matches != 10:
        raise ValueError(
            "profiler changed MPK output: "
            f"profiled-vs-Torch={matches}/10, "
            f"profiled-vs-unprofiled={profile_vs_plain}/10"
        )

    profile_csv = args.output_dir / "mpk_profile.csv"
    summary_dir = args.output_dir / "summary"
    print("Summarizing MPK profiler output...", flush=True)
    subprocess.run([
        sys.executable, str(SUMMARIZER), str(profile_csv),
        "--output-dir", str(summary_dir),
    ], cwd=ROOT, check=True)
    profile = json.loads((summary_dir / "profile_summary.json").read_text(encoding="utf-8"))
    summary = {
        "step": 15,
        "phase": "mpk_profiler_health",
        "status": "passed",
        "model": args.model,
        "batch_size": 1,
        "s_in": 128,
        "s_out": 16,
        "first10_matches": matches,
        "prefill_ms": mpk_data.get("prefill_time_ms"),
        "decode_ms": mpk_data.get("decode_time_ms"),
        "decode_step_ms": mpk_data.get("decode_step_time_ms"),
        "paired_profile_events": profile["paired_events"],
        "profile_categories": [row["category"] for row in profile["categories"]],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Step 15 MPK profiler health check: PASS")


if __name__ == "__main__":
    main()
