"""Measure MPK RMSNorm+Linear fusion on Qwen3 decode."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
MODES = ("none", "qkv", "qkv-mlp")
S_IN, S_OUT, MAX_SEQ = 1024, 128, 1280


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


def run(command, output, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return None, "timeout"
    if code:
        return None, f"exit code {code}"
    if not output.is_file():
        return None, "missing output JSON"
    return json.loads(output.read_text(encoding="utf-8")), ""


def command(args, output, mode=None, cache=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-new-tokens", str(S_OUT),
        "--max-seq-length", str(MAX_SEQ), "--page-size", str(MAX_SEQ),
        "--max-num-pages", "1", "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8", "--ignore-eos",
        "--save-tokens", str(output),
    ]
    if mode is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", "256",
            "--mpk-auto-attention-target-tasks", "128",
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-attention-tma-kv-auto",
            "--mpk-fused-rmsnorm-linear", mode,
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = args.output_dir / "torch.json"
    print("Running Torch correctness reference...", flush=True)
    reference, error = run(
        command(args, ref_path), ref_path, args.output_dir / "torch.log",
        args.timeout,
    )
    if reference is None:
        raise RuntimeError(f"Torch reference failed: {error}")
    expected = reference["token_ids"][:10]

    rows = []
    for mode in MODES:
        case_dir = args.output_dir / mode
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_path = case_dir / "warmup.json"
        out_path = case_dir / "tokens.json"
        print(f"Warmup/compile fusion={mode}...", flush=True)
        warm, error = run(
            command(args, warm_path, mode, cache), warm_path,
            case_dir / "warmup.log", args.timeout,
        )
        data = None
        if warm is not None:
            print(f"Measuring fusion={mode}...", flush=True)
            data, error = run(
                command(args, out_path, mode, cache), out_path,
                case_dir / "run.log", args.timeout,
            )

        reasons = [error] if error else []
        matches = invalid = incomplete = None
        if data is not None:
            tokens = data.get("token_ids", [])
            matches = sum(a == b for a, b in zip(expected, tokens[:10]))
            invalid = data.get("invalid_token_count")
            incomplete = data.get("generate_length") != S_OUT
            if matches != 10:
                reasons.append(f"first-10={matches}")
            if invalid != 0:
                reasons.append(f"invalid={invalid}")
            if incomplete:
                reasons.append("incomplete output")
            if data.get("mpk_fused_rmsnorm_linear") != mode:
                reasons.append("wrong fusion metadata")

        row = {
            "mode": mode,
            "status": "failed" if reasons else "passed",
            "first10_matches": matches,
            "invalid_token_count": invalid,
            "incomplete": incomplete,
            "decode_ms": data.get("decode_time_ms") if data else None,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "tokens_per_second": (
                1000.0 / data["decode_step_time_ms"]
                if data and data.get("decode_step_time_ms") else None
            ),
            "resolved_attention": data.get("mpk_attention") if data else None,
            "tma_kv": data.get("mpk_attention_tma_kv") if data else None,
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        print(
            f"fusion={mode}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={matches}; decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['tokens_per_second']}", flush=True,
        )

    baseline = next((r["decode_step_ms"] for r in rows
                     if r["mode"] == "none" and r["status"] == "passed"), None)
    for row in rows:
        row["speedup_vs_none"] = (
            baseline / row["decode_step_ms"]
            if baseline and row["status"] == "passed" else None
        )
    valid = [r for r in rows if r["status"] == "passed"]
    best = min(valid, key=lambda r: r["decode_step_ms"])["mode"] if valid else None
    status = "passed" if len(valid) == len(MODES) else "failed"
    summary = {
        "step": 50,
        "phase": "fused_rmsnorm_linear_ablation",
        "status": status,
        "model": args.model,
        "batch_size": 1,
        "s_in": S_IN,
        "s_out": S_OUT,
        "best_mode": best,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best fusion mode: {best}")
    print(f"Step 50 fused RMSNorm+Linear: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
