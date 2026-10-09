"""A/B test warp-per-head RMSNorm+RoPE inside Hopper attention."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
MODES = ("baseline", "warp-per-head")
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
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            terminate(process)
            if isinstance(sys.exc_info()[1], KeyboardInterrupt):
                raise
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
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
        if mode == "warp-per-head":
            cmd.append("--mpk-attention-warp-norm")
    return cmd


def tail(log, lines=60):
    if log.is_file():
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"--- failure tail: {log} ---")
        print("\n".join(content[-lines:]))


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
        print(f"Warmup/compile mode={mode}...", flush=True)
        warm, error = run(
            command(args, warm_path, mode, cache), warm_path,
            case_dir / "warmup.log", args.timeout,
        )
        data = None
        if warm is not None:
            print(f"Measuring mode={mode}...", flush=True)
            data, error = run(
                command(args, out_path, mode, cache), out_path,
                case_dir / "run.log", args.timeout,
            )
        if data is None:
            tail(case_dir / ("run.log" if warm is not None else "warmup.log"))

        reasons = [error] if error else []
        matches = invalid = incomplete = None
        if data is not None:
            tokens = data.get("token_ids", [])
            matches = sum(a == b for a, b in zip(expected, tokens[:10]))
            invalid = data.get("invalid_token_count")
            incomplete = data.get("generate_length") != S_OUT
            expected_flag = mode == "warp-per-head"
            if matches != 10:
                reasons.append(f"first-10={matches}")
            if invalid != 0:
                reasons.append(f"invalid={invalid}")
            if incomplete:
                reasons.append("incomplete output")
            if data.get("mpk_attention_warp_norm") != expected_flag:
                reasons.append("wrong warp-norm metadata")

        step_ms = data.get("decode_step_time_ms") if data else None
        row = {
            "mode": mode,
            "status": "failed" if reasons else "passed",
            "first10_matches": matches,
            "invalid_token_count": invalid,
            "incomplete": incomplete,
            "decode_step_ms": step_ms,
            "tokens_per_second": 1000.0 / step_ms if step_ms else None,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "split_kv_chunk_size": (
                data.get("mpk_split_kv_chunk_size") if data else None),
            "tma_kv": data.get("mpk_attention_tma_kv") if data else None,
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        print(
            f"mode={mode}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={matches}; decode={step_ms} ms/step; "
            f"throughput={row['tokens_per_second']}", flush=True,
        )

    base = rows[0]["decode_step_ms"] if rows[0]["status"] == "passed" else None
    candidate = rows[1]["decode_step_ms"] if rows[1]["status"] == "passed" else None
    speedup = base / candidate if base and candidate else None
    status = "passed" if all(r["status"] == "passed" for r in rows) else "failed"
    summary = {
        "step": 53,
        "phase": "warp_per_head_qk_norm_rope",
        "status": status,
        "model": args.model,
        "batch_size": 1,
        "s_in": S_IN,
        "s_out": S_OUT,
        "split_kv_chunk_size": 128,
        "candidate_speedup": speedup,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Warp-per-head speedup: {speedup}")
    print(f"Step 53 warp norm/RoPE: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
