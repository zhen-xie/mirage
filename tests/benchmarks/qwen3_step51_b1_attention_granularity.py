"""Tune B=1 MPK attention task granularity with end-to-end decode timing."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
MODES = (("default", None), ("split_64", 64),
         ("split_128", 128), ("split_256", 256))
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


def print_failure_tail(log, lines=40):
    if log.is_file():
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"--- failure tail: {log} ---", flush=True)
        print("\n".join(content[-lines:]), flush=True)


def command(args, output, mode=None, chunk=None, cache=None):
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
            "--mpk-attention", "default" if chunk is None else "split-kv",
            "--mpk-attention-tma-kv",
            "--mpk-fused-rmsnorm-linear", "none",
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
        if chunk is not None:
            cmd += ["--mpk-split-kv-chunk-size", str(chunk)]
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
    for mode, chunk in MODES:
        case_dir = args.output_dir / mode
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_path = case_dir / "warmup.json"
        out_path = case_dir / "tokens.json"
        warm_log = case_dir / "warmup.log"
        run_log = case_dir / "run.log"
        print(f"Warmup/compile mode={mode}...", flush=True)
        warm, error = run(
            command(args, warm_path, mode, chunk, cache), warm_path,
            warm_log, args.timeout,
        )
        data = None
        if warm is not None:
            print(f"Measuring mode={mode}...", flush=True)
            data, error = run(
                command(args, out_path, mode, chunk, cache), out_path,
                run_log, args.timeout,
            )
            if data is None:
                print_failure_tail(run_log)
        else:
            print_failure_tail(warm_log)

        reasons = [error] if error else []
        matches = invalid = incomplete = None
        if data is not None:
            tokens = data.get("token_ids", [])
            matches = sum(a == b for a, b in zip(expected, tokens[:10]))
            invalid = data.get("invalid_token_count")
            incomplete = data.get("generate_length") != S_OUT
            wanted_attention = "default" if chunk is None else "split-kv"
            if matches != 10:
                reasons.append(f"first-10={matches}")
            if invalid != 0:
                reasons.append(f"invalid={invalid}")
            if incomplete:
                reasons.append("incomplete output")
            if data.get("mpk_attention") != wanted_attention:
                reasons.append("wrong attention metadata")
            if chunk is not None and data.get(
                    "mpk_split_kv_chunk_size") != chunk:
                reasons.append("wrong chunk metadata")

        step_ms = data.get("decode_step_time_ms") if data else None
        row = {
            "mode": mode,
            "chunk_size": chunk,
            "status": "failed" if reasons else "passed",
            "first10_matches": matches,
            "invalid_token_count": invalid,
            "incomplete": incomplete,
            "num_chunks": data.get("mpk_split_kv_num_chunks") if data else None,
            "decode_ms": data.get("decode_time_ms") if data else None,
            "decode_step_ms": step_ms,
            "tokens_per_second": 1000.0 / step_ms if step_ms else None,
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        print(
            f"mode={mode}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={matches}; chunks={row['num_chunks']}; "
            f"decode={step_ms} ms/step; throughput={row['tokens_per_second']}",
            flush=True,
        )

    baseline = next((r["decode_step_ms"] for r in rows
                     if r["mode"] == "split_128" and r["status"] == "passed"),
                    None)
    for row in rows:
        row["speedup_vs_split_128"] = (
            baseline / row["decode_step_ms"]
            if baseline and row["status"] == "passed" else None
        )
    valid = [r for r in rows if r["status"] == "passed"]
    best = min(valid, key=lambda r: r["decode_step_ms"])["mode"] if valid else None
    status = "passed" if len(valid) == len(MODES) else "failed"
    summary = {
        "step": 51,
        "phase": "b1_attention_task_granularity",
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
    print(f"Best attention mode: {best}")
    print(f"Step 51 B=1 attention granularity: {status.upper()}")
    raise SystemExit(0 if status == "passed" else 1)


if __name__ == "__main__":
    main()
