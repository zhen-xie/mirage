"""Validate workload-aware Hopper TMA KV selection and decode latency."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
CASES = ((8, 128), (8, 1024), (32, 128), (32, 1024))
S_OUT = 128
COMPARE = 10


def run(command, output, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
            return None, "timeout"
    if code or not output.is_file():
        return None, f"exit code {code}" if code else "missing output JSON"
    return json.loads(output.read_text(encoding="utf-8")), ""


def command(args, output, batch, s_in, mode=None, cache=None):
    max_seq = ((s_in + S_OUT + 127) // 128) * 128
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(S_OUT), "--page-size", str(max_seq),
        "--max-num-pages", str(batch if mode else 1),
        "--max-num-batched-requests", str(batch if mode else 1),
        "--max-num-batched-tokens", str(max(8, batch if mode else 1)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if mode:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-auto-attention-target-tasks", str(args.target_tasks),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-kernel-cache-dir", str(cache),
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
        ]
        if mode == "auto":
            cmd.append("--mpk-attention-tma-kv-auto")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    references = {}
    for s_in in sorted({case[1] for case in CASES}):
        output = args.output_dir / f"torch_in{s_in}.json"
        data, error = run(command(args, output, 1, s_in), output,
                          args.output_dir / f"torch_in{s_in}.log", args.timeout)
        if error:
            raise RuntimeError(f"Torch S_IN={s_in}: {error}")
        references[s_in] = data["token_ids"][:COMPARE]

    rows = []
    failures = 0
    for batch, s_in in CASES:
        measured = {}
        for mode in ("off", "auto"):
            case_dir = args.output_dir / f"b{batch}_in{s_in}_{mode}"
            cache = case_dir / "cache"
            case_dir.mkdir(parents=True, exist_ok=True)
            cache.mkdir(exist_ok=True)
            warm_output = case_dir / "warmup.json"
            output = case_dir / "tokens.json"
            print(f"Warmup B={batch} S_IN={s_in} TMA={mode}...", flush=True)
            _, error = run(command(args, warm_output, batch, s_in, mode, cache),
                           warm_output, case_dir / "warmup.log", args.timeout)
            data = None
            if not error:
                print(f"Measure B={batch} S_IN={s_in} TMA={mode}...", flush=True)
                data, error = run(command(args, output, batch, s_in, mode, cache),
                                  output, case_dir / "run.log", args.timeout)
            reasons = [error] if error else []
            matches = []
            if data:
                tokens = data.get("token_ids_by_request", [])
                matches = [sum(a == b for a, b in zip(
                    references[s_in], request[:COMPARE])) for request in tokens]
                invalid = sum(data.get("invalid_token_counts_by_request", []))
                lengths = data.get("generate_lengths_by_request", [])
                incomplete = sum(length != S_OUT for length in lengths)
                if len(tokens) != batch or not matches or min(matches) != COMPARE:
                    reasons.append(f"first-10={min(matches) if matches else None}")
                if invalid or incomplete:
                    reasons.append(f"invalid={invalid}, incomplete={incomplete}")
                expected_tma = mode == "auto" and (
                    data["mpk_attention_tma_kv"] is True)
                wanted_tma = mode == "auto" and (
                    s_in + S_OUT > args.threshold or batch * 8 < args.target_tasks)
                if expected_tma != wanted_tma:
                    reasons.append(
                        f"TMA resolved={expected_tma}, expected={wanted_tma}")
            row = {
                "batch_size": batch, "s_in": s_in, "s_out": S_OUT,
                "mode": mode, "status": "failed" if reasons else "passed",
                "attention": data.get("mpk_attention") if data else None,
                "tma_enabled": data.get("mpk_attention_tma_kv") if data else None,
                "minimum_first10_matches": min(matches) if matches else None,
                "prefill_ms": data.get("prefill_time_ms") if data else None,
                "decode_ms": data.get("decode_time_ms") if data else None,
                "decode_step_ms": data.get("decode_step_time_ms") if data else None,
                "speedup_vs_off": None, "reason": "; ".join(reasons),
            }
            measured[mode] = row
            rows.append(row)
            failures += bool(reasons)
        if measured["off"]["decode_ms"] and measured["auto"]["decode_ms"]:
            measured["auto"]["speedup_vs_off"] = (
                measured["off"]["decode_ms"] / measured["auto"]["decode_ms"])
        print(
            f"B={batch} S_IN={s_in}: auto TMA={measured['auto']['tma_enabled']}; "
            f"speedup={measured['auto']['speedup_vs_off']}; "
            f"status={measured['auto']['status'].upper()}", flush=True)

    summary = {"step": 41, "phase": "workload_aware_tma",
               "status": "passed" if not failures else "failed", "rows": rows}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Step 41 workload-aware TMA: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
