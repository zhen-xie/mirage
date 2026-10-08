"""Compare Hopper attention consumer-completion policies."""

import argparse
import csv
import json
import sys
from pathlib import Path

import qwen3_step28_attention_pipeline as common


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
MODES = ("warpgroup-sync", "warp-arrive")
FIELDS = (
    "completion_mode", "status", "minimum_first10_matches",
    "minimum_full_matches_vs_sync", "passing_requests",
    "invalid_token_count", "incomplete_requests", "decode_ms",
    "decode_step_ms", "decode_tokens_per_second", "speedup_vs_sync",
    "cache_status", "kernel_prepare_ms", "output_path", "log_path",
    "reason",
)


def command(args, output, mode=None, cache=None):
    batch = common.BATCH if mode is not None else 1
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(common.S_IN),
        "--max-seq-length", str(common.MAX_SEQ),
        "--max-new-tokens", str(common.S_OUT),
        "--page-size", str(common.MAX_SEQ),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if mode is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-auto-attention-target-tasks", str(args.target_tasks),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", "fifo",
            "--mpk-attention-kv-pipeline-stages", "2",
            "--mpk-attention-consumer-completion", mode,
            "--normal-prefill-attention", "sdpa",
            "--prefill-warmup-runs", "1", "--normal-prefill-cuda-graph",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    return cmd


def write(args, rows):
    baseline = next((
        row["decode_ms"] for row in rows
        if row["completion_mode"] == "warpgroup-sync" and
        row["status"] == "completed"), None)
    for row in rows:
        row["speedup_vs_sync"] = (
            baseline / row["decode_ms"]
            if baseline and row["status"] == "completed" else None)
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    passed = len(rows) == len(MODES) and all(
        row["status"] == "completed" for row in rows)
    summary = {
        "step": 29,
        "phase": "attention_consumer_completion",
        "status": "passed" if passed else "failed",
        "model": args.model,
        "batch_size": common.BATCH,
        "s_in": common.S_IN,
        "s_out": common.S_OUT,
        "pipeline_stages": 2,
        "completion_modes": list(MODES),
        "warmup_runs": 1,
        "measured_runs": 1,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = args.output_dir / "torch.json"
    reference_log = args.output_dir / "torch.log"
    print("Running Torch correctness reference...", flush=True)
    reference, error = common.run(
        command(args, reference_path), reference_path, reference_log,
        args.timeout)
    if reference is None:
        raise RuntimeError(
            f"Torch reference failed: {error}; see {reference_log}")
    expected = reference["token_ids"][:common.COMPARE]

    rows = []
    sync_tokens = None
    failures = 0
    for mode in MODES:
        case_dir = args.output_dir / mode.replace("-", "_")
        cache = case_dir / "cache"
        case_dir.mkdir(exist_ok=True)
        cache.mkdir(exist_ok=True)
        warm_output = case_dir / "warmup.json"
        output = case_dir / "tokens.json"
        warm_log = case_dir / "warmup.log"
        log = case_dir / "run.log"
        print(f"Warmup/compile completion={mode}...", flush=True)
        warm, error = common.run(
            command(args, warm_output, mode, cache), warm_output,
            warm_log, args.timeout)
        data = None
        failure_log = warm_log
        if warm is not None:
            print(f"Measuring completion={mode}...", flush=True)
            data, error = common.run(
                command(args, output, mode, cache), output, log,
                args.timeout)
            failure_log = log

        reasons = [error] if error else []
        matches = []
        full = invalid = incomplete = None
        if data is not None:
            tokens = common.token_batches(data)
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            matches = [
                sum(a == b for a, b in zip(expected, value[:common.COMPARE]))
                for value in tokens
            ]
            invalid = (
                sum(invalid_counts)
                if len(invalid_counts) == common.BATCH else None)
            incomplete = (
                sum(value != common.S_OUT for value in lengths)
                if len(lengths) == common.BATCH else None)
            if (len(tokens) != common.BATCH or
                    min(matches, default=-1) != common.COMPARE):
                reasons.append(
                    f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            if data.get("mpk_attention") != "default":
                reasons.append(f"attention={data.get('mpk_attention')}")
            if data.get("mpk_attention_consumer_completion") != mode:
                reasons.append("wrong completion-mode metadata")
            if mode == "warpgroup-sync":
                sync_tokens = tokens
                full = common.S_OUT
            elif sync_tokens is not None and len(tokens) == common.BATCH:
                full = min(
                    sum(a == b for a, b in zip(base, value))
                    for base, value in zip(sync_tokens, tokens))
                if full != common.S_OUT:
                    reasons.append(
                        f"minimum full matches={full}/{common.S_OUT}")

        decode_ms = data.get("decode_time_ms") if data else None
        steps = data.get("decode_steps") if data else None
        row = {
            "completion_mode": mode,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": min(matches) if matches else None,
            "minimum_full_matches_vs_sync": full,
            "passing_requests": sum(
                value == common.COMPARE for value in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "decode_ms": decode_ms,
            "decode_step_ms": data.get("decode_step_time_ms") if data else None,
            "decode_tokens_per_second": (
                1000.0 * common.BATCH * steps / decode_ms
                if decode_ms and steps else None),
            "speedup_vs_sync": None,
            "cache_status": data.get("mpk_kernel_cache_status") if data else None,
            "kernel_prepare_ms": data.get("mpk_kernel_prepare_time_ms") if data else None,
            "output_path": str(output),
            "log_path": str(failure_log),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        write(args, rows)
        print(
            f"completion={mode}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"full={full}/{common.S_OUT}; "
            f"decode={row['decode_step_ms']} ms/step; "
            f"throughput={row['decode_tokens_per_second']}", flush=True)

    write(args, rows)
    print(
        f"Step 29 attention completion completed with "
        f"{failures} failed case(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
