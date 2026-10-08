"""Measure default MPK attention task scaling with KV length."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"
BATCH, S_OUT, COMPARE = 32, 128, 10
CASES = ((128, 256), (512, 640))
PROFILE_START, PROFILE_STEPS = 1, 9
HEAD_DIM, Q_PER_KV, DTYPE_BYTES = 128, 4, 2


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
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return "timeout"
    return "" if code == 0 else f"exit code {code}"


def command(args, s_in, max_seq, output, batch, cache=None, trace=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(S_OUT), "--page-size", str(max_seq),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if cache is not None:
        cmd += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-auto-attention-target-tasks", str(args.target_tasks),
            "--mpk-split-kv-chunk-size", "128",
            "--mpk-scheduler-policy", "round-robin",
            "--mpk-worker-policy", "fifo",
            "--normal-prefill-attention", "sdpa",
            "--mpk-kernel-cache-dir", str(cache),
        ]
    if trace is not None:
        cmd += [
            "--profiling", "--trace-name", str(trace),
            "--profiler-buffer-entries-per-block",
            str(args.profiler_entries_per_block),
            "--profiler-decode-start-step", str(PROFILE_START),
            "--profiler-decode-num-steps", str(PROFILE_STEPS),
        ]
    return cmd


def token_batches(data):
    return data.get("token_ids_by_request") or [data.get("token_ids", [])]


def batch_health(data, expected_batch):
    lengths = data.get("generate_lengths_by_request", [])
    invalid_counts = data.get("invalid_token_counts_by_request", [])
    invalid = (
        sum(invalid_counts) if len(invalid_counts) == expected_batch else None)
    incomplete = (
        sum(value != S_OUT for value in lengths)
        if len(lengths) == expected_batch else None)
    return invalid, incomplete


def attention_metrics(profile, representative_kv_len):
    attention = next(
        row for row in profile["categories"]
        if row["category"] == "attention")
    mean_us = attention["mean_us"]
    kv_bytes = 2 * representative_kv_len * HEAD_DIM * DTYPE_BYTES
    flops = 4 * Q_PER_KV * representative_kv_len * HEAD_DIM
    return {
        "attention_events": attention["events"],
        "attention_worker_ms": attention["worker_time_ms"],
        "attention_worker_share": attention["worker_time_share"],
        "attention_mean_us": mean_us,
        "estimated_kv_bytes_per_task": kv_bytes,
        "estimated_effective_kv_gbps_per_task": (
            kv_bytes / (mean_us * 1000.0) if mean_us else None),
        "estimated_attention_gflops_per_task": (
            flops / (mean_us * 1000.0) if mean_us else None),
    }


def regression(rows):
    xs = [row["representative_kv_length"] for row in rows]
    ys = [row["attention_mean_us"] for row in rows]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    residual = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "attention_us_per_kv_token": slope,
        "estimated_fixed_attention_us": intercept,
        "linear_fit_r_squared": 1.0 - residual / total if total else 1.0,
    }


def load_step26(args):
    profile_path = args.step26_dir / "early/summary/profile_summary.json"
    token_path = args.step26_dir / "early/tokens.json"
    summary_path = args.step26_dir / "summary.json"
    if not all(path.is_file() for path in (
            profile_path, token_path, summary_path)):
        raise FileNotFoundError(
            f"Missing Step 26 early profile under {args.step26_dir}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    data = json.loads(token_path.read_text(encoding="utf-8"))
    step26 = json.loads(summary_path.read_text(encoding="utf-8"))
    early = next(row for row in step26["rows"] if row["window"] == "early")
    representative = 1024 + (PROFILE_START - 1) + (PROFILE_STEPS + 1) / 2
    return {
        "s_in": 1024,
        "max_seq_length": 1280,
        "representative_kv_length": representative,
        "status": early["status"],
        "minimum_first10_matches": early["minimum_first10_matches"],
        "minimum_full_matches": early["minimum_full_matches"],
        "invalid_token_count": early["invalid_token_count"],
        "incomplete_requests": early["incomplete_requests"],
        "resolved_attention": data.get("mpk_attention"),
        "base_tasks": data.get("mpk_auto_attention_base_tasks"),
        **attention_metrics(profile, representative),
        "source": str(profile_path),
        "reason": early["reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument("--step26-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.step26_dir = args.step26_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = 0
    for s_in, max_seq in CASES:
        case_dir = args.output_dir / f"in{s_in}"
        case_dir.mkdir(exist_ok=True)
        reference_output = case_dir / "torch.json"
        reference_log = case_dir / "torch.log"
        print(f"Running Torch reference S_IN={s_in}...", flush=True)
        error = run(
            command(args, s_in, max_seq, reference_output, 1),
            reference_log, args.timeout)
        if error:
            raise RuntimeError(
                f"Torch S_IN={s_in} failed: {error}; see {reference_log}")
        reference = json.loads(
            reference_output.read_text(encoding="utf-8"))["token_ids"]

        control_output = case_dir / "control.json"
        control_log = case_dir / "control.log"
        control_cache = case_dir / "cache_control"
        print(f"Running unprofiled control S_IN={s_in}...", flush=True)
        error = run(
            command(
                args, s_in, max_seq, control_output, BATCH, control_cache),
            control_log, args.timeout)
        if error:
            raise RuntimeError(
                f"Control S_IN={s_in} failed: {error}; see {control_log}")
        control = json.loads(control_output.read_text(encoding="utf-8"))
        control_tokens = token_batches(control)
        control_first10 = min((
            sum(a == b for a, b in zip(reference[:COMPARE], row[:COMPARE]))
            for row in control_tokens), default=None)
        control_invalid, control_incomplete = batch_health(control, BATCH)
        if (len(control_tokens) != BATCH or control_first10 != COMPARE or
                control_invalid != 0 or control_incomplete != 0):
            raise ValueError(
                f"Control S_IN={s_in} correctness failed: "
                f"batch={len(control_tokens)}/{BATCH}, "
                f"first-10={control_first10}/{COMPARE}, "
                f"invalid={control_invalid}, incomplete={control_incomplete}")

        output = case_dir / "tokens.json"
        log = case_dir / "run.log"
        trace = case_dir / "mpk_profile"
        cache = case_dir / "cache_profile"
        print(f"Profiling default attention S_IN={s_in}...", flush=True)
        error = run(
            command(args, s_in, max_seq, output, BATCH, cache, trace),
            log, args.timeout)
        reasons = [error] if error else []
        first10 = full = invalid = incomplete = None
        data = profile = None
        if not error:
            data = json.loads(output.read_text(encoding="utf-8"))
            actual = token_batches(data)
            first10 = min((
                sum(a == b for a, b in zip(reference[:COMPARE], row[:COMPARE]))
                for row in actual), default=None)
            full = min((
                sum(a == b for a, b in zip(expected, row))
                for expected, row in zip(control_tokens, actual)), default=None)
            invalid, incomplete = batch_health(data, BATCH)
            if len(actual) != BATCH or first10 != COMPARE:
                reasons.append(f"minimum first-10={first10}")
            if full != S_OUT:
                reasons.append(f"minimum full matches={full}/{S_OUT}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            if data.get("mpk_attention") != "default":
                reasons.append(f"attention={data.get('mpk_attention')}")
            result = subprocess.run([
                sys.executable, str(SUMMARIZER), str(trace) + ".csv",
                "--output-dir", str(case_dir / "summary"),
            ], cwd=ROOT)
            if result.returncode:
                reasons.append("profile summary failed")
            else:
                profile = json.loads(
                    (case_dir / "summary/profile_summary.json").read_text(
                        encoding="utf-8"))

        representative = s_in + (PROFILE_START - 1) + (PROFILE_STEPS + 1) / 2
        metrics = attention_metrics(profile, representative) if profile else {}
        row = {
            "s_in": s_in,
            "max_seq_length": max_seq,
            "representative_kv_length": representative,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": first10,
            "minimum_full_matches": full,
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "base_tasks": data.get("mpk_auto_attention_base_tasks") if data else None,
            **metrics,
            "source": str(case_dir / "summary/profile_summary.json"),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        print(
            f"S_IN={s_in}: {'PASS' if not reasons else 'FAIL'}; "
            f"attention mean={row.get('attention_mean_us')} us; "
            f"estimated KV bandwidth="
            f"{row.get('estimated_effective_kv_gbps_per_task')} GB/s",
            flush=True,
        )

    step26_row = load_step26(args)
    if (step26_row["status"] != "completed" or
            step26_row["minimum_first10_matches"] != COMPARE or
            step26_row["minimum_full_matches"] != S_OUT or
            step26_row["invalid_token_count"] != 0 or
            step26_row["incomplete_requests"] != 0 or
            step26_row["resolved_attention"] != "default"):
        step26_row["status"] = "failed"
        step26_row["reason"] = (
            step26_row["reason"] or
            "Step 26 correctness or default-attention gate failed")
        failures += 1
    rows.append(step26_row)
    rows.sort(key=lambda row: row["s_in"])
    fit = regression([row for row in rows if row["status"] == "completed"])

    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 27,
        "phase": "default_attention_length_scaling",
        "status": "passed" if not failures else "failed",
        "model": args.model,
        "batch_size": BATCH,
        "s_out": S_OUT,
        "profile_window": {
            "start": PROFILE_START, "steps": PROFILE_STEPS,
        },
        "kernel_facts": {
            "q_heads_per_kv_head": Q_PER_KV,
            "head_dim": HEAD_DIM,
            "kv_tile_tokens": 64,
            "kv_copy_width_bytes": 16,
            "pipeline_stages": 2,
            "gqa_reuses_kv_across_query_heads": True,
        },
        "model_fit": fit,
        "measurement_note": (
            "Estimated bytes include one BF16 K read and one BF16 V read per "
            "KV token. The measured task also includes Q/K norm, RoPE, online "
            "softmax, PV accumulation, synchronization, and output storage, so "
            "the reported effective bandwidth is a lower-bound diagnostic."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "Attention scaling: "
        f"fixed={fit['estimated_fixed_attention_us']:.3f} us, "
        f"slope={fit['attention_us_per_kv_token']:.6f} us/token, "
        f"R2={fit['linear_fit_r_squared']:.4f}")
    print(f"Step 27 attention length scaling: {'PASS' if not failures else 'FAIL'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
