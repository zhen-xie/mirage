"""Profile workload-aware MPK attention in early, middle, and late decode."""

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
BATCH, S_IN, S_OUT, MAX_SEQ = 32, 1024, 128, 1280
COMPARE = 10
WINDOWS = (("early", 1, 9), ("middle", 60, 9), ("late", 119, 9))


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


def command(args, output, batch, cache=None, trace=None, window=None):
    cmd = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(S_IN), "--max-seq-length", str(MAX_SEQ),
        "--max-new-tokens", str(S_OUT), "--page-size", str(MAX_SEQ),
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
        _, start, steps = window
        cmd += [
            "--profiling", "--trace-name", str(trace),
            "--profiler-buffer-entries-per-block",
            str(args.profiler_entries_per_block),
            "--profiler-decode-start-step", str(start),
            "--profiler-decode-num-steps", str(steps),
        ]
    return cmd


def token_batches(data):
    return data.get("token_ids_by_request") or [data.get("token_ids", [])]


def category_shares(profile):
    return {
        row["category"]: row["worker_time_share"]
        for row in profile.get("categories", [])
    }


def old_rows(path):
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["window"]: row for row in data.get("rows", [])
        if row.get("model") == "Qwen/Qwen3-8B"
        and row.get("case") == "long_context"
        and int(row.get("batch_size", 0)) == BATCH
        and row.get("status") == "completed"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument("--step15-summary", type=Path)
    parser.add_argument("--step25-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch_output = args.output_dir / "torch.json"
    torch_log = args.output_dir / "torch.log"
    print("Running Torch correctness reference...", flush=True)
    error = run(command(args, torch_output, 1), torch_log, args.timeout)
    if error:
        raise RuntimeError(f"Torch reference failed: {error}; see {torch_log}")
    reference = json.loads(torch_output.read_text(encoding="utf-8"))["token_ids"]

    control_output = args.output_dir / "control.json"
    control_log = args.output_dir / "control.log"
    control_cache = args.output_dir / "cache" / "control"
    print("Running unprofiled workload-aware control...", flush=True)
    error = run(
        command(args, control_output, BATCH, control_cache),
        control_log, args.timeout)
    if error:
        raise RuntimeError(f"MPK control failed: {error}; see {control_log}")
    control = json.loads(control_output.read_text(encoding="utf-8"))
    control_tokens = token_batches(control)
    control_matches = [
        sum(a == b for a, b in zip(reference[:COMPARE], tokens[:COMPARE]))
        for tokens in control_tokens
    ]
    if (
        len(control_tokens) != BATCH
        or min(control_matches, default=0) != COMPARE
        or control.get("mpk_attention") != "default"
    ):
        raise ValueError(
            "Workload-aware control failed: "
            f"requests={len(control_tokens)}, first10="
            f"{min(control_matches, default=None)}, "
            f"attention={control.get('mpk_attention')}"
        )

    previous = old_rows(args.step15_summary)
    rows = []
    failures = 0
    for window_name, start, steps in WINDOWS:
        case_dir = args.output_dir / window_name
        case_dir.mkdir(exist_ok=True)
        output = case_dir / "tokens.json"
        log = case_dir / "run.log"
        trace = case_dir / "mpk_profile"
        cache = args.output_dir / "cache" / window_name
        print(
            f"Profiling workload-aware window={window_name} "
            f"steps={start}-{start + steps - 1}...", flush=True)
        error = run(
            command(
                args, output, BATCH, cache, trace,
                (window_name, start, steps)),
            log, args.timeout)
        reasons = [error] if error else []
        data = profile = None
        first10 = full_matches = invalid = incomplete = None
        if not error:
            data = json.loads(output.read_text(encoding="utf-8"))
            actual = token_batches(data)
            first10_values = [
                sum(a == b for a, b in zip(reference[:COMPARE], value[:COMPARE]))
                for value in actual
            ]
            full_values = [
                sum(a == b for a, b in zip(expected, value))
                for expected, value in zip(control_tokens, actual)
            ]
            first10 = min(first10_values, default=None)
            full_matches = min(full_values, default=None)
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            invalid = sum(invalid_counts) if len(invalid_counts) == BATCH else None
            incomplete = (
                sum(value != S_OUT for value in lengths)
                if len(lengths) == BATCH else None)
            if len(actual) != BATCH or first10 != COMPARE:
                reasons.append(f"minimum first-10={first10}")
            if full_matches != S_OUT:
                reasons.append(f"minimum full matches={full_matches}/{S_OUT}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            if data.get("mpk_attention") != "default":
                reasons.append(f"attention={data.get('mpk_attention')}")

            summary_dir = case_dir / "summary"
            result = subprocess.run([
                sys.executable, str(SUMMARIZER), str(trace) + ".csv",
                "--output-dir", str(summary_dir),
            ], cwd=ROOT)
            if result.returncode:
                reasons.append("profile summary failed")
            else:
                profile = json.loads(
                    (summary_dir / "profile_summary.json").read_text(
                        encoding="utf-8"))

        shares = category_shares(profile or {})
        old = previous.get(window_name, {})
        row = {
            "window": window_name,
            "window_start": start,
            "window_steps": steps,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": first10,
            "minimum_full_matches": full_matches,
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "base_tasks": data.get("mpk_auto_attention_base_tasks") if data else None,
            "target_tasks": data.get("mpk_auto_attention_target_tasks") if data else None,
            "target_splits": data.get("mpk_auto_attention_target_splits") if data else None,
            "paired_events": (profile or {}).get("paired_events"),
            "linear_worker_share": shares.get("linear"),
            "attention_worker_share": shares.get("attention"),
            "activation_worker_share": shares.get("activation"),
            "norm_worker_share": shares.get("norm"),
            "sampling_worker_share": shares.get("sampling"),
            "old_linear_worker_share": old.get("linear_worker_share"),
            "old_attention_worker_share": old.get("attention_worker_share"),
            "old_paired_events": old.get("profile_paired_events"),
            "case_dir": str(case_dir),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        print(
            f"{window_name:6s}: {'PASS' if not reasons else 'FAIL'}; "
            f"first10={first10}; full={full_matches}/{S_OUT}; "
            f"linear={100 * (shares.get('linear') or 0):.1f}%; "
            f"attention={100 * (shares.get('attention') or 0):.1f}%; "
            f"old attention={100 * (old.get('attention_worker_share') or 0):.1f}%",
            flush=True,
        )

    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    step25_row = None
    if args.step25_summary and args.step25_summary.is_file():
        step25 = json.loads(args.step25_summary.read_text(encoding="utf-8"))
        step25_row = next((
            row for row in step25.get("rows", [])
            if int(row.get("batch_size", 0)) == BATCH
            and row.get("mode") == "auto"
        ), None)
    summary = {
        "step": 26,
        "phase": "workload_aware_attention_profile",
        "status": "passed" if not failures else "failed",
        "model": args.model,
        "batch_size": BATCH,
        "s_in": S_IN,
        "s_out": S_OUT,
        "max_seq_length": MAX_SEQ,
        "step25_performance": step25_row,
        "old_profile_source": str(args.step15_summary) if args.step15_summary else None,
        "measurement_note": (
            "Profiler worker-time shares are activity measures and do not equal "
            "decode wall time. Compare category mix and event count between profiles; "
            "use Step 25 for uninstrumented performance."
        ),
        "correctness_gate": (
            "All 32 requests match Torch for the first 10 tokens and the "
            "unprofiled workload-aware control for all 128 tokens."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Step 26 workload-aware profile completed with "
        f"{failures} failed window(s).")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
