"""Profile representative MPK cases while retaining first-10 correctness."""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"
COMPARE_TOKENS = 10


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


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
            return None, "timeout"
    if code:
        return None, f"exit code {code}"
    return code, ""


def demo_command(
    args, batch, s_in, max_seq, page_size, output, cache=None, trace=None
):
    command = [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(COMPARE_TOKENS), "--page-size", str(page_size),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)),
        "--ignore-eos", "--save-tokens", str(output),
    ]
    if cache is not None:
        command += [
            "--use-mirage", "--mpk-policy", "decode-only",
            "--mpk-attention", "auto",
            "--mpk-auto-split-kv-threshold", str(args.threshold),
            "--mpk-split-kv-chunk-size", "128",
            "--normal-prefill-attention", "sdpa",
            "--mpk-kernel-cache-dir", str(cache),
            "--profiling", "--trace-name", str(trace),
            "--profiler-buffer-entries-per-block",
            str(args.profiler_entries_per_block),
        ]
    return command


def load_step14(path):
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["model"], row["case"], int(row["batch_size"])): row
        for row in data.get("rows", [])
        if row.get("status") == "completed"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument(
        "--cases", nargs="+",
        choices=("short_b1", "short_b32", "long_context_b32"),
        default=("short_b1", "short_b32", "long_context_b32"),
    )
    parser.add_argument("--step14-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step14 = load_step14(args.step14_summary)
    all_cases = {
        "short_b1": ("short", 1, 128, 256),
        "short_b32": ("short", 32, 128, 256),
        "long_context_b32": ("long_context", 32, 1024, 1152),
    }
    cases = [all_cases[name] for name in args.cases]

    references = {}
    rows = []
    failed = 0
    model_name = safe(args.model)
    for index, (case_name, batch, s_in, page_size) in enumerate(cases, 1):
        # Keep the Step 14 static sequence and attention layout.  The MPK
        # max_generation_length compile parameter stops this profiling run
        # after COMPARE_TOKENS outputs.
        max_seq = page_size
        reference_key = (s_in, max_seq)
        reference_output = args.output_dir / f"reference_in{s_in}.json"
        reference_log = args.output_dir / f"reference_in{s_in}.log"
        if reference_key not in references:
            if not reference_output.is_file():
                print(f"Running Torch reference S_IN={s_in}...", flush=True)
                _, error = run(
                    demo_command(
                        args, 1, s_in, max_seq, page_size, reference_output
                    ),
                    reference_log, args.timeout,
                )
                if error:
                    raise RuntimeError(f"Torch reference failed: {error}; see {reference_log}")
            references[reference_key] = json.loads(
                reference_output.read_text(encoding="utf-8")
            )["token_ids"][:COMPARE_TOKENS]

        stem = f"{model_name}_{case_name}_b{batch}"
        case_dir = args.output_dir / stem
        case_dir.mkdir(exist_ok=True)
        output = case_dir / "tokens.json"
        log = case_dir / "run.log"
        trace = case_dir / "mpk_profile"
        cache = args.output_dir / "cache" / (
            f"{model_name}_b{batch}_seq{max_seq}_page{page_size}"
        )
        cache.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(cases)}] Profiling {case_name} B={batch} "
            f"S_IN={s_in} S_OUT={COMPARE_TOKENS}...",
            flush=True,
        )
        _, error = run(
            demo_command(
                args, batch, s_in, max_seq, page_size, output, cache, trace
            ),
            log, args.timeout,
        )
        reasons = [error] if error else []
        data = None
        matches = []
        invalid = incomplete = None
        profile = None
        if not error:
            data = json.loads(output.read_text(encoding="utf-8"))
            expected = references[reference_key]
            tokens = data.get("token_ids_by_request", [])
            matches = [
                sum(a == b for a, b in zip(expected, value[:COMPARE_TOKENS]))
                for value in tokens
            ]
            lengths = data.get("generate_lengths_by_request", [])
            invalid_counts = data.get("invalid_token_counts_by_request", [])
            invalid = sum(invalid_counts) if len(invalid_counts) == batch else None
            incomplete = sum(value != COMPARE_TOKENS for value in lengths) if len(lengths) == batch else None
            if len(tokens) != batch or not matches or min(matches) != COMPARE_TOKENS:
                reasons.append(f"minimum first-10={min(matches) if matches else None}")
            if invalid != 0:
                reasons.append(f"invalid tokens={invalid}")
            if incomplete != 0:
                reasons.append(f"incomplete requests={incomplete}")
            summary_dir = case_dir / "summary"
            result = subprocess.run([
                sys.executable, str(SUMMARIZER), str(trace) + ".csv",
                "--output-dir", str(summary_dir),
            ], cwd=ROOT)
            if result.returncode:
                reasons.append("profile summary failed")
            else:
                profile = json.loads(
                    (summary_dir / "profile_summary.json").read_text(encoding="utf-8")
                )

        baseline = step14.get((args.model, case_name, batch), {})
        category_shares = {
            item["category"]: item["worker_time_share"]
            for item in (profile or {}).get("categories", [])
        }
        row = {
            "model": args.model,
            "case": case_name,
            "batch_size": batch,
            "s_in": s_in,
            "profiled_s_out": COMPARE_TOKENS,
            "status": "failed" if reasons else "completed",
            "minimum_first10_matches": min(matches) if matches else None,
            "passing_requests": sum(value == COMPARE_TOKENS for value in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
            "resolved_attention": data.get("mpk_attention") if data else None,
            "step14_decode_step_ms": baseline.get("decode_step_ms"),
            "step14_decode_tokens_per_second": baseline.get("decode_tokens_per_second"),
            "profile_paired_events": (profile or {}).get("paired_events"),
            "linear_worker_share": category_shares.get("linear"),
            "attention_worker_share": category_shares.get("attention"),
            "norm_worker_share": category_shares.get("norm"),
            "activation_worker_share": category_shares.get("activation"),
            "sampling_worker_share": category_shares.get("sampling"),
            "case_dir": str(case_dir),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failed += bool(reasons)
        print(
            f"{stem}: {'PASS' if not reasons else 'FAIL'}; "
            f"first-10={row['minimum_first10_matches']}; "
            f"linear={100 * (row['linear_worker_share'] or 0):.1f}%; "
            f"attention={100 * (row['attention_worker_share'] or 0):.1f}%",
            flush=True,
        )

    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 15,
        "phase": "mpk_operator_profile",
        "status": "passed" if not failed else "failed",
        "correctness_gate": "Every request matches Torch for the first 10 generated tokens",
        "performance_source": str(args.step14_summary),
        "profile_note": (
            "Instrumented wall time is intentionally excluded. Worker shares are sums "
            "of task durations across parallel blocks and describe activity composition."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Step 15 MPK operator profile completed with {failed} failed case(s).")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
