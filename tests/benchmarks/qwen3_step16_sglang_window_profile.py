"""Run matching early, middle, and late SGLang decode profiles."""

import argparse
import csv
import gzip
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_sglang_profile.py"
S_OUT = 128
WINDOWS = (("early", 0, 9), ("middle", 59, 9), ("late", 118, 9))
CASES = (
    ("short", 1, 128),
    ("short", 32, 128),
    ("long_context", 1, 1024),
    ("long_context", 32, 1024),
)


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
            return "timeout"
    return "" if code == 0 else f"exit code {code}"


def load_last_jsonl(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty result file: {path}")
    return json.loads(lines[-1])


def matching_step14(step14_dir, model, case, batch):
    path = step14_dir / f"{safe(model)}_{case}_b{batch}.jsonl"
    return path, load_last_jsonl(path)


def find_trace(case_dir, prefix):
    candidates = sorted(
        list(case_dir.glob(prefix.name + "*.trace.json.gz"))
        + list(case_dir.glob(prefix.name + "*.trace.json")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No profiler trace matching {prefix.name} in {case_dir}")
    return candidates[-1]


def command(args, batch, s_in, result, prefix, start, steps):
    return [
        sys.executable, "-m", "sglang.benchmark.one_batch",
        "--model-path", args.model, "--tp-size", "1", "--dtype", "bfloat16",
        "--batch-size", str(batch), "--input-len", str(s_in),
        "--output-len", str(S_OUT), "--run-name", prefix.name,
        "--result-filename", str(result), "--profile",
        "--profile-stage", "decode", "--profile-activities", "GPU",
        "--profile-prefix", str(prefix), "--profile-start-step", str(start),
        "--profile-steps", str(steps),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--step14-sglang-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=sorted({item[0] + "_b" + str(item[1]) for item in CASES}),
        default=None,
    )
    parser.add_argument(
        "--windows", nargs="+", choices=[item[0] for item in WINDOWS], default=None,
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.step14_sglang_dir = args.step14_sglang_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0
    selected_cases = set(args.cases or [item[0] + "_b" + str(item[1]) for item in CASES])
    selected_windows = set(args.windows or [item[0] for item in WINDOWS])
    for case, batch, s_in in CASES:
        if case + "_b" + str(batch) not in selected_cases:
            continue
        baseline_path, baseline = matching_step14(
            args.step14_sglang_dir, args.model, case, batch
        )
        for window, start, steps in WINDOWS:
            if window not in selected_windows:
                continue
            stem = f"{safe(args.model)}_{case}_b{batch}_{window}"
            case_dir = args.output_dir / stem
            case_dir.mkdir(exist_ok=True)
            result = case_dir / "result.jsonl"
            log = case_dir / "run.log"
            prefix = case_dir / "sglang_profile"
            if result.exists():
                result.unlink()
            for old in case_dir.glob(prefix.name + "*.trace.json*"):
                old.unlink()
            print(
                f"Profiling {case} B={batch} S_IN={s_in} window={window} "
                f"steps={start + 1}-{start + steps}...",
                flush=True,
            )
            error = run(
                command(args, batch, s_in, result, prefix, start, steps),
                log, args.timeout,
            )
            reasons = [error] if error else []
            profile = measured = None
            trace = None
            if not error:
                try:
                    measured = load_last_jsonl(result)
                    trace = find_trace(case_dir, prefix)
                    summary_dir = case_dir / "summary"
                    code = subprocess.run([
                        sys.executable, str(SUMMARIZER), str(trace),
                        "--output-dir", str(summary_dir),
                    ], cwd=ROOT).returncode
                    if code:
                        reasons.append("profile classification incomplete")
                        summary_path = summary_dir / "profile_summary.json"
                        if summary_path.is_file():
                            profile = json.loads(summary_path.read_text(encoding="utf-8"))
                    else:
                        profile = json.loads(
                            (summary_dir / "profile_summary.json").read_text(encoding="utf-8")
                        )
                except Exception as exc:
                    reasons.append(f"{type(exc).__name__}: {exc}")

            categories = {
                item["category"]: item["time_share"]
                for item in (profile or {}).get("categories", [])
            }
            row = {
                "model": args.model, "case": case, "batch_size": batch,
                "s_in": s_in, "s_out": S_OUT, "window": window,
                "window_start_zero_based": start, "window_steps": steps,
                "status": "failed" if reasons else "completed",
                "step14_baseline": str(baseline_path),
                "profile_trace": str(trace) if trace else None,
                "cuda_kernel_events": (profile or {}).get("cuda_kernel_events"),
                "cuda_kernel_time_ms": (profile or {}).get("cuda_kernel_time_ms"),
                "linear_kernel_share": categories.get("linear"),
                "attention_kernel_share": categories.get("attention"),
                "norm_kernel_share": categories.get("norm"),
                "activation_kernel_share": categories.get("activation"),
                "sampling_kernel_share": categories.get("sampling"),
                "other_kernel_share": categories.get("other"),
                "reason": "; ".join(reasons),
            }
            rows.append(row)
            failed += bool(reasons)
            print(
                f"{stem}: {'PASS' if not reasons else 'FAIL'}; "
                f"kernels={row['cuda_kernel_events']}; "
                f"linear={100 * (categories.get('linear') or 0):.1f}%; "
                f"attention={100 * (categories.get('attention') or 0):.1f}%; "
                f"reason={row['reason'] or '--'}",
                flush=True,
            )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 16, "phase": "sglang_window_profile",
        "status": "passed" if not failed else "failed",
        "windows": [
            {"name": name, "start_zero_based": start, "steps": steps}
            for name, start, steps in WINDOWS
        ],
        "correctness_limitation": (
            "sglang.benchmark.one_batch does not export generated token IDs. "
            "The matching successful Step 14 result is required as a baseline, but "
            "the profiled invocation cannot be first-10-token compared."
        ),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Step 16 SGLang window profile completed with {failed} failed window(s).")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
