"""Aggregate repeated MPK and SGLang sweeps for batch sizes 1 through 8."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def number(record, *names):
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_sglang(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty SGLang result: {path}")
    return rows[-1]


def sglang_metrics(record, batch, s_out):
    prefill_s = number(record, "prefill_latency", "mean_prefill_latency",
                       "prefill_latency_s", "prefill_time")
    step_s = number(record, "median_decode_latency", "mean_decode_latency",
                    "decode_latency", "decode_latency_s")
    decode_s = number(record, "decode_time", "decode_latency_total")
    if decode_s is None and step_s is not None:
        decode_s = step_s * max(0, s_out - 1)
    return {
        "prefill_ms": 1000 * prefill_s if prefill_s is not None else None,
        "decode_ms": 1000 * decode_s if decode_s is not None else None,
        "decode_step_ms": 1000 * step_s if step_s is not None else None,
        "decode_tokens_per_second": (
            batch * max(0, s_out - 1) / decode_s
            if decode_s and s_out > 1 else None),
    }


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-failed-cases", type=int, default=0,
        help="Allow this many failed matrix cases while preserving their failed rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpk_runs = []
    for repeat in range(1, args.repeats + 1):
        path = args.root_dir / f"mpk_repeat{repeat}/summary.json"
        mpk_runs.append(json.loads(path.read_text(encoding="utf-8")))
    keys = [(row["model"], row["case"], int(row["batch_size"]))
            for row in mpk_runs[0]["rows"]]
    rows, failures = [], 0
    for model, case, batch in keys:
        mpk_samples, sg_samples, reasons = [], [], []
        source = None
        for repeat, summary in enumerate(mpk_runs, 1):
            matches = [row for row in summary["rows"] if
                       (row["model"], row["case"], int(row["batch_size"])) ==
                       (model, case, batch)]
            if len(matches) != 1:
                reasons.append(f"MPK repeat {repeat}: missing or duplicate row")
                continue
            source = matches[0]
            if source["status"] != "completed" or source.get(
                    "minimum_first10_matches") != min(10, int(source["s_out"])):
                reasons.append(f"MPK repeat {repeat}: correctness failed")
            mpk_samples.append(source)
            sg_path = (args.root_dir / f"sglang_repeat{repeat}" /
                       f"{safe(model)}_{case}_b{batch}.jsonl")
            try:
                sg_samples.append(sglang_metrics(
                    load_sglang(sg_path), batch, int(source["s_out"])))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                reasons.append(f"SGLang repeat {repeat}: {error}")
        if source is None:
            failures += 1
            continue
        row = {
            "model": model, "case": case, "batch_size": batch,
            "s_in": source["s_in"], "s_out": source["s_out"],
            "status": "failed" if reasons else "passed",
            "mpk_first10_matches": min(
                (sample.get("minimum_first10_matches", 0) for sample in mpk_samples),
                default=None),
        }
        for metric in ("prefill_ms", "decode_ms", "decode_step_ms",
                       "decode_tokens_per_second"):
            mpk_median, mpk_min, mpk_max = stats(
                [sample.get(metric) for sample in mpk_samples])
            sg_median, sg_min, sg_max = stats(
                [sample.get(metric) for sample in sg_samples])
            row.update({
                f"mpk_{metric}_median": mpk_median,
                f"mpk_{metric}_min": mpk_min, f"mpk_{metric}_max": mpk_max,
                f"sglang_{metric}_median": sg_median,
                f"sglang_{metric}_min": sg_min, f"sglang_{metric}_max": sg_max,
                f"{metric}_mpk_over_sglang": (
                    mpk_median / sg_median
                    if mpk_median is not None and sg_median else None),
            })
        row["reason"] = "; ".join(reasons)
        rows.append(row); failures += bool(reasons)
        print(
            f"{model} {case} B={batch}: {row['status'].upper()}; "
            f"prefill MPK/SGLang={row['prefill_ms_mpk_over_sglang']}; "
            f"decode MPK/SGLang={row['decode_step_ms_mpk_over_sglang']}",
            flush=True)
    within_failure_budget = failures <= args.max_failed_cases
    overall_status = (
        "passed" if failures == 0 else
        "passed_with_exceptions" if within_failure_budget else "failed"
    )
    summary = {
        "step": 47, "phase": "batch_1_8_mpk_sglang_comparison",
        "status": overall_status, "repeats": args.repeats,
        "failed_cases": failures,
        "max_failed_cases": args.max_failed_cases,
        "statistic": "median with min/max range", "rows": rows,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (args.output_dir / "comparison.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(f"Step 47 batch 1-8 comparison: {summary['status'].upper()}")
    raise SystemExit(0 if within_failure_budget else 1)


if __name__ == "__main__":
    main()
