"""Combine matching Step 14 MPK and SGLang sweep results."""

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = (
    "model", "case", "batch_size", "s_in", "s_out", "status",
    "mpk_first10_matches", "mpk_prefill_ms", "sglang_prefill_ms",
    "prefill_mpk_over_sglang", "mpk_decode_ms", "sglang_decode_ms",
    "decode_mpk_over_sglang", "mpk_decode_step_ms", "sglang_decode_step_ms",
    "decode_step_mpk_over_sglang", "mpk_decode_tokens_per_second",
    "sglang_decode_tokens_per_second", "reason",
)


def safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def number(record, *names):
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def ratio(lhs, rhs):
    return lhs / rhs if lhs is not None and rhs is not None and rhs > 0 else None


def load_last(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty SGLang result: {path}")
    return records[-1]


def parse_sglang(record, batch, s_out):
    prefill_s = number(record, "prefill_latency", "mean_prefill_latency", "prefill_latency_s", "prefill_time")
    step_s = number(record, "median_decode_latency", "mean_decode_latency", "decode_latency", "decode_latency_s")
    decode_s = number(record, "decode_time", "decode_latency_total")
    if decode_s is None and step_s is not None:
        decode_s = step_s * max(0, s_out - 1)
    prefill_ms = 1000.0 * prefill_s if prefill_s is not None else None
    decode_ms = 1000.0 * decode_s if decode_s is not None else None
    step_ms = 1000.0 * step_s if step_s is not None else None
    throughput = (
        1000.0 * batch * max(0, s_out - 1) / decode_ms
        if decode_ms and s_out > 1 else None
    )
    return prefill_ms, decode_ms, step_ms, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-summary", type=Path, required=True)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mpk = json.loads(args.mpk_summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_passed = mpk.get("status") == "passed"
    for source in mpk["rows"]:
        model, case = source["model"], source["case"]
        batch, s_out = int(source["batch_size"]), int(source["s_out"])
        path = args.sglang_dir / f"{safe(model)}_{case}_b{batch}.jsonl"
        errors = []
        if source.get("status") != "completed":
            errors.append(f"MPK status={source.get('status')}")
        if source.get("minimum_first10_matches") != min(10, s_out):
            errors.append("MPK first-10 gate failed")
        metrics = (None, None, None, None)
        try:
            metrics = parse_sglang(load_last(path), batch, s_out)
            if any(value is None for value in metrics):
                errors.append("SGLang result is missing timing metrics")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
        sg_prefill, sg_decode, sg_step, sg_throughput = metrics
        mpk_prefill = source.get("prefill_ms")
        mpk_decode = source.get("decode_ms")
        mpk_step = source.get("decode_step_ms")
        row = {
            "model": model, "case": case, "batch_size": batch,
            "s_in": source["s_in"], "s_out": s_out,
            "status": "failed" if errors else "completed",
            "mpk_first10_matches": source.get("minimum_first10_matches"),
            "mpk_prefill_ms": mpk_prefill, "sglang_prefill_ms": sg_prefill,
            "prefill_mpk_over_sglang": ratio(mpk_prefill, sg_prefill),
            "mpk_decode_ms": mpk_decode, "sglang_decode_ms": sg_decode,
            "decode_mpk_over_sglang": ratio(mpk_decode, sg_decode),
            "mpk_decode_step_ms": mpk_step, "sglang_decode_step_ms": sg_step,
            "decode_step_mpk_over_sglang": ratio(mpk_step, sg_step),
            "mpk_decode_tokens_per_second": source.get("decode_tokens_per_second"),
            "sglang_decode_tokens_per_second": sg_throughput,
            "reason": "; ".join(errors),
        }
        rows.append(row)
        all_passed &= not errors
        print(
            f"{model} {case} B={batch}: {'PASS' if not errors else 'FAIL'}; "
            f"prefill MPK/SGLang={row['prefill_mpk_over_sglang']}; "
            f"decode MPK/SGLang={row['decode_mpk_over_sglang']}; "
            f"MPK first-10={row['mpk_first10_matches']}/10",
            flush=True,
        )
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 14, "status": "passed" if all_passed else "failed",
        "warmup_runs": 1, "measured_runs": 1,
        "timing_note": "First generated token is prefill; decode contains S_OUT-1 steps.",
        "correctness_note": "MPK requires first-10 equality with Torch. SGLang one_batch does not export token IDs.",
        "rows": rows,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'comparison.csv'}")
    print(f"Wrote {args.output_dir / 'comparison.json'}")
    print(f"Step 14 comparison: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
