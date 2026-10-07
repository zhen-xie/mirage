"""Combine validated Step 7 MPK data with SGLang one-batch results."""

import argparse
import csv
import json
import re
from pathlib import Path


CSV_FIELDS = (
    "model", "status", "batch_size", "s_in", "s_out",
    "mpk_first10_matches", "mpk_prefill_ms", "sglang_prefill_ms",
    "prefill_mpk_over_sglang", "mpk_decode_ms", "sglang_decode_ms",
    "decode_mpk_over_sglang", "mpk_decode_step_ms",
    "sglang_decode_step_ms", "decode_step_mpk_over_sglang",
    "mpk_decode_tokens_per_second", "sglang_decode_tokens_per_second",
    "sglang_result_path", "reason",
)


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def number(record, *names):
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_last_record(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No JSON records in {path}")
    return records[-1]


def seconds_to_ms(value):
    return None if value is None else 1000.0 * value


def ratio(lhs, rhs):
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)) or rhs <= 0:
        return None
    return lhs / rhs


def parse_sglang(record, s_out):
    prefill_s = number(
        record, "prefill_latency", "mean_prefill_latency",
        "prefill_latency_s", "prefill_time",
    )
    step_s = number(
        record, "median_decode_latency", "mean_decode_latency",
        "decode_latency", "decode_latency_s",
    )
    total_s = number(record, "total_latency", "mean_total_latency")
    decode_s = number(record, "decode_time", "decode_latency_total")
    if decode_s is None and step_s is not None:
        decode_s = step_s * max(0, s_out - 1)
    if decode_s is None and total_s is not None and prefill_s is not None:
        decode_s = max(0.0, total_s - prefill_s)
    throughput = number(
        record, "median_decode_throughput", "mean_decode_throughput",
        "decode_throughput", "output_throughput",
    )
    if throughput is None and step_s is not None and step_s > 0:
        throughput = 1.0 / step_s
    return {
        "prefill_ms": seconds_to_ms(prefill_s),
        "decode_ms": seconds_to_ms(decode_s),
        "decode_step_ms": seconds_to_ms(step_s),
        "decode_tokens_per_second": throughput,
    }


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
    if mpk.get("warmup_runs") != 1 or mpk.get("measured_runs") != 1:
        raise ValueError(
            "MPK summary must use warmup_runs=1 and measured_runs=1"
        )
    s_out = int(mpk["s_out"])

    for mpk_row in mpk["rows"]:
        model = mpk_row["model"]
        path = args.sglang_dir / f"{safe_name(model)}.jsonl"
        errors = []
        if mpk_row.get("status") != "completed":
            errors.append("MPK Step 7 validation did not pass")
        if mpk_row.get("first10_matches") != mpk.get("compare_tokens", 10):
            errors.append("MPK first-10 token gate did not pass")
        sglang = {}
        if not path.is_file():
            errors.append(f"missing SGLang result: {path}")
        else:
            try:
                record = load_last_record(path)
                sglang = parse_sglang(record, s_out)
                missing = [key for key, value in sglang.items() if value is None]
                if missing:
                    errors.append(
                        f"SGLang metrics missing {missing}; available keys={sorted(record)}"
                    )
            except (ValueError, json.JSONDecodeError) as error:
                errors.append(str(error))

        mpk_prefill = mpk_row.get("prefill_time_ms")
        mpk_decode = mpk_row.get("decode_time_ms")
        mpk_step = mpk_row.get("decode_step_time_ms")
        status = "failed" if errors else "completed"
        all_passed &= status == "completed"
        row = {
            "model": model,
            "status": status,
            "batch_size": mpk["batch_size"],
            "s_in": mpk["s_in"],
            "s_out": s_out,
            "mpk_first10_matches": mpk_row.get("first10_matches"),
            "mpk_prefill_ms": mpk_prefill,
            "sglang_prefill_ms": sglang.get("prefill_ms"),
            "prefill_mpk_over_sglang": ratio(mpk_prefill, sglang.get("prefill_ms")),
            "mpk_decode_ms": mpk_decode,
            "sglang_decode_ms": sglang.get("decode_ms"),
            "decode_mpk_over_sglang": ratio(mpk_decode, sglang.get("decode_ms")),
            "mpk_decode_step_ms": mpk_step,
            "sglang_decode_step_ms": sglang.get("decode_step_ms"),
            "decode_step_mpk_over_sglang": ratio(mpk_step, sglang.get("decode_step_ms")),
            "mpk_decode_tokens_per_second": mpk_row.get("decode_tokens_per_second"),
            "sglang_decode_tokens_per_second": sglang.get("decode_tokens_per_second"),
            "sglang_result_path": str(path),
            "reason": "; ".join(errors),
        }
        rows.append(row)
        print(
            f"{model}: {'PASS' if status == 'completed' else 'FAIL'}; "
            f"prefill MPK/SGLang={row['prefill_mpk_over_sglang']}; "
            f"decode-step MPK/SGLang={row['decode_step_mpk_over_sglang']}; "
            f"MPK first-10={row['mpk_first10_matches']}/10",
            flush=True,
        )

    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 8,
        "status": "passed" if all_passed else "failed",
        "timing": {
            "warmup_runs": 1,
            "measured_runs": 1,
            "decode_steps": max(0, s_out - 1),
            "note": "The first output token is attributed to prefill.",
        },
        "correctness": {
            "mpk": "Compared with Torch; first 10 generated tokens must match.",
            "sglang": "Not token-assessed because sglang.benchmark.one_batch does not export generated token IDs.",
        },
        "mpk_summary": str(args.mpk_summary),
        "sglang_dir": str(args.sglang_dir),
        "rows": rows,
    }
    json_path = args.output_dir / "comparison.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Step 8 comparison: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
