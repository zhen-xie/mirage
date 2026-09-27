"""Merge Mirage Qwen3 backend results with SGLang one_batch JSONL results."""

import argparse
import csv
import json
from pathlib import Path


OUTPUT_FIELDS = (
    "batch_size",
    "context_length",
    "output_length",
    "backend",
    "prefill_ms",
    "decode_ms",
    "prefill_plus_decode_ms",
    "decode_step_ms",
    "decode_tokens_per_second",
    "generated_tokens_per_second_including_prefill",
    "decode_speedup_vs_sdpa",
    "phase_sum_speedup_vs_sdpa",
    "minimum_first30_matches_vs_sdpa",
    "timing_source",
)


def as_float(value):
    return None if value in (None, "") else float(value)


def read_mirage(path):
    rows = []
    with path.open(newline="") as source:
        for item in csv.DictReader(source):
            rows.append({
                "batch_size": int(item["batch_size"]),
                "context_length": int(item["context_length"]),
                "output_length": int(item["output_length"]),
                "backend": item["backend"],
                "prefill_ms": as_float(item["mean_prefill_ms"]),
                "decode_ms": as_float(item["mean_decode_ms"]),
                "prefill_plus_decode_ms": as_float(
                    item["mean_prefill_plus_decode_ms"]
                ),
                "decode_step_ms": as_float(item["mean_decode_step_ms"]),
                "decode_tokens_per_second": as_float(
                    item["decode_tokens_per_second"]
                ),
                "generated_tokens_per_second_including_prefill": as_float(
                    item["generated_tokens_per_second_including_prefill"]
                ),
                "decode_speedup_vs_sdpa": as_float(
                    item["decode_speedup_vs_sdpa"]
                ),
                "phase_sum_speedup_vs_sdpa": as_float(
                    item["phase_sum_speedup_vs_sdpa"]
                ),
                "minimum_first30_matches_vs_sdpa": item.get(
                    "minimum_first30_matches_vs_sdpa", ""
                ),
                "timing_source": "Mirage CUDA phase events",
            })
    return rows


def read_sglang(path, wanted_cases):
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
        key = (
            int(item["batch_size"]),
            int(item["input_len"]),
            int(item["output_len"]),
        )
        if key in wanted_cases:
            records.setdefault(key, []).append(item)

    rows = []
    for key in sorted(wanted_cases):
        samples = records.get(key, [])
        if not samples:
            print(f"WARNING: SGLang result is missing B={key[0]} "
                  f"S_IN={key[1]} S_OUT={key[2]}")
            continue
        if len(samples) > 1:
            print(f"WARNING: using the last of {len(samples)} SGLang records "
                  f"for B={key[0]} S_IN={key[1]} S_OUT={key[2]}")
        item = samples[-1]
        batch_size, context_length, output_length = key
        prefill_ms = float(item["prefill_latency"]) * 1000
        decode_step_ms = float(item["median_decode_latency"]) * 1000
        decode_ms = decode_step_ms * max(0, output_length - 1)
        total_ms = float(item["total_latency"]) * 1000
        rows.append({
            "batch_size": batch_size,
            "context_length": context_length,
            "output_length": output_length,
            "backend": "sglang",
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_plus_decode_ms": total_ms,
            "decode_step_ms": decode_step_ms,
            "decode_tokens_per_second": float(
                item["median_decode_throughput"]
            ),
            "generated_tokens_per_second_including_prefill": (
                1000 * batch_size * output_length / total_ms
            ),
            "decode_speedup_vs_sdpa": None,
            "phase_sum_speedup_vs_sdpa": None,
            "minimum_first30_matches_vs_sdpa": "",
            "timing_source": "SGLang one_batch wall-clock timing",
        })
    return rows


def add_relative_metrics(rows):
    sdpa = {
        (row["batch_size"], row["context_length"], row["output_length"]): row
        for row in rows
        if row["backend"] == "normal_sdpa"
    }
    for row in rows:
        reference = sdpa.get(
            (row["batch_size"], row["context_length"], row["output_length"])
        )
        if reference is None:
            continue
        if row["decode_ms"]:
            row["decode_speedup_vs_sdpa"] = (
                reference["decode_ms"] / row["decode_ms"]
            )
        if row["prefill_plus_decode_ms"]:
            row["phase_sum_speedup_vs_sdpa"] = (
                reference["prefill_plus_decode_ms"]
                / row["prefill_plus_decode_ms"]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirage-csv", type=Path, required=True)
    parser.add_argument("--sglang-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    mirage_rows = read_mirage(args.mirage_csv)
    wanted_cases = {
        (row["batch_size"], row["context_length"], row["output_length"])
        for row in mirage_rows
    }
    sglang_rows = read_sglang(args.sglang_jsonl, wanted_cases)
    rows = mirage_rows + sglang_rows
    add_relative_metrics(rows)
    rows.sort(key=lambda row: (
        row["batch_size"],
        row["context_length"],
        row["output_length"],
        row["backend"],
    ))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Mirage rows: {len(mirage_rows)}")
    print(f"SGLang rows: {len(sglang_rows)}")
    print(f"Combined rows: {len(rows)}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
