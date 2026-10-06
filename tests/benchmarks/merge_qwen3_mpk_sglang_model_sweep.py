"""Merge matched MPK and SGLang model/shape sweep results."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "model", "batch_size", "s_in", "s_out",
    "mpk_prefill_ms", "mpk_decode_ms", "mpk_decode_step_ms",
    "mpk_decode_tokens_per_second", "mpk_status", "mpk_policy",
    "mpk_prefill_backend", "mpk_decode_backend",
    "sglang_prefill_ms", "sglang_decode_ms", "sglang_decode_step_ms",
    "sglang_decode_tokens_per_second",
    "mpk_over_sglang_prefill", "mpk_over_sglang_decode_step",
    "mpk_speedup_vs_sglang_decode", "timing_note",
)


def model_name(folder):
    return json.loads((folder / "model.json").read_text())["model"]


def optional_float(value):
    return float(value) if value not in (None, "") else None


def load_mpk(root):
    records = {}
    for path in sorted(root.glob("*/raw_results.csv")):
        model = model_name(path.parent)
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                if row["policy"] != "decode-only":
                    continue
                key = (model, int(row["batch_size"]), int(row["s_in"]),
                       int(row["s_out"]))
                records[key] = {
                    "prefill": optional_float(row["mean_prefill_ms"]),
                    "decode": optional_float(row["mean_decode_ms"]),
                    "step": optional_float(row["mean_step_latency_ms"]),
                    "throughput": optional_float(
                        row["decode_tokens_per_second"]),
                    "status": row["status"],
                    "policy": row["policy"],
                    "prefill_backend": row["prefill_backend"],
                    "decode_backend": row["decode_backend"],
                }
    return records


def load_sglang(root):
    records = {}
    for path in sorted(root.glob("*/results.jsonl")):
        model = model_name(path.parent)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (model, int(item["batch_size"]), int(item["input_len"]),
                   int(item["output_len"]))
            step = float(item["median_decode_latency"]) * 1000
            records[key] = {
                "prefill": float(item["prefill_latency"]) * 1000,
                "decode": step * max(0, key[3] - 1),
                "step": step,
                "throughput": float(item["median_decode_throughput"]),
            }
    return records


def ratio(left, right):
    return left / right if left is not None and right else None


def encode_keys(keys):
    names = ("model", "batch_size", "s_in", "s_out")
    return [dict(zip(names, key)) for key in keys]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.root / "comparison.csv"
    mpk = load_mpk(args.root / "mpk")
    sglang = load_sglang(args.root / "sglang")
    keys = sorted(set(mpk) | set(sglang))
    rows = []
    for model, batch, s_in, s_out in keys:
        key = (model, batch, s_in, s_out)
        m = mpk.get(key, {})
        s = sglang.get(key, {})
        rows.append({
            "model": model, "batch_size": batch, "s_in": s_in,
            "s_out": s_out,
            "mpk_prefill_ms": m.get("prefill"),
            "mpk_decode_ms": m.get("decode"),
            "mpk_decode_step_ms": m.get("step"),
            "mpk_decode_tokens_per_second": m.get("throughput"),
            "mpk_status": m.get("status", "missing"),
            "mpk_policy": m.get("policy"),
            "mpk_prefill_backend": m.get("prefill_backend"),
            "mpk_decode_backend": m.get("decode_backend"),
            "sglang_prefill_ms": s.get("prefill"),
            "sglang_decode_ms": s.get("decode"),
            "sglang_decode_step_ms": s.get("step"),
            "sglang_decode_tokens_per_second": s.get("throughput"),
            "mpk_over_sglang_prefill": ratio(
                m.get("prefill"), s.get("prefill")),
            "mpk_over_sglang_decode_step": ratio(
                m.get("step"), s.get("step")),
            "mpk_speedup_vs_sglang_decode": ratio(
                s.get("decode"), m.get("decode")),
            "timing_note": (
                "Mirage decode-only uses NORMAL prefill and MPK decode with "
                "CUDA phase events; SGLang uses one_batch wall-clock prefill "
                "and median decode latency"
            ),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    paired = [key for key in keys if key in mpk and key in sglang]
    mpk_only = [key for key in keys if key in mpk and key not in sglang]
    sglang_only = [key for key in keys if key in sglang and key not in mpk]
    noncompleted = [key for key, value in mpk.items()
                    if value.get("status") != "completed"]
    coverage = {
        "mpk_cases": len(mpk),
        "sglang_cases": len(sglang),
        "paired_cases": len(paired),
        "mpk_only_cases": encode_keys(mpk_only),
        "sglang_only_cases": encode_keys(sglang_only),
        "noncompleted_mpk_cases": encode_keys(noncompleted),
    }
    coverage_path = output.with_name("coverage.json")
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
    print(f"MPK cases: {len(mpk)}")
    print(f"SGLang cases: {len(sglang)}")
    print(f"Paired cases: {len(paired)}")
    print(f"Noncompleted MPK cases: {len(noncompleted)}")
    print(f"Wrote {output}")
    print(f"Wrote {coverage_path}")


if __name__ == "__main__":
    main()
