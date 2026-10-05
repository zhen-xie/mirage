"""Merge model/shape sweeps produced by run_qwen3_mpk_sglang_sweep.sh."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "model", "batch_size", "s_in", "s_out",
    "mpk_prefill_ms", "mpk_decode_ms", "mpk_decode_step_ms",
    "mpk_decode_tokens_per_second", "mpk_status",
    "sglang_prefill_ms", "sglang_decode_ms", "sglang_decode_step_ms",
    "sglang_decode_tokens_per_second",
    "mpk_over_sglang_prefill", "mpk_over_sglang_decode_step",
    "mpk_speedup_vs_sglang_decode", "timing_note",
)


def model_name(folder):
    path = folder / "model.json"
    return json.loads(path.read_text())["model"]


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
                    "prefill": float(row["mean_prefill_ms"])
                    if row["mean_prefill_ms"] else None,
                    "decode": float(row["mean_decode_ms"])
                    if row["mean_decode_ms"] else None,
                    "step": float(row["mean_step_latency_ms"])
                    if row["mean_step_latency_ms"] else None,
                    "throughput": float(row["decode_tokens_per_second"])
                    if row["decode_tokens_per_second"] else None,
                    "status": row["status"],
                }
    return records


def load_sglang(root):
    records = {}
    for path in sorted(root.glob("*/results.jsonl")):
        model = model_name(path.parent)
        for number, line in enumerate(path.read_text().splitlines(), 1):
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
    for key in keys:
        model, batch, s_in, s_out = key
        m = mpk.get(key, {})
        s = sglang.get(key, {})
        rows.append({
            "model": model, "batch_size": batch, "s_in": s_in, "s_out": s_out,
            "mpk_prefill_ms": m.get("prefill"),
            "mpk_decode_ms": m.get("decode"),
            "mpk_decode_step_ms": m.get("step"),
            "mpk_decode_tokens_per_second": m.get("throughput"),
            "mpk_status": m.get("status", "missing"),
            "sglang_prefill_ms": s.get("prefill"),
            "sglang_decode_ms": s.get("decode"),
            "sglang_decode_step_ms": s.get("step"),
            "sglang_decode_tokens_per_second": s.get("throughput"),
            "mpk_over_sglang_prefill": ratio(m.get("prefill"), s.get("prefill")),
            "mpk_over_sglang_decode_step": ratio(m.get("step"), s.get("step")),
            "mpk_speedup_vs_sglang_decode": ratio(s.get("decode"), m.get("decode")),
            "timing_note": (
                "MPK uses CUDA phase events; SGLang uses one_batch wall-clock "
                "prefill and median decode latency"
            ),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    paired = sum(key in mpk and key in sglang for key in keys)
    print(f"MPK cases: {len(mpk)}")
    print(f"SGLang cases: {len(sglang)}")
    print(f"Paired cases: {paired}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
