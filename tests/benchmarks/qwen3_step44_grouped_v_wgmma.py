"""Compare serialized and grouped V-WGMMA attention issue."""

import argparse
import csv
import json
from pathlib import Path


def keyed(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(row["batch_size"], row["kv_length"]): row for row in data["rows"]}


def divide(a, b):
    return a / b if a is not None and b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--grouped", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--grouped-profile", type=Path, required=True)
    parser.add_argument("--baseline-phases", type=Path, required=True)
    parser.add_argument("--grouped-phases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, grouped = keyed(args.baseline), keyed(args.grouped)
    base_prof, grouped_prof = keyed(args.baseline_profile), keyed(args.grouped_profile)
    base_phase, grouped_phase = keyed(args.baseline_phases), keyed(args.grouped_phases)
    rows, failures = [], 0
    for key in sorted(base):
        b, g = base[key], grouped[key]
        bp, gp = base_prof[key], grouped_prof[key]
        bph, gph = base_phase[key], grouped_phase[key]
        reasons = []
        for label, row in (("baseline", b), ("grouped", g),
                           ("baseline-profile", bp), ("grouped-profile", gp),
                           ("baseline-phases", bph), ("grouped-phases", gph)):
            if row["status"] != "passed":
                reasons.append(f"{label}: {row.get('reason') or 'failed'}")
        for label, row in (("baseline", b), ("grouped", g)):
            if row.get("minimum_first10_matches") != 10:
                reasons.append(f"{label}: first-10={row.get('minimum_first10_matches')}")
            if row.get("invalid_tokens") or row.get("incomplete_requests"):
                reasons.append(f"{label}: invalid or incomplete output")
        row = {
            "batch_size": key[0], "kv_length": key[1],
            "status": "failed" if reasons else "passed",
            "baseline_decode_step_ms": b.get("decode_step_ms"),
            "grouped_decode_step_ms": g.get("decode_step_ms"),
            "decode_ratio_grouped_baseline": divide(
                g.get("decode_step_ms"), b.get("decode_step_ms")),
            "baseline_attention_task_us": bp.get("mpk_attention_mean_task_us"),
            "grouped_attention_task_us": gp.get("mpk_attention_mean_task_us"),
            "attention_task_ratio_grouped_baseline": divide(
                gp.get("mpk_attention_mean_task_us"),
                bp.get("mpk_attention_mean_task_us")),
            "baseline_consumer_compute_cycles_per_tile": bph.get(
                "consumer_compute_cycles_per_tile"),
            "grouped_consumer_compute_cycles_per_tile": gph.get(
                "consumer_compute_cycles_per_tile"),
            "consumer_compute_ratio_grouped_baseline": divide(
                gph.get("consumer_compute_cycles_per_tile"),
                bph.get("consumer_compute_cycles_per_tile")),
            "baseline_consumer_wait_cycles_per_task": bph.get(
                "consumer_ready_wait_cycles_per_task"),
            "grouped_consumer_wait_cycles_per_task": gph.get(
                "consumer_ready_wait_cycles_per_task"),
            "consumer_wait_ratio_grouped_baseline": divide(
                gph.get("consumer_ready_wait_cycles_per_task"),
                bph.get("consumer_ready_wait_cycles_per_task")),
            "reason": "; ".join(reasons),
        }
        rows.append(row); failures += bool(reasons)
        print(
            f"B={key[0]} KV={key[1]}: {row['status'].upper()}; "
            f"decode ratio={row['decode_ratio_grouped_baseline']}; "
            f"task ratio={row['attention_task_ratio_grouped_baseline']}; "
            f"consumer-compute ratio={row['consumer_compute_ratio_grouped_baseline']}",
            flush=True)
    summary = {
        "step": 44, "phase": "grouped_v_wgmma_ablation",
        "status": "failed" if failures else "passed",
        "ratio_definition": "grouped / serialized; lower is better", "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Step 44 grouped V-WGMMA: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
