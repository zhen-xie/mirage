"""Compare separate and combined TMA K/V readiness barriers."""

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
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--combined-profile", type=Path, required=True)
    parser.add_argument("--baseline-phases", type=Path, required=True)
    parser.add_argument("--combined-phases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, combined = keyed(args.baseline), keyed(args.combined)
    base_prof, combined_prof = keyed(args.baseline_profile), keyed(args.combined_profile)
    base_phase, combined_phase = keyed(args.baseline_phases), keyed(args.combined_phases)
    rows, failures = [], 0
    for key in sorted(base):
        b, g = base[key], combined[key]
        bp, gp = base_prof[key], combined_prof[key]
        bph, gph = base_phase[key], combined_phase[key]
        reasons = []
        for label, row in (("baseline", b), ("combined", g),
                           ("baseline-profile", bp), ("combined-profile", gp),
                           ("baseline-phases", bph), ("combined-phases", gph)):
            if row["status"] != "passed":
                reasons.append(f"{label}: {row.get('reason') or 'failed'}")
        for label, row in (("baseline", b), ("combined", g)):
            if row.get("minimum_first10_matches") != 10:
                reasons.append(f"{label}: first-10={row.get('minimum_first10_matches')}")
            if row.get("invalid_tokens") or row.get("incomplete_requests"):
                reasons.append(f"{label}: invalid or incomplete output")
        row = {
            "batch_size": key[0], "kv_length": key[1],
            "status": "failed" if reasons else "passed",
            "baseline_decode_step_ms": b.get("decode_step_ms"),
            "combined_decode_step_ms": g.get("decode_step_ms"),
            "decode_ratio_combined_baseline": divide(
                g.get("decode_step_ms"), b.get("decode_step_ms")),
            "baseline_attention_task_us": bp.get("mpk_attention_mean_task_us"),
            "combined_attention_task_us": gp.get("mpk_attention_mean_task_us"),
            "attention_task_ratio_combined_baseline": divide(
                gp.get("mpk_attention_mean_task_us"),
                bp.get("mpk_attention_mean_task_us")),
            "baseline_consumer_compute_cycles_per_tile": bph.get(
                "consumer_compute_cycles_per_tile"),
            "combined_consumer_compute_cycles_per_tile": gph.get(
                "consumer_compute_cycles_per_tile"),
            "consumer_compute_ratio_combined_baseline": divide(
                gph.get("consumer_compute_cycles_per_tile"),
                bph.get("consumer_compute_cycles_per_tile")),
            "baseline_consumer_wait_cycles_per_task": bph.get(
                "consumer_ready_wait_cycles_per_task"),
            "combined_consumer_wait_cycles_per_task": gph.get(
                "consumer_ready_wait_cycles_per_task"),
            "consumer_wait_ratio_combined_baseline": divide(
                gph.get("consumer_ready_wait_cycles_per_task"),
                bph.get("consumer_ready_wait_cycles_per_task")),
            "reason": "; ".join(reasons),
        }
        rows.append(row); failures += bool(reasons)
        print(
            f"B={key[0]} KV={key[1]}: {row['status'].upper()}; "
            f"decode ratio={row['decode_ratio_combined_baseline']}; "
            f"task ratio={row['attention_task_ratio_combined_baseline']}; "
            f"consumer-compute ratio={row['consumer_compute_ratio_combined_baseline']}",
            flush=True)
    summary = {
        "step": 46, "phase": "tma_combined_kv_barrier_ablation",
        "status": "failed" if failures else "passed",
        "ratio_definition": "combined / separate; lower is better", "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"Step 46 TMA combined K/V barrier: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
