"""Compare the combined K/V readiness barrier against the Step 37 baseline."""

import argparse
import csv
import json
from pathlib import Path


def keyed(rows):
    return {(int(row["batch_size"]), int(row["kv_length"])): row
            for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = keyed(json.loads(args.baseline.read_text())["rows"])
    candidate_data = json.loads(args.candidate.read_text())
    candidate = keyed(candidate_data["rows"])
    rows = []
    failures = 0
    for key in sorted(candidate):
        old = baseline[key]
        new = candidate[key]
        old_us = old["mpk_attention_mean_task_us"]
        new_us = new["mpk_attention_mean_task_us"]
        old_lb = old["mpk_attention_work_lower_bound_ms_per_step"]
        new_lb = new["mpk_attention_work_lower_bound_ms_per_step"]
        reasons = []
        if new["status"] != "passed":
            reasons.append(new.get("reason") or "candidate failed")
        if new["minimum_first10_matches"] != 10:
            reasons.append(
                f"first-10={new['minimum_first10_matches']}/10")
        row = {
            "batch_size": key[0],
            "kv_length": key[1],
            "status": "failed" if reasons else "passed",
            "minimum_first10_matches": new["minimum_first10_matches"],
            "baseline_task_us": old_us,
            "combined_task_us": new_us,
            "task_speedup": old_us / new_us,
            "baseline_work_lower_bound_ms": old_lb,
            "combined_work_lower_bound_ms": new_lb,
            "work_lower_bound_speedup": old_lb / new_lb,
            "baseline_worker_share": old["mpk_attention_worker_share"],
            "combined_worker_share": new["mpk_attention_worker_share"],
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        print(
            f"B={key[0]} KV={key[1]}: "
            f"{'PASS' if not reasons else 'FAIL'}; "
            f"task speedup={row['task_speedup']:.3f}x; "
            f"work speedup={row['work_lower_bound_speedup']:.3f}x")

    target = next(row for row in rows
                  if row["batch_size"] == 32 and row["kv_length"] == 1024)
    summary = {
        "step": 38,
        "phase": "combined_kv_readiness_barrier",
        "status": "passed" if not failures else "failed",
        "target_case": target,
        "rows": rows,
        "acceptance_note": (
            "Correctness is mandatory. Performance remains an ablation: "
            "retain only if the B=32 KV=1024 target improves without a "
            "material short-KV regression."),
    }
    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"Step 38 combined KV barrier: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
