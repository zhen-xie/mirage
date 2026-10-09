"""Summarize detailed B=1 Hopper attention phase and task timings."""

import argparse
import csv
import json
from pathlib import Path


PHASE_NAMES = ("pre_qk", "qk_wgmma", "online_softmax", "pv_wgmma",
               "completion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, default=9)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_dir = args.candidate_dir / "b1_kv1024"
    data = json.loads((case_dir / "tokens.json").read_text(encoding="utf-8"))
    counters = data.get("mpk_attention_phase_counters") or []
    reasons = []
    if len(counters) < 15:
        reasons.append(f"expected at least 15 counters, got {len(counters)}")
        counters = list(counters) + [0] * (15 - len(counters))

    consumer_tiles = counters[7]
    detailed = dict(zip(PHASE_NAMES, counters[8:13]))
    detailed_total = sum(detailed.values())
    rows = []
    for name in PHASE_NAMES:
        cycles = detailed[name]
        rows.append({
            "phase": name,
            "cycles": cycles,
            "cycles_per_consumer_tile": (
                cycles / consumer_tiles if consumer_tiles else None),
            "consumer_compute_share": (
                cycles / counters[3] if counters[3] else None),
            "detailed_phase_share": cycles / detailed_total if detailed_total else None,
        })

    task_totals = {}
    profile_csv = case_dir / "mpk_profile.csv"
    if profile_csv.is_file():
        with profile_csv.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = row["task_type_name"]
                if name in {
                    "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER",
                    "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100",
                }:
                    entry = task_totals.setdefault(name, {"calls": 0, "ns": 0})
                    entry["calls"] += 1
                    entry["ns"] += int(row["duration_ns"])

    attention = task_totals.get("TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER", {})
    merge = task_totals.get("TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100", {})
    if not consumer_tiles:
        reasons.append("empty consumer tile count")
    if not counters[14]:
        reasons.append("empty split-KV merge counter")

    summary = {
        "step": 52,
        "phase": "b1_attention_internal_profile",
        "status": "failed" if reasons else "passed",
        "model": "Qwen/Qwen3-8B",
        "batch_size": 1,
        "kv_length": 1024,
        "profile_steps": args.profile_steps,
        "counter_units": "GPU clock64 cycles summed across consumer warpgroups",
        "consumer_compute_cycles": counters[3],
        "consumer_tiles": consumer_tiles,
        "detailed_cycles": detailed,
        "detailed_cycles_sum": detailed_total,
        "detailed_coverage": detailed_total / counters[3] if counters[3] else None,
        "merge_cycles": counters[13],
        "merge_tasks": counters[14],
        "merge_cycles_per_task": (
            counters[13] / counters[14] if counters[14] else None),
        "attention_worker_ms_per_step": (
            attention.get("ns", 0) / 1e6 / args.profile_steps
            if attention.get("calls") else None),
        "attention_task_calls": attention.get("calls", 0),
        "merge_worker_ms_per_step": (
            merge.get("ns", 0) / 1e6 / args.profile_steps
            if merge.get("calls") else None),
        "merge_task_calls": merge.get("calls", 0),
        "decode_step_ms": data.get("decode_step_time_ms"),
        "rows": rows,
        "reason": "; ".join(reasons),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['phase']:15s}: cycles/tile="
            f"{row['cycles_per_consumer_tile']:.1f}; "
            f"consumer share={100 * row['consumer_compute_share']:.1f}%")
    print(
        "Task-profiler worker time: disabled for this low-overhead run; "
        f"merge cycles/task={summary['merge_cycles_per_task']}")
    print(f"Detailed counter coverage: {100 * summary['detailed_coverage']:.1f}%")
    print(f"Step 52 attention internal profile: {summary['status'].upper()}")
    raise SystemExit(1 if reasons else 0)


if __name__ == "__main__":
    main()
