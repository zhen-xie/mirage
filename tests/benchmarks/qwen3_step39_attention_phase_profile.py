"""Summarize Hopper attention producer/consumer phase counters."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-sizes", default="8 32")
    parser.add_argument("--kv-lengths", default="128 1024")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    for batch in map(int, args.batch_sizes.split()):
        for kv in map(int, args.kv_lengths.split()):
            path = args.candidate_dir / f"b{batch}_kv{kv}/tokens.json"
            data = json.loads(path.read_text())
            c = data.get("mpk_attention_phase_counters")
            reasons = []
            if not c or len(c) < 8:
                reasons.append("missing attention phase counters")
                c = [0] * 8
            producer_tasks, producer_tiles = c[4], c[5]
            consumer_tasks, consumer_tiles = c[6], c[7]
            if not producer_tasks or not consumer_tasks or not producer_tiles:
                reasons.append("empty attention phase counters")
            row = {
                "batch_size": batch,
                "kv_length": kv,
                "status": "failed" if reasons else "passed",
                "producer_wait_cycles_per_task": c[0] / producer_tasks if producer_tasks else None,
                "producer_load_cycles_per_tile": c[1] / producer_tiles if producer_tiles else None,
                "consumer_ready_wait_cycles_per_task": c[2] / consumer_tasks if consumer_tasks else None,
                "consumer_compute_cycles_per_tile": c[3] / consumer_tiles if consumer_tiles else None,
                "producer_tasks": producer_tasks,
                "producer_tiles": producer_tiles,
                "consumer_tasks": consumer_tasks,
                "consumer_tiles": consumer_tiles,
                "reason": "; ".join(reasons),
            }
            rows.append(row)
            failures += bool(reasons)
            print(
                f"B={batch} KV={kv}: {row['status'].upper()}; "
                f"producer wait/task={row['producer_wait_cycles_per_task']}; "
                f"load/tile={row['producer_load_cycles_per_tile']}; "
                f"consumer wait/task={row['consumer_ready_wait_cycles_per_task']}; "
                f"compute/tile={row['consumer_compute_cycles_per_tile']}")
    summary = {
        "step": 39,
        "phase": "attention_pipeline_phase_profile",
        "status": "failed" if failures else "passed",
        "counter_units": "GPU clock64 cycles",
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Step 39 attention phase profile: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
