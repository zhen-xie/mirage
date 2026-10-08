"""Compare Hopper attention phase counters before and after TMA KV loads."""
import argparse, csv, json
from pathlib import Path


def keyed(rows):
    return {(int(row["batch_size"]), int(row["kv_length"])): row
            for row in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    old = keyed(json.loads(a.baseline.read_text())["rows"])
    new = keyed(json.loads(a.candidate.read_text())["rows"])
    rows = []; failures = 0
    metrics = (
        "producer_wait_cycles_per_task",
        "producer_load_cycles_per_tile",
        "consumer_ready_wait_cycles_per_task",
        "consumer_compute_cycles_per_tile",
    )
    for key in sorted(new):
        baseline = old.get(key); candidate = new[key]; reasons = []
        if baseline is None:
            reasons.append("missing Step 39 baseline")
        if candidate.get("status") != "passed":
            reasons.append(candidate.get("reason") or "candidate failed")
        row = {"batch_size": key[0], "kv_length": key[1],
               "status": "failed" if reasons else "passed"}
        for metric in metrics:
            before = baseline.get(metric) if baseline else None
            after = candidate.get(metric)
            row[f"baseline_{metric}"] = before
            row[f"tma_{metric}"] = after
            row[f"ratio_{metric}"] = (
                after / before if before not in (None, 0) and after is not None
                else None)
        row["reason"] = "; ".join(reasons)
        rows.append(row); failures += bool(reasons)
        print(
            f"B={key[0]} KV={key[1]}: {row['status'].upper()}; "
            f"producer-wait ratio={row['ratio_producer_wait_cycles_per_task']}; "
            f"consumer-wait ratio={row['ratio_consumer_ready_wait_cycles_per_task']}; "
            f"compute ratio={row['ratio_consumer_compute_cycles_per_tile']}")
    summary = {"step": 42, "phase": "tma_attention_phase_comparison",
               "status": "failed" if failures else "passed", "rows": rows}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Step 42 TMA phase comparison: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__": main()
