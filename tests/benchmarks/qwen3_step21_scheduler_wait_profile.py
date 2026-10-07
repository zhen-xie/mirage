"""Summarize MPK worker dependency waits for a profiled decode window."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MOD = 1 << 32


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else 0.0


def circular_span(rows):
    points = sorted({int(r["begin_ts"]) for r in rows} |
                    {int(r["end_ts"]) for r in rows})
    if len(points) < 2:
        return 0
    gaps = [(points[(i + 1) % len(points)] - points[i]) % MOD
            for i in range(len(points))]
    return MOD - max(gaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--mpk", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch_data = json.loads(args.torch.read_text(encoding="utf-8"))
    mpk = json.loads(args.mpk.read_text(encoding="utf-8"))
    expected = torch_data["token_ids"][:10]
    tokens = mpk.get("token_ids_by_request", [])
    matches = [sum(a == b for a, b in zip(expected, row[:10]))
               for row in tokens]
    lengths = mpk.get("generate_lengths_by_request", [])
    invalid_counts = mpk.get("invalid_token_counts_by_request", [])
    invalid = sum(invalid_counts) if len(invalid_counts) == 32 else None
    incomplete = sum(n != 128 for n in lengths) if len(lengths) == 32 else None

    with args.profile.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    waits = [r for r in rows if r["task_type_name"] == "TASK_GET_EVENT"]
    compute = [r for r in rows if r["task_type_name"] != "TASK_GET_EVENT"]
    wait_ns = sum(int(r["duration_ns"]) for r in waits)
    compute_ns = sum(int(r["duration_ns"]) for r in compute)
    span_ns = circular_span(rows)
    blocks = sorted({int(r["block_idx"]) for r in rows})
    wait_by_block = defaultdict(int)
    compute_by_block = defaultdict(int)
    for row in waits:
        wait_by_block[int(row["block_idx"])] += int(row["duration_ns"])
    for row in compute:
        compute_by_block[int(row["block_idx"])] += int(row["duration_ns"])
    wait_shares = []
    for block in blocks:
        total = wait_by_block[block] + compute_by_block[block]
        wait_shares.append(wait_by_block[block] / total if total else 0.0)

    average_waiting = wait_ns / span_ns if span_ns else None
    average_computing = compute_ns / span_ns if span_ns else None
    accounted = ((wait_ns + compute_ns) / (span_ns * len(blocks))
                 if span_ns and blocks else None)
    correct = (len(tokens) == 32 and matches and min(matches) == 10 and
               invalid == 0 and incomplete == 0)
    passed = correct and bool(waits) and span_ns > 0
    summary = {
        "step": 21,
        "phase": "scheduler_dependency_wait_profile",
        "status": "passed" if passed else "failed",
        "model": "Qwen/Qwen3-8B",
        "batch_size": 32,
        "s_in": 1024,
        "s_out": 128,
        "profile_window": {"start_step": 60, "num_steps": 9},
        "scheduler_policy": mpk.get("mpk_scheduler_policy"),
        "split_kv_chunk_size": mpk.get("mpk_split_kv_chunk_size"),
        "correctness": {
            "minimum_first10_matches": min(matches) if matches else None,
            "passing_requests": sum(m == 10 for m in matches),
            "invalid_token_count": invalid,
            "incomplete_requests": incomplete,
        },
        "profile": {
            "paired_events": len(rows),
            "dependency_wait_events": len(waits),
            "worker_blocks": len(blocks),
            "window_span_ms": span_ns / 1e6,
            "compute_worker_time_ms": compute_ns / 1e6,
            "dependency_wait_worker_time_ms": wait_ns / 1e6,
            "dependency_wait_share_of_accounted_worker_time": (
                wait_ns / (wait_ns + compute_ns)
                if wait_ns + compute_ns else None),
            "average_computing_workers": average_computing,
            "average_dependency_waiting_workers": average_waiting,
            "accounted_worker_fraction": accounted,
            "dependency_wait_share_by_block_p10": percentile(wait_shares, .1),
            "dependency_wait_share_by_block_p50": percentile(wait_shares, .5),
            "dependency_wait_share_by_block_p90": percentile(wait_shares, .9),
        },
        "measurement_note": (
            "Dependency wait is measured after a task reaches a worker queue "
            "and before its dependent event becomes ready. Remaining unaccounted "
            "worker time includes empty-queue polling and profiler overhead."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    p = summary["profile"]
    print(f"Correctness: {summary['correctness']}")
    print(f"Profile span: {p['window_span_ms']:.3f} ms")
    print(f"Average computing workers: {p['average_computing_workers']:.1f}")
    print(f"Average dependency-waiting workers: {p['average_dependency_waiting_workers']:.1f}")
    print(f"Accounted worker fraction: {100*p['accounted_worker_fraction']:.1f}%")
    print(
        "Dependency wait share p10/p50/p90: "
        f"{100*p['dependency_wait_share_by_block_p10']:.1f}/"
        f"{100*p['dependency_wait_share_by_block_p50']:.1f}/"
        f"{100*p['dependency_wait_share_by_block_p90']:.1f}%")
    print(f"Step 21 scheduler wait profile: {summary['status'].upper()}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
