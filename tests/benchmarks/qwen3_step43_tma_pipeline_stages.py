"""Compare two- and three-stage Hopper TMA attention pipelines."""

import argparse
import csv
import json
from pathlib import Path


def keyed(rows):
    return {(row["batch_size"], row["kv_length"]): row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--stage2-phases", type=Path, required=True)
    parser.add_argument("--stage3-phases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage2 = json.loads(args.stage2.read_text(encoding="utf-8"))
    stage3 = json.loads(args.stage3.read_text(encoding="utf-8"))
    phase2 = keyed(json.loads(args.stage2_phases.read_text(encoding="utf-8"))["rows"])
    phase3 = keyed(json.loads(args.stage3_phases.read_text(encoding="utf-8"))["rows"])
    rows = []
    failures = 0
    for old in stage2["rows"]:
        key = (old["batch_size"], old["kv_length"])
        new = keyed(stage3["rows"])[key]
        p2, p3 = phase2[key], phase3[key]
        reasons = []
        for label, candidate in (("stage2", old), ("stage3", new)):
            if candidate["status"] != "passed":
                reasons.append(f"{label}: {candidate.get('reason') or 'failed'}")
            if candidate.get("minimum_first10_matches") != 10:
                reasons.append(
                    f"{label}: first-10={candidate.get('minimum_first10_matches')}")
            if candidate.get("invalid_tokens") or candidate.get("incomplete_requests"):
                reasons.append(f"{label}: invalid or incomplete output")
        for label, candidate in (("stage2", p2), ("stage3", p3)):
            if candidate["status"] != "passed":
                reasons.append(f"{label} phases: {candidate.get('reason') or 'failed'}")

        def ratio(numerator, denominator):
            return numerator / denominator if numerator is not None and denominator else None

        row = {
            "batch_size": key[0],
            "kv_length": key[1],
            "status": "failed" if reasons else "passed",
            "stage2_attention_task_us": old.get("mpk_attention_mean_task_us"),
            "stage3_attention_task_us": new.get("mpk_attention_mean_task_us"),
            "attention_task_ratio_stage3_stage2": ratio(
                new.get("mpk_attention_mean_task_us"),
                old.get("mpk_attention_mean_task_us")),
            "stage2_decode_step_ms": old.get("decode_step_ms"),
            "stage3_decode_step_ms": new.get("decode_step_ms"),
            "decode_step_ratio_stage3_stage2": ratio(
                new.get("decode_step_ms"), old.get("decode_step_ms")),
            "stage2_decode_tokens_per_second": old.get("decode_tokens_per_second"),
            "stage3_decode_tokens_per_second": new.get("decode_tokens_per_second"),
            "stage2_producer_wait_cycles_per_task": p2.get("producer_wait_cycles_per_task"),
            "stage3_producer_wait_cycles_per_task": p3.get("producer_wait_cycles_per_task"),
            "producer_wait_ratio_stage3_stage2": ratio(
                p3.get("producer_wait_cycles_per_task"),
                p2.get("producer_wait_cycles_per_task")),
            "stage2_consumer_wait_cycles_per_task": p2.get("consumer_ready_wait_cycles_per_task"),
            "stage3_consumer_wait_cycles_per_task": p3.get("consumer_ready_wait_cycles_per_task"),
            "consumer_wait_ratio_stage3_stage2": ratio(
                p3.get("consumer_ready_wait_cycles_per_task"),
                p2.get("consumer_ready_wait_cycles_per_task")),
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        failures += bool(reasons)
        print(
            f"B={key[0]} KV={key[1]}: {row['status'].upper()}; "
            f"task ratio={row['attention_task_ratio_stage3_stage2']}; "
            f"decode ratio={row['decode_step_ratio_stage3_stage2']}; "
            f"producer-wait ratio={row['producer_wait_ratio_stage3_stage2']}; "
            f"consumer-wait ratio={row['consumer_wait_ratio_stage3_stage2']}",
            flush=True,
        )

    summary = {
        "step": 43,
        "phase": "tma_attention_pipeline_stage_ablation",
        "status": "failed" if failures else "passed",
        "ratio_definition": "stage3 / stage2; lower is better",
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Step 43 TMA pipeline stages: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
