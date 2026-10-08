"""Compare Hopper TMA KV loading with the established MPK baseline."""
import argparse, csv, json
from pathlib import Path

def keyed(rows):
    return {(int(r["batch_size"]), int(r["kv_length"])): r for r in rows}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    old = keyed(json.loads(a.baseline.read_text())["rows"])
    new = keyed(json.loads(a.candidate.read_text())["rows"])
    rows=[]; failures=0
    for key in sorted(new):
        b, n = old[key], new[key]
        reasons=[]
        if n["status"] != "passed" or n["minimum_first10_matches"] != 10:
            reasons.append(n.get("reason") or "correctness failed")
        baseline_us = b.get("mpk_attention_mean_task_us")
        candidate_us = n.get("mpk_attention_mean_task_us")
        speedup = (
            baseline_us / candidate_us
            if baseline_us is not None and candidate_us not in (None, 0)
            else None
        )
        row={"batch_size":key[0],"kv_length":key[1],
             "status":"failed" if reasons else "passed",
             "minimum_first10_matches":n["minimum_first10_matches"],
             "baseline_task_us":baseline_us,
             "tma_task_us":candidate_us,
             "task_speedup":speedup,"reason":"; ".join(reasons)}
        rows.append(row); failures += bool(reasons)
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
        print(f"B={key[0]} KV={key[1]}: {row['status'].upper()}; task speedup={speedup_text}")
    summary={"step":40,"phase":"hopper_tma_kv","status":"failed" if failures else "passed","rows":rows}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (a.output_dir/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Step 40 Hopper TMA KV: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)
if __name__ == "__main__": main()
