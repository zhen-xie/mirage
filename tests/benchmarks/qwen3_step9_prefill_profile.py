"""Summarize CUDA-event prefill stages with the Step 8 SGLang baseline."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "model", "status", "first10_matches", "prefill_total_ms",
    "embedding_ms", "transformer_layers_ms", "mean_layer_ms",
    "slowest_layer", "slowest_layer_ms", "final_norm_ms", "lm_head_ms",
    "unaccounted_ms", "layers_fraction", "sglang_prefill_ms",
    "mpk_over_sglang", "reason",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpk-summary", type=Path, required=True)
    parser.add_argument("--step8-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mpk = json.loads(args.mpk_summary.read_text(encoding="utf-8"))
    comparison = json.loads(args.step8_comparison.read_text(encoding="utf-8"))
    sglang_by_model = {
        row["model"]: row.get("sglang_prefill_ms")
        for row in comparison["rows"]
    }
    rows = []
    all_passed = mpk.get("status") == "passed"

    for source in mpk["rows"]:
        errors = []
        if source.get("status") != "completed":
            errors.append("profile run failed correctness validation")
        if source.get("first10_matches") != 10:
            errors.append("first-10 token gate failed")
        raw_profile = source.get("prefill_stage_profile_ms")
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) else raw_profile
        if not isinstance(profile, dict):
            profile = {}
            errors.append("missing prefill stage profile")
        layer_items = sorted(
            (
                (name, float(value)) for name, value in profile.items()
                if name.startswith("layer_") and name[6:].isdigit()
            ),
            key=lambda item: int(item[0][6:]),
        )
        total = source.get("prefill_time_ms")
        instrumented = profile.get("total_instrumented")
        layers = profile.get("transformer_layers")
        if not layer_items:
            errors.append("missing per-layer timings")
        if not isinstance(total, (int, float)) or total <= 0:
            errors.append("invalid total prefill timing")
        if not isinstance(instrumented, (int, float)) or instrumented <= 0:
            errors.append("invalid instrumented timing")
        slowest = max(layer_items, key=lambda item: item[1]) if layer_items else (None, None)
        mean_layer = layers / len(layer_items) if layer_items and layers is not None else None
        sglang = sglang_by_model.get(source["model"])
        if not isinstance(sglang, (int, float)) or sglang <= 0:
            errors.append("missing Step 8 SGLang prefill timing")
        status = "failed" if errors else "completed"
        all_passed &= status == "completed"
        row = {
            "model": source["model"],
            "status": status,
            "first10_matches": source.get("first10_matches"),
            "prefill_total_ms": total,
            "embedding_ms": profile.get("embedding"),
            "transformer_layers_ms": layers,
            "mean_layer_ms": mean_layer,
            "slowest_layer": slowest[0],
            "slowest_layer_ms": slowest[1],
            "final_norm_ms": profile.get("final_norm"),
            "lm_head_ms": profile.get("lm_head"),
            "unaccounted_ms": total - instrumented if isinstance(total, (int, float)) and isinstance(instrumented, (int, float)) else None,
            "layers_fraction": layers / total if isinstance(layers, (int, float)) and isinstance(total, (int, float)) and total > 0 else None,
            "sglang_prefill_ms": sglang,
            "mpk_over_sglang": total / sglang if isinstance(total, (int, float)) and isinstance(sglang, (int, float)) and sglang > 0 else None,
            "reason": "; ".join(errors),
        }
        rows.append(row)
        print(
            f"{source['model']}: {'PASS' if status == 'completed' else 'FAIL'}; "
            f"prefill={total:.3f} ms; layers={layers:.3f} ms "
            f"({100.0 * row['layers_fraction']:.1f}%); "
            f"MPK/SGLang={row['mpk_over_sglang']:.3f}x",
            flush=True,
        )

    csv_path = args.output_dir / "prefill_profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 9,
        "status": "passed" if all_passed else "failed",
        "correctness_gate": "MPK first 10 generated tokens equal Torch",
        "timing_method": "CUDA events recorded on the inference stream",
        "rows": rows,
    }
    json_path = args.output_dir / "prefill_profile.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Step 9 prefill profile: {'PASS' if all_passed else 'FAIL'}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
