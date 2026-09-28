"""Summarize representative Optimized Normal, adaptive MPK, and SGLang runs."""

import argparse
import csv
import json
from pathlib import Path


NORMAL_BACKEND = "normal_flashinfer_cuda_graph_fused_rope_kv"
CASES = (
    (
        "short",
        128,
        128,
        "qwen3_experiment_c_normal_stage1_fix_smoke/summary.json",
    ),
    (
        "long_context",
        1024,
        128,
        "qwen3_experiment_c_normal_stage1_b1_in1024_out128/summary.json",
    ),
    (
        "long_generation",
        128,
        1024,
        "qwen3_experiment_c_normal_stage1_b1_in128_out1024/summary.json",
    ),
)
FIELDS = (
    "case",
    "batch_size",
    "context_length",
    "output_length",
    "backend",
    "prefill_ms",
    "decode_ms",
    "decode_steps",
    "decode_step_ms",
    "decode_tokens_per_second",
    "speedup_vs_optimized_normal",
    "speedup_vs_sglang",
    "timing_source",
)


def load_normal(path, case_name, context_length, output_length):
    data = json.loads(path.read_text())
    result = data["backends"][NORMAL_BACKEND]
    return {
        "case": case_name,
        "batch_size": 1,
        "context_length": context_length,
        "output_length": output_length,
        "backend": "optimized_normal",
        "prefill_ms": float(result["mean_prefill_ms"]),
        "decode_ms": float(result["mean_decode_ms"]),
        "decode_steps": output_length - 1,
        "decode_step_ms": float(result["mean_decode_step_ms"]),
        "decode_tokens_per_second": float(
            result["decode_tokens_per_second"]
        ),
        "speedup_vs_optimized_normal": 1.0,
        "speedup_vs_sglang": None,
        "timing_source": "Mirage CUDA phase events",
    }


def load_mpk(path, case_name, context_length, output_length):
    data = json.loads(path.read_text())
    timing = data["phase_timing"]
    steps = int(timing["decode_steps"])
    generated = int(data["generate_length"])
    invalid = data.get("invalid_token_ids_by_request", {})
    if generated != output_length or steps != output_length - 1 or invalid:
        raise ValueError(
            f"Invalid adaptive MPK result in {path}: generated={generated}, "
            f"decode_steps={steps}, invalid={invalid}"
        )
    decode_ms = float(timing["decode_ms"])
    step_ms = decode_ms / steps
    return {
        "case": case_name,
        "batch_size": 1,
        "context_length": context_length,
        "output_length": output_length,
        "backend": "adaptive_mpk",
        "prefill_ms": float(timing["prefill_ms"]),
        "decode_ms": decode_ms,
        "decode_steps": steps,
        "decode_step_ms": step_ms,
        "decode_tokens_per_second": 1000.0 / step_ms,
        "speedup_vs_optimized_normal": None,
        "speedup_vs_sglang": None,
        "timing_source": "Mirage CUDA phase events; prefill not warmed",
    }


def load_sglang(path):
    results = {}
    if not path.is_file():
        return results
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        key = (
            int(item["batch_size"]),
            int(item["input_len"]),
            int(item["output_len"]),
        )
        results[key] = item
    return results


def sglang_row(item, case_name, context_length, output_length):
    step_ms = float(item["median_decode_latency"]) * 1000.0
    steps = output_length - 1
    return {
        "case": case_name,
        "batch_size": 1,
        "context_length": context_length,
        "output_length": output_length,
        "backend": "sglang",
        "prefill_ms": float(item["prefill_latency"]) * 1000.0,
        "decode_ms": step_ms * steps,
        "decode_steps": steps,
        "decode_step_ms": step_ms,
        "decode_tokens_per_second": float(item["median_decode_throughput"]),
        "speedup_vs_optimized_normal": None,
        "speedup_vs_sglang": 1.0,
        "timing_source": "SGLang one_batch wall-clock timing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--mpk-root",
        type=Path,
        default=Path("results/qwen3_mpk_adaptive_attention_validation"),
    )
    parser.add_argument(
        "--sglang-jsonl",
        type=Path,
        default=Path("results/sglang_qwen3_b1_matrix/results.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen3_representative_backend_summary"),
    )
    args = parser.parse_args()

    sglang = load_sglang(args.sglang_jsonl)
    rows = []
    missing_sglang = []

    for case_name, context_length, output_length, normal_relative in CASES:
        normal_path = args.normal_root / normal_relative
        mpk_path = args.mpk_root / case_name / "tokens.json"
        if not normal_path.is_file():
            raise FileNotFoundError(normal_path)
        if not mpk_path.is_file():
            raise FileNotFoundError(mpk_path)

        normal = load_normal(
            normal_path, case_name, context_length, output_length
        )
        mpk = load_mpk(mpk_path, case_name, context_length, output_length)
        mpk["speedup_vs_optimized_normal"] = (
            normal["decode_step_ms"] / mpk["decode_step_ms"]
        )
        rows.extend((normal, mpk))

        key = (1, context_length, output_length)
        if key in sglang:
            row = sglang_row(
                sglang[key], case_name, context_length, output_length
            )
            row["speedup_vs_optimized_normal"] = (
                normal["decode_step_ms"] / row["decode_step_ms"]
            )
            rows.append(row)
        else:
            missing_sglang.append(key)

    sglang_steps = {
        row["case"]: row["decode_step_ms"]
        for row in rows
        if row["backend"] == "sglang"
    }
    for row in rows:
        sglang_step = sglang_steps.get(row["case"])
        if sglang_step is not None:
            row["speedup_vs_sglang"] = (
                sglang_step / row["decode_step_ms"]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "rows": rows,
        "missing_sglang_cases": missing_sglang,
        "prefill_comparison_valid": False,
        "prefill_note": (
            "Adaptive MPK validation used no in-process warmup; compare decode "
            "metrics only."
        ),
    }, indent=2))

    print(
        "case             backend              step ms     tokens/s   "
        "vs normal  vs SGLang"
    )
    for row in rows:
        print(
            f"{row['case']:<16} {row['backend']:<20} "
            f"{row['decode_step_ms']:>8.3f} "
            f"{row['decode_tokens_per_second']:>12.2f} "
            f"{row['speedup_vs_optimized_normal']:>10.3f}x "
            + (
                f"{row['speedup_vs_sglang']:>9.3f}x"
                if row["speedup_vs_sglang"] is not None
                else f"{'N/A':>10}"
            )
        )
    for batch_size, context_length, output_length in missing_sglang:
        print(
            "WARNING: missing SGLang result for "
            f"B={batch_size}, S_IN={context_length}, S_OUT={output_length}"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
