"""Run stage-level parity gates for Optimized Normal and MPK decode.

Each case starts from the same eager FlashInfer prefill and compares the first
decode step, before autoregressive token divergence can contaminate later
layers. CUDA Graph is intentionally disabled because diagnostic snapshots are
incompatible with its required in-process warmup; graph replay does not change
operator math. The snapshots cover every major Qwen3 decoder stage.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
STRICT_LAYER_KEYS = (
    "layer0_norm",
    "layer0_qkv",
    "layer0_attention_output",
    "layer0_after_attention",
    "layer0_post_attention_norm",
    "layer0_mlp_mid",
    "layer0_silu_mul",
    "layer0_output",
    "normalized_hidden_state",
    "logits",
)
INFORMATIONAL_LAYER_KEYS = ("layer0_input",)
LAYER_KEYS = STRICT_LAYER_KEYS + INFORMATIONAL_LAYER_KEYS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("CONTEXT_LENGTH", "PROMPT_FILE"),
        required=True,
        help="May be repeated; prompt must render to the stated token length",
    )
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--mpk-kernel-cache-root",
        type=Path,
        default=Path("results/mpk_kernel_cache/qwen3_decode_parity"),
    )
    parser.add_argument("--max-absolute-error", type=float, default=1.0)
    parser.add_argument("--max-mean-absolute-error", type=float, default=0.1)
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def tensor_metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} vs {actual.shape}")
    reference = reference.float()
    actual = actual.float()
    difference = (reference - actual).abs()
    denominator = torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(
        actual
    )
    return {
        "num_elements": reference.numel(),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": (
            (torch.dot(reference.flatten(), actual.flatten()) / denominator).item()
            if denominator.item()
            else None
        ),
    }


def common_command(args, context_length, prompt, snapshot):
    max_seq_length = context_length + 2
    max_num_pages = (max_seq_length + args.page_size - 1) // args.page_size
    return [
        sys.executable,
        str(DEMO),
        "--prompt",
        prompt,
        "--max-seq-length",
        str(max_seq_length),
        "--max-new-tokens",
        "2",
        "--ignore-eos",
        "--page-size",
        str(args.page_size),
        "--max-num-pages",
        str(max_num_pages),
        "--max-num-batched-tokens",
        "8",
        "--model",
        args.model,
        "--save-intermediates",
        str(snapshot),
        "--normal-attention",
        "flashinfer",
        "--normal-flashinfer-kv-page-size",
        str(args.page_size),
        "--normal-fused-projections",
        "--normal-flashinfer-rmsnorm",
        "--normal-flashinfer-fused-add-rmsnorm",
        "--normal-flashinfer-prefill-backend",
        "auto",
    ]


def run_backend(args, context_length, prompt, backend, case_dir):
    snapshot = case_dir / f"{backend}.pt"
    log = case_dir / f"{backend}.log"
    command = common_command(args, context_length, prompt, snapshot)
    if backend == "optimized_normal":
        command += [
            "--backend",
            "normal",
            "--normal-fused-decode-rope-kv-cache",
        ]
    elif backend == "mpk_page128_split_kv":
        cache_dir = (
            args.mpk_kernel_cache_root
            / f"b1_s{context_length + 2}_page{args.page_size}"
        ).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        command += [
            "--backend",
            "mpk",
            "--mpk-policy",
            "decode-only",
            "--split-kv-cache",
            "--mpk-kernel-cache-dir",
            str(cache_dir),
        ]
    else:
        raise ValueError(backend)
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"{backend} exited {completed.returncode}; see {log}\n{tail}"
        )
    return torch.load(snapshot, map_location="cpu", weights_only=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    failed = False
    for context_text, prompt_name in args.case:
        context_length = int(context_text)
        prompt_path = Path(prompt_name).resolve()
        case_dir = args.output_dir / f"c{context_length}"
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt = prompt_path.read_text()
        print(f"Running context={context_length} Optimized Normal...", flush=True)
        normal = run_backend(
            args, context_length, prompt, "optimized_normal", case_dir
        )
        print(f"Running context={context_length} MPK...", flush=True)
        mpk = run_backend(
            args, context_length, prompt, "mpk_page128_split_kv", case_dir
        )
        if not torch.equal(normal["prefix_token_ids"], mpk["prefix_token_ids"]):
            raise ValueError("The two probes did not use the same decode input")

        stage_metrics = {}
        for key in LAYER_KEYS:
            missing = [
                name
                for name, probe in (("optimized_normal", normal), ("mpk", mpk))
                if key not in probe
            ]
            if missing:
                raise ValueError(
                    f"Missing required snapshot {key} from: {', '.join(missing)}"
                )
            metrics = tensor_metrics(normal[key], mpk[key])
            metrics["within_thresholds"] = (
                metrics["max_absolute_error"] <= args.max_absolute_error
                and metrics["mean_absolute_error"]
                <= args.max_mean_absolute_error
                and (
                    metrics["cosine_similarity"] is None
                    or metrics["cosine_similarity"]
                    >= args.minimum_cosine_similarity
                )
            )
            metrics["gated"] = key in STRICT_LAYER_KEYS
            metrics["passed"] = (
                metrics["within_thresholds"] or not metrics["gated"]
            )
            failed |= metrics["gated"] and not metrics["within_thresholds"]
            stage_metrics[key] = metrics
        report = {
            "context_length": context_length,
            "page_size": args.page_size,
            "decode_step_index": normal["decode_step_index"],
            "generated_token_matches": torch.equal(
                normal["generated_token_ids"], mpk["generated_token_ids"]
            ),
            "stages": stage_metrics,
            "passed": all(item["passed"] for item in stage_metrics.values()),
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    output = {
        "thresholds": {
            "max_absolute_error": args.max_absolute_error,
            "max_mean_absolute_error": args.max_mean_absolute_error,
            "minimum_cosine_similarity": args.minimum_cosine_similarity,
        },
        "cases": reports,
        "passed": not failed,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    if failed:
        raise SystemExit(1)
    print("MPK decode stage parity: PASS")


if __name__ == "__main__":
    main()
