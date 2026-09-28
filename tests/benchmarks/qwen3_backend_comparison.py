"""Compare Qwen3 SDPA, FlashInfer, CUDA Graph, and MPK decode backends.

This benchmark intentionally starts a fresh demo process for every recorded
sample.  Warmup generations run inside that process after model setup.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
BACKENDS = (
    "normal_sdpa",
    "normal_flashinfer",
    "normal_flashinfer_cuda_graph",
    "normal_flashinfer_cuda_graph_fused",
    "normal_flashinfer_cuda_graph_fused_rmsnorm",
    "normal_flashinfer_cuda_graph_fused_add_rmsnorm",
    "normal_flashinfer_cuda_graph_fused_norm_silu",
    "normal_flashinfer_cuda_graph_fused_norm_cuda_cores",
    "normal_flashinfer_cuda_graph_fused_rope_kv",
    "normal_flashinfer_cuda_graph_fused_rope_kv_argmax",
    "mpk_decode_only",
    "mpk_decode_only_optimized_prefill",
    "mpk_decode_only_page128_split_kv",
    "mpk_decode_only_split_kv",
)


# This manifest is deliberately explicit.  A backend may only be described as
# aligned with Optimized Normal after every decode optimization below is marked
# equivalent and backed by a focused correctness test.  Sharing a page size or
# using split-KV is not sufficient.
OPTIMIZED_NORMAL_DECODE_FEATURES = {
    "launch_amortization": "cuda_graph",
    "paged_attention": "flashinfer_paged_attention",
    "kv_page_size": 128,
    "split_kv": "flashinfer_split_kv",
    "qkv_projection": "fused_qkv_projection",
    "qk_norm_rope_kv_write": "fused_decode_rope_kv_write",
    "attention_output_projection": "projection_then_fused_add_rmsnorm",
    "rmsnorm": "flashinfer_rmsnorm",
    "gate_up_projection": "fused_gate_up_projection",
    "activation": "silu_and_mul",
    "mlp_output_projection": "projection_then_residual_add",
    "token_selection": "greedy_argmax",
}


MPK_DECODE_FEATURE_PARITY = {
    "launch_amortization": {
        "equivalent": True,
        "implementation": "persistent_kernel",
    },
    "paged_attention": {
        "equivalent": False,
        "implementation": "mirage_paged_attention",
        "missing": "FlashInfer-equivalent work partition and online-softmax path",
    },
    "kv_page_size": {
        "equivalent": True,
        "implementation": 128,
    },
    "split_kv": {
        "equivalent": False,
        "implementation": "mirage_fixed_chunk_split_kv",
        "missing": "FlashInfer-equivalent split heuristic and LSE merge",
    },
    "qkv_projection": {
        "equivalent": True,
        "implementation": "single_shuffled_qkv_projection",
    },
    "qk_norm_rope_kv_write": {
        "equivalent": True,
        "implementation": "fused_in_mpk_attention_task",
    },
    "attention_output_projection": {
        "equivalent": True,
        "implementation": "fused_projection_and_residual_then_rmsnorm",
        "note": "The fusion boundary differs but avoids the same residual materialization",
    },
    "rmsnorm": {
        "equivalent": True,
        "implementation": "mirage_rmsnorm_hopper",
        "evidence": "qwen3_rmsnorm_parity_probe.py: exact BF16 match on H100",
    },
    "gate_up_projection": {
        "equivalent": True,
        "implementation": "single_shuffled_gate_up_projection",
    },
    "activation": {
        "equivalent": True,
        "implementation": "mirage_silu_mul",
    },
    "mlp_output_projection": {
        "equivalent": True,
        "implementation": "fused_down_projection_and_residual",
    },
    "token_selection": {
        "equivalent": True,
        "implementation": "partial_reduce_argmax_with_lowest_id_tie_break",
    },
}


def mpk_decode_parity_complete():
    return all(item["equivalent"] for item in MPK_DECODE_FEATURE_PARITY.values())


def command_for(args, prompt, backend, output):
    page_size = (
        args.mpk_page_size
        if backend.startswith("mpk_") and args.mpk_page_size is not None
        else args.page_size
    )
    max_num_pages = (
        args.context_length + args.decode_steps + page_size
    ) // page_size
    command = [
        sys.executable,
        str(DEMO),
        "--prompt",
        prompt,
        "--max-seq-length",
        str(args.context_length + args.decode_steps + 1),
        "--max-new-tokens",
        str(args.decode_steps + 1),
        "--ignore-eos",
        "--phase-timing",
        "--in-process-warmup",
        str(args.warmup),
        "--page-size",
        str(page_size),
        "--max-num-pages",
        str(max_num_pages),
        "--max-num-batched-tokens",
        "8",
        "--model",
        args.model,
        "--save-tokens",
        str(output),
        "--save-token-limit",
        str(args.decode_steps + 1),
    ]
    if args.no_system_message:
        command.append("--no-system-message")
    if backend.startswith("mpk_") and args.mpk_kernel_cache_dir is not None:
        command += [
            "--mpk-kernel-cache-dir",
            str(args.mpk_kernel_cache_dir),
        ]
    if (
        backend.startswith("normal_flashinfer")
        and args.flashinfer_kv_page_size is not None
    ):
        command += [
            "--normal-flashinfer-kv-page-size",
            str(args.flashinfer_kv_page_size),
        ]
    if backend.startswith("normal_flashinfer") and args.cuda_graph_prefill:
        command.append("--normal-cuda-graph-prefill")
    if backend.startswith("normal_flashinfer"):
        command += [
            "--normal-flashinfer-prefill-backend",
            args.flashinfer_prefill_backend,
        ]
    if (
        backend.startswith("normal_flashinfer")
        and args.flashinfer_silu_prefill_only
    ):
        command.append("--normal-flashinfer-silu-prefill-only")
    if backend == "normal_sdpa":
        command += ["--backend", "normal", "--normal-attention", "sdpa"]
    elif backend == "normal_flashinfer":
        command += ["--backend", "normal", "--normal-attention", "flashinfer"]
    elif backend == "normal_flashinfer_cuda_graph":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_rmsnorm":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_add_rmsnorm":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_norm_silu":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-flashinfer-silu-and-mul",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_norm_cuda_cores":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-flashinfer-no-tensor-cores",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_rope_kv":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-fused-decode-rope-kv-cache",
        ]
    elif backend == "normal_flashinfer_cuda_graph_fused_rope_kv_argmax":
        command += [
            "--backend",
            "normal",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph",
            "--normal-cuda-graph-argmax",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-fused-decode-rope-kv-cache",
        ]
    elif backend == "mpk_decode_only":
        command += ["--backend", "mpk", "--mpk-policy", "decode-only"]
    elif backend == "mpk_decode_only_optimized_prefill":
        command += [
            "--backend",
            "mpk",
            "--mpk-policy",
            "decode-only",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph-prefill",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-flashinfer-silu-prefill-only",
            "--normal-flashinfer-prefill-backend",
            "auto",
        ]
    elif backend == "mpk_decode_only_page128_split_kv":
        command += [
            "--backend",
            "mpk",
            "--mpk-policy",
            "decode-only",
            "--split-kv-cache",
            "--normal-attention",
            "flashinfer",
            "--normal-cuda-graph-prefill",
            "--normal-fused-projections",
            "--normal-flashinfer-rmsnorm",
            "--normal-flashinfer-fused-add-rmsnorm",
            "--normal-flashinfer-silu-prefill-only",
            "--normal-flashinfer-prefill-backend",
            "auto",
        ]
    elif backend == "mpk_decode_only_split_kv":
        command += [
            "--backend",
            "mpk",
            "--mpk-policy",
            "decode-only",
            "--split-kv-cache",
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return command


def run_sample(args, prompt, backend, repeat_index):
    stem = f"repeat_{repeat_index}_{backend}"
    output = args.output_dir / f"{stem}.json"
    log = args.output_dir / f"{stem}.log"
    command = command_for(args, prompt, backend, output)
    with log.open("w") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{backend} timed out; see {log}") from error
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"{backend} exited {completed.returncode}; see {log}\n{tail}"
        )
    data = json.loads(output.read_text())
    if data.get("prompt_length") != args.context_length:
        raise ValueError(f"Unexpected prompt length in {output}")
    if data.get("generate_length") != args.decode_steps + 1:
        raise ValueError(f"Unexpected generation length in {output}")
    timing = data.get("phase_timing", {})
    if timing.get("prefill_ms", 0) <= 0 or timing.get("decode_ms", 0) <= 0:
        raise ValueError(f"Missing phase timing in {output}")
    return data


def decode_step_count(data):
    timing = data["phase_timing"]
    if timing.get("decode_steps") is not None:
        return int(timing["decode_steps"])
    steps = timing.get("decode_step_ms")
    if isinstance(steps, list):
        return len(steps)
    return int(data["generate_length"]) - 1


def summarize(samples, decode_steps):
    prefills = [sample["phase_timing"]["prefill_ms"] for sample in samples]
    decodes = [sample["phase_timing"]["decode_ms"] for sample in samples]
    totals = [prefill + decode for prefill, decode in zip(prefills, decodes)]
    return {
        "repeat_prefill_ms": prefills,
        "repeat_decode_ms": decodes,
        "mean_prefill_ms": statistics.mean(prefills),
        "mean_decode_ms": statistics.mean(decodes),
        "mean_prefill_plus_decode_ms": statistics.mean(totals),
        "mean_decode_step_ms": statistics.mean(decodes) / decode_steps,
        "decode_tokens_per_second": 1000 * decode_steps / statistics.mean(decodes),
        "generated_tokens_per_second_including_prefill": (
            1000 * (decode_steps + 1) / statistics.mean(totals)
        ),
        "decode_relative_range": (
            (max(decodes) - min(decodes)) / statistics.mean(decodes)
            if len(decodes) > 1
            else None
        ),
    }


def positional_matches(reference, actual, count):
    return sum(a == b for a, b in zip(reference[:count], actual[:count]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument(
        "--mpk-page-size",
        type=int,
        default=None,
        help="Optional MPK-specific page size for mixed backend comparisons",
    )
    parser.add_argument("--mpk-kernel-cache-dir", type=Path, default=None)
    parser.add_argument("--flashinfer-kv-page-size", type=int, default=None)
    parser.add_argument(
        "--flashinfer-prefill-backend",
        choices=("auto", "fa2", "fa3"),
        default="auto",
    )
    parser.add_argument("--cuda-graph-prefill", action="store_true")
    parser.add_argument("--flashinfer-silu-prefill-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--no-system-message", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=None,
        help="Run only these backends; normal_sdpa is added as the reference",
    )
    parser.add_argument("--minimum-token-match-fraction", type=float, default=2 / 3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.context_length < 1 or args.decode_steps < 1:
        parser.error("context-length and decode-steps must be positive")
    if args.warmup < 1:
        parser.error("warmup must be at least 1 because CUDA Graph capture requires it")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.context_length + args.decode_steps + 1 > args.page_size:
        parser.error("context and generated tokens must fit in one KV page")
    if not 0 <= args.minimum_token_match_fraction <= 1:
        parser.error("minimum-token-match-fraction must be in [0, 1]")

    selected_backends = list(args.backends or BACKENDS)
    if "normal_sdpa" not in selected_backends:
        selected_backends.insert(0, "normal_sdpa")
    selected_backends = tuple(dict.fromkeys(selected_backends))

    args.prompt_file = args.prompt_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text()

    samples = {backend: [] for backend in selected_backends}
    for repeat_index in range(args.repeat):
        order = (
            selected_backends
            if repeat_index % 2 == 0
            else tuple(reversed(selected_backends))
        )
        for backend in order:
            print(f"{backend} repeat {repeat_index + 1}/{args.repeat}", flush=True)
            sample = run_sample(args, prompt, backend, repeat_index)
            if decode_step_count(sample) != args.decode_steps:
                raise ValueError(f"Wrong decode step count for {backend}")
            samples[backend].append(sample)

    compared_tokens = min(30, args.decode_steps + 1)
    required_matches = int(
        compared_tokens * args.minimum_token_match_fraction + 0.999999
    )
    compared_backends = tuple(
        backend for backend in selected_backends if backend != "normal_sdpa"
    )
    match_counts = {backend: [] for backend in compared_backends}
    full_match_counts = {backend: [] for backend in compared_backends}
    full_match_fractions = {backend: [] for backend in compared_backends}
    first_mismatches = {backend: [] for backend in compared_backends}
    match_counts_by_128_token_window = {
        backend: [] for backend in compared_backends
    }
    for repeat_index in range(args.repeat):
        reference = samples["normal_sdpa"][repeat_index]["token_ids"]
        for backend in compared_backends:
            actual = samples[backend][repeat_index]["token_ids"]
            if min(len(reference), len(actual)) < compared_tokens:
                raise ValueError(f"Insufficient saved tokens for {backend}")
            matches = positional_matches(reference, actual, compared_tokens)
            match_counts[backend].append(matches)
            full_compared = min(len(reference), len(actual))
            full_matches = positional_matches(reference, actual, full_compared)
            full_match_counts[backend].append(full_matches)
            full_match_fractions[backend].append(
                full_matches / full_compared if full_compared else 0.0
            )
            first_mismatches[backend].append(next(
                (
                    index
                    for index, (expected, observed) in enumerate(
                        zip(reference, actual)
                    )
                    if expected != observed
                ),
                None,
            ))
            match_counts_by_128_token_window[backend].append([
                positional_matches(
                    reference[start:start + 128],
                    actual[start:start + 128],
                    min(128, full_compared - start),
                )
                for start in range(0, full_compared, 128)
            ])
            if matches < required_matches:
                raise ValueError(
                    f"{backend} matched {matches}/{compared_tokens}; "
                    f"required {required_matches}/{compared_tokens}"
                )

    summary = {
        "batch_size": 1,
        "context_length": args.context_length,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "reference_backend": "normal_sdpa",
        "compared_token_positions": compared_tokens,
        "required_token_matches": required_matches,
        "optimized_normal_decode_features": OPTIMIZED_NORMAL_DECODE_FEATURES,
        "mpk_decode_feature_parity": MPK_DECODE_FEATURE_PARITY,
        "mpk_decode_feature_parity_complete": mpk_decode_parity_complete(),
        "positional_matches_vs_sdpa": match_counts,
        "full_positional_matches_vs_sdpa": full_match_counts,
        "full_positional_match_fractions_vs_sdpa": full_match_fractions,
        "first_mismatch_positions_vs_sdpa": first_mismatches,
        "positional_matches_by_128_token_window_vs_sdpa": (
            match_counts_by_128_token_window
        ),
        "backends": {
            backend: summarize(samples[backend], args.decode_steps)
            for backend in selected_backends
        },
    }
    reference_decode = summary["backends"]["normal_sdpa"]["mean_decode_ms"]
    reference_total = summary["backends"]["normal_sdpa"][
        "mean_prefill_plus_decode_ms"
    ]
    for backend, result in summary["backends"].items():
        result["decode_speedup_vs_sdpa"] = reference_decode / result["mean_decode_ms"]
        result["phase_sum_speedup_vs_sdpa"] = (
            reference_total / result["mean_prefill_plus_decode_ms"]
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = args.output_dir / "summary.csv"
    fieldnames = [
        "backend",
        "mean_prefill_ms",
        "mean_decode_ms",
        "mean_prefill_plus_decode_ms",
        "mean_decode_step_ms",
        "decode_tokens_per_second",
        "generated_tokens_per_second_including_prefill",
        "decode_speedup_vs_sdpa",
        "phase_sum_speedup_vs_sdpa",
        "minimum_first30_matches_vs_sdpa",
        "minimum_full_match_fraction_vs_sdpa",
    ]
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for backend in selected_backends:
            result = summary["backends"][backend]
            row = {name: result.get(name) for name in fieldnames}
            row["backend"] = backend
            row["minimum_first30_matches_vs_sdpa"] = (
                min(match_counts[backend]) if backend in match_counts else compared_tokens
            )
            row["minimum_full_match_fraction_vs_sdpa"] = (
                min(full_match_fractions[backend])
                if backend in full_match_fractions else 1.0
            )
            writer.writerow(row)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
