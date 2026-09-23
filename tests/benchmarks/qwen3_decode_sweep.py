"""Run a resumable Qwen3 batch/input/output length sweep across all policies.

Each case calls qwen3_decode_backend.py and keeps its own log and summary.
Each request uses one KV page; memory-limited cells are recorded as skipped.
"""

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).with_name("qwen3_decode_backend.py")
FIELDS = (
    "batch_size", "s_in", "s_out", "context_length", "decode_steps", "repeat", "warmup",
    "page_size", "max_num_pages", "max_num_batched_tokens",
    "estimated_kv_gib", "gpu_total_gib", "status", "status_reason",
    "model", "prompt_sha256", "git_commit", "gpu_name", "driver_version",
    "torch_version", "torch_cuda_version", "transformers_version",
    "flashinfer_version", "policy",
    "prefill_backend", "decode_backend", "timing_mode",
    "mean_prefill_ms", "mean_decode_ms", "mean_prefill_plus_decode_ms",
    "mean_continuous_total_ms", "continuous_generated_tokens_per_second",
    "mean_step_latency_ms", "median_step_latency_ms", "p90_step_latency_ms",
    "p99_step_latency_ms", "decode_tokens_per_second",
    "generated_tokens_per_second_including_prefill", "decode_relative_range",
    "repeat_prefill_ms", "repeat_decode_ms", "repeat_continuous_total_ms",
    "minimum_first30_matches",
    "compared_token_positions", "required_token_matches", "correctness_gate_applicable",
    "first30_matches_by_request_per_repeat", "decode_speedup_vs_normal",
    "phase_sum_speedup_vs_normal", "continuous_total_speedup_vs_normal",
    "summary_path",
)
POLICIES = ("normal", "always", "always-continuous", "decode-only", "prefill-only")
BACKENDS = {
    "normal": ("NORMAL", "NORMAL"),
    "always": ("MPK", "MPK"),
    "always-continuous": ("MPK", "MPK"),
    "decode-only": ("NORMAL", "MPK"),
    "prefill-only": ("MPK", "NORMAL"),
}
DEFAULT_SEEDS = (
    "Explain a number theory example.",
    "Review an asynchronous Python service.",
    "Compare two public transport schedules.",
    "Describe a seasonal ocean sensor trend.",
    "Evaluate a historical primary source.",
    "Explain a clinical trial design.",
    "Describe a database consistency problem.",
    "Analyze an energy demand forecast.",
)


def required_token_matches(compared_tokens):
    """Scale the agreed 20/30 positional gate to shorter generations."""
    return (compared_tokens * 20 + 29) // 30


def format_duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class SweepProgress:
    def __init__(self, total, path):
        self.total = total
        self.path = path
        self.completed = 0
        self.started = time.monotonic()
        self.run_durations = []
        self.show("Starting sweep")

    def show(self, status):
        elapsed = time.monotonic() - self.started
        remaining = self.total - self.completed
        mean_duration = (sum(self.run_durations) / len(self.run_durations)
                         if self.run_durations else None)
        eta = remaining * mean_duration if mean_duration is not None else None
        width = 24
        filled = width * self.completed // self.total
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {self.completed}/{self.total} "
              f"({100 * self.completed / self.total:.1f}%) "
              f"elapsed={format_duration(elapsed)} ETA~{format_duration(eta)} "
              f"| {status}", flush=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "completed_cases": self.completed,
            "total_cases": self.total,
            "elapsed_seconds": round(elapsed, 1),
            "estimated_remaining_seconds": round(eta, 1) if eta is not None else None,
            "status": status,
        }, indent=2) + "\n")
        temporary.replace(self.path)

    def finish_case(self, label, outcome, duration=None):
        if duration is not None:
            self.run_durations.append(duration)
        self.completed += 1
        self.show(f"{label}: {outcome}")


def run_benchmark(command, log_path, progress, case_label,
                  progress_interval_seconds):
    """Save the child log while reporting policy starts and idle heartbeats."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        events = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    events.put(line)
            finally:
                events.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        stage = "starting benchmark"
        stage_started = time.monotonic()
        try:
            while True:
                try:
                    timeout = (progress_interval_seconds
                               if progress_interval_seconds > 0 else None)
                    line = events.get(timeout=timeout)
                except queue.Empty:
                    progress.show(f"{case_label}: {stage} "
                                  f"({format_duration(time.monotonic() - stage_started)})")
                    continue
                if line is None:
                    break
                log_file.write(line)
                log_file.flush()
                match = re.match(
                    r"^(normal|always|always-continuous|decode-only|prefill-only) "
                    r"(warmup|repeat) \d+/\d+", line,
                )
                if match:
                    stage = line.strip()
                    stage_started = time.monotonic()
                    progress.show(f"{case_label}: {stage}")
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return process.wait()


def prompt_length(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(rendered)["input_ids"])


def make_prompt(tokenizer, seed, target_length):
    seed_length = prompt_length(tokenizer, seed)
    if seed_length == target_length:
        return seed
    if seed_length > target_length:
        low, high = 0, len(seed)
        while low < high:
            middle = (low + high + 1) // 2
            if prompt_length(tokenizer, seed[:middle]) <= target_length:
                low = middle
            else:
                high = middle - 1
        seed = seed[:low]
        seed_length = prompt_length(tokenizer, seed)
        if seed_length > target_length:
            raise ValueError(f"Minimum chat template length {seed_length} exceeds {target_length}")
        if seed_length == target_length:
            return seed

    repeats = max(1, target_length // max(seed_length, 1) - 1)
    prefix = (seed + "\n") * repeats
    while prompt_length(tokenizer, prefix) >= target_length and repeats > 1:
        repeats -= 1
        prefix = (seed + "\n") * repeats
    if prompt_length(tokenizer, prefix) >= target_length:
        prefix = seed

    low, high = 0, 1
    while prompt_length(tokenizer, prefix + " hello" * high) < target_length:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if prompt_length(tokenizer, prefix + " hello" * middle) < target_length:
            low = middle + 1
        else:
            high = middle
    prompt = prefix + " hello" * low
    actual = prompt_length(tokenizer, prompt)
    if actual != target_length:
        raise ValueError(f"Constructed {actual} tokens; target is {target_length}")
    return prompt


def make_distinct_seeds(tokenizer, source_seeds, count, minimum_length):
    seeds = []
    for request_id in range(count):
        seed = source_seeds[request_id % len(source_seeds)]
        if request_id >= len(source_seeds):
            variant = f" {request_id}"
            prefix_length = max(1, len(seed) // 2)
            while prompt_length(tokenizer, seed[:prefix_length] + variant) > minimum_length:
                prefix_length //= 2
                if prefix_length == 0:
                    seed = variant.strip()
                    break
            else:
                seed = seed[:prefix_length] + variant
        seeds.append(seed)
    return seeds


def page_size_for(context_length, decode_steps, minimum=64):
    max_sequence = context_length + decode_steps + 1
    return max(minimum, 1 << (max_sequence - 1).bit_length())


def kv_bytes_for(model_config, batch_size, page_size):
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = model_config.hidden_size // model_config.num_attention_heads
    return (model_config.num_hidden_layers * batch_size * page_size *
            model_config.num_key_value_heads * head_dim * 2 * 2)


def collect_environment(args):
    import torch

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()[0].strip()
    return {
        "model": args.model,
        "git_commit": git_commit,
        "gpu_name": torch.cuda.get_device_name(0),
        "driver_version": driver,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": metadata.version("transformers"),
        "flashinfer_version": metadata.version("flashinfer-python"),
    }


def load_rows(summary_path, batch_size, context_length, s_out, args, environment,
              prompt_sha256, page_size, estimated_kv_gib, gpu_total_gib):
    summary = json.loads(summary_path.read_text())
    expected = {
        "batch_size": batch_size,
        "context_length": context_length,
        "decode_steps": s_out - 1,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "page_size": page_size,
        "max_num_pages": batch_size,
        "max_num_batched_tokens": max(8, batch_size),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing summary has different parameters: {summary_path}")
    keys = {"normal", "mpk_always", "mpk_always_continuous", "mpk_decode_only",
            "mpk_prefill_only"}
    if not keys.issubset(summary):
        raise ValueError(f"Existing summary is missing one or more policies: {summary_path}")
    match_counts = summary["first_30_token_matches_by_policy"]
    if any(policy not in match_counts for policy in POLICIES[1:]):
        raise ValueError(f"Existing summary is missing correctness counts: {summary_path}")

    normal = summary["normal"]
    rows = []
    for policy in POLICIES:
        item = normal if policy == "normal" else summary[f"mpk_{policy.replace('-', '_')}"]
        continuous = policy == "always-continuous"
        counts = None if policy == "normal" else match_counts[policy]
        prefill_backend, decode_backend = BACKENDS[policy]
        minimum = min(min(repeat_counts) for repeat_counts in counts) if counts else None
        compared_positions = summary["compared_token_positions"]
        required_matches = summary.get(
            "required_token_matches", required_token_matches(compared_positions))
        gate_applicable = compared_positions > 0
        gate_passed = minimum is None or minimum >= required_matches
        if policy == "normal":
            status = "completed"
            reason = None
        elif gate_passed:
            status = "completed"
            reason = None
        else:
            status = "correctness_failed"
            reason = (f"Minimum positional matches: {minimum}/{compared_positions}; "
                      f"required: {required_matches}/{compared_positions}")
        rows.append({
            **expected,
            "s_in": context_length,
            "s_out": s_out,
            **environment,
            "prompt_sha256": prompt_sha256,
            "estimated_kv_gib": estimated_kv_gib,
            "gpu_total_gib": gpu_total_gib,
            "status": status,
            "status_reason": reason,
            "policy": policy,
            "prefill_backend": prefill_backend,
            "decode_backend": decode_backend,
            "timing_mode": ("continuous_one_mpk_launch" if continuous
                            else "split_two_mpk_launches" if policy == "always"
                            else "one_prefill_phase_then_decode"),
            "mean_prefill_ms": None if continuous else item["mean_prefill_ms"],
            "mean_decode_ms": None if continuous else item["mean_total_decode_ms"],
            "mean_prefill_plus_decode_ms": (
                None if continuous else item["mean_prefill_plus_decode_ms"]),
            "mean_continuous_total_ms": item["mean_total_ms"] if continuous else None,
            "continuous_generated_tokens_per_second": (
                item["generated_tokens_per_second"] if continuous else None),
            "mean_step_latency_ms": None if continuous else item["mean_step_latency_ms"],
            "median_step_latency_ms": None if continuous else item["median_step_latency_ms"],
            "p90_step_latency_ms": None if continuous else item["p90_step_latency_ms"],
            "p99_step_latency_ms": None if continuous else item["p99_step_latency_ms"],
            "decode_tokens_per_second": None if continuous else item["tokens_per_second"],
            "generated_tokens_per_second_including_prefill": (
                None if continuous else item["generated_tokens_per_second_including_prefill"]),
            "decode_relative_range": item["relative_total_range"],
            "repeat_prefill_ms": None if continuous else json.dumps(item["repeat_prefill_ms"]),
            "repeat_decode_ms": (None if continuous
                                 else json.dumps(item["repeat_total_decode_ms"])),
            "repeat_continuous_total_ms": (
                json.dumps(item["repeat_total_ms"]) if continuous else None),
            "minimum_first30_matches": minimum,
            "compared_token_positions": compared_positions,
            "required_token_matches": required_matches,
            "correctness_gate_applicable": gate_applicable,
            "first30_matches_by_request_per_repeat": json.dumps(counts) if counts else None,
            "decode_speedup_vs_normal": (
                None if continuous else normal["mean_total_decode_ms"] /
                item["mean_total_decode_ms"]),
            "phase_sum_speedup_vs_normal": (
                None if continuous else normal["mean_prefill_plus_decode_ms"] /
                item["mean_prefill_plus_decode_ms"]),
            "continuous_total_speedup_vs_normal": (
                normal["mean_prefill_plus_decode_ms"] / item["mean_total_ms"]
                if continuous else None),
            "summary_path": str(summary_path),
        })
    return rows


def unavailable_rows(batch_size, context_length, s_out, args, environment, page_size,
                     estimated_kv_gib, gpu_total_gib, status, reason):
    rows = []
    for policy in POLICIES:
        prefill_backend, decode_backend = BACKENDS[policy]
        row = {field: None for field in FIELDS}
        row.update({
            "batch_size": batch_size,
            "context_length": context_length,
            "s_in": context_length,
            "s_out": s_out,
            "decode_steps": s_out - 1,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "page_size": page_size,
            "max_num_pages": batch_size,
            "max_num_batched_tokens": max(8, batch_size),
            "estimated_kv_gib": estimated_kv_gib,
            "gpu_total_gib": gpu_total_gib,
            "status": status,
            "status_reason": reason,
            "policy": policy,
            "prefill_backend": prefill_backend,
            "decode_backend": decode_backend,
            **environment,
        })
        rows.append(row)
    return rows


def write_csv(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[int(value) for value in os.getenv("B_VALUES", "1 2 4 8 16 32 64 128").split()])
    parser.add_argument("--s-in-values", nargs="+", type=int,
                        default=[int(value) for value in os.getenv("S_IN_VALUES", "16 32 64 128 256 512 1024").split()])
    parser.add_argument("--s-out-values", nargs="+", type=int,
                        default=[int(value) for value in os.getenv("S_OUT_VALUES", "16 32 64 128 256 512 1024").split()])
    parser.add_argument("--cases", nargs="+", metavar="B:S_IN:S_OUT",
                        help="Run selected triples instead of the Cartesian grid")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout in seconds for each demo process")
    parser.add_argument("--fail-on-failed-cases", action="store_true",
                        help="Return a nonzero status if any selected case fails")
    parser.add_argument("--allow-code-change-resume", action="store_true",
                        help="Reuse case summaries when only the recorded Git commit changed")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--reserve-gib", type=float, default=32.0,
                        help="Keep this much GPU memory outside the estimated KV cache")
    parser.add_argument("--min-page-size", type=int, default=64,
                        help="Minimum KV page size; use 4096 to compare with established runs")
    parser.add_argument("--max-mpk-batch-size", type=int, default=128,
                        help="Largest batch enabled for MPK; Hopper uses the large-batch CUTLASS path above 16")
    parser.add_argument("--progress-interval-seconds", type=int, default=30,
                        help="Seconds between idle progress heartbeats; use 0 to disable them")
    parser.add_argument("--source-prompts-file", type=Path,
                        default=ROOT / "tests/benchmarks/baselines/batch_b8_smoke/eight_prompts.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/qwen3_decode_sweep")
    args = parser.parse_args()

    selected_cases = None
    if args.cases:
        try:
            cases = [tuple(map(int, item.split(":"))) for item in args.cases]
        except ValueError:
            parser.error("--cases entries must use B:S_IN:S_OUT with integer values")
        if any(len(case) != 3 for case in cases) or len(set(cases)) != len(cases):
            parser.error("--cases entries must be distinct B:S_IN:S_OUT triples")
        selected_cases = set(cases)
        args.batch_sizes = sorted({batch for batch, _, _ in cases})
        args.s_in_values = sorted({s_in for _, s_in, _ in cases})
        args.s_out_values = sorted({s_out for _, _, s_out in cases})

    if (not args.batch_sizes or len(set(args.batch_sizes)) != len(args.batch_sizes)
        or any(batch_size < 1 or batch_size > 128 for batch_size in args.batch_sizes)):
        parser.error("Batch sizes must be distinct integers in [1, 128]")
    if (not args.s_in_values or len(set(args.s_in_values)) != len(args.s_in_values)
        or any(length < 1 or length > 8192 for length in args.s_in_values)):
        parser.error("Input lengths must be distinct integers in [1, 8192]")
    if (not args.s_out_values or len(set(args.s_out_values)) != len(args.s_out_values)
        or any(length < 2 or length > 8192 for length in args.s_out_values)):
        parser.error("Output lengths must be distinct integers in [2, 8192]")
    if (args.repeat < 1 or args.warmup < 0
        or args.timeout < 1 or args.reserve_gib < 0
        or args.progress_interval_seconds < 0):
        parser.error(
            "Repeat and timeout must be positive; warmup, reserve, and progress "
            "interval must be nonnegative"
        )
    if args.min_page_size < 64 or args.min_page_size & (args.min_page_size - 1):
        parser.error("--min-page-size must be a power of two and at least 64")
    if args.max_mpk_batch_size < 1:
        parser.error("--max-mpk-batch-size must be positive")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = (json.loads(args.source_prompts_file.read_text(encoding="utf-8"))
             if args.source_prompts_file.is_file() else list(DEFAULT_SEEDS))
    if (not isinstance(seeds, list) or not seeds
        or any(not isinstance(seed, str) for seed in seeds)):
        raise ValueError("Source file must contain at least one string seed")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_config = AutoConfig.from_pretrained(args.model)
    environment_path = args.output_dir / "environment.json"
    environment = collect_environment(args)
    import torch
    gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_total_gib = gpu_total_bytes / 2**30
    kv_budget_bytes = gpu_total_bytes - args.reserve_gib * 2**30
    if environment_path.is_file():
        previous_environment = json.loads(environment_path.read_text())
        if previous_environment != environment:
            completed_summaries = list(args.output_dir.glob("b*_in*_out*/summary.json"))
            previous_without_commit = {
                key: value for key, value in previous_environment.items()
                if key != "git_commit"
            }
            current_without_commit = {
                key: value for key, value in environment.items()
                if key != "git_commit"
            }
            code_only_change = previous_without_commit == current_without_commit
            if completed_summaries and not (
                args.allow_code_change_resume and code_only_change
            ):
                raise ValueError(
                    f"Environment changed after successful cases were recorded in "
                    f"{args.output_dir}; start a new output directory or use "
                    f"--allow-code-change-resume when only the Git commit changed"
                )
            environment_path.write_text(json.dumps(environment, indent=2) + "\n")
            print("Updated environment metadata for resumed sweep", flush=True)
    else:
        environment_path.write_text(json.dumps(environment, indent=2) + "\n")
    rows = []
    csv_path = args.output_dir / "raw_results.csv"
    total_cases = (len(selected_cases) if selected_cases is not None else
                   len(args.batch_sizes) * len(args.s_in_values) * len(args.s_out_values))
    progress = SweepProgress(total_cases, args.output_dir / "progress.json")

    for context_length in args.s_in_values:
        for s_out in args.s_out_values:
            page_size = page_size_for(context_length, (s_out - 1), args.min_page_size)
            feasible_batches = [
                batch_size for batch_size in args.batch_sizes
                if selected_cases is None or (batch_size, context_length, s_out) in selected_cases
                if batch_size <= args.max_mpk_batch_size
                if kv_bytes_for(model_config, batch_size, page_size) <= kv_budget_bytes
            ]
            prompts = []
            prompt_error = None
            if feasible_batches:
                try:
                    distinct_seeds = make_distinct_seeds(
                        tokenizer, seeds, max(feasible_batches), context_length
                    )
                    prompts = [make_prompt(tokenizer, seed, context_length)
                               for seed in distinct_seeds]
                    if len(set(prompts)) != len(prompts):
                        raise ValueError(f"Prompts are not distinct at S_IN={context_length}")
                except ValueError as error:
                    prompt_error = str(error)
            for batch_size in args.batch_sizes:
                if selected_cases is not None and (batch_size, context_length, s_out) not in selected_cases:
                    continue
                estimated_kv_gib = kv_bytes_for(model_config, batch_size, page_size) / 2**30
                case_dir = args.output_dir / f"b{batch_size}_in{context_length}_out{s_out}"
                case_label = f"B={batch_size} S_IN={context_length} S_OUT={s_out}"
                case_dir.mkdir(parents=True, exist_ok=True)
                if batch_size > args.max_mpk_batch_size:
                    reason = (f"Batch size exceeds the configured MPK limit of "
                              f"{args.max_mpk_batch_size}")
                    case_rows = unavailable_rows(batch_size, context_length, s_out, args,
                                                 environment, page_size, estimated_kv_gib,
                                                 gpu_total_gib, "unsupported_kernel", reason)
                    rows.extend(case_rows)
                    write_csv(csv_path, rows)
                    progress.finish_case(case_label, "unsupported_kernel")
                    continue
                if batch_size not in feasible_batches:
                    reason = (f"Estimated KV cache {estimated_kv_gib:.2f} GiB exceeds "
                              f"the {max(0, kv_budget_bytes / 2**30):.2f} GiB budget "
                              f"after reserving {args.reserve_gib:.2f} GiB")
                    case_rows = unavailable_rows(batch_size, context_length, s_out, args,
                                                 environment, page_size, estimated_kv_gib,
                                                 gpu_total_gib, "skipped_memory", reason)
                    rows.extend(case_rows)
                    write_csv(csv_path, rows)
                    print(f"Skipping B={batch_size}, S_IN={context_length}, S_OUT={s_out}: {reason}",
                          flush=True)
                    progress.finish_case(case_label, "skipped_memory")
                    continue
                if prompt_error is not None:
                    case_rows = unavailable_rows(batch_size, context_length, s_out, args,
                                                 environment, page_size, estimated_kv_gib,
                                                 gpu_total_gib, "failed_prompt", prompt_error)
                    rows.extend(case_rows)
                    write_csv(csv_path, rows)
                    print(f"Prompt preparation failed for B={batch_size}, "
                          f"S_IN={context_length}, S_OUT={s_out}: {prompt_error}",
                          flush=True)
                    progress.finish_case(case_label, "failed_prompt")
                    continue
                prompt_args = []
                if batch_size == 1:
                    prompt_path = case_dir / "prompt.txt"
                    prompt_path.write_text(prompts[0], encoding="utf-8")
                    prompt_args = ["--prompt-file", str(prompt_path)]
                else:
                    prompt_path = case_dir / "prompts.json"
                    prompt_path.write_text(json.dumps(prompts[:batch_size], ensure_ascii=False,
                                                      indent=2), encoding="utf-8")
                    prompt_args = ["--batch-prompts-file", str(prompt_path)]

                summary_path = case_dir / "summary.json"
                case_duration = None
                prompt_sha256 = hashlib.sha256(
                    json.dumps(prompts[:batch_size], ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                manifest_path = case_dir / "case_config.json"
                manifest = {
                    "batch_size": batch_size,
                    "context_length": context_length,
                    "s_in": context_length,
                    "s_out": s_out,
                    "decode_steps": (s_out - 1),
                    "chat_template": "user_only",
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "page_size": page_size,
                    "min_page_size": args.min_page_size,
                    "max_num_pages": batch_size,
                    "max_num_batched_tokens": max(8, batch_size),
                    "reserve_gib": args.reserve_gib,
                    "model": args.model,
                    "prompt_sha256": prompt_sha256,
                    "policies": list(POLICIES),
                }
                if summary_path.is_file():
                    if (not manifest_path.is_file()
                        or json.loads(manifest_path.read_text()) != manifest):
                        raise ValueError(f"Existing case has different inputs: {case_dir}")
                    print(f"Reusing B={batch_size}, S_IN={context_length}, S_OUT={s_out}: {summary_path}", flush=True)
                else:
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    command = [
                        sys.executable, str(BENCHMARK),
                        "--batch-size", str(batch_size),
                        "--context-length", str(context_length),
                        "--decode-steps", str((s_out - 1)),
                        "--warmup", str(args.warmup),
                        "--repeat", str(args.repeat),
                        "--timeout", str(args.timeout),
                        "--page-size", str(page_size),
                        "--max-num-pages", str(batch_size),
                        "--max-num-batched-tokens", str(max(8, batch_size)),
                        "--policies", "always", "decode-only", "prefill-only",
                        "--include-continuous-always",
                        "--allow-correctness-failures",
                        "--no-system-message",
                        "--model", args.model,
                        "--output-dir", str(case_dir),
                        *prompt_args,
                    ]
                    log_path = case_dir / "benchmark.log"
                    progress.show(f"{case_label}: starting")
                    case_started = time.monotonic()
                    returncode = run_benchmark(
                        command, log_path, progress, case_label,
                        args.progress_interval_seconds,
                    )
                    case_duration = time.monotonic() - case_started
                    if returncode:
                        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
                        reason = f"Benchmark exited {returncode}; see {log_path}"
                        case_rows = unavailable_rows(batch_size, context_length, s_out, args,
                                                     environment, page_size,
                                                     estimated_kv_gib, gpu_total_gib,
                                                     "failed", reason)
                        rows.extend(case_rows)
                        write_csv(csv_path, rows)
                        print(f"Case B={batch_size}, S_IN={context_length}, S_OUT={s_out} failed:\n{tail}",
                              flush=True)
                        progress.finish_case(case_label, "failed", case_duration)
                        continue
                case_rows = load_rows(summary_path, batch_size, context_length, s_out, args,
                                      environment, prompt_sha256, page_size,
                                      estimated_kv_gib, gpu_total_gib)
                rows.extend(case_rows)
                write_csv(csv_path, rows)
                for row in case_rows:
                    if row["policy"] == "always-continuous":
                        print(f"  continuous always: total="
                              f"{row['mean_continuous_total_ms']:.3f} ms, "
                              f"minimum first-30 matches={row['minimum_first30_matches']}",
                              flush=True)
                    else:
                        print(f"  {row['policy']}: decode speedup="
                              f"{row['decode_speedup_vs_normal']:.3f}, "
                              f"phase-sum speedup={row['phase_sum_speedup_vs_normal']:.3f}, "
                              f"minimum first-30 matches={row['minimum_first30_matches']}",
                              flush=True)
                outcome = ("correctness_failed" if any(
                    row["status"] == "correctness_failed" for row in case_rows
                ) else "completed")
                progress.finish_case(case_label, outcome, case_duration)

    case_statuses = {}
    for row in rows:
        case_statuses.setdefault((row["batch_size"], row["s_in"], row["s_out"]), set()).add(row["status"])
    completed = sum(states == {"completed"} for states in case_statuses.values())
    correctness_failed = sum("correctness_failed" in states for states in case_statuses.values())
    skipped = sum("skipped_memory" in states for states in case_statuses.values())
    unsupported = sum("unsupported_kernel" in states for states in case_statuses.values())
    failed = sum(bool({"failed", "failed_prompt"} & states) for states in case_statuses.values())
    print(f"Completed={completed}, correctness_failed={correctness_failed}, "
          f"skipped_memory={skipped}, unsupported_kernel={unsupported}, failed={failed}")
    print(f"Wrote {csv_path}")
    if args.fail_on_failed_cases and (failed or correctness_failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
