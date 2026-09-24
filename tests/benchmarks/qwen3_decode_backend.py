"""Exploratory Qwen3 decode timing across normal and MPK policies.

Each sample starts a fresh demo process. CUDA event timing excludes model load,
prefill, and MPK compilation, but each process still has cold runtime state.
This is not yet the steady-state multi-request sweep from Steps 8-9.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo" / "qwen3" / "demo.py"
TOKEN_MATCH_NUMERATOR = 20
TOKEN_MATCH_DENOMINATOR = 30


def required_token_matches(compared_tokens):
    """Scale the agreed 20/30 positional gate to shorter generations."""
    return ((compared_tokens * TOKEN_MATCH_NUMERATOR
             + TOKEN_MATCH_DENOMINATOR - 1) // TOKEN_MATCH_DENOMINATOR)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load_prompt(args):
    if args.batch_size > 1:
        prompts = json.loads(args.batch_prompts_file.read_text())
        if (not isinstance(prompts, list) or len(prompts) != args.batch_size
            or any(not isinstance(prompt, str) for prompt in prompts)):
            raise ValueError("Batch prompt file must contain one string per request")
    else:
        path = args.prompt_file or (ROOT / "tests" / "benchmarks" / "baselines"
                                    / f"new_server_prompt_{args.context_length}.txt")
        prompts = [path.read_text()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for request_id, prompt in enumerate(prompts):
        messages = []
        if not args.no_system_message:
            messages.append({"role": "system", "content":
                             "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        actual_length = len(tokenizer(rendered)["input_ids"])
        if actual_length != args.context_length:
            raise ValueError(f"Request {request_id} has {actual_length} input tokens, expected {args.context_length}")
    return prompts[0]


def run_case(args, prompt, policy, index, warmup):
    label = "warmup" if warmup else "repeat"
    stem = f"{label}_{index}_{policy}"
    output = args.output_dir / f"{stem}.json"
    log = args.output_dir / f"{stem}.log"
    command = [
        sys.executable, str(DEMO), "--backend", "normal" if policy == "normal" else "mpk",
        "--prompt", prompt,
        "--max-seq-length", str(args.context_length + args.decode_steps + 1),
        "--max-new-tokens", str(args.decode_steps + 1),
        "--ignore-eos", "--save-tokens", str(output),
        "--model", args.model,
    ]
    if policy != "always-continuous":
        command.append("--phase-timing")
    if args.no_system_message:
        command.append("--no-system-message")
    if policy != "normal":
        command += ["--mpk-policy", "always" if policy == "always-continuous" else policy]
    if args.batch_size > 1:
        command += ["--max-num-batched-requests", str(args.batch_size),
                    "--batch-prompts-file", str(args.batch_prompts_file)]
    command += ["--page-size", str(args.page_size),
                "--max-num-pages", str(args.max_num_pages),
                "--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    with log.open("w") as log_file:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log_file,
                                       stderr=subprocess.STDOUT, timeout=args.timeout,
                                       check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{policy} timed out; see {log}") from error
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{policy} exited {completed.returncode}; {log}\n{tail}")
    data = json.loads(output.read_text())
    if data["prompt_length"] != args.context_length:
        raise ValueError(f"Unexpected prompt length in {output}")
    if data["generate_length"] != args.decode_steps + 1:
        raise ValueError(f"Unexpected generation length in {output}")
    if args.batch_size > 1:
        requests = data.get("token_ids_by_request")
        if data.get("batch_size") != args.batch_size or not isinstance(requests, list) or len(requests) != args.batch_size:
            raise ValueError(f"Missing per-request tokens in {output}")
    if policy == "always-continuous":
        if data["mode"] != "mpk" or data["latency_ms_per_token"] <= 0:
            raise ValueError(f"Missing continuous always timing in {output}")
        return data
    timing = data["phase_timing"]
    if timing["prefill_ms"] <= 0 or timing["decode_ms"] <= 0:
        raise ValueError(f"Invalid phase timing in {output}")
    if policy in ("normal", "prefill-only") and len(timing["decode_step_ms"]) != args.decode_steps:
        raise ValueError(f"Missing normal decode steps in {output}")
    if policy in ("always", "decode-only") and timing["decode_steps"] != args.decode_steps:
        raise ValueError(f"Wrong MPK decode step count in {output}")
    return data


def summarize(samples, backend, decode_steps, batch_size):
    prefills = [sample["phase_timing"]["prefill_ms"] for sample in samples]
    totals = [sample["phase_timing"]["decode_ms"] for sample in samples]
    mean_total = statistics.mean(totals)
    mean_prefill = statistics.mean(prefills)
    per_step = [value for sample in samples
                for value in (sample["phase_timing"]["decode_step_ms"] or [])]
    return {
        "backend": backend,
        "repeat_prefill_ms": prefills,
        "mean_prefill_ms": mean_prefill,
        "repeat_total_decode_ms": totals,
        "mean_total_decode_ms": mean_total,
        "mean_prefill_plus_decode_ms": statistics.mean(
            prefill + decode for prefill, decode in zip(prefills, totals)),
        "mean_step_latency_ms": mean_total / decode_steps,
        "median_step_latency_ms": percentile(per_step, 0.5) if per_step else None,
        "p90_step_latency_ms": percentile(per_step, 0.9) if per_step else None,
        "p99_step_latency_ms": percentile(per_step, 0.99) if per_step else None,
        "tokens_per_second": 1000 * decode_steps * batch_size / mean_total,
        "generated_tokens_per_second_including_prefill": (
            1000 * (decode_steps + 1) * batch_size / (mean_prefill + mean_total)),
        "relative_total_range": ((max(totals) - min(totals)) / mean_total
                                 if len(totals) > 1 else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-num-pages", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--decode-steps", type=int, required=True,
                        help="Decode iterations after the first token from prefill")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--policies", nargs="+", choices=("always", "decode-only", "prefill-only"),
                        default=["decode-only"], help="MPK policies to compare with normal")
    parser.add_argument("--include-continuous-always", action="store_true",
                        help="Also run continuous MPK always and record its full kernel duration")
    parser.add_argument("--allow-correctness-failures", action="store_true",
                        help="Record first-30 mismatches without rejecting the timing sample")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--no-system-message", action="store_true")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--batch-prompts-file", type=Path,
                        help="JSON array of distinct, equal-length prompts for batch-size > 1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "qwen3_decode_b1")
    args = parser.parse_args()
    if len(set(args.policies)) != len(args.policies):
        parser.error("--policies must not contain duplicates")
    if args.batch_size < 1 or args.batch_size > 128:
        parser.error("batch-size must be in [1, 128] for this benchmark")
    args.max_num_pages = args.max_num_pages or max(16, args.batch_size)
    args.max_num_batched_tokens = args.max_num_batched_tokens or max(8, args.batch_size)
    if args.max_num_pages < args.batch_size or args.max_num_batched_tokens < args.batch_size:
        parser.error("Page and batched-token capacities must cover every request")
    if args.batch_size > 1 and (args.batch_prompts_file is None or args.prompt_file is not None):
        parser.error("batch-size > 1 requires --batch-prompts-file and no --prompt-file")
    if args.batch_size == 1 and args.batch_prompts_file is not None:
        parser.error("--batch-prompts-file requires batch-size > 1")
    if args.context_length < 1 or args.context_length >= args.page_size:
        parser.error("Context length must be positive and shorter than one KV page")
    if args.decode_steps < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("decode-steps and repeat must be positive; warmup must be nonnegative")
    if args.context_length + args.decode_steps + 1 > args.page_size:
        parser.error("Each batched sequence must fit in one KV page")
    if args.batch_prompts_file is not None:
        args.batch_prompts_file = args.batch_prompts_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(args)

    policies = ["normal", *args.policies]
    if args.include_continuous_always:
        if "always" not in args.policies:
            parser.error("--include-continuous-always requires --policies always")
        policies.append("always-continuous")
    samples = {policy: [] for policy in policies}
    for index in range(args.warmup + args.repeat):
        warmup = index < args.warmup
        for policy in (policies if index % 2 == 0 else list(reversed(policies))):
            print(f"{policy} {'warmup' if warmup else 'repeat'} {index + 1}/{args.warmup + args.repeat}",
                  flush=True)
            data = run_case(args, prompt, policy, index, warmup)
            if not warmup:
                samples[policy].append(data)

    match_counts = {}
    invalid_token_counts = {}
    incomplete_request_counts = {}
    compared_policies = [*args.policies]
    if args.include_continuous_always:
        compared_policies.append("always-continuous")
    compared_tokens = min(30, args.decode_steps + 1)
    required_matches = required_token_matches(compared_tokens)
    for policy in compared_policies:
        token_match_counts = []
        policy_invalid_counts = []
        policy_incomplete_counts = []
        for normal, mpk in zip(samples["normal"], samples[policy]):
            normal_requests = (normal["token_ids_by_request"] if args.batch_size > 1
                               else [normal["token_ids"]])
            mpk_requests = (mpk["token_ids_by_request"] if args.batch_size > 1
                            else [mpk["token_ids"]])
            repeat_counts = []
            for request_id, (normal_tokens, mpk_tokens) in enumerate(zip(normal_requests, mpk_requests)):
                if min(len(normal_tokens), len(mpk_tokens)) < compared_tokens:
                    raise ValueError(f"Request {request_id} has fewer than {compared_tokens} saved tokens")
                matches = sum(a == b for a, b in zip(normal_tokens[:compared_tokens], mpk_tokens[:compared_tokens]))
                repeat_counts.append(matches)
            total_matches = sum(repeat_counts)
            total_positions = len(repeat_counts) * compared_tokens
            passing_requests = sum(
                matches >= required_matches for matches in repeat_counts
            )
            aggregate_passed = (
                total_matches * TOKEN_MATCH_DENOMINATOR
                >= total_positions * TOKEN_MATCH_NUMERATOR
            )
            request_pass_rate_passed = (
                passing_requests * TOKEN_MATCH_DENOMINATOR
                >= len(repeat_counts) * TOKEN_MATCH_NUMERATOR
            )
            invalid_count = sum(
                len(entries)
                for entries in mpk.get(
                    "invalid_token_ids_by_request", {}
                ).values()
            )
            generation_lengths = mpk.get(
                "generate_lengths_by_request",
                [mpk["generate_length"]] * len(repeat_counts),
            )
            incomplete_count = sum(
                length != args.decode_steps + 1
                for length in generation_lengths
            )
            policy_invalid_counts.append(invalid_count)
            policy_incomplete_counts.append(incomplete_count)
            if ((not aggregate_passed or not request_pass_rate_passed
                 or invalid_count > 0 or incomplete_count > 0)
                and not args.allow_correctness_failures):
                raise ValueError(
                    f"{policy}: batch positional matches are "
                    f"{total_matches}/{total_positions}, and "
                    f"{passing_requests}/{len(repeat_counts)} requests meet "
                    f"the per-request {required_matches}/{compared_tokens} "
                    "gate; both batch ratios must be at least two thirds; "
                    f"invalid token IDs: {invalid_count}; "
                    f"incomplete requests: {incomplete_count}")
            token_match_counts.append(repeat_counts)
        match_counts[policy] = token_match_counts
        invalid_token_counts[policy] = policy_invalid_counts
        incomplete_request_counts[policy] = policy_incomplete_counts

    summary = {
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "page_size": args.page_size,
        "max_num_pages": args.max_num_pages,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing_scope": "CUDA events around each phase; continuous always uses one full-kernel CUDA duration; separate cold demo process per sample",
        "warmup_note": "Warmup runs are discarded processes; they do not warm subsequent processes",
        "per_step_note": "MPK persistent kernel exposes only whole-decode timing; its per-step percentiles are null",
        "always_timing_mode": "two MPK launches with a prefill boundary; continuous always may generate different tokens",
        "continuous_always_note": "Continuous always has one kernel duration; prefill/decode timing is unavailable",
        "first_30_token_matches_by_policy": match_counts,
        "compared_token_positions": compared_tokens,
        "required_token_matches": required_matches,
        "batch_match_fraction_by_policy_per_repeat": {
            policy: [
                sum(counts) / (len(counts) * compared_tokens)
                for counts in repeats
            ]
            for policy, repeats in match_counts.items()
        },
        "passing_request_fraction_by_policy_per_repeat": {
            policy: [
                sum(matches >= required_matches for matches in counts)
                / len(counts)
                for counts in repeats
            ]
            for policy, repeats in match_counts.items()
        },
        "invalid_token_count_by_policy_per_repeat": invalid_token_counts,
        "incomplete_request_count_by_policy_per_repeat": (
            incomplete_request_counts
        ),
        "first_30_gate_passed_by_policy": {
            policy: all(
                sum(counts) * TOKEN_MATCH_DENOMINATOR
                    >= len(counts) * compared_tokens * TOKEN_MATCH_NUMERATOR
                and sum(matches >= required_matches for matches in counts)
                    * TOKEN_MATCH_DENOMINATOR
                    >= len(counts) * TOKEN_MATCH_NUMERATOR
                and invalid_count == 0
                and incomplete_count == 0
                for counts, invalid_count, incomplete_count in zip(
                    repeats,
                    invalid_token_counts[policy],
                    incomplete_request_counts[policy],
                )
            )
            for policy, repeats in match_counts.items()
        },
        "token_match_gate": (
            "For every repeat, at least two thirds of all compared token "
            "positions match and at least two thirds of requests meet the "
            "per-request two-thirds positional threshold"),
    }
    if args.policies == ["decode-only"]:
        summary["first_30_token_matches_by_request_per_repeat"] = match_counts["decode-only"]
        summary["first_30_token_matches_per_repeat"] = (
            [counts[0] for counts in match_counts["decode-only"]]
            if args.batch_size == 1 else None
        )
    summary["normal"] = summarize(samples["normal"], "normal", args.decode_steps, args.batch_size)
    for policy in args.policies:
        summary[f"mpk_{policy.replace('-', '_')}"] = summarize(
            samples[policy], "mpk" if policy != "prefill-only" else "normal",
            args.decode_steps, args.batch_size)
    if args.include_continuous_always:
        totals = [sample["latency_ms_per_token"] *
                  (sample["prompt_length"] + sample["generate_length"])
                  for sample in samples["always-continuous"]]
        mean_total = statistics.mean(totals)
        summary["mpk_always_continuous"] = {
            "backend": "mpk",
            "repeat_total_ms": totals,
            "mean_total_ms": mean_total,
            "generated_tokens_per_second": (
                1000 * (args.decode_steps + 1) * args.batch_size / mean_total),
            "relative_total_range": ((max(totals) - min(totals)) / mean_total
                                     if len(totals) > 1 else None),
        }
    destination = args.output_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.repeat < 3:
        print("WARNING: fewer than 3 repeats; stability is not established")
    if any(summary[name]["relative_total_range"] is not None
           and summary[name]["relative_total_range"] > 0.1
           for name in ("normal", *(f"mpk_{policy.replace('-', '_')}" for policy in args.policies),
                        *(("mpk_always_continuous",) if args.include_continuous_always else ()))):
        print("WARNING: timing range exceeds 10%; investigate benchmark stability")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
