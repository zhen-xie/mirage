"""Exploratory single-request Qwen3 decode timing on normal and MPK.

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


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def load_prompt(args):
    path = args.prompt_file or (ROOT / "tests" / "benchmarks" / "baselines"
                                / f"new_server_prompt_{args.context_length}.txt")
    prompt = path.read_text()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    actual_length = len(tokenizer(rendered)["input_ids"])
    if actual_length != args.context_length:
        raise ValueError(f"Prompt has {actual_length} tokens, expected {args.context_length}")
    return prompt


def run_case(args, prompt, backend, index, warmup):
    label = "warmup" if warmup else "repeat"
    stem = f"{label}_{index}_{backend}"
    output = args.output_dir / f"{stem}.json"
    log = args.output_dir / f"{stem}.log"
    command = [
        sys.executable, str(DEMO), "--backend", "normal" if backend == "normal" else "mpk",
        "--prompt", prompt,
        "--max-seq-length", str(args.context_length + args.decode_steps + 1),
        "--max-new-tokens", str(args.decode_steps + 1),
        "--ignore-eos", "--phase-timing", "--save-tokens", str(output),
        "--model", args.model,
    ]
    if backend == "mpk":
        command += ["--mpk-policy", "decode-only"]
    with log.open("w") as log_file:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log_file,
                                       stderr=subprocess.STDOUT, timeout=args.timeout,
                                       check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{backend} timed out; see {log}") from error
    if completed.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{backend} exited {completed.returncode}; {log}\n{tail}")
    data = json.loads(output.read_text())
    if data["prompt_length"] != args.context_length:
        raise ValueError(f"Unexpected prompt length in {output}")
    if data["generate_length"] != args.decode_steps + 1:
        raise ValueError(f"Unexpected generation length in {output}")
    timing = data["phase_timing"]
    if timing["prefill_ms"] <= 0 or timing["decode_ms"] <= 0:
        raise ValueError(f"Invalid phase timing in {output}")
    if backend == "normal" and len(timing["decode_step_ms"]) != args.decode_steps:
        raise ValueError(f"Missing normal decode steps in {output}")
    if backend == "mpk" and timing["decode_steps"] != args.decode_steps:
        raise ValueError(f"Wrong MPK decode step count in {output}")
    return data


def summarize(samples, backend, decode_steps):
    totals = [sample["phase_timing"]["decode_ms"] for sample in samples]
    mean_total = statistics.mean(totals)
    per_step = [value for sample in samples
                for value in (sample["phase_timing"]["decode_step_ms"] or [])]
    return {
        "backend": backend,
        "repeat_total_decode_ms": totals,
        "mean_total_decode_ms": mean_total,
        "mean_step_latency_ms": mean_total / decode_steps,
        "median_step_latency_ms": percentile(per_step, 0.5) if per_step else None,
        "p90_step_latency_ms": percentile(per_step, 0.9) if per_step else None,
        "p99_step_latency_ms": percentile(per_step, 0.99) if per_step else None,
        "tokens_per_second": 1000 * decode_steps / mean_total,
        "relative_total_range": ((max(totals) - min(totals)) / mean_total
                                 if len(totals) > 1 else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, required=True,
                        help="Decode iterations after the first token from prefill")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "qwen3_decode_b1")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("Current normal/decode-only demo supports batch-size=1 only")
    if args.context_length < 1 or args.context_length >= 4096:
        parser.error("Current decode-only resume requires context-length in [1, 4095]")
    if args.decode_steps < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("decode-steps and repeat must be positive; warmup must be nonnegative")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(args)

    samples = {"normal": [], "mpk": []}
    for index in range(args.warmup + args.repeat):
        warmup = index < args.warmup
        for backend in (("normal", "mpk") if index % 2 == 0 else ("mpk", "normal")):
            print(f"{backend} {'warmup' if warmup else 'repeat'} {index + 1}/{args.warmup + args.repeat}",
                  flush=True)
            data = run_case(args, prompt, backend, index, warmup)
            if not warmup:
                samples[backend].append(data)

    for normal, mpk in zip(samples["normal"], samples["mpk"]):
        if normal["token_ids"][:30] != mpk["token_ids"][:30]:
            raise ValueError("Normal and MPK decode-only differ in first 30 saved tokens")

    summary = {
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "decode_steps": args.decode_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing_scope": "CUDA events around decode only; separate cold demo process per sample",
        "warmup_note": "Warmup runs are discarded processes; they do not warm subsequent processes",
        "per_step_note": "MPK persistent kernel exposes only whole-decode timing; its per-step percentiles are null",
        "normal": summarize(samples["normal"], "normal", args.decode_steps),
        "mpk_decode_only": summarize(samples["mpk"], "mpk", args.decode_steps),
    }
    destination = args.output_dir / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.repeat < 3:
        print("WARNING: fewer than 3 repeats; stability is not established")
    if any(summary[name]["relative_total_range"] is not None
           and summary[name]["relative_total_range"] > 0.1
           for name in ("normal", "mpk_decode_only")):
        print("WARNING: decode timing range exceeds 10%; investigate benchmark stability")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
