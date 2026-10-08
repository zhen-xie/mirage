"""Build a matched MPK/FlashInfer attention baseline over batch and KV size."""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo/qwen3/demo.py"
SUMMARIZER = ROOT / "tests/benchmarks/summarize_qwen3_mpk_profile.py"
Q_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 8, 128, 128
LAYERS, PROFILE_STEPS, S_OUT, COMPARE = 36, 9, 10, 10
DTYPE = torch.bfloat16


def terminate(process):
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run(command, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate(process)
            return "timeout"
    return "" if code == 0 else f"exit code {code}"


def token_batches(data):
    return data.get("token_ids_by_request") or [data.get("token_ids", [])]


def torch_command(args, s_in, output):
    max_seq = math.ceil((s_in + S_OUT) / 128) * 128
    return [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(S_OUT), "--page-size", str(max_seq),
        "--max-num-pages", "1", "--max-num-batched-requests", "1",
        "--max-num-batched-tokens", "8", "--ignore-eos",
        "--save-tokens", str(output),
    ]


def mpk_command(args, batch, s_in, case_dir):
    max_seq = math.ceil((s_in + S_OUT) / 128) * 128
    return [
        sys.executable, str(DEMO), "--model", args.model,
        "--input-length", str(s_in), "--max-seq-length", str(max_seq),
        "--max-new-tokens", str(S_OUT), "--page-size", str(max_seq),
        "--max-num-pages", str(batch),
        "--max-num-batched-requests", str(batch),
        "--max-num-batched-tokens", str(max(8, batch)), "--ignore-eos",
        "--save-tokens", str(case_dir / "tokens.json"),
        "--use-mirage", "--mpk-policy", "decode-only",
        "--mpk-attention", "auto",
        "--mpk-auto-split-kv-threshold", str(args.threshold),
        "--mpk-auto-attention-target-tasks", str(args.target_tasks),
        "--mpk-split-kv-chunk-size", "128",
        "--mpk-scheduler-policy", "round-robin",
        "--mpk-worker-policy", "fifo",
        "--normal-prefill-attention", "sdpa",
        "--mpk-kernel-cache-dir", str(case_dir / "cache"),
        "--profiling", "--trace-name", str(case_dir / "mpk_profile"),
        "--profiler-buffer-entries-per-block",
        str(args.profiler_entries_per_block),
        "--profiler-decode-start-step", "1",
        "--profiler-decode-num-steps", str(PROFILE_STEPS),
    ]


def plan_wrapper(wrapper, indptr, indices, last_page_len):
    positional = (indptr, indices, last_page_len, Q_HEADS, KV_HEADS,
                  HEAD_DIM, PAGE_SIZE)
    common = {"pos_encoding_mode": "NONE", "q_data_type": DTYPE}
    try:
        wrapper.plan(*positional, **common, kv_data_type=DTYPE)
    except TypeError:
        wrapper.plan(*positional, **common, data_type=DTYPE)


def flashinfer_case(flashinfer, workspace, batch, kv_length, warmup, repeat):
    pages = math.ceil(kv_length / PAGE_SIZE)
    total_pages = batch * pages
    q = torch.randn(batch, Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(total_pages, PAGE_SIZE, KV_HEADS, HEAD_DIM,
                    dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)
    indptr = torch.arange(0, total_pages + 1, pages,
                          dtype=torch.int32, device="cuda")
    indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last = torch.full((batch,), kv_length - (pages - 1) * PAGE_SIZE,
                      dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True)
    plan_wrapper(wrapper, indptr, indices, last)
    output = wrapper.run(q, (k, v))
    dense_k = k.reshape(batch, kv_length, KV_HEADS, HEAD_DIM)
    dense_v = v.reshape(batch, kv_length, KV_HEADS, HEAD_DIM)
    reference = F.scaled_dot_product_attention(
        q.unsqueeze(2), dense_k.permute(0, 2, 1, 3),
        dense_v.permute(0, 2, 1, 3), enable_gqa=True).squeeze(2)
    error = (output.float() - reference.float()).abs()
    for _ in range(warmup):
        wrapper.run(q, (k, v))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        wrapper.run(q, (k, v))
    end.record()
    end.synchronize()
    return {
        "flashinfer_ms_per_layer": start.elapsed_time(end) / repeat,
        "flashinfer_max_error": error.max().item(),
        "flashinfer_mean_error": error.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-sizes", default="1 8 32")
    parser.add_argument("--kv-lengths", default="128 512 1024")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--target-tasks", type=int, default=128)
    parser.add_argument("--profiler-entries-per-block", type=int, default=32768)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    batches = [int(value) for value in args.batch_sizes.split()]
    lengths = [int(value) for value in args.kv_lengths.split()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import flashinfer

    references = {}
    for length in lengths:
        output = args.output_dir / f"torch_in{length}.json"
        log = args.output_dir / f"torch_in{length}.log"
        print(f"Running Torch reference KV={length}...", flush=True)
        error = run(torch_command(args, length, output), log, args.timeout)
        if error:
            raise RuntimeError(f"Torch KV={length}: {error}; see {log}")
        references[length] = json.loads(
            output.read_text(encoding="utf-8"))["token_ids"]

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8,
                            device="cuda")
    rows = []
    failures = 0
    for batch in batches:
        for length in lengths:
            case_dir = args.output_dir / f"b{batch}_kv{length}"
            case_dir.mkdir(exist_ok=True)
            print(f"Running MPK B={batch} KV={length}...", flush=True)
            error = run(mpk_command(args, batch, length, case_dir),
                        case_dir / "run.log", args.timeout)
            reasons = [error] if error else []
            data = profile = None
            first10 = invalid = incomplete = None
            if not error:
                data = json.loads((case_dir / "tokens.json").read_text())
                actual = token_batches(data)
                first10 = min(sum(a == b for a, b in zip(
                    references[length][:COMPARE], row[:COMPARE]))
                    for row in actual)
                invalid = sum(data.get("invalid_token_counts_by_request", []))
                incomplete = sum(value != S_OUT for value in
                                 data.get("generate_lengths_by_request", []))
                if len(actual) != batch or first10 != COMPARE:
                    reasons.append(f"first-10={first10}, batch={len(actual)}")
                if invalid or incomplete:
                    reasons.append(
                        f"invalid={invalid}, incomplete={incomplete}")
                result = subprocess.run([
                    sys.executable, str(SUMMARIZER),
                    str(case_dir / "mpk_profile.csv"),
                    "--output-dir", str(case_dir / "summary")], cwd=ROOT)
                if result.returncode:
                    reasons.append("profile summary failed")
                else:
                    profile = json.loads((
                        case_dir / "summary/profile_summary.json").read_text())

            fi = flashinfer_case(
                flashinfer, workspace, batch, length,
                args.warmup, args.repeat)
            if fi["flashinfer_max_error"] > 0.05 or fi[
                    "flashinfer_mean_error"] > 0.005:
                reasons.append("FlashInfer correctness failed")

            attention = None
            if profile:
                attention = next(row for row in profile["categories"]
                                 if row["category"] == "attention")
            events = attention["events"] if attention else None
            tasks_per_layer_step = (
                events / (LAYERS * PROFILE_STEPS) if events else None)
            worker_ms = attention["worker_time_ms"] if attention else None
            lower_bound_ms = (
                worker_ms / (128 * PROFILE_STEPS)
                if worker_ms is not None else None)
            kv_bytes_step = (
                batch * LAYERS * KV_HEADS * 2 * length * HEAD_DIM * 2)
            row = {
                "batch_size": batch,
                "kv_length": length,
                "status": "failed" if reasons else "passed",
                "minimum_first10_matches": first10,
                "invalid_tokens": invalid,
                "incomplete_requests": incomplete,
                "mpk_attention": data.get("mpk_attention") if data else None,
                "mpk_split_kv_chunk_size": (
                    data.get("mpk_split_kv_chunk_size") if data else None),
                "mpk_attention_events": events,
                "mpk_tasks_per_layer_step": tasks_per_layer_step,
                "mpk_attention_mean_task_us": (
                    attention["mean_us"] if attention else None),
                "mpk_attention_worker_share": (
                    attention["worker_time_share"] if attention else None),
                "mpk_attention_work_lower_bound_ms_per_step": lower_bound_ms,
                "mpk_effective_kv_gbps_lower_bound": (
                    kv_bytes_step / (lower_bound_ms * 1e6)
                    if lower_bound_ms else None),
                **fi,
                "flashinfer_36_layer_ms": fi["flashinfer_ms_per_layer"] * LAYERS,
                "reason": "; ".join(reasons),
            }
            rows.append(row)
            failures += bool(reasons)
            print(
                f"B={batch} KV={length}: "
                f"{'PASS' if not reasons else 'FAIL'}; "
                f"MPK={row['mpk_attention']}, tasks/layer="
                f"{tasks_per_layer_step}; FI={fi['flashinfer_ms_per_layer']:.4f} "
                "ms/layer", flush=True)

    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "step": 37,
        "phase": "mpk_flashinfer_attention_baseline",
        "status": "passed" if not failures else "failed",
        "model": args.model,
        "batch_sizes": batches,
        "kv_lengths": lengths,
        "profile_steps": PROFILE_STEPS,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Step 37 attention baseline: {summary['status'].upper()}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
