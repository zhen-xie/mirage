"""MPK vs SGLang head-to-head at batch size 1.

Drives both conda envs, repeats each run, and reports prefill (TTFT),
per-token decode latency (TPOT) and decode throughput with median + spread.

Run from the repo root with any python3 that has conda on PATH:

    python benchmark/mpk_vs_sglang/run_bench.py \
        --input-len 128 --output-len 256 --reps 5
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]


def sh(cmd, env=None, cwd=None):
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd or REPO, env=env, text=True,
                       capture_output=True)
    dt = time.perf_counter() - t0
    tail = "\n".join(r.stdout.strip().splitlines()[-25:])
    print(tail, flush=True)
    if r.returncode != 0:
        print("--- stderr ---\n" + "\n".join(
            r.stderr.strip().splitlines()[-40:]), flush=True)
        raise SystemExit(f"command failed ({r.returncode}) after {dt:.1f}s")
    return r.stdout


def conda(env_name, *argv):
    return ["conda", "run", "-n", env_name, "--no-capture-output", *argv]


def run_mpk(args, outdir, use_mirage):
    tag = "mpk" if use_mirage else "torch"
    runs = []
    cache = outdir / "mpk_kernel_cache"
    for i in range(args.warmup + args.reps):
        jf = outdir / f"{tag}_run{i}.json"
        cmd = conda(
            args.mpk_env, "python", "demo/qwen3/demo.py",
            "--model", args.model,
            "--input-length", str(args.input_len),
            "--max-new-tokens", str(args.output_len),
            "--ignore-eos",
            "--save-tokens", str(jf),
        )
        if use_mirage:
            cmd += ["--use-mirage", "--mpk-policy", "decode-only",
                    "--mpk-kernel-cache-dir", str(cache)]
        env = dict(os.environ, MIRAGE_HOME=str(REPO),
                   CUDA_VISIBLE_DEVICES=args.gpu)
        sh(cmd, env=env)
        d = json.loads(jf.read_text())
        # Sanity gate: a run that decoded the wrong number of steps, or emitted
        # out-of-vocab ids, is not a latency measurement. Fail loudly.
        got = d.get("generate_length")
        bad = d.get("invalid_token_count") or 0
        if got != args.output_len:
            raise SystemExit(
                f"[{tag} run{i}] generated {got} tokens, expected "
                f"{args.output_len}. This is a correctness failure, not a "
                f"slow run -- the latency numbers are meaningless. See {jf}"
            )
        if bad:
            raise SystemExit(
                f"[{tag} run{i}] {bad} out-of-vocab token ids. The decode "
                f"path is producing garbage; fix correctness first. See {jf}"
            )
        if i < args.warmup:
            continue
        runs.append({
            "prefill_time_ms": d.get("prefill_time_ms"),
            "decode_time_ms": d.get("decode_time_ms"),
            "decode_steps": d.get("decode_steps"),
            "decode_step_time_ms": d.get("decode_step_time_ms"),
            "total_time_ms": d.get("total_time_ms"),
            "generate_length": d.get("generate_length"),
        })
    return {"framework": tag, "runs": runs}


def run_sglang(args, outdir):
    jf = outdir / "sglang.json"
    cmd = conda(
        args.sgl_env, "python",
        str(REPO / "benchmark/mpk_vs_sglang/sgl_bench.py"),
        "--model", args.model,
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
        "--out", str(jf),
    )
    if args.sgl_no_cuda_graph:
        cmd.append("--no-cuda-graph")
    sh(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu))
    return json.loads(jf.read_text())


def agg(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return None
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def fmt(a, unit="ms"):
    if a is None:
        return "n/a"
    return f"{a['median']:.3f} {unit}  [{a['min']:.3f}–{a['max']:.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--gpu", default="0")
    p.add_argument("--mpk-env", default="mirage-mpk")
    p.add_argument("--sgl-env", default="sglang-bench")
    p.add_argument("--include-torch", action="store_true",
                   help="also run the demo's Triton/FlashInfer path as a third bar")
    p.add_argument("--sgl-no-cuda-graph", action="store_true")
    p.add_argument("--skip-mpk", action="store_true")
    p.add_argument("--skip-sglang", action="store_true")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = pathlib.Path(args.outdir or (REPO / f"outputs/bench_{stamp}"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {outdir}")

    results = {}
    if not args.skip_mpk:
        results["mpk"] = run_mpk(args, outdir, use_mirage=True)
    if args.include_torch:
        results["torch"] = run_mpk(args, outdir, use_mirage=False)
    if not args.skip_sglang:
        results["sglang"] = run_sglang(args, outdir)

    rows = []
    for name, res in results.items():
        runs = res["runs"]
        pre = agg(runs, "prefill_time_ms")
        step = agg(runs, "decode_step_time_ms")
        tput = None
        if step:
            tput = {k: (1e3 / v if k != "n" else v)
                    for k, v in step.items()}
            tput["min"], tput["max"] = tput["max"], tput["min"]
        rows.append((name, pre, step, tput))

    lines = [
        f"# MPK vs SGLang — {args.model}, bs=1, "
        f"in={args.input_len} out={args.output_len}, {args.reps} reps "
        f"(+{args.warmup} warmup)",
        "",
        "| framework | prefill / TTFT | decode step (TPOT) | decode tok/s |",
        "|---|---|---|---|",
    ]
    for name, pre, step, tput in rows:
        lines.append(
            f"| {name} | {fmt(pre)} | {fmt(step)} | "
            f"{fmt(tput, 'tok/s') if tput else 'n/a'} |"
        )
    if "mpk" in results and "sglang" in results:
        m = agg(results["mpk"]["runs"], "decode_step_time_ms")
        s = agg(results["sglang"]["runs"], "decode_step_time_ms")
        if m and s:
            lines += ["", f"**MPK decode speedup vs SGLang: "
                          f"{s['median'] / m['median']:.2f}x**"]
    lines += ["", "Median [min–max] over reps. Warmup runs discarded."]

    report = "\n".join(lines)
    print("\n" + report)
    (outdir / "summary.md").write_text(report)
    (outdir / "raw.json").write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2))
    print(f"\nwrote {outdir/'summary.md'} and {outdir/'raw.json'}")


if __name__ == "__main__":
    main()
