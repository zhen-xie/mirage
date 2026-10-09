"""SGLang bs=1 decode-latency probe. Runs INSIDE the sglang conda env.

Measures TTFT (prefill) and per-token decode latency via the offline Engine
with streaming, so the numbers line up with what demo/qwen3/demo.py reports
for MPK (--mpk-policy decode-only): prefill_time_ms / decode_step_time_ms.
"""
import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    args = p.parse_args()

    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Deterministic synthetic prompt of exactly input_len tokens.
    filler = tok("hello", add_special_tokens=False).input_ids[0]
    input_ids = [filler] * args.input_len

    engine_kwargs = dict(
        model_path=args.model,
        tp_size=1,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        disable_radix_cache=True,   # no prefix-cache shortcut across reps
        max_running_requests=1,
        log_level="error",
    )
    if args.no_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True

    engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": args.output_len,
        "ignore_eos": True,
    }

    def one_run():
        stamps = []
        t0 = time.perf_counter()
        for _ in engine.generate(
            input_ids=input_ids, sampling_params=sampling_params, stream=True
        ):
            stamps.append(time.perf_counter())
        if len(stamps) < 2:
            raise RuntimeError("stream produced <2 chunks; cannot split TTFT/TPOT")
        ttft_ms = (stamps[0] - t0) * 1e3
        decode_ms = (stamps[-1] - stamps[0]) * 1e3
        steps = len(stamps) - 1
        return {
            "prefill_time_ms": ttft_ms,
            "decode_time_ms": decode_ms,
            "decode_steps": steps,
            "decode_step_time_ms": decode_ms / steps,
            "total_time_ms": ttft_ms + decode_ms,
            "generate_length": len(stamps),
        }

    for _ in range(args.warmup):
        one_run()

    runs = [one_run() for _ in range(args.reps)]
    engine.shutdown()

    with open(args.out, "w") as f:
        json.dump(
            {
                "framework": "sglang",
                "model": args.model,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "cuda_graph": not args.no_cuda_graph,
                "runs": runs,
            },
            f,
            indent=2,
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
