from models.modeling_qwen3 import Qwen3ForCausalLM, plan_qwen3_kv_cache
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json, time

from models.qwen3_shard_loader import Qwen3ShardLoader
from mirage.mpk.base_dynamic_shard_loader import ShardType
from mirage.mpk.models.utils import grid_for_splitk_linear_layer
from mirage.mpk.kv_planner import resolve_pool_size


mapping = {
    "embed_tokens": {"name": "embed", "shard_type": [(ShardType.NONE,)]},
    "input_layernorm": {"name": "attn_norm", "shard_type": [(ShardType.NONE,)]},
    "q_proj" : {"name": "wq", "shard_type": [(ShardType.COL_PARALLEL,)]},
	"q_norm" : {"name": "wq", "shard_type": [(ShardType.NONE,)]}, 
    "k_proj": {"name": "wk", "shard_type": [(ShardType.COL_PARALLEL,)]},
    "k_norm": {"name": "wk", "shard_type": [(ShardType.NONE,)]},
    "v_proj": {"name": "wv", "shard_type": [(ShardType.COL_PARALLEL,)]},
	"o_proj" : {"name": "wo", "shard_type": [(ShardType.ROW_PARALLEL)]},
    "post_attention_layernorm": {"name": "post_norm", "shard_type": [(ShardType.NONE)]}, 
	"gate": {"name": "gate", "shard_type": [(ShardType.NONE)]}, # router gate
	"gate_proj": {"name": "w1", "shard_type": [(ShardType.COL_PARALLEL)]}, 
	"down_proj": {"name": "w2", "shard_type": [(ShardType.ROW_PARALLEL)]}, 
	"up_proj": {"name": "w3", "shard_type": [(ShardType.COL_PARALLEL)]}, 
    "norm": {"name": "norm", "shard_type": [(ShardType.NONE)]},
    "lm_head": {"name": "head", "shard_type": [(ShardType.NONE)]}
}

DEFAULT_SAVE_DIR = os.path.join("outputs", "qwen3")
MAX_SAVE_TOKENS = 4096

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size % 64 == 0 and not use_cutlass_kernel:
        # TODO(Wenqin): If we set OUTPUT_SIZE too much for PTX linear kernel,
        # there is some regression.
        return size // 64
    if size / 96 > 400:
        # TODO: An add-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpected result (not same output for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, "FATAL: Linear layer size not supported, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument(
        "--mpk-policy",
        choices=("always", "decode-only"),
        default="always",
        help=(
            "Select whether MPK runs the whole request or only decode after "
            "a Torch prefill. This option requires --use-mirage."
        ),
    )
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Tokens per page")
    parser.add_argument("--kv-budget", type=str, default=None,
                        help="Memory budget for KV cache as a size('24GiB'). Exclusive with --max-num-pages")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages. Exclusive with --kv-budget")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument(
        "--mpk-kernel-cache-dir",
        help=(
            "Load a compatible MPK kernel from this directory, or compile "
            "and populate it on a cache miss."
        ),
    )
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
    )
    parser.add_argument(
        "--profiler-buffer-entries-per-block",
        type=int,
        default=32768,
        help=(
            "Profiler uint64 entries reserved per persistent worker block. "
            "Each paired task event consumes two entries."
        ),
    )
    parser.add_argument(
        "--profiler-decode-start-step",
        type=int,
        default=1,
        help="One-based first decode iteration recorded by the MPK profiler.",
    )
    parser.add_argument(
        "--profiler-decode-num-steps",
        type=int,
        default=None,
        help="Number of consecutive decode iterations recorded by the MPK profiler.",
    )
    parser.add_argument(
        "--profile-prefill-stages",
        action="store_true",
        help="Record CUDA-event timing for embedding, layers, norm, and LM head.",
    )
    parser.add_argument(
        "--normal-prefill-attention",
        choices=("sdpa", "flashinfer"),
        default="sdpa",
        help="Attention implementation used by the Torch prefill path.",
    )
    parser.add_argument(
        "--prefill-warmup-runs",
        type=int,
        default=0,
        help="Run unmeasured Torch prefill forwards in the current process.",
    )
    parser.add_argument(
        "--normal-prefill-cuda-graph",
        action="store_true",
        help="Capture and replay the fixed-shape Torch prefill forward.",
    )
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=512,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    parser.add_argument(
        "--no-use-cutlass-kernel",
        action="store_false",
        dest="use_cutlass_kernel",
        default=True,
        help="Not use the cutlass version kernel.",
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos token during generation")

    # -------- Args for CI tests ----------
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Decode cap for CI determinism")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default off)")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top_k logits when sampling (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    parser.add_argument(
        "--sampling-topk-max",
        type=int,
        default=32,
        help=(
            "Candidates kept per vocabulary chunk when sampling. Upper bound "
            "on --top_k and on the nucleus size that can be served exactly."
        ),
    )
    parser.add_argument(
        "--save-tokens",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Optionally dump first N generated token_ids, text, and latency to JSON. "
            "If path omitted, saves to outputs/qwen3/{torch_output.json|mpk_output.json}."
        ),
    )
    parser.add_argument(
        "--capture-final-logits-topk",
        type=int,
        default=0,
        help=(
            "Attach the MPK LM-head output and save the final step's top-k "
            "logits. Intended for short correctness diagnostics."
        ),
    )
    parser.add_argument("--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Custom prompt text to generate from.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help=(
            "Use a deterministic synthetic prompt with exactly this many "
            "tokens. This bypasses chat-template tokenization."
        ),
    )

    parser.add_argument("--split-kv-cache", action="store_true", help="Use split-kv cache")
    parser.add_argument(
        "--mpk-attention",
        choices=("default", "split-kv", "auto"),
        default="default",
        help=(
            "Select the MPK attention implementation. Auto uses default "
            "attention for short workloads and split-KV for longer workloads."
        ),
    )
    parser.add_argument(
        "--mpk-auto-split-kv-threshold",
        type=int,
        default=256,
        help=(
            "With --mpk-attention auto, use split-KV when max sequence "
            "length exceeds this value."
        ),
    )
    parser.add_argument(
        "--mpk-split-kv-chunk-size",
        type=int,
        default=128,
        help="Number of KV tokens processed by each MPK split-KV task.",
    )
    args = parser.parse_args()
    if args.mpk_policy != "always" and not args.use_mirage:
        parser.error("--mpk-policy requires --use-mirage")
    if args.prefill_warmup_runs < 0:
        parser.error("--prefill-warmup-runs must be non-negative")
    if args.capture_final_logits_topk < 0:
        parser.error("--capture-final-logits-topk must be non-negative")
    if args.normal_prefill_cuda_graph:
        if not args.use_mirage or args.mpk_policy != "decode-only":
            parser.error(
                "--normal-prefill-cuda-graph requires MPK decode-only"
            )
        if args.prefill_warmup_runs < 1:
            parser.error(
                "--normal-prefill-cuda-graph requires at least one "
                "--prefill-warmup-runs"
            )
        if args.profile_prefill_stages:
            parser.error(
                "--normal-prefill-cuda-graph cannot be combined with "
                "--profile-prefill-stages"
            )
    if args.split_kv_cache and args.mpk_attention != "default":
        parser.error(
            "--split-kv-cache cannot be combined with a non-default "
            "--mpk-attention value"
        )
    if args.mpk_auto_split_kv_threshold <= 0:
        parser.error("--mpk-auto-split-kv-threshold must be positive")
    requested_mpk_attention = (
        "split-kv" if args.split_kv_cache else args.mpk_attention
    )
    if requested_mpk_attention == "auto":
        resolved_mpk_attention = (
            "split-kv"
            if args.max_seq_length > args.mpk_auto_split_kv_threshold
            else "default"
        )
    else:
        resolved_mpk_attention = requested_mpk_attention
    args.split_kv_cache = resolved_mpk_attention == "split-kv"
    if requested_mpk_attention != "default" and not args.use_mirage:
        parser.error("--mpk-attention requires --use-mirage")
    if args.mpk_split_kv_chunk_size <= 0:
        parser.error("--mpk-split-kv-chunk-size must be positive")
    if args.profiler_decode_start_step < 1:
        parser.error("--profiler-decode-start-step must be at least 1")
    if (args.profiler_decode_num_steps is not None
            and args.profiler_decode_num_steps < 1):
        parser.error("--profiler-decode-num-steps must be at least 1")
    if args.split_kv_cache and (
        args.max_seq_length % args.mpk_split_kv_chunk_size != 0
    ):
        parser.error(
            "Split-KV currently requires --max-seq-length to be divisible "
            "by --mpk-split-kv-chunk-size"
        )
    if args.mpk_policy == "decode-only":
        if args.spec_decode is not None:
            parser.error("decode-only does not support speculative decoding")
        if args.do_sample:
            parser.error("decode-only currently supports greedy decoding only")
        if not args.ignore_eos or args.max_new_tokens is None:
            parser.error(
                "decode-only requires --ignore-eos and --max-new-tokens"
            )
        if args.max_new_tokens < 2:
            parser.error("decode-only requires --max-new-tokens >= 2")
    if args.do_sample and args.temperature <= 0.0:
        parser.error("--do-sample needs --temperature > 0 "
                     "(temperature 0 is greedy decoding, i.e. no --do-sample)")
    if args.input_length is not None:
        if args.input_length <= 0:
            parser.error("--input-length must be positive")
        if args.input_length >= args.max_seq_length:
            parser.error("--input-length must be smaller than --max-seq-length")
        if (args.max_new_tokens is not None
                and args.input_length + args.max_new_tokens > args.max_seq_length):
            parser.error("--input-length + --max-new-tokens exceeds --max-seq-length")
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
    except ImportError:
        world_size = 1
        rank = 0

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)

    kv_plan = plan_qwen3_kv_cache(
        AutoConfig.from_pretrained(args.model_path or model_name),
        world_size, args.page_size)
    try:
        max_num_pages = resolve_pool_size(
            kv_plan, kv_budget=args.kv_budget,
            max_num_pages=None if args.kv_budget else args.max_num_pages,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            device=rank, verbose=args.use_mirage)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.model_path is not None or world_size == 1:
      with torch.device("cuda"):
          if args.model_path is not None:
              # load model locally (necessary for multi-GPU case)
              print(f"Load model from model path: {args.model_path}")
              config = AutoConfig.from_pretrained(args.model_path)
              model = Qwen3ForCausalLM(config, world_size, max_num_pages, args.page_size, kv_plan=kv_plan)
              load_model(
                  model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
              )
              # model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(args.model_path)
          else:
              # No kv_plan here: from_pretrained serialises unknown kwargs
              # through GenerationConfig, which a plan does not survive. The
              # constructor rebuilds one from the same config.
              model = Qwen3ForCausalLM.from_pretrained(
                  model_name, world_size, max_num_pages=max_num_pages,
                  page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(model_name)
    else: # Use dynamic shard loader to load directly from HF and shard.
        print("Detected multi-GPU run without a local path specified. Will use the DynamicShardLoader class.")
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(model_name)
            model = Qwen3ForCausalLM(config, world_size, max_num_pages, args.page_size, kv_plan=kv_plan)

        device = torch.device(f"cuda:{rank}")
        loader = Qwen3ShardLoader(model, model_name, mapping, rank, world_size, device)
        loader.load()

        with torch.device("cuda"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Adopt whichever plan the model ended up holding, so exactly one is live.
    kv_plan = model.model.kv_plan
    kv_plan.max_num_pages = max_num_pages
    model.set_prefill_attention_backend(args.normal_prefill_attention)
    print(f"Normal prefill attention: {args.normal_prefill_attention.upper()}")

    total_num_requests = 1 if not args.use_mirage else args.max_num_batched_requests
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    if args.input_length is None:
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": args.prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer([text], return_tensors="pt").input_ids[0].to(
            model.device
        )
    else:
        # Keep every id inside the ordinary vocabulary while avoiding special
        # tokens. The same sequence is used by Torch and MPK baselines.
        token_span = max(1, min(32000, model.config.vocab_size - 100))
        prompt_ids = 100 + (
            torch.arange(args.input_length, dtype=torch.long, device=model.device)
            * 1543
        ) % token_span
    prompt_length = int(prompt_ids.numel())
    tokens[:, :prompt_length] = prompt_ids
    prompt_lengths = torch.full(
        (total_num_requests,), prompt_length, dtype=torch.int, device="cuda"
    )
    if args.mpk_policy == "decode-only":
        if args.page_size < args.max_seq_length:
            parser.error(
                "decode-only currently requires one KV page per request: "
                "--page-size must be at least --max-seq-length"
            )
        if max_num_pages < total_num_requests:
            parser.error(
                "decode-only requires at least one KV page per request"
            )
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)

    # get all model weight tensors
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
    prev_pos = 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    step = torch.full((total_num_requests, ), 0, dtype=torch.int32, device="cuda")
    num_new_tokens = torch.full((total_num_requests, ), 1, dtype=torch.int32, device="cuda")
    mpk_kernel_cache_status = None
    mpk_kernel_prepare_time_ms = None
    captured_mpk_logits = None

    if args.use_mirage:
        import mirage as mi

        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
        # pad vocab_size to facilitate task graph creation
        lm_head_weight = torch.cat(
            (
                model.lm_head.weight,
                torch.full(
                    (153600 - model.config.vocab_size, hidden_size), 0, device="cuda"
                ),
            ),
            0,
        )
        assert lm_head_weight.stride()[0] == hidden_size
        vocab_size = 153600
        num_q_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_local_q_heads = num_q_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = model.config.head_dim
        fused_outdim_1 = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused_outdim_2 = 2 * intermediate_size
        split_kv_chunk_size = (
            args.mpk_split_kv_chunk_size if args.split_kv_cache else 256
        )
        num_kv_cache_chunks = max(
            1,
            (args.max_seq_length + split_kv_chunk_size - 1)
            // split_kv_chunk_size,
        )
        if args.split_kv_cache:
            print(
                "MPK attention: SPLIT-KV "
                f"(chunk_size={args.mpk_split_kv_chunk_size}, "
                f"chunks={num_kv_cache_chunks})"
            )
        else:
            print("MPK attention: DEFAULT")
        if requested_mpk_attention == "auto":
            print(
                "MPK attention policy: AUTO "
                f"(resolved: {resolved_mpk_attention.upper()}, "
                f"threshold={args.mpk_auto_split_kv_threshold})"
            )

        if args.profiling:
            if args.profiler_buffer_entries_per_block < 2:
                parser.error("--profiler-buffer-entries-per-block must be at least 2")
            # runtime_header.h permits up to 160 workers (B200).  Allocate for
            # that limit so the final worker cannot run past the tensor even
            # when the current GPU uses more than the historical 128 blocks.
            profiler_tensor = torch.zeros(
                1 + args.profiler_buffer_entries_per_block * 160,
                dtype=torch.uint64,
                device="cuda",
            ).contiguous()
            print(
                "MPK profiler buffer: "
                f"{args.profiler_buffer_entries_per_block} entries/block, "
                f"{profiler_tensor.numel() * profiler_tensor.element_size() / (1024 ** 2):.2f} MiB"
            )
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.mpk.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        # Create auxiliary buffers for paged (with kv_plan builder) KV and QO
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        kv_meta_tensors = kv_plan.build_meta_tensors(
            max_num_batched_requests=args.max_num_batched_requests,
            max_seq_length=args.max_seq_length)
        mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=max_num_pages,
            kv_groups=kv_plan.group_specs(),
            eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                **kv_meta_tensors,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=args.use_cutlass_kernel,
            # Decode-only contributes exactly one query token per request to
            # each persistent-kernel step.  Size per-request attention CTA
            # buffers from that fact rather than from the whole batch.
            max_tokens_per_request=(
                1 if args.mpk_policy == "decode-only" else None
            ),
            max_generation_length=(
                args.max_new_tokens
                if args.mpk_policy == "decode-only"
                else args.max_seq_length
            ),
            profiler_start_iteration=args.profiler_decode_start_step,
            profiler_num_iterations=args.profiler_decode_num_steps,
        )
        print(
            "MPK max tokens per request: "
            f"{mpk.max_tokens_per_request}"
        )
        print(f"MPK max generation length: {mpk.max_generation_length}")
        if args.profiling:
            print(
                "MPK profiler decode window: "
                f"start={mpk.profiler_start_iteration}, "
                f"steps={mpk.profiler_num_iterations}"
            )
        
        if spec_decode_config and spec_decode_config.method == "promptlookup":
            all_tokens = mpk.attach_input(torch_tensor=tokens, name="all_tokens")
            num_tokens_extend = spec_decode_config.spec_length + 1
        else:
            num_tokens_extend = 1
        
        # TODO: Make the code run well even if 96 % max_num_batched_tokens != 0
        # assert(96 % args.max_num_batched_tokens == 0)
        
        x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
        cos_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[0][0, :4096, :],
            name="cos_position_embedding",
        )
        sin_pos_embed = mpk.attach_input(
            torch_tensor=position_embeddings[1][0, :4096, :],
            name="sin_position_embedding",
        )

        y = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="embed_out",
            io_category="cuda_tensor",
        )
        rmsnorm_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )
        attn_in = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_1 // world_size), # [6, 6144]
            dtype=mi.bfloat16,
            name="attn_in",
            io_category="cuda_tensor",
        )
        lse = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        attn_out_tmp = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim, num_local_kv_heads),
            strides=(num_kv_cache_chunks * num_local_q_heads * head_dim, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )
        attn_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, num_local_q_heads * head_dim),
            dtype=mi.bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        attn_proj_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_proj_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        allreduce_buf = mpk.new_tensor(
            dims=(world_size, args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="all_reduce_buf",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        attn_allreduce_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="attn_allreduce_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_mid = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, fused_outdim_2 // world_size),
            dtype=mi.bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        silu_mul_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, intermediate_size // world_size),
            dtype=mi.bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        mlp_out = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_out",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        mlp_final = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, hidden_size),
            dtype=mi.bfloat16,
            name="mlp_final",
            io_category="nvshmem_tensor" if world_size > 1 else "cuda_tensor",
        )
        if args.capture_final_logits_topk:
            captured_mpk_logits = torch.empty(
                args.max_num_batched_tokens,
                vocab_size,
                dtype=torch.bfloat16,
                device="cuda",
            )
            argmax_in = mpk.attach_input(
                torch_tensor=captured_mpk_logits, name="argmax_in"
            )
        else:
            argmax_in = mpk.new_tensor(
                dims=(args.max_num_batched_tokens, vocab_size),
                dtype=mi.bfloat16,
                name="argmax_in",
                io_category="cuda_tensor",
            )
        argmax_part_value = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, mpk.num_workers),
            dtype=mi.bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        argmax_part_index = mpk.new_tensor(
            dims=(args.max_num_batched_tokens, mpk.num_workers),
            dtype=mi.int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )
        # Temperature/top-k/top-p sampling keeps its candidates in fp32 and
        # needs topk_max + 2 slots per worker (candidates, chunk max, chunk
        # exp-sum); greedy decoding keeps using the argmax buffers above.
        if args.do_sample:
            sampling_part_value = mpk.new_tensor(
                dims=(args.max_num_batched_tokens,
                      mpk.num_workers * (args.sampling_topk_max + 2)),
                dtype=mi.float32,
                name="sampling_part_value",
                io_category="cuda_tensor",
            )
            sampling_part_index = mpk.new_tensor(
                dims=(args.max_num_batched_tokens,
                      mpk.num_workers * args.sampling_topk_max),
                dtype=mi.int64,
                name="sampling_part_index",
                io_category="cuda_tensor",
            )
        argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")
        #argmax_out = mpk.new_tensor(
        #    dims=(args.max_num_batched_tokens, 1),
        #    dtype=mi.int64,
        #    name="argmax_out",
        #    io_category="cuda_tensor",
        #)

        # add spec tokens layer
        if spec_decode_config:
            spec_tokens = mpk.draft_forward_layer_dispatcher(
                spec_decode_config = spec_decode_config, 
                tokens = all_tokens,
                grid_dim=(96, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = spec_tokens
        # Add Embed
        w = mpk.attach_input(
            torch_tensor=model.model.embed_tokens.weight, name="embed_tokens"
        )
        
        mpk.embed_layer(
            input=x, 
            weight=w, 
            output=y, 
            # grid_dim=(max_factor_leq_n(hidden_size, 96 // args.max_num_batched_tokens), total_tokens_per_iter, 1), 
            grid_dim=(1, 1, 1), 
            block_dim=(128, 1, 1),
            input_source=1,
        )
        x = y
        target_cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
        # A current workaround to use splitk for only B200 GPUs
        use_splitk = (target_cc == 100)
        for i, layer in enumerate(model.model.layers):
            # if i > 0:
            #     break
            # add rmsnorm + linear
            w_norm = mpk.attach_input(
                torch_tensor=layer.input_layernorm.weight,
                name=f"layer_{i}_input_layernorm",
            )
            w_q = mpk.attach_input(
                torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj"
            )
            w_k = mpk.attach_input(
                torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj"
            )
            w_v = mpk.attach_input(
                torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj"
            )
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v],
                shuffled_dim=0,
                num_groups=model.config.num_key_value_heads // world_size,
                name=f"layer_{i}_qkv_proj",
            )
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
                block_dim=(128, 1, 1),
            )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_qkv,
            #    output=attn_in,
            #    grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0)), 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            # add attention
            w_q_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm"
            )
            w_k_norm = mpk.attach_input(
                torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm"
            )
            # kv_plan.layer_info() resolves (group_id, slot_id) for this layer.
            # For single spec, slot_id == layer_idx.
            group_id, slot_id = kv_plan.layer_info(i)
            k_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[0][slot_id], name=f"layer_{i}_k_cache"
            ) 
            v_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[1][slot_id], name=f"layer_{i}_v_cache"
            )
            # TODO: Later attention kernels should be merged as one
            if spec_decode_config:
                mpk.single_batch_extend_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(1, num_local_kv_heads, 1), #TODO: further divide across batch dim
                    block_dim=(128, 1, 1),
                )
            elif args.split_kv_cache:
                mpk.paged_attention_split_kv_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    lse=lse,
                    output=attn_out_tmp,
                    attention_params=(num_local_q_heads, num_kv_cache_chunks),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, num_kv_cache_chunks),
                    block_dim=(128, 1, 1),
                    group_id=group_id,
                )

                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse,
                    output_tmp=attn_out_tmp,
                    output=attn_out,
                    attention_params=(num_local_q_heads, head_dim),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                    group_id=group_id,
                )
            else:
                mpk.paged_attention_layer(
                    input=attn_in,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    q_norm=w_q_norm,
                    k_norm=w_k_norm,
                    cos_pos_embed=cos_pos_embed,
                    sin_pos_embed=sin_pos_embed,
                    output=attn_out,
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
                    group_id=group_id,
                )
            
            
            # add linear w/ residual
            w = mpk.attach_input(
                torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj"
            )
            if use_splitk:
                attn_proj_out = x
                mpk.splitk_linear_layer(
                    input=attn_out,
                    weight=w,
                    output=attn_proj_out,
                    grid_dim=grid_for_splitk_linear_layer(hidden_size, w.dim(1)),
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_with_residual_layer(
                    input=attn_out,
                    weight=w,
                    residual=x,
                    output=attn_proj_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
            # reset residual input as x
            x = attn_proj_out
            # add allreduce if needed
            if world_size > 1:
                mpk.allreduce_layer(
                    input=attn_proj_out,
                    buffer=allreduce_buf,
                    output=attn_allreduce_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = attn_allreduce_out
            # add rmsnorm_linear layer
            w_norm = mpk.attach_input(
                torch_tensor=layer.post_attention_layernorm.weight,
                name=f"layer_{i}_post_attn_layernorm",
            )
            w_gate_proj = mpk.attach_input(
                torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj"
            )
            w_up_proj = mpk.attach_input(
                torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj"
            )
            rmsnorm_num_tasks = grid_for_rmsnorm_linear_layer(w_gate_proj.dim(0) + w_up_proj.dim(0), args.use_cutlass_kernel)
            w_gatedup = mpk.shuffle_tensors(
                inputs=[w_gate_proj, w_up_proj],
                shuffled_dim=0,
                num_groups=rmsnorm_num_tasks//2,
                name=f"layer_{i}_gatedup_proj",
            )
            mpk.rmsnorm_layer(
                input=x,
                weight=w_norm,
                output=rmsnorm_out,
                grid_dim=(mpk.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_gatedup,
                output=mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )
            #mpk.rmsnorm_linear_layer(
            #    input=x,
            #    weight_norm=w_norm,
            #    weight_linear=w_gatedup,
            #    output=mlp_mid,
            #    grid_dim=(rmsnorm_num_tasks, 1, 1),
            #    block_dim=(128, 1, 1),
            #)
            mpk.silu_mul_layer(
                input=mlp_mid,
                output=silu_mul_out,
                grid_dim=(rmsnorm_num_tasks//2, 1, 1),
                block_dim=(128, 1, 1),
            )
            # add silu_mul_linear layer
            w = mpk.attach_input(
                torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj"
            )
            if use_splitk:
                mlp_out = x
                mpk.splitk_linear_layer(
                    input=silu_mul_out,
                    weight=w,
                    output=mlp_out,
                    grid_dim=grid_for_splitk_linear_layer(hidden_size, w.dim(1)),
                    block_dim=(256, 1, 1),
                )
            else:
                mpk.linear_with_residual_layer(
                    input=silu_mul_out,
                    weight=w,
                    residual=x,
                    output=mlp_out,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
            # reset residual input as x
            x = mlp_out
            if world_size > 1:
                mpk.allreduce_layer(
                    input=mlp_out,
                    buffer=allreduce_buf,
                    output=mlp_final,
                    grid_dim=(hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                x = mlp_final

        # add rmsnorm_linear layer
        w_norm = mpk.attach_input(
            torch_tensor=model.model.norm.weight, name="model_norm_weight"
        )
        w_proj = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
        mpk.rmsnorm_layer(
            input=x,
            weight=w_norm,
            output=rmsnorm_out,
            grid_dim=(mpk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rmsnorm_out,
            weight=w_proj,
            output=argmax_in,
            grid_dim=(mpk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )
        #mpk.rmsnorm_linear_layer(
        #    input=x,
        #    weight_norm=w_norm,
        #    weight_linear=w_proj,
        #    output=argmax_in,
        #    grid_dim=(grid_for_rmsnorm_linear_layer(w_proj.dim(0)), 1, 1),
        #    block_dim=(128, 1, 1),
        #)
        # add argmax layer
        if args.do_sample:
            mpk.sampling_partial_layer(
                input=argmax_in,
                output=(sampling_part_value, sampling_part_index),
                grid_dim=(mpk.num_workers, 1, 1),
                block_dim=(128, 1, 1),
                vocab_size=model.config.vocab_size,
                topk_max=args.sampling_topk_max,
                temperature=args.temperature,
            )
            mpk.sampling_reduce_layer(
                input=(sampling_part_value, sampling_part_index),
                output=argmax_out,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            )
        else:
            if spec_decode_config and spec_decode_config.method == "promptlookup":
                argmax_partial_grid_dim = (max_factor_leq_n(153600, 96 // (spec_decode_config.spec_length + 1)), 
                                           spec_decode_config.spec_length + 1, 
                                           1)
                argmax_reduce_grid_dim = (1, spec_decode_config.spec_length + 1, 1)
            else:
                argmax_partial_grid_dim = (mpk.num_workers, 1, 1)
                argmax_reduce_grid_dim = (1, 1, 1)
            mpk.argmax_partial_layer(
                input=argmax_in,
                output=(argmax_part_value, argmax_part_index),
                grid_dim=argmax_partial_grid_dim,
                block_dim=(128, 1, 1),
                vocab_size=model.config.vocab_size,
            )
            mpk.argmax_reduce_layer(
                input=(argmax_part_value, argmax_part_index),
                output=argmax_out,
                grid_dim=argmax_reduce_grid_dim,
                block_dim=(128, 1, 1),
            )
        if spec_decode_config:
            verify_out = mpk.verify_layer_dispatcher(
                spec_decode_config = spec_decode_config,
                spec_tokens = spec_tokens,
                target_output = argmax_out,
                grid_dim = (1, 1, 1),
                block_dim = (128, 1, 1),
            )

        results = mpk.kn_graph.generate_task_graph(num_gpus=world_size, my_gpu_id=rank)
        with open(f"task_graph_{rank}.json", "w") as f:
            f.write(results["json_file"])
        with open(f"kernel_{rank}.cu", "w") as f:
            f.write(results["cuda_code"])

        kernel_prepare_started = time.perf_counter()
        if args.mpk_kernel_cache_dir:
            cache_dir = os.path.abspath(args.mpk_kernel_cache_dir)
            try:
                mpk.load_mpk_kernel(
                    cache_dir,
                    expected_task_graph_json=results["json_file"],
                )
                mpk_kernel_cache_status = "hit"
                print(f"MPK kernel cache: HIT ({cache_dir})")
            except (FileNotFoundError, ValueError, ImportError, OSError) as error:
                print(f"MPK kernel cache: MISS ({error})")
                os.makedirs(cache_dir, exist_ok=True)
                mpk.compile(output_dir=cache_dir)
                mpk_kernel_cache_status = "miss_compiled"
                print(f"MPK kernel cache populated: {cache_dir}")
        else:
            mpk.compile(output_dir=args.output_dir)
            mpk_kernel_cache_status = "disabled_compiled"
        mpk_kernel_prepare_time_ms = (
            time.perf_counter() - kernel_prepare_started
        ) * 1000.0
        print(
            "MPK kernel preparation: status={}, time={:.3f} ms".format(
                mpk_kernel_cache_status, mpk_kernel_prepare_time_ms
            )
        )

    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    if not args.use_mirage:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        prefill_starter = torch.cuda.Event(enable_timing=True)
        prefill_ender = torch.cuda.Event(enable_timing=True)
        decode_starter = torch.cuda.Event(enable_timing=True)
        decode_ender = torch.cuda.Event(enable_timing=True)
        prefill_starter.record()
        for cur_pos in range(prompt_len, decode_limit):
            if cur_pos == prompt_len + 1:
                decode_starter.record()
            step.fill_(cur_pos - 1)
            input_ids = tokens[:, prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:, prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:, prev_pos:cur_pos]
            logits = model.forward(
                input_ids=input_ids,
                position_embeddings=(cos_embeddings, sin_embeddings),
                step=step,
                stream=stream,
                num_logits_to_keep=1,
            )
            next_token = logits.argmax(dim=-1)
            if args.do_sample:
                # Match the megakernel path: temperature → top-k → top-p → draw.
                row = logits[0, -1, : model.config.vocab_size].float()
                row = row / args.temperature
                if args.top_k > 0:
                    kth = torch.topk(row, min(args.top_k, row.numel())).values[-1]
                    row = row.masked_fill(row < kth, float("-inf"))
                if args.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(row, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    keep = cum <= args.top_p
                    keep[..., 0] = True
                    row = row.clone()
                    row[sorted_idx[~keep]] = float("-inf")
                probs = torch.softmax(row, dim=-1)
                g = torch.Generator(device=probs.device)
                g.manual_seed(args.seed + cur_pos)
                next_token = torch.multinomial(probs, 1, generator=g)
                next_token = next_token.view(1, 1)
            next_token = next_token[0, -1]
            tokens[0, cur_pos] = next_token
            prev_pos = cur_pos
            if cur_pos == prompt_len:
                prefill_ender.record()
            if (not args.ignore_eos
                    and next_token == model.config.eos_token_id):
                break

        decode_ender.record()
        torch.cuda.synchronize()
        prefill_time = prefill_starter.elapsed_time(prefill_ender)
        decode_steps = max(0, output_len - 1)
        decode_time = (
            decode_starter.elapsed_time(decode_ender) if decode_steps else 0.0
        )
        run_time = prefill_time + decode_time

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]
        tokens_generated = max(0, end_idx - prompt_len)
        decode_step_ms = decode_time / max(decode_steps, 1)
        per_tok_ms = run_time / max(tokens_generated, 1)

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        print(
            "Prompt length {}, generate length {}, prefill {:.3f} ms, "
            "decode {:.3f} ms, decode-step {:.3f} ms".format(
                prompt_len, tokens_generated, prefill_time, decode_time,
                decode_step_ms,
            )
        )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            all_generated_ids = tokens[0, prompt_len:end_idx]
            invalid_token_count = int(
                ((all_generated_ids < 0)
                 | (all_generated_ids >= model.config.vocab_size)).sum().item()
            )
            final_logits_topk = None
            if args.capture_final_logits_topk:
                k = min(args.capture_final_logits_topk, model.config.vocab_size)
                values, indices = torch.topk(
                    logits[0, -1, : model.config.vocab_size].float(), k
                )
                final_logits_topk = [
                    {"token_id": int(index), "logit": float(value)}
                    for value, index in zip(values.cpu(), indices.cpu())
                ]
            out = {
                "token_ids": token_ids,
                "text": tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True),
                "total_time_ms": run_time,
                "prefill_time_ms": prefill_time,
                "decode_time_ms": decode_time,
                "decode_steps": decode_steps,
                "decode_step_time_ms": decode_step_ms,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "requested_generate_length": output_len,
                "vocab_size": model.config.vocab_size,
                "invalid_token_count": invalid_token_count,
                "final_logits_topk": final_logits_topk,
                "mode": "torch",
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        prompt_len = prompt_lengths[0].item()
        prefill_time = None
        prefill_stage_profile = None
        decode_time = None
        decode_steps = None
        if args.mpk_policy == "decode-only":
            # Torch and MPK share model.model.kv_cache. Torch writes the prompt
            # K/V entries and produces the first generated token. Seeding step
            # at prompt_len lets the offline MPK scheduler resume from there.
            step.fill_(prompt_len - 1)
            for _ in range(args.prefill_warmup_runs):
                model.forward(
                    input_ids=tokens[:, :prompt_len],
                    position_embeddings=(
                        position_embeddings[0][:, :prompt_len],
                        position_embeddings[1][:, :prompt_len],
                    ),
                    step=step,
                    stream=stream,
                    num_logits_to_keep=1,
                )
            if args.prefill_warmup_runs:
                torch.cuda.synchronize()
            prefill_graph = None
            prefill_logits = None
            if args.normal_prefill_cuda_graph:
                static_prefill_input = tokens[:, :prompt_len].clone()
                prefill_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(prefill_graph):
                    prefill_logits = model.forward(
                        input_ids=static_prefill_input,
                        position_embeddings=(
                            position_embeddings[0][:, :prompt_len],
                            position_embeddings[1][:, :prompt_len],
                        ),
                        step=step,
                        stream=stream,
                        num_logits_to_keep=1,
                    )
                torch.cuda.synchronize()
            prefill_starter = torch.cuda.Event(enable_timing=True)
            prefill_ender = torch.cuda.Event(enable_timing=True)
            if args.profile_prefill_stages:
                model.enable_prefill_profile()
            prefill_starter.record()
            if prefill_graph is not None:
                prefill_graph.replay()
            else:
                prefill_logits = model.forward(
                    input_ids=tokens[:, :prompt_len],
                    position_embeddings=(
                        position_embeddings[0][:, :prompt_len],
                        position_embeddings[1][:, :prompt_len],
                    ),
                    step=step,
                    stream=stream,
                    num_logits_to_keep=1,
                )
            tokens[:, prompt_len] = prefill_logits[:, -1].argmax(dim=-1)
            step.fill_(prompt_len)
            prefill_ender.record()
            starter.record()
        else:
            starter.record()
        mpk(resume_after_prefill=args.mpk_policy == "decode-only")
        ender.record()
        torch.cuda.synchronize()
        mpk_time = starter.elapsed_time(ender)
        if args.mpk_policy == "decode-only":
            prefill_time = prefill_starter.elapsed_time(prefill_ender)
            prefill_stage_profile = (
                model.prefill_profile_ms()
                if args.profile_prefill_stages else None
            )
            if args.profile_prefill_stages:
                model.disable_prefill_profile()
            decode_time = mpk_time
            decode_steps = max(0, output_len - 1)
            run_time = prefill_time + decode_time
        else:
            run_time = mpk_time

        print("tokens.shape = ", tokens.shape)
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            valid_for_display = generated_ids[
                (generated_ids >= 0)
                & (generated_ids < model.config.vocab_size)
            ]
            response = tokenizer.decode(
                valid_for_display, skip_special_tokens=True
            )
            print(response)
        
        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        tokens_generated = step.max().item() + 1 - prompt_lengths[0].item()
        decode_step_ms = (
            decode_time / max(decode_steps, 1)
            if decode_time is not None else None
        )
        per_tok_ms = run_time / max(tokens_generated, 1)

        if args.mpk_policy == "decode-only":
            print(
                "Prompt length {}, generate length {}, prefill {:.3f} ms, "
                "decode {:.3f} ms, decode-step {:.3f} ms".format(
                    prompt_lengths[0], tokens_generated, prefill_time,
                    decode_time, decode_step_ms,
                )
            )
        else:
            print(
                "Prompt length {}, generate length {}, total latency: "
                "{:.3f} ms".format(
                    prompt_lengths[0], tokens_generated, run_time
                )
            )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            prompt_len = prompt_lengths[0].item()
            end_indices = [int(step[r].item()) + 1 for r in range(total_num_requests)]
            end_idx = end_indices[0]
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = per_tok_ms
            token_ids_by_request = []
            generate_lengths_by_request = []
            invalid_token_counts_by_request = []
            for request_id, request_end_idx in enumerate(end_indices):
                slice_end = min(request_end_idx, prompt_len + MAX_SAVE_TOKENS)
                token_ids_by_request.append(
                    tokens[request_id, prompt_len:slice_end].tolist()
                )
                generate_lengths_by_request.append(
                    max(0, request_end_idx - prompt_len)
                )
                generated = tokens[
                    request_id, prompt_len:request_end_idx
                ]
                invalid_token_counts_by_request.append(int(
                    ((generated < 0)
                     | (generated >= model.config.vocab_size)).sum().item()
                ))
            token_ids = token_ids_by_request[0]
            invalid_token_count = sum(invalid_token_counts_by_request)
            response_text = tokenizer.decode(tokens[0, :end_idx], skip_special_tokens=True)
            final_logits_topk = None
            if args.capture_final_logits_topk:
                k = min(args.capture_final_logits_topk, model.config.vocab_size)
                values, indices = torch.topk(
                    captured_mpk_logits[0, : model.config.vocab_size].float(), k
                )
                final_logits_topk = [
                    {"token_id": int(index), "logit": float(value)}
                    for value, index in zip(values.cpu(), indices.cpu())
                ]
            out = {
                "token_ids": token_ids,
                "token_ids_by_request": token_ids_by_request,
                "text": response_text,
                "total_time_ms": run_time,
                "prefill_time_ms": prefill_time,
                "prefill_stage_profile_ms": prefill_stage_profile,
                "normal_prefill_attention": args.normal_prefill_attention,
                "prefill_warmup_runs": args.prefill_warmup_runs,
                "normal_prefill_cuda_graph": args.normal_prefill_cuda_graph,
                "decode_time_ms": decode_time,
                "decode_steps": decode_steps,
                "decode_step_time_ms": decode_step_ms,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "generate_lengths_by_request": generate_lengths_by_request,
                "requested_generate_length": output_len,
                "vocab_size": model.config.vocab_size,
                "invalid_token_count": invalid_token_count,
                "final_logits_topk": final_logits_topk,
                "invalid_token_counts_by_request": (
                    invalid_token_counts_by_request
                ),
                "batch_size": total_num_requests,
                "mpk_kernel_cache_status": mpk_kernel_cache_status,
                "mpk_kernel_prepare_time_ms": mpk_kernel_prepare_time_ms,
                "mpk_max_tokens_per_request": (
                    mpk.max_tokens_per_request if args.use_mirage else None
                ),
                "mpk_kernel_cache_dir": (
                    os.path.abspath(args.mpk_kernel_cache_dir)
                    if args.mpk_kernel_cache_dir else None
                ),
                "mpk_attention": (
                    "split-kv" if args.split_kv_cache else "default"
                ),
                "mpk_attention_requested": requested_mpk_attention,
                "mpk_auto_split_kv_threshold": (
                    args.mpk_auto_split_kv_threshold
                    if requested_mpk_attention == "auto" else None
                ),
                "mpk_split_kv_chunk_size": (
                    args.mpk_split_kv_chunk_size
                    if args.split_kv_cache else None
                ),
                "mpk_split_kv_num_chunks": (
                    num_kv_cache_chunks if args.split_kv_cache else None
                ),
                "mode": (
                    "normal_prefill_mpk_decode"
                    if args.mpk_policy == "decode-only" else "mpk_always"
                ),
            }
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
