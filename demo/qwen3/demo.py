from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os, json

from models.qwen3_shard_loader import Qwen3ShardLoader
from mirage.mpk.base_dynamic_shard_loader import ShardType
from mirage.mpk.models.utils import grid_for_splitk_linear_layer
from execution.workload import WorkloadDescriptor
from policy import make_policy


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
MAX_SAVE_TOKENS = 100

# print limitation
# torch.set_printoptions(threshold=2000)

def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    # Hopper linear tasks currently compute one 64-column output tile. Keep
    # every task slice at 64 columns for QKV, gate/up, and LM-head projections.
    if size % 64 == 0:
        return size // 64
    if size % 96 == 0:
        return 96
    raise ValueError(f"Unsupported linear output size: {size}")
    
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


def execute_prefill(model, tokens, prompt_len, position_embeddings, step, stream):
    """Run the normal backend over the prompt and populate its KV cache."""
    step.fill_(prompt_len - 1)
    return model.forward(
        input_ids=tokens[:, :prompt_len],
        position_embeddings=(
            position_embeddings[0][:, :prompt_len],
            position_embeddings[1][:, :prompt_len],
        ),
        step=step,
        stream=stream,
    )


def execute_decode_step(model, tokens, cur_pos, position_embeddings, step, stream):
    """Consume the previous token using the normal backend's KV cache."""
    step.fill_(cur_pos - 1)
    return model.forward(
        input_ids=tokens[:, cur_pos - 1:cur_pos],
        position_embeddings=(
            position_embeddings[0][:, cur_pos - 1:cur_pos],
            position_embeddings[1][:, cur_pos - 1:cur_pos],
        ),
        step=step,
        stream=stream,
    )


def save_prefill_kv_snapshot(path, model, tokens, prompt_len, backend, policy):
    """Capture only the populated first KV page for offline diagnostics."""
    torch.cuda.synchronize()
    key_cache, value_cache = model.model.kv_cache
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "backend": backend,
        "policy": policy,
        "prompt_length": prompt_len,
        "prompt_token_ids": tokens[0, :prompt_len].detach().cpu(),
        "key_cache": key_cache[:, 0, :prompt_len].detach().cpu(),
        "value_cache": value_cache[:, 0, :prompt_len].detach().cpu(),
    }, path)
    print(f"Saved prefill KV to {path}")


def load_prefill_kv_snapshot(path, model, tokens, prompt_len):
    """Replace normal prefill KV with a matching MPK snapshot for diagnosis."""
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot["policy"] != "prefill-only" or snapshot["prompt_length"] != prompt_len:
        raise ValueError("Expected an MPK prefill-only KV snapshot of this length")
    if not torch.equal(snapshot["prompt_token_ids"], tokens[0, :prompt_len].cpu()):
        raise ValueError("KV snapshot prompt token IDs differ from this run")
    key_cache, value_cache = model.model.kv_cache
    for name, destination in (("key_cache", key_cache), ("value_cache", value_cache)):
        source = snapshot[name]
        target = destination[:, 0, :prompt_len]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"KV snapshot {name} shape or dtype differs")
        target.copy_(source.to(target.device))
    torch.cuda.synchronize()
    print(f"Loaded MPK prefill KV from {path}")


def select_normal_token(logits, args, model, cur_pos):
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
    return next_token[0, -1]


def decode_tokens_safely(tokenizer, token_ids, vocab_size):
    """Decode valid IDs and preserve invalid values for kernel diagnostics."""
    ids = token_ids.detach().cpu().reshape(-1).tolist()
    invalid = [
        {"position": position, "token_id": token_id}
        for position, token_id in enumerate(ids)
        if token_id < 0 or token_id >= vocab_size
    ]
    if not invalid:
        return tokenizer.decode(ids, skip_special_tokens=True), invalid
    replacement = tokenizer.unk_token_id
    if replacement is None or replacement < 0 or replacement >= vocab_size:
        replacement = 0
    sanitized = [
        token_id if 0 <= token_id < vocab_size else replacement
        for token_id in ids
    ]
    return tokenizer.decode(sanitized, skip_special_tokens=True), invalid

if __name__ == "__main__":
    global print
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("normal", "mpk"), default=None,
                        help="Execution backend (default: normal)")
    parser.add_argument("--mpk-policy", choices=("always", "decode-only", "prefill-only", "workload-aware"), default=None,
                        help="MPK execution policy")
    parser.add_argument("--use-mirage", action="store_true",
                        help="Deprecated alias for --backend mpk --mpk-policy always")
    parser.add_argument("--max-num-batched-tokens", default=8, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", help="Output files directory")
    parser.add_argument("--trace-name", default="", help="Perfetto trace output name")
    parser.add_argument("--phase-timing", action="store_true",
                        help="Record normal prefill and per-step decode CUDA timings")
    parser.add_argument("--debug-split-mpk-prefill", action="store_true",
                        help="Diagnostic: stop MPK after prefill and resume MPK decode")
    parser.add_argument(
        "--profiling", action="store_true", help="Use Profiler to generate trace"
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
        "--save-intermediates", type=str, default=None,
        help="Save final decode logits and normalized hidden state for a correctness probe",
    )
    parser.add_argument("--save-prefill-kv", type=str, default=None,
                        help="Save populated KV cache at the prefill boundary for diagnostics")
    parser.add_argument("--debug-load-prefill-kv", type=str, default=None,
                        help="Diagnostic: replace normal prefill KV with an MPK snapshot before decode")
    parser.add_argument("--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Custom prompt text to generate from.",
    )
    parser.add_argument("--batch-prompts-file", type=str, default=None,
                        help="JSON array of equal-token-length prompts for batched execution")
    parser.add_argument("--no-system-message", action="store_true",
                        help="Use a user-only chat template for short input-length sweeps")

    parser.add_argument("--split-kv-cache", action="store_true", help="Use split-kv cache")
    args = parser.parse_args()
    if args.use_mirage:
        if args.backend not in (None, "mpk") or args.mpk_policy not in (None, "always"):
            parser.error("--use-mirage conflicts with the selected backend or MPK policy")
        print("[Deprecated] --use-mirage is deprecated.\n"
              "Use --backend mpk --mpk-policy always instead.")
        args.backend = "mpk"
        args.mpk_policy = "always"
    else:
        args.backend = args.backend or "normal"
        if args.backend == "normal" and args.mpk_policy is not None:
            parser.error("--mpk-policy is only valid when --backend=mpk")
        if args.backend == "mpk":
            args.mpk_policy = args.mpk_policy or "always"
    # Keep the existing execution paths intact while migrating their CLI.
    args.use_mirage = args.backend == "mpk"
    if args.max_num_batched_requests < 1:
        parser.error("--max-num-batched-requests must be positive")
    if args.batch_prompts_file and (
        args.max_num_batched_requests < 2
    ):
        parser.error("--batch-prompts-file requires at least two requests")
    if args.backend == "normal" and args.max_num_batched_requests > 1:
        if (args.max_num_batched_requests > args.max_num_pages
            or not args.ignore_eos or args.do_sample or args.spec_decode
            or args.profiling or args.save_prefill_kv):
            parser.error("Batched normal currently requires one KV page per request, "
                         "--ignore-eos, greedy decoding, and no profiling, "
                         "speculative decoding, or prefill KV snapshots")
        if args.max_seq_length > args.page_size:
            parser.error("Batched normal currently requires each sequence to fit in one KV page")
    if args.mpk_policy == "workload-aware":
        parser.error("workload-aware requires a measured MPK advantage map; complete Steps 9–10 first")
    if args.debug_split_mpk_prefill and (
        args.backend != "mpk" or args.mpk_policy != "always"
        or args.do_sample or args.spec_decode or args.profiling
    ):
        parser.error("--debug-split-mpk-prefill requires MPK always, greedy decoding, no speculative decoding, and no profiling")
    if args.mpk_policy in ("prefill-only", "decode-only"):
        if args.spec_decode or args.do_sample or args.profiling:
            parser.error("Mixed backend policies require greedy decoding, no speculative decoding, and no profiling")
    if args.backend == "mpk" and args.max_num_batched_requests > 1:
        if (args.max_num_batched_requests > args.max_num_pages
            or args.max_num_batched_requests > args.max_num_batched_tokens
            or not args.ignore_eos or args.max_seq_length > args.page_size
            or args.save_prefill_kv
            or args.debug_load_prefill_kv):
            parser.error("Batched MPK requires one KV page per request, "
                         "enough batched-token slots, --ignore-eos, and no prefill KV snapshots")
    if args.save_intermediates and (
        args.backend == "mpk" and args.mpk_policy not in ("decode-only", "prefill-only", "always")
        or args.max_new_tokens is None
        or args.max_new_tokens < 1
        or not args.ignore_eos
        or args.do_sample
    ):
        parser.error("--save-intermediates requires a static policy, at least one output token, ignore-eos, and greedy decoding")
    if args.save_prefill_kv and (
        args.backend == "mpk" and args.mpk_policy != "prefill-only"
        or args.do_sample or args.spec_decode or args.profiling
    ):
        parser.error("--save-prefill-kv requires normal or prefill-only, greedy decoding, no speculative decoding, and no profiling")
    if args.debug_load_prefill_kv and (args.backend != "mpk" or args.mpk_policy != "decode-only"):
        parser.error("--debug-load-prefill-kv requires MPK decode-only")
    if args.do_sample and args.temperature <= 0.0:
        parser.error("--do-sample needs --temperature > 0 "
                     "(temperature 0 is greedy decoding, i.e. no --do-sample)")
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
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"Execution backend: {args.backend.upper()}"
          + (f", MPK policy: {args.mpk_policy}" if args.backend == "mpk" else ""))
    print(f"world_size({world_size}) rank({rank})")
    if args.mpk_policy in ("prefill-only", "decode-only") and world_size != 1:
        parser.error("Mixed backend policies currently require a single GPU")
    if args.backend == "normal" and args.max_num_batched_requests > 1 and world_size != 1:
        parser.error("Batched normal currently requires a single GPU")
    if args.save_intermediates and world_size != 1:
        parser.error("--save-intermediates currently requires a single GPU")
    if args.save_prefill_kv and world_size != 1:
        parser.error("--save-prefill-kv currently requires a single GPU")
    if args.debug_load_prefill_kv and world_size != 1:
        parser.error("--debug-load-prefill-kv currently requires a single GPU")
    if args.debug_split_mpk_prefill and world_size != 1:
        parser.error("--debug-split-mpk-prefill currently requires a single GPU")
    model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    torch.cuda.set_device(rank)
    if args.model_path is not None or world_size == 1:
      with torch.device("cuda"):
          if args.model_path is not None:
              # load model locally (necessary for multi-GPU case)
              print(f"Load model from model path: {args.model_path}")
              config = AutoConfig.from_pretrained(args.model_path)
              model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)
              load_model(
                  model, f"{args.model_path}/model{rank}-mp{world_size}.safetensors"
              )
              # model = Qwen3ForCausalLM.from_pretrained(args.model_path, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(args.model_path)
          else:
              model = Qwen3ForCausalLM.from_pretrained(model_name, world_size, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
              tokenizer = AutoTokenizer.from_pretrained(model_name)
    else: # Use dynamic shard loader to load directly from HF and shard.
        print("Detected multi-GPU run without a local path specified. Will use the DynamicShardLoader class.")
        with torch.device("meta"):
            config = AutoConfig.from_pretrained(model_name)
            model = Qwen3ForCausalLM(config, world_size, args.max_num_pages, args.page_size)

        device = torch.device(f"cuda:{rank}")
        loader = Qwen3ShardLoader(model, model_name, mapping, rank, world_size, device)
        loader.load()

        with torch.device("cuda"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    total_num_requests = args.max_num_batched_requests
    normal_hidden = {}
    normal_layer0 = {}
    if args.save_intermediates and (args.backend == "normal" or args.mpk_policy == "prefill-only"):
        def capture_normal_hidden(_module, _inputs, output):
            normal_hidden["last"] = output[:, -1, :].detach()
        model.model.norm.register_forward_hook(capture_normal_hidden)
    if args.save_intermediates and args.backend == "normal":
        def capture_layer0_input(_module, _inputs, output):
            normal_layer0["input"] = output[:, -1, :].detach()

        def capture_layer0_norm(_module, _inputs, output):
            normal_layer0["norm"] = output[:, -1, :].detach()

        def capture_layer0_attention_output(_module, inputs):
            normal_layer0["attention_output"] = inputs[0][:, -1, :].detach()

        def capture_layer0_q(_module, _inputs, output):
            normal_layer0["q"] = output[:, -1, :].detach()

        def capture_layer0_k(_module, _inputs, output):
            normal_layer0["k"] = output[:, -1, :].detach()

        def capture_layer0_v(_module, _inputs, output):
            normal_layer0["v"] = output[:, -1, :].detach()

        def capture_layer0_after_attention(_module, inputs):
            normal_layer0["after_attention"] = inputs[1][:, -1, :].detach()

        def capture_layer0_post_attention_norm(_module, _inputs, output):
            normal_layer0["post_attention_norm"] = output[:, -1, :].detach()

        def capture_layer0_gate(_module, _inputs, output):
            normal_layer0["gate"] = output[:, -1, :].detach()

        def capture_layer0_up(_module, _inputs, output):
            normal_layer0["up"] = output[:, -1, :].detach()
            gate = normal_layer0.get("gate")
            if gate is not None:
                normal_layer0["silu_mul"] = (
                    torch.nn.functional.silu(gate) * output[:, -1, :]
                ).detach()

        def capture_layer0_output(_module, _inputs, output):
            normal_layer0["output"] = output[0][:, -1, :].detach()

        model.model.embed_tokens.register_forward_hook(capture_layer0_input)
        model.model.layers[0].input_layernorm.register_forward_hook(
            capture_layer0_norm
        )
        model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
            capture_layer0_attention_output
        )
        model.model.layers[0].self_attn.q_proj.register_forward_hook(
            capture_layer0_q
        )
        model.model.layers[0].self_attn.k_proj.register_forward_hook(
            capture_layer0_k
        )
        model.model.layers[0].self_attn.v_proj.register_forward_hook(
            capture_layer0_v
        )
        model.model.layers[0].mlp.register_forward_pre_hook(
            capture_layer0_after_attention
        )
        model.model.layers[0].post_attention_layernorm.register_forward_hook(
            capture_layer0_post_attention_norm
        )
        model.model.layers[0].mlp.gate_proj.register_forward_hook(
            capture_layer0_gate
        )
        model.model.layers[0].mlp.up_proj.register_forward_hook(
            capture_layer0_up
        )
        model.model.layers[0].register_forward_hook(capture_layer0_output)
    # get all model weight tensors
    tokens = torch.full((total_num_requests, args.max_seq_length), 0, dtype=torch.long, device="cuda")

    prompt = args.prompt
    # This prompt is copied from https://github.com/apoorvumang/prompt-lookup-decoding/blob/main/demo-pld.ipynb
    code_text = """import numpy as np
                import matplotlib.pyplot as plt

                # Calculate the average
                average_throughput = np.mean(tokens_per_sec_arr)
                print(f"Average Throughput: {average_throughput} tokens/sec")

                # Plotting the histogram
                plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
                plt.title('Histogram of Throughput Values')
                plt.xlabel('Tokens per Second')
                plt.ylabel('Frequency')
                plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
                plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {average_throughput:.2f}', color = 'red')
                plt.show()
                """
    #question = "Can you please change x axis to start from 0"
    #prompt = code_text + "\n" + question
    if args.batch_prompts_file:
        with open(args.batch_prompts_file) as f:
            prompts = json.load(f)
        if (not isinstance(prompts, list) or len(prompts) != total_num_requests
            or any(not isinstance(item, str) for item in prompts)):
            parser.error("--batch-prompts-file must contain one string per request")
    else:
        prompts = [prompt] * total_num_requests
    texts = []
    for request_prompt in prompts:
        messages = []
        if not args.no_system_message:
            messages.append({
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            })
        messages.append({"role": "user", "content": request_prompt})
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    encoded = tokenizer(texts)
    if len({len(ids) for ids in encoded["input_ids"]}) != 1:
        parser.error("Batched normal prompts must have equal input token lengths")
    model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)
    if total_num_requests == 1:
        for i in range(model_inputs.input_ids.shape[-1]):
            tokens[0, i] = model_inputs.input_ids[0, i]
    else:
        tokens[:, :model_inputs.input_ids.shape[-1]] = model_inputs.input_ids
    prompt_lengths = torch.full((total_num_requests,), model_inputs.input_ids.shape[-1], dtype=torch.int, device="cuda")
    input_length = model_inputs.input_ids.shape[-1]
    execution_policy = make_policy(args.mpk_policy) if args.backend == "mpk" else None
    prefill_workload = WorkloadDescriptor.for_prefill(total_num_requests, input_length)
    decode_workload = WorkloadDescriptor.for_decode(total_num_requests, input_length, 0)
    prefill_use_mpk = execution_policy.should_use_mpk(prefill_workload) if execution_policy else False
    decode_use_mpk = execution_policy.should_use_mpk(decode_workload) if execution_policy else False
    print(f"Prefill backend: {'MPK' if prefill_use_mpk else 'NORMAL'}")
    print(f"Decode backend: {'MPK' if decode_use_mpk else 'NORMAL'}")
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
        num_kv_cache_chunks = max(1, args.max_seq_length // 256)

        if args.profiling:
            profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            profiler_tensor = None
            
        spec_decode_config = mi.mpk.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        qo_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indptr_buffer = torch.empty(
            args.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        paged_kv_indices_buffer = torch.empty(
            args.max_num_pages, dtype=torch.int32, device="cuda")
        # Keep this deterministic before a scheduler seeds its first batch.
        # Resume-after-prefill overwrites active slots with their valid lengths.
        paged_kv_last_page_len_buffer = torch.zeros(
            args.max_num_batched_requests, dtype=torch.int32, device="cuda")
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
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=model.config.eos_token_id if not args.ignore_eos else -1,
            meta_tensors={
                "step": step,
                "tokens": tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "num_new_tokens": num_new_tokens,
                "prompt_lengths": prompt_lengths,
                "qo_indptr_buffer": qo_indptr_buffer,
                "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
            },
            profiler_tensor=profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=spec_decode_config,
            use_cutlass_kernel=args.use_cutlass_kernel
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
        if args.save_intermediates:
            mpk_hidden = torch.empty((args.max_num_batched_tokens, hidden_size),
                                     dtype=torch.bfloat16, device="cuda")
            rmsnorm_out = mpk.attach_input(torch_tensor=mpk_hidden, name="rmsnorm_out")
            mpk_layer0_input = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_input_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_input,
                name="layer0_input_snapshot",
            )
            mpk_layer0_norm = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_norm_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_norm,
                name="layer0_norm_snapshot",
            )
            mpk_layer0_attention_output = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_attention_output_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_attention_output,
                name="layer0_attention_output_snapshot",
            )
            mpk_layer0_qkv = torch.empty(
                (args.max_num_batched_tokens, fused_outdim_1 // world_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_qkv_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_qkv,
                name="layer0_qkv_snapshot",
            )
            mpk_layer0_after_attention = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_after_attention_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_after_attention,
                name="layer0_after_attention_snapshot",
            )
            mpk_layer0_output = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_output_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_output,
                name="layer0_output_snapshot",
            )
            mpk_layer0_post_attention_norm = torch.empty(
                (args.max_num_batched_tokens, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_post_attention_norm_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_post_attention_norm,
                name="layer0_post_attention_norm_snapshot",
            )
            mpk_layer0_mlp_mid = torch.empty(
                (args.max_num_batched_tokens, fused_outdim_2 // world_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_mlp_mid_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_mlp_mid,
                name="layer0_mlp_mid_snapshot",
            )
            mpk_layer0_silu_mul = torch.empty(
                (args.max_num_batched_tokens, intermediate_size // world_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            mpk_layer0_silu_mul_dt = mpk.attach_input(
                torch_tensor=mpk_layer0_silu_mul,
                name="layer0_silu_mul_snapshot",
            )
        else:
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
            strides=(num_kv_cache_chunks * num_local_q_heads, 1, num_kv_cache_chunks * num_local_q_heads // num_local_kv_heads * head_dim),
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
        if args.save_intermediates:
            mpk_logits = torch.empty((args.max_num_batched_tokens, vocab_size),
                                     dtype=torch.bfloat16, device="cuda")
            argmax_in = mpk.attach_input(torch_tensor=mpk_logits, name="argmax_in")
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
        if args.save_intermediates:
            mpk.copy_layer(
                input=x,
                output=mpk_layer0_input_dt,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
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
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=rmsnorm_out,
                    output=mpk_layer0_norm_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_qkv,
                output=attn_in,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
                block_dim=(128, 1, 1),
            )
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=attn_in,
                    output=mpk_layer0_qkv_dt,
                    grid_dim=(1, 1, 1),
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
            k_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[0][i], name=f"layer_{i}_k_cache"
            ) 
            v_cache = mpk.attach_input(
                torch_tensor=model.model.kv_cache[1][i], name=f"layer_{i}_v_cache"
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
                )

                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse,
                    output_tmp=attn_out_tmp,
                    output=attn_out,
                    attention_params=(num_local_q_heads, head_dim),
                    grid_dim=(mpk.max_num_batched_requests, num_local_kv_heads, 1),
                    block_dim=(128, 1, 1),
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
                )
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=attn_out,
                    output=mpk_layer0_attention_output_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
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
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=x,
                    output=mpk_layer0_after_attention_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
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
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=rmsnorm_out,
                    output=mpk_layer0_post_attention_norm_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
            mpk.linear_layer(
                input=rmsnorm_out,
                weight=w_gatedup,
                output=mlp_mid,
                grid_dim=(rmsnorm_num_tasks, 1, 1),
                block_dim=(128, 1, 1),
            )
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=mlp_mid,
                    output=mpk_layer0_mlp_mid_dt,
                    grid_dim=(1, 1, 1),
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
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=silu_mul_out,
                    output=mpk_layer0_silu_mul_dt,
                    grid_dim=(1, 1, 1),
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
            if args.save_intermediates and i == 0:
                mpk.copy_layer(
                    input=x,
                    output=mpk_layer0_output_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )

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
            grid_dim=(
                grid_for_rmsnorm_linear_layer(
                    w_proj.dim(0), args.use_cutlass_kernel
                ),
                1,
                1,
            ),
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

        mpk.compile(output_dir=args.output_dir)

    # g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    warmup = 0
    # Decode up to user cap or buffer size
    output_len = args.max_new_tokens if args.max_new_tokens is not None else (tokens.size(1) - prompt_lengths[0].item())
    output_len = max(0, min(output_len, tokens.size(1) - prompt_lengths[0].item()))
    split_mpk_prefill = args.debug_split_mpk_prefill or (
        args.phase_timing and args.mpk_policy == "always"
    )
    if split_mpk_prefill and (
        prompt_lengths[0].item() >= args.page_size or output_len < 2
        or args.max_seq_length != prompt_lengths[0].item() + output_len
    ):
        parser.error("Split MPK prefill requires prompt shorter than one KV page and max-seq-length = prompt length + at least two output tokens")
    if prefill_use_mpk and not decode_use_mpk:
        if output_len == 0:
            parser.error("prefill-only requires at least one output token")
        starter.record()
        if args.phase_timing:
            mpk_prefill_start = torch.cuda.Event(enable_timing=True)
            mpk_prefill_end = torch.cuda.Event(enable_timing=True)
            mpk_prefill_start.record()
        mpk(stop_after_prefill=True)
        if args.phase_timing:
            mpk_prefill_end.record()
        torch.cuda.synchronize()
        prompt_len = prompt_lengths[0].item()
        if not torch.all(step == prompt_len).item():
            raise RuntimeError("MPK did not stop every request at the prefill boundary")
        if args.save_prefill_kv:
            save_prefill_kv_snapshot(args.save_prefill_kv, model, tokens,
                                     prompt_len, args.backend, args.mpk_policy)
        prev_pos = prompt_len
    elif decode_use_mpk and not prefill_use_mpk:
        prompt_len = prompt_lengths[0].item()
        if output_len < 2 or args.max_seq_length != prompt_len + output_len:
            parser.error("decode-only requires at least two output tokens and max-seq-length = prompt length + output length")
        if prompt_len >= args.page_size:
            parser.error("decode-only currently requires the prompt to fit in one KV page")
        starter.record()
        if args.phase_timing:
            normal_prefill_start = torch.cuda.Event(enable_timing=True)
            normal_prefill_end = torch.cuda.Event(enable_timing=True)
            normal_prefill_start.record()
        logits = execute_prefill(model, tokens, prompt_len, position_embeddings, step, stream)
        if total_num_requests == 1:
            tokens[0, prompt_len] = select_normal_token(logits, args, model, prompt_len)
        else:
            tokens[:, prompt_len] = logits[:, -1, :model.config.vocab_size].argmax(dim=-1)
        if args.debug_load_prefill_kv:
            load_prefill_kv_snapshot(args.debug_load_prefill_kv, model, tokens, prompt_len)
        if args.phase_timing:
            normal_prefill_end.record()

    if not decode_use_mpk:
        prompt_len = prompt_lengths[0].item()
        decode_limit = prompt_len + output_len
        start_pos = prompt_len + (args.mpk_policy == "prefill-only")
        phase_events = []
        for cur_pos in range(start_pos, decode_limit):
            phase = "prefill" if cur_pos == prompt_len else "decode"
            if args.phase_timing:
                phase_start = torch.cuda.Event(enable_timing=True)
                phase_end = torch.cuda.Event(enable_timing=True)
                phase_start.record()
            if phase == "prefill":
                logits = execute_prefill(model, tokens, prompt_len, position_embeddings, step, stream)
                if args.save_prefill_kv:
                    save_prefill_kv_snapshot(args.save_prefill_kv, model, tokens,
                                             prompt_len, args.backend, args.mpk_policy)
            else:
                logits = execute_decode_step(model, tokens, cur_pos, position_embeddings, step, stream)
            if total_num_requests == 1:
                next_token = select_normal_token(logits, args, model, cur_pos)
                tokens[0, cur_pos] = next_token
            else:
                next_token = logits[:, -1, :model.config.vocab_size].argmax(dim=-1)
                tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            if args.phase_timing:
                phase_end.record()
                phase_events.append((phase, phase_start, phase_end))
            if (total_num_requests == 1 and not args.ignore_eos
                and next_token == model.config.eos_token_id):
                break
            if args.mpk_policy != "prefill-only" and cur_pos == prompt_len + warmup:
                torch.cuda.synchronize()
                starter.record()

        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)
        phase_timing_data = None
        if args.phase_timing and (phase_events or args.mpk_policy == "prefill-only"):
            prefill_ms = (mpk_prefill_start.elapsed_time(mpk_prefill_end)
                          if args.mpk_policy == "prefill-only"
                          else phase_events[0][1].elapsed_time(phase_events[0][2]))
            decode_step_ms = [start.elapsed_time(end) for phase, start, end in phase_events
                              if phase == "decode"]
            phase_timing_data = {
                "prefill_ms": prefill_ms,
                "decode_ms": sum(decode_step_ms),
                "decode_step_ms": decode_step_ms,
            }
            print(f"Phase timing: prefill={prefill_ms:.3f} ms, "
                  f"decode={sum(decode_step_ms):.3f} ms, "
                  f"decode_steps={len(decode_step_ms)}")

        end_idx = prev_pos + 1
        generated_ids = tokens[:, :end_idx]
        tokens_generated = max(0, end_idx - prompt_len)
        per_tok_ms = run_time / max(prompt_len + tokens_generated, 1)

        invalid_token_ids_by_request = {}
        for request_id in range(total_num_requests):
            response, invalid = decode_tokens_safely(
                tokenizer, generated_ids[request_id], model.config.vocab_size
            )
            if invalid:
                invalid_token_ids_by_request[str(request_id)] = invalid
            if total_num_requests > 1:
                print(f"Request {request_id}:")
            print(response)
            if invalid:
                print(f"Invalid token IDs for request {request_id}: {invalid}")
        print(
            "Prompt length {}, generate length {}, per-token latency {:.3f} ms".format(
                prompt_len, tokens_generated, per_tok_ms
            )
        )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            out = {
                "token_ids": token_ids,
                "text": decode_tokens_safely(
                    tokenizer, tokens[0, :end_idx], model.config.vocab_size
                )[0],
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "mpk_prefill_normal_decode" if args.mpk_policy == "prefill-only" else "torch",
            }
            if total_num_requests > 1:
                out["batch_size"] = total_num_requests
                out["token_ids_by_request"] = [
                    tokens[r, prompt_len:slice_end].tolist()
                    for r in range(total_num_requests)
                ]
            if invalid_token_ids_by_request:
                out["invalid_token_ids_by_request"] = invalid_token_ids_by_request
            if phase_timing_data is not None:
                out["phase_timing"] = phase_timing_data
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    else:
        if args.mpk_policy != "decode-only":
            starter.record()
        if args.phase_timing and not split_mpk_prefill:
            mpk_decode_start = torch.cuda.Event(enable_timing=True)
            mpk_decode_end = torch.cuda.Event(enable_timing=True)
            mpk_decode_start.record()
        if split_mpk_prefill:
            if args.phase_timing:
                mpk_decode_start = torch.cuda.Event(enable_timing=True)
                mpk_decode_end = torch.cuda.Event(enable_timing=True)
                mpk_prefill_start = torch.cuda.Event(enable_timing=True)
                mpk_prefill_end = torch.cuda.Event(enable_timing=True)
                mpk_prefill_start.record()
            mpk(stop_after_prefill=True)
            if args.phase_timing:
                mpk_prefill_end.record()
            torch.cuda.synchronize()
            if not torch.all(step == prompt_lengths[0]).item():
                raise RuntimeError("MPK did not stop every request at the prefill boundary")
            if args.phase_timing:
                mpk_decode_start.record()
        mpk(resume_after_prefill=args.mpk_policy == "decode-only" or split_mpk_prefill)
        if args.phase_timing:
            mpk_decode_end.record()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)
        phase_timing_data = None
        if args.phase_timing:
            phase_timing_data = {
                "prefill_ms": (mpk_prefill_start.elapsed_time(mpk_prefill_end)
                               if args.mpk_policy == "always"
                               else normal_prefill_start.elapsed_time(normal_prefill_end)),
                "decode_ms": mpk_decode_start.elapsed_time(mpk_decode_end),
                "decode_steps": output_len - 1,
                "decode_step_ms": None,
            }
            print(f"Phase timing: prefill={phase_timing_data['prefill_ms']:.3f} ms, "
                  f"decode={phase_timing_data['decode_ms']:.3f} ms")

        print("tokens.shape = ", tokens.shape)
        invalid_token_ids_by_request = {}
        for r in range(total_num_requests):
            generated_ids = tokens[r, : step[r] + 1]
            response, invalid = decode_tokens_safely(
                tokenizer, generated_ids, model.config.vocab_size
            )
            if invalid:
                invalid_token_ids_by_request[str(r)] = invalid
            print(response)
            if invalid:
                print(f"Invalid token IDs for request {r}: {invalid}")
        
        if total_num_requests > 1:
            print(f"Output length of each batch is same: {(step.max() == step.min()).item()}")

        tokens_generated = step.max().item() + 1 - prompt_lengths[0].item()
        per_tok_ms = run_time / max(prompt_lengths[0].item() + tokens_generated, 1)

        print("Prompt length {}, generate length {}, per-token latency: {:.3f} ms".format(
              prompt_lengths[0], tokens_generated, per_tok_ms
            )
        )

        # -------- CI dumps outputs to json files ----------
        if save_path and rank == 0:
            end_idx = step[0].item() + 1
            prompt_len = prompt_lengths[0].item()
            tokens_generated = max(0, end_idx - prompt_len)
            per_tok_ms = per_tok_ms
            slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
            token_ids = tokens[0, prompt_len:slice_end].tolist()
            response_text = decode_tokens_safely(
                tokenizer, tokens[0, :end_idx], model.config.vocab_size
            )[0]
            out = {
                "token_ids": token_ids,
                "text": response_text,
                "latency_ms_per_token": per_tok_ms,
                "prompt_length": prompt_len,
                "generate_length": tokens_generated,
                "mode": "normal_prefill_mpk_decode" if args.mpk_policy == "decode-only" else "mpk",
            }
            if total_num_requests > 1:
                out["batch_size"] = total_num_requests
                out["token_ids_by_request"] = [
                    tokens[r, prompt_len:min(step[r].item() + 1,
                                              prompt_len + MAX_SAVE_TOKENS)].tolist()
                    for r in range(total_num_requests)
                ]
            if invalid_token_ids_by_request:
                out["invalid_token_ids_by_request"] = invalid_token_ids_by_request
            if phase_timing_data is not None:
                out["phase_timing"] = phase_timing_data
            with open(save_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved tokens to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    if args.save_intermediates:
        prompt_len = prompt_lengths[0].item()
        if (args.backend == "normal"
                or (args.mpk_policy == "prefill-only" and output_len > 1)):
            hidden_all = normal_hidden["last"]
            output_logits_all = logits[:, -1, :model.config.vocab_size]
            layer0_after_attention_all = normal_layer0.get("after_attention")
            layer0_output_all = normal_layer0.get("output")
            layer0_input_all = normal_layer0.get("input")
            layer0_norm_all = normal_layer0.get("norm")
            layer0_attention_output_all = normal_layer0.get("attention_output")
            layer0_post_attention_norm_all = normal_layer0.get(
                "post_attention_norm"
            )
            normal_gate = normal_layer0.get("gate")
            normal_up = normal_layer0.get("up")
            if normal_gate is None or normal_up is None:
                layer0_mlp_mid_all = None
            else:
                layer0_mlp_mid_all = torch.cat((normal_gate, normal_up), dim=-1)
            layer0_silu_mul_all = normal_layer0.get("silu_mul")
            normal_q = normal_layer0.get("q")
            normal_k = normal_layer0.get("k")
            normal_v = normal_layer0.get("v")
            if normal_q is None or normal_k is None or normal_v is None:
                layer0_qkv_all = None
            else:
                q_heads = model.config.num_attention_heads
                kv_heads = model.config.num_key_value_heads
                q_per_kv = q_heads // kv_heads
                qkv_head_dim = model.config.head_dim
                batch_size = normal_q.shape[0]
                grouped_q = normal_q.reshape(
                    batch_size, kv_heads, q_per_kv, qkv_head_dim
                )
                grouped_k = normal_k.reshape(
                    batch_size, kv_heads, 1, qkv_head_dim
                )
                grouped_v = normal_v.reshape(
                    batch_size, kv_heads, 1, qkv_head_dim
                )
                layer0_qkv_all = torch.cat(
                    (grouped_q, grouped_k, grouped_v), dim=2
                ).reshape(batch_size, -1)
        else:
            hidden_all = mpk_hidden
            output_logits_all = mpk_logits[:, :model.config.vocab_size]
            snapshot_indices = torch.arange(
                total_num_requests, device=mpk_hidden.device
            )
            if output_len == 1 and args.mpk_policy in ("always", "prefill-only"):
                # MPK prefill schedules at most 16 tokens per request in one
                # internal batch. Snapshot tensors therefore use positions in
                # the final packed chunk, rather than logical prompt offsets.
                final_chunk_lengths = ((prompt_lengths - 1) % 16) + 1
                total_final_chunk_tokens = int(final_chunk_lengths.sum().item())
                if total_final_chunk_tokens > args.max_num_batched_tokens:
                    raise RuntimeError(
                        "Final prefill chunks do not fit in one internal MPK "
                        "batch; reduce the number of requests or increase "
                        "--max-num-batched-tokens"
                    )
                terminal_slots = torch.cumsum(final_chunk_lengths, dim=0) - 1
                snapshot_indices = terminal_slots
                hidden_all = hidden_all.index_select(0, terminal_slots)
                output_logits_all = output_logits_all.index_select(
                    0, terminal_slots
                )
            layer0_after_attention_all = mpk_layer0_after_attention.index_select(
                0, snapshot_indices
            )
            layer0_output_all = mpk_layer0_output.index_select(
                0, snapshot_indices
            )
            layer0_input_all = mpk_layer0_input.index_select(
                0, snapshot_indices
            )
            layer0_norm_all = mpk_layer0_norm.index_select(
                0, snapshot_indices
            )
            layer0_attention_output_all = (
                mpk_layer0_attention_output.index_select(0, snapshot_indices)
            )
            layer0_qkv_all = mpk_layer0_qkv.index_select(0, snapshot_indices)
            layer0_post_attention_norm_all = (
                mpk_layer0_post_attention_norm.index_select(
                    0, snapshot_indices
                )
            )
            layer0_mlp_mid_all = mpk_layer0_mlp_mid.index_select(
                0, snapshot_indices
            )
            layer0_silu_mul_all = mpk_layer0_silu_mul.index_select(
                0, snapshot_indices
            )
        if total_num_requests == 1:
            hidden = hidden_all[0]
            output_logits = output_logits_all[0]
            candidate_ids = torch.topk(output_logits.float(), 8).indices
            candidate_weights = model.lm_head.weight.index_select(0, candidate_ids)
            fp32_candidate_logits = torch.mv(
                candidate_weights.float(), hidden.float()
            )
            prefix_token_ids = tokens[0, :prompt_len + max(output_len - 1, 0)]
            generated_token_ids = tokens[0, prompt_len:prompt_len + output_len]
            layer0_after_attention = (
                None if layer0_after_attention_all is None
                else layer0_after_attention_all[0]
            )
            layer0_output = (
                None if layer0_output_all is None else layer0_output_all[0]
            )
            layer0_input = (
                None if layer0_input_all is None else layer0_input_all[0]
            )
            layer0_norm = (
                None if layer0_norm_all is None else layer0_norm_all[0]
            )
            layer0_attention_output = (
                None if layer0_attention_output_all is None
                else layer0_attention_output_all[0]
            )
            layer0_qkv = (
                None if layer0_qkv_all is None else layer0_qkv_all[0]
            )
            layer0_post_attention_norm = (
                None if layer0_post_attention_norm_all is None
                else layer0_post_attention_norm_all[0]
            )
            layer0_mlp_mid = (
                None if layer0_mlp_mid_all is None else layer0_mlp_mid_all[0]
            )
            layer0_silu_mul = (
                None if layer0_silu_mul_all is None
                else layer0_silu_mul_all[0]
            )
        else:
            hidden = hidden_all
            output_logits = output_logits_all
            candidate_ids = torch.topk(output_logits.float(), 8, dim=-1).indices
            candidate_weights = model.lm_head.weight[candidate_ids]
            fp32_candidate_logits = torch.einsum(
                "bkh,bh->bk", candidate_weights.float(), hidden.float()
            )
            prefix_token_ids = tokens[:, :prompt_len + max(output_len - 1, 0)]
            generated_token_ids = tokens[:, prompt_len:prompt_len + output_len]
            layer0_after_attention = layer0_after_attention_all
            layer0_output = layer0_output_all
            layer0_input = layer0_input_all
            layer0_norm = layer0_norm_all
            layer0_attention_output = layer0_attention_output_all
            layer0_qkv = layer0_qkv_all
            layer0_post_attention_norm = layer0_post_attention_norm_all
            layer0_mlp_mid = layer0_mlp_mid_all
            layer0_silu_mul = layer0_silu_mul_all
        os.makedirs(os.path.dirname(os.path.abspath(args.save_intermediates)), exist_ok=True)
        snapshot = {
            "backend": args.backend,
            "policy": args.mpk_policy,
            "prompt_length": prompt_len,
            "prefix_token_ids": prefix_token_ids.cpu(),
            "generated_token_ids": generated_token_ids.cpu(),
            "decode_step_index": output_len - 2,
            "debug_split_mpk_prefill": args.debug_split_mpk_prefill,
            "debug_load_prefill_kv": args.debug_load_prefill_kv,
            "logits": output_logits.detach().cpu(),
            "normalized_hidden_state": hidden.detach().cpu(),
            "candidate_token_ids": candidate_ids.detach().cpu(),
            "fp32_recomputed_candidate_logits": fp32_candidate_logits.detach().cpu(),
        }
        if layer0_after_attention is not None:
            snapshot["layer0_after_attention"] = (
                layer0_after_attention.detach().cpu()
            )
        if layer0_output is not None:
            snapshot["layer0_output"] = layer0_output.detach().cpu()
        if layer0_input is not None:
            snapshot["layer0_input"] = layer0_input.detach().cpu()
        if layer0_norm is not None:
            snapshot["layer0_norm"] = layer0_norm.detach().cpu()
        if layer0_attention_output is not None:
            snapshot["layer0_attention_output"] = (
                layer0_attention_output.detach().cpu()
            )
        if layer0_qkv is not None:
            snapshot["layer0_qkv"] = layer0_qkv.detach().cpu()
        if layer0_post_attention_norm is not None:
            snapshot["layer0_post_attention_norm"] = (
                layer0_post_attention_norm.detach().cpu()
            )
        if layer0_mlp_mid is not None:
            snapshot["layer0_mlp_mid"] = layer0_mlp_mid.detach().cpu()
        if layer0_silu_mul is not None:
            snapshot["layer0_silu_mul"] = layer0_silu_mul.detach().cpu()
        torch.save(snapshot, args.save_intermediates)
        print(f"Saved intermediates to {args.save_intermediates}")
