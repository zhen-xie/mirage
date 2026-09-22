"""Compare normal and MPK Qwen3 KV values immediately after prefill."""

import argparse
import json

import torch


def summarize(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"KV shapes differ: {reference.shape} vs {actual.shape}")
    reference = reference.float()
    actual = actual.float()
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Non-finite KV values")
    difference = (reference - actual).abs()
    layer_means = difference.flatten(1).mean(dim=1)
    worst_layers = torch.argsort(layer_means, descending=True)[:5]
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), actual.reshape(1, -1)
    ).item()
    return {
        "shape": list(reference.shape),
        "exact_equal_fraction": (difference == 0).float().mean().item(),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": cosine,
        "worst_layers_by_mean_absolute_error": [
            {"layer": i.item(), "mean_absolute_error": layer_means[i].item()}
            for i in worst_layers
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normal")
    parser.add_argument("mpk_prefill")
    args = parser.parse_args()
    normal = torch.load(args.normal, map_location="cpu", weights_only=True)
    mpk = torch.load(args.mpk_prefill, map_location="cpu", weights_only=True)
    if normal["backend"] != "normal" or mpk["policy"] != "prefill-only":
        raise ValueError("Expected normal and MPK prefill-only KV snapshots")
    if normal["prompt_length"] != mpk["prompt_length"]:
        raise ValueError("Prompt lengths differ")
    if not torch.equal(normal["prompt_token_ids"], mpk["prompt_token_ids"]):
        raise ValueError("Prompt token IDs differ")
    report = {
        "prompt_length": normal["prompt_length"],
        "key_cache": summarize(normal["key_cache"], mpk["key_cache"]),
        "value_cache": summarize(normal["value_cache"], mpk["value_cache"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
