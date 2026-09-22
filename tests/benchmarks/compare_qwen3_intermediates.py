"""Compare a two-token normal/decode-only Qwen3 correctness probe."""

import argparse
import json

import torch


def tensor_metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Tensor shapes differ: {reference.shape} vs {actual.shape}")
    reference = reference.float()
    actual = actual.float()
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Non-finite values in intermediate tensors")
    difference = (reference - actual).abs()
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), actual.reshape(1, -1)
    ).item()
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": cosine,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("normal")
    parser.add_argument("decode_only")
    args = parser.parse_args()

    normal = torch.load(args.normal, map_location="cpu", weights_only=True)
    mpk = torch.load(args.decode_only, map_location="cpu", weights_only=True)
    if normal["backend"] != "normal" or mpk["policy"] != "decode-only":
        raise ValueError("Expected normal and MPK decode-only probes")
    if normal["prompt_length"] != mpk["prompt_length"]:
        raise ValueError("Prompt lengths differ")
    if not torch.equal(normal["prefix_token_ids"], mpk["prefix_token_ids"]):
        raise ValueError("Input and first generated token differ")
    for name, probe in (("normal", normal), ("decode-only", mpk)):
        predicted = probe["logits"].argmax().item()
        generated = probe["generated_token_ids"][1].item()
        if predicted != generated:
            raise ValueError(
                f"{name} captured logits predict token {predicted}, "
                f"but generation produced {generated}; probe is not aligned"
            )

    report = {
        "prompt_length": normal["prompt_length"],
        "first_two_generated_tokens_match": torch.equal(
            normal["generated_token_ids"], mpk["generated_token_ids"]
        ),
        "normal_generated_token_ids": normal["generated_token_ids"].tolist(),
        "decode_only_generated_token_ids": mpk["generated_token_ids"].tolist(),
        "logits": tensor_metrics(normal["logits"], mpk["logits"]),
        "normalized_hidden_state": tensor_metrics(
            normal["normalized_hidden_state"], mpk["normalized_hidden_state"]
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["first_two_generated_tokens_match"]:
        raise SystemExit("Generated tokens differ")


if __name__ == "__main__":
    main()
