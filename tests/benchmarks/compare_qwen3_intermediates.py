"""Compare normal and MPK Qwen3 correctness probes at one decode step."""

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
    reference64 = reference.double().flatten()
    actual64 = actual.double().flatten()
    cosine = (torch.dot(reference64, actual64) /
              (torch.linalg.vector_norm(reference64) *
               torch.linalg.vector_norm(actual64))).item()
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "cosine_similarity": cosine,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("normal")
    parser.add_argument("mpk_probe")
    args = parser.parse_args()

    normal = torch.load(args.normal, map_location="cpu", weights_only=True)
    mpk = torch.load(args.mpk_probe, map_location="cpu", weights_only=True)
    if normal["backend"] != "normal" or mpk["policy"] not in ("decode-only", "prefill-only", "always"):
        raise ValueError("Expected normal and a static MPK-policy probe")
    if normal["prompt_length"] != mpk["prompt_length"]:
        raise ValueError("Prompt lengths differ")
    if not torch.equal(normal["prefix_token_ids"], mpk["prefix_token_ids"]):
        raise ValueError("Input or generated tokens before the probed step differ")
    if normal["decode_step_index"] != mpk["decode_step_index"]:
        raise ValueError("Probes target different decode steps")
    for name, probe in (("normal", normal), (mpk["policy"], mpk)):
        predicted = probe["logits"].argmax(dim=-1)
        generated = probe["generated_token_ids"][..., -1]
        if not torch.equal(predicted, generated):
            raise ValueError(
                f"{name} captured logits do not predict the saved generated "
                "tokens; probe is not aligned"
            )

    def top_tokens(probe):
        values, indices = torch.topk(probe["logits"].float(), 5, dim=-1)
        if probe["logits"].ndim == 1:
            return [{"token_id": token.item(), "logit": value.item()}
                    for token, value in zip(indices, values)]
        return [
            [{"token_id": token.item(), "logit": value.item()}
             for token, value in zip(row_indices, row_values)]
            for row_indices, row_values in zip(indices, values)
        ]

    def fp32_candidates(probe):
        if "fp32_recomputed_candidate_logits" not in probe:
            return None
        token_ids = probe["candidate_token_ids"]
        scores = probe["fp32_recomputed_candidate_logits"]
        if token_ids.ndim == 1:
            candidates = [
                {"token_id": token.item(), "logit": score.item()}
                for token, score in zip(token_ids, scores)
            ]
            return sorted(
                candidates, key=lambda item: item["logit"], reverse=True
            )
        return [
            sorted(
                [{"token_id": token.item(), "logit": score.item()}
                 for token, score in zip(row_tokens, row_scores)],
                key=lambda item: item["logit"],
                reverse=True,
            )
            for row_tokens, row_scores in zip(token_ids, scores)
        ]

    report = {
        "mpk_policy": mpk["policy"],
        "prompt_length": normal["prompt_length"],
        "decode_step_index": normal["decode_step_index"],
        "probed_generated_token_matches": torch.equal(
            normal["generated_token_ids"][..., -1],
            mpk["generated_token_ids"][..., -1],
        ),
        "generated_tokens_match_through_probe": torch.equal(
            normal["generated_token_ids"], mpk["generated_token_ids"]
        ),
        "normal_generated_token_ids": normal["generated_token_ids"].tolist(),
        "mpk_generated_token_ids": mpk["generated_token_ids"].tolist(),
        "normal_top_5": top_tokens(normal),
        "mpk_top_5": top_tokens(mpk),
        "normal_fp32_candidates": fp32_candidates(normal),
        "mpk_fp32_candidates": fp32_candidates(mpk),
        "logits": tensor_metrics(normal["logits"], mpk["logits"]),
        "normalized_hidden_state": tensor_metrics(
            normal["normalized_hidden_state"], mpk["normalized_hidden_state"]
        ),
    }
    if normal["logits"].ndim == 2:
        report["per_request"] = []
        for request_id in range(normal["logits"].shape[0]):
            report["per_request"].append({
                "request_id": request_id,
                "generated_token_matches": (
                    normal["generated_token_ids"][request_id, -1].item()
                    == mpk["generated_token_ids"][request_id, -1].item()
                ),
                "normal_generated_token": normal[
                    "generated_token_ids"
                ][request_id, -1].item(),
                "mpk_generated_token": mpk[
                    "generated_token_ids"
                ][request_id, -1].item(),
                "logits": tensor_metrics(
                    normal["logits"][request_id],
                    mpk["logits"][request_id],
                ),
                "normalized_hidden_state": tensor_metrics(
                    normal["normalized_hidden_state"][request_id],
                    mpk["normalized_hidden_state"][request_id],
                ),
            })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
