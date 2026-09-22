"""Check saved outputs from a single Qwen3 GPU environment.

Set QWEN3_CORRECTNESS_DIR to a directory containing the four
new_server_{normal,always,decode-only,prefill-only}_128.json files.
The GPU runs must use the same model, prompt, weights, and generation options.
"""

import json
import os
from pathlib import Path

import pytest


MODES = {
    "normal": "torch",
    "always": "mpk",
    "decode-only": "normal_prefill_mpk_decode",
    "prefill-only": "mpk_prefill_normal_decode",
}
PREFIX_TOKENS = 30


def test_saved_backend_outputs_match_normal_prefix():
    directory = os.environ.get("QWEN3_CORRECTNESS_DIR")
    if not directory:
        pytest.skip("Set QWEN3_CORRECTNESS_DIR after running all four GPU modes")

    base = Path(directory)
    outputs = {}
    for policy, expected_mode in MODES.items():
        path = base / f"new_server_{policy}_128.json"
        data = json.loads(path.read_text())
        assert data["mode"] == expected_mode, policy
        assert data["prompt_length"] == 128, policy
        assert data["generate_length"] == 128, policy
        assert len(data["token_ids"]) >= PREFIX_TOKENS, policy
        outputs[policy] = data["token_ids"]

    reference = outputs["normal"][:PREFIX_TOKENS]
    for policy, tokens in outputs.items():
        assert tokens[:PREFIX_TOKENS] == reference, policy
