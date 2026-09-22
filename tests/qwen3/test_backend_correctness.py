"""Check saved outputs from a single Qwen3 GPU environment.

Set QWEN3_CORRECTNESS_DIR to a directory containing four
new_server_*_128.json files and, when available, four diverse_*_128.json files.
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
MIN_MATCHES = 20


@pytest.mark.parametrize("case", ("new_server", "diverse"))
def test_saved_backend_outputs_match_normal_prefix(case):
    directory = os.environ.get("QWEN3_CORRECTNESS_DIR")
    if not directory:
        pytest.skip("Set QWEN3_CORRECTNESS_DIR after running all four GPU modes")

    base = Path(directory)
    if not (base / f"{case}_normal_128.json").exists():
        pytest.skip(f"{case} artifacts are not available")
    outputs = {}
    for policy, expected_mode in MODES.items():
        path = base / f"{case}_{policy}_128.json"
        data = json.loads(path.read_text())
        assert data["mode"] == expected_mode, policy
        assert data["prompt_length"] == 128, policy
        assert data["generate_length"] == 128, policy
        assert len(data["token_ids"]) >= PREFIX_TOKENS, policy
        outputs[policy] = data["token_ids"]

    reference = outputs["normal"][:PREFIX_TOKENS]
    for policy, tokens in outputs.items():
        matches = sum(a == b for a, b in zip(reference, tokens[:PREFIX_TOKENS]))
        print(f"{case} {policy}: {matches}/{PREFIX_TOKENS} positional token matches")
        assert matches >= MIN_MATCHES, f"{case} {policy}: {matches}/{PREFIX_TOKENS}"
