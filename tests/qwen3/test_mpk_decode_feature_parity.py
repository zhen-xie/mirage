"""Keep MPK/Optimized-Normal decode parity claims honest.

This test is CPU-only.  Numerical operator parity is covered by GPU probes;
this file prevents a partially aligned backend from being advertised as fully
aligned in benchmark results.
"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "tests" / "benchmarks" / "qwen3_backend_comparison.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_backend_comparison", BENCHMARK
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mpk_backend_name_does_not_claim_unverified_alignment():
    benchmark = load_benchmark_module()
    assert "mpk_decode_only_aligned_attention" not in benchmark.BACKENDS
    assert "mpk_decode_only_page128_split_kv" in benchmark.BACKENDS
    assert "mpk_decode_only_adaptive_attention" in benchmark.BACKENDS


def test_parity_manifest_covers_every_optimized_normal_feature():
    benchmark = load_benchmark_module()
    assert set(benchmark.MPK_DECODE_FEATURE_PARITY) == set(
        benchmark.OPTIMIZED_NORMAL_DECODE_FEATURES
    )
    for feature, result in benchmark.MPK_DECODE_FEATURE_PARITY.items():
        assert isinstance(result["equivalent"], bool), feature
        assert result["implementation"], feature


def test_full_parity_requires_every_feature_to_pass():
    benchmark = load_benchmark_module()
    expected = all(
        result["equivalent"]
        for result in benchmark.MPK_DECODE_FEATURE_PARITY.values()
    )
    assert benchmark.mpk_decode_parity_complete() is expected
