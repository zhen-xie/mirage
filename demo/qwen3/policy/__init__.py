"""Qwen3 execution policy selection."""

from .always import AlwaysPolicy
from .decode_only import DecodeOnlyPolicy
from .prefill_only import PrefillOnlyPolicy
from .workload_aware import WorkloadAwarePolicy


POLICIES = {
    "always": AlwaysPolicy,
    "decode-only": DecodeOnlyPolicy,
    "prefill-only": PrefillOnlyPolicy,
    "workload-aware": WorkloadAwarePolicy,
}


def make_policy(name):
    try:
        return POLICIES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown MPK policy: {name}") from error
