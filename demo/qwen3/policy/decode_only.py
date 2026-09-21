from .base import MPKExecutionPolicy


class DecodeOnlyPolicy(MPKExecutionPolicy):
    def should_use_mpk(self, workload) -> bool:
        return workload.phase == "decode"
