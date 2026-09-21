from .base import MPKExecutionPolicy


class PrefillOnlyPolicy(MPKExecutionPolicy):
    def should_use_mpk(self, workload) -> bool:
        return workload.phase == "prefill"
