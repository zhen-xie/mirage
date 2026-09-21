from .base import MPKExecutionPolicy


class AlwaysPolicy(MPKExecutionPolicy):
    def should_use_mpk(self, workload) -> bool:
        return True
