from .base import MPKExecutionPolicy


class WorkloadAwarePolicy(MPKExecutionPolicy):
    def should_use_mpk(self, workload) -> bool:
        raise NotImplementedError(
            "Workload-aware decisions require the measured advantage map"
        )
