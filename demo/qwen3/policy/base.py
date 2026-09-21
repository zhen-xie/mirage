"""Execution policy interface."""

from abc import ABC, abstractmethod


class MPKExecutionPolicy(ABC):
    @abstractmethod
    def should_use_mpk(self, workload) -> bool:
        """Return whether this computation should run through MPK."""
