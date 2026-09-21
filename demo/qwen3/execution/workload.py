"""Small, backend-independent description of the current Qwen3 work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadDescriptor:
    phase: str
    batch_size: int
    context_length: int
    decode_step: int
    active_tokens: int

    def __post_init__(self):
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"Unknown phase: {self.phase}")
        if self.batch_size < 1 or self.context_length < 1:
            raise ValueError("batch_size and context_length must be positive")
        if self.decode_step < 0 or self.active_tokens < 1:
            raise ValueError("decode_step must be nonnegative and active_tokens positive")

    @classmethod
    def for_prefill(cls, batch_size: int, input_length: int):
        return cls("prefill", batch_size, input_length, 0,
                   batch_size * input_length)

    @classmethod
    def for_decode(cls, batch_size: int, input_length: int, decode_step: int):
        if decode_step < 0:
            raise ValueError("decode_step must be nonnegative")
        return cls("decode", batch_size, input_length + decode_step,
                   decode_step, batch_size)
