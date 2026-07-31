from __future__ import annotations

from collections.abc import Mapping

import torch

from experiments.runtime.expert_slot import ExpertSlot


class TransientSingleBuffer:
    """One transient slot for the no-prefetch micro-ablation."""

    def __init__(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda:0",
    ):
        self.slot = ExpertSlot(
            0, tensor_shapes, dtype, device, host_staging=True
        )
        self.slots = [self.slot]

    @property
    def bytes(self) -> int:
        return self.slot.bytes

    def replace(self, current: ExpertSlot, replacement: ExpertSlot) -> None:
        if current is not self.slot:
            raise ValueError("current slot does not belong to this buffer")
        self.slot = replacement
        self.slots[0] = replacement

