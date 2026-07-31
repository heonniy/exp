from __future__ import annotations

from collections.abc import Mapping

import torch

from experiments.runtime.expert_slot import ExpertSlot


class TransientDoubleBuffer:
    """Exactly two transient Expert slots; never a persistent cache."""

    def __init__(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda:0",
    ):
        self.slots = [
            ExpertSlot(0, tensor_shapes, dtype, device, host_staging=True),
            ExpertSlot(1, tensor_shapes, dtype, device, host_staging=True),
        ]

    @property
    def bytes(self) -> int:
        return sum(slot.bytes for slot in self.slots)

    def other(self, slot: ExpertSlot) -> ExpertSlot:
        if slot is self.slots[0]:
            return self.slots[1]
        if slot is self.slots[1]:
            return self.slots[0]
        raise ValueError("slot does not belong to this double buffer")

    def replace(self, current: ExpertSlot, replacement: ExpertSlot) -> None:
        for index, slot in enumerate(self.slots):
            if slot is current:
                self.slots[index] = replacement
                return
        raise ValueError("current slot does not belong to this double buffer")
