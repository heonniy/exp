from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class SlotIdentity:
    layer_id: int | None = None
    expert_id: int | None = None


class ExpertSlot:
    """One GPU Expert buffer guarded by copy/compute CUDA events."""

    def __init__(
        self,
        slot_id: int,
        tensor_shapes: Mapping[str, tuple[int, ...]],
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda:0",
    ):
        self.slot_id = slot_id
        self.tensors = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, shape in tensor_shapes.items()
        }
        self.identity = SlotIdentity()
        self.copy_done = torch.cuda.Event(blocking=False)
        self.compute_done = torch.cuda.Event(blocking=False)
        self._has_compute_event = False
        self._has_copy_event = False

    @property
    def bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors.values())

    def enqueue_copy(
        self,
        *,
        layer_id: int,
        expert_id: int,
        source: Mapping[str, torch.Tensor],
        copy_stream: torch.cuda.Stream,
    ) -> None:
        if set(source) != set(self.tensors):
            raise ValueError("source tensors do not match the Expert slot layout")
        with torch.cuda.stream(copy_stream):
            if self._has_compute_event:
                copy_stream.wait_event(self.compute_done)
            for name, destination in self.tensors.items():
                host_tensor = source[name]
                if host_tensor.device.type != "cpu":
                    raise ValueError("Expert H2D source must reside on CPU")
                if not host_tensor.is_pinned():
                    raise ValueError("Expert H2D source must use pinned memory")
                if tuple(host_tensor.shape) != tuple(destination.shape):
                    raise ValueError(f"shape mismatch for {name}")
                destination.copy_(host_tensor, non_blocking=True)
            self.copy_done.record(copy_stream)
        self.identity = SlotIdentity(layer_id=layer_id, expert_id=expert_id)
        self._has_copy_event = True

    def wait_until_ready(self, compute_stream: torch.cuda.Stream) -> None:
        if not self._has_copy_event:
            raise RuntimeError("cannot compute from a slot before its first H2D copy")
        compute_stream.wait_event(self.copy_done)

    def record_compute_done(self, compute_stream: torch.cuda.Stream) -> None:
        self.compute_done.record(compute_stream)
        self._has_compute_event = True


class ResidentExpert:
    """GPU-resident Expert weights used by permanent and quota policies."""

    def __init__(self, layer_id: int, expert_id: int, tensors: Mapping[str, torch.Tensor]):
        if any(tensor.device.type != "cuda" for tensor in tensors.values()):
            raise ValueError("resident Expert weights must be CUDA tensors")
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.tensors = dict(tensors)

