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
        host_staging: bool = False,
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
        self._copy_source: dict[str, torch.Tensor] | None = None
        self._host_staging: dict[str, torch.Tensor] | None = None
        if host_staging:
            self.ensure_host_staging()

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
        record_timing: bool = False,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if set(source) != set(self.tensors):
            raise ValueError("source tensors do not match the Expert slot layout")
        if self._has_copy_event:
            # The pinned allocation must stay alive until its asynchronous H2D
            # completes. Slot reuse occurs infrequently enough that this host
            # wait is normally already satisfied by the intervening compute.
            self.copy_done.synchronize()
        if all(tensor.is_pinned() for tensor in source.values()):
            copy_source = dict(source)
        else:
            self.ensure_host_staging()
            copy_source = self._host_staging
            for name, host_tensor in source.items():
                copy_source[name].copy_(host_tensor)
        self._copy_source = copy_source
        copy_started = (
            torch.cuda.Event(enable_timing=True) if record_timing else None
        )
        if record_timing:
            self.copy_done = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            if self._has_compute_event:
                copy_stream.wait_event(self.compute_done)
            if copy_started is not None:
                copy_started.record(copy_stream)
            for name, destination in self.tensors.items():
                host_tensor = copy_source[name]
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
        if copy_started is None:
            return None
        return copy_started, self.copy_done

    def wait_until_ready(self, compute_stream: torch.cuda.Stream) -> None:
        if not self._has_copy_event:
            raise RuntimeError("cannot compute from a slot before its first H2D copy")
        compute_stream.wait_event(self.copy_done)

    def record_compute_done(self, compute_stream: torch.cuda.Stream) -> None:
        self.compute_done.record(compute_stream)
        self._has_compute_event = True

    def release_copy_source(self) -> None:
        if self._has_copy_event:
            self.copy_done.synchronize()
        self._copy_source = None

    def ensure_host_staging(self) -> None:
        if self._host_staging is None:
            self._host_staging = {
                name: torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                for name, tensor in self.tensors.items()
            }

    def release_host_staging(self) -> None:
        self.release_copy_source()
        self._host_staging = None


class ResidentExpert:
    """GPU-resident Expert weights used by permanent and quota policies."""

    def __init__(
        self,
        layer_id: int,
        expert_id: int,
        tensors: Mapping[str, torch.Tensor],
        slot: ExpertSlot | None = None,
    ):
        if any(tensor.device.type != "cuda" for tensor in tensors.values()):
            raise ValueError("resident Expert weights must be CUDA tensors")
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.tensors = dict(tensors)
        self.slot = slot
