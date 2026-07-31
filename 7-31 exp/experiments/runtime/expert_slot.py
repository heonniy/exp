from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


def contiguous_projection_views(
    buffer: torch.Tensor,
    tensor_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    views = {}
    offsets = {}
    offset = 0
    for name, shape in tensor_shapes.items():
        elements = int(torch.Size(shape).numel())
        offsets[name] = offset
        views[name] = buffer[offset : offset + elements].view(shape)
        offset += elements
    if offset != buffer.numel():
        raise ValueError("projection shapes do not cover the Expert buffer")
    gate_shape = tensor_shapes.get("gate_proj")
    up_shape = tensor_shapes.get("up_proj")
    if gate_shape is not None and up_shape is not None:
        gate_elements = int(torch.Size(gate_shape).numel())
        if (
            len(gate_shape) == 2
            and tuple(gate_shape) == tuple(up_shape)
            and offsets["up_proj"] == offsets["gate_proj"] + gate_elements
        ):
            start = offsets["gate_proj"]
            views["gate_up_proj"] = buffer[
                start : start + 2 * gate_elements
            ].view(gate_shape[0] * 2, gate_shape[1])
    return views


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
        self.tensor_shapes = dict(tensor_shapes)
        total_elements = sum(
            int(torch.Size(shape).numel()) for shape in self.tensor_shapes.values()
        )
        self.buffer = torch.empty(
            total_elements, dtype=dtype, device=device
        )
        self.tensors = self._views(self.buffer)
        self.identity = SlotIdentity()
        self.copy_done = torch.cuda.Event(blocking=False)
        self.compute_done = torch.cuda.Event(blocking=False)
        self._has_compute_event = False
        self._has_copy_event = False
        self._copy_source: torch.Tensor | None = None
        self._host_staging: torch.Tensor | None = None
        self._host_staging_views: dict[str, torch.Tensor] | None = None
        if host_staging:
            self.ensure_host_staging()

    @property
    def bytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()

    def _views(self, buffer: torch.Tensor) -> dict[str, torch.Tensor]:
        return contiguous_projection_views(buffer, self.tensor_shapes)

    def enqueue_copy(
        self,
        *,
        layer_id: int,
        expert_id: int,
        source: Mapping[str, torch.Tensor] | torch.Tensor,
        copy_stream: torch.cuda.Stream,
        record_timing: bool = False,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if self._has_copy_event:
            # The pinned allocation must stay alive until its asynchronous H2D
            # completes. Slot reuse occurs infrequently enough that this host
            # wait is normally already satisfied by the intervening compute.
            self.copy_done.synchronize()
        if isinstance(source, torch.Tensor):
            copy_source = source
            if source.device.type != "cpu":
                raise ValueError("Expert H2D source must reside on CPU")
            if not source.is_pinned():
                raise ValueError("Expert H2D source must use pinned memory")
            if source.dtype != self.buffer.dtype:
                raise ValueError("Expert H2D source dtype does not match slot")
            if source.numel() != self.buffer.numel() or not source.is_contiguous():
                raise ValueError(
                    "Expert H2D source must be one contiguous flat Expert buffer"
                )
        else:
            if set(source) != set(self.tensor_shapes):
                raise ValueError("source tensors do not match the Expert slot layout")
            self.ensure_host_staging()
            copy_source = self._host_staging
            assert copy_source is not None
            assert self._host_staging_views is not None
            for name, host_tensor in source.items():
                destination = self._host_staging_views[name]
                if tuple(host_tensor.shape) != tuple(destination.shape):
                    raise ValueError(f"shape mismatch for {name}")
                destination.copy_(host_tensor)
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
            # Exactly one H2D operation per Expert fetch. gate/up/down are views
            # over this contiguous destination buffer and are never copied
            # separately.
            self.buffer.copy_(copy_source, non_blocking=True)
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
            self._host_staging = torch.empty(
                self.buffer.numel(),
                dtype=self.buffer.dtype,
                device="cpu",
                pin_memory=True,
            )
            self._host_staging_views = self._views(self._host_staging)

    def release_host_staging(self) -> None:
        self.release_copy_source()
        self._host_staging = None
        self._host_staging_views = None


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
