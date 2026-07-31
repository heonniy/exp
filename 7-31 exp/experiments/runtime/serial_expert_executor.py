from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Protocol

import torch
import torch.nn.functional as functional

from experiments.runtime.expert_slot import ExpertSlot, ResidentExpert
from experiments.runtime.prefetch_scheduler import (
    first_future_nonresident,
    ordered_active_experts,
)
from experiments.runtime.transient_double_buffer import TransientDoubleBuffer


@dataclass(frozen=True)
class RoutedExpertTokens:
    token_indices: torch.Tensor
    routing_weights: torch.Tensor


class HostWeightProvider(Protocol):
    def __call__(
        self, layer_id: int, expert_id: int
    ) -> Mapping[str, torch.Tensor] | torch.Tensor:
        """Return a packed Expert or projection tensors for slot-side packing."""


def interval_overlap_ms(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    left = sorted(first)
    right = sorted(second)
    i = j = 0
    overlap = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        stop = min(left[i][1], right[j][1])
        overlap += max(0.0, stop - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return overlap


class SerialExpertExecutor:
    """Individual per-Expert MLP with current-layer one-ahead H2D."""

    def __init__(
        self,
        double_buffer: TransientDoubleBuffer,
        host_weights: HostWeightProvider,
        resident_lookup: Callable[[int, int], ResidentExpert | None],
        on_resident_hit: Callable[[int, int], None] | None = None,
        on_transient_complete: (
            Callable[[int, int, ExpertSlot, TransientDoubleBuffer], None] | None
        ) = None,
        track_timeline: bool = False,
        prefetch_submit_order: str = "compute_first",
    ):
        self.double_buffer = double_buffer
        self.host_weights = host_weights
        self.resident_lookup = resident_lookup
        self.on_resident_hit = on_resident_hit or (lambda _layer, _expert: None)
        self.on_transient_complete = on_transient_complete or (
            lambda _layer, _expert, _slot, _buffer: None
        )
        self.track_timeline = track_timeline
        if prefetch_submit_order not in {"compute_first", "copy_first"}:
            raise ValueError("invalid prefetch submit order")
        self.prefetch_submit_order = prefetch_submit_order
        self.compute_stream = torch.cuda.Stream(device=0)
        self.copy_stream = torch.cuda.Stream(device=0)
        self.timeline_origin = None
        if self.track_timeline:
            self.timeline_origin = torch.cuda.Event(enable_timing=True)
            self.timeline_origin.record(torch.cuda.current_stream())
            self.compute_stream.wait_event(self.timeline_origin)
            self.copy_stream.wait_event(self.timeline_origin)
        self.fetches = 0
        self.h2d_bytes = 0
        self.expert_executions = 0
        self.host_prepare_seconds = 0.0
        self.total_h2d_ms = 0.0
        self.overlapped_h2d_ms = 0.0
        self.compute_stream_h2d_wait_ms = 0.0
        self.first_miss_stall_ms = 0.0
        self.layers_with_misses = 0
        self.expert_compute_ms = 0.0

    @staticmethod
    def _mlp(x: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "gate_up_proj" in weights:
            gate, up = functional.linear(x, weights["gate_up_proj"]).chunk(
                2, dim=-1
            )
        else:
            gate = functional.linear(x, weights["gate_proj"])
            up = functional.linear(x, weights["up_proj"])
        hidden = functional.silu(gate) * up
        return functional.linear(hidden, weights["down_proj"])

    def execute_layer(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        routed_tokens: Mapping[int, RoutedExpertTokens],
    ) -> torch.Tensor:
        active = ordered_active_experts(routed_tokens)
        output = torch.zeros_like(hidden_states)
        if not active:
            return output
        copy_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        wait_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        compute_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        def resident(expert_id: int) -> bool:
            return self.resident_lookup(layer_id, expert_id) is not None

        staged: dict[int, ExpertSlot] = {}
        next_slot = self.double_buffer.slots[0]
        first_miss = next(
            (expert_id for expert_id in active if not resident(expert_id)),
            None,
        )
        if first_miss is not None:
            host_started = time.perf_counter()
            source = self.host_weights(layer_id, first_miss)
            timing = next_slot.enqueue_copy(
                layer_id=layer_id,
                expert_id=first_miss,
                source=source,
                copy_stream=self.copy_stream,
                record_timing=self.track_timeline,
            )
            if timing is not None:
                copy_timings.append(timing)
            self.host_prepare_seconds += time.perf_counter() - host_started
            self.fetches += 1
            self.h2d_bytes += next_slot.bytes
            staged[first_miss] = next_slot

        for position, expert_id in enumerate(active):
            resident_expert = self.resident_lookup(layer_id, expert_id)
            if resident_expert is None and expert_id not in staged:
                # A future quota-LRU hit can be evicted by admissions that occur
                # earlier in the fixed ascending execution order. Revalidate at
                # use time and synchronously stage it if that happened.
                occupied = set(staged.values())
                current_fallback_slot = next(
                    slot for slot in self.double_buffer.slots if slot not in occupied
                )
                host_started = time.perf_counter()
                source = self.host_weights(layer_id, expert_id)
                timing = current_fallback_slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    source=source,
                    copy_stream=self.copy_stream,
                    record_timing=self.track_timeline,
                )
                if timing is not None:
                    copy_timings.append(timing)
                self.host_prepare_seconds += time.perf_counter() - host_started
                self.fetches += 1
                self.h2d_bytes += current_fallback_slot.bytes
                staged[expert_id] = current_fallback_slot
            current_slot = None if resident_expert is not None else staged.pop(expert_id)
            current_weights = (
                resident_expert.tensors
                if resident_expert is not None
                else current_slot.tensors
            )

            # Depth=1 means one staged future miss at most. Do not skip over an
            # already staged miss and accidentally create a deeper queue.
            future = (
                None
                if staged
                else first_future_nonresident(active, position, resident)
            )
            def stage_future() -> None:
                if future is None:
                    return
                if current_slot is not None:
                    prefetch_slot = self.double_buffer.other(current_slot)
                else:
                    occupied = set(staged.values())
                    prefetch_slot = next(
                        slot for slot in self.double_buffer.slots if slot not in occupied
                    )
                host_started = time.perf_counter()
                source = self.host_weights(layer_id, future)
                timing = prefetch_slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=future,
                    source=source,
                    copy_stream=self.copy_stream,
                    record_timing=self.track_timeline,
                )
                if timing is not None:
                    copy_timings.append(timing)
                self.host_prepare_seconds += time.perf_counter() - host_started
                self.fetches += 1
                self.h2d_bytes += prefetch_slot.bytes
                staged[future] = prefetch_slot

            if self.prefetch_submit_order == "copy_first":
                stage_future()

            with torch.cuda.stream(self.compute_stream):
                if current_slot is not None:
                    wait_started = wait_stopped = None
                    if self.track_timeline:
                        wait_started = torch.cuda.Event(enable_timing=True)
                        wait_stopped = torch.cuda.Event(enable_timing=True)
                        wait_started.record(self.compute_stream)
                    current_slot.wait_until_ready(self.compute_stream)
                    if wait_stopped is not None:
                        wait_stopped.record(self.compute_stream)
                        wait_timings.append((wait_started, wait_stopped))
                compute_started = compute_stopped = None
                if self.track_timeline:
                    compute_started = torch.cuda.Event(enable_timing=True)
                    compute_stopped = torch.cuda.Event(enable_timing=True)
                    compute_started.record(self.compute_stream)
                routed = routed_tokens[expert_id]
                token_indices = routed.token_indices
                expert_input = hidden_states.index_select(0, token_indices)
                expert_output = self._mlp(expert_input, current_weights)
                expert_output = expert_output * routed.routing_weights[:, None]
                output.index_add_(0, token_indices, expert_output)
                if compute_stopped is not None:
                    compute_stopped.record(self.compute_stream)
                    compute_timings.append((compute_started, compute_stopped))
                if current_slot is not None:
                    current_slot.record_compute_done(self.compute_stream)
                elif resident_expert.slot is not None:
                    resident_expert.slot.record_compute_done(self.compute_stream)
            if self.prefetch_submit_order == "compute_first":
                stage_future()
            self.expert_executions += 1
            if current_slot is None:
                self.on_resident_hit(layer_id, expert_id)
            else:
                self.on_transient_complete(
                    layer_id, expert_id, current_slot, self.double_buffer
                )
        self.compute_stream.synchronize()
        if self.track_timeline:
            copy_ms = sum(start.elapsed_time(stop) for start, stop in copy_timings)
            waits = [start.elapsed_time(stop) for start, stop in wait_timings]
            copy_intervals = [
                (
                    self.timeline_origin.elapsed_time(start),
                    self.timeline_origin.elapsed_time(stop),
                )
                for start, stop in copy_timings
            ]
            compute_intervals = [
                (
                    self.timeline_origin.elapsed_time(start),
                    self.timeline_origin.elapsed_time(stop),
                )
                for start, stop in compute_timings
            ]
            self.total_h2d_ms += copy_ms
            self.overlapped_h2d_ms += interval_overlap_ms(
                copy_intervals, compute_intervals
            )
            self.compute_stream_h2d_wait_ms += sum(waits)
            self.expert_compute_ms += sum(
                start.elapsed_time(stop) for start, stop in compute_timings
            )
            if waits:
                self.first_miss_stall_ms += waits[0]
                self.layers_with_misses += 1
        return output

    def metrics(self) -> dict:
        return {
            "expert_h2d_fetches": self.fetches,
            "expert_h2d_copy_operations": self.fetches,
            "h2d_copy_operations_per_fetch": 1,
            "gpu_expert_layout": "single_contiguous_buffer_with_projection_views",
            "expert_gate_up_execution": "single_zero_copy_gate_up_projection",
            "expert_h2d_bytes": self.h2d_bytes,
            "expert_executions": self.expert_executions,
            "compute_streams": 1,
            "copy_streams": 1,
            "prefetch_depth": 1,
            "prefetch_submit_order": self.prefetch_submit_order,
            "transient_slots": 2,
            "host_prepare_seconds": self.host_prepare_seconds,
            "timeline_events_enabled": self.track_timeline,
            "total_h2d_duration_ms": (
                self.total_h2d_ms if self.track_timeline else None
            ),
            "exposed_h2d_stall_ms": (
                max(0.0, self.total_h2d_ms - self.overlapped_h2d_ms)
                if self.track_timeline
                else None
            ),
            "overlapped_h2d_ms": (
                self.overlapped_h2d_ms if self.track_timeline else None
            ),
            "overlap_ratio": (
                max(0.0, min(1.0, self.overlapped_h2d_ms / self.total_h2d_ms))
                if self.track_timeline and self.total_h2d_ms
                else None
            ),
            "compute_stream_h2d_wait_ms": (
                self.compute_stream_h2d_wait_ms if self.track_timeline else None
            ),
            "first_miss_stall_ms": (
                self.first_miss_stall_ms if self.track_timeline else None
            ),
            "expert_compute_ms": (
                self.expert_compute_ms if self.track_timeline else None
            ),
            "layers_with_misses": self.layers_with_misses,
        }
