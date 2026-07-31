from __future__ import annotations

import time
from collections.abc import Callable

import torch

from experiments.runtime.expert_slot import ExpertSlot, ResidentExpert
from experiments.runtime.prefetch_scheduler import ordered_active_experts
from experiments.runtime.serial_expert_executor import (
    HostWeightProvider,
    RoutedExpertTokens,
    SerialExpertExecutor,
    interval_overlap_ms,
)
from experiments.runtime.transient_single_buffer import TransientSingleBuffer


class SerialNoPrefetchExecutor:
    """Strict fetch-wait-compute execution with one transient Expert slot."""

    def __init__(
        self,
        buffer: TransientSingleBuffer,
        host_weights: HostWeightProvider,
        resident_lookup: Callable[[int, int], ResidentExpert | None],
        on_resident_hit: Callable[[int, int], None] | None = None,
        on_transient_complete: (
            Callable[[int, int, ExpertSlot, TransientSingleBuffer], None] | None
        ) = None,
        track_timeline: bool = False,
    ):
        self.buffer = buffer
        self.host_weights = host_weights
        self.resident_lookup = resident_lookup
        self.on_resident_hit = on_resident_hit or (lambda _layer, _expert: None)
        self.on_transient_complete = on_transient_complete or (
            lambda _layer, _expert, _slot, _buffer: None
        )
        self.track_timeline = track_timeline
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

    def execute_layer(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        routed_tokens: dict[int, RoutedExpertTokens],
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        copy_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        wait_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        compute_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for expert_id in ordered_active_experts(routed_tokens):
            resident = self.resident_lookup(layer_id, expert_id)
            slot = None
            weights = resident.tensors if resident is not None else None
            if resident is None:
                slot = self.buffer.slot
                host_started = time.perf_counter()
                source = self.host_weights(layer_id, expert_id)
                timing = slot.enqueue_copy(
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
                self.h2d_bytes += slot.bytes
                weights = slot.tensors
            with torch.cuda.stream(self.compute_stream):
                if slot is not None:
                    wait_started = wait_stopped = None
                    if self.track_timeline:
                        wait_started = torch.cuda.Event(enable_timing=True)
                        wait_stopped = torch.cuda.Event(enable_timing=True)
                        wait_started.record(self.compute_stream)
                    slot.wait_until_ready(self.compute_stream)
                    if wait_stopped is not None:
                        wait_stopped.record(self.compute_stream)
                        wait_timings.append((wait_started, wait_stopped))
                compute_started = compute_stopped = None
                if self.track_timeline:
                    compute_started = torch.cuda.Event(enable_timing=True)
                    compute_stopped = torch.cuda.Event(enable_timing=True)
                    compute_started.record(self.compute_stream)
                routed = routed_tokens[expert_id]
                expert_input = hidden_states.index_select(0, routed.token_indices)
                expert_output = SerialExpertExecutor._mlp(expert_input, weights)
                expert_output = expert_output * routed.routing_weights[:, None]
                output.index_add_(0, routed.token_indices, expert_output)
                if compute_stopped is not None:
                    compute_stopped.record(self.compute_stream)
                    compute_timings.append((compute_started, compute_stopped))
                if slot is not None:
                    slot.record_compute_done(self.compute_stream)
                elif resident.slot is not None:
                    resident.slot.record_compute_done(self.compute_stream)
            self.expert_executions += 1
            if slot is None:
                self.on_resident_hit(layer_id, expert_id)
            else:
                self.on_transient_complete(
                    layer_id, expert_id, slot, self.buffer
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
            "expert_h2d_bytes": self.h2d_bytes,
            "expert_executions": self.expert_executions,
            "compute_streams": 1,
            "copy_streams": 1,
            "prefetch_depth": 0,
            "transient_slots": 1,
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
