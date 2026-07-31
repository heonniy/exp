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
    ):
        self.buffer = buffer
        self.host_weights = host_weights
        self.resident_lookup = resident_lookup
        self.on_resident_hit = on_resident_hit or (lambda _layer, _expert: None)
        self.on_transient_complete = on_transient_complete or (
            lambda _layer, _expert, _slot, _buffer: None
        )
        self.compute_stream = torch.cuda.Stream(device=0)
        self.copy_stream = torch.cuda.Stream(device=0)
        self.fetches = 0
        self.h2d_bytes = 0
        self.expert_executions = 0
        self.host_prepare_seconds = 0.0

    def execute_layer(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        routed_tokens: dict[int, RoutedExpertTokens],
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_id in ordered_active_experts(routed_tokens):
            resident = self.resident_lookup(layer_id, expert_id)
            slot = None
            weights = resident.tensors if resident is not None else None
            if resident is None:
                slot = self.buffer.slot
                host_started = time.perf_counter()
                source = self.host_weights(layer_id, expert_id)
                slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    source=source,
                    copy_stream=self.copy_stream,
                )
                self.host_prepare_seconds += time.perf_counter() - host_started
                self.fetches += 1
                self.h2d_bytes += slot.bytes
                weights = slot.tensors
            with torch.cuda.stream(self.compute_stream):
                if slot is not None:
                    slot.wait_until_ready(self.compute_stream)
                routed = routed_tokens[expert_id]
                expert_input = hidden_states.index_select(0, routed.token_indices)
                expert_output = SerialExpertExecutor._mlp(expert_input, weights)
                expert_output = expert_output * routed.routing_weights[:, None]
                output.index_add_(0, routed.token_indices, expert_output)
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
        }

