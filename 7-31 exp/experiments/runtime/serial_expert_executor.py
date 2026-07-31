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
    def __call__(self, layer_id: int, expert_id: int) -> Mapping[str, torch.Tensor]:
        """Return pinned CPU gate/up/down tensors."""


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
    ):
        self.double_buffer = double_buffer
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

    @staticmethod
    def _mlp(x: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
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
            next_slot.enqueue_copy(
                layer_id=layer_id,
                expert_id=first_miss,
                source=source,
                copy_stream=self.copy_stream,
            )
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
                current_fallback_slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    source=source,
                    copy_stream=self.copy_stream,
                )
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
            if future is not None:
                if current_slot is not None:
                    prefetch_slot = self.double_buffer.other(current_slot)
                else:
                    occupied = set(staged.values())
                    prefetch_slot = next(
                        slot for slot in self.double_buffer.slots if slot not in occupied
                    )
                host_started = time.perf_counter()
                source = self.host_weights(layer_id, future)
                prefetch_slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=future,
                    source=source,
                    copy_stream=self.copy_stream,
                )
                self.host_prepare_seconds += time.perf_counter() - host_started
                self.fetches += 1
                self.h2d_bytes += prefetch_slot.bytes
                staged[future] = prefetch_slot

            with torch.cuda.stream(self.compute_stream):
                if current_slot is not None:
                    current_slot.wait_until_ready(self.compute_stream)
                routed = routed_tokens[expert_id]
                token_indices = routed.token_indices
                expert_input = hidden_states.index_select(0, token_indices)
                expert_output = self._mlp(expert_input, current_weights)
                expert_output = expert_output * routed.routing_weights[:, None]
                output.index_add_(0, token_indices, expert_output)
                if current_slot is not None:
                    current_slot.record_compute_done(self.compute_stream)
                elif resident_expert.slot is not None:
                    resident_expert.slot.record_compute_done(self.compute_stream)
            self.expert_executions += 1
            if current_slot is None:
                self.on_resident_hit(layer_id, expert_id)
            else:
                self.on_transient_complete(
                    layer_id, expert_id, current_slot, self.double_buffer
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
            "prefetch_depth": 1,
            "transient_slots": 2,
            "host_prepare_seconds": self.host_prepare_seconds,
        }
