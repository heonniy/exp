from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import torch
import torch.nn.functional as functional

from experiments.runtime.expert_slot import ExpertSlot, ResidentExpert
from experiments.runtime.prefetch_scheduler import (
    first_future_nonresident,
    ordered_active_experts,
)
from experiments.runtime.transient_double_buffer import TransientDoubleBuffer


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
    ):
        self.double_buffer = double_buffer
        self.host_weights = host_weights
        self.resident_lookup = resident_lookup
        self.compute_stream = torch.cuda.Stream(device=0)
        self.copy_stream = torch.cuda.Stream(device=0)

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
        routed_tokens: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        active = ordered_active_experts(routed_tokens)
        output = torch.zeros_like(hidden_states)
        if not active:
            return output

        def resident(expert_id: int) -> bool:
            return self.resident_lookup(layer_id, expert_id) is not None

        staged: dict[int, ExpertSlot] = {}
        next_slot = self.double_buffer.slots[0]
        first = first(
            (expert_id for expert_id in active if not resident(expert_id)),
            None,
        )
        if first is not None:
            next_slot.enqueue_copy(
                layer_id=layer_id,
                expert_id=first,
                source=self.host_weights(layer_id, first),
                copy_stream=self.copy_stream,
            )
            staged[first] = next_slot

        for position, expert_id in enumerate(active):
            resident_expert = self.resident_lookup(layer_id, expert_id)
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
                prefetch_slot.enqueue_copy(
                    layer_id=layer_id,
                    expert_id=future,
                    source=self.host_weights(layer_id, future),
                    copy_stream=self.copy_stream,
                )
                staged[future] = prefetch_slot

            with torch.cuda.stream(self.compute_stream):
                if current_slot is not None:
                    current_slot.wait_until_ready(self.compute_stream)
                token_indices = routed_tokens[expert_id]
                expert_input = hidden_states.index_select(0, token_indices)
                expert_output = self._mlp(expert_input, current_weights)
                output.index_add_(0, token_indices, expert_output)
                if current_slot is not None:
                    current_slot.record_compute_done(self.compute_stream)
            next_slot = (
                self.double_buffer.other(current_slot)
                if current_slot is not None
                else next_slot
            )

        self.compute_stream.synchronize()
        return output
