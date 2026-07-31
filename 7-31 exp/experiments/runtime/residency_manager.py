from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

import torch

from experiments.runtime.expert_slot import ExpertSlot, ResidentExpert
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.transient_double_buffer import TransientDoubleBuffer


class RuntimeResidencyManager:
    name: str
    k: int

    def lookup(self, layer_id: int, expert_id: int) -> ResidentExpert | None:
        raise NotImplementedError

    def on_resident_hit(self, layer_id: int, expert_id: int) -> None:
        pass

    def on_transient_complete(
        self,
        layer_id: int,
        expert_id: int,
        slot: ExpertSlot,
        double_buffer: TransientDoubleBuffer,
    ) -> None:
        pass

    def metrics(self) -> dict:
        return {}


class StreamingRuntimeManager(RuntimeResidencyManager):
    name = "stream2"
    k = 0

    def lookup(self, layer_id: int, expert_id: int) -> None:
        return None


def _preload_slot(
    layer_id: int,
    expert_id: int,
    tensor_shapes: dict[str, tuple[int, ...]],
    dtype: torch.dtype,
    host_store: PinnedExpertStore,
    copy_stream: torch.cuda.Stream,
) -> ExpertSlot:
    slot = ExpertSlot(-1, tensor_shapes, dtype=dtype, device="cuda:0")
    slot.enqueue_copy(
        layer_id=layer_id,
        expert_id=expert_id,
        source=host_store.get(layer_id, expert_id),
        copy_stream=copy_stream,
    )
    slot.copy_done.synchronize()
    slot.release_copy_source()
    slot.release_host_staging()
    return slot


class PermanentRuntimeManager(RuntimeResidencyManager):
    name = "permanent_k"

    def __init__(
        self,
        selections: Iterable[Iterable[int]],
        tensor_shapes: dict[str, tuple[int, ...]],
        dtype: torch.dtype,
        host_store: PinnedExpertStore,
        name: str = "permanent_k",
    ):
        if name not in {"permanent_k", "full_resident"}:
            raise ValueError(f"invalid permanent residency name: {name}")
        self.name = name
        selected = [tuple(int(value) for value in layer) for layer in selections]
        self.k = len(selected[0]) if selected else 0
        if any(len(layer) != self.k for layer in selected):
            raise ValueError("permanent selections must have a uniform layer quota")
        self._residents: dict[tuple[int, int], ResidentExpert] = {}
        stream = torch.cuda.Stream(device=0)
        for layer_id, experts in enumerate(selected):
            for expert_id in experts:
                slot = _preload_slot(
                    layer_id,
                    expert_id,
                    tensor_shapes,
                    dtype,
                    host_store,
                    stream,
                )
                self._residents[(layer_id, expert_id)] = ResidentExpert(
                    layer_id, expert_id, slot.tensors, slot
                )
        self.hits = 0

    def lookup(self, layer_id: int, expert_id: int) -> ResidentExpert | None:
        return self._residents.get((layer_id, expert_id))

    def on_resident_hit(self, layer_id: int, expert_id: int) -> None:
        self.hits += 1

    def metrics(self) -> dict:
        return {
            "permanent_resident_experts": len(self._residents),
            "permanent_hits": self.hits,
            "pinned_evictions": 0,
            "non_pinned_admissions": 0,
        }


class QuotaLRURuntimeManager(RuntimeResidencyManager):
    name = "quota_lru_k"

    def __init__(
        self,
        num_layers: int,
        k: int,
        tensor_shapes: dict[str, tuple[int, ...]],
        dtype: torch.dtype,
    ):
        self.k = k
        self._lru: list[OrderedDict[int, ExpertSlot]] = [
            OrderedDict() for _ in range(num_layers)
        ]
        self._empty: list[list[ExpertSlot]] = [
            [
                ExpertSlot(
                    slot_id=2 + layer_id * max(1, k) + index,
                    tensor_shapes=tensor_shapes,
                    dtype=dtype,
                    device="cuda:0",
                )
                for index in range(k)
            ]
            for layer_id in range(num_layers)
        ]
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.logical_swaps = 0

    def lookup(self, layer_id: int, expert_id: int) -> ResidentExpert | None:
        slot = self._lru[layer_id].get(expert_id)
        if slot is None:
            return None
        return ResidentExpert(layer_id, expert_id, slot.tensors, slot)

    def on_resident_hit(self, layer_id: int, expert_id: int) -> None:
        layer = self._lru[layer_id]
        slot = layer.pop(expert_id)
        layer[expert_id] = slot
        self.hits += 1

    def on_transient_complete(
        self,
        layer_id: int,
        expert_id: int,
        slot: ExpertSlot,
        double_buffer: TransientDoubleBuffer,
    ) -> None:
        self.misses += 1
        if self.k == 0:
            return
        slot.release_host_staging()
        layer = self._lru[layer_id]
        if self._empty[layer_id]:
            replacement = self._empty[layer_id].pop()
        else:
            _, replacement = layer.popitem(last=False)
            self.evictions += 1
        replacement.ensure_host_staging()
        double_buffer.replace(slot, replacement)
        layer[expert_id] = slot
        self.logical_swaps += 1
        if len(layer) > self.k:
            raise AssertionError("layer-local quota exceeded")

    def metrics(self) -> dict:
        counts = [len(layer) for layer in self._lru]
        return {
            "quota_hits": self.hits,
            "quota_misses": self.misses,
            "quota_evictions": self.evictions,
            "logical_ownership_swaps": self.logical_swaps,
            "d2d_admission_copies": 0,
            "resident_count_max": max(counts, default=0),
            "resident_count_by_layer": counts,
        }
