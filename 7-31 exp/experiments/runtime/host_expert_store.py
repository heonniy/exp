from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def pack_projection_tensors(
    tensors: dict[str, torch.Tensor], *, pin_memory: bool
) -> torch.Tensor:
    if set(tensors) != set(PROJECTIONS):
        raise ValueError("Expert projection set is incomplete")
    dtypes = {tensor.dtype for tensor in tensors.values()}
    if len(dtypes) != 1:
        raise ValueError("Expert projections must have a common dtype")
    if not pin_memory:
        return torch.cat(
            [tensors[projection].reshape(-1) for projection in PROJECTIONS]
        )
    total_elements = sum(tensor.numel() for tensor in tensors.values())
    dtype = next(iter(dtypes))
    packed = torch.empty(total_elements, dtype=dtype, pin_memory=True)
    copy_projection_tensors_into(tensors, packed)
    return packed


def copy_projection_tensors_into(
    tensors: dict[str, torch.Tensor], destination: torch.Tensor
) -> None:
    if destination.ndim != 1 or not destination.is_contiguous():
        raise ValueError("packed Expert destination must be flat and contiguous")
    if destination.numel() != sum(tensor.numel() for tensor in tensors.values()):
        raise ValueError("packed Expert destination has the wrong size")
    offset = 0
    for projection in PROJECTIONS:
        source = tensors[projection].reshape(-1)
        destination[offset : offset + source.numel()].copy_(source)
        offset += source.numel()
    if offset != destination.numel():
        raise AssertionError("projection copies do not cover packed Expert")


class PinnedExpertStore:
    """Lazy safetensors-backed CPU Expert store with a bounded Expert LRU.

    In pinned-weight mode, each layer owns one contiguous pinned slab and each
    Expert is a flat row view. In staging mode, fixed pinned buffers belong to
    the transient GPU slots and the store retains CPU projection tensors.
    """

    def __init__(
        self,
        model_path: str | Path,
        max_pinned_experts: int = 16,
        pin_weights: bool = False,
    ):
        if max_pinned_experts < 2:
            raise ValueError("at least two pinned host Experts are required")
        self.model_path = Path(model_path)
        with (self.model_path / "model.safetensors.index.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._archives = {}
        self._cache: OrderedDict[
            tuple[int, int], dict[str, torch.Tensor] | torch.Tensor
        ] = (
            OrderedDict()
        )
        self.max_pinned_experts = max_pinned_experts
        self.pin_weights = pin_weights
        self.stage_calls = 0
        self.stage_cache_hits = 0
        self.stage_seconds = 0.0
        self._layer_slabs: list[torch.Tensor] = []

    def _archive(self, shard: str):
        archive = self._archives.get(shard)
        if archive is None:
            archive = safe_open(
                self.model_path / shard, framework="pt", device="cpu"
            )
            self._archives[shard] = archive
        return archive

    @staticmethod
    def tensor_name(layer_id: int, expert_id: int, projection: str) -> str:
        return (
            f"model.layers.{layer_id}.mlp.experts.{expert_id}."
            f"{projection}.weight"
        )

    def get(
        self, layer_id: int, expert_id: int
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        self.stage_calls += 1
        key = (layer_id, expert_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self.stage_cache_hits += 1
            self._cache[key] = cached
            return cached

        started = time.perf_counter()
        tensors = self._load_projection_tensors(layer_id, expert_id)
        value: dict[str, torch.Tensor] | torch.Tensor
        if self.pin_weights:
            # Lazy fallback. Full experiments call preload_all(), which uses one
            # large pinned slab per layer instead of one allocation per Expert.
            value = pack_projection_tensors(tensors, pin_memory=True)
        else:
            value = tensors
        self.stage_seconds += time.perf_counter() - started
        self._cache[key] = value
        while len(self._cache) > self.max_pinned_experts:
            self._cache.popitem(last=False)
        return value

    def _load_projection_tensors(
        self, layer_id: int, expert_id: int
    ) -> dict[str, torch.Tensor]:
        tensors = {}
        for projection in PROJECTIONS:
            name = self.tensor_name(layer_id, expert_id, projection)
            try:
                shard = self.weight_map[name]
            except KeyError as error:
                raise KeyError(f"checkpoint is missing {name}") from error
            tensor = self._archive(shard).get_tensor(name)
            tensors[projection] = tensor
        return tensors

    def preload_all(self, num_layers: int, num_experts: int) -> float:
        required = num_layers * num_experts
        if not self.pin_weights:
            raise ValueError("preload_all requires pinned_weights mode")
        if self.max_pinned_experts < required:
            raise ValueError(
                f"host cache holds {self.max_pinned_experts} Experts; need {required}"
            )
        if self._cache:
            raise RuntimeError("preload_all must run before lazy Expert staging")
        started = time.perf_counter()
        for layer_id in range(num_layers):
            first = self._load_projection_tensors(layer_id, 0)
            expert_elements = sum(tensor.numel() for tensor in first.values())
            dtype = next(iter(first.values())).dtype
            slab = torch.empty(
                (num_experts, expert_elements),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
            self._layer_slabs.append(slab)
            for expert_id in range(num_experts):
                tensors = (
                    first
                    if expert_id == 0
                    else self._load_projection_tensors(layer_id, expert_id)
                )
                copy_projection_tensors_into(tensors, slab[expert_id])
                self._cache[(layer_id, expert_id)] = slab[expert_id]
                self.stage_calls += 1
            print(
                f"host-pinned Experts {len(self._cache)}/{required}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        self.stage_seconds += elapsed
        return elapsed

    def metrics(self) -> dict:
        return {
            "host_stage_calls": self.stage_calls,
            "host_stage_cache_hits": self.stage_cache_hits,
            "host_stage_cache_hit_rate": (
                self.stage_cache_hits / self.stage_calls if self.stage_calls else 0.0
            ),
            "host_stage_seconds": self.stage_seconds,
            "max_pinned_experts": self.max_pinned_experts,
            "current_pinned_experts": len(self._cache),
            "host_memory_mode": (
                "pinned_weights" if self.pin_weights else "two_slot_pinned_staging"
            ),
            "host_expert_layout": (
                "single_contiguous_pinned_tensor"
                if self.pin_weights
                else "projection_tensors_packed_into_single_slot_staging"
            ),
            "host_pinned_allocation_granularity": (
                "one_layer_slab" if self._layer_slabs else "one_expert_or_slot"
            ),
        }
