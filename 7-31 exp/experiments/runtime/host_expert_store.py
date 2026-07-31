from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class PinnedExpertStore:
    """Lazy safetensors-backed CPU Expert store with bounded tensor-view LRU.

    Fixed pinned staging buffers belong to the two transient GPU slots. This
    store retains cheap CPU/mmap tensor views and never pins an Expert per miss.
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
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self.max_pinned_experts = max_pinned_experts
        self.pin_weights = pin_weights
        self.stage_calls = 0
        self.stage_cache_hits = 0
        self.stage_seconds = 0.0

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

    def get(self, layer_id: int, expert_id: int) -> dict[str, torch.Tensor]:
        self.stage_calls += 1
        key = (layer_id, expert_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self.stage_cache_hits += 1
            self._cache[key] = cached
            return cached

        started = time.perf_counter()
        tensors = {}
        for projection in PROJECTIONS:
            name = self.tensor_name(layer_id, expert_id, projection)
            try:
                shard = self.weight_map[name]
            except KeyError as error:
                raise KeyError(f"checkpoint is missing {name}") from error
            tensor = self._archive(shard).get_tensor(name)
            tensors[projection] = tensor.pin_memory() if self.pin_weights else tensor
        self.stage_seconds += time.perf_counter() - started
        self._cache[key] = tensors
        while len(self._cache) > self.max_pinned_experts:
            self._cache.popitem(last=False)
        return tensors

    def preload_all(self, num_layers: int, num_experts: int) -> float:
        required = num_layers * num_experts
        if not self.pin_weights:
            raise ValueError("preload_all requires pinned_weights mode")
        if self.max_pinned_experts < required:
            raise ValueError(
                f"host cache holds {self.max_pinned_experts} Experts; need {required}"
            )
        started = time.perf_counter()
        for layer_id in range(num_layers):
            for expert_id in range(num_experts):
                self.get(layer_id, expert_id)
            print(
                f"host-pinned Experts {len(self._cache)}/{required}",
                flush=True,
            )
        return time.perf_counter() - started

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
        }
