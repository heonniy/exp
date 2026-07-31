from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AccessResult:
    hit: bool
    hit_kind: str | None = None
    admitted: bool = False
    evicted_expert: int | None = None
    resident_lifetime: int | None = None


class ExpertPolicy(ABC):
    name: str

    def __init__(self, num_layers: int, num_experts: int, k: int):
        if not 0 <= k <= num_experts:
            raise ValueError(f"k={k} is outside [0, {num_experts}]")
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k = k

    def _validate(self, layer_id: int, expert_id: int) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(f"invalid layer ID: {layer_id}")
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(f"invalid Expert ID: {expert_id}")

    @abstractmethod
    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        raise NotImplementedError

    @abstractmethod
    def resident_counts(self) -> tuple[int, ...]:
        raise NotImplementedError

    def reset_dynamic_state(self) -> None:
        """Reset wave-local adaptive state. Static policies need no action."""


class Stream2Policy(ExpertPolicy):
    name = "stream2"

    def __init__(self, num_layers: int, num_experts: int):
        super().__init__(num_layers, num_experts, k=0)

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        return AccessResult(hit=False)

    def resident_counts(self) -> tuple[int, ...]:
        return (0,) * self.num_layers


class PermanentPolicy(ExpertPolicy):
    name = "permanent_k"

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        k: int,
        permanent_experts: Iterable[Iterable[int]],
    ):
        super().__init__(num_layers, num_experts, k)
        layers = tuple(frozenset(int(expert) for expert in values) for values in permanent_experts)
        if len(layers) != num_layers:
            raise ValueError("permanent selections must cover every layer")
        for layer_id, selected in enumerate(layers):
            if len(selected) != k:
                raise ValueError(
                    f"layer {layer_id} has {len(selected)} permanent Experts; expected {k}"
                )
            for expert_id in selected:
                self._validate(layer_id, expert_id)
        self.permanent_experts = layers

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        if expert_id in self.permanent_experts[layer_id]:
            return AccessResult(hit=True, hit_kind="permanent")
        return AccessResult(hit=False)

    def resident_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.permanent_experts)


class QuotaLRUPolicy(ExpertPolicy):
    name = "quota_lru_k"

    def __init__(self, num_layers: int, num_experts: int, k: int):
        super().__init__(num_layers, num_experts, k)
        self._layers: list[OrderedDict[int, int]] = [
            OrderedDict() for _ in range(num_layers)
        ]

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        cache = self._layers[layer_id]
        if expert_id in cache:
            admitted_at = cache.pop(expert_id)
            cache[expert_id] = admitted_at
            return AccessResult(hit=True, hit_kind="local_lru")
        if self.k == 0:
            return AccessResult(hit=False)

        evicted = None
        lifetime = None
        if len(cache) == self.k:
            evicted, admitted_at = cache.popitem(last=False)
            lifetime = tick - admitted_at
        # Logical ownership swap: no D2D copy is represented or required.
        cache[expert_id] = tick
        return AccessResult(
            hit=False,
            admitted=True,
            evicted_expert=evicted,
            resident_lifetime=lifetime,
        )

    def resident_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self._layers)

    def reset_dynamic_state(self) -> None:
        for cache in self._layers:
            cache.clear()


class FullResidentPolicy(ExpertPolicy):
    name = "full_resident"

    def __init__(self, num_layers: int, num_experts: int):
        super().__init__(num_layers, num_experts, k=num_experts)

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        return AccessResult(hit=True, hit_kind="full_resident")

    def resident_counts(self) -> tuple[int, ...]:
        return (self.num_experts,) * self.num_layers

