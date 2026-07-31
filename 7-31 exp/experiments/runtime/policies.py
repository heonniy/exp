from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True)
class AccessResult:
    hit: bool
    hit_kind: str | None = None
    admitted: bool = False
    evicted_expert: int | None = None
    resident_lifetime: int | None = None
    bypassed: bool = False


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

    def begin_layer_step(
        self, layer_id: int, active_experts: tuple[int, ...]
    ) -> None:
        """Observe a batch layer-step before its Expert execution order is set."""

    def order_active_experts(
        self, layer_id: int, active_experts: tuple[int, ...]
    ) -> tuple[int, ...]:
        return tuple(sorted(active_experts))

    def simulation_metadata(self) -> dict[str, object]:
        return {
            "access_order": "ascending_expert_id",
            "admission_policy": "not_applicable",
            "random_seed": None,
            "window_size": None,
            "window_min_frequency": None,
        }


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
        name: str = "permanent_k",
    ):
        super().__init__(num_layers, num_experts, k)
        if name not in {"permanent_k", "permanent_oracle"}:
            raise ValueError(f"invalid permanent policy name: {name}")
        self.name = name
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

    VALID_ACCESS_ORDERS = {
        "ascending_expert_id",
        "resident_hit_first",
        "router_order",
        "random_expert_order",
    }
    VALID_ADMISSION_POLICIES = {
        "always_admit",
        "miss_bypass_when_full",
        "no_admission",
        "window_frequency",
    }

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        k: int,
        *,
        access_order: str = "ascending_expert_id",
        admission_policy: str = "always_admit",
        random_seed: int = 731,
        window_size: int = 8,
        window_min_frequency: int = 2,
        name: str | None = None,
    ):
        super().__init__(num_layers, num_experts, k)
        if access_order not in self.VALID_ACCESS_ORDERS:
            raise ValueError(f"invalid Quota-LRU access order: {access_order}")
        if admission_policy not in self.VALID_ADMISSION_POLICIES:
            raise ValueError(
                f"invalid Quota-LRU admission policy: {admission_policy}"
            )
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 1 <= window_min_frequency <= window_size:
            raise ValueError("window_min_frequency must be in [1, window_size]")
        self.access_order = access_order
        self.admission_policy = admission_policy
        self.random_seed = random_seed
        self.window_size = window_size
        self.window_min_frequency = window_min_frequency
        self.name = name or "quota_lru_k"
        self._rng = random.Random(random_seed)
        self._layers: list[OrderedDict[int, int]] = [
            OrderedDict() for _ in range(num_layers)
        ]
        self._windows: list[deque[frozenset[int]]] = [
            deque() for _ in range(num_layers)
        ]
        self._window_counts: list[Counter[int]] = [
            Counter() for _ in range(num_layers)
        ]

    def begin_layer_step(
        self, layer_id: int, active_experts: tuple[int, ...]
    ) -> None:
        if self.admission_policy != "window_frequency":
            return
        window = self._windows[layer_id]
        counts = self._window_counts[layer_id]
        current = frozenset(active_experts)
        window.append(current)
        counts.update(current)
        if len(window) > self.window_size:
            expired = window.popleft()
            counts.subtract(expired)
            for expert_id in expired:
                if counts[expert_id] == 0:
                    del counts[expert_id]

    def order_active_experts(
        self, layer_id: int, active_experts: tuple[int, ...]
    ) -> tuple[int, ...]:
        ordered = tuple(sorted(active_experts))
        if self.access_order == "ascending_expert_id":
            return ordered
        if self.access_order == "resident_hit_first":
            residents = self._layers[layer_id]
            return tuple(
                sorted(
                    ordered,
                    key=lambda expert_id: (
                        expert_id not in residents,
                        expert_id,
                    ),
                )
            )
        if self.access_order == "router_order":
            return tuple(active_experts)
        shuffled = list(ordered)
        self._rng.shuffle(shuffled)
        return tuple(shuffled)

    def _should_admit(self, layer_id: int, expert_id: int) -> bool:
        if self.admission_policy == "always_admit":
            return True
        if self.admission_policy == "no_admission":
            return False
        if self.admission_policy == "miss_bypass_when_full":
            return len(self._layers[layer_id]) < self.k
        return (
            self._window_counts[layer_id].get(expert_id, 0)
            >= self.window_min_frequency
        )

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        cache = self._layers[layer_id]
        if expert_id in cache:
            admitted_at = cache.pop(expert_id)
            cache[expert_id] = admitted_at
            return AccessResult(hit=True, hit_kind="local_lru")
        if self.k == 0:
            return AccessResult(hit=False)
        if not self._should_admit(layer_id, expert_id):
            return AccessResult(hit=False, bypassed=True)

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
        for window in self._windows:
            window.clear()
        for counts in self._window_counts:
            counts.clear()
        self._rng = random.Random(self.random_seed)

    def simulation_metadata(self) -> dict[str, object]:
        return {
            "access_order": self.access_order,
            "admission_policy": self.admission_policy,
            "random_seed": (
                self.random_seed
                if self.access_order == "random_expert_order"
                else None
            ),
            "window_size": (
                self.window_size
                if self.admission_policy == "window_frequency"
                else None
            ),
            "window_min_frequency": (
                self.window_min_frequency
                if self.admission_policy == "window_frequency"
                else None
            ),
        }


class FullResidentPolicy(ExpertPolicy):
    name = "full_resident"

    def __init__(self, num_layers: int, num_experts: int):
        super().__init__(num_layers, num_experts, k=num_experts)

    def access(self, layer_id: int, expert_id: int, tick: int) -> AccessResult:
        self._validate(layer_id, expert_id)
        return AccessResult(hit=True, hit_kind="full_resident")

    def resident_counts(self) -> tuple[int, ...]:
        return (self.num_experts,) * self.num_layers
