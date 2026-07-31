from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from experiments.runtime.policies import ExpertPolicy
from experiments.trace.trace_schema import RoutingTrace


@dataclass
class LayerStats:
    layer_id: int
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    fetches: int = 0
    compulsory_loads: int = 0
    refetches: int = 0
    evictions: int = 0
    admissions: int = 0
    bypasses: int = 0


@dataclass
class SimulationResult:
    policy: str
    k: int
    batch_size: int
    requests: int
    waves: int
    generated_tokens: int
    expert_token_assignments: int
    expert_executions: int
    hits: int
    misses: int
    fetches: int
    compulsory_loads: int
    refetches: int
    evictions: int
    admissions: int
    bypasses: int
    h2d_bytes: int
    h2d_bytes_per_generated_token: float
    hit_rate: float
    refetch_ratio: float
    tokens_per_expert_fetch: float | None
    average_evicted_resident_lifetime: float | None
    trace_sha256: str
    retain_state_across_waves: bool
    access_order: str
    admission_policy: str
    random_seed: int | None
    window_size: int | None
    window_min_frequency: int | None
    per_layer: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def simulate(
    trace: RoutingTrace,
    policy: ExpertPolicy,
    expert_bytes: int,
    batch_size: int,
    retain_state_across_waves: bool = True,
) -> SimulationResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")
    if policy.num_layers != trace.num_layers:
        raise ValueError("policy and trace layer counts do not match")

    per_layer = [LayerStats(layer_id=i) for i in range(trace.num_layers)]
    seen: set[tuple[int, int]] = set()
    hits = misses = fetches = compulsory = refetches = evictions = executions = 0
    admissions = bypasses = 0
    lifetimes: list[int] = []
    tick = 0

    for wave_index, start in enumerate(range(0, trace.num_requests, batch_size)):
        if wave_index and not retain_state_across_waves:
            policy.reset_dynamic_state()
        stop = min(start + batch_size, trace.num_requests)
        for step in range(trace.output_tokens):
            for layer_id in range(trace.num_layers):
                active = tuple(
                    int(value)
                    for value in np.unique(
                        trace.routing_expert_ids[start:stop, step, layer_id, :]
                    )
                )
                policy.begin_layer_step(layer_id, active)
                ordered = policy.order_active_experts(layer_id, active)
                if len(ordered) != len(active) or set(ordered) != set(active):
                    raise AssertionError(
                        "policy execution order must preserve the active Expert set"
                    )
                for expert_id in ordered:
                    result = policy.access(layer_id, expert_id, tick)
                    tick += 1
                    executions += 1
                    layer = per_layer[layer_id]
                    layer.accesses += 1
                    key = (layer_id, expert_id)
                    if result.hit:
                        hits += 1
                        layer.hits += 1
                    else:
                        misses += 1
                        fetches += 1
                        layer.misses += 1
                        layer.fetches += 1
                        if key in seen:
                            refetches += 1
                            layer.refetches += 1
                        else:
                            compulsory += 1
                            layer.compulsory_loads += 1
                            seen.add(key)
                    if result.evicted_expert is not None:
                        evictions += 1
                        layer.evictions += 1
                    if result.resident_lifetime is not None:
                        lifetimes.append(result.resident_lifetime)
                    if result.admitted:
                        admissions += 1
                        layer.admissions += 1
                    if result.bypassed:
                        bypasses += 1
                        layer.bypasses += 1

                resident = policy.resident_counts()
                if any(count > policy.k for count in resident):
                    raise AssertionError("layer-local residency quota exceeded")

    assignments = int(trace.routing_expert_ids.size)
    generated = trace.num_requests * trace.output_tokens
    metadata = policy.simulation_metadata()
    return SimulationResult(
        policy=policy.name,
        k=policy.k,
        batch_size=batch_size,
        requests=trace.num_requests,
        waves=ceil(trace.num_requests / batch_size),
        generated_tokens=generated,
        expert_token_assignments=assignments,
        expert_executions=executions,
        hits=hits,
        misses=misses,
        fetches=fetches,
        compulsory_loads=compulsory,
        refetches=refetches,
        evictions=evictions,
        admissions=admissions,
        bypasses=bypasses,
        h2d_bytes=fetches * expert_bytes,
        h2d_bytes_per_generated_token=(fetches * expert_bytes / generated),
        hit_rate=(hits / executions if executions else 0.0),
        refetch_ratio=(refetches / fetches if fetches else 0.0),
        tokens_per_expert_fetch=(assignments / fetches if fetches else None),
        average_evicted_resident_lifetime=(
            float(np.mean(lifetimes)) if lifetimes else None
        ),
        trace_sha256=trace.digest(),
        retain_state_across_waves=retain_state_across_waves,
        access_order=str(metadata["access_order"]),
        admission_policy=str(metadata["admission_policy"]),
        random_seed=(
            int(metadata["random_seed"])
            if metadata["random_seed"] is not None
            else None
        ),
        window_size=(
            int(metadata["window_size"])
            if metadata["window_size"] is not None
            else None
        ),
        window_min_frequency=(
            int(metadata["window_min_frequency"])
            if metadata["window_min_frequency"] is not None
            else None
        ),
        per_layer=[asdict(layer) for layer in per_layer],
    )
