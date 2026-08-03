from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    dtype: str
    num_moe_layers: int
    num_experts_per_layer: int
    router_top_k: int


@dataclass(frozen=True)
class DatasetConfig:
    parquet_glob: str
    language: str
    calibration_requests: int
    evaluation_requests: int
    input_tokens: int
    output_tokens: int
    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    transient_expert_slots: int
    hbm_safety_margin_gib: float
    effective_hbm_gib: float | None
    expert_execution_order: str
    global_lru: bool


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    gpu_physical_index: int
    model: ModelConfig
    dataset: DatasetConfig
    runtime: RuntimeConfig
    trace_k: tuple[int, ...]
    runtime_k: tuple[int, ...]
    policies: tuple[str, ...]

    @property
    def peak_sequence_length(self) -> int:
        return self.dataset.input_tokens + self.dataset.output_tokens


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    exp = _required(raw, "experiment")
    model = _required(raw, "model")
    dataset = _required(raw, "dataset")
    runtime = _required(raw, "runtime")
    sweep = _required(raw, "sweep")

    if bool(runtime.get("global_lru", False)):
        raise ValueError("global_lru must remain false for this experiment")
    if int(runtime.get("prefetch_depth", -1)) != 1:
        raise ValueError("the primary experiment requires prefetch_depth=1")
    if int(runtime.get("transient_expert_slots", -1)) != 2:
        raise ValueError("the primary experiment requires exactly two transient slots")
    if runtime.get("expert_execution_order") != "ascending_expert_id":
        raise ValueError("expert execution order must be ascending_expert_id")

    parsed = ExperimentConfig(
        name=str(_required(exp, "name")),
        seed=int(_required(exp, "seed")),
        gpu_physical_index=int(_required(exp, "gpu_physical_index")),
        model=ModelConfig(
            path=Path(_required(model, "path")),
            dtype=str(_required(model, "dtype")),
            num_moe_layers=int(_required(model, "num_moe_layers")),
            num_experts_per_layer=int(_required(model, "num_experts_per_layer")),
            router_top_k=int(_required(model, "router_top_k")),
        ),
        dataset=DatasetConfig(
            parquet_glob=str(_required(dataset, "parquet_glob")),
            language=str(_required(dataset, "language")),
            calibration_requests=int(_required(dataset, "calibration_requests")),
            evaluation_requests=int(_required(dataset, "evaluation_requests")),
            input_tokens=int(_required(dataset, "input_tokens")),
            output_tokens=int(_required(dataset, "output_tokens")),
            output_dir=Path(_required(dataset, "output_dir")),
        ),
        runtime=RuntimeConfig(
            transient_expert_slots=int(_required(runtime, "transient_expert_slots")),
            hbm_safety_margin_gib=float(_required(runtime, "hbm_safety_margin_gib")),
            effective_hbm_gib=(
                float(runtime["effective_hbm_gib"])
                if runtime.get("effective_hbm_gib") is not None
                else None
            ),
            expert_execution_order=str(_required(runtime, "expert_execution_order")),
            global_lru=bool(_required(runtime, "global_lru")),
        ),
        trace_k=tuple(int(value) for value in _required(sweep, "trace_k")),
        runtime_k=tuple(int(value) for value in _required(sweep, "runtime_k")),
        policies=tuple(str(value) for value in _required(sweep, "policies")),
    )
    validate_config(parsed)
    return parsed


def validate_config(config: ExperimentConfig) -> None:
    if config.gpu_physical_index != 0:
        raise ValueError("this workspace is restricted to physical GPU 0")
    if config.model.router_top_k <= 0:
        raise ValueError("router_top_k must be positive")
    if not 0 < config.model.num_experts_per_layer <= 256:
        raise ValueError("num_experts_per_layer must fit the uint8 trace representation")
    for value in (*config.trace_k, *config.runtime_k):
        if not 0 <= value <= config.model.num_experts_per_layer:
            raise ValueError(f"invalid resident Expert count: {value}")
    if config.dataset.input_tokens <= 0 or config.dataset.output_tokens <= 0:
        raise ValueError("fixed input/output lengths must be positive")
    if (
        config.runtime.effective_hbm_gib is not None
        and config.runtime.effective_hbm_gib <= 0
    ):
        raise ValueError("effective_hbm_gib must be positive when configured")
    valid_policies = {"stream2", "permanent_k", "quota_lru_k", "full_resident"}
    invalid_policies = sorted(set(config.policies) - valid_policies)
    if invalid_policies:
        raise ValueError(f"invalid runtime policies: {invalid_policies}")
