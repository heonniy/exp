from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.residency_manager import RuntimeResidencyManager
from experiments.runtime.serial_expert_executor import (
    RoutedExpertTokens,
    SerialExpertExecutor,
)
from experiments.runtime.transient_double_buffer import TransientDoubleBuffer


EXPERT_SHAPES = {
    "gate_proj": (768, 2048),
    "up_proj": (768, 2048),
    "down_proj": (2048, 768),
}


def build_routed_tokens(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor
) -> dict[int, RoutedExpertTokens]:
    routed = {}
    for expert_id in sorted(int(value) for value in torch.unique(top_k_index).tolist()):
        token_index, top_k_position = torch.where(top_k_index == expert_id)
        routed[expert_id] = RoutedExpertTokens(
            token_indices=token_index,
            routing_weights=top_k_weights[token_index, top_k_position],
        )
    return routed


class OffloadedExpertEngine:
    def __init__(
        self,
        host_store: PinnedExpertStore,
        manager: RuntimeResidencyManager,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.host_store = host_store
        self.manager = manager
        self.double_buffer = TransientDoubleBuffer(
            EXPERT_SHAPES, dtype=dtype, device="cuda:0"
        )
        self.executor = SerialExpertExecutor(
            self.double_buffer,
            self.host_store.get,
            self.manager.lookup,
            self.manager.on_resident_hit,
            self.manager.on_transient_complete,
        )
        self.capture_routes = False
        self.captured_routes: list[tuple[int, np.ndarray]] = []
        self.forced_routing: np.ndarray | None = None
        self.decode_step: int | None = None
        self.natural_route_assignments = 0
        self.natural_route_mismatches = 0

    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.forced_routing is not None:
            if self.decode_step is None:
                raise RuntimeError("forced routing is active without a decode step")
            forced = torch.as_tensor(
                self.forced_routing[:, self.decode_step, layer_id, :],
                dtype=top_k_index.dtype,
                device=top_k_index.device,
            )
            if forced.shape != top_k_index.shape:
                raise ValueError(
                    f"forced route shape {tuple(forced.shape)} does not match "
                    f"runtime shape {tuple(top_k_index.shape)}"
                )
            self.natural_route_assignments += forced.numel()
            self.natural_route_mismatches += int(
                (forced != top_k_index).count_nonzero().item()
            )
            top_k_index = forced
        if self.capture_routes:
            self.captured_routes.append(
                (
                    layer_id,
                    top_k_index.detach().to(dtype=torch.uint8, device="cpu").numpy(),
                )
            )
        # CPU synchronization is intentional: Expert execution order is a fixed
        # ascending ID sequence and no grouped GEMM is allowed in Phase A.
        routed = build_routed_tokens(top_k_index, top_k_weights)
        return self.executor.execute_layer(
            layer_id=layer_id,
            hidden_states=hidden_states,
            routed_tokens=routed,
        )

    def reset_captured_routes(self) -> None:
        self.captured_routes.clear()

    def set_forced_routing(self, routing: np.ndarray | None) -> None:
        if routing is not None and routing.ndim != 4:
            raise ValueError("forced routing must be [batch, token, layer, top_k]")
        self.forced_routing = routing
        self.decode_step = None
        self.natural_route_assignments = 0
        self.natural_route_mismatches = 0

    def metrics(self) -> dict:
        return {
            **self.executor.metrics(),
            **self.manager.metrics(),
            **self.host_store.metrics(),
            "global_lru": False,
            "expert_execution_order": "ascending_expert_id",
            "grouped_gemm": False,
            "batched_gemm": False,
            "forced_routing": self.forced_routing is not None,
            "natural_route_assignments": self.natural_route_assignments,
            "natural_route_mismatches": self.natural_route_mismatches,
            "natural_route_mismatch_rate": (
                self.natural_route_mismatches / self.natural_route_assignments
                if self.natural_route_assignments
                else 0.0
            ),
        }


class OffloadedQwenExperts(nn.Module):
    def __init__(self, layer_id: int, engine: OffloadedExpertEngine):
        super().__init__()
        self.layer_id = layer_id
        # Bypass Module.__setattr__ registration: the shared engine owns CUDA
        # streams/buffers but is not part of the model state dict.
        object.__setattr__(self, "_engine", engine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._engine.execute(
            self.layer_id, hidden_states, top_k_index, top_k_weights
        )


def _load_dense_parameters(
    model: nn.Module,
    model_path: Path,
    dtype: torch.dtype,
) -> None:
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map: dict[str, str] = index["weight_map"]
    expected = dict(model.named_parameters())
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in expected:
        if ".mlp.experts." in name:
            raise AssertionError("offloaded Expert parameter remained in dense model")
        try:
            by_shard[weight_map[name]].append(name)
        except KeyError as error:
            raise KeyError(f"dense checkpoint tensor not found: {name}") from error

    loaded = set()
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_path / shard, framework="pt", device="cpu") as archive:
            for name in names:
                value = archive.get_tensor(name)
                set_module_tensor_to_device(
                    model,
                    name,
                    device="cuda:0",
                    value=value,
                    dtype=dtype,
                    non_blocking=False,
                )
                loaded.add(name)
    missing = set(expected) - loaded
    if missing:
        raise RuntimeError(f"failed to load {len(missing)} dense parameters")
    meta = [name for name, value in model.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"dense model still has meta parameters: {meta[:5]}")


def load_offloaded_qwen(
    model_path: str | Path,
    engine: OffloadedExpertEngine,
    dtype: torch.dtype = torch.bfloat16,
):
    model_dir = Path(model_path)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.output_router_logits = False
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    layers = model.model.layers
    for layer_id, layer in enumerate(layers):
        layer.mlp.experts = OffloadedQwenExperts(layer_id, engine)
    _load_dense_parameters(model, model_dir, dtype)
    model.model.rotary_emb.to("cuda:0")
    model.eval()
    return model


def attach_engine(model, engine: OffloadedExpertEngine) -> None:
    for layer in model.model.layers:
        if not isinstance(layer.mlp.experts, OffloadedQwenExperts):
            raise TypeError("model does not contain OffloadedQwenExperts")
        object.__setattr__(layer.mlp.experts, "_engine", engine)
