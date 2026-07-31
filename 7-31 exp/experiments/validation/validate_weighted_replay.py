from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import OffloadedExpertEngine, load_offloaded_qwen
from experiments.runtime.residency_manager import StreamingRuntimeManager
from experiments.trace.trace_schema import RoutingTrace


def _read_row(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise ValueError(f"{path} does not contain row {index}")


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class ReferenceRoutingReplay:
    """Inject one recorded route into the full eager Expert implementation."""

    def __init__(self, ids: np.ndarray, weights: np.ndarray):
        self.ids = ids
        self.weights = weights
        self.decode_step: int | None = None
        self.natural_id_exact = np.zeros(ids.shape[1:3], dtype=np.bool_)
        self.natural_weight_exact = np.zeros(ids.shape[1:3], dtype=np.bool_)

    def force(
        self,
        layer_id: int,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decode_step is None:
            return top_k_index, top_k_weights
        recorded_ids = self.ids[:, self.decode_step, layer_id, :]
        recorded_weights = self.weights[:, self.decode_step, layer_id, :]
        natural_ids = top_k_index.detach().to(dtype=torch.uint8, device="cpu").numpy()
        natural_weights = (
            top_k_weights.detach().to(dtype=torch.float32, device="cpu").numpy()
        )
        self.natural_id_exact[self.decode_step, layer_id] = np.array_equal(
            natural_ids, recorded_ids
        )
        self.natural_weight_exact[self.decode_step, layer_id] = np.array_equal(
            natural_weights, recorded_weights
        )
        return (
            torch.as_tensor(
                recorded_ids,
                dtype=top_k_index.dtype,
                device=top_k_index.device,
            ),
            torch.as_tensor(
                recorded_weights,
                dtype=top_k_weights.dtype,
                device=top_k_weights.device,
            ),
        )


class ForcedReferenceExperts(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        layer_id: int,
        replay: ReferenceRoutingReplay,
    ):
        super().__init__()
        self.original = original
        self.layer_id = layer_id
        object.__setattr__(self, "_replay", replay)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        forced_ids, forced_weights = self._replay.force(
            self.layer_id, top_k_index, top_k_weights
        )
        return self.original(hidden_states, forced_ids, forced_weights)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-resident and offloaded logits using a real 4K prefill "
            "cache and exact schema-v2 Expert ID plus routing-weight replay."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    trace = RoutingTrace.load(args.trace)
    trace.validate(config.model.num_experts_per_layer, require_weights=True)
    trace.require_serial_reference()
    if not 0 <= args.request_index < trace.num_requests:
        raise ValueError("request index is outside the trace")
    if not 0 < args.decode_steps <= trace.output_tokens:
        raise ValueError("decode steps are outside the trace")
    row = _read_row(args.workload, args.request_index)
    if str(row["conversation_id"]) != str(trace.conversation_ids[args.request_index]):
        raise ValueError("trace and workload conversation IDs differ")
    forced = np.asarray(row["forced_output_ids"][: args.decode_steps], dtype=np.int32)
    if not np.array_equal(
        forced, trace.forced_output_ids[args.request_index, : args.decode_steps]
    ):
        raise ValueError("trace and workload forced tokens differ")

    dtype = getattr(torch, config.model.dtype)
    full_model = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        experts_implementation="eager",
    ).eval()
    replay_ids = trace.routing_expert_ids[
        args.request_index : args.request_index + 1, : args.decode_steps
    ]
    replay_weights = trace.require_routing_weights()[
        args.request_index : args.request_index + 1, : args.decode_steps
    ]
    reference_replay = ReferenceRoutingReplay(replay_ids, replay_weights)
    for layer_id, layer in enumerate(full_model.model.layers):
        layer.mlp.experts = ForcedReferenceExperts(
            layer.mlp.experts, layer_id, reference_replay
        )
    prompt = torch.as_tensor(
        row["input_ids"], dtype=torch.long, device="cuda:0"
    ).unsqueeze(0)
    prefill = full_model(
        input_ids=prompt,
        use_cache=True,
        output_router_logits=False,
        logits_to_keep=1,
        return_dict=True,
    )
    # The untouched copy is consumed by the offloaded decoder. It is a real
    # full-model 4K prefill cache, so static-zero KV cannot affect correctness.
    offloaded_past = copy.deepcopy(prefill.past_key_values)
    reference_past = prefill.past_key_values
    reference_logits: list[torch.Tensor] = []
    route_id_exact = []
    route_weight_exact = []
    for step in range(args.decode_steps):
        reference_replay.decode_step = step
        token = torch.as_tensor([[int(forced[step])]], device="cuda:0")
        output = full_model(
            input_ids=token,
            past_key_values=reference_past,
            use_cache=True,
            output_router_logits=False,
            logits_to_keep=1,
            return_dict=True,
        )
        reference_past = output.past_key_values
        route_id_exact.append(bool(reference_replay.natural_id_exact[step].all()))
        route_weight_exact.append(
            bool(reference_replay.natural_weight_exact[step].all())
        )
        reference_logits.append(output.logits.detach().to("cpu"))

    del output, prefill, reference_past, prompt, full_model
    torch.cuda.empty_cache()

    host_store = PinnedExpertStore(
        config.model.path, max_pinned_experts=16, pin_weights=False
    )
    engine = OffloadedExpertEngine(
        host_store,
        StreamingRuntimeManager(),
        prefetch_depth=1,
        track_timeline=False,
        prefetch_submit_order="compute_first",
    )
    offloaded_model = load_offloaded_qwen(config.model.path, engine).eval()
    engine.set_forced_routing(replay_ids, replay_weights)
    comparisons = []
    for step in range(args.decode_steps):
        engine.decode_step = step
        token = torch.as_tensor([[int(forced[step])]], device="cuda:0")
        output = offloaded_model(
            input_ids=token,
            past_key_values=offloaded_past,
            use_cache=True,
            output_router_logits=False,
            logits_to_keep=1,
            return_dict=True,
        )
        offloaded_past = output.past_key_values
        actual = output.logits.detach().to("cpu")
        expected = reference_logits[step]
        difference = (actual.float() - expected.float()).abs()
        comparisons.append(
            {
                "decode_step": step,
                "reference_logits_sha256": _digest(expected),
                "offloaded_logits_sha256": _digest(actual),
                "bitwise_equal": bool(torch.equal(actual, expected)),
                "allclose_atol_0p1_rtol_0p01": bool(
                    torch.allclose(actual, expected, atol=0.1, rtol=0.01)
                ),
                "max_absolute_error": float(difference.max().item()),
                "mean_absolute_error": float(difference.mean().item()),
                "argmax_equal": bool(
                    torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
                ),
            }
        )

    engine_metrics = engine.metrics()
    engine_metrics.pop("natural_route_mismatch_by_step_layer", None)
    correctness_gate_passed = all(row["bitwise_equal"] for row in comparisons)
    result = {
        "validation": "full_vs_offloaded_weighted_routing_replay",
        "gpu_physical_index": 0,
        "config": config.name,
        "request_index": args.request_index,
        "conversation_id": str(row["conversation_id"]),
        "input_tokens": len(row["input_ids"]),
        "decode_steps": args.decode_steps,
        "kv_setup": "real_full_model_prefill_cache",
        "reference_experts_implementation": "eager",
        "offloaded_experts_implementation": "serial_per_expert",
        "trace_schema_version": trace.metadata.get("schema_version", 2),
        "forced_routing_weight_source": "recorded_trace_weights",
        "forced_routing_trace_sha256": trace.digest(),
        "reference_routing_mode": "recorded_ids_and_weights_replayed_into_full_eager",
        "reference_forced_routing_assignments": int(replay_ids.size),
        "natural_reference_route_ids_exact_by_step": route_id_exact,
        "natural_reference_route_weights_exact_by_step": route_weight_exact,
        "natural_reference_routes_exact": all(route_id_exact),
        "natural_reference_weights_exact": all(route_weight_exact),
        "reference_route_ids_exact_by_step": [True] * args.decode_steps,
        "reference_route_weights_exact_by_step": [True] * args.decode_steps,
        "reference_routes_exact": True,
        "reference_weights_exact": True,
        "logits_allclose_all_steps": all(
            row["allclose_atol_0p1_rtol_0p01"] for row in comparisons
        ),
        "argmax_equal_all_steps": all(row["argmax_equal"] for row in comparisons),
        "correctness_gate": (
            "bitwise_equal_full_eager_and_offloaded_logits_under_identical_"
            "recorded_ids_and_weights"
        ),
        "correctness_gate_passed": correctness_gate_passed,
        "comparisons": comparisons,
        "engine_metrics": engine_metrics,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not correctness_gate_passed:
        raise RuntimeError("weighted replay correctness gate failed")


if __name__ == "__main__":
    main()
