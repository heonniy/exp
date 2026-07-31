from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.benchmark.find_runtime_max_batch import (
    _clear_cuda_probe_state,
    _detach_engine,
)
from experiments.benchmark.run_offloaded_decode import _manager, _read_examples
from experiments.benchmark.run_runtime_sweep import configurations
from experiments.common.config import load_config
from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json
from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.kv_cache import make_static_kv_cache
from experiments.runtime.offloaded_model import (
    OffloadedExpertEngine,
    attach_engine,
    load_offloaded_qwen,
)
from experiments.runtime.residency_manager import StreamingRuntimeManager
from experiments.trace.trace_schema import RoutingTrace


def assert_closed_monotonic_boundary(
    probes: dict[int, dict], validated: int
) -> None:
    required = {validated, validated + 1}
    if validated > 1:
        required.add(validated - 1)
    missing = sorted(required - set(probes))
    if missing:
        raise AssertionError(f"Bmax boundary probes are missing: {missing}")
    if not probes[validated]["feasible"]:
        raise AssertionError("validated Bmax is not feasible")
    if probes[validated + 1]["feasible"]:
        raise AssertionError("Bmax+1 is feasible; boundary is not closed")
    if validated > 1 and not probes[validated - 1]["feasible"]:
        raise AssertionError("Bmax-1 is not feasible; boundary is non-monotonic")
    ordered = sorted(probes)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not probes[smaller]["feasible"] and probes[larger]["feasible"]:
            raise AssertionError(
                f"non-monotonic feasibility: B={smaller} failed but "
                f"B={larger} succeeded"
            )


def _probe_256(
    *,
    model,
    host_store: PinnedExpertStore,
    calibration: RoutingTrace,
    trace: RoutingTrace,
    forced_tokens: np.ndarray,
    policy: str,
    k: int,
    batch_size: int,
    decode_steps: int,
    input_tokens: int,
    peak_sequence_length: int,
    safety_margin_bytes: int,
    num_layers: int,
    num_experts: int,
    permanent_method: str,
) -> dict:
    manager = engine = cache = reserve = output = None
    _detach_engine(model)
    _clear_cuda_probe_state()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        manager, policy_initialization_seconds = _manager(
            policy,
            k,
            num_layers,
            num_experts,
            host_store,
            calibration,
            permanent_method=permanent_method,
            permanent_batch_size=batch_size,
        )
        engine = OffloadedExpertEngine(
            host_store,
            manager,
            prefetch_depth=1,
            track_timeline=False,
            prefetch_submit_order="compute_first",
        )
        attach_engine(model, engine)
        reserve = torch.empty(safety_margin_bytes, dtype=torch.uint8, device="cuda:0")
        reserve.zero_()
        cache = make_static_kv_cache(
            model,
            batch_size=batch_size,
            max_cache_length=peak_sequence_length,
            initial_sequence_length=input_tokens,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        engine.set_forced_routing(
            trace.routing_expert_ids[:batch_size, :decode_steps],
            trace.require_routing_weights()[:batch_size, :decode_steps],
        )
        with torch.inference_mode():
            for step in range(decode_steps):
                engine.decode_step = step
                token = torch.as_tensor(
                    forced_tokens[:batch_size, step, None],
                    dtype=torch.long,
                    device="cuda:0",
                )
                output = model(
                    input_ids=token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    output_router_logits=False,
                    return_dict=True,
                )
                cache = output.past_key_values
                if (step + 1) % 32 == 0 or step + 1 == decode_steps:
                    print(
                        f"policy={policy} k={k} B={batch_size} "
                        f"step={step + 1}/{decode_steps}",
                        flush=True,
                    )
        torch.cuda.synchronize()
        metrics = engine.metrics()
        if metrics["h2d_copy_operations_per_fetch"] != 1.0:
            raise AssertionError("packed runtime did not issue one H2D copy per fetch")
        return {
            "batch_size": batch_size,
            "feasible": True,
            "decode_steps_completed": decode_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "policy_initialization_seconds": policy_initialization_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "expert_h2d_fetches": metrics["expert_h2d_fetches"],
            "expert_h2d_copy_operations": metrics["expert_h2d_copy_operations"],
            "h2d_copy_operations_per_fetch": metrics[
                "h2d_copy_operations_per_fetch"
            ],
        }
    except torch.OutOfMemoryError as error:
        return {
            "batch_size": batch_size,
            "feasible": False,
            "decode_steps_completed": (
                step if "step" in locals() else 0
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error).splitlines()[0],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        _detach_engine(model)
        del output, cache, reserve, engine, manager
        _clear_cuda_probe_state()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turn provisional one-step Bmax values into physical Bmax values by "
            "validating the B-1/B/B+1 boundary for all 256 decode steps."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--forced-routing-trace", type=Path, required=True)
    parser.add_argument("--provisional-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument(
        "--permanent-method",
        choices=["batch_step_union_presence", "token_frequency"],
        default="batch_step_union_presence",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    if args.decode_steps != config.dataset.output_tokens:
        raise ValueError("physical Bmax validation must run all configured tokens")
    trace = RoutingTrace.load(args.forced_routing_trace)
    trace.validate(config.model.num_experts_per_layer, require_weights=True)
    trace.require_serial_reference()
    calibration = RoutingTrace.load(args.calibration_trace)
    runs = list(configurations(config.runtime_k, config.model.num_experts_per_layer))
    provisional = {}
    maximum_candidate = 0
    for policy, k in runs:
        path = args.provisional_dir / f"{policy}_k{k}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        candidate = int(value["measured_bmax"])
        provisional[(policy, k)] = value
        maximum_candidate = max(maximum_candidate, candidate + 1)
    if maximum_candidate > trace.num_requests:
        raise ValueError("trace has too few requests for the Bmax boundary")
    examples = _read_examples(args.workload, maximum_candidate)
    expected_ids = [str(item) for item in trace.conversation_ids[:maximum_candidate]]
    if [str(row["conversation_id"]) for row in examples] != expected_ids:
        raise ValueError("trace and workload row order differ")
    forced_tokens = np.asarray(
        [row["forced_output_ids"][: args.decode_steps] for row in examples],
        dtype=np.int64,
    )
    if not np.array_equal(
        forced_tokens.astype(np.int32),
        trace.forced_output_ids[:maximum_candidate, : args.decode_steps],
    ):
        raise ValueError("trace and workload forced token IDs differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host_store = PinnedExpertStore(
        config.model.path,
        max_pinned_experts=(
            config.model.num_moe_layers * config.model.num_experts_per_layer
        ),
        pin_weights=True,
    )
    preload_seconds = host_store.preload_all(
        config.model.num_moe_layers, config.model.num_experts_per_layer
    )
    bootstrap = OffloadedExpertEngine(host_store, StreamingRuntimeManager())
    model = load_offloaded_qwen(config.model.path, bootstrap)
    _detach_engine(model)
    del bootstrap
    _clear_cuda_probe_state()

    manifest_runs = []
    for policy, k in runs:
        output_path = args.output_dir / f"{policy}_k{k}.json"
        if output_path.exists():
            manifest_runs.append(
                {"policy": policy, "k": k, "output": str(output_path), "completed": True}
            )
            continue
        source = provisional[(policy, k)]
        candidate = int(source["measured_bmax"])
        probes: dict[int, dict] = {}

        def probe(batch_size: int) -> dict:
            if batch_size not in probes:
                attempts = []
                for _ in range(2):
                    value = _probe_256(
                        model=model,
                        host_store=host_store,
                        calibration=calibration,
                        trace=trace,
                        forced_tokens=forced_tokens,
                        policy=policy,
                        k=k,
                        batch_size=batch_size,
                        decode_steps=args.decode_steps,
                        input_tokens=config.dataset.input_tokens,
                        peak_sequence_length=config.peak_sequence_length,
                        safety_margin_bytes=int(source["safety_margin_bytes"]),
                        num_layers=config.model.num_moe_layers,
                        num_experts=config.model.num_experts_per_layer,
                        permanent_method=args.permanent_method,
                    )
                    attempts.append(value)
                    if value["feasible"]:
                        break
                selected = dict(attempts[-1])
                selected["attempt_count"] = len(attempts)
                selected["transient_oom_recovered"] = bool(
                    len(attempts) > 1 and selected["feasible"]
                )
                selected["attempts"] = attempts
                probes[batch_size] = selected
            return probes[batch_size]

        for batch_size in sorted({max(1, candidate - 1), candidate, candidate + 1}):
            probe(batch_size)
        if probe(candidate)["feasible"]:
            validated = candidate
            while probe(validated + 1)["feasible"]:
                validated += 1
        else:
            validated = candidate - 1
            while validated > 0 and not probe(validated)["feasible"]:
                validated -= 1
        if validated <= 0:
            raise RuntimeError(f"no feasible batch found for {policy} k={k}")
        probe(max(1, validated - 1))
        probe(validated)
        probe(validated + 1)
        assert_closed_monotonic_boundary(probes, validated)

        result = {
            **{
                key: value
                for key, value in source.items()
                if key not in {"probes", "measured_bmax", "probe_mode"}
            },
            "probe_mode": "real_runtime_static_peak_kv_full_256_step_boundary",
            "provisional_one_step_bmax": candidate,
            "measured_bmax": validated,
            "decode_steps": args.decode_steps,
            "boundary_batches_required": [
                max(1, validated - 1),
                validated,
                validated + 1,
            ],
            "boundary_closed": True,
            "host_store_preload_seconds": preload_seconds,
            "probes": [probes[key] for key in sorted(probes)],
        }
        atomic_write_json(output_path, result)
        manifest_runs.append(
            {"policy": policy, "k": k, "output": str(output_path), "completed": True}
        )
        atomic_write_json(
            args.output_dir / "manifest.json",
            {
                "config": config.name,
                "probe_mode": "real_runtime_static_peak_kv_full_256_step_boundary",
                "decode_steps": args.decode_steps,
                "forced_routing_trace_sha256": trace.digest(),
                "runs": manifest_runs,
            },
        )

    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "config": config.name,
            "probe_mode": "real_runtime_static_peak_kv_full_256_step_boundary",
            "decode_steps": args.decode_steps,
            "forced_routing_trace_sha256": trace.digest(),
            "runs": manifest_runs,
        },
    )
    print(json.dumps({"completed": len(manifest_runs), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
