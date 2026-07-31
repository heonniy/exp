from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from experiments.common.config import ExperimentConfig, load_config
from experiments.common.gpu import require_gpu0
from experiments.trace.trace_schema import concatenate_parts


def _examples(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _save_part(
    path: Path,
    conversation_id: str,
    forced_output_ids: np.ndarray,
    routing_expert_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                conversation_id=np.asarray(conversation_id, dtype=np.str_),
                forced_output_ids=forced_output_ids.astype(np.int32),
                routing_expert_ids=routing_expert_ids.astype(np.uint8),
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _router_topk(
    outputs, expected_layers: int, top_k: int, batch_size: int
) -> np.ndarray:
    router_logits = outputs.router_logits
    if router_logits is None or len(router_logits) != expected_layers:
        raise RuntimeError(
            f"expected router logits for {expected_layers} layers, got "
            f"{None if router_logits is None else len(router_logits)}"
        )
    layer_routes = []
    for logits in router_logits:
        final_position = logits.reshape(batch_size, -1, logits.shape[-1])[:, -1, :]
        expert_ids = torch.topk(final_position, k=top_k, dim=-1).indices
        layer_routes.append(expert_ids.to(dtype=torch.uint8, device="cpu").numpy())
    return np.stack(layer_routes, axis=1)


@torch.inference_mode()
def collect_batch(
    model, examples: list[dict], config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda:0")
    prompt = torch.tensor(
        [example["input_ids"] for example in examples],
        dtype=torch.long,
        device=device,
    )
    forced = np.asarray(
        [example["forced_output_ids"] for example in examples], dtype=np.int32
    )
    if prompt.shape[1] != config.dataset.input_tokens:
        raise ValueError("input does not have the configured fixed length")
    if forced.shape[1] != config.dataset.output_tokens:
        raise ValueError("forced output does not have the configured fixed length")
    batch_size = len(examples)

    prefill = model(
        input_ids=prompt,
        use_cache=True,
        output_router_logits=False,
        logits_to_keep=1,
        return_dict=True,
    )
    past = prefill.past_key_values
    routes = np.empty(
        (
            batch_size,
            config.dataset.output_tokens,
            config.model.num_moe_layers,
            config.model.router_top_k,
        ),
        dtype=np.uint8,
    )
    for step in range(config.dataset.output_tokens):
        token = torch.as_tensor(
            forced[:, step, None], dtype=torch.long, device=device
        )
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            output_router_logits=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past = output.past_key_values
        routes[:, step] = _router_topk(
            output,
            config.model.num_moe_layers,
            config.model.router_top_k,
            batch_size,
        )
    return forced, routes


def load_model(config: ExperimentConfig):
    dtype = getattr(torch, config.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.output_router_logits = True
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect resumable Qwen forced-routing traces on physical GPU 0."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Collect only the first N requests for a smoke test."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Requests traced together; reduce if full-resident tracing OOMs.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Build the final NPZ from already collected part files.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    require_gpu0(torch)
    parts_dir = args.output.with_suffix(args.output.suffix + ".parts")
    examples = list(_examples(args.input))
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("input split is empty")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if not args.finalize_only:
        model = load_model(config)
        missing_indices = [
            index
            for index in range(len(examples))
            if not (parts_dir / f"{index:06d}.npz").exists()
        ]
        for offset in range(0, len(missing_indices), args.batch_size):
            indices = missing_indices[offset : offset + args.batch_size]
            batch = [examples[index] for index in indices]
            forced, routes = collect_batch(model, batch, config)
            for batch_index, example_index in enumerate(indices):
                example = examples[example_index]
                _save_part(
                    parts_dir / f"{example_index:06d}.npz",
                    str(example["conversation_id"]),
                    forced[batch_index],
                    routes[batch_index],
                )
            print(
                f"collected {min(offset + len(indices), len(missing_indices))}/"
                f"{len(missing_indices)} missing requests",
                flush=True,
            )

    expected_parts = [parts_dir / f"{index:06d}.npz" for index in range(len(examples))]
    missing = [str(part) for part in expected_parts if not part.exists()]
    if missing:
        raise RuntimeError(f"cannot finalize; {len(missing)} trace parts are missing")
    concatenate_parts(
        expected_parts,
        args.output,
        metadata={
            "config_name": config.name,
            "model_path": str(config.model.path),
            "num_experts": config.model.num_experts_per_layer,
            "gpu_physical_index": 0,
            "source_split": str(args.input),
        },
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
