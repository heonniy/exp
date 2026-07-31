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


def _router_topk(outputs, expected_layers: int, top_k: int) -> np.ndarray:
    router_logits = outputs.router_logits
    if router_logits is None or len(router_logits) != expected_layers:
        raise RuntimeError(
            f"expected router logits for {expected_layers} layers, got "
            f"{None if router_logits is None else len(router_logits)}"
        )
    layer_routes = []
    for logits in router_logits:
        final_position = logits.reshape(-1, logits.shape[-1])[-1]
        expert_ids = torch.topk(final_position, k=top_k, dim=-1).indices
        layer_routes.append(expert_ids.to(dtype=torch.uint8, device="cpu").numpy())
    return np.stack(layer_routes)


@torch.inference_mode()
def collect_one(model, example: dict, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda:0")
    prompt = torch.tensor(example["input_ids"], dtype=torch.long, device=device)[None]
    forced = np.asarray(example["forced_output_ids"], dtype=np.int32)
    if prompt.shape[1] != config.dataset.input_tokens:
        raise ValueError("input does not have the configured fixed length")
    if len(forced) != config.dataset.output_tokens:
        raise ValueError("forced output does not have the configured fixed length")

    prefill = model(
        input_ids=prompt,
        use_cache=True,
        output_router_logits=False,
        return_dict=True,
    )
    past = prefill.past_key_values
    routes = np.empty(
        (
            config.dataset.output_tokens,
            config.model.num_moe_layers,
            config.model.router_top_k,
        ),
        dtype=np.uint8,
    )
    for step, token_id in enumerate(forced):
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        output = model(
            input_ids=token,
            past_key_values=past,
            use_cache=True,
            output_router_logits=True,
            return_dict=True,
        )
        past = output.past_key_values
        routes[step] = _router_topk(
            output, config.model.num_moe_layers, config.model.router_top_k
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

    if not args.finalize_only:
        model = load_model(config)
        for index, example in enumerate(examples):
            part = parts_dir / f"{index:06d}.npz"
            if part.exists():
                continue
            forced, routes = collect_one(model, example, config)
            _save_part(part, str(example["conversation_id"]), forced, routes)
            print(f"collected {index + 1}/{len(examples)}", flush=True)

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

