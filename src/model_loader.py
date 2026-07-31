"""Model / tokenizer loading and runtime-config extraction.

All expert/layer counts are read from the runtime config -- never hardcoded.
GPU selection is enforced via CUDA_VISIBLE_DEVICES (must be set to 6,7 by the
launcher). The local model checkpoint lives at exp/model/Qwen3-30B-A3B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT.parent / "model" / "Qwen3-30B-A3B"
MODEL_NAME = "Qwen/Qwen3-30B-A3B"


@dataclass
class RuntimeConfig:
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    max_position_embeddings: int
    moe_layer_ids: list[int]


def _resolve_field(config, *names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    if default is not None:
        return default
    raise AttributeError(f"config has none of {names}")


def extract_runtime_config(config) -> RuntimeConfig:
    from src.router_trace import moe_layer_ids_from_config

    return RuntimeConfig(
        num_hidden_layers=_resolve_field(config, "num_hidden_layers"),
        num_experts=_resolve_field(config, "num_experts", "num_local_experts"),
        num_experts_per_tok=_resolve_field(config, "num_experts_per_tok"),
        norm_topk_prob=bool(_resolve_field(config, "norm_topk_prob", default=True)),
        max_position_embeddings=_resolve_field(config, "max_position_embeddings"),
        moe_layer_ids=moe_layer_ids_from_config(config),
    )


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)


def enable_determinism() -> None:
    """Best-effort bitwise determinism for the forward pass.

    The decode-time attention kernel (SDPA flash-decoding) and MoE scatter use
    non-deterministic split/atomic reductions; forcing eager attention +
    deterministic algorithms removes that run-to-run jitter so greedy decode
    and router top-k reproduce exactly.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_model(
    dtype: torch.dtype = torch.bfloat16,
    single_gpu: bool = True,
    attn_implementation: str | None = None,
):
    """Load the causal LM onto the visible GPUs (expected: 6,7).

    By default the whole 30B (~60GB in bf16) is placed on the first visible GPU
    (device_map={"": 0} -> physical GPU 6). This avoids per-token cross-GPU
    pipeline latency that makes step-by-step router-trace capture ~10x slower.
    The second visible GPU (7) stays as headroom. Set single_gpu=False to shard
    across both with device_map="auto" (needed only if a context OOMs).
    """
    from transformers import AutoModelForCausalLM

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be set (expected '6,7') before loading."
        )
    device_map = {"": 0} if single_gpu else "auto"
    kwargs = dict(
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **kwargs)
    model.eval()
    return model


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device
