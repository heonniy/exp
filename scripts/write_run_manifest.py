#!/usr/bin/env python3
"""Write outputs/run_manifest.json with model/env/runtime metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import MODEL_DIR, MODEL_NAME, extract_runtime_config  # noqa: E402
from src.utils import read_json, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/run_manifest.json")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    rc = extract_runtime_config(cfg)

    commits = {}
    cpath = Path("data/manifests/dataset_commits.json")
    if cpath.exists():
        commits = read_json(cpath)

    manifest = {
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_DIR),
        "model_revision": getattr(cfg, "transformers_version", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "bfloat16",
        "attn_implementation_prefill": "sdpa",
        "attn_implementation_decode": "sdpa:MATH (deterministic)",
        "determinism": {
            "use_deterministic_algorithms": True,
            "tf32": False,
            "cublas_workspace_config": ":4096:8",
        },
        "num_hidden_layers": rc.num_hidden_layers,
        "num_experts": rc.num_experts,
        "num_experts_per_tok": rc.num_experts_per_tok,
        "norm_topk_prob": rc.norm_topk_prob,
        "moe_layer_ids": rc.moe_layer_ids,
        "max_position_embeddings": rc.max_position_embeddings,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
        },
        "dataset_commits": commits,
    }
    write_json(args.out, manifest)
    print(f"[manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
