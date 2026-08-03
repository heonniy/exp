from __future__ import annotations

import os
from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True)
class VisibleGpu:
    logical_index: int
    physical_index: int
    name: str
    total_memory: int


def require_gpu0(torch_module=None) -> VisibleGpu:
    """Fail closed unless the process can see exactly physical GPU 0.

    CUDA remaps physical GPU 0 to logical device 0 after
    ``CUDA_VISIBLE_DEVICES=0``. Requiring that exact environment value prevents
    accidental multi-GPU use even when the host has more devices.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            "GPU commands must run with CUDA_VISIBLE_DEVICES=0 "
            "(use ./scripts/gpu0.sh)"
        )

    if torch_module is None:
        import torch as torch_module

    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch")
    if torch_module.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible GPU after masking, found "
            f"{torch_module.cuda.device_count()}"
        )
    props = torch_module.cuda.get_device_properties(0)
    return VisibleGpu(
        logical_index=0,
        physical_index=0,
        name=props.name,
        total_memory=int(props.total_memory),
    )


def effective_hbm_bytes(gpu: VisibleGpu, configured_gib: float | None) -> int:
    """Resolve the accounted HBM capacity without exceeding physical HBM."""

    if configured_gib is None:
        return gpu.total_memory
    requested = int(configured_gib * GIB)
    if requested <= 0:
        raise ValueError("configured HBM limit must be positive")
    return min(gpu.total_memory, requested)


def apply_effective_hbm_limit(
    torch_module, gpu: VisibleGpu, configured_gib: float | None
) -> int:
    """Apply an allocator-enforced HBM cap and return its exact byte value.

    Accounting alone would let a run on an 80 GB card silently consume more
    than the emulated device capacity. PyTorch's per-process allocator limit
    makes the configured cap an actual runtime constraint as well.
    """

    effective = effective_hbm_bytes(gpu, configured_gib)
    if effective < gpu.total_memory:
        torch_module.cuda.set_per_process_memory_fraction(
            effective / gpu.total_memory,
            device=0,
        )
    return effective
