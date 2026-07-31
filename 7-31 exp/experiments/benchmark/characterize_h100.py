from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import platform
import subprocess
from pathlib import Path

import psutil
import torch

from experiments.common.gpu import require_gpu0
from experiments.common.io import atomic_write_json


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _cuda_device_attribute(attribute: int) -> int | None:
    candidates = [
        ctypes.util.find_library("cudart"),
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
            value = ctypes.c_int()
            function = library.cudaDeviceGetAttribute
            function.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
            ]
            function.restype = ctypes.c_int
            if function(ctypes.byref(value), attribute, 0) == 0:
                return int(value.value)
        except (AttributeError, OSError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the physical GPU 0 and host topology."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = require_gpu0(torch)
    properties = torch.cuda.get_device_properties(0)
    sm_count = int(properties.multi_processor_count)
    if "PCIe" in gpu.name:
        variant = "PCIe"
    elif "H100" in gpu.name and "HBM3" in gpu.name:
        variant = "SXM (inferred from NVIDIA product name)"
    else:
        variant = "unknown"
    result = {
        "gpu_physical_index": 0,
        "gpu_logical_index": 0,
        "gpu_name": gpu.name,
        "gpu_total_hbm_bytes": gpu.total_memory,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_sm_count": sm_count,
        "gpu_variant": variant,
        # cudaDevAttrAsyncEngineCount is enum value 40 in CUDA 12.x.
        "cuda_async_engine_count": _cuda_device_attribute(40),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "python_version": platform.python_version(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "host_memory_bytes": psutil.virtual_memory().total,
        "nvidia_smi_gpu0": _command(
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ),
        "numa_hardware": _command("numactl", "--hardware")
        or _command("lscpu", "--json"),
        "gpu0_topology": _command("nvidia-smi", "topo", "-m"),
        "cuda_visible_devices": "0",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
