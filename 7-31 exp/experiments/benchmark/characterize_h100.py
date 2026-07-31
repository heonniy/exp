from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record the physical GPU 0 and host topology."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = require_gpu0(torch)
    properties = torch.cuda.get_device_properties(0)
    sm_count = int(properties.multi_processor_count)
    result = {
        "gpu_physical_index": 0,
        "gpu_logical_index": 0,
        "gpu_name": gpu.name,
        "gpu_total_hbm_bytes": gpu.total_memory,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_sm_count": sm_count,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "python_version": platform.python_version(),
        "kernel": platform.release(),
        "cpu_model": platform.processor()
        or _command("sh", "-c", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"),
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
        "numa_hardware": _command("numactl", "--hardware"),
        "gpu0_topology": _command("nvidia-smi", "topo", "-m"),
        "cuda_visible_devices": "0",
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

