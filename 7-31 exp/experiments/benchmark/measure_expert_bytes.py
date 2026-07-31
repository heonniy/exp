from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.benchmark.checkpoint_layout import inspect_checkpoint
from experiments.common.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Qwen Expert bytes from safetensors metadata."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_checkpoint(args.model)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

