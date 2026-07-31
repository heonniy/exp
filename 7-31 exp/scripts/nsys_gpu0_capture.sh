#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_PREFIX COMMAND [ARG ...]" >&2
  exit 2
fi

output_prefix="$1"
shift
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "${project_dir}/scripts/gpu0.sh" \
  nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${output_prefix}" \
  "$@"
