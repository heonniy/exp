#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LABEL COMMAND [ARG ...]" >&2
  exit 2
fi

label="$1"
shift
if [[ ! "${label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LABEL may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${project_dir}/experiments/results/nsys"
mkdir -p "${output_dir}"

exec "${project_dir}/scripts/gpu0.sh" \
  nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${output_dir}/${label}" \
  "$@"
