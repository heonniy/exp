#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${project_dir}/.cache/huggingface"
export XDG_CACHE_HOME="${project_dir}/.cache"
export MPLCONFIGDIR="${project_dir}/.cache/matplotlib"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$@"
