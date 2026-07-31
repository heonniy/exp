#!/usr/bin/env bash
# Full observation-study pipeline, unattended. Runs generation for both
# datasets, then signatures + Exp1 + Exp2 + figures + summaries.
# GPU: restricted to 6,7 (single-GPU load on the first visible device).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

export CUDA_VISIBLE_DEVICES=6,7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
MNT=192

echo "[pipe] $(date) START"

echo "[pipe] === run manifest ==="
$PY scripts/write_run_manifest.py --max-new-tokens $MNT

echo "[pipe] === generate SQuALITY ==="
$PY scripts/generate_and_trace.py \
  --requests data/processed/squality_requests.jsonl \
  --output-dir outputs/traces/squality \
  --max-new-tokens $MNT --skip-existing

echo "[pipe] === generate QMSum ==="
$PY scripts/generate_and_trace.py \
  --requests data/processed/qmsum_requests.jsonl \
  --output-dir outputs/traces/qmsum \
  --max-new-tokens $MNT --skip-existing

echo "[pipe] === reproducibility check (5 each) ==="
$PY scripts/check_reproducibility.py \
  --requests data/processed/squality_requests.jsonl \
  --trace-dir outputs/traces/squality \
  --gen-jsonl outputs/generations/squality/generations.jsonl --n 5 || echo "[pipe] WARN repro squality"
$PY scripts/check_reproducibility.py \
  --requests data/processed/qmsum_requests.jsonl \
  --trace-dir outputs/traces/qmsum \
  --gen-jsonl outputs/generations/qmsum/generations.jsonl --n 5 || echo "[pipe] WARN repro qmsum"

echo "[pipe] === build signatures ==="
$PY scripts/build_signatures.py --trace-dir outputs/traces --datasets squality,qmsum

echo "[pipe] === Experiment 1 ==="
$PY scripts/analyze_exp1.py

echo "[pipe] === Experiment 2 ==="
$PY scripts/analyze_exp2.py

echo "[pipe] === figures ==="
$PY scripts/make_figures.py

echo "[pipe] === summaries ==="
$PY scripts/build_summary.py --dataset squality
$PY scripts/build_summary.py --dataset qmsum

echo "[pipe] $(date) PIPELINE_DONE_OK"
