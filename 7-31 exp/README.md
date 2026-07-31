# H100 KV–Expert Residency Experiment

Implementation of [`h100_lmsys_kv_expert_residency_spec.md`](./h100_lmsys_kv_expert_residency_spec.md)
for Qwen3-30B-A3B on one H100.

The implementation intentionally has no global Expert LRU. Its main policies are:

- `stream2`: no persistent Experts and two transient streaming slots.
- `permanent_k`: calibration-selected, layer-local permanent Experts.
- `quota_lru_k`: adaptive layer-local LRU quotas.
- `full_resident`: the common `k=128` endpoint.

## Safety and reproducibility

All scripts refuse to use a physical GPU other than GPU 0. Run GPU commands with the
wrapper below; it sets `CUDA_VISIBLE_DEVICES=0` and keeps Hugging Face, Torch, and
Matplotlib caches inside this directory.

```bash
./scripts/gpu0.sh .venv/bin/python -m experiments.benchmark.characterize_h100 \
  --output experiments/results/environment.json
```

The local model and dataset defaults are:

```text
/home/hwlee/model/Qwen3-30B-A3B
/home/hwlee/dataset/*.parquet
```

They are read-only inputs. Generated samples, traces, plots, and benchmark results
go under `artifacts/` or `experiments/results/`.

## Quick start

```bash
# CPU tests
.venv/bin/python -m pytest

# Measure checkpoint Expert sizes without loading the model.
.venv/bin/python -m experiments.benchmark.measure_expert_bytes \
  --model /home/hwlee/model/Qwen3-30B-A3B \
  --output experiments/results/expert_bytes.json

# Prepare deterministic 4K/256 calibration and evaluation splits.
.venv/bin/python -m experiments.data.prepare_lmsys \
  --config experiments/configs/h100_lmsys_4k256.yaml

# Validate exact token lengths and split disjointness.
.venv/bin/python -m experiments.data.validate_fixed_lengths \
  --calibration artifacts/data/lmsys_4k256_calibration.jsonl \
  --evaluation artifacts/data/lmsys_4k256_evaluation.jsonl

# Simulate all cache policies after collecting a routing trace.
.venv/bin/python -m experiments.benchmark.run_residency_sweep \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --trace artifacts/traces/evaluation_4k256.npz \
  --calibration-trace artifacts/traces/calibration_4k256.npz
```

See `--help` on each module for smaller smoke-test sizes.

