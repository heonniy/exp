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
  --calibration-trace artifacts/traces/calibration_4k256.npz \
  --expert-bytes 9437184 \
  --batch-size 40 --requests 1200 \
  --permanent-method batch_step_union_presence \
  --quota-controls ascending_always_admit resident_hit_first \
    miss_bypass no_admission window_frequency random_order \
  --bmax-csv experiments/results/bmax/bmax.csv \
  --output-dir experiments/results/trace_controls_b40_n1200

# Real serial Expert offload uses 58GB of host-pinned model weights.
./scripts/gpu0.sh .venv/bin/python -m experiments.benchmark.run_offloaded_decode \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --policy stream2 --k 0 --batch-size 1 \
  --host-memory-mode pinned_weights --max-pinned-experts 6144 \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --output experiments/results/stream2_b1.json
```

The primary decode-only path can allocate the full 4,352-token KV shape without
timing prompt prefill:

```bash
./scripts/gpu0.sh .venv/bin/python -m experiments.benchmark.run_offloaded_decode \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --policy stream2 --k 0 --batch-size 1 --kv-setup static_zero \
  --host-memory-mode pinned_weights --max-pinned-experts 6144 \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --output experiments/results/stream2_static_kv_b1.json
```

`static_zero` is a decode-only memory/performance fixture. It preallocates the real
Transformers `StaticCache`, initializes its sequence length to 4,096, and runs the
real attention/router/Expert code for the 256 replayed tokens. Use `real_prefill`
when prompt-dependent KV values or end-to-end timing matter.

Natural routing from a `static_zero` run is not comparable with a reference trace
collected after real prompt prefill. Routing diagnostics therefore report
position, active-set, and order-only mismatch separately and label reference
comparability explicitly.

## Trace collection

Trace collection writes resumable part files, so an interrupted run continues from
the first missing request:

```bash
./scripts/gpu0.sh .venv/bin/python \
  -m experiments.trace.collect_forced_routing_trace \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --input artifacts/data/lmsys_4k256_evaluation.jsonl \
  --output artifacts/traces/evaluation_4k256.npz \
  --batch-size 20
```

## Measured Bmax and runtime sweeps

The real Bmax probe loads the dense model and selected residency policy, reserves
the common 2 GiB safety margin, allocates the full peak KV cache, and executes one
forced-routing decode step. It is distinct from the synthetic allocator probe.

The complete 1,200-request 4K/256 matrix, including intermediate k=48, measured
Bmax, common physical B=40, 256-step makespan, and CUDA-event timing breakdown is
restartable with one command:

```bash
./scripts/gpu0.sh .venv/bin/python \
  -m experiments.benchmark.run_4k256_completion
```

```bash
./scripts/gpu0.sh .venv/bin/python \
  -m experiments.benchmark.run_bmax_sweep \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --calibration-trace artifacts/traces/calibration_4k256.npz \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --expert-bytes 9437184 --dense-bytes 3082186752 \
  --output-dir experiments/results/bmax

./scripts/gpu0.sh .venv/bin/python \
  -m experiments.benchmark.run_runtime_sweep \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --calibration-trace artifacts/traces/calibration_4k256.npz \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --bmax-dir experiments/results/bmax \
  --output-dir experiments/results/runtime_at_bmax
```

After selecting a configuration, run every strict evaluation row in full and
last-partial waves without reloading the model:

```bash
./scripts/gpu0.sh .venv/bin/python \
  -m experiments.benchmark.run_fixed_workload \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --calibration-trace artifacts/traces/calibration_4k256.npz \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --policy quota_lru_k --k 8 --batch-size 178 \
  --output experiments/results/quota_k8_fixed_workload.json
```

Quota state is warm across waves by default. Add `--cold-each-wave` for the cold
Quota-LRU control. Replace the example batch 178 with the corresponding measured
Bmax; each wave records its exact batch size and decode time.

For the `stream1_no_prefetch` micro-ablation, run the decode command with
`--prefetch-depth 0`; the primary curve and sweep default to depth 1.
The primary depth-1 executor uses `--prefetch-submit-order compute_first`.
Add `--timeline-events` only to separately labeled representative profiles to
record CUDA-event H2D duration, compute-stream exposed wait, overlap ratio,
first-miss stall, and copy-engine utilization. Per-Expert events are intrusive
at B=40 and must not be enabled for throughput or fixed-workload makespan runs.
Nsight Systems capture is also constrained to GPU 0 and stored under the project:

```bash
./scripts/nsys_gpu0.sh stream2_k0 \
  .venv/bin/python -m experiments.benchmark.run_offloaded_decode \
  --config experiments/configs/h100_lmsys_4k256.yaml \
  --workload artifacts/data/lmsys_4k256_evaluation.jsonl \
  --policy stream2 --k 0 --batch-size 1 --decode-steps 4 \
  --kv-setup static_zero --host-memory-mode pinned_weights \
  --max-pinned-experts 6144 \
  --forced-routing-trace artifacts/traces/evaluation_4k256.npz \
  --output experiments/results/nsys_stream2_k0.json
```

## Current dataset limitation

The local LMSYS parquet snapshot contains 1,529 examples satisfying every strict
4K/256 filter. The completion experiment uses a fixed 256-row calibration split
and the first 1,200 evaluation rows as explicitly requested. It never duplicates
or pads examples. Exact token lengths and split disjointness are validated.

See `--help` on each module for smaller smoke-test sizes and dry-run sweep manifests.
