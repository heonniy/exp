# Analysis Agent Guide

This document tells a downstream analysis agent what evidence to inspect, how to
interpret it, and which conclusions are not yet supported.

## 1. Analysis objective

Evaluate the H100 HBM trade-off between GPU KV capacity and layer-local Qwen3
Expert residency, without a global Expert cache. The primary question is whether
reduced Expert H2D refetch compensates for the smaller feasible KV batch.

The implementation target and invariant definitions are in
`h100_lmsys_kv_expert_residency_spec.md`.

## 2. Read these files in order

1. `experiments/results/progress.md`
   - Concise status, validated measurements, current findings, and limitations.
2. `experiments/results/bmax/bmax.csv`
   - Real-runtime HBM endpoints. Prefer `measured_bmax` over theoretical Bmax.
3. `experiments/results/trace_sweep_b50/cache_summary.csv`
   - Full 1,273-request cache simulation at fixed batch 50.
4. `experiments/results/trace_sweep_b50/per_layer.csv`
   - Layer-level hits, refetches, evictions, and traffic reduction.
5. `experiments/results/trace_sweep_b50/trace_diagnostics.png`
   - Visual summary of hit rate, H2D traffic, refetches, and layer variation.
6. `experiments/results/offloaded_smoke_summary.csv`
   - Real pinned-weight offload integration smoke for stream2, Permanent-k=8,
     and Quota-LRU-k=8. It is only four decode tokens at batch 1.
7. `experiments/results/offloaded_stream1_static_timeline_smoke.json` and
   `experiments/results/offloaded_stream2_static_timeline_smoke.json`
   - One-slot no-prefetch versus two-slot one-ahead CUDA-event validation.
8. `experiments/results/environment.json`, `expert_bytes.json`,
   `h2d_bandwidth.json`, `compute_copy_overlap.json`, and
   `memory_breakdown.csv`
   - Hardware, checkpoint, transfer, overlap, and theoretical accounting inputs.

Inspect implementation details only after understanding the results:

- `experiments/runtime/serial_expert_executor.py`
- `experiments/runtime/residency_manager.py`
- `experiments/runtime/offloaded_model.py`
- `experiments/benchmark/find_runtime_max_batch.py`
- `experiments/benchmark/run_offloaded_decode.py`
- `experiments/benchmark/run_fixed_workload.py`
- `experiments/trace/simulator.py`

## 3. Evidence hierarchy

Use the following priority when evidence differs:

1. Real-runtime measured Bmax is stronger than theoretical memory accounting.
2. Full 1,273-request trace results are stronger than smoke trace results for
   cache behavior.
3. Pinned-weight offload measurements are stronger than pinned-staging wall time
   for throughput. Pinned-staging timeline runs are functional/timeline checks.
4. Trace simulation predicts Expert traffic and cache behavior; it does not prove
   end-to-end throughput.
5. Four-token and one-token smoke runs establish integration correctness only.

All forced-token/routing comparisons should use the recorded hashes. The full
evaluation routing trace SHA-256 is:

`2dd67a486095614f1ec2071bbf0ac2cfc34fe097def9062bf1c9511f2dbaadac`

## 4. Findings currently supported

### HBM endpoints

- Stream2 k=0: theoretical Bmax 186, measured Bmax 157.
- Full-resident k=128: theoretical Bmax 50, measured Bmax 41.
- Full residency therefore costs 116 feasible requests relative to stream2.
- Successful full-resident probes have zero decode Expert H2D fetches.

### Fixed-batch-50 cache behavior

- Stream2 performs 33,691,149 Expert fetches.
- Permanent-k reaches 58.97% hit rate at k=64 and 84.98% at k=96.
- Quota-LRU reaches only 0.072% at k=64 and 7.16% at k=96.
- The Quota result is consistent with cyclic ascending-ID scans over a
  batch-level active set approaching all 128 Experts. It is batch-dependent, not
  a universal statement that Quota-LRU is ineffective.

### Prefetch correctness

- Depth 0 / one slot has zero measured compute-copy interval overlap.
- Depth 1 / two slots overlaps 71.78 ms, or 50.4%, in the timeline smoke.
- Both paths produce the same final-logits digest, supporting execution
  correctness across the micro-ablation.

## 5. Conclusions that are not yet supported

Do not declare the globally optimal k or winning runtime policy yet. The repository
does not yet contain all of the following primary measurements:

- Real Bmax for every intermediate Permanent/Quota k.
- Full 256-token decode throughput at each configuration's measured Bmax.
- Fixed-batch throughput at one common feasible batch.
- Exact fixed-workload makespan for every policy, including the final partial wave.
- Nsight Systems validation for k=0, the eventual optimal k, and k=128.
- Representative pinned-weight timeline results beyond the integration smokes.

The strict local LMSYS snapshot also has only 1,273 eligible evaluation rows rather
than the requested 2,048. Never describe the results as a 2,048-request evaluation.

## 6. Required decision procedure

For each policy/k, build one row containing:

- measured Bmax and batch cost versus stream2;
- persistent and transient HBM;
- decode tokens/s at measured Bmax;
- decode tokens/s at the fixed common batch;
- Expert H2D bytes/generated token;
- compulsory loads and refetches;
- exposed H2D stall and Expert compute time;
- cold-start and steady-state costs;
- fixed-workload makespan, including the partial wave.

Then judge in this order:

1. Reject configurations that fail HBM or correctness invariants.
2. Compare maximum-feasible-batch throughput as the primary operating curve.
3. Use fixed-batch throughput to isolate residency effects from batch effects.
4. Confirm that H2D savings reduce exposed stall rather than only total traffic.
5. Compare full-workload makespan, including cold/warm state and partial waves.
6. Report uncertainty from the 1,273-request sample and incomplete intermediate
   runtime points.

The final recommendation should distinguish a deployable Permanent/Quota policy
from `permanent_oracle`, which is an evaluation-trace upper bound.

## 7. Reproduction boundaries

Large/generated inputs are intentionally not committed to GitHub:

- `/home/hwlee/model/Qwen3-30B-A3B`
- `/home/hwlee/dataset/*.parquet`
- `artifacts/data/*.jsonl`
- `artifacts/traces/*.npz` and resumable part files
- `.venv/` and local caches

The committed JSON/CSV results contain trace digests and configuration identities
so an analysis agent can audit existing evidence. Re-running the experiments needs
the local model, strict workload, and routing traces above. All GPU commands must
use `scripts/gpu0.sh`; no other physical GPU is in scope.
