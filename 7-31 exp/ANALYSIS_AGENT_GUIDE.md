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

1. `FEEDBACK_RESPONSE.md`
   - Review findings, implementation mapping, and corrected experiment protocol.
2. `experiments/results/completion_4k256_1200/manifest.json`
   - Restartable 1,200-request completion matrix and per-run completion state.
3. `experiments/results/completion_4k256_1200/bmax/bmax.csv`
   - Real-runtime Bmax for every primary policy/k after the completion run.
4. `experiments/results/completion_4k256_1200/runtime_at_bmax/`
   - Full 256-step, 1,200-request workloads at each measured Bmax.
5. `experiments/results/completion_4k256_1200/runtime_common_b40/`
   - Physical common-batch runtime; B=40 is no larger than full-resident Bmax 41.
6. `experiments/results/natural_routing_diagnostic_summary.json`
   - Same-prompt real-prefill and non-comparable static-zero mismatch breakdown.
7. `experiments/results/single_buffer_h2d_smoke_summary.json`
   - Verifies one contiguous H2D operation per Expert fetch and unchanged logits.
8. `experiments/results/trace_controls_b40_n1200/decision_summary.json`
   - Full-trace Quota order/admission sensitivity at physical common B=40.
9. `experiments/results/progress.md`
   - Historical status and the earlier 1,273-request/B=50 cache sweep.
10. `experiments/results/bmax/bmax.csv`
   - Earlier real-runtime HBM endpoints.
11. `experiments/results/trace_sweep_b50/cache_summary.csv`
   - Historical 1,273-request cache simulation at fixed batch 50. The
     full-resident row is a cache upper bound, not a physical runtime point.
12. `experiments/results/trace_sweep_b50/per_layer.csv`
    - Layer-level hits, refetches, evictions, and traffic reduction.
13. `experiments/results/trace_sweep_b50/trace_diagnostics.png`
    - Visual summary of hit rate, H2D traffic, refetches, and layer variation.
14. `experiments/results/offloaded_smoke_summary.csv`
    - Superseded three-copy integration smokes; do not use their timing.
15. `experiments/results/offloaded_stream1_static_timeline_smoke.json` and
    `experiments/results/offloaded_stream2_static_timeline_smoke.json`
    - One-slot no-prefetch versus two-slot one-ahead validation.
16. `experiments/results/environment.json`, `expert_bytes.json`,
    `h2d_bandwidth.json`, `compute_copy_overlap.json`, and
    `memory_breakdown.csv`
    - Hardware, checkpoint, transfer, overlap, and accounting inputs.

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
2. Full 1,200-request completion results are the official comparison set. The
   earlier 1,273-request trace remains useful historical cache evidence.
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

### Historical fixed-batch-50 cache behavior

- Stream2 performs 33,691,149 Expert fetches.
- Permanent-k reaches 58.97% hit rate at k=64 and 84.98% at k=96.
- Quota-LRU reaches only 0.072% at k=64 and 7.16% at k=96.
- The Quota result is consistent with cyclic ascending-ID scans over a
  batch-level active set approaching all 128 Experts. It is batch-dependent, not
  a universal statement that Quota-LRU is ineffective.
- Full-resident B=50 is not physical because measured Bmax is 41. Treat it as a
  cache upper bound only.
- Read the resident-first, bypass, window-frequency, and random-order controls
  before attributing the baseline miss rate to Quota-LRU generally.

### Natural-routing diagnostic

- Same-prompt real-prefill has 103/1,536 positional mismatches.
- At route-row level, 51/192 rows differ; 40 are order-only and 11 change the
  active Expert set.
- Static-zero KV is a different context and is non-comparable with the reference
  routing trace.
- Forced IDs use natural runtime router weights. Relative policy comparisons are
  supported when final-logits hashes agree, but exact reference equivalence is not.

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

The official completion comparison uses the first 1,200 eligible evaluation rows.
Never describe it as a 1,273- or 2,048-request completion run.

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
6. Report uncertainty from the 1,200-request sample and incomplete intermediate
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
