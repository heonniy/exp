# 40 GiB / 4K+128 / 200-request analysis guide

This is the entry point for judging the reduced-HBM experiment requested after
the original 4K/256 plan. Its scope is deliberately narrower: physical GPU 0,
an allocator-enforced 40 GiB HBM ceiling, 4,096-token real prefill, 128-token
forced-routing decode, and exactly 200 evaluation requests. Only two policies
are in scope: two-slot streaming (`stream2`, k=0) and Permanent residency.
Quota/LRU is excluded.

## Read these files in order

1. `experiments/configs/h100_lmsys_4k128_n200_hbm40.yaml`
   - Authoritative workload, HBM cap, policy list, and candidate k values.
2. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/summary.json`
   - Machine-readable validated aggregate and selected minimum-E2E point.
3. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/summary.csv`
   - One row per feasible k, including Bmax, phase throughput, E2E, cold/steady
     timing, traffic, and instrumented component attribution.
4. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/hbm40_4k128_n200_report.md`
   - Answer-first interpretation, caveats, and comparison tables.
5. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/operating_curve.png`
   - Visual operating curve; confirm numerical claims against the CSV/JSON.
6. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/bmax_prefill_decode/manifest.json`
   - Feasible configurations and the automatic k=96/128 exclusions.
7. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/bmax_prefill_decode/*.json`
   - Real-prefill plus full-128-decode B/B+1 boundary evidence.
8. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/runtime_at_bmax/*.json`
   - Uninstrumented 200-request measurements. These files alone are eligible
     for makespan and throughput comparison.
9. `experiments/results/by_commit/05ed1c7/h100_lmsys_4k128_n200_hbm40gib/profiles_at_bmax/*.json`
   - One-wave intrusive profiles used only for H2D/exposed-stall/compute/
     attention attribution, separately for prefill and decode.

## Required checks

Reject a row before comparing performance unless all of these hold:

- `effective_hbm_cap_bytes == 42949672960` and
  `allocator_hbm_cap_enforced == true`;
- `gpu_physical_index == 0`, `kv_setup == "real_prefill"`, 4,096 prompt tokens
  per request, and 128 generated tokens per request;
- the Bmax file has `boundary_closed == true` and includes real prefill plus all
  128 decode steps;
- runtime files contain 200 requests, have timeline events disabled, and are
  eligible for throughput/makespan comparison;
- profile files have timeline events enabled and are explicitly ineligible for
  throughput/makespan comparison;
- both prefill and decode have exactly one contiguous H2D copy operation per
  Expert fetch;
- Permanent uses `batch_step_union_presence`, has Permanent hits during real
  prefill, and retains the same initialized residency manager into decode;
- E2E wall time equals prefill wall time plus decode wall time.

`experiments.analysis.aggregate_reduced_hbm` enforces these invariants before it
writes the aggregate.

## Decision procedure

Use `fixed_workload_e2e_makespan_seconds` from the uninstrumented runtime as the
primary outcome. It includes every wave and the final partial wave. Then explain
the winning point using, in order:

1. physical Bmax and resulting wave count;
2. prefill wall time and prompt tokens/s;
3. decode wall time and generated tokens/s;
4. Expert fetch reduction and Permanent hits in both phases;
5. one-wave profile totals for H2D, exposed stall, overlap, Expert compute, and
   attention;
6. initialization cold-start and steady full-wave decode timing.

Do not choose the largest k merely because it has fewer fetches. Permanent HBM
reduces KV capacity, so Bmax collapse and additional waves can dominate the
traffic saving. Also do not use the intrusive profile's wall time as throughput
evidence.

## Interpretation boundaries

- The 40 GiB limit is 40 GiB (binary), not 40,000,000,000 decimal bytes.
- This is a reduced 200-request study with the five-repeat requirement waived;
  treat the optimum as a measured point requiring replication for uncertainty,
  not a universal model-wide optimum.
- Decode Expert IDs and router weights are replayed from the recorded trace so
  policies see identical work. Natural-route mismatch remains a numerical-drift
  diagnostic, not evidence that forced-policy comparisons used different work.
- Permanent experts are selected with decode calibration
  `batch_step_union_presence`. The same buffers accelerate prefill, but the
  selection itself is not prefill-optimized.
- A common-batch comparison across every feasible k would be limited to B=2 by
  k=80. This sweep instead answers the deployment-style question at each
  configuration's physical Bmax.
- Large model, workload JSONL, and trace NPZ inputs are local-only. Committed
  results contain configuration identities, commands, git SHAs, and trace
  digests for auditability.
