# 40 GiB HBM: 4K prefill + 128 decode, 200-request result

## Result

Permanent k=8 is the minimum-E2E measured point in this reduced sweep. At its
physical Bmax=54, the 200-request real-prefill plus decode workload finishes in
653.552 s, 52.143 s (7.389%) faster than two-slot streaming at Bmax=60
(705.695 s). Decode throughput improves from 38.635 to 41.598 generated
tokens/s (+7.668%). Prompt prefill throughput improves from 19,014 to 21,482
prompt tokens/s (+12.98%).

k=8 remains at four waves for 200 requests, as do k=0 through k=8. At k=16,
Bmax falls to 48 and the workload needs five waves; from that point onward the
additional-wave cost dominates the continued reduction in Expert fetches. The
largest feasible k=80 cuts decode fetch count by about 50.7% versus streaming,
but Bmax=2 creates 100 waves and makes E2E 3.64x slower.

This is a measured optimum for the specified 40 GiB/4K+128/200-request
operating point, not a claim of a universal optimum.

## Scope and validity

- Physical GPU: 0 only.
- HBM ceiling: exactly 40 GiB = 42,949,672,960 bytes, enforced through the
  PyTorch CUDA allocator. All successful runtime peaks are below the cap.
- Workload: exactly 200 evaluation requests, 4,096 prompt tokens and 128 decode
  tokens per request.
- Policies: `stream2` at k=0 and `permanent_k` for positive k. Quota/LRU is not
  run or included.
- Candidate k: 0, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128. k=96 and k=128 are
  automatically excluded because reduced-HBM memory accounting produces no
  feasible provisional batch.
- Bmax is closed with real 4K prefill plus all 128 decode steps at B, and a
  failed B+1 boundary.
- Performance numbers come only from uninstrumented 200-request runs.
- Phase breakdown comes from separate, intrusive, one-wave timeline profiles;
  their wall time and throughput are not used as performance evidence.
- The Permanent manager and packed GPU buffers are initialized before real
  prefill and retained into decode. Positive-k runs record Permanent hits in
  both phases.
- Prefill and decode each validate exactly one contiguous H2D copy operation
  per Expert fetch. The superseded three-tensor copy path is not used.

The validator output is `summary.json`; the complete one-row-per-k table is
`summary.csv`; `operating_curve.png` is a visual summary.

## Physical Bmax and phase-separated performance

All times below are the uninstrumented makespan for the same 200 requests. E2E
is checked to equal prefill + decode, including the last partial wave.

| k | Policy | Bmax | Waves | Prefill (s) | Prompt tok/s | Decode (s) | Gen tok/s | E2E (s) | E2E change vs k=0 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | stream2 | 60 | 4 | 43.085 | 19,014 | 662.610 | 38.635 | 705.695 | baseline |
| 2 | Permanent | 59 | 4 | 38.769 | 21,130 | 620.892 | 41.231 | 659.661 | 6.523% faster |
| 4 | Permanent | 57 | 4 | 37.876 | 21,628 | 617.223 | 41.476 | 655.099 | 7.170% faster |
| 8 | Permanent | 54 | 4 | 38.134 | 21,482 | 615.418 | 41.598 | 653.552 | **7.389% faster** |
| 16 | Permanent | 48 | 5 | 40.578 | 20,188 | 656.942 | 38.968 | 697.520 | 1.158% faster |
| 32 | Permanent | 37 | 6 | 41.558 | 19,712 | 794.573 | 32.219 | 836.132 | 18.483% slower |
| 48 | Permanent | 25 | 8 | 43.871 | 18,673 | 826.102 | 30.989 | 869.973 | 23.279% slower |
| 64 | Permanent | 13 | 16 | 54.732 | 14,968 | 1,165.298 | 21.969 | 1,220.030 | 72.883% slower |
| 80 | Permanent | 2 | 100 | 129.552 | 6,323 | 2,438.956 | 10.496 | 2,568.508 | 263.969% slower |

The k=4 prefill is marginally fastest, while k=8 has the fastest decode and the
lowest combined E2E. The distinction is only 1.547 s E2E between k=4 and k=8,
so replication is needed before treating that small separation as robust.

## Expert traffic and Permanent use

| k | Prefill fetches | Prefill Permanent hits | Decode fetches | Decode Permanent hits |
|---:|---:|---:|---:|---:|
| 0 | 24,525 | 0 | 2,488,403 | 0 |
| 2 | 24,141 | 384 | 2,468,915 | 48,616 |
| 4 | 23,757 | 768 | 2,447,788 | 97,641 |
| 8 | 22,991 | 1,536 | 2,371,655 | 195,896 |
| 16 | 26,763 | 3,840 | 2,386,555 | 457,013 |
| 32 | 27,510 | 9,216 | 2,274,026 | 1,130,897 |
| 48 | 30,485 | 18,432 | 2,049,264 | 2,155,889 |
| 64 | 48,145 | 49,145 | 1,739,529 | 4,272,260 |
| 80 | 198,879 | 377,032 | 1,227,420 | 8,055,031 |

Higher k consistently reduces total decode fetches. Prefill fetches reverse and
increase at high k because smaller Bmax creates many more waves; each wave has
its own compulsory non-permanent loads. This is why fetch count alone is not the
deployment objective.

## Intrusive component profiles

The following tables normalize each one-wave profile by phase token count so
different Bmax values are not compared using raw totals. H2D, exposed stall,
Expert compute, and attention are microseconds per phase token. H2D overlap is
`(total H2D - exposed H2D) / total H2D`. These component sums are event totals,
not wall-time partitions; concurrent activity can overlap.

### Prefill profile

| k | Profile B | H2D us/prompt tok | Exposed us/tok | H2D overlap | Expert compute us/tok | Attention us/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 4.461 | 0.478 | 89.29% | 24.154 | 15.346 |
| 2 | 59 | 4.459 | 0.466 | 89.55% | 24.202 | 15.962 |
| 4 | 57 | 4.548 | 0.498 | 89.06% | 24.392 | 15.237 |
| 8 | 54 | 4.655 | 0.570 | 87.76% | 24.970 | 15.449 |
| 16 | 48 | 4.907 | 0.697 | 85.80% | 26.176 | 15.059 |
| 32 | 37 | 5.521 | 1.334 | 75.84% | 31.709 | 15.082 |
| 48 | 25 | 6.905 | 2.924 | 57.65% | 48.750 | 15.004 |
| 64 | 13 | 10.580 | 5.151 | 51.31% | 104.456 | 15.954 |
| 80 | 2 | 47.484 | 24.153 | 49.13% | 504.502 | 29.647 |

### Decode profile

| k | Profile B | H2D us/generated tok | Exposed us/tok | H2D overlap | Expert compute us/tok | Attention us/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 15,254.8 | 12,014.7 | 21.24% | 3,381.9 | 1,289.0 |
| 2 | 59 | 15,225.2 | 11,911.3 | 21.77% | 3,470.2 | 1,299.3 |
| 4 | 57 | 15,375.3 | 11,964.1 | 22.19% | 3,586.8 | 1,318.4 |
| 8 | 54 | 15,487.6 | 12,015.2 | 22.42% | 3,907.3 | 1,357.3 |
| 16 | 48 | 15,684.1 | 11,906.4 | 24.09% | 4,768.9 | 1,388.8 |
| 32 | 37 | 15,595.5 | 11,100.9 | 28.82% | 7,528.0 | 1,544.9 |
| 48 | 25 | 14,883.8 | 9,625.7 | 35.33% | 12,343.3 | 1,879.4 |
| 64 | 13 | 14,594.9 | 8,254.2 | 43.44% | 19,714.6 | 2,722.1 |
| 80 | 2 | 7,966.9 | 3,788.6 | 52.45% | 33,993.1 | 13,935.9 |

The corrected stream2 decode profile attributes only 21.24% of H2D duration to
overlap, not the historical 50.4%. The old figure came from the superseded
three-copy/staging smoke. Prefill has much higher overlap (89.29% at k=0), but
decode H2D remains substantially exposed. Because timeline events are intrusive,
these ratios explain the run; they do not replace the uninstrumented 7.389% E2E
result.

The old prefetch-ON compute inflation is not present as a comparable runtime
metric here: packed one-copy execution is used throughout, and compute events
are collected only in the separately labeled intrusive profiles. The established
uninstrumented prefetch ablation remains the evidence for prefetch benefit.

## Cold-start and steady-state separation

`Cold init` includes model load, the 6,144-Expert host-pinned preload, and policy
initialization. It is deliberately excluded from the 200-request phase/E2E
makespan above and varies with host page/cache state, so it should not be used to
rank policies. Warmup and steady figures below are uninstrumented decode times
for a full Bmax wave; the final partial wave is included in E2E but not in the
steady full-wave median.

| k | Cold init (s) | Warmup prefill (s) | Warmup decode (s) | Steady full-wave count | Steady decode median (s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 258.120 | 11.718 | 165.681 | 2 | 186.017 |
| 2 | 177.155 | 11.509 | 162.576 | 2 | 163.853 |
| 4 | 43.749 | 11.131 | 159.127 | 2 | 160.658 |
| 8 | 62.760 | 10.676 | 156.432 | 2 | 156.666 |
| 16 | 43.617 | 9.763 | 145.126 | 3 | 150.937 |
| 32 | 437.668 | 8.880 | 191.813 | 4 | 130.007 |
| 48 | 190.154 | 7.901 | 103.883 | 7 | 104.161 |
| 64 | 61.133 | 7.657 | 77.620 | 14 | 74.623 |
| 80 | 45.953 | 5.478 | 25.210 | 99 | 24.283 |

k=32 demonstrates why the split matters: its warmup decode wave is 191.813 s,
while steady full waves have a 130.007 s median. The fixed-workload E2E includes
both states and is therefore the primary decision metric.

## Caveats and decision boundary

- The user-requested reduction to 200 samples waives the original five-repeat
  protocol. The k=4 versus k=8 difference in particular needs replicated runs
  for uncertainty.
- At-Bmax comparison changes batch size and wave count by design. A common batch
  across all feasible k would be only B=2 because of k=80, which would not
  isolate the useful deployment region.
- Decode Expert IDs and recorded router weights are forced so every policy sees
  identical work. Same-prompt natural routing has about a 6.3% Expert-set
  mismatch in these runs; that remains a numerical-drift diagnostic and limits
  claims of exact equivalence to an unconstrained full-model reference.
- Permanent selection uses decode calibration batch-step union presence. The
  same buffers are active during prefill, but the selection is not separately
  optimized for prefill.
- All rows use packed single-buffer Expert transfers. The pure bandwidth
  benchmark's one-copy shape therefore matches the executor's copy granularity.
