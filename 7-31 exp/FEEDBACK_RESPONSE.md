# Feedback incorporation and completion run

This note maps the four review findings to code and to the new 4K/256 completion
experiment. The official validation prefix is 1,200 requests. The remaining 73
eligible local rows are not used to enlarge that comparison set.

## 1. Permanent score

The legacy `presence` score is explicitly retained as token-assignment frequency.
The new `batch_step_union_presence` score counts an Expert once per:

```text
batch wave × decode step × layer
```

This is the H2D fetch unit used by the simulator. The completion run selects
Permanent Experts with this method at the actual runtime batch size. Token
frequency remains a baseline rather than being described as fetch-optimal.

Implementation: `experiments/trace/select_permanent.py`.

## 2. Quota-LRU sensitivity controls

The trace simulator now records access order, admission policy, seed, admissions,
bypasses, evictions, and H2D for these controls:

- ascending-ID, always-admit LRU (original baseline);
- resident-hit-first ordering;
- miss bypass once a layer quota is full;
- no admission;
- recent-window batch-step frequency admission;
- random Expert order with seeds 731, 732, and 733.

These controls change only layer-local Quota behavior. They do not introduce a
global cache. The real-runtime primary Quota curve remains the original
ascending-ID, always-admit policy so it can be interpreted alongside the
sensitivity controls rather than silently changing the baseline.

The full 1,200-request B=40 control completed with the following hit rates:

| k | Permanent union | Ascending admit | Resident-first | Miss bypass | Random order |
|---:|---:|---:|---:|---:|---:|
| 8 | 7.87% | 0.00% | 6.92% | 6.98% | 0.32–0.33% |
| 32 | 31.15% | ~0.00% | 27.96% | 27.99% | 5.17% |
| 64 | 60.62% | 0.09% | 56.22% | 55.53% | 23.37% |
| 96 | 86.14% | 16.38% | 84.91% | 81.20% | 64.29–64.30% |

The three random seeds have little within-control variance, while changing the
ordering/admission rule changes hit rate by tens of percentage points. This
confirms the review concern rather than treating the original Quota curve as a
policy-independent result.

Implementation: `experiments/runtime/policies.py`,
`experiments/trace/simulator.py`, and
`experiments/benchmark/run_residency_sweep.py`.

## 3. Physical fixed batch

The old B=50 trace result is labeled cache simulation only. In particular,
full-resident B=50 is a cache upper bound because its measured Bmax is 41. The
completion runner uses B=40 as the common physical fixed batch and refuses to run
if B=40 exceeds any measured policy/k Bmax.

The measured-Bmax workload and common-B=40 workload each process 1,200 requests,
256 decode steps, and the exact final partial wave. No B=50 runtime point is used
as physical evidence.

Implementation: `experiments/benchmark/run_4k256_completion.py` and the Bmax
annotations in `run_residency_sweep.py`.

## 4. Natural routing mismatch

Mismatch is now decomposed into position mismatch, row mismatch, active-set
mismatch, and order-only mismatch for every decode-step/layer.

The same-prompt real-prefill diagnostic measured 103/1,536 positional mismatches.
At the token-layer row level, 51/192 rows differed, but 40 of those 51 were
order-only and only 11/192 changed the active Expert set. The static-zero KV
fixture changes the context and produced set mismatch in all 48 first-step layer
rows, so static-zero natural mismatch is explicitly marked non-comparable.

Forced IDs still use natural runtime router weights. Matching final-logits hashes
across residency policies supports a fair relative comparison, but does not prove
exact numerical equivalence to the original full-model reference.

Tracked summary: `experiments/results/natural_routing_diagnostic_summary.json`.

## Completion matrix

For k = 0, 2, 4, 8, 16, 32, 48, 64, 96, 128, with Permanent and Quota at every
intermediate k, the resumable GPU-0 runner performs:

1. real measured Bmax;
2. 1,200-request × 256-step workload at measured Bmax;
3. the same workload at common physical B=40;
4. Expert H2D, exposed stall, Expert compute, attention, router, and residual
   timing breakdown in a separately labeled two-wave instrumented profile;
5. cold-start, KV setup, and steady decode makespan as separate fields.

The 1,200-request throughput and makespan runs do not enable per-Expert CUDA
timeline events. Those events materially perturb B=40 execution, so the common
B=40 two-wave profile is evidence for component diagnosis only and is excluded
from throughput and makespan comparisons.

Run it with:

```bash
./scripts/gpu0.sh .venv/bin/python \
  -m experiments.benchmark.run_4k256_completion
```

Each atomic JSON output is a restart checkpoint. Existing completed outputs are
skipped on rerun.

## 5. Single-buffer Expert H2D correction

The original executor allocated three GPU tensors and enqueued separate
gate/up/down H2D copies, while the pure bandwidth benchmark copied one contiguous
9 MiB buffer. That made the benchmark and runtime transfer granularity differ.

The host-pinned cache and every GPU Expert slot now use one contiguous 9,437,184
byte buffer. gate/up/down are views used only by GEMM; one Expert fetch enqueues
exactly one H2D `copy_`. A GPU-0 smoke recorded 384 Expert fetches and 384 H2D copy
operations, with the same final-logits digest as the pre-change execution.

All pre-change executor timing/overlap measurements are superseded. Cache trace
counts and HBM byte accounting remain structurally useful, but runtime timing is
remeasured by the completion run.

## 6. Prefetch performance revalidation

The old 50.4% overlap value did not establish a performance benefit. Its
prefetch-OFF and ON wall times were 13.912 and 13.888 seconds, respectively,
while roughly 13.1 seconds in each run came from projection-level host staging.
It was also one unreplicated cold run per mode using three H2D copies per Expert.

The corrected benchmark reuses one model and one layer-slab pinned host store,
issues one H2D copy per Expert, warms every mode, rotates execution order, and
checks identical final-logits digests. Uninstrumented wall-time results are:

| Batch | Repeats | OFF | ON copy-first | ON compute-first |
|---:|---:|---:|---:|---:|
| 1 | 3 | 727.57 ms | 672.20 ms (-7.61%) | 670.98 ms (-7.78%) |
| 40 | 9 | 5.345 s | 4.678 s (-12.47%) | 4.699 s (-12.09%) |

The old 3.3x compute increase is not reproduced after the single-buffer and
shared-process corrections. At B=40, per-Expert event profiling itself expanded
wall time from roughly 4.7--5.4 seconds to 9.0--37.6 seconds. Therefore overlap,
compute, and exposed-stall event totals are labeled instrumented diagnostics and
never substituted for uninstrumented wall performance.

Both ON submission orders have a real wall-time benefit. Their B=40 median
difference is only 0.44%, smaller than run-to-run dispersion. `compute_first` is
the primary default because it submits the current Expert compute before the
future H2D and avoids giving DMA priority over the work on the critical path.

Tracked summary:
`experiments/results/prefetch_revalidation_summary.json`.
