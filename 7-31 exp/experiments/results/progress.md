# H100 4K/256 implementation progress

> Historical note: this file describes the original 1,273-request/B=50 phase.
> The corrected official comparison uses a 1,200-request prefix, batch-step union
> Permanent scoring, Quota sensitivity controls, and physical common B=40. Read
> `FEEDBACK_RESPONSE.md` and `completion_4k256_1200/` first.
>
> The executor measurements in this historical file used three projection-level
> H2D enqueues per Expert. They are superseded by the single-contiguous-buffer
> implementation. Do not use the old runtime timing/overlap values in the final
> decision; the completion run remeasures them.

All GPU measurements in this directory used physical GPU 0 through
`scripts/gpu0.sh`.

## Verified foundations

- GPU: NVIDIA H100 80GB HBM3, 85,028,372,480 HBM bytes.
- CUDA async engine count: 3; PCIe link: generation 5, width 16.
- Qwen3-30B-A3B dense/non-Expert weights: 3,082,186,752 bytes.
- One Expert: 9,437,184 bytes; all 6,144 Experts have the same size.
- Logical BF16 KV: 98,304 bytes/token, or 427,819,008 bytes/request at
  4,352 peak tokens.
- One pinned 9 MiB H2D copy: 0.1779 ms median, 49.4 GiB/s.
- Continuous pinned H2D: 50.5 GiB/s.

Theoretical Bmax currently uses zero fixed workspace plus a common 2 GiB safety
margin. It ranges from 186 at `k=0` to 50 at `k=128`; real-runtime Bmax probing is
implemented separately and must be used for the primary operating curve.

Real-runtime extreme probes now show:

| Endpoint | Theoretical Bmax | Measured Bmax | Peak allocated at Bmax |
|---|---:|---:|---:|
| stream2, k=0 | 186 | 157 | 83,784,424,448 bytes |
| full-resident, k=128 | 50 | 41 | 83,830,419,456 bytes |

The measured KV batch cost of full residency is therefore 116 requests. Batch 158
for stream2 and batch 42 for full residency failed in the real attention path,
despite fitting the zero-workspace theoretical accounting. Full-resident successful
probes had zero decode Expert H2D fetches.

## Strict LMSYS workload

- Calibration: 256 requests.
- Evaluation: 1,273 requests.
- Every stored row has exactly 4,096 real input tokens and 256 assistant tokens.
- Calibration/evaluation conversation IDs are disjoint.
- The local snapshot has a strict shortfall of 775 evaluation rows; no rows were
  duplicated or padded.

## Full evaluation routing trace

- Shape: `[1273, 256, 48, 8]`.
- Routing dtype: `uint8`; forced-token dtype: `int32`.
- Trace SHA-256:
  `2dd67a486095614f1ec2071bbf0ac2cfc34fe097def9062bf1c9511f2dbaadac`.
- The trace validates unique conversation IDs, in-range/non-duplicate top-8 IDs,
  and exact workload forced-token identity.

At fixed batch 50, stream2 makes 33,691,149 Expert fetches. The key cache result is
that ascending-ID Quota-LRU thrashes when the batch-level active set approaches all
128 Experts:

| Policy | k | Hit rate | H2D reduction vs stream2 |
|---|---:|---:|---:|
| permanent-k | 8 | 7.56% | 7.56% |
| quota-LRU-k | 8 | 0.00% | 0.00% |
| permanent-k | 64 | 58.97% | 58.97% |
| quota-LRU-k | 64 | 0.072% | 0.072% |
| permanent-k | 96 | 84.98% | 84.98% |
| quota-LRU-k | 96 | 7.16% | 7.16% |
| full-resident | 128 | 100.00% | 100.00% |

This is batch-dependent: the earlier batch-1 runtime smoke showed useful
Quota-LRU-k=8 reuse, while the fixed-batch-50 trace does not.

## Real offload smoke result

The first evaluation request was replayed for four decode tokens with identical
forced Expert IDs. These are integration smoke measurements, not the final
256-token operating curve.

| Policy | k | tok/s | H2D fetches | H2D reduction | Throughput gain |
|---|---:|---:|---:|---:|---:|
| stream2 | 0 | 4.959 | 1,536 | 0.00% | 1.000x |
| permanent-k | 8 | 5.753 | 1,298 | 15.49% | 1.160x |
| quota-LRU-k | 8 | 6.073 | 1,126 | 26.69% | 1.225x |

Quota-LRU kept exactly eight residents in every layer after warm-up, reported 742
evictions and 1,126 logical ownership swaps, and performed zero D2D admission
copies. Natural routing differed from the recorded forced trace for 8.53% of
assignments before the forced-ID override; effective execution used the recorded
IDs for every policy.

The historical Static-KV one-token timeline micro-ablation used projection-level
pinned staging only as a functional check. Its 13-second wall time was dominated
by CPU staging, and its three-copy transfer granularity is now superseded. The
old GPU-event interval result was:

| Prefetch depth | Slots | H2D duration | Compute-copy overlap | Overlap ratio |
|---:|---:|---:|---:|---:|
| 0 | 1 | 138.39 ms | 0.00 ms | 0.0% |
| 1 | 2 | 142.38 ms | 71.78 ms | 50.4% |

That 50.4% value produced no meaningful wall-time improvement and must not be
used as performance evidence. The single-buffer shared-process revalidation is:

| Batch | Repeats | OFF | ON copy-first | ON compute-first |
|---:|---:|---:|---:|---:|
| 1 | 3 | 727.57 ms | 672.20 ms | 670.98 ms |
| 40 | 9 | 5.345 s | 4.678 s | 4.699 s |

At B=40, prefetch reduces uninstrumented median wall time by 12.1--12.5%.
Per-Expert timeline events materially perturb this workload, so full-workload
throughput is measured without them and breakdowns are stored as separate
instrumented profiles.

Both paths produced the same final-logits SHA-256:
`cc101da1e5b190768df8a9524554f77c9a9a0879e3717ea271c078331dd1fafe`.

## Implementation invariants

- No global LRU implementation exists.
- Active Experts execute in ascending Expert ID order.
- One compute stream and one H2D copy stream are used.
- Main mode uses two transient slots and depth-1 current-layer prefetch.
- The micro-ablation uses one transient slot and depth-0 fetch/wait/compute.
- Different Experts are never grouped, batched, or concurrently executed.
- `k=128` is emitted once as `full_resident`.
