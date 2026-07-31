# H100 4K/256 implementation progress

All GPU measurements in this directory used physical GPU 0 through
`scripts/gpu0.sh`.

## Verified foundations

- GPU: NVIDIA H100 80GB HBM3, 85,028,372,480 HBM bytes.
- Qwen3-30B-A3B dense/non-Expert weights: 3,082,186,752 bytes.
- One Expert: 9,437,184 bytes; all 6,144 Experts have the same size.
- Logical BF16 KV: 98,304 bytes/token, or 427,819,008 bytes/request at
  4,352 peak tokens.
- One pinned 9 MiB H2D copy: 0.1779 ms median, 49.4 GiB/s.
- Continuous pinned H2D: 50.5 GiB/s.

Theoretical Bmax currently uses zero fixed workspace plus a common 2 GiB safety
margin. It ranges from 186 at `k=0` to 50 at `k=128`; real-runtime Bmax probing is
implemented separately and must be used for the primary operating curve.

## Strict LMSYS workload

- Calibration: 256 requests.
- Evaluation: 1,273 requests.
- Every stored row has exactly 4,096 real input tokens and 256 assistant tokens.
- Calibration/evaluation conversation IDs are disjoint.
- The local snapshot has a strict shortfall of 775 evaluation rows; no rows were
  duplicated or padded.

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

## Implementation invariants

- No global LRU implementation exists.
- Active Experts execute in ascending Expert ID order.
- One compute stream and one H2D copy stream are used.
- Main mode uses two transient slots and depth-1 current-layer prefetch.
- The micro-ablation uses one transient slot and depth-0 fetch/wait/compute.
- Different Experts are never grouped, batched, or concurrently executed.
- `k=128` is emitted once as `full_resident`.
