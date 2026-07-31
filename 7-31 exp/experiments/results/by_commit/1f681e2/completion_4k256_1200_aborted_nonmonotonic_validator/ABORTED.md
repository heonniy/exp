# Aborted Bmax validation diagnostic

This run is not valid completion evidence. It was stopped after the Quota-LRU
`k=32` validator recorded the non-monotonic sequence below:

- B=127: OOM before decode;
- B=128: 256/256 decode steps succeeded;
- B=129: OOM before decode.

The old validator incorrectly labeled this `boundary_closed=true` because it
checked only B and B+1, not that B-1 also succeeded. The run was interrupted
while starting Permanent `k=48`. Commit `1f681e2` therefore remains a diagnostic
only.

The replacement validator performs CUDA synchronization, Python GC, cuBLAS
workspace cleanup, and allocator cleanup between probes; retries an OOM once;
and rejects any non-monotonic B-1/B/B+1 boundary. The official completion run is
stored under the later fixing commit.
