# Superseded shared-process Bmax sweep

Do not use this directory for the operating curve.

This first shared-process run created a new compute stream for every probe but
did not clear cuBLAS per-stream workspaces between probes. The accumulated
workspace state lowered the final full-resident result to Bmax=38. A fresh-process
control measured Bmax=41 with the same single-buffer executor, proving that the
drop was a measurement-process artifact rather than an HBM cost of packed Expert
weights.

The active sibling `bmax/` directory is regenerated with
`torch._C._cuda_clearCublasWorkspaces()` and `torch.cuda.empty_cache()` between
probes. Use only that active directory and the independent control
`experiments/results/full_resident_bmax_fresh_single_buffer.json`.
