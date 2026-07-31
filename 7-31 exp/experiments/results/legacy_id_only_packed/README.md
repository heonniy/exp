# Legacy packed, ID-only replay results

These files use the packed one-copy Expert implementation, but force only
Expert IDs while retaining natural router weights. The resulting hidden states
and logits are not a correct replay when natural and recorded routes differ.
They are also missing the final 256-step Bmax boundary protocol. They are
retained only for audit history and must not be reused by new sweeps.

