from __future__ import annotations

from dataclasses import asdict, dataclass


GIB = 1024**3


@dataclass(frozen=True)
class MemoryBreakdown:
    total_hbm_bytes: int
    dense_resident_bytes: int
    fixed_workspace_bytes: int
    safety_margin_bytes: int
    persistent_expert_bytes: int
    transient_expert_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_request: int
    theoretical_bmax: int

    def as_dict(self) -> dict:
        return asdict(self)


def account_memory(
    *,
    total_hbm_bytes: int,
    dense_resident_bytes: int,
    fixed_workspace_bytes: int,
    safety_margin_bytes: int,
    expert_bytes: int,
    num_layers: int,
    k: int,
    transient_slots: int,
    kv_bytes_per_token: int,
    peak_sequence_length: int,
) -> MemoryBreakdown:
    values = {
        "total_hbm_bytes": total_hbm_bytes,
        "dense_resident_bytes": dense_resident_bytes,
        "fixed_workspace_bytes": fixed_workspace_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "expert_bytes": expert_bytes,
        "num_layers": num_layers,
        "k": k,
        "transient_slots": transient_slots,
        "kv_bytes_per_token": kv_bytes_per_token,
        "peak_sequence_length": peak_sequence_length,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("memory accounting inputs cannot be negative")
    persistent = num_layers * k * expert_bytes
    transient = transient_slots * expert_bytes
    kv_budget = (
        total_hbm_bytes
        - dense_resident_bytes
        - fixed_workspace_bytes
        - safety_margin_bytes
        - persistent
        - transient
    )
    per_request = kv_bytes_per_token * peak_sequence_length
    bmax = max(0, kv_budget // per_request) if per_request else 0
    return MemoryBreakdown(
        total_hbm_bytes=total_hbm_bytes,
        dense_resident_bytes=dense_resident_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
        safety_margin_bytes=safety_margin_bytes,
        persistent_expert_bytes=persistent,
        transient_expert_bytes=transient,
        kv_budget_bytes=max(0, kv_budget),
        kv_bytes_per_request=per_request,
        theoretical_bmax=bmax,
    )

