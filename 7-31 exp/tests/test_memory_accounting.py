from experiments.benchmark.memory_accounting import account_memory
from experiments.benchmark.measure_kv_bytes import logical_kv_bytes_per_token

import torch


def test_qwen_logical_kv_bytes_per_token() -> None:
    assert logical_kv_bytes_per_token(48, 4, 128, torch.bfloat16) == 98_304


def test_persistent_and_transient_are_separate() -> None:
    result = account_memory(
        total_hbm_bytes=1000,
        dense_resident_bytes=100,
        fixed_workspace_bytes=100,
        safety_margin_bytes=100,
        expert_bytes=10,
        num_layers=2,
        k=3,
        transient_slots=2,
        kv_bytes_per_token=2,
        peak_sequence_length=10,
    )
    assert result.persistent_expert_bytes == 60
    assert result.transient_expert_bytes == 20
    assert result.kv_budget_bytes == 620
    assert result.theoretical_bmax == 31

