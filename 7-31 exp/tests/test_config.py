from pathlib import Path

import pytest

from experiments.common.config import load_config


ROOT = Path(__file__).parents[1]


def test_primary_config_matches_spec() -> None:
    config = load_config(ROOT / "experiments/configs/h100_lmsys_4k256.yaml")
    assert config.gpu_physical_index == 0
    assert config.peak_sequence_length == 4352
    assert config.model.num_moe_layers == 48
    assert config.model.num_experts_per_layer == 128
    assert config.runtime.transient_expert_slots == 2
    assert config.runtime.global_lru is False


def test_reduced_hbm_config_matches_requested_scope() -> None:
    config = load_config(
        ROOT / "experiments/configs/h100_lmsys_4k128_n200_hbm40.yaml"
    )
    assert config.runtime.effective_hbm_gib == 40
    assert config.dataset.input_tokens == 4096
    assert config.dataset.output_tokens == 128
    assert config.dataset.evaluation_requests == 200
    assert config.policies == ("stream2", "permanent_k")
    assert config.runtime_k == (0, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128)


def test_global_lru_is_rejected(tmp_path: Path) -> None:
    path = ROOT / "experiments/configs/h100_lmsys_4k256.yaml"
    text = path.read_text().replace("global_lru: false", "global_lru: true")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(text)
    with pytest.raises(ValueError, match="global_lru"):
        load_config(invalid)
