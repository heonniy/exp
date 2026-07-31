from experiments.benchmark.checkpoint_layout import EXPERT_PATTERN


def test_qwen_expert_tensor_name_parsing() -> None:
    match = EXPERT_PATTERN.match(
        "model.layers.47.mlp.experts.127.down_proj.weight"
    )
    assert match is not None
    assert match.groupdict() == {
        "layer": "47",
        "expert": "127",
        "projection": "down_proj",
        "parameter": "weight",
    }

