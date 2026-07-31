import torch

from experiments.runtime.expert_slot import contiguous_projection_views
from experiments.runtime.host_expert_store import (
    PinnedExpertStore,
    pack_projection_tensors,
)
from experiments.runtime.offloaded_model import OffloadedQwenExperts, build_routed_tokens
from experiments.runtime.serial_expert_executor import SerialExpertExecutor
from experiments.runtime.serial_expert_executor import interval_overlap_ms


class FakeEngine:
    def execute(self, layer_id, hidden_states, top_k_index, top_k_weights):
        assert layer_id == 7
        return hidden_states + 1


def test_offloaded_expert_module_has_no_parameters() -> None:
    module = OffloadedQwenExperts(7, FakeEngine())
    hidden = torch.zeros(2, 4)
    output = module(
        hidden,
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1),
    )
    assert not list(module.parameters())
    assert torch.equal(output, torch.ones_like(hidden))


def test_checkpoint_expert_tensor_name() -> None:
    assert (
        PinnedExpertStore.tensor_name(47, 127, "down_proj")
        == "model.layers.47.mlp.experts.127.down_proj.weight"
    )


def test_routed_token_axis_order() -> None:
    expert_ids = torch.tensor([[2, 5], [5, 3], [2, 3]])
    weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    routed = build_routed_tokens(expert_ids, weights)
    assert routed[2].token_indices.tolist() == [0, 2]
    assert torch.equal(routed[2].routing_weights, torch.tensor([0.2, 0.4]))
    assert routed[5].token_indices.tolist() == [0, 1]
    assert torch.equal(routed[5].routing_weights, torch.tensor([0.8, 0.7]))


def test_timeline_overlap_uses_interval_intersection() -> None:
    copies = [(0.0, 4.0), (6.0, 10.0)]
    computes = [(2.0, 7.0), (8.0, 9.0)]
    assert interval_overlap_ms(copies, computes) == 4.0


def test_expert_projections_pack_into_one_buffer_with_zero_copy_views() -> None:
    source = {
        "gate_proj": torch.arange(6, dtype=torch.float32).view(2, 3),
        "up_proj": torch.arange(6, 12, dtype=torch.float32).view(2, 3),
        "down_proj": torch.arange(12, 18, dtype=torch.float32).view(3, 2),
    }
    packed = pack_projection_tensors(source, pin_memory=False)
    views = contiguous_projection_views(
        packed,
        {"gate_proj": (2, 3), "up_proj": (2, 3), "down_proj": (3, 2)},
    )
    assert packed.numel() == 18
    assert all(view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr() for view in views.values())
    for name in source:
        assert torch.equal(views[name], source[name])


def test_packed_views_produce_identical_mlp_output() -> None:
    generator = torch.Generator().manual_seed(731)
    source = {
        "gate_proj": torch.randn(3, 5, generator=generator),
        "up_proj": torch.randn(3, 5, generator=generator),
        "down_proj": torch.randn(5, 3, generator=generator),
    }
    packed = pack_projection_tensors(source, pin_memory=False)
    views = contiguous_projection_views(
        packed,
        {"gate_proj": (3, 5), "up_proj": (3, 5), "down_proj": (5, 3)},
    )
    inputs = torch.randn(7, 5, generator=generator)
    expected = SerialExpertExecutor._mlp(inputs, source)
    actual = SerialExpertExecutor._mlp(inputs, views)
    assert torch.equal(actual, expected)
