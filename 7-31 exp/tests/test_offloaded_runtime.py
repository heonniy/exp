import torch

from experiments.runtime.host_expert_store import PinnedExpertStore
from experiments.runtime.offloaded_model import OffloadedQwenExperts, build_routed_tokens
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
