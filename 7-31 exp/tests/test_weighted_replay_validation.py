import numpy as np
import torch

from experiments.validation.validate_weighted_replay import ReferenceRoutingReplay


def test_reference_replay_forces_recorded_ids_and_weights() -> None:
    ids = np.asarray([[[[3, 1]], [[2, 0]]]], dtype=np.uint8)
    weights = np.asarray([[[[0.75, 0.25]], [[0.6, 0.4]]]], dtype=np.float32)
    replay = ReferenceRoutingReplay(ids, weights)
    replay.decode_step = 0

    forced_ids, forced_weights = replay.force(
        0,
        torch.tensor([[9, 8]], dtype=torch.int64),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )

    torch.testing.assert_close(forced_ids, torch.tensor([[3, 1]]))
    torch.testing.assert_close(forced_weights, torch.tensor([[0.75, 0.25]]))
    assert not replay.natural_id_exact[0, 0]
    assert not replay.natural_weight_exact[0, 0]


def test_reference_replay_preserves_natural_prefill_before_decode() -> None:
    ids = np.asarray([[[[3, 1]]]], dtype=np.uint8)
    weights = np.asarray([[[[0.75, 0.25]]]], dtype=np.float32)
    replay = ReferenceRoutingReplay(ids, weights)
    natural_ids = torch.tensor([[9, 8]])
    natural_weights = torch.tensor([[0.5, 0.5]])

    actual_ids, actual_weights = replay.force(
        0, natural_ids, natural_weights
    )

    assert actual_ids is natural_ids
    assert actual_weights is natural_weights
