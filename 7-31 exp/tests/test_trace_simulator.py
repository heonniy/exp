from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.trace.collect_forced_routing_trace import _router_topk
from experiments.runtime.policies import (
    FullResidentPolicy,
    PermanentPolicy,
    Stream2Policy,
)
from experiments.trace.select_permanent import select_topk
from experiments.trace.simulator import simulate
from experiments.trace.trace_schema import RoutingTrace


def synthetic_trace() -> RoutingTrace:
    routing = np.asarray(
        [
            [[[0, 1]], [[0, 2]]],
            [[[0, 1]], [[0, 2]]],
        ],
        dtype=np.uint8,
    )
    return RoutingTrace(
        conversation_ids=np.asarray(["a", "b"]),
        forced_output_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int32),
        routing_expert_ids=routing,
        metadata={"num_experts": 4},
    )


def test_trace_round_trip(tmp_path: Path) -> None:
    trace = synthetic_trace()
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = RoutingTrace.load(path)
    assert loaded.digest() == trace.digest()
    np.testing.assert_array_equal(loaded.routing_expert_ids, trace.routing_expert_ids)


def test_stream2_refetch_accounting() -> None:
    trace = synthetic_trace()
    result = simulate(trace, Stream2Policy(1, 4), 100, batch_size=2)
    assert result.fetches == 4
    assert result.compulsory_loads == 3
    assert result.refetches == 1
    assert result.h2d_bytes == 400


def test_permanent_selection_and_full_resident() -> None:
    trace = synthetic_trace()
    selected = select_topk(trace, 1)
    assert selected.tolist() == [[0]]
    permanent = PermanentPolicy(1, 4, 1, selected.tolist())
    result = simulate(trace, permanent, 100, batch_size=2)
    assert result.hits == 2
    assert result.fetches == 2

    full = simulate(trace, FullResidentPolicy(1, 4), 100, batch_size=2)
    assert full.fetches == 0
    assert full.hit_rate == 1.0


def test_batched_router_logits_keep_requests_separate() -> None:
    outputs = SimpleNamespace(
        router_logits=(
            torch.tensor([[0.0, 3.0, 2.0], [5.0, 0.0, 4.0]]),
            torch.tensor([[7.0, 1.0, 6.0], [0.0, 9.0, 8.0]]),
        )
    )
    routes = _router_topk(outputs, expected_layers=2, top_k=2, batch_size=2)
    assert routes.shape == (2, 2, 2)
    assert routes[0].tolist() == [[1, 2], [0, 2]]
    assert routes[1].tolist() == [[0, 2], [1, 2]]
