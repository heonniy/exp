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
from experiments.trace.select_permanent import score_experts, select_topk
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
        metadata={
            "num_experts": 4,
            "reference_experts_implementation": "eager",
        },
        routing_expert_weights=np.full(routing.shape, 0.5, dtype=np.float32),
    )


def test_trace_round_trip(tmp_path: Path) -> None:
    trace = synthetic_trace()
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = RoutingTrace.load(path)
    assert loaded.digest() == trace.digest()
    np.testing.assert_array_equal(loaded.routing_expert_ids, trace.routing_expert_ids)
    np.testing.assert_array_equal(
        loaded.routing_expert_weights, trace.routing_expert_weights
    )


def test_trace_request_prefix_has_derived_identity() -> None:
    trace = synthetic_trace()
    prefix = trace.first_requests(1)
    assert prefix.num_requests == 1
    assert prefix.metadata["source_trace_sha256"] == trace.digest()
    assert prefix.digest() != trace.digest()


def test_stream2_refetch_accounting() -> None:
    trace = synthetic_trace()
    result = simulate(trace, Stream2Policy(1, 4), 100, batch_size=2)
    assert result.fetches == 4
    assert result.compulsory_loads == 3
    assert result.refetches == 1
    assert result.h2d_bytes == 400


def test_request_order_seed_is_recorded_and_deterministic() -> None:
    trace = synthetic_trace()
    first = simulate(
        trace, Stream2Policy(1, 4), 100, batch_size=1, request_order_seed=731
    )
    second = simulate(
        trace, Stream2Policy(1, 4), 100, batch_size=1, request_order_seed=731
    )
    assert first.request_order_seed == 731
    assert first.as_dict() == second.as_dict()


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


def test_batch_step_union_presence_matches_fetch_unit() -> None:
    trace = RoutingTrace(
        conversation_ids=np.asarray(["a", "b"]),
        forced_output_ids=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        routing_expert_ids=np.asarray(
            [
                [[(0,)], [(1,)], [(1,)]],
                [[(0,)], [(2,)], [(2,)]],
            ],
            dtype=np.uint8,
        ),
        metadata={"num_experts": 3},
    )
    token_scores = score_experts(trace, "token_frequency")
    union_scores = score_experts(
        trace, "batch_step_union_presence", batch_size=2
    )
    assert token_scores.tolist() == [[2, 2, 2]]
    assert union_scores.tolist() == [[1, 2, 2]]
    assert select_topk(trace, 1, "token_frequency").tolist() == [[0]]
    assert select_topk(
        trace, 1, "batch_step_union_presence", batch_size=2
    ).tolist() == [[1]]


def test_batched_router_logits_keep_requests_separate() -> None:
    outputs = SimpleNamespace(
        router_logits=(
            torch.tensor([[0.0, 3.0, 2.0], [5.0, 0.0, 4.0]]),
            torch.tensor([[7.0, 1.0, 6.0], [0.0, 9.0, 8.0]]),
        )
    )
    routes, weights = _router_topk(
        outputs, expected_layers=2, top_k=2, batch_size=2
    )
    assert routes.shape == (2, 2, 2)
    assert routes[0].tolist() == [[1, 2], [0, 2]]
    assert routes[1].tolist() == [[0, 2], [1, 2]]
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)


def test_legacy_trace_is_readable_but_rejected_for_weighted_replay(
    tmp_path: Path,
) -> None:
    trace = synthetic_trace()
    legacy = RoutingTrace(
        conversation_ids=trace.conversation_ids,
        forced_output_ids=trace.forced_output_ids,
        routing_expert_ids=trace.routing_expert_ids,
        metadata=trace.metadata,
    )
    path = tmp_path / "legacy.npz"
    legacy.save(path)
    loaded = RoutingTrace.load(path)
    assert not loaded.has_routing_weights
    try:
        loaded.validate(4, require_weights=True)
    except ValueError as error:
        assert "legacy ID-only" in str(error)
    else:
        raise AssertionError("legacy trace unexpectedly passed weighted validation")


def test_weighted_trace_requires_eager_serial_reference() -> None:
    trace = synthetic_trace()
    assert trace.require_serial_reference() is trace.routing_expert_weights
    wrong = RoutingTrace(
        conversation_ids=trace.conversation_ids,
        forced_output_ids=trace.forced_output_ids,
        routing_expert_ids=trace.routing_expert_ids,
        routing_expert_weights=trace.routing_expert_weights,
        metadata={"num_experts": 4, "reference_experts_implementation": "grouped_mm"},
    )
    try:
        wrong.require_serial_reference()
    except ValueError as error:
        assert "eager serial" in str(error)
    else:
        raise AssertionError("grouped-mm trace unexpectedly passed serial gate")
