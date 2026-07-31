import pytest

from experiments.benchmark.validate_bmax_256 import (
    assert_closed_monotonic_boundary,
)


def _probe(feasible: bool) -> dict:
    return {"feasible": feasible}


def test_closed_bmax_boundary_requires_b_minus_one_b_and_b_plus_one() -> None:
    assert_closed_monotonic_boundary(
        {41: _probe(True), 42: _probe(True), 43: _probe(False)}, 42
    )


def test_nonmonotonic_lower_failure_is_rejected() -> None:
    with pytest.raises(AssertionError, match="Bmax-1"):
        assert_closed_monotonic_boundary(
            {127: _probe(False), 128: _probe(True), 129: _probe(False)}, 128
        )


def test_feasible_bmax_plus_one_is_rejected() -> None:
    with pytest.raises(AssertionError, match="Bmax\\+1"):
        assert_closed_monotonic_boundary(
            {9: _probe(True), 10: _probe(True), 11: _probe(True)}, 10
        )
