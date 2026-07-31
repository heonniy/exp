from experiments.runtime.policies import (
    FullResidentPolicy,
    PermanentPolicy,
    QuotaLRUPolicy,
    Stream2Policy,
)
from experiments.runtime.prefetch_scheduler import (
    first_future_nonresident,
    ordered_active_experts,
)


def test_stream2_never_reuses_expert() -> None:
    policy = Stream2Policy(2, 4)
    assert not policy.access(0, 1, 0).hit
    assert not policy.access(0, 1, 1).hit
    assert policy.resident_counts() == (0, 0)


def test_permanent_never_admits_a_miss() -> None:
    policy = PermanentPolicy(2, 4, 1, [[1], [2]])
    assert policy.access(0, 1, 0).hit
    assert not policy.access(0, 2, 1).hit
    assert policy.resident_counts() == (1, 1)


def test_oracle_permanent_policy_has_distinct_label() -> None:
    policy = PermanentPolicy(
        1, 4, 1, [[1]], name="permanent_oracle"
    )
    assert policy.name == "permanent_oracle"


def test_quota_lru_is_strictly_layer_local() -> None:
    policy = QuotaLRUPolicy(2, 8, 2)
    assert policy.access(0, 1, 0).admitted
    assert policy.access(0, 2, 1).admitted
    assert policy.access(0, 1, 2).hit
    result = policy.access(0, 3, 3)
    assert result.evicted_expert == 2
    assert policy.resident_counts() == (2, 0)
    policy.access(1, 7, 4)
    assert policy.resident_counts() == (2, 1)


def test_full_resident_has_no_miss() -> None:
    policy = FullResidentPolicy(2, 4)
    for layer in range(2):
        for expert in range(4):
            assert policy.access(layer, expert, 0).hit


def test_current_layer_one_ahead_selection() -> None:
    active = ordered_active_experts([5, 2, 5, 1])
    assert active == (1, 2, 5)
    residents = {2}
    assert first_future_nonresident(active, 0, residents.__contains__) == 5
    assert first_future_nonresident(active, 0, residents.__contains__, {5}) is None
