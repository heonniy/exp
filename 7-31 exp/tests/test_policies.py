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


def test_quota_resident_hit_first_preserves_hits_before_admission() -> None:
    policy = QuotaLRUPolicy(
        1, 8, 2, access_order="resident_hit_first"
    )
    policy.access(0, 1, 0)
    policy.access(0, 2, 1)
    assert policy.order_active_experts(0, (0, 1, 2)) == (1, 2, 0)


def test_quota_miss_bypass_does_not_thrash_a_full_layer() -> None:
    policy = QuotaLRUPolicy(
        1, 8, 2, admission_policy="miss_bypass_when_full"
    )
    policy.access(0, 1, 0)
    policy.access(0, 2, 1)
    result = policy.access(0, 3, 2)
    assert result.bypassed
    assert not result.admitted
    assert policy.access(0, 1, 3).hit


def test_quota_no_admission_is_an_explicit_control() -> None:
    policy = QuotaLRUPolicy(1, 8, 2, admission_policy="no_admission")
    result = policy.access(0, 1, 0)
    assert result.bypassed
    assert policy.resident_counts() == (0,)


def test_quota_window_frequency_requires_repeated_layer_step_presence() -> None:
    policy = QuotaLRUPolicy(
        1,
        8,
        2,
        admission_policy="window_frequency",
        window_size=4,
        window_min_frequency=2,
    )
    policy.begin_layer_step(0, (3,))
    assert policy.access(0, 3, 0).bypassed
    policy.begin_layer_step(0, (3,))
    assert policy.access(0, 3, 1).admitted


def test_quota_random_order_is_seed_reproducible() -> None:
    first = QuotaLRUPolicy(
        1, 8, 2, access_order="random_expert_order", random_seed=9
    )
    second = QuotaLRUPolicy(
        1, 8, 2, access_order="random_expert_order", random_seed=9
    )
    assert first.order_active_experts(0, (0, 1, 2, 3)) == second.order_active_experts(
        0, (0, 1, 2, 3)
    )


def test_quota_router_order_preserves_first_occurrence() -> None:
    policy = QuotaLRUPolicy(1, 8, 2, access_order="router_order")
    assert policy.order_active_experts(0, (5, 2, 7, 1)) == (5, 2, 7, 1)


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
