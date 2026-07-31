from experiments.benchmark.benchmark_prefetch_4way import MODES, rotated_modes


def test_rotated_modes_covers_each_mode_at_each_sequence_position() -> None:
    orders = [rotated_modes(repeat) for repeat in range(len(MODES))]
    assert all(set(order) == set(MODES) for order in orders)
    for position in range(len(MODES)):
        assert {order[position] for order in orders} == set(MODES)


def test_rotated_modes_wraps_after_full_cycle() -> None:
    assert rotated_modes(len(MODES)) == MODES
