from __future__ import annotations

from collections.abc import Callable, Iterable


def ordered_active_experts(expert_ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(int(expert_id) for expert_id in expert_ids)))


def first_future_nonresident(
    ordered_experts: tuple[int, ...],
    after_position: int,
    is_resident: Callable[[int], bool],
    already_staged: set[int] | None = None,
) -> int | None:
    staged = already_staged or set()
    for expert_id in ordered_experts[after_position + 1 :]:
        if not is_resident(expert_id) and expert_id not in staged:
            return expert_id
    return None

