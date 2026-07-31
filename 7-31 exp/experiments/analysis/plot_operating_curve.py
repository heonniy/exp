from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "stream2": "#64748b",
    "permanent_k": "#2563eb",
    "quota_lru_k": "#d97706",
    "full_resident": "#334155",
}


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot measured HBM operating curves and runtime breakdown."
    )
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--bmax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = _read(args.runtime)
    bmax = _read(args.bmax)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for policy, values in _series(runtime).items():
        ordered = sorted(values, key=lambda row: int(row["k"]))
        k = [int(row["k"]) for row in ordered]
        axes[0, 0].plot(
            k,
            [float(row["decode_tokens_per_second"]) for row in ordered],
            marker="o",
            label=policy,
            color=COLORS.get(policy),
        )
        axes[1, 0].plot(
            k,
            [
                int(row["expert_h2d_bytes"])
                / max(1, int(row["generated_tokens"]))
                / (1024**2)
                for row in ordered
            ],
            marker="o",
            label=policy,
            color=COLORS.get(policy),
        )
    for policy, values in _series(bmax).items():
        ordered = sorted(values, key=lambda row: int(row["k"]))
        axes[0, 1].plot(
            [int(row["k"]) for row in ordered],
            [int(row["measured_bmax"]) for row in ordered],
            marker="o",
            label=policy,
            color=COLORS.get(policy),
        )

    timed = [
        row
        for row in runtime
        if row.get("timeline_events_enabled", "").lower() == "true"
    ]
    labels = [f"{row['policy']}\nk={row['k']}" for row in timed]
    bottom = [0.0] * len(timed)
    categories = [
        ("Attention", "attention_ms", "#0891b2"),
        ("Router", "router_ms", "#7c3aed"),
        ("Expert compute", "expert_compute_ms", "#16a34a"),
        ("Exposed H2D", "exposed_h2d_stall_ms", "#dc2626"),
        ("Other dense/host/idle", "other_dense_host_idle_ms", "#94a3b8"),
    ]
    for label, field, color in categories:
        values = [float(row[field]) for row in timed]
        axes[1, 1].bar(labels, values, bottom=bottom, label=label, color=color)
        bottom = [old + value for old, value in zip(bottom, values)]

    axes[0, 0].set(
        title="HBM operating curve",
        xlabel="Resident Experts / layer (k)",
        ylabel="Generated tokens / second",
    )
    axes[0, 1].set(
        title="Measured maximum feasible batch",
        xlabel="Resident Experts / layer (k)",
        ylabel="Bmax",
    )
    axes[1, 0].set(
        title="Expert H2D traffic",
        xlabel="Resident Experts / layer (k)",
        ylabel="MiB / generated token",
    )
    axes[1, 1].set(title="Representative runtime breakdown", ylabel="Milliseconds")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
