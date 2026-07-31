from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.analysis.plot_operating_curve import COLORS


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trace cache behavior, refetches, and a per-layer point."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-layer", type=Path, required=True)
    parser.add_argument("--detail-policy", default="quota_lru_k")
    parser.add_argument("--detail-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = _read(args.summary)
    per_layer = _read(args.per_layer)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary:
        grouped[row["policy"]].append(row)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for policy, values in grouped.items():
        ordered = sorted(values, key=lambda row: int(row["k"]))
        k = [int(row["k"]) for row in ordered]
        color = COLORS.get(policy)
        axes[0, 0].plot(
            k,
            [float(row["hit_rate"]) for row in ordered],
            marker="o",
            label=policy,
            color=color,
        )
        axes[0, 1].plot(
            k,
            [
                float(row["h2d_bytes_per_generated_token"]) / (1024**2)
                for row in ordered
            ],
            marker="o",
            label=policy,
            color=color,
        )
        axes[1, 0].plot(
            k,
            [int(row["compulsory_loads"]) for row in ordered],
            marker="o",
            linestyle=":",
            color=color,
            label=f"{policy}: compulsory",
        )
        axes[1, 0].plot(
            k,
            [int(row["refetches"]) for row in ordered],
            marker="o",
            color=color,
            label=f"{policy}: refetch",
        )

    detail = sorted(
        (
            row
            for row in per_layer
            if row["policy"] == args.detail_policy and int(row["k"]) == args.detail_k
        ),
        key=lambda row: int(row["layer_id"]),
    )
    layers = [int(row["layer_id"]) for row in detail]
    hit_rate = [
        int(row["hits"]) / int(row["accesses"]) if int(row["accesses"]) else 0.0
        for row in detail
    ]
    axes[1, 1].plot(layers, hit_rate, color="#2563eb", label="Hit rate")
    twin = axes[1, 1].twinx()
    twin.plot(
        layers,
        [int(row["refetches"]) for row in detail],
        color="#dc2626",
        alpha=0.75,
        label="Refetches",
    )

    axes[0, 0].set(title="Cache hit rate", xlabel="k", ylabel="Hit rate")
    axes[0, 1].set(
        title="Expert H2D traffic", xlabel="k", ylabel="MiB / generated token"
    )
    axes[1, 0].set(
        title="Compulsory load and refetch decomposition",
        xlabel="k",
        ylabel="Expert fetch count",
    )
    axes[1, 1].set(
        title=f"Per-layer: {args.detail_policy}, k={args.detail_k}",
        xlabel="Layer",
        ylabel="Hit rate",
    )
    twin.set_ylabel("Refetch count")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    twin.spines["top"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].legend(frameon=False, loc="upper left")
    twin.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
