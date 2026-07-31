from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trace-sweep cache behavior.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.summary)
    series: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        series[row["policy"]].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {
        "stream2": "#64748b",
        "permanent_k": "#2563eb",
        "quota_lru_k": "#d97706",
        "full_resident": "#334155",
    }
    for policy, values in series.items():
        ordered = sorted(values, key=lambda row: int(row["k"]))
        x = [int(row["k"]) for row in ordered]
        axes[0].plot(
            x,
            [float(row["hit_rate"]) for row in ordered],
            marker="o",
            label=policy,
            color=colors.get(policy),
        )
        axes[1].plot(
            x,
            [
                float(row["h2d_bytes_per_generated_token"]) / (1024**2)
                for row in ordered
            ],
            marker="o",
            label=policy,
            color=colors.get(policy),
        )
    axes[0].set(title="Layer-local Expert cache hit rate", xlabel="Resident Experts/layer", ylabel="Hit rate")
    axes[1].set(
        title="Expert H2D traffic",
        xlabel="Resident Experts/layer",
        ylabel="MiB / generated token",
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()

