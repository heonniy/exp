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
SHORT = {
    "stream2": "S",
    "permanent_k": "P",
    "quota_lru_k": "Q",
    "full_resident": "F",
}


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    return grouped


def _operating_plot(path: Path, rows: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = [
        ("measured_bmax", "Physical Bmax", "requests"),
        ("at_bmax_tokens_per_second", "Throughput at physical Bmax", "generated tokens/s"),
        ("common_tokens_per_second", "Throughput at common B=40", "generated tokens/s"),
        ("common_h2d_bytes_per_generated_token", "Common-B Expert H2D traffic", "bytes/generated token"),
    ]
    for policy, policy_rows in _group(rows).items():
        ordered = sorted(policy_rows, key=lambda item: int(item["k"]))
        ks = [int(row["k"]) for row in ordered]
        for axis, (field, title, ylabel) in zip(axes.flat, fields):
            axis.plot(
                ks,
                [float(row[field]) for row in ordered],
                marker="o",
                label=policy,
                color=COLORS[policy],
            )
            axis.set(title=title, xlabel="resident Experts/layer (k)", ylabel=ylabel)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout(pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _profile_plot(path: Path, rows: list[dict]) -> None:
    labels = [f"{SHORT[row['policy']]}{row['k']}" for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    fields = [
        ("wall_seconds", "Decode wall time", "seconds"),
        ("exposed_h2d_stall_ms", "Exposed Expert H2D stall", "milliseconds"),
        ("expert_compute_ms", "Expert compute", "milliseconds"),
        ("attention_ms", "Attention", "milliseconds"),
    ]
    for axis, (suffix, title, ylabel) in zip(axes.flat, fields):
        cold = [float(row[f"profile_cold_wave_{suffix}"]) for row in rows]
        steady = [float(row[f"profile_steady_wave_{suffix}"]) for row in rows]
        axis.bar(
            [position - width / 2 for position in positions],
            cold,
            width,
            label="cold/warmup wave",
            color="#94a3b8",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            steady,
            width,
            label="steady wave",
            color="#2563eb",
        )
        axis.set(title=title, ylabel=ylabel, xticks=positions, xticklabels=labels)
        axis.tick_params(axis="x", labelrotation=55)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Intrusive common-B40 profile: component attribution only")
    figure.tight_layout(pad=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the validated 4K/256 completion operating matrix."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--operating-output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.summary_csv)
    if not rows:
        raise ValueError("completion summary is empty")
    _operating_plot(args.operating_output, rows)
    _profile_plot(args.profile_output, rows)
    print(args.operating_output)
    print(args.profile_output)


if __name__ == "__main__":
    main()
