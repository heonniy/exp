#!/usr/bin/env python3
"""Generate the primary figures for Exp1/Exp2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import per_layer_cosine  # noqa: E402
from src.sig_store import SignatureStore  # noqa: E402


def fig_e1_within_distribution(exp1_pairs: pd.DataFrame, out: Path):
    windows = ["first_64", "first_128"]
    datasets = sorted(exp1_pairs["dataset"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(11, 4.5), sharey=True)
    if len(windows) == 1:
        axes = [axes]
    for ax, win in zip(axes, windows):
        data = [
            exp1_pairs[
                (exp1_pairs["dataset"] == d) & (exp1_pairs["analysis_window"] == win)
            ]["cosine_mean_layers"].to_numpy()
            for d in datasets
        ]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(datasets) + 1))
        ax.set_xticklabels(datasets)
        ax.set_title(f"Within-prefix expert cosine ({win})")
        ax.set_ylabel("mean per-layer cosine")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Figure E1-1: Within-prefix expert similarity distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_e1_heatmap(store: SignatureStore, dataset: str, window: str, out: Path):
    sub = store.subset(dataset, window)
    sub = sub[sub["valid_window"]]
    # pick the first prefix group with a full complement of members
    grp = None
    for pid, g in sub.groupby("prefix_id"):
        if len(g) >= 3:
            grp = g
            break
    if grp is None:
        return
    members = grp.sort_values("request_id").to_dict("records")
    n = len(members)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(
                np.mean(
                    per_layer_cosine(
                        store.normalized(members[i]["key"]),
                        store.normalized(members[j]["key"]),
                    )
                )
            )
            M[i, j] = M[j, i] = c
    labels = [m["request_id"].split(":")[-1] for m in members]
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title(f"E1-2 heatmap: {members[0]['prefix_id']} ({window})")
    fig.colorbar(im, ax=ax, label="mean per-layer cosine")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_e2_scatter(nn: pd.DataFrame, out: Path, window: str = "first_128"):
    sub = nn[nn["analysis_window"] == window]
    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(datasets), 1)))
    for d, c in zip(datasets, colors):
        s = sub[sub["dataset"] == d]
        ax.scatter(s["best_same_similarity"], s["best_cross_similarity"],
                   s=22, alpha=0.6, label=d, color=c)
    lims = [0, 1]
    ax.plot(lims, lims, "k--", alpha=0.6, label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("best same-prefix similarity")
    ax.set_ylabel("best cross-prefix similarity")
    ax.set_title(f"E2-1: best same vs best cross ({window})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_e2_cross_better(metrics: dict, out: Path):
    rows = list(metrics.values())
    windows = ["first_64", "first_128"]
    datasets = sorted({r["dataset"] for r in rows})
    x = np.arange(len(datasets))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for k, win in enumerate(windows):
        vals = []
        for d in datasets:
            key = f"{d}|{win}"
            vals.append(metrics.get(key, {}).get("cross_better_rate", 0.0))
        ax.bar(x + (k - 0.5) * width, vals, width, label=win)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Cross-Better Rate")
    ax.set_ylim(0, 1)
    ax.set_title("E2-2: Cross-Better Rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="outputs/metrics")
    ap.add_argument("--output-dir", default="outputs/figures")
    ap.add_argument("--npz", default="outputs/metrics/signatures.npz")
    ap.add_argument("--meta", default="outputs/metrics/signatures_meta.parquet")
    args = ap.parse_args()

    md = Path(args.metrics_dir)
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    exp1 = pd.read_parquet(md / "exp1_within_prefix_pairs.parquet")
    fig_e1_within_distribution(exp1, od / "exp1_within_prefix_similarity.png")

    store = SignatureStore(args.npz, args.meta)
    for d in store.datasets():
        fig_e1_heatmap(store, d, "first_128", od / f"exp1_example_heatmap_{d}.png")

    nn_path = md / "exp2_cross_prefix_nn.parquet"
    if nn_path.exists():
        nn = pd.read_parquet(nn_path)
        if len(nn):
            fig_e2_scatter(nn, od / "exp2_best_same_vs_cross_scatter.png")
    import json

    metrics_path = md / "exp2_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        if metrics:
            fig_e2_cross_better(metrics, od / "exp2_cross_better_rate.png")

    print("FIGURES_DONE_OK ->", sorted(p.name for p in od.glob("*.png")))


if __name__ == "__main__":
    main()
