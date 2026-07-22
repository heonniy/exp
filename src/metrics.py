"""Similarity metrics between two expert signatures.

Primary: per-layer cosine similarity of the normalized activation histograms,
aggregated (mean/median/p10/p90) over MoE layers.
Secondary: top-N expert-set Jaccard, and Jensen-Shannon divergence.
"""

from __future__ import annotations

import numpy as np


def per_layer_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: [L, E] normalized histograms -> [L] cosine per layer.

    Layers where either side is all-zero yield cosine 0.0 (no signal).
    """
    num = (a * b).sum(axis=-1)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb
    out = np.zeros_like(num)
    mask = denom > 0
    out[mask] = num[mask] / denom[mask]
    return out


def cosine_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    cos = per_layer_cosine(a, b)
    return {
        "cosine_mean_layers": float(np.mean(cos)),
        "cosine_median_layers": float(np.median(cos)),
        "cosine_p10_layers": float(np.percentile(cos, 10)),
        "cosine_p90_layers": float(np.percentile(cos, 90)),
    }


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(per_layer_cosine(a, b)))


def topn_jaccard(counts_a: np.ndarray, counts_b: np.ndarray, n: int) -> float:
    """Mean over layers of the Jaccard between the top-N activated expert sets."""
    L = counts_a.shape[0]
    vals = []
    for li in range(L):
        a_top = _topn_set(counts_a[li], n)
        b_top = _topn_set(counts_b[li], n)
        if not a_top and not b_top:
            continue
        inter = len(a_top & b_top)
        union = len(a_top | b_top)
        vals.append(inter / union if union else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _topn_set(counts_row: np.ndarray, n: int) -> set[int]:
    nz = np.nonzero(counts_row)[0]
    if len(nz) == 0:
        return set()
    # top-n by count (ties broken by expert id via stable sort on -count)
    order = np.argsort(-counts_row[nz], kind="stable")
    top = nz[order[:n]]
    return set(int(x) for x in top)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-layer Jensen-Shannon divergence (base 2) of normalized histograms."""
    L = p.shape[0]
    out = np.zeros(L)
    for li in range(L):
        out[li] = _js_row(p[li], q[li])
    return out


def _js_row(p: np.ndarray, q: np.ndarray) -> float:
    ps = p.sum()
    qs = q.sum()
    if ps == 0 or qs == 0:
        return 0.0
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))
