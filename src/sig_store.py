"""Loader for the signatures.npz produced by build_signatures.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.signatures import normalized_freq


class SignatureStore:
    def __init__(self, npz_path: str | Path, meta_path: str | Path):
        self._npz = np.load(npz_path)
        self.meta = pd.read_parquet(meta_path)
        self.moe_layer_ids = self._npz["__moe_layer_ids__"].tolist()
        self.num_experts = int(self._npz["__num_experts__"][0])
        self._norm_cache: dict[str, np.ndarray] = {}

    def counts(self, key: str) -> np.ndarray:
        return self._npz[f"c::{key}"]

    def weight_mass(self, key: str) -> np.ndarray:
        return self._npz[f"w::{key}"]

    def normalized(self, key: str) -> np.ndarray:
        if key not in self._norm_cache:
            self._norm_cache[key] = normalized_freq(self.counts(key))
        return self._norm_cache[key]

    def subset(self, dataset: str, window: str) -> pd.DataFrame:
        m = self.meta
        return m[(m["dataset"] == dataset) & (m["window"] == window)].copy()

    def datasets(self) -> list[str]:
        return sorted(self.meta["dataset"].unique().tolist())
