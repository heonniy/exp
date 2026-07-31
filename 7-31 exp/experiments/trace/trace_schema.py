from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RoutingTrace:
    conversation_ids: np.ndarray
    forced_output_ids: np.ndarray
    routing_expert_ids: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_requests(self) -> int:
        return int(self.routing_expert_ids.shape[0])

    @property
    def output_tokens(self) -> int:
        return int(self.routing_expert_ids.shape[1])

    @property
    def num_layers(self) -> int:
        return int(self.routing_expert_ids.shape[2])

    @property
    def top_k(self) -> int:
        return int(self.routing_expert_ids.shape[3])

    def validate(self, num_experts: int = 128) -> None:
        routing = self.routing_expert_ids
        output = self.forced_output_ids
        ids = self.conversation_ids
        if routing.ndim != 4:
            raise ValueError("routing_expert_ids must have [request, token, layer, top_k]")
        if output.ndim != 2:
            raise ValueError("forced_output_ids must have [request, token]")
        if len(ids) != routing.shape[0] or output.shape[:2] != routing.shape[:2]:
            raise ValueError("request/token dimensions do not agree")
        if routing.dtype != np.uint8:
            raise ValueError("routing_expert_ids must use uint8")
        if output.dtype != np.int32:
            raise ValueError("forced_output_ids must use int32")
        if routing.size and int(routing.max()) >= num_experts:
            raise ValueError("routing trace contains an out-of-range Expert ID")
        if routing.size and int(routing.min()) < 0:
            raise ValueError("routing trace contains a negative Expert ID")
        if len(set(str(item) for item in ids.tolist())) != len(ids):
            raise ValueError("conversation IDs must be unique")
        sorted_ids = np.sort(routing, axis=-1)
        if routing.shape[-1] > 1 and np.any(np.diff(sorted_ids, axis=-1) == 0):
            raise ValueError("top-k routing contains a duplicate Expert")

    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.routing_expert_ids.tobytes(order="C"))
        hasher.update(self.forced_output_ids.tobytes(order="C"))
        for conversation_id in self.conversation_ids:
            hasher.update(str(conversation_id).encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    def first_requests(self, count: int) -> "RoutingTrace":
        if not 0 < count <= self.num_requests:
            raise ValueError("requested trace prefix is outside available requests")
        metadata = dict(self.metadata)
        metadata["source_trace_sha256"] = self.digest()
        metadata["request_prefix_count"] = count
        metadata.pop("trace_sha256", None)
        return RoutingTrace(
            conversation_ids=self.conversation_ids[:count].copy(),
            forced_output_ids=self.forced_output_ids[:count].copy(),
            routing_expert_ids=self.routing_expert_ids[:count].copy(),
            metadata=metadata,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.validate(int(self.metadata.get("num_experts", 128)))
        metadata = dict(self.metadata)
        metadata.update(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "num_requests": self.num_requests,
                "output_tokens": self.output_tokens,
                "num_layers": self.num_layers,
                "top_k": self.top_k,
                "trace_sha256": self.digest(),
            }
        )
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    conversation_ids=self.conversation_ids,
                    forced_output_ids=self.forced_output_ids,
                    routing_expert_ids=self.routing_expert_ids,
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True), dtype=np.str_
                    ),
                )
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "RoutingTrace":
        with np.load(path, allow_pickle=False) as archive:
            trace = cls(
                conversation_ids=archive["conversation_ids"].copy(),
                forced_output_ids=archive["forced_output_ids"].astype(
                    np.int32, copy=True
                ),
                routing_expert_ids=archive["routing_expert_ids"].astype(
                    np.uint8, copy=True
                ),
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
        trace.validate(int(trace.metadata.get("num_experts", 128)))
        expected = trace.metadata.get("trace_sha256")
        if expected and trace.digest() != expected:
            raise ValueError(f"{path}: trace digest mismatch")
        return trace


def concatenate_parts(parts: list[Path], output: Path, metadata: dict[str, Any]) -> None:
    if not parts:
        raise ValueError("no routing trace parts found")
    conversation_ids: list[str] = []
    forced_ids: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    for part in parts:
        with np.load(part, allow_pickle=False) as archive:
            conversation_ids.append(str(archive["conversation_id"].item()))
            forced_ids.append(archive["forced_output_ids"].astype(np.int32))
            routes.append(archive["routing_expert_ids"].astype(np.uint8))
    trace = RoutingTrace(
        conversation_ids=np.asarray(conversation_ids, dtype=np.str_),
        forced_output_ids=np.stack(forced_ids),
        routing_expert_ids=np.stack(routes),
        metadata=metadata,
    )
    trace.save(output)
