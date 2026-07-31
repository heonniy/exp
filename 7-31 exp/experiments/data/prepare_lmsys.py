from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb
from transformers import AutoTokenizer

from experiments.common.config import ExperimentConfig, load_config
from experiments.common.io import atomic_write_json, atomic_write_jsonl


@dataclass
class PreparationStats:
    scanned: int = 0
    no_eligible_target: int = 0
    short_output: int = 0
    short_input: int = 0
    accepted: int = 0


_WORKER_TOKENIZER = None
_WORKER_INPUT_TOKENS = 0
_WORKER_OUTPUT_TOKENS = 0


def _init_worker(model_path: str, input_tokens: int, output_tokens: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_INPUT_TOKENS, _WORKER_OUTPUT_TOKENS
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    _WORKER_INPUT_TOKENS = input_tokens
    _WORKER_OUTPUT_TOKENS = output_tokens


def _worker_build(row: tuple[str, Sequence[Any]]) -> tuple[dict[str, Any] | None, str | None]:
    conversation_id, conversation = row
    return build_fixed_example(
        _WORKER_TOKENIZER,
        str(conversation_id),
        conversation,
        _WORKER_INPUT_TOKENS,
        _WORKER_OUTPUT_TOKENS,
    )


def _token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected an unbatched tokenizer result")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _normalize_messages(conversation: Sequence[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in conversation:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            role, content = item[1], item[0]
        if role not in {"user", "assistant", "system"} or content is None:
            continue
        messages.append({"role": str(role), "content": str(content)})
    return messages


def _candidate_targets(messages: Sequence[dict[str, str]]) -> Iterator[int]:
    for index in range(len(messages) - 1, 0, -1):
        if (
            messages[index]["role"] == "assistant"
            and messages[index - 1]["role"] == "user"
        ):
            yield index


def _encode_recent_history(
    tokenizer: Any,
    history: Sequence[dict[str, str]],
    input_tokens: int,
) -> list[int] | None:
    """Tokenize the newest history first and stop once left truncation is exact.

    Once a suffix exceeds ``input_tokens``, content before that suffix cannot
    affect the final left-truncated token IDs. This avoids tokenizing megabytes of
    old context for pathological LMSYS rows.
    """

    if not history:
        return None
    for start in range(len(history) - 1, -1, -1):
        encoded = tokenizer.apply_chat_template(
            history[start:],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = _token_ids(encoded)
        if len(prompt) >= input_tokens:
            return prompt[-input_tokens:]
    return None


def build_fixed_example(
    tokenizer: Any,
    conversation_id: str,
    conversation: Sequence[Any],
    input_tokens: int,
    output_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build one exact-length teacher-forced example.

    The latest qualifying assistant turn is preferred. If it is too short, an
    earlier assistant target in the same conversation may be used. Truncating
    from the left preserves the final user turn and the generation prompt.
    """

    messages = _normalize_messages(conversation)
    saw_long_output = False
    for target_index in _candidate_targets(messages):
        forced = tokenizer.encode(
            messages[target_index]["content"], add_special_tokens=False
        )
        if len(forced) < output_tokens:
            continue
        saw_long_output = True
        history = messages[:target_index]
        fixed_prompt = _encode_recent_history(tokenizer, history, input_tokens)
        if fixed_prompt is None:
            continue
        fixed_output = [int(token) for token in forced[:output_tokens]]
        return (
            {
                "conversation_id": conversation_id,
                "input_ids": fixed_prompt,
                "forced_output_ids": fixed_output,
                "input_length": len(fixed_prompt),
                "output_length": len(fixed_output),
                "source_target_message_index": target_index,
            },
            None,
        )
    return None, "short_input" if saw_long_output else "short_output"


def _eligible_rows(
    parquet_glob: str,
    language: str,
    seed: int,
    input_tokens: int,
    output_tokens: int,
    max_scan: int | None,
) -> Iterator[tuple[str, Sequence[Any]]]:
    connection = duckdb.connect()
    query = f"""
        SELECT conversation_id, conversation
        FROM read_parquet(?)
        WHERE language = ?
          AND conversation_id IS NOT NULL
          AND conversation IS NOT NULL
          AND NOT coalesce(
            list_has(
              list_transform(openai_moderation, item -> item.flagged),
              true
            ),
            false
          )
          AND list_sum(
            list_transform(conversation, item -> length(item.content))
          ) >= ?
          AND list_count(
            list_filter(
              conversation,
              item -> item.role = 'assistant'
                      AND length(item.content) >= ?
            )
          ) > 0
        QUALIFY row_number() OVER (
          PARTITION BY conversation_id
          ORDER BY hash(conversation_id || ?)
        ) = 1
        ORDER BY hash(conversation_id || ?)
    """
    # The half-length character predicates are deliberately loose. They remove
    # obvious short rows while retaining byte-fallback/code-heavy candidates
    # whose token/character ratio can exceed one.
    parameters: list[Any] = [
        parquet_glob,
        language,
        max(1, input_tokens // 2),
        max(1, output_tokens // 2),
        f":{seed}",
        f":{seed}",
    ]
    if max_scan is not None:
        query += " LIMIT ?"
        parameters.append(max_scan)
    cursor = connection.execute(query, parameters)
    try:
        while rows := cursor.fetchmany(128):
            yield from rows
    finally:
        connection.close()


def prepare(
    config: ExperimentConfig,
    max_scan: int | None = None,
    workers: int = 1,
    allow_shortfall: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    required = (
        config.dataset.calibration_requests + config.dataset.evaluation_requests
    )
    examples: list[dict[str, Any]] = []
    stats = PreparationStats()

    source = _eligible_rows(
        config.dataset.parquet_glob,
        config.dataset.language,
        config.seed,
        config.dataset.input_tokens,
        config.dataset.output_tokens,
        max_scan,
    )
    if workers == 1:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.path,
            local_files_only=True,
            trust_remote_code=False,
        )
        built = (
            build_fixed_example(
                tokenizer,
                str(conversation_id),
                conversation,
                config.dataset.input_tokens,
                config.dataset.output_tokens,
            )
            for conversation_id, conversation in source
        )
        tokenizer_class = type(tokenizer).__name__
        pool = None
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(
            workers,
            initializer=_init_worker,
            initargs=(
                str(config.model.path),
                config.dataset.input_tokens,
                config.dataset.output_tokens,
            ),
        )
        built = pool.imap(_worker_build, source, chunksize=16)
        tokenizer_class = "AutoTokenizer(worker)"

    try:
        for example, rejection in built:
            stats.scanned += 1
            if stats.scanned % 1000 == 0:
                print(
                    f"scanned={stats.scanned:,} accepted={stats.accepted:,}/"
                    f"{required:,}",
                    flush=True,
                )
            if example is None:
                if rejection == "short_input":
                    stats.short_input += 1
                elif rejection == "short_output":
                    stats.short_output += 1
                else:
                    stats.no_eligible_target += 1
                continue
            examples.append(example)
            stats.accepted += 1
            if len(examples) == required:
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if len(examples) < required and not allow_shortfall:
        raise RuntimeError(
            f"found {len(examples)} eligible requests after scanning "
            f"{stats.scanned}; need {required}. Re-run with --allow-shortfall "
            "to preserve all strict-eligible rows without duplication."
        )
    if len(examples) < config.dataset.calibration_requests:
        raise RuntimeError(
            f"only {len(examples)} eligible requests; cannot fill the "
            f"{config.dataset.calibration_requests}-request calibration split"
        )

    output_dir = config.dataset.output_dir
    stem = f"lmsys_{config.dataset.input_tokens // 1024}k{config.dataset.output_tokens}"
    calibration_path = output_dir / f"{stem}_calibration.jsonl"
    evaluation_path = output_dir / f"{stem}_evaluation.jsonl"
    calibration = examples[: config.dataset.calibration_requests]
    evaluation = examples[
        config.dataset.calibration_requests :
        config.dataset.calibration_requests + config.dataset.evaluation_requests
    ]
    atomic_write_jsonl(calibration_path, calibration)
    atomic_write_jsonl(evaluation_path, evaluation)

    conversation_digest = hashlib.sha256(
        "\n".join(item["conversation_id"] for item in examples).encode("utf-8")
    ).hexdigest()
    metadata = {
        "config_name": config.name,
        "model_path": str(config.model.path),
        "tokenizer_class": tokenizer_class,
        "tokenizer_workers": workers,
        "input_tokens": config.dataset.input_tokens,
        "output_tokens": config.dataset.output_tokens,
        "calibration_requests": len(calibration),
        "evaluation_requests": len(evaluation),
        "requested_calibration_requests": config.dataset.calibration_requests,
        "requested_evaluation_requests": config.dataset.evaluation_requests,
        "strict_sample_shortfall": required - len(examples),
        "conversation_id_sha256": conversation_digest,
        "stats": asdict(stats),
        "outputs": {
            "calibration": str(calibration_path),
            "evaluation": str(evaluation_path),
        },
    }
    atomic_write_json(output_dir / f"{stem}_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic fixed-length LMSYS requests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/h100_lmsys_4k256.yaml"),
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        help="Optional smoke-test cap on source conversations scanned.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
        help="Parallel CPU tokenizer workers (default: up to 16).",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help=(
            "Write every strict-eligible unique request when the source dataset "
            "cannot fill the configured split; never duplicates or pads rows."
        ),
    )
    args = parser.parse_args()
    metadata = prepare(
        load_config(args.config),
        max_scan=args.max_scan,
        workers=args.workers,
        allow_shortfall=args.allow_shortfall,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
