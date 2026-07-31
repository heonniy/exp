from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class SplitValidation:
    path: str
    rows: int
    unique_conversation_ids: int
    input_length: int
    output_length: int


def _rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def validate_split(path: Path) -> tuple[SplitValidation, set[str]]:
    count = 0
    ids: set[str] = set()
    input_length: int | None = None
    output_length: int | None = None
    for row in _rows(path):
        count += 1
        conversation_id = str(row["conversation_id"])
        if conversation_id in ids:
            raise ValueError(f"{path}: duplicate conversation_id {conversation_id}")
        ids.add(conversation_id)
        actual_input = len(row["input_ids"])
        actual_output = len(row["forced_output_ids"])
        if row["input_length"] != actual_input:
            raise ValueError(f"{path}: stored input_length does not match list length")
        if row["output_length"] != actual_output:
            raise ValueError(f"{path}: stored output_length does not match list length")
        input_length = input_length if input_length is not None else actual_input
        output_length = output_length if output_length is not None else actual_output
        if actual_input != input_length or actual_output != output_length:
            raise ValueError(f"{path}: non-uniform fixed lengths")
        if not all(isinstance(token, int) and token >= 0 for token in row["input_ids"]):
            raise ValueError(f"{path}: invalid input token ID")
        if not all(
            isinstance(token, int) and token >= 0
            for token in row["forced_output_ids"]
        ):
            raise ValueError(f"{path}: invalid forced output token ID")

    if count == 0:
        raise ValueError(f"{path}: empty split")
    return (
        SplitValidation(
            path=str(path),
            rows=count,
            unique_conversation_ids=len(ids),
            input_length=int(input_length),
            output_length=int(output_length),
        ),
        ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed LMSYS JSONL splits.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()

    calibration, calibration_ids = validate_split(args.calibration)
    evaluation, evaluation_ids = validate_split(args.evaluation)
    overlap = calibration_ids & evaluation_ids
    if overlap:
        raise ValueError(
            f"calibration/evaluation conversation IDs overlap ({len(overlap)} IDs)"
        )
    print(
        json.dumps(
            {
                "calibration": calibration.__dict__,
                "evaluation": evaluation.__dict__,
                "split_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

