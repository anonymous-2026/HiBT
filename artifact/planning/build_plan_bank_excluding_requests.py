#!/usr/bin/env python3
"""Build a prototype bank after excluding request-overlapping base problems.

This script is a strict isolation wrapper around ``build_plan_bank.py``.
It removes all dataset rows whose normalized base problem id overlaps with the
normalized base problem ids found in a request file such as ``test_requests_60``.

Normalization removes common derived-task suffixes including:

- ``_subtask_<N>``
- ``_subset``
- ``_goal_already_satisfied``

The output can therefore become empty if the request benchmark fully covers the
available train pool at the base-problem level.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from planning.build_plan_bank import (
    _load_frozen_builder,
    _read_jsonl,
    build_bank,
)


LOGGER = logging.getLogger("build_plan_bank_excluding_requests")

DERIVED_SUFFIX_PATTERNS = (
    re.compile(r"_subtask_\d+$"),
    re.compile(r"_subset$"),
    re.compile(r"_goal_already_satisfied$"),
)


def normalize_base_problem_id(problem_id: str) -> str:
    normalized = problem_id
    for pattern in DERIVED_SUFFIX_PATTERNS:
        normalized = pattern.sub("", normalized)
    return normalized


def load_request_base_ids(requests_file: Path) -> set[str]:
    requests = json.loads(requests_file.read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise ValueError(f"Expected list in requests file: {requests_file}")

    base_ids: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        raw_id = request.get("problem_id") or request.get("sample_id")
        if not raw_id:
            continue
        base_ids.add(normalize_base_problem_id(str(raw_id)))
    return base_ids


def row_problem_id(row: dict[str, Any]) -> str:
    return str(row.get("problem_id") or row["main_id"])


def filter_rows_by_request_overlap(
    rows: list[dict[str, Any]],
    excluded_base_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row in rows:
        base_id = normalize_base_problem_id(row_problem_id(row))
        if base_id in excluded_base_ids:
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed


def build_report(
    dataset_jsonl: Path,
    requests_file: Path,
    kept_rows: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    excluded_base_ids: set[str],
) -> dict[str, Any]:
    removed_main_ids = [row["main_id"] for row in removed_rows]
    removed_base_ids = sorted(
        {
            normalize_base_problem_id(row_problem_id(row))
            for row in removed_rows
        }
    )
    kept_base_ids = sorted(
        {
            normalize_base_problem_id(row_problem_id(row))
            for row in kept_rows
        }
    )
    return {
        "report_version": "test60-bank-exclusion-v1",
        "dataset_jsonl": str(dataset_jsonl),
        "requests_file": str(requests_file),
        "excluded_request_base_ids": sorted(excluded_base_ids),
        "input_rows": len(kept_rows) + len(removed_rows),
        "kept_rows": len(kept_rows),
        "removed_rows": len(removed_rows),
        "kept_base_ids": kept_base_ids,
        "removed_base_ids": removed_base_ids,
        "removed_main_ids": removed_main_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prototype bank after excluding request-overlapping base problems."
    )
    parser.add_argument(
        "--builder-config",
        required=True,
        help="Path to the Builder YAML config.",
    )
    parser.add_argument(
        "--builder-checkpoint",
        required=True,
        help="Path to the frozen Builder checkpoint.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        required=True,
        help="Path to the exported predictor dataset JSONL.",
    )
    parser.add_argument(
        "--requests-file",
        default=str(ARTIFACT_ROOT / "data" / "requests" / "test_requests_60.json"),
        help="Benchmark request file whose base problems should be excluded.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the filtered prototype bank (.pt).",
    )
    parser.add_argument(
        "--filtered-jsonl-output",
        default=None,
        help="Optional path to save the filtered rows as JSONL.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional path to save the exclusion report as JSON.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for running the frozen Builder.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    builder_config_path = Path(args.builder_config)
    builder_checkpoint_path = Path(args.builder_checkpoint)
    dataset_jsonl_path = Path(args.dataset_jsonl)
    requests_file_path = Path(args.requests_file)
    output_path = Path(args.output)
    filtered_jsonl_output = Path(args.filtered_jsonl_output) if args.filtered_jsonl_output else None
    report_output = Path(args.report_output) if args.report_output else None

    rows = _read_jsonl(dataset_jsonl_path)
    excluded_base_ids = load_request_base_ids(requests_file_path)
    kept_rows, removed_rows = filter_rows_by_request_overlap(rows, excluded_base_ids)

    LOGGER.info("Loaded %d dataset rows from %s", len(rows), dataset_jsonl_path)
    LOGGER.info("Loaded %d excluded base ids from %s", len(excluded_base_ids), requests_file_path)
    LOGGER.info(
        "After exclusion: kept=%d removed=%d",
        len(kept_rows),
        len(removed_rows),
    )

    if filtered_jsonl_output is not None:
        filtered_jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with filtered_jsonl_output.open("w", encoding="utf-8") as handle:
            for row in kept_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOGGER.info("Saved filtered dataset rows to %s", filtered_jsonl_output)

    report = build_report(
        dataset_jsonl=dataset_jsonl_path,
        requests_file=requests_file_path,
        kept_rows=kept_rows,
        removed_rows=removed_rows,
        excluded_base_ids=excluded_base_ids,
    )
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Saved exclusion report to %s", report_output)

    builder = _load_frozen_builder(
        builder_config_path=builder_config_path,
        builder_checkpoint_path=builder_checkpoint_path,
        device=args.device,
    )
    bank = build_bank(rows=kept_rows, builder=builder, device=args.device)
    bank["exclusion_report"] = report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)
    LOGGER.info("Saved filtered prototype bank with %d entries to %s", bank["num_entries"], output_path)


if __name__ == "__main__":
    main()
