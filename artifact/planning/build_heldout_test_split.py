#!/usr/bin/env python3
"""Build a more balanced held-out test split from full-task plan examples.

This split is stricter than ``artifact_request_pool.json`` in two ways:
1. it excludes selector-subtask-derived samples and keeps only full tasks
2. it fixes a dedicated test subset that should be excluded from training,
   prototype-bank construction, and decoder-rule tuning

Because the artifact only ships a limited number of non-gearset1 full tasks,
the split is best-effort balanced rather than perfectly uniform.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = THIS_DIR.parent
DATA_DIR = ARTIFACT_ROOT / "data"
PYRAMID_INPUT = DATA_DIR / "pyramids" / "plan_examples_v2.json"
REQUEST_INPUT = DATA_DIR / "requests" / "artifact_request_pool.json"
REQUEST_OUTPUT = DATA_DIR / "requests" / "test_requests_heldout_full_12.json"
META_OUTPUT = DATA_DIR / "requests" / "test_requests_heldout_full_12_metadata.json"
SPLIT_OUTPUT = DATA_DIR / "datasets" / "plan_predictor_v3_split.json"


# Best-effort target matrix over full-task-only samples.
TARGET_MATRIX: dict[str, dict[str, int]] = {
    "gearset": {"easy": 0, "medium": 0, "hard": 1, "very_hard": 0},
    "gearset1": {"easy": 1, "medium": 1, "hard": 2, "very_hard": 2},
    "chair": {"easy": 0, "medium": 1, "hard": 1, "very_hard": 1},
    "lamp": {"easy": 0, "medium": 0, "hard": 1, "very_hard": 1},
}


def _load_requests() -> list[dict[str, Any]]:
    return json.loads(REQUEST_INPUT.read_text(encoding="utf-8"))


def _load_examples() -> list[dict[str, Any]]:
    return json.loads(PYRAMID_INPUT.read_text(encoding="utf-8"))["examples"]


def _is_full_task(sample_id: str) -> bool:
    return "_subtask_" not in sample_id


def _eligible_full_requests() -> list[dict[str, Any]]:
    return [item for item in _load_requests() if _is_full_task(item["sample_id"])]


def _select_test_requests(
    requests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_domain_diff: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in requests:
        by_domain_diff[item["domain"]][item["difficulty"]].append(item)

    for domain in by_domain_diff:
        for bucket in by_domain_diff[domain]:
            by_domain_diff[domain][bucket].sort(key=lambda x: x["sample_id"])

    selected: list[dict[str, Any]] = []
    audit: dict[str, list[str]] = {}
    for domain, buckets in TARGET_MATRIX.items():
        for difficulty, k in buckets.items():
            pool = by_domain_diff[domain][difficulty]
            chosen = pool[:k]
            if len(chosen) != k:
                raise ValueError(
                    f"Not enough samples for {domain}/{difficulty}: "
                    f"need {k}, have {len(pool)}"
                )
            selected.extend(chosen)
            audit[f"{domain}:{difficulty}"] = [item["sample_id"] for item in chosen]

    selected.sort(key=lambda x: x["sample_id"])
    return selected, audit


def build_heldout_split() -> dict[str, Any]:
    all_requests = _load_requests()
    all_examples = _load_examples()
    eligible = _eligible_full_requests()
    selected, audit = _select_test_requests(eligible)

    selected_ids = {item["sample_id"] for item in selected}
    remaining_ids = [item["sample_id"] for item in all_requests if item["sample_id"] not in selected_ids]

    example_ids = {example["sample_id"] for example in all_examples}
    missing = sorted(selected_ids - example_ids)
    if missing:
        raise ValueError(f"Selected IDs missing from plan examples: {missing}")

    by_domain = Counter(item["domain"] for item in selected)
    by_diff = Counter(item["difficulty"] for item in selected)
    by_domain_diff: dict[str, Counter[str]] = defaultdict(Counter)
    for item in selected:
        by_domain_diff[item["domain"]][item["difficulty"]] += 1

    actual_matrix = {
        domain: {
            bucket: by_domain_diff[domain].get(bucket, 0)
            for bucket in ("easy", "medium", "hard", "very_hard")
        }
        for domain in ("gearset", "gearset1", "chair", "lamp")
    }

    report = {
        "description": (
            "Strict held-out test split built from full-task-only plan examples. "
            "The selected sample IDs should be excluded from predictor training, "
            "prototype-bank construction, and decoder/repair tuning."
        ),
        "input_examples": "artifact/data/pyramids/plan_examples_v2.json",
        "input_requests": "artifact/data/requests/artifact_request_pool.json",
        "selection_policy": {
            "full_task_only": True,
            "best_effort_balanced_matrix": TARGET_MATRIX,
            "selection_order": "deterministic sample_id sort within each domain/difficulty bucket",
        },
        "eligible_full_task_count": len(eligible),
        "heldout_test_size": len(selected),
        "actual_domain_difficulty_matrix": actual_matrix,
        "counts": {
            "by_domain": dict(by_domain),
            "by_difficulty": dict(by_diff),
        },
        "selected_audit": audit,
        "selected_sample_ids": [item["sample_id"] for item in selected],
        "remaining_non_test_sample_ids": remaining_ids,
    }

    REQUEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REQUEST_OUTPUT.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    META_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    SPLIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUTPUT.write_text(
        json.dumps(
            {
                "dataset_name": "plan_predictor_v3_split",
                "policy": "full-task-only held-out test split",
                "test": [item["sample_id"] for item in selected],
                "non_test_pool": remaining_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build_heldout_split()
    print(
        json.dumps(
            {
                "eligible_full_task_count": report["eligible_full_task_count"],
                "heldout_test_size": report["heldout_test_size"],
                "by_domain": report["counts"]["by_domain"],
                "by_difficulty": report["counts"]["by_difficulty"],
                "request_output": str(REQUEST_OUTPUT),
                "meta_output": str(META_OUTPUT),
                "split_output": str(SPLIT_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
