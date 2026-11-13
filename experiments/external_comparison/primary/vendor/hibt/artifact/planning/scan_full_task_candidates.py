#!/usr/bin/env python3
"""Scan all full-task planning candidates and summarize difficulty capacity.

This utility re-scans the full-task subset from ``plan_examples_v2.json``,
re-applies the current difficulty labeling logic, and writes a compact report
that can be used to design a balanced benchmark later.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_artifact_request_pool import _complexity_bucket, _domain_from_sample_id


THIS_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = THIS_DIR.parent
DATA_DIR = ARTIFACT_ROOT / "data"
PYRAMID_INPUT = DATA_DIR / "pyramids" / "plan_examples_v2.json"
REPORT_OUTPUT = DATA_DIR / "requests" / "full_task_candidate_scan.json"

DIFFICULTIES = ("easy", "medium", "hard", "very_hard")
DOMAINS = ("gearset", "gearset1", "chair", "lamp")


def _call(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"


def _is_full_task(sample_id: str) -> bool:
    return "_subtask_" not in sample_id


def _load_examples() -> list[dict[str, Any]]:
    return json.loads(PYRAMID_INPUT.read_text(encoding="utf-8"))["examples"]


def _action_calls(example: dict[str, Any]) -> list[str]:
    action_items = example["instance"]["pyramid"][3]["items"]
    return [_call(item["name"], item["args"]) for item in action_items]


def build_report() -> dict[str, Any]:
    examples = _load_examples()
    full_examples = [example for example in examples if _is_full_task(example["sample_id"])]

    records: list[dict[str, Any]] = []
    by_domain = Counter()
    by_difficulty = Counter()
    by_domain_difficulty: dict[str, Counter[str]] = defaultdict(Counter)

    for example in sorted(full_examples, key=lambda item: item["sample_id"]):
        sample_id = example["sample_id"]
        domain = _domain_from_sample_id(sample_id)
        actions = _action_calls(example)
        difficulty, signals = _complexity_bucket(actions)
        by_domain[domain] += 1
        by_difficulty[difficulty] += 1
        by_domain_difficulty[domain][difficulty] += 1
        records.append(
            {
                "sample_id": sample_id,
                "domain": domain,
                "difficulty": difficulty,
                "target": example["instance"]["input"]["target"],
                "action_count": signals["action_count"],
                "has_tool_switch": signals["has_tool_switch"],
                "has_put_down": signals["has_put_down"],
                "has_putdown_unload_load_chain": signals["has_putdown_unload_load_chain"],
                "source_path": example.get("source_path"),
            }
        )

    max_balanced_per_bucket = min(by_difficulty.values()) if by_difficulty else 0
    balanced_capacity = {
        "per_difficulty_max": max_balanced_per_bucket,
        "max_balanced_total": max_balanced_per_bucket * len(DIFFICULTIES),
    }

    report = {
        "description": (
            "Full-task-only candidate scan derived from plan_examples_v2.json. "
            "Difficulty labels are recomputed with the current artifact rules."
        ),
        "input_examples": "artifact/data/pyramids/plan_examples_v2.json",
        "full_task_count": len(records),
        "counts": {
            "by_domain": {domain: by_domain.get(domain, 0) for domain in DOMAINS},
            "by_difficulty": {difficulty: by_difficulty.get(difficulty, 0) for difficulty in DIFFICULTIES},
            "by_domain_difficulty": {
                domain: {
                    difficulty: by_domain_difficulty[domain].get(difficulty, 0)
                    for difficulty in DIFFICULTIES
                }
                for domain in DOMAINS
            },
        },
        "balanced_sampling_capacity": balanced_capacity,
        "samples": records,
    }
    return report


def main() -> None:
    report = build_report()
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "full_task_count": report["full_task_count"],
                "by_domain": report["counts"]["by_domain"],
                "by_difficulty": report["counts"]["by_difficulty"],
                "balanced_sampling_capacity": report["balanced_sampling_capacity"],
                "report_output": str(REPORT_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
