#!/usr/bin/env python3
"""Build a 50-instance held-out test request file from plan examples v2.

The current artifact's validated candidate pool is exactly the 50 examples in
``artifact/data/pyramids/plan_examples_v2.json``.  We therefore:

1. enumerate all candidates,
2. assign a complexity label to each candidate,
3. compare the observed domain/difficulty distribution against an ideal target
   matrix, and
4. export the full 50-instance pool as a fixed test request file.

This keeps the test-set construction reproducible and makes the sampling
limitation explicit in the report.
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
REQUEST_OUTPUT = DATA_DIR / "requests" / "test_requests_50.json"
META_OUTPUT = DATA_DIR / "requests" / "test_requests_50_metadata.json"


IDEAL_MATRIX: dict[str, dict[str, int]] = {
    "gearset": {"easy": 3, "medium": 4, "hard": 3, "very_hard": 0},
    "gearset1": {"easy": 1, "medium": 3, "hard": 6, "very_hard": 5},
    "chair": {"easy": 2, "medium": 3, "hard": 5, "very_hard": 2},
    "lamp": {"easy": 2, "medium": 4, "hard": 4, "very_hard": 3},
}


def _call(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"


def _domain_from_sample_id(sample_id: str) -> str:
    if sample_id.startswith("gearset_insert_"):
        return "gearset"
    if sample_id.startswith("gearset1_"):
        return "gearset1"
    if sample_id.startswith("chair_"):
        return "chair"
    if sample_id.startswith("lamp_"):
        return "lamp"
    raise ValueError(f"Unable to infer domain from sample_id: {sample_id}")


def _action_calls(example: dict[str, Any]) -> list[str]:
    action_items = example["instance"]["pyramid"][3]["items"]
    return [_call(item["name"], item["args"]) for item in action_items]


def _complexity_bucket(actions: list[str]) -> tuple[str, dict[str, bool | int]]:
    action_count = len(actions)
    has_put_down = any(call.startswith("put_down(") for call in actions)
    has_unload = any(call.startswith("unload_tool(") for call in actions)
    has_load = any(call.startswith("load_tool(") for call in actions)
    has_change = any(call.startswith("change_tool(") for call in actions)
    has_tool_switch = has_unload or has_load or has_change
    has_chain = has_put_down and (has_unload or has_change) and (
        has_load or has_change
    )

    if action_count == 0 or action_count == 1:
        bucket = "easy"
    elif has_chain:
        bucket = "very_hard"
    elif action_count >= 4 or has_tool_switch:
        bucket = "hard"
    else:
        bucket = "medium"

    signals = {
        "action_count": action_count,
        "has_tool_switch": has_tool_switch,
        "has_put_down": has_put_down,
        "has_putdown_unload_load_chain": has_chain,
    }
    return bucket, signals


def _make_request_record(example: dict[str, Any]) -> dict[str, Any]:
    sample_id = example["sample_id"]
    domain = _domain_from_sample_id(sample_id)
    actions = _action_calls(example)
    difficulty, signals = _complexity_bucket(actions)
    instance = example["instance"]
    return {
        "sample_id": sample_id,
        "domain": domain,
        "difficulty": difficulty,
        "target": instance["input"]["target"],
        "world_state": instance["input"]["initial_state"],
        "action_sequence_gold": actions,
        **signals,
        "problem_id": instance.get("problem_id", sample_id),
        "source_path": example.get("source_path"),
    }


def build_test_requests(
    input_path: Path = PYRAMID_INPUT,
    request_output: Path = REQUEST_OUTPUT,
    meta_output: Path = META_OUTPUT,
) -> dict[str, Any]:
    examples = json.loads(input_path.read_text(encoding="utf-8"))["examples"]
    requests = [_make_request_record(example) for example in examples]

    by_domain = Counter(item["domain"] for item in requests)
    by_diff = Counter(item["difficulty"] for item in requests)
    by_domain_diff: dict[str, Counter[str]] = defaultdict(Counter)
    for item in requests:
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
            "Formal 50-instance test set derived from the validated plan example "
            "pool. The artifact currently exposes exactly 50 usable candidates, "
            "so the final test set keeps all candidates and reports the resulting "
            "domain/difficulty distribution against the ideal target matrix."
        ),
        "input_examples": str(input_path),
        "total_candidates": len(requests),
        "final_test_size": len(requests),
        "ideal_domain_difficulty_matrix": IDEAL_MATRIX,
        "actual_domain_difficulty_matrix": actual_matrix,
        "counts": {
            "by_domain": dict(by_domain),
            "by_difficulty": dict(by_diff),
        },
        "samples": [
            {
                "sample_id": item["sample_id"],
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "target": item["target"],
                "action_count": item["action_count"],
                "has_tool_switch": item["has_tool_switch"],
                "has_putdown_unload_load_chain": item[
                    "has_putdown_unload_load_chain"
                ],
            }
            for item in requests
        ],
    }

    request_output.parent.mkdir(parents=True, exist_ok=True)
    request_output.write_text(
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = build_test_requests()
    print(
        json.dumps(
            {
                "total_candidates": report["total_candidates"],
                "final_test_size": report["final_test_size"],
                "by_domain": report["counts"]["by_domain"],
                "by_difficulty": report["counts"]["by_difficulty"],
                "request_output": str(REQUEST_OUTPUT),
                "meta_output": str(META_OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
