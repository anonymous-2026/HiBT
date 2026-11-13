#!/usr/bin/env python3
"""Expand hand-labeled plan samples from ground-truth BT problems.

The script builds a larger pyramid collection by combining:
1. the existing manually curated v1 examples
2. full-task examples extracted from experiment problem files with gold BTs
3. nested selector-subtask examples extracted from those same gold BTs

Every generated sample is validated by compiling the pyramid back to a BT and
running ``sk_sim_run``. Only successful examples are kept.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR, ARTIFACT_EVAL_DIR, bootstrap_runtime

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))

from compile_plan_to_bt import PyramidCompiler, run_sk_simulation
from decode_plan_latents import _build_goal_only_pyramid, _build_pyramid_from_actions


PROBLEM_GLOBS = [
    str(ARTIFACT_DATA_DIR / "examples" / "problems" / "gearset1_problem_*.json"),
    str(ARTIFACT_DATA_DIR / "examples" / "problems" / "chair_problem_*.json"),
    str(ARTIFACT_DATA_DIR / "examples" / "problems" / "lamp_problem_*.json"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_target(raw_target: str) -> str:
    if raw_target.startswith("target:"):
        return raw_target.split(":", 1)[1].strip()
    return raw_target.strip()


def _extract_action_calls(node: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    name = node.get("name", "")
    if isinstance(name, str) and name.startswith("action:"):
        actions.append(name.split(":", 1)[1].strip())
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            actions.extend(_extract_action_calls(child))
    return actions


def _dedup_first(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_call(call: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", call)
    if not match:
        raise ValueError(f"Invalid grounded call: {call}")
    name = match.group(1)
    args_part = match.group(2).strip()
    args = [] if not args_part else [arg.strip() for arg in args_part.split(",")]
    return name, args


def _iter_nested_selectors(root: dict[str, Any]) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    queue = list(root.get("children", []) or [])
    while queue:
        node = queue.pop(0)
        if not isinstance(node, dict):
            continue
        name = node.get("name", "")
        if isinstance(name, str) and name.startswith("selector:"):
            selectors.append(node)
        queue.extend(node.get("children", []) or [])
    return selectors


def _selector_target_call(selector_node: dict[str, Any]) -> str | None:
    for child in selector_node.get("children", []) or []:
        if not isinstance(child, dict):
            continue
        name = child.get("name", "")
        if isinstance(name, str) and name.startswith("target:"):
            return name.split(":", 1)[1].strip()
    return None


def _build_instance_from_actions(
    problem_id: str,
    target_call: str,
    initial_state: dict[str, Any],
    action_calls: list[str],
) -> dict[str, Any]:
    if not action_calls:
        return _build_goal_only_pyramid(target_call, copy.deepcopy(initial_state), problem_id)

    actions = []
    for call in action_calls:
        name, args = _parse_call(call)
        actions.append((name, args))
    return _build_pyramid_from_actions(
        target_call=target_call,
        initial_state=copy.deepcopy(initial_state),
        action_sequence=actions,
        problem_id=problem_id,
    )


def _validate_instance(instance: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    behavior_tree = PyramidCompiler(instance).compile()
    result = run_sk_simulation(
        copy.deepcopy(instance["input"]["initial_state"]),
        copy.deepcopy(behavior_tree),
    )
    return result.get("result") == "success", result


def _problem_files() -> list[Path]:
    paths: list[Path] = []
    for pattern in PROBLEM_GLOBS:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return paths


def expand_collection(
    base_collection_path: Path,
    target_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _read_json(base_collection_path)
    examples = copy.deepcopy(base["examples"])
    existing_ids = {example["sample_id"] for example in examples}
    report: dict[str, Any] = {
        "base_examples": len(examples),
        "added_full": 0,
        "added_subtasks": 0,
        "skipped_validation_failures": [],
        "skipped_duplicates": [],
    }

    for problem_path in _problem_files():
        if len(examples) >= target_count:
            break

        payload = _read_json(problem_path)
        target_call = _normalize_target(payload["target"])
        initial_state = payload["initial_world_state"]
        result_tree = payload["result"]
        stem = problem_path.stem
        family = problem_path.parents[1].name
        sample_id = f"{family}_{stem}"

        action_calls = _dedup_first(_extract_action_calls(result_tree))
        full_instance = _build_instance_from_actions(
            problem_id=sample_id,
            target_call=target_call,
            initial_state=initial_state,
            action_calls=action_calls,
        )
        ok, validation = _validate_instance(full_instance)
        if sample_id not in existing_ids:
            if ok:
                examples.append(
                    {
                        "sample_id": sample_id,
                        "source_path": str(problem_path.relative_to(REPO_ROOT)),
                        "derivation": "full_task_from_gold_bt",
                        "instance": full_instance,
                    }
                )
                existing_ids.add(sample_id)
                report["added_full"] += 1
            else:
                report["skipped_validation_failures"].append(
                    {
                        "sample_id": sample_id,
                        "reason": validation,
                    }
                )
        else:
            report["skipped_duplicates"].append(sample_id)

        if len(examples) >= target_count:
            break

        # Extract nested selector subtasks in BFS order until we hit the target.
        subtask_index = 0
        for selector in _iter_nested_selectors(result_tree):
            if len(examples) >= target_count:
                break
            nested_target = _selector_target_call(selector)
            if not nested_target:
                continue
            nested_actions = _dedup_first(_extract_action_calls(selector))
            if not nested_actions and nested_target != target_call:
                continue
            nested_sample_id = f"{family}_{stem}_subtask_{subtask_index:02d}"
            subtask_index += 1
            if nested_sample_id in existing_ids:
                continue
            nested_instance = _build_instance_from_actions(
                problem_id=nested_sample_id,
                target_call=nested_target,
                initial_state=initial_state,
                action_calls=nested_actions,
            )
            ok, validation = _validate_instance(nested_instance)
            if ok:
                examples.append(
                    {
                        "sample_id": nested_sample_id,
                        "source_path": str(problem_path.relative_to(REPO_ROOT)),
                        "derivation": "selector_subtask_from_gold_bt",
                        "parent_sample_id": sample_id,
                        "instance": nested_instance,
                    }
                )
                existing_ids.add(nested_sample_id)
                report["added_subtasks"] += 1
            else:
                report["skipped_validation_failures"].append(
                    {
                        "sample_id": nested_sample_id,
                        "reason": validation,
                    }
                )

    collection = {
        "collection_version": "plan-examples-v2",
        "domain": "assembly_planning",
        "description": (
            "Expanded hand-labeled and validated plan examples built from "
            "manual v1 samples plus full-task and selector-subtask derivations."
        ),
        "examples": examples,
    }
    report["final_examples"] = len(examples)
    return collection, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand plan examples to roughly the requested target count."
    )
    parser.add_argument(
        "--base-collection",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_examples_v1.json"),
        help="Base pyramid collection to expand from.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50,
        help="Target total number of examples to keep.",
    )
    parser.add_argument(
        "--output",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_examples_v2.json"),
        help="Output collection path.",
    )
    parser.add_argument(
        "--report-output",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_examples_v2_report.json"),
        help="Where to write the generation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collection, report = expand_collection(
        base_collection_path=Path(args.base_collection),
        target_count=args.target_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n")
    Path(args.report_output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "final_examples": report["final_examples"],
                "added_full": report["added_full"],
                "added_subtasks": report["added_subtasks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
