#!/usr/bin/env python3
"""Generate extra concept-training samples that stay disjoint from main_benchmark.

The script expands only from seed problem families that are absent from both:

- artifact/data/datasets/plan_predictor_v3/train.jsonl
- artifact/data/requests/benchmark_main.json

Each generated sample is:
1. created from a programmatic world-state variant
2. assigned a target call
3. solved deterministically into a gold action sequence
4. converted into ``pyramid_json``
5. validated by BT compilation + symbolic execution

The output is written directly in predictor JSONL format so it can be used
immediately to build a strict-isolation prototype bank.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
if str(ARTIFACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_ROOT))

from common_paths import ARTIFACT_DATA_DIR, ARTIFACT_EVAL_DIR, bootstrap_runtime

bootstrap_runtime()
if str(ARTIFACT_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_EVAL_DIR))

from compile_plan_to_bt import PyramidCompiler, run_sk_simulation
from planning.export_plan_dataset import (
    _render_groundtruth,
    _render_pyramid_text,
    _render_question,
)
from planning.decode_plan_latents import (
    _build_pyramid_from_actions,
    _call,
    _normalize_state_for_kios,
    _parse_call,
    _synthesize_action_sequence,
)


REPO_ROOT = ARTIFACT_ROOT.parent.parent
KIOS_EXPERIMENTS = Path(os.environ.get("KIOS_EXPERIMENTS_ROOT", REPO_ROOT / "external" / "kios" / "experiments"))


@dataclass(frozen=True)
class SeedSpec:
    sample_id: str
    source_path: Path


@dataclass(frozen=True)
class VariantSpec:
    suffix: str
    target_builder: Callable[[dict[str, Any]], str]
    state_transform: Callable[[dict[str, Any]], dict[str, Any]]
    description: str


SEEDS = [
    SeedSpec(
        sample_id="gearset1_problem_017",
        source_path=KIOS_EXPERIMENTS / "gearset1" / "problem_set" / "problem_017.json",
    ),
    SeedSpec(
        sample_id="lamp_problem_002",
        source_path=KIOS_EXPERIMENTS / "lamp" / "problem_set" / "problem_002.json",
    ),
    SeedSpec(
        sample_id="lamp_problem_003",
        source_path=KIOS_EXPERIMENTS / "lamp" / "problem_set" / "problem_003.json",
    ),
]


def _load_problem(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_target(raw_target: str) -> str:
    if raw_target.startswith("target:"):
        return raw_target.split(":", 1)[1].strip()
    return raw_target.strip()


def _remove_relations(
    state: dict[str, Any],
    *,
    source: str | None = None,
    name: str | None = None,
    target: str | None = None,
) -> None:
    kept = []
    for relation in state.get("relations", []):
        if source is not None and relation.get("source") != source:
            kept.append(relation)
            continue
        if name is not None and relation.get("name") != name:
            kept.append(relation)
            continue
        if target is not None and relation.get("target") != target:
            kept.append(relation)
            continue
        if source is None and name is None and target is None:
            kept.append(relation)
    state["relations"] = kept


def _add_relation(state: dict[str, Any], source: str, name: str, target: str) -> None:
    for relation in state.get("relations", []):
        if (
            relation.get("source") == source
            and relation.get("name") == name
            and relation.get("target") == target
        ):
            return
    state.setdefault("relations", []).append(
        {"source": source, "name": name, "target": target}
    )


def _set_hand_tool(state: dict[str, Any], hand: str, tool: str | None) -> None:
    _remove_relations(state, source=hand, name="hold")
    if tool is not None:
        _add_relation(state, hand, "hold", tool)


def _set_tool_payload(state: dict[str, Any], tool: str, part: str | None) -> None:
    _remove_relations(state, source=tool, name="hold")
    if part is not None:
        _add_relation(state, tool, "hold", part)


def _ensure_goal_satisfied(state: dict[str, Any], target_call: str) -> None:
    predicate, args = _parse_call(target_call)
    if len(args) == 2:
        _add_relation(state, args[0], predicate, args[1])


def _base_state(problem: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(problem["initial_world_state"])


def _identity(state: dict[str, Any]) -> dict[str, Any]:
    return state


def _hand_empty(state: dict[str, Any], hand: str = "left_hand") -> dict[str, Any]:
    _set_hand_tool(state, hand, None)
    return state


def _hold_tool(state: dict[str, Any], tool: str, hand: str = "left_hand") -> dict[str, Any]:
    _set_hand_tool(state, hand, tool)
    return state


def _hold_tool_with_payload(
    state: dict[str, Any],
    tool: str,
    payload: str,
    hand: str = "left_hand",
) -> dict[str, Any]:
    _set_hand_tool(state, hand, tool)
    _set_tool_payload(state, tool, payload)
    return state


def _required_tool_with_target_part(
    state: dict[str, Any],
    tool: str,
    part: str,
    hand: str = "left_hand",
) -> dict[str, Any]:
    _set_hand_tool(state, hand, tool)
    _set_tool_payload(state, tool, part)
    return state


def _goal_satisfied_variant(state: dict[str, Any], target_call: str) -> dict[str, Any]:
    _ensure_goal_satisfied(state, target_call)
    return state


def _target_literal(target_call: str) -> Callable[[dict[str, Any]], str]:
    return lambda _problem: target_call


def _target_from_problem(problem: dict[str, Any]) -> str:
    return _normalize_target(problem["target"])


def _instance_from_actions(
    target_call: str,
    initial_state: dict[str, Any],
    action_calls: list[tuple[str, list[str]]],
    problem_id: str,
) -> dict[str, Any]:
    return _build_pyramid_from_actions(
        target_call=target_call,
        initial_state=copy.deepcopy(initial_state),
        action_sequence=action_calls,
        problem_id=problem_id,
    )


def _validate_instance(instance: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    behavior_tree = PyramidCompiler(instance).compile()
    result = run_sk_simulation(
        copy.deepcopy(instance["input"]["initial_state"]),
        copy.deepcopy(behavior_tree),
    )
    return result.get("result") == "success", result


def _to_record(
    sample_id: str,
    problem_id: str,
    source_path: Path,
    instance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "main_id": sample_id,
        "question": _render_question(instance),
        "cot_answer": _render_pyramid_text(instance),
        "groundtruth": _render_groundtruth(instance),
        "problem_id": problem_id,
        "source_path": str(source_path),
        "target": instance["input"]["target"],
        "pyramid_json": instance["pyramid"],
    }


def _variant_specs_for_seed(seed: SeedSpec, target_call: str) -> list[VariantSpec]:
    if seed.sample_id == "gearset1_problem_017":
        required_tool = "outwardgripper"
        target_part = "compoundgear"
        wrong_tool = "defaultgripper"
        wrong_payload = "largeshaft"
        return [
            VariantSpec("full", _target_literal(target_call), _identity, "base full task"),
            VariantSpec("hand_empty", _target_literal(target_call), lambda s: _hand_empty(s), "full task from empty hand"),
            VariantSpec("wrong_tool", _target_literal(target_call), lambda s: _hold_tool(s, wrong_tool), "full task from wrong tool"),
            VariantSpec("wrong_tool_payload", _target_literal(target_call), lambda s: _hold_tool_with_payload(s, wrong_tool, wrong_payload), "wrong tool with payload"),
            VariantSpec("target_part_picked", _target_literal(target_call), lambda s: _required_tool_with_target_part(s, required_tool, target_part), "required tool already holds target part"),
            VariantSpec("goal_already_satisfied", _target_literal(target_call), lambda s: _goal_satisfied_variant(s, target_call), "goal already satisfied"),
            VariantSpec("subtask_hold_tool", _target_literal(_call("hold", ["left_hand", required_tool])), _identity, "subtask hold required tool"),
            VariantSpec("subtask_hold_part", _target_literal(_call("hold", [required_tool, target_part])), _identity, "subtask hold target part"),
        ]

    if seed.sample_id == "lamp_problem_002":
        required_tool = "outwardgripper"
        target_part = "lampshade"
        wrong_tool = "defaultgripper"
        payload_tool = "clampgripper"
        payload_part = "cube"
        return [
            VariantSpec("full", _target_literal(target_call), _identity, "base full task"),
            VariantSpec("hand_empty", _target_literal(target_call), lambda s: _hand_empty(s), "full task from empty hand"),
            VariantSpec("wrong_tool", _target_literal(target_call), lambda s: _hold_tool(s, wrong_tool), "full task from wrong tool"),
            VariantSpec("wrong_tool_payload", _target_literal(target_call), lambda s: _hold_tool_with_payload(s, payload_tool, payload_part), "wrong tool with payload"),
            VariantSpec("required_tool_loaded", _target_literal(target_call), lambda s: _hold_tool(s, required_tool), "required tool already loaded"),
            VariantSpec("target_part_picked", _target_literal(target_call), lambda s: _required_tool_with_target_part(s, required_tool, target_part), "required tool already holds target part"),
            VariantSpec("goal_already_satisfied", _target_literal(target_call), lambda s: _goal_satisfied_variant(s, target_call), "goal already satisfied"),
            VariantSpec("subtask_hold_tool", _target_literal(_call("hold", ["left_hand", required_tool])), _identity, "subtask hold required tool"),
            VariantSpec("subtask_hold_part", _target_literal(_call("hold", [required_tool, target_part])), _identity, "subtask hold target part"),
        ]

    if seed.sample_id == "lamp_problem_003":
        required_tool = "outwardgripper"
        target_part = "lampshade"
        wrong_tool = "defaultgripper"
        return [
            VariantSpec("goal_satisfied", _target_literal(target_call), _identity, "goal already satisfied base sample"),
            VariantSpec("goal_satisfied_hand_empty", _target_literal(target_call), lambda s: _hand_empty(s), "goal satisfied from empty hand"),
            VariantSpec("subtask_hold_tool", _target_literal(_call("hold", ["left_hand", required_tool])), _identity, "subtask hold required tool"),
            VariantSpec("subtask_hold_tool_hand_empty", _target_literal(_call("hold", ["left_hand", required_tool])), lambda s: _hand_empty(s), "subtask hold required tool from empty hand"),
            VariantSpec("subtask_hold_tool_wrong_tool", _target_literal(_call("hold", ["left_hand", required_tool])), lambda s: _hold_tool(s, wrong_tool), "subtask hold required tool from wrong tool"),
            VariantSpec("subtask_hold_part", _target_literal(_call("hold", [required_tool, target_part])), _identity, "subtask hold target part"),
            VariantSpec("subtask_hold_part_required_tool_loaded", _target_literal(_call("hold", [required_tool, target_part])), lambda s: _hold_tool(s, required_tool), "subtask hold target part with tool preloaded"),
            VariantSpec("subtask_hold_part_already_picked", _target_literal(_call("hold", [required_tool, target_part])), lambda s: _required_tool_with_target_part(s, required_tool, target_part), "subtask target already satisfied"),
        ]

    raise ValueError(f"Unsupported seed family: {seed.sample_id}")


def build_extra_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "generator_version": "main_benchmark-excluded-extra-v1",
        "seed_families": [seed.sample_id for seed in SEEDS],
        "kept": [],
        "dropped": [],
    }

    for seed in SEEDS:
        problem = _load_problem(seed.source_path)
        target_call = _target_from_problem(problem)
        variants = _variant_specs_for_seed(seed, target_call)
        for variant in variants:
            sample_id = f"{seed.sample_id}_{variant.suffix}"
            state = _base_state(problem)
            state = variant.state_transform(state)
            state = _normalize_state_for_kios(state)
            variant_target = variant.target_builder(problem)
            actions, notes = _synthesize_action_sequence(
                variant_target,
                state,
            )
            instance = _instance_from_actions(
                target_call=variant_target,
                initial_state=state,
                action_calls=actions,
                problem_id=sample_id,
            )
            ok, validation = _validate_instance(instance)
            item_report = {
                "sample_id": sample_id,
                "seed_family": seed.sample_id,
                "target": variant_target,
                "description": variant.description,
                "repair_notes": notes,
                "action_count": len(actions),
            }
            if ok:
                records.append(
                    _to_record(
                        sample_id=sample_id,
                        problem_id=seed.sample_id,
                        source_path=seed.source_path,
                        instance=instance,
                    )
                )
                report["kept"].append(item_report)
            else:
                item_report["validation"] = validation
                report["dropped"].append(item_report)

    return records, report


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate strict-isolation extra predictor rows outside current train/main_benchmark."
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(
            ARTIFACT_DATA_DIR
            / "datasets"
            / "plan_predictor_v3"
            / "train_excl_main_benchmark_extra.jsonl"
        ),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--report-output",
        default=str(
            ARTIFACT_DATA_DIR
            / "datasets"
            / "plan_predictor_v3"
            / "train_excl_main_benchmark_extra_report.json"
        ),
        help="Output report JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, report = build_extra_records()
    write_jsonl(Path(args.output_jsonl), records)
    Path(args.report_output).write_text(
        json.dumps(
            {
                **report,
                "record_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "report_output": args.report_output,
                "record_count": len(records),
                "dropped_count": len(report["dropped"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
