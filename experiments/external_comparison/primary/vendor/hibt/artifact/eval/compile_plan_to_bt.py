#!/usr/bin/env python3
"""Compile structured plan examples into behavior-tree skeletons.

This script is the deterministic bridge between the intermediate plan
representation (`plan-schema-v1`) and executable behavior-tree skeletons.

It supports two modes:

1. Compile only:
   - read one example collection
   - emit compiled BT skeleton JSON
2. Compile + evaluate:
   - compile each example
   - run sk_sim_run on the compiled BT
   - write a per-sample report plus a summary
"""

from __future__ import annotations

import argparse
import copy
import json
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

from runtime import run_sk_simulation
from runtime.minimal_bt import ground_action


def _call(name: str, args: list[str]) -> str:
    return f"{name}({', '.join(args)})"


def _node_summary(kind: str, body: str) -> str:
    return f"{kind} {body}"


def _condition_call_from_object_property(op: Any) -> str:
    args = [op.object_name]
    if op.property_value is not None:
        args.append(op.property_value)
    return _call(op.property_name, args)


def _condition_node(kind: str, call: str) -> dict[str, Any]:
    return {
        "summary": _node_summary(kind, call),
        "name": f"{kind}: {call}",
    }


def _action_node(action_name: str, args: list[str]) -> dict[str, Any]:
    call = _call(action_name, args)
    return {
        "summary": _node_summary("action", call),
        "name": f"action: {call}",
    }


def _selector_node(call: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": _node_summary("selector", call),
        "name": f"selector: {call}",
        "children": children,
    }


def _sequence_node(call: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": _node_summary("sequence", call),
        "name": f"sequence: {call}",
        "children": children,
    }


class PyramidCompiler:
    def __init__(self, instance: dict[str, Any]) -> None:
        self.instance = instance
        levels = instance["pyramid"]
        self.goal_level = levels[0]["items"]
        self.subgoal_level = levels[1]["items"]
        self.method_level = levels[2]["items"]
        self.action_level = levels[3]["items"]

        self.goals = {item["id"]: item for item in self.goal_level}
        self.subgoals = {item["id"]: item for item in self.subgoal_level}
        self.methods = {item["id"]: item for item in self.method_level}
        self.method_by_achieves = {
            item["achieves"]: item for item in self.method_level
        }

    def compile(self) -> dict[str, Any]:
        root_goal = self.goal_level[0]
        return self._compile_reference(root_goal["id"])

    def _compile_reference(self, ref_id: str) -> dict[str, Any]:
        if ref_id in self.goals:
            return self._compile_goal(self.goals[ref_id])
        if ref_id in self.subgoals:
            return self._compile_subgoal(self.subgoals[ref_id])
        raise KeyError(f"Unknown goal/subgoal reference: {ref_id}")

    def _compile_goal(self, goal: dict[str, Any]) -> dict[str, Any]:
        target_call = _call(goal["predicate"], goal["args"])
        method = self.method_by_achieves.get(goal["id"])
        if method is None:
            return _selector_node(
                target_call,
                [_condition_node("target", target_call)],
            )
        return self._compile_method_targeting(method, target_call)

    def _compile_subgoal(self, subgoal: dict[str, Any]) -> dict[str, Any]:
        target_call = _call(subgoal["predicate"], subgoal["args"])
        method = self.method_by_achieves.get(subgoal["id"])
        if method is None:
            return _selector_node(
                target_call,
                [_condition_node("target", target_call)],
            )
        return self._compile_method_targeting(method, target_call)

    def _compile_method_targeting(
        self, method: dict[str, Any], target_call: str
    ) -> dict[str, Any]:
        action_name = method["action"]["name"]
        action_args = method["action"]["args"]
        action_call = _call(action_name, action_args)

        sequence_children: list[dict[str, Any]] = []

        for req_id in method.get("requires", []):
            sequence_children.append(self._compile_reference(req_id))

        preconditions, _ = ground_action(action_name, action_args)
        for op in preconditions:
            precondition_call = _condition_call_from_object_property(op)
            sequence_children.append(_condition_node("precondition", precondition_call))

        sequence_children.append(_action_node(action_name, action_args))

        return _selector_node(
            action_call,
            [
                _condition_node("target", target_call),
                _sequence_node(action_call, sequence_children),
            ],
        )


def _load_examples(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def compile_collection(
    collection: dict[str, Any], sample_ids: set[str] | None
) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for example in collection["examples"]:
        sample_id = example["sample_id"]
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        instance = example["instance"]
        compiler = PyramidCompiler(instance)
        behavior_tree = compiler.compile()
        compiled.append(
            {
                "sample_id": sample_id,
                "source_path": example.get("source_path"),
                "instance": instance,
                "behavior_tree": behavior_tree,
            }
        )
    return compiled


def evaluate_compiled_samples(compiled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in compiled:
        instance = item["instance"]
        world_state = copy.deepcopy(instance["input"]["initial_state"])
        tree = copy.deepcopy(item["behavior_tree"])

        evaluation_error: str | None = None
        evaluation_result: dict[str, Any] | None = None
        try:
            evaluation_result = run_sk_simulation(world_state, tree)
        except Exception as exc:
            evaluation_error = f"{type(exc).__name__}: {exc}"

        results.append(
            {
                "sample_id": item["sample_id"],
                "source_path": item.get("source_path"),
                "target": instance["input"]["target"],
                "behavior_tree": item["behavior_tree"],
                "evaluation_result": evaluation_result,
                "evaluation_error": evaluation_error,
                "success": evaluation_result is not None
                and evaluation_result.get("result") == "success",
            }
        )
    return results


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successes = sum(1 for item in results if item["success"])
    errors = sum(1 for item in results if item["evaluation_error"] is not None)
    failures = total - successes - errors
    return {
        "total": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "success_rate": (successes / total) if total else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile structured plans into BT skeletons and optionally run "
            "sk_sim_run on the compiled trees."
        )
    )
    parser.add_argument(
        "--examples-file",
        default=str(ARTIFACT_DATA_DIR / "pyramids" / "plan_examples_v1.json"),
        help="Path to the hand-labeled plan collection JSON.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Optional sample_id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write compiled BT skeletons as JSON.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run sk_sim_run on each compiled tree.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Where to write per-sample evaluation results.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Where to write the evaluation summary JSON.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    examples_file = Path(args.examples_file).expanduser().resolve()
    collection = _load_examples(examples_file)
    sample_ids = set(args.sample_id) if args.sample_id else None

    compiled = compile_collection(collection, sample_ids)
    if args.output_file:
        _write_json(Path(args.output_file).expanduser().resolve(), compiled)

    print(f"compiled_samples={len(compiled)}", flush=True)

    if not args.evaluate:
        return

    results = evaluate_compiled_samples(compiled)
    summary = summarize_evaluation(results)

    for item in results:
        print(f"=== {item['sample_id']} ===", flush=True)
        if item["evaluation_error"] is not None:
            print(f"[evaluation_error] {item['evaluation_error']}", flush=True)
        else:
            print(
                f"[sk_sim_result] {item['evaluation_result'].get('result')}",
                flush=True,
            )
            if not item["success"]:
                final_node = item["evaluation_result"].get("final_node")
                if final_node is not None:
                    print(f"[final_node] {final_node}", flush=True)

    print("=== summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.report_file:
        _write_json(Path(args.report_file).expanduser().resolve(), results)
    if args.summary_file:
        _write_json(Path(args.summary_file).expanduser().resolve(), summary)


if __name__ == "__main__":
    main()
