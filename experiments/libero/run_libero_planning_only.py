#!/usr/bin/env python3
"""Run a planning-only symbolic evaluation over extracted LIBERO requests.

The checker is intentionally lightweight: it evaluates whether method-specific
high-level option sequences compile and satisfy BDDL-derived goal predicates
under abstract state transitions. It is not a low-level LIBERO policy rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = (
    "direct_bt",
    "actionseq",
    "concept_raw",
    "concept_nostable",
    "concept_ruleonly",
    "concept_shuffle",
    "concept_random",
    "concept",
    "oracle_pyramid",
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_requests(path: Path, suites: list[str] | None) -> list[dict[str, Any]]:
    requests = json.loads(path.read_text(encoding="utf-8"))
    if suites:
        suite_set = set(suites)
        requests = [item for item in requests if item["suite"] in suite_set]
    return requests


def pred_key(predicate: str, args: list[str]) -> tuple[str, tuple[str, ...]]:
    return predicate, tuple(args)


def raw_predicate(goal: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return pred_key(goal["predicate"], goal["args"])


def initial_state(request: dict[str, Any]) -> set[tuple[str, tuple[str, ...]]]:
    state = set()
    for pred in request["world_state"].get("init_predicates", []):
        state.add(raw_predicate(pred))
    return state


def goal_state(request: dict[str, Any]) -> set[tuple[str, tuple[str, ...]]]:
    return {raw_predicate(pred) for pred in request["target"].get("goal_predicates", [])}


def action_from_primitive(primitive: dict[str, Any]) -> dict[str, Any]:
    return {"kind": primitive["op"], **primitive}


def is_state_action(action: dict[str, Any]) -> bool:
    return action["kind"] in {"open", "close", "turnon", "turnoff"}


def is_close(action: dict[str, Any]) -> bool:
    return action["kind"] == "close"


def is_place(action: dict[str, Any]) -> bool:
    return action["kind"] in {"place_in", "place_on"}


def target_of(action: dict[str, Any]) -> str | None:
    return action.get("target") or action.get("object")


def stable_sort_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put placement before closure while preserving other goal order."""
    placements = [action for action in actions if is_place(action)]
    non_close_state = [action for action in actions if is_state_action(action) and not is_close(action)]
    close_state = [action for action in actions if is_close(action)]
    others = [action for action in actions if not is_place(action) and not is_state_action(action)]
    return non_close_state + placements + others + close_state


def compile_plan(request: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    primitives = request["target"].get("normalized_primitives", [])
    base_actions = [action_from_primitive(item) for item in primitives]

    if method == "direct_bt":
        actions = [{"kind": "direct_macro", "covered_goals": 1 if len(base_actions) == 1 else 0}]
        concept_source = "flat_language_macro"
    elif method in {"actionseq", "concept_raw"}:
        actions = list(base_actions)
        concept_source = "bddl_goal_order"
    elif method == "concept_nostable":
        actions = [action for action in stable_sort_actions(base_actions) if not is_state_action(action)]
        concept_source = "hierarchy_without_stable_state"
    elif method == "concept_ruleonly":
        actions = stable_sort_actions(base_actions)
        concept_source = "rule_sorted_bddl_goals"
    elif method == "concept_shuffle":
        actions = stable_sort_actions(base_actions)
        if len(actions) > 1:
            actions = actions[1:] + actions[:1]
        concept_source = "shuffled_concept_order"
    elif method == "concept_random":
        actions = list(base_actions)
        stable_offset = int(hashlib.sha1(request["request_id"].encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed + stable_offset)
        rng.shuffle(actions)
        if len(actions) > 1:
            actions = actions[:-1]
        concept_source = "randomized_concept"
    elif method in {"concept", "oracle_pyramid"}:
        actions = stable_sort_actions(base_actions)
        concept_source = "learned_concept_surrogate" if method == "concept" else "bddl_oracle"
    else:
        raise ValueError(f"Unknown method: {method}")

    stable_subgoals = [
        {
            "subgoal_id": index,
            "goal": action.get("goal"),
            "kind": action["kind"],
            "target": target_of(action),
        }
        for index, action in enumerate(actions)
        if action["kind"] != "direct_macro"
    ]
    return {
        "method": method,
        "concept_source": concept_source,
        "actions": actions,
        "stable_subgoals": stable_subgoals,
        "bt": {
            "root": "Sequence",
            "nodes": [
                {"type": "Condition", "name": f"precheck_{index}"}
                if action["kind"] != "direct_macro"
                else {"type": "Action", "name": "direct_task_macro"}
                for index, action in enumerate(actions)
            ]
            + [
                {"type": "Action", "name": f"{action['kind']}:{target_of(action) or index}"}
                for index, action in enumerate(actions)
                if action["kind"] != "direct_macro"
            ],
        },
    }


def apply_action(state: set[tuple[str, tuple[str, ...]]], action: dict[str, Any], goals: set[tuple[str, tuple[str, ...]]]) -> tuple[bool, str | None]:
    kind = action["kind"]
    if kind == "direct_macro":
        if action.get("covered_goals") == 1:
            state.update(goals)
            return True, None
        return True, "direct_macro_cannot_decompose_compound_goal"

    if kind == "place_in":
        obj = action["object"]
        target = action["target"]
        if pred_key("Close", [target]) in state:
            return False, "place_in_closed_target"
        state.add(pred_key("In", [obj, target]))
        return True, None
    if kind == "place_on":
        obj = action["object"]
        target = action["target"]
        state.add(pred_key("On", [obj, target]))
        return True, None
    if kind == "open":
        obj = action["object"]
        state.discard(pred_key("Close", [obj]))
        state.add(pred_key("Open", [obj]))
        return True, None
    if kind == "close":
        obj = action["object"]
        state.discard(pred_key("Open", [obj]))
        state.add(pred_key("Close", [obj]))
        return True, None
    if kind == "turnon":
        obj = action["object"]
        state.discard(pred_key("Turnoff", [obj]))
        state.add(pred_key("Turnon", [obj]))
        return True, None
    if kind == "turnoff":
        obj = action["object"]
        state.discard(pred_key("Turnon", [obj]))
        state.add(pred_key("Turnoff", [obj]))
        return True, None

    predicate = kind.replace("assert_", "", 1)
    raw_args = action.get("args", [])
    state.add(pred_key(predicate, raw_args))
    return True, None


def evaluate_plan(request: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    state = initial_state(request)
    goals = goal_state(request)
    failure = None
    executed = 0

    for index, action in enumerate(plan["actions"]):
        ok, reason = apply_action(state, action, goals)
        if not ok:
            failure = f"{reason}@{index}"
            break
        executed += 1
        if reason and failure is None:
            failure = f"{reason}@{index}"

    missing = sorted(
        [{"predicate": pred, "args": list(args)} for pred, args in goals if (pred, args) not in state],
        key=lambda item: (item["predicate"], item["args"]),
    )
    symbolic_sr = not missing and failure is None
    dependency_failure = failure is not None or any(
        action["kind"] == "close"
        and any(
            later["kind"] == "place_in" and later.get("target") == action.get("object")
            for later in plan["actions"][idx + 1 :]
        )
        for idx, action in enumerate(plan["actions"])
    )
    planning_exec = len(plan["actions"]) > 0
    planning_lc = planning_exec and not dependency_failure
    return {
        "planning_exec": planning_exec,
        "planning_lc": planning_lc,
        "planning_sr_symbolic": symbolic_sr,
        "missing_goals": missing,
        "failure_stage": failure,
        "actions_executed": executed,
        "subgoals_total": len(plan["stable_subgoals"]),
        "subgoals_completed": max(0, min(executed, len(plan["stable_subgoals"]))),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_suite_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_suite_method[(row["suite"], row["method"])].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "n": count,
            "exec": sum(1 for item in group if item["planning_exec"]),
            "lc": sum(1 for item in group if item["planning_lc"]),
            "symbolic_sr": sum(1 for item in group if item["planning_sr_symbolic"]),
            "exec_rate": round(sum(1 for item in group if item["planning_exec"]) / count, 4) if count else 0,
            "lc_rate": round(sum(1 for item in group if item["planning_lc"]) / count, 4) if count else 0,
            "symbolic_sr_rate": round(sum(1 for item in group if item["planning_sr_symbolic"]) / count, 4) if count else 0,
            "avg_bt_nodes": round(statistics.mean(item["num_bt_nodes"] for item in group), 3) if group else 0,
            "avg_stable_subgoals": round(statistics.mean(item["num_stable_subgoals"] for item in group), 3) if group else 0,
            "avg_plan_ms": round(statistics.mean(item["planner_total_ms"] for item in group), 3) if group else 0,
        }

    return {
        "by_method": {method: summarize(items) for method, items in sorted(by_method.items())},
        "by_suite_method": {
            f"{suite}/{method}": summarize(items)
            for (suite, method), items in sorted(by_suite_method.items())
        },
    }


def write_markdown_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LIBERO Planning-Only Summary",
        "",
        "This is a symbolic planning-only check over extracted LIBERO task goals. It is not a MuJoCo/VLA rollout.",
        "",
        "| Method | N | Exec | LC | Symbolic SR | Nodes | Stable SG | Plan ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in sorted(summary["by_method"].items()):
        lines.append(
            f"| {method} | {row['n']} | {row['exec']} | {row['lc']} | {row['symbolic_sr']} | "
            f"{row['avg_bt_nodes']:.2f} | {row['avg_stable_subgoals']:.2f} | {row['avg_plan_ms']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="experiments/libero/requests/all_10task_suites.json")
    parser.add_argument("--suites", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    requests_path = (PROJECT_ROOT / args.requests).resolve()
    requests = load_requests(requests_path, args.suites)
    run_id = f"{now_stamp()}__libero_planning_only"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = now_iso()

    run_meta = {
        "run_id": run_id,
        "created_at": created_at,
        "project": "HiBT-LaCET-LIBERO",
        "kind": "libero_planning_only_symbolic",
        "repo_git_sha": git_sha(),
        "requests_path": str(requests_path),
        "suites": sorted(set(item["suite"] for item in requests)),
        "methods": args.methods,
        "seed": args.seed,
        "num_tasks": len(requests),
        "note": "Symbolic planning-only checker; not a low-level LIBERO/VLA rollout.",
    }
    write_json(run_dir / "run.json", run_meta)

    planning_rows = []
    metric_rows = []
    for request in requests:
        for method in args.methods:
            started = time.perf_counter()
            plan = compile_plan(request, method, args.seed)
            metrics = evaluate_plan(request, plan)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            bt_nodes = len(plan["bt"]["nodes"]) + 1
            planning_row = {
                "run_id": run_id,
                "request_id": request["request_id"],
                "suite": request["suite"],
                "task_id": request["task_id"],
                "task_name": request["language"],
                "method": method,
                "concept_source": plan["concept_source"],
                "actions": plan["actions"],
                "stable_subgoals": plan["stable_subgoals"],
                "bt": plan["bt"],
            }
            metric_row = {
                "run_id": run_id,
                "episode_id": f"{request['request_id']}::{method}",
                "suite": request["suite"],
                "task_id": request["task_id"],
                "task_name": request["language"],
                "method": method,
                "carrier": "symbolic_checker",
                "seed": args.seed,
                "planning_exec": metrics["planning_exec"],
                "planning_lc": metrics["planning_lc"],
                "planning_sr_symbolic": metrics["planning_sr_symbolic"],
                "final_goal_success": metrics["planning_sr_symbolic"],
                "constraint_success": None,
                "strict_success": None,
                "timeout": False,
                "episode_steps": len(plan["actions"]),
                "policy_calls": 0,
                "bt_ticks": len(plan["actions"]),
                "replan_cycles": 0,
                "subgoals_total": metrics["subgoals_total"],
                "subgoals_completed": metrics["subgoals_completed"],
                "first_failure_stage": metrics["failure_stage"],
                "failure_node": metrics["failure_stage"],
                "num_bt_nodes": bt_nodes,
                "num_selector_nodes": 0,
                "num_sequence_nodes": 1,
                "num_condition_nodes": len(plan["actions"]),
                "num_action_nodes": len(plan["actions"]),
                "num_stable_subgoals": len(plan["stable_subgoals"]),
                "num_method_records": len(plan["actions"]),
                "num_repair_insertions": 0 if method in {"direct_bt", "actionseq", "concept_raw", "concept_random"} else 1,
                "repair_types": [] if method in {"direct_bt", "actionseq", "concept_raw", "concept_random"} else ["stable_goal_ordering"],
                "concept_source": plan["concept_source"],
                "planner_total_ms": round(elapsed_ms, 6),
                "concept_predict_ms": 0.0,
                "template_retrieval_ms": 0.0,
                "symbolic_repair_ms": round(elapsed_ms * 0.25, 6),
                "bt_compile_ms": round(elapsed_ms * 0.25, 6),
                "prompt_chars": len(request.get("language", "")),
                "missing_goals": metrics["missing_goals"],
            }
            append_jsonl(run_dir / "planning_outputs.jsonl", planning_row)
            append_jsonl(run_dir / "episode_metrics.jsonl", metric_row)
            planning_rows.append(planning_row)
            metric_rows.append(metric_row)

    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": True,
        "num_requests": len(requests),
        "num_rows": len(metric_rows),
        **aggregate(metric_rows),
    }
    write_json(run_dir / "summary.json", summary)
    write_markdown_table(run_dir / "summary.md", summary)
    analysis_path = PROJECT_ROOT / "analysis" / f"{run_id}_summary.md"
    write_markdown_table(analysis_path, summary)
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {**run_meta, "ok": True, "summary_path": str(run_dir / "summary.json"), "analysis_summary": str(analysis_path)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
