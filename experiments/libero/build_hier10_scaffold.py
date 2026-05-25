#!/usr/bin/env python3
"""Create a project-local LIBERO-Hier10 scaffold from the process design."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    {
        "id": 0,
        "type": "multi_object",
        "language": "put alphabet soup and tomato sauce in basket",
        "key_test": "subgoal decomposition",
        "source_suite_hint": "libero_10",
        "source_task_hint": 0,
    },
    {
        "id": 1,
        "type": "distractor_aware",
        "language": "put soup and tomato sauce in basket while ignoring ketchup",
        "key_test": "wrong-object / distractor",
        "source_suite_hint": "libero_90",
        "source_task_hint": "LIVING_ROOM_SCENE1/2 basket tasks",
    },
    {
        "id": 2,
        "type": "articulated",
        "language": "put black bowl in bottom drawer and close it",
        "key_test": "open/close stable state",
        "source_suite_hint": "libero_10",
        "source_task_hint": 3,
    },
    {
        "id": 3,
        "type": "multi_stage",
        "language": "put bowl in drawer and close it, then place wine bottle on rack",
        "key_test": "cross-receptacle planning",
        "source_suite_hint": "libero_90",
        "source_task_hint": "KITCHEN_SCENE4 drawer/rack tasks",
    },
    {
        "id": 4,
        "type": "stove_state",
        "language": "turn on stove and put moka pot on it",
        "key_test": "state + placement",
        "source_suite_hint": "libero_10",
        "source_task_hint": 2,
    },
    {
        "id": 5,
        "type": "microwave_state",
        "language": "put mug in microwave and close it",
        "key_test": "articulated closure",
        "source_suite_hint": "libero_10",
        "source_task_hint": 9,
    },
    {
        "id": 6,
        "type": "two_plates",
        "language": "put white mug on left plate and yellow-white mug on right plate",
        "key_test": "spatial binding/order",
        "source_suite_hint": "libero_10",
        "source_task_hint": 4,
    },
    {
        "id": 7,
        "type": "stack_place",
        "language": "stack bowls then place in tray",
        "key_test": "compositional manipulation",
        "source_suite_hint": "libero_90",
        "source_task_hint": "LIVING_ROOM_SCENE4 stack bowl tasks",
    },
    {
        "id": 8,
        "type": "caddy_compartment",
        "language": "place book in back compartment then mug right of caddy",
        "key_test": "target-region specificity",
        "source_suite_hint": "libero_10/libero_90",
        "source_task_hint": "STUDY_SCENE1/3 caddy tasks",
    },
    {
        "id": 9,
        "type": "constraint",
        "language": "put target object in tray without moving distractor",
        "key_test": "strict success vs final success",
        "source_suite_hint": "libero_90",
        "source_task_hint": "tray tasks with distractors",
    },
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/libero/hier10")
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()

    run_id = f"{now_stamp()}__libero_hier10_scaffold"
    created_at = now_iso()
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "suite_name": "libero_hier10",
        "created_at": created_at,
        "status": "scaffold_only",
        "tasks": [
            {
                **task,
                "bddl_status": "todo",
                "constraint_fields": ["final_goal_success", "constraint_success", "strict_success"],
                "recommended_methods": ["flat_vla", "actionseq_vla", "hibt_vla", "hibt_vla_no_stable", "manual_subgoal_vla"],
            }
            for task in TASKS
        ],
    }
    write_json(output_dir / "libero_hier10_scaffold.json", payload)
    lines = [
        "# LIBERO-Hier10 Scaffold",
        "",
        "Status: scaffold only. BDDL generation and validation are the next implementation step.",
        "",
        "| ID | Type | Task | Key Test | Source Hint |",
        "|---:|---|---|---|---|",
    ]
    for task in TASKS:
        lines.append(
            f"| {task['id']} | {task['type']} | {task['language']} | {task['key_test']} | "
            f"{task['source_suite_hint']}:{task['source_task_hint']} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": True,
        "suite_name": "libero_hier10",
        "num_tasks": len(TASKS),
        "output_dir": str(output_dir),
        "status": "scaffold_only",
    }
    write_json(run_dir / "run.json", {"run_id": run_id, "kind": "libero_hier10_scaffold", "created_at": created_at})
    write_json(run_dir / "summary.json", summary)
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "libero_hier10_scaffold", "created_at": created_at, "ok": True, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
