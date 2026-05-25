#!/usr/bin/env python3
"""Extract LIBERO tasks into project-local planning requests."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SECTION_NAMES = ("regions", "fixtures", "objects", "obj_of_interest", "init", "goal")


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


def task_to_dict(task: Any) -> dict[str, Any]:
    if hasattr(task, "_asdict"):
        return dict(task._asdict())
    return {
        key: getattr(task, key, None)
        for key in ("name", "language", "problem", "problem_folder", "bddl_file", "init_states_file")
    }


def find_section(text: str, section: str) -> str:
    marker = f"(:{section}"
    start = text.find(marker)
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def parse_language(text: str) -> str | None:
    match = re.search(r"\(:language\s+([^\n\)]+)\)", text)
    return match.group(1).strip() if match else None


def parse_typed_entries(section_text: str) -> list[dict[str, str]]:
    rows = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^([A-Za-z0-9_]+)\s+-\s+([A-Za-z0-9_]+)$", line)
        if match:
            rows.append({"name": match.group(1), "type": match.group(2)})
    return rows


def parse_obj_of_interest(section_text: str) -> list[str]:
    items = []
    for raw_line in section_text.splitlines()[1:]:
        line = raw_line.strip().strip(")")
        if line and not line.startswith(":"):
            items.extend(part for part in line.split() if part)
    return items


def parse_predicates(section_text: str) -> list[dict[str, Any]]:
    rows = []
    for name, args in re.findall(r"\(([A-Za-z_][A-Za-z0-9_]*)\s+([^\(\)]*?)\)", section_text):
        if name in {"And", "Or", "Not"}:
            continue
        arg_list = [item for item in args.split() if item]
        if arg_list:
            rows.append({"predicate": name, "args": arg_list, "raw": f"({name} {args.strip()})"})
    return rows


def normalize_target(arg: str) -> str:
    for suffix in (
        "_contain_region",
        "_heating_region",
        "_top_region",
        "_middle_region",
        "_bottom_region",
        "_left_region",
        "_right_region",
    ):
        if arg.endswith(suffix):
            return arg[: -len(suffix)]
    return arg


def primitive_from_goal(goal: dict[str, Any]) -> dict[str, Any]:
    predicate = goal["predicate"]
    args = goal["args"]
    if predicate == "In" and len(args) >= 2:
        return {
            "op": "place_in",
            "object": args[0],
            "target": args[1],
            "display_target": normalize_target(args[1]),
            "goal": goal["raw"],
        }
    if predicate == "On" and len(args) >= 2:
        return {
            "op": "place_on",
            "object": args[0],
            "target": args[1],
            "display_target": normalize_target(args[1]),
            "goal": goal["raw"],
        }
    if predicate in {"Open", "Close", "Turnon", "TurnOn", "Turnoff", "TurnOff"} and args:
        return {"op": predicate.lower(), "object": args[0], "goal": goal["raw"]}
    if predicate in {"LeftOf", "RightOf", "FrontOf", "Behind"} and len(args) >= 2:
        return {"op": predicate.lower(), "object": args[0], "target": args[1], "goal": goal["raw"]}
    return {"op": f"assert_{predicate.lower()}", "args": args, "goal": goal["raw"]}


def build_request(
    suite_name: str,
    task_id: int,
    task_info: dict[str, Any],
    bddl_path: Path,
    bddl_root: Path,
) -> dict[str, Any]:
    text = bddl_path.read_text(encoding="utf-8")
    sections = {name: find_section(text, name) for name in SECTION_NAMES}
    goals = parse_predicates(sections["goal"])
    init = parse_predicates(sections["init"])
    language = task_info.get("language") or parse_language(text) or task_info.get("name")
    primitives = [primitive_from_goal(goal) for goal in goals]
    return {
        "request_id": f"{suite_name}:{task_id}",
        "benchmark": "LIBERO",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task_info.get("name"),
        "language": language,
        "problem_folder": task_info.get("problem_folder"),
        "bddl_file": task_info.get("bddl_file"),
        "bddl_path": str(bddl_path.relative_to(bddl_root)),
        "world_state": {
            "fixtures": parse_typed_entries(sections["fixtures"]),
            "objects": parse_typed_entries(sections["objects"]),
            "objects_of_interest": parse_obj_of_interest(sections["obj_of_interest"]),
            "init_predicates": init,
        },
        "target": {
            "goal_predicates": goals,
            "normalized_primitives": primitives,
            "compound_goal": "; ".join(goal["raw"] for goal in goals),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--output-dir", default="experiments/libero/requests")
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    run_id = f"{now_stamp()}__libero_request_extraction"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    benchmark_dict = benchmark.get_benchmark_dict()
    bddl_root = Path(get_libero_path("bddl_files"))
    created_at = now_iso()
    all_requests = []
    suite_counts = []

    for suite_name in args.suites:
        suite = benchmark_dict[suite_name]()
        suite_requests = []
        for task_id in range(suite.n_tasks):
            task_info = task_to_dict(suite.get_task(task_id))
            bddl_path = bddl_root / task_info["problem_folder"] / task_info["bddl_file"]
            request = build_request(suite_name, task_id, task_info, bddl_path, bddl_root)
            suite_requests.append(request)
            all_requests.append(request)
        write_json(output_dir / f"{suite_name}.json", suite_requests)
        suite_counts.append({"suite": suite_name, "n_requests": len(suite_requests)})

    write_json(output_dir / "all_10task_suites.json", all_requests)
    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": True,
        "repo_git_sha": git_sha(),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "bddl_root": str(bddl_root),
        "output_dir": str(output_dir),
        "total_requests": len(all_requests),
        "suites": suite_counts,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "run.json", {**summary, "kind": "libero_request_extraction"})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {**summary, "kind": "libero_request_extraction", "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
