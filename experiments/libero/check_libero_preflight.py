#!/usr/bin/env python3
"""Project-local LIBERO resource preflight.

This script reads the installed LIBERO benchmark resources and writes a
runpack-style audit under this repository's runs/ directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


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


def parse_reset_smoke(values: list[str]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for raw in values:
        if ":" not in raw:
            raise SystemExit(f"--reset-smoke expects SUITE:TASK_ID, got {raw!r}")
        suite, task_id = raw.split(":", 1)
        result.append((suite, int(task_id)))
    return result


def run_reset_smoke(
    suite_name: str,
    task_id: int,
    benchmark_dict: dict[str, Any],
    get_libero_path: Any,
) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    started = time.perf_counter()
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    task_info = task_to_dict(task)
    bddl_path = Path(get_libero_path("bddl_files")) / task_info["problem_folder"] / task_info["bddl_file"]

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=64,
        camera_widths=64,
    )
    try:
        obs = env.reset()
        init_states = suite.get_task_init_states(task_id)
        if len(init_states):
            obs = env.set_init_state(init_states[0])
        zero_action = np.zeros(env.env.action_dim)
        obs, reward, done, info = env.step(zero_action)
        image_shape = None
        if isinstance(obs, dict) and "agentview_image" in obs:
            image_shape = list(obs["agentview_image"].shape)
        return {
            "suite": suite_name,
            "task_id": task_id,
            "task_name": task_info.get("language") or task_info.get("name"),
            "bddl_path": str(bddl_path),
            "ok": True,
            "reward": float(reward),
            "done": bool(done),
            "action_dim": int(env.env.action_dim),
            "agentview_image_shape": image_shape,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--reset-smoke", action="append", default=[])
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path

    run_id = f"{now_stamp()}__libero_preflight"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    events_path = run_dir / "events.jsonl"
    tasks_path = run_dir / "tasks.jsonl"
    benchmark_dict = benchmark.get_benchmark_dict()
    created_at = now_iso()

    run_meta = {
        "run_id": run_id,
        "created_at": created_at,
        "project": "HiBT-LaCET-LIBERO",
        "kind": "libero_resource_preflight",
        "repo_git_sha": git_sha(),
        "suites": args.suites,
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "bddl_root": get_libero_path("bddl_files"),
        "output_dir": str(run_dir),
        "command": " ".join(["experiments/libero/check_libero_preflight.py", *os.sys.argv[1:]]),
    }
    write_json(run_dir / "run.json", run_meta)
    append_jsonl(events_path, {"time": now_iso(), "event": "start", "run_id": run_id})

    suite_summaries = []
    total_tasks = 0
    missing_bddl = []
    init_state_errors = []

    for suite_name in args.suites:
        suite = benchmark_dict[suite_name]()
        suite_rows = []
        for task_id in range(suite.n_tasks):
            task_info = task_to_dict(suite.get_task(task_id))
            bddl_path = Path(get_libero_path("bddl_files")) / task_info["problem_folder"] / task_info["bddl_file"]
            init_states_count = None
            init_state_error = None
            try:
                init_states_count = int(len(suite.get_task_init_states(task_id)))
            except Exception as exc:  # pragma: no cover - depends on local LIBERO data.
                init_state_error = repr(exc)
                init_state_errors.append({"suite": suite_name, "task_id": task_id, "error": init_state_error})
            row = {
                "suite": suite_name,
                "task_id": task_id,
                "task_name": task_info.get("language") or task_info.get("name"),
                "problem_folder": task_info.get("problem_folder"),
                "bddl_file": task_info.get("bddl_file"),
                "bddl_path": str(bddl_path),
                "bddl_exists": bddl_path.is_file(),
                "init_states_file": task_info.get("init_states_file"),
                "init_states_count": init_states_count,
                "init_state_error": init_state_error,
            }
            if not row["bddl_exists"]:
                missing_bddl.append({"suite": suite_name, "task_id": task_id, "bddl_path": str(bddl_path)})
            append_jsonl(tasks_path, row)
            suite_rows.append(row)

        total_tasks += len(suite_rows)
        suite_summaries.append(
            {
                "suite": suite_name,
                "n_tasks": suite.n_tasks,
                "bddl_present": sum(1 for row in suite_rows if row["bddl_exists"]),
                "init_states_loaded": sum(1 for row in suite_rows if row["init_states_count"] is not None),
            }
        )

    reset_smokes = []
    for suite_name, task_id in parse_reset_smoke(args.reset_smoke):
        try:
            result = run_reset_smoke(suite_name, task_id, benchmark_dict, get_libero_path)
        except Exception as exc:  # pragma: no cover - environment dependent.
            result = {"suite": suite_name, "task_id": task_id, "ok": False, "error": repr(exc)}
        reset_smokes.append(result)
        append_jsonl(events_path, {"time": now_iso(), "event": "reset_smoke", **result})

    ok = not missing_bddl and not init_state_errors and all(item.get("ok") for item in reset_smokes)
    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": ok,
        "total_tasks": total_tasks,
        "suites": suite_summaries,
        "missing_bddl": missing_bddl,
        "init_state_errors": init_state_errors,
        "reset_smokes": reset_smokes,
    }
    write_json(run_dir / "summary.json", summary)
    append_jsonl(events_path, {"time": now_iso(), "event": "finish", "ok": ok})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {**run_meta, "ok": ok, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
