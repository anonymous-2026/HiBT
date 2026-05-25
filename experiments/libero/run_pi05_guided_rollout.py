#!/usr/bin/env python3
"""Run LIBERO with pi0.5/OpenPI using planning-derived subgoal prompts."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_OUTPUTS = PROJECT_ROOT / "runs" / "20260525_060213__libero_planning_only" / "planning_outputs.jsonl"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
ORDINALS = ("first", "second", "third", "fourth", "fifth")


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


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def transform_camera(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.ascontiguousarray(frame)
    if mode == "flip_y":
        return np.ascontiguousarray(frame[::-1, :])
    if mode == "rotate180":
        return np.ascontiguousarray(frame[::-1, ::-1])
    raise ValueError(f"Unknown camera transform: {mode}")


def save_frame(path: Path, frame: np.ndarray) -> str | None:
    try:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
        return str(path)
    except Exception:
        return None


def plan_lookup_method(method: str) -> str:
    if method == "concept_repaired":
        return "concept"
    return method


def load_plan(plan_outputs: Path, suite: str, task_id: int, method: str) -> dict[str, Any]:
    lookup_method = plan_lookup_method(method)
    with plan_outputs.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("suite") == suite and row.get("task_id") == task_id and row.get("method") == lookup_method:
                row = dict(row)
                row["planner_method"] = lookup_method
                row["execution_method"] = method
                return row
    raise SystemExit(f"No plan found for suite={suite} task_id={task_id} method={method} in {plan_outputs}")


def clean_entity(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"_\d+($|_)", "_", value)
    value = re.sub(r"_\d+$", "", value)
    value = value.replace("_", " ")
    replacements = {
        "flat stove cook region": "stove",
        "flat stove": "stove",
        "wooden cabinet": "cabinet",
        "akita black bowl": "black bowl",
    }
    for raw, clean in replacements.items():
        value = value.replace(raw, clean)
    value = re.sub(r"\b(main table|region)\b", "", value)
    return " ".join(value.split())


def object_phrase(action: dict[str, Any], object_counts: dict[str, int], object_seen: dict[str, int]) -> str:
    raw = action.get("object")
    base = clean_entity(raw)
    if not base:
        return ""
    if object_counts.get(base, 0) <= 1:
        return f"the {base}"
    seen = object_seen.get(base, 0)
    object_seen[base] = seen + 1
    if seen == 0:
        return f"one {base}"
    if seen == 1:
        return f"the other {base}"
    ordinal = ORDINALS[seen] if seen < len(ORDINALS) else f"next"
    return f"the {ordinal} {base}"


def execution_sort_actions(actions: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method != "concept_repaired":
        return actions
    open_actions = [action for action in actions if action.get("kind") == "open"]
    placements = [action for action in actions if action.get("kind") in {"place_in", "place_on"}]
    other_actions = [action for action in actions if action.get("kind") not in {"open", "place_in", "place_on", "turnon", "turnoff", "close"}]
    state_after_placement = [action for action in actions if action.get("kind") in {"turnon", "turnoff", "close"}]
    return open_actions + placements + other_actions + state_after_placement


def subgoal_prompts(plan: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    actions = [action for action in plan.get("actions", []) if action.get("kind") != "direct_macro"]
    actions = execution_sort_actions(actions, plan.get("execution_method", plan.get("method", "")))
    cleaned_objects = [clean_entity(action.get("object")) for action in actions if action.get("object")]
    object_counts = collections.Counter(cleaned_objects)
    object_seen: dict[str, int] = {}
    prompts = []
    for index, action in enumerate(actions):
        kind = action.get("kind")
        obj = object_phrase(action, object_counts, object_seen)
        target = clean_entity(action.get("target") or action.get("display_target"))
        if kind == "place_on":
            prompt = f"pick up {obj} and place it on the {target}"
        elif kind == "place_in":
            prompt = f"pick up {obj} and place it in the {target}"
        elif kind == "open":
            prompt = f"open the {clean_entity(action.get('object'))}"
        elif kind == "close":
            prompt = f"close the {clean_entity(action.get('object'))}"
        elif kind == "turnon":
            prompt = f"turn on the {clean_entity(action.get('object'))}"
        elif kind == "turnoff":
            prompt = f"turn off the {clean_entity(action.get('object'))}"
        else:
            prompt = plan["task_name"]
        if mode == "task_prefix":
            prompt = f"{prompt}; this is part of the task: {plan['task_name']}"
        prompts.append({"subgoal_id": index, "kind": kind, "prompt": prompt, "action": action})
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", ""))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--method", choices=["actionseq", "concept", "concept_repaired", "concept_nostable", "concept_ruleonly"], default="concept")
    parser.add_argument("--plan-outputs", default=str(DEFAULT_PLAN_OUTPUTS))
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--stage-steps", type=int, default=180)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--prompt-mode", choices=["subgoal_only", "task_prefix"], default="task_prefix")
    parser.add_argument("--camera-transform", choices=["rotate180", "flip_y", "raw"], default="rotate180")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--export-frames", action="store_true")
    args = parser.parse_args()

    openpi_root = Path(args.openpi_root)
    sys.path.insert(0, str(openpi_root / "packages" / "openpi-client" / "src"))
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    plan = load_plan(Path(args.plan_outputs), args.suite, args.task_id, args.method)
    prompts = subgoal_prompts(plan, args.prompt_mode)
    if not prompts:
        raise SystemExit(f"Plan method {args.method} has no executable subgoal prompts")

    run_id = f"{now_stamp()}__pi05_guided_{args.method}_libero"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    task_info = dict(task._asdict()) if hasattr(task, "_asdict") else {
        "language": getattr(task, "language", None),
        "problem_folder": getattr(task, "problem_folder"),
        "bddl_file": getattr(task, "bddl_file"),
    }
    bddl_path = Path(get_libero_path("bddl_files")) / task_info["problem_folder"] / task_info["bddl_file"]
    created_at = now_iso()
    run_meta = {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "pi05_guided_libero_rollout",
        "repo_git_sha": git_sha(),
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task_name": task_info["language"],
        "method": f"{args.method}_vla",
        "planner_method": plan.get("planner_method", args.method),
        "execution_method": args.method,
        "carrier": "pi05_openpi_server",
        "server": {"host": args.host, "port": args.port},
        "camera_transform": args.camera_transform,
        "prompt_mode": args.prompt_mode,
        "subgoal_prompts": prompts,
        "bddl_path": str(Path(task_info["problem_folder"]) / task_info["bddl_file"]),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    }
    write_json(run_dir / "run.json", run_meta)
    append_jsonl(events_path, {"time": now_iso(), "event": "start", **run_meta})

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    frame_paths = []
    policy_latencies_ms = []
    rewards = []
    subgoal_metrics = []
    done = False
    failure = None
    started = time.perf_counter()
    try:
        env.seed(0)
        obs = env.reset()
        init_states = suite.get_task_init_states(args.task_id)
        obs = env.set_init_state(init_states[args.init_state_id])
        for wait_step in range(args.num_steps_wait):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            rewards.append(float(reward))
            append_jsonl(events_path, {"time": now_iso(), "event": "wait_step", "step": wait_step, "reward": float(reward), "done": bool(done)})

        total_policy_steps = 0
        for subgoal in prompts:
            if done or total_policy_steps >= args.max_steps:
                break
            subgoal_start_step = len(rewards)
            subgoal_calls_before = len(policy_latencies_ms)
            subgoal_frame_paths = []
            action_plan.clear()
            local_steps = 0
            append_jsonl(events_path, {"time": now_iso(), "event": "subgoal_start", **subgoal})
            while local_steps < args.stage_steps and total_policy_steps < args.max_steps:
                base_img = transform_camera(obs["agentview_image"], args.camera_transform)
                wrist_img = transform_camera(obs["robot0_eye_in_hand_image"], args.camera_transform)
                if args.export_frames and local_steps % max(1, args.replan_steps) == 0:
                    frame_path = run_dir / "media" / "keyframes" / f"subgoal_{subgoal['subgoal_id']:02d}_step_{local_steps:04d}.png"
                    saved = save_frame(frame_path, base_img)
                    if saved:
                        frame_paths.append(saved)
                        subgoal_frame_paths.append(saved)

                base_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                if not action_plan:
                    element = {
                        "observation/image": base_img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": subgoal["prompt"],
                    }
                    query_started = time.perf_counter()
                    result = client.infer(element)
                    policy_latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
                    action_chunk = result["actions"]
                    if len(action_chunk) < args.replan_steps:
                        raise RuntimeError(f"Policy returned {len(action_chunk)} actions, need {args.replan_steps}")
                    action_plan.extend(action_chunk[: args.replan_steps])
                    append_jsonl(
                        events_path,
                        {
                            "time": now_iso(),
                            "event": "policy_query",
                            "subgoal_id": subgoal["subgoal_id"],
                            "prompt": subgoal["prompt"],
                            "step": len(rewards),
                            "latency_ms": round(policy_latencies_ms[-1], 3),
                            "chunk_len": int(len(action_chunk)),
                        },
                    )

                action = action_plan.popleft()
                obs, reward, done, info = env.step(np.asarray(action).tolist())
                rewards.append(float(reward))
                local_steps += 1
                total_policy_steps += 1
                append_jsonl(
                    events_path,
                    {
                        "time": now_iso(),
                        "event": "step",
                        "subgoal_id": subgoal["subgoal_id"],
                        "step": len(rewards),
                        "reward": float(reward),
                        "done": bool(done),
                    },
                )
                if done:
                    break
            subgoal_metric = {
                "subgoal_id": subgoal["subgoal_id"],
                "kind": subgoal["kind"],
                "prompt": subgoal["prompt"],
                "steps": len(rewards) - subgoal_start_step,
                "policy_calls": len(policy_latencies_ms) - subgoal_calls_before,
                "final_goal_done_after_subgoal": bool(done),
                "frame_paths": subgoal_frame_paths,
            }
            subgoal_metrics.append(subgoal_metric)
            append_jsonl(run_dir / "subgoal_metrics.jsonl", subgoal_metric)
            append_jsonl(events_path, {"time": now_iso(), "event": "subgoal_finish", **subgoal_metric})
            if done:
                break
    except Exception as exc:
        failure = repr(exc)
        append_jsonl(events_path, {"time": now_iso(), "event": "error", "error": failure})
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    success = bool(done) and failure is None
    method_name = f"{args.method}_vla"
    metric = {
        "run_id": run_id,
        "episode_id": f"{args.suite}:{args.task_id}:{method_name}:init{args.init_state_id}",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_info["language"],
        "method": method_name,
        "planner_method": plan.get("planner_method", args.method),
        "execution_method": args.method,
        "carrier": "pi05_openpi_server",
        "seed": 0,
        "init_state_id": args.init_state_id,
        "success": success,
        "final_goal_success": success,
        "timeout": not success and failure is None,
        "episode_steps": len(rewards),
        "policy_calls": len(policy_latencies_ms),
        "bt_ticks": sum(item["steps"] for item in subgoal_metrics),
        "subgoals_total": len(prompts),
        "subgoals_completed": next((item["subgoal_id"] + 1 for item in subgoal_metrics if item["final_goal_done_after_subgoal"]), len(subgoal_metrics) if not failure else 0),
        "first_failure_stage": failure or (None if success else "timeout"),
        "wall_clock_s": round(elapsed, 4),
        "vla_policy_mean_ms": round(float(np.mean(policy_latencies_ms)), 3) if policy_latencies_ms else None,
        "vla_policy_p95_ms": round(float(np.percentile(policy_latencies_ms, 95)), 3) if policy_latencies_ms else None,
        "camera_transform": args.camera_transform,
        "frame_paths": frame_paths,
    }
    append_jsonl(run_dir / "episode_metrics.jsonl", metric)
    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": failure is None,
        "success": success,
        "failure": failure,
        "method": method_name,
        "policy_calls": len(policy_latencies_ms),
        "steps": len(rewards),
        "subgoals": subgoal_metrics,
        "frames_exported": len(frame_paths),
        "summary_metric": metric,
    }
    write_json(run_dir / "summary.json", summary)
    append_jsonl(events_path, {"time": now_iso(), "event": "finish", **summary})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "pi05_guided_libero_rollout", "created_at": created_at, "ok": failure is None, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
