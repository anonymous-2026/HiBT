#!/usr/bin/env python3
"""Run a small LIBERO rollout against an OpenPI pi0.5 policy server.

Start an OpenPI LIBERO policy server in a separate environment, then point this
client at its host and port. All rollout logs, frames, and metrics are written
inside this repository.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", ""))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
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

    run_id = f"{now_stamp()}__pi05_openpi_libero_smoke"
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
        "kind": "pi05_openpi_libero_smoke",
        "repo_git_sha": git_sha(),
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task_name": task_info["language"],
        "carrier": "pi05_openpi_server",
        "server": {"host": args.host, "port": args.port},
        "camera_transform": args.camera_transform,
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
    done = False
    failure = None
    started = time.perf_counter()
    try:
        env.seed(0)
        obs = env.reset()
        init_states = suite.get_task_init_states(args.task_id)
        obs = env.set_init_state(init_states[args.init_state_id])
        for step in range(args.max_steps + args.num_steps_wait):
            if step < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                rewards.append(float(reward))
                continue

            base_img = transform_camera(obs["agentview_image"], args.camera_transform)
            wrist_img = transform_camera(obs["robot0_eye_in_hand_image"], args.camera_transform)
            if args.export_frames and (step - args.num_steps_wait) % max(1, args.replan_steps) == 0:
                saved = save_frame(run_dir / "media" / "keyframes" / f"step_{step:04d}.png", base_img)
                if saved:
                    frame_paths.append(saved)

            base_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(base_img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            if not action_plan:
                element = {
                    "observation/image": base_img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(task_info["language"]),
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
                        "step": step,
                        "latency_ms": round(policy_latencies_ms[-1], 3),
                        "chunk_len": int(len(action_chunk)),
                    },
                )

            action = action_plan.popleft()
            obs, reward, done, info = env.step(np.asarray(action).tolist())
            rewards.append(float(reward))
            append_jsonl(events_path, {"time": now_iso(), "event": "step", "step": step, "reward": float(reward), "done": bool(done)})
            if done:
                break
    except Exception as exc:
        failure = repr(exc)
        append_jsonl(events_path, {"time": now_iso(), "event": "error", "error": failure})
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    success = bool(done) and failure is None
    metric = {
        "run_id": run_id,
        "episode_id": f"{args.suite}:{args.task_id}:pi05_openpi:init{args.init_state_id}",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_info["language"],
        "method": "flat_vla",
        "carrier": "pi05_openpi_server",
        "seed": 0,
        "init_state_id": args.init_state_id,
        "success": success,
        "final_goal_success": success,
        "timeout": not success and failure is None,
        "episode_steps": len(rewards),
        "policy_calls": len(policy_latencies_ms),
        "bt_ticks": len(rewards),
        "subgoals_total": 0,
        "subgoals_completed": 0,
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
        "policy_calls": len(policy_latencies_ms),
        "steps": len(rewards),
        "frames_exported": len(frame_paths),
        "summary_metric": metric,
    }
    write_json(run_dir / "summary.json", summary)
    append_jsonl(events_path, {"time": now_iso(), "event": "finish", **summary})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "pi05_openpi_libero_smoke", "created_at": created_at, "ok": failure is None, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
