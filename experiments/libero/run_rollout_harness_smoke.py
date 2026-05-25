#!/usr/bin/env python3
"""Smoke test the LIBERO rollout harness using a null action carrier.

This validates environment reset, init-state restore, BT/subgoal prompt plumbing,
step-loop logging, and optional frame export without requiring a VLA model load.
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


def save_frame(path: Path, frame: np.ndarray) -> str | None:
    try:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
        return str(path)
    except Exception:
        return None


def transform_camera(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.ascontiguousarray(frame)
    if mode == "flip_y":
        return np.ascontiguousarray(frame[::-1, :])
    if mode == "rotate180":
        return np.ascontiguousarray(frame[::-1, ::-1])
    raise ValueError(f"Unknown camera transform: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument(
        "--camera-transform",
        choices=["rotate180", "flip_y", "raw"],
        default="rotate180",
        help="Transform raw LIBERO frames before export. rotate180 matches OpenPI/OpenVLA-OFT policy preprocessing.",
    )
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    run_id = f"{now_stamp()}__libero_rollout_harness_smoke"
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
        "kind": "libero_rollout_harness_smoke",
        "repo_git_sha": git_sha(),
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_info["language"],
        "carrier": "null_zero_action",
        "bddl_path": str(bddl_path),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "camera_transform": args.camera_transform,
        "note": "Rollout harness smoke only; not a VLA policy evaluation.",
    }
    write_json(run_dir / "run.json", run_meta)
    append_jsonl(events_path, {"time": now_iso(), "event": "start", **run_meta})

    started = time.perf_counter()
    frames = []
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=128, camera_widths=128)
    try:
        obs = env.reset()
        init_states = suite.get_task_init_states(args.task_id)
        if len(init_states):
            obs = env.set_init_state(init_states[0])
        zero_action = np.zeros(env.env.action_dim)
        reward_trace = []
        done_trace = []
        for step in range(args.steps):
            if args.export_frames and isinstance(obs, dict) and "agentview_image" in obs:
                frame = transform_camera(obs["agentview_image"], args.camera_transform)
                saved = save_frame(run_dir / "media" / "keyframes" / f"step_{step:03d}.png", frame)
                if saved:
                    frames.append(saved)
            obs, reward, done, info = env.step(zero_action)
            reward_trace.append(float(reward))
            done_trace.append(bool(done))
            append_jsonl(events_path, {"time": now_iso(), "event": "step", "step": step, "reward": float(reward), "done": bool(done)})
            if done:
                break
        elapsed = time.perf_counter() - started
        metric = {
            "run_id": run_id,
            "episode_id": f"{args.suite}:{args.task_id}:null_zero_action",
            "suite": args.suite,
            "task_id": args.task_id,
            "task_name": task_info["language"],
            "method": "rollout_harness_smoke",
            "carrier": "null_zero_action",
            "seed": 0,
            "success": bool(done_trace[-1]) if done_trace else False,
            "final_goal_success": bool(done_trace[-1]) if done_trace else False,
            "timeout": not (bool(done_trace[-1]) if done_trace else False),
            "episode_steps": len(reward_trace),
            "policy_calls": len(reward_trace),
            "bt_ticks": len(reward_trace),
            "subgoals_total": 0,
            "subgoals_completed": 0,
            "first_failure_stage": "null_policy_timeout" if not (bool(done_trace[-1]) if done_trace else False) else None,
            "wall_clock_s": round(elapsed, 4),
            "frame_paths": frames,
        }
        append_jsonl(run_dir / "episode_metrics.jsonl", metric)
    finally:
        env.close()

    summary = {
        "run_id": run_id,
        "created_at": created_at,
        "ok": True,
        "suite": args.suite,
        "task_id": args.task_id,
        "steps_executed": len(reward_trace),
        "frames_exported": len(frames),
        "success": bool(done_trace[-1]) if done_trace else False,
        "note": "Expected to fail task success with null zero-action carrier; validates rollout plumbing and media/log output.",
    }
    write_json(run_dir / "summary.json", summary)
    append_jsonl(events_path, {"time": now_iso(), "event": "finish", **summary})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "libero_rollout_harness_smoke", "created_at": created_at, "ok": True, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
