#!/usr/bin/env python3
"""Run a small LIBERO rollout with OpenVLA-OFT weights.

Set ``OPENVLA_OFT_LIBERO10_CHECKPOINT`` or pass ``--checkpoint`` to the local
OpenVLA-OFT checkpoint directory. The caller is responsible for providing a
Python environment with LIBERO and the OpenVLA-OFT dependencies on
``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = os.environ.get("OPENVLA_OFT_LIBERO10_CHECKPOINT", "")
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
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


def center_crop_resize(frame: np.ndarray, size: int = 224, crop_scale: float = 0.9) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    side = int(min(image.size) * math.sqrt(crop_scale))
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def save_frame(path: Path, frame: np.ndarray) -> str | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(path)
        return str(path)
    except Exception:
        return None


def find_checkpoint_file(ckpt: Path, pattern: str) -> Path:
    matches = [path for path in ckpt.iterdir() if pattern in path.name and "checkpoint" in path.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} checkpoint in {ckpt}, found {len(matches)}")
    return matches[0]


def load_component_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, weights_only=True, map_location="cpu")
    return {key.removeprefix("module."): value for key, value in state.items()}


def normalize_proprio(proprio: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    high = np.asarray(stats["q99"])
    low = np.asarray(stats["q01"])
    return np.clip(np.where(mask, 2 * (proprio - low) / (high - low + 1e-8) - 1, proprio), -1.0, 1.0)


def normalize_and_invert_gripper(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action).copy()
    action[..., -1] = 2 * action[..., -1] - 1
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] *= -1.0
    return action


def load_openvla_oft(ckpt: Path, device: str, suite: str):
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(ckpt),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.vision_backbone.set_num_images_in_input(2)
    model.eval().to(device)
    with (ckpt / "dataset_statistics.json").open("r", encoding="utf-8") as handle:
        model.norm_stats = json.load(handle)
    unnorm_key = suite
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8).to(torch.bfloat16).to(device).eval()
    proprio.load_state_dict(load_component_state_dict(find_checkpoint_file(ckpt, "proprio_projector")))
    action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
    action_head = action_head.to(torch.bfloat16).to(device).eval()
    action_head.load_state_dict(load_component_state_dict(find_checkpoint_file(ckpt, "action_head")))
    return processor, model, proprio, action_head, unnorm_key, round(time.perf_counter() - started, 3)


def predict_action_chunk(
    processor: Any,
    model: torch.nn.Module,
    proprio_projector: torch.nn.Module,
    action_head: torch.nn.Module,
    unnorm_key: str,
    obs: dict[str, Any],
    task: str,
    device: str,
) -> list[np.ndarray]:
    prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
    images = [center_crop_resize(obs["full_image"]), center_crop_resize(obs["wrist_image"])]
    inputs = processor(prompt, images[0]).to(device, dtype=torch.bfloat16)
    wrist_inputs = processor(prompt, images[1]).to(device, dtype=torch.bfloat16)
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
    proprio_stats = model.norm_stats[unnorm_key]["proprio"]
    proprio = normalize_proprio(obs["state"], proprio_stats)
    with torch.inference_mode():
        action, _ = model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=False,
        )
    return [normalize_and_invert_gripper(action[i]) for i in range(len(action))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--camera-transform", choices=["rotate180", "flip_y", "raw"], default="rotate180")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--export-frames", action="store_true")
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_id = f"{now_stamp()}__openvla_oft_libero_smoke"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"

    ckpt = Path(args.checkpoint)
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
        "kind": "openvla_oft_libero_smoke",
        "repo_git_sha": git_sha(),
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "task_name": task_info["language"],
        "carrier": "openvla_oft_libero10",
        "checkpoint": str(ckpt),
        "device": device,
        "camera_transform": args.camera_transform,
        "bddl_path": str(Path(task_info["problem_folder"]) / task_info["bddl_file"]),
    }
    write_json(run_dir / "run.json", run_meta)
    append_jsonl(events_path, {"time": now_iso(), "event": "start", **run_meta})

    failure = None
    frame_paths = []
    policy_latencies_ms = []
    rewards = []
    done = False
    started = time.perf_counter()
    env = None
    try:
        processor, model, proprio_projector, action_head, unnorm_key, load_s = load_openvla_oft(ckpt, device, args.suite)
        append_jsonl(events_path, {"time": now_iso(), "event": "model_loaded", "load_s": load_s, "unnorm_key": unnorm_key})
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
        env.seed(0)
        obs = env.reset()
        init_states = suite.get_task_init_states(args.task_id)
        obs = env.set_init_state(init_states[args.init_state_id])
        action_plan: collections.deque[np.ndarray] = collections.deque()
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
            if not action_plan:
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                query_started = time.perf_counter()
                actions = predict_action_chunk(
                    processor,
                    model,
                    proprio_projector,
                    action_head,
                    unnorm_key,
                    {"full_image": base_img, "wrist_image": wrist_img, "state": state},
                    task_info["language"],
                    device,
                )
                policy_latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
                action_plan.extend(actions[: args.replan_steps])
                append_jsonl(events_path, {"time": now_iso(), "event": "policy_query", "step": step, "latency_ms": round(policy_latencies_ms[-1], 3), "chunk_len": len(actions)})
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
        if env is not None:
            env.close()

    elapsed = time.perf_counter() - started
    success = bool(done) and failure is None
    metric = {
        "run_id": run_id,
        "episode_id": f"{args.suite}:{args.task_id}:openvla_oft:init{args.init_state_id}",
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_info["language"],
        "method": "flat_vla",
        "carrier": "openvla_oft_libero10",
        "init_state_id": args.init_state_id,
        "success": success,
        "final_goal_success": success,
        "timeout": not success and failure is None,
        "episode_steps": len(rewards),
        "policy_calls": len(policy_latencies_ms),
        "bt_ticks": len(rewards),
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
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "openvla_oft_libero_smoke", "created_at": created_at, "ok": failure is None, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
