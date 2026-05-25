#!/usr/bin/env python3
"""Run a small LIBERO VLA experiment matrix through project-local entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_PY = os.environ.get("LIBERO_PYTHON", sys.executable)
OPENPI_ROOT = os.environ.get("OPENPI_ROOT", "")
OPENPI_PY = os.environ.get(
    "OPENPI_PYTHON",
    str(Path(OPENPI_ROOT) / ".venv" / "bin" / "python") if OPENPI_ROOT else sys.executable,
)
OPENVLA_ROOT = os.environ.get("OPENVLA_ROOT", "")
LIBERO_SRC = os.environ.get("LIBERO_SRC", "")
LIBERO_CONFIG = os.environ.get("LIBERO_CONFIG_PATH", "")
OPENVLA_OFT_CHECKPOINT = os.environ.get("OPENVLA_OFT_LIBERO10_CHECKPOINT", "")


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


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            items.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            items.append(int(part))
    return sorted(dict.fromkeys(items))


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def env_for_carrier(carrier: str, gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    if LIBERO_CONFIG:
        env["LIBERO_CONFIG_PATH"] = env.get("LIBERO_CONFIG_PATH", LIBERO_CONFIG)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if carrier in {"pi05_openpi", "pi05_guided"}:
        openpi_client = f"{OPENPI_ROOT}/packages/openpi-client/src"
        env["PYTHONPATH"] = ":".join(path for path in [openpi_client, env.get("PYTHONPATH", "")] if path)
    elif carrier == "openvla_oft":
        env["PYTHONPATH"] = ":".join(path for path in [OPENVLA_ROOT, LIBERO_SRC, env.get("PYTHONPATH", "")] if path)
    else:
        raise ValueError(f"Unknown carrier: {carrier}")
    return env


def command_for_episode(args: argparse.Namespace, task_id: int, init_state_id: int) -> list[str]:
    common = [
        "--suite",
        args.suite,
        "--task-id",
        str(task_id),
        "--init-state-id",
        str(init_state_id),
        "--max-steps",
        str(args.max_steps),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--replan-steps",
        str(args.replan_steps),
        "--camera-transform",
        args.camera_transform,
        "--output-root",
        args.child_output_root,
    ]
    if args.export_frames:
        common.append("--export-frames")

    if args.carrier == "pi05_openpi":
        return [
            args.libero_python,
            "experiments/libero/run_pi05_openpi_smoke.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            *common,
        ]
    if args.carrier == "pi05_guided":
        return [
            args.libero_python,
            "experiments/libero/run_pi05_guided_rollout.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--method",
            args.guided_method,
            "--stage-steps",
            str(args.stage_steps),
            *common,
        ]
    if args.carrier == "openvla_oft":
        return [
            args.openvla_python,
            "experiments/libero/run_openvla_oft_smoke.py",
            "--checkpoint",
            args.openvla_checkpoint,
            *common,
        ]
    raise ValueError(f"Unknown carrier: {args.carrier}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", choices=["pi05_openpi", "pi05_guided", "openvla_oft"], required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--init-state-ids", default="0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--stage-steps", type=int, default=180)
    parser.add_argument("--guided-method", choices=["actionseq", "concept", "concept_repaired", "concept_nostable", "concept_ruleonly"], default="actionseq")
    parser.add_argument("--camera-transform", choices=["rotate180", "flip_y", "raw"], default="rotate180")
    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--child-output-root", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--libero-python", default=LIBERO_PY)
    parser.add_argument("--openvla-python", default=OPENPI_PY)
    parser.add_argument("--openvla-checkpoint", default=OPENVLA_OFT_CHECKPOINT)
    args = parser.parse_args()

    run_id = f"{now_stamp()}__libero_vla_matrix_{args.carrier}"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    created_at = now_iso()
    task_ids = parse_int_list(args.task_ids)
    init_state_ids = parse_int_list(args.init_state_ids)
    config = {
        "run_id": run_id,
        "created_at": created_at,
        "kind": "libero_vla_matrix",
        "carrier": args.carrier,
        "suite": args.suite,
        "task_ids": task_ids,
        "init_state_ids": init_state_ids,
        "gpu_id": args.gpu_id,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "stage_steps": args.stage_steps,
        "guided_method": args.guided_method if args.carrier == "pi05_guided" else None,
        "camera_transform": args.camera_transform,
        "export_frames": args.export_frames,
    }
    write_json(run_dir / "matrix_config.json", config)

    index_path = PROJECT_ROOT / args.child_output_root / "index.jsonl"
    child_kind = {
        "pi05_openpi": "pi05_openpi_libero_smoke",
        "pi05_guided": "pi05_guided_libero_rollout",
        "openvla_oft": "openvla_oft_libero_smoke",
    }[args.carrier]
    rows = []
    env = env_for_carrier(args.carrier, args.gpu_id)
    ok = True
    for task_id in task_ids:
        for init_state_id in init_state_ids:
            before = len(read_index(index_path))
            cmd = command_for_episode(args, task_id, init_state_id)
            started_at = now_iso()
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
            stdout_path = run_dir / "child_logs" / f"task{task_id:02d}_init{init_state_id:02d}.stdout.txt"
            stderr_path = run_dir / "child_logs" / f"task{task_id:02d}_init{init_state_id:02d}.stderr.txt"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")
            after_rows = read_index(index_path)
            new_children = [row for row in after_rows[before:] if row.get("kind") == child_kind]
            child = new_children[-1] if new_children else None
            row = {
                "time": now_iso(),
                "carrier": args.carrier,
                "suite": args.suite,
                "task_id": task_id,
                "init_state_id": init_state_id,
                "returncode": proc.returncode,
                "started_at": started_at,
                "command": cmd,
                "child_run_id": child.get("run_id") if child else None,
                "child_summary_path": child.get("summary_path") if child else None,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            rows.append(row)
            append_jsonl(run_dir / "children.jsonl", row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            if proc.returncode != 0:
                ok = False
                if args.stop_on_failure:
                    break
        if args.stop_on_failure and not ok:
            break

    summary = {
        **config,
        "ok": ok,
        "num_requested": len(task_ids) * len(init_state_ids),
        "num_completed_processes": sum(1 for row in rows if row["returncode"] == 0),
        "child_runs": [row["child_run_id"] for row in rows if row["child_run_id"]],
    }
    write_json(run_dir / "summary.json", summary)
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "libero_vla_matrix", "created_at": created_at, "ok": ok, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
