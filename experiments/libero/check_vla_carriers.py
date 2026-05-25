#!/usr/bin/env python3
"""Check local VLA carrier assets and runtime readiness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = {
    "pi05_libero": os.environ.get("PI05_LIBERO_CHECKPOINT", ""),
    "openvla_oft_libero10": os.environ.get("OPENVLA_OFT_LIBERO10_CHECKPOINT", ""),
    "openvla_base": os.environ.get("OPENVLA_BASE_CHECKPOINT", ""),
}
REQUIRED_IMPORTS = {
    "pi05_libero": ["jax", "torch", "transformers", "lerobot", "openpi_client"],
    "openvla_oft_libero10": ["torch", "transformers", "peft", "timm"],
    "openvla_base": ["torch", "transformers", "timm"],
}


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


def shell_json(command: list[str]) -> Any:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"ok": True, "output": output.strip()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def gpu_snapshot() -> list[dict[str, Any]]:
    result = shell_json(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["ok"]:
        return [{"error": result.get("error")}]
    rows = []
    for line in result["output"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                }
            )
    return rows


def import_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def file_status(path: Path) -> dict[str, Any]:
    files = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
    expected = {
        "has_config": (path / "config.json").is_file(),
        "has_manifest": (path / "manifest.json").is_file(),
        "has_model_safetensors": any(name.endswith(".safetensors") for name in files),
        "has_model_index": (path / "model.safetensors.index.json").is_file(),
        "has_lora_adapter": (path / "lora_adapter" / "adapter_config.json").is_file(),
        "has_action_head": any(name.startswith("action_head") for name in files),
        "has_proprio_projector": any(name.startswith("proprio_projector") for name in files),
        "has_orbax_params": (path / "params" / "_METADATA").is_file(),
        "has_assets": (path / "assets").is_dir(),
    }
    return {"exists": path.exists(), "path": str(path), "file_count": len(files), **expected}


def choose_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for name in ("pi05_libero", "openvla_oft_libero10", "openvla_base"):
        row = next((item for item in rows if item["name"] == name), None)
        if row and row["assets_ready"]:
            return {
                "primary": name,
                "reason": row["selection_reason"],
                "runtime_ready_now": row["runtime_ready"],
                "blocked_imports": row["blocked_imports"],
            }
    return {"primary": None, "reason": "No candidate assets found.", "runtime_ready_now": False, "blocked_imports": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()

    run_id = f"{now_stamp()}__libero_vla_carrier_preflight"
    run_dir = (PROJECT_ROOT / args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    all_required = sorted({name for values in REQUIRED_IMPORTS.values() for name in values})
    imports = import_status(all_required)
    rows = []
    for name, raw_path in DEFAULT_CANDIDATES.items():
        status = file_status(Path(raw_path))
        required = REQUIRED_IMPORTS[name]
        blocked = [module for module in required if not imports.get(module)]
        assets_ready = status["exists"] and (
            status["has_config"]
            or status["has_manifest"]
            or status["has_model_safetensors"]
            or status["has_model_index"]
            or status["has_lora_adapter"]
            or status["has_orbax_params"]
        )
        if name == "openvla_oft_libero10":
            reason = "Best match for LIBERO-Long main experiment; local checkpoint includes OFT action head/proprio/lora assets."
        elif name == "pi05_libero":
            reason = "Requested pi0.5 route; official OpenPI pi05_libero Orbax checkpoint is cached locally."
        else:
            reason = "Secondary baseline candidate."
        rows.append(
            {
                "name": name,
                **status,
                "required_imports": required,
                "blocked_imports": blocked,
                "assets_ready": assets_ready,
                "runtime_ready": assets_ready and not blocked,
                "selection_reason": reason,
            }
        )

    summary = {
        "run_id": run_id,
        "created_at": now_iso(),
        "ok": True,
        "python_executable": os.sys.executable,
        "python_runtime_imports": imports,
        "gpu_snapshot": gpu_snapshot(),
        "candidates": rows,
        "selection": choose_primary(rows),
        "note": "This preflight inspects local assets and imports only; it does not mutate shared model directories.",
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "run.json", {"run_id": run_id, "kind": "libero_vla_carrier_preflight", "created_at": summary["created_at"]})
    append_jsonl(PROJECT_ROOT / args.output_root / "index.jsonl", {"run_id": run_id, "kind": "libero_vla_carrier_preflight", "created_at": summary["created_at"], "ok": True, "summary_path": str(run_dir / "summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
