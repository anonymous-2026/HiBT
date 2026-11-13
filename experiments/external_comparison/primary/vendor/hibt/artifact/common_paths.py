from __future__ import annotations

import os
import sys
from pathlib import Path


ARTIFACT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_ROOT.parent

ARTIFACT_PLANNING_DIR = ARTIFACT_ROOT / "planning"
ARTIFACT_EVAL_DIR = ARTIFACT_ROOT / "eval"
ARTIFACT_CONFIGS_DIR = ARTIFACT_ROOT / "configs"
ARTIFACT_DATA_DIR = ARTIFACT_ROOT / "data"
ARTIFACT_RUNTIME_DIR = ARTIFACT_ROOT / "runtime"
BT_DATA_DIR = ARTIFACT_DATA_DIR


def _add(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def bootstrap_runtime() -> None:
    _add(ARTIFACT_EVAL_DIR)
    _add(ARTIFACT_PLANNING_DIR)
    _add(ARTIFACT_ROOT)
    _add(REPO_ROOT)
    os.environ.setdefault("BT_DATA_DIR", str(BT_DATA_DIR))


def bootstrap_planner_runtime() -> None:
    _add(ARTIFACT_EVAL_DIR)
    _add(ARTIFACT_PLANNING_DIR)
    _add(ARTIFACT_ROOT)
    _add(REPO_ROOT)


def bootstrap_all() -> None:
    bootstrap_runtime()
    bootstrap_planner_runtime()
