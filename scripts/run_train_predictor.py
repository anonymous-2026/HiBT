#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_TRAIN_SCRIPT = (
    REPO_ROOT
    / "planner"
    / "train_predictor.py"
)

if __name__ == "__main__":
    sys.argv[0] = str(PLANNER_TRAIN_SCRIPT)
    runpy.run_path(str(PLANNER_TRAIN_SCRIPT), run_name="__main__")
