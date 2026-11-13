"""Minimal dataset registry for planner experiments."""

from __future__ import annotations

from typing import Any

from planner.local_dataset import PlanLocalDataset


def get(data_cfg: dict[str, Any], split: str):
    """Instantiate a dataset from planner-local config.

    Currently supports only the local JSONL-backed planner dataset.
    """

    data_name = data_cfg.get("data_name", "planlocal")
    if data_name != "planlocal":
        raise ValueError(
            f"Unsupported planner dataset '{data_name}'. Only 'planlocal' is supported."
        )
    return PlanLocalDataset(split=split, config=data_cfg)
