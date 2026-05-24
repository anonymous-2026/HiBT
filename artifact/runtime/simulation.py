"""Thin simulation wrapper around the repository-owned minimal BT runtime."""

from __future__ import annotations

import copy
from typing import Any

from .minimal_bt import MinimalBehaviorTreeRuntime


def evaluate_behavior_tree(
    world_state: dict[str, Any], behavior_tree: dict[str, Any]
) -> dict[str, Any]:
    """Run BT simulation and return a normalized result dict."""
    runtime = MinimalBehaviorTreeRuntime(
        world_state=copy.deepcopy(world_state),
        behavior_tree=copy.deepcopy(behavior_tree),
    )
    return runtime.evaluate()


def run_sk_simulation(
    world_state: dict[str, Any], behavior_tree: dict[str, Any]
) -> dict[str, Any]:
    """Backward-compatible alias for existing scripts."""
    return evaluate_behavior_tree(world_state, behavior_tree)
