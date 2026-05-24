#!/usr/bin/env python3
"""Evaluate Success Rate metric for one-step BT generation."""

from evaluate_bt_metric_common import passes_sr, run_metric


if __name__ == "__main__":
    raise SystemExit(run_metric("sr", passes_sr))
