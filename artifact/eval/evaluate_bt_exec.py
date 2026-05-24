#!/usr/bin/env python3
"""Evaluate Exec metric for one-step BT generation."""

from evaluate_bt_metric_common import passes_exec, run_metric


if __name__ == "__main__":
    raise SystemExit(run_metric("exec", passes_exec))
