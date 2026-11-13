#!/usr/bin/env python3
"""Evaluate proxy Logical Coherence metric for one-step BT generation."""

from evaluate_bt_metric_common import passes_lc, run_metric


if __name__ == "__main__":
    raise SystemExit(run_metric("lc", passes_lc))
