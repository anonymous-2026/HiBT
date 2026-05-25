#!/usr/bin/env python3
"""Aggregate LIBERO rollout episode metrics from project-local run directories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_metrics(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/episode_metrics.jsonl")):
        for row in read_jsonl(metrics_path):
            if row.get("carrier") in {"pi05_openpi_server", "openvla_oft_libero10"}:
                row = dict(row)
                row["metrics_path"] = str(metrics_path)
                rows.append(row)
    return rows


def collect_matrix_metrics(matrix_runs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matrix_run in matrix_runs:
        children_path = matrix_run / "children.jsonl"
        if not children_path.is_file():
            raise FileNotFoundError(f"Missing matrix children file: {children_path}")
        for child in read_jsonl(children_path):
            summary_path = Path(child["child_summary_path"])
            metrics_path = summary_path.parent / "episode_metrics.jsonl"
            for row in read_jsonl(metrics_path):
                row = dict(row)
                row["matrix_run_id"] = matrix_run.name
                row["metrics_path"] = str(metrics_path)
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("suite", ""), row.get("carrier", ""), row.get("method", ""))].append(row)

    summary_rows = []
    for (suite, carrier, method), group in sorted(groups.items()):
        successes = [1.0 if row.get("success") else 0.0 for row in group]
        steps = [float(row["episode_steps"]) for row in group if row.get("episode_steps") is not None]
        calls = [float(row["policy_calls"]) for row in group if row.get("policy_calls") is not None]
        latency = [float(row["vla_policy_mean_ms"]) for row in group if row.get("vla_policy_mean_ms") is not None]
        summary_rows.append(
            {
                "suite": suite,
                "carrier": carrier,
                "method": method,
                "episodes": len(group),
                "successes": int(sum(successes)),
                "success_rate": round(mean(successes), 4) if successes else 0.0,
                "mean_steps": round(mean(steps), 2) if steps else None,
                "mean_policy_calls": round(mean(calls), 2) if calls else None,
                "mean_policy_latency_ms": round(mean(latency), 3) if latency else None,
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "carrier", "method", "episodes", "successes", "success_rate", "mean_steps", "mean_policy_calls", "mean_policy_latency_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LIBERO Rollout Aggregation",
        "",
        "| Suite | Carrier | Method | Episodes | Successes | SR | Steps | Policy Calls | Latency ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {carrier} | {method} | {episodes} | {successes} | {success_rate:.3f} | {mean_steps} | {mean_policy_calls} | {mean_policy_latency_ms} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| Run | Suite | Task | Carrier | Success | Steps | Calls | Failure |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(detail_rows, key=lambda item: (item.get("suite", ""), item.get("task_id", -1), item.get("run_id", ""))):
        lines.append(
            f"| {row.get('run_id')} | {row.get('suite')} | {row.get('task_id')} | {row.get('carrier')} | {row.get('success')} | {row.get('episode_steps')} | {row.get('policy_calls')} | {row.get('first_failure_stage')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--matrix-run", action="append", default=[], help="Restrict aggregation to one matrix run directory or run id. Can be repeated.")
    parser.add_argument("--output-prefix", default="analysis/libero_rollouts_latest")
    args = parser.parse_args()

    runs_root = (PROJECT_ROOT / args.runs_root).resolve()
    output_prefix = PROJECT_ROOT / args.output_prefix
    if args.matrix_run:
        matrix_runs = []
        for raw in args.matrix_run:
            path = Path(raw)
            if not path.is_absolute():
                path = runs_root / raw
            matrix_runs.append(path)
        detail_rows = collect_matrix_metrics(matrix_runs)
    else:
        detail_rows = collect_metrics(runs_root)
    summary_rows = summarize(detail_rows)
    write_csv(output_prefix.with_suffix(".csv"), summary_rows)
    write_md(output_prefix.with_suffix(".md"), summary_rows, detail_rows)
    print(json.dumps({"episodes": len(detail_rows), "groups": len(summary_rows), "output_prefix": str(output_prefix)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
