#!/usr/bin/env python3
"""Build task-8 recovery tables and paper-ready plots."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE = Path(os.environ.get("MPLSTYLE", ""))
OUT_PREFIX = PROJECT_ROOT / "analysis" / "libero_task8_multiinit_recovery"
RUNS = {
    "flat_vla": [
        "20260525_072900__pi05_openpi_libero_smoke",
        "20260525_075047__pi05_openpi_libero_smoke",
        "20260525_075104__pi05_openpi_libero_smoke",
        "20260525_075127__pi05_openpi_libero_smoke",
        "20260525_075149__pi05_openpi_libero_smoke",
    ],
    "actionseq_vla": [
        "20260525_073207__pi05_guided_actionseq_libero",
        "20260525_075222__pi05_guided_actionseq_libero",
        "20260525_075237__pi05_guided_actionseq_libero",
        "20260525_075258__pi05_guided_actionseq_libero",
        "20260525_075318__pi05_guided_actionseq_libero",
    ],
    "concept_repaired_vla": [
        "20260525_074630__pi05_guided_concept_repaired_libero",
        "20260525_075351__pi05_guided_concept_repaired_libero",
        "20260525_075407__pi05_guided_concept_repaired_libero",
        "20260525_075424__pi05_guided_concept_repaired_libero",
        "20260525_075441__pi05_guided_concept_repaired_libero",
    ],
}


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, run_ids in RUNS.items():
        for run_id in run_ids:
            summary_path = PROJECT_ROOT / "runs" / run_id / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metric = summary["summary_metric"]
            rows.append(
                {
                    "method": method,
                    "run_id": run_id,
                    "suite": metric["suite"],
                    "task_id": metric["task_id"],
                    "init_state_id": metric["init_state_id"],
                    "success": bool(metric["success"]),
                    "episode_steps": metric["episode_steps"],
                    "policy_calls": metric["policy_calls"],
                    "first_failure_stage": metric["first_failure_stage"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["method", "run_id", "suite", "task_id", "init_state_id", "success", "episode_steps", "policy_calls", "first_failure_stage"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    order = ["flat_vla", "actionseq_vla", "concept_repaired_vla"]
    out = []
    for method in order:
        group = groups[method]
        successes = [1 if row["success"] else 0 for row in group]
        out.append(
            {
                "method": method,
                "episodes": len(group),
                "successes": sum(successes),
                "success_rate": mean(successes),
                "mean_steps": mean(float(row["episode_steps"]) for row in group),
                "mean_policy_calls": mean(float(row["policy_calls"]) for row in group),
            }
        )
    return out


def write_md(path: Path, rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# LIBERO Task 8 Multi-Init Recovery",
        "",
        "Task: `libero_10:8`, \"put both moka pots on the stove\"",
        "",
        "| Method | Episodes | Successes | SR | Mean Steps | Mean Policy Calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['method']}` | {row['episodes']} | {row['successes']} | {row['success_rate']:.2f} | {row['mean_steps']:.1f} | {row['mean_policy_calls']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per Init State",
            "",
            "| Method | Init | Success | Steps | Policy Calls | Run |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item["method"]), int(item["init_state_id"]))):
        lines.append(
            f"| `{row['method']}` | {row['init_state_id']} | {row['success']} | {row['episode_steps']} | {row['policy_calls']} | `{row['run_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "`concept_repaired_vla` improves task-8 robustness over the flat pi0.5 baseline on this five-init pilot. The repair is execution-aware: placement subgoals are executed before the final stove activation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary_rows: list[dict[str, object]]) -> None:
    if STYLE.is_file():
        plt.style.use(STYLE)
    labels = ["Flat", "ActionSeq", "Repaired"]
    sr = [float(row["success_rate"]) for row in summary_rows]
    steps = [float(row["mean_steps"]) for row in summary_rows]
    colors = ["#787878", "#4C78A8", "#54A24B"]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))
    axes[0].bar(labels, sr, color=colors, width=0.62)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Task 8 SR")
    for idx, value in enumerate(sr):
        axes[0].text(idx, value + 0.035, f"{int(value * 5)}/5", ha="center", va="bottom")

    axes[1].bar(labels, steps, color=colors, width=0.62)
    axes[1].set_ylabel("Mean episode steps")
    axes[1].set_title("Execution length")
    axes[1].set_ylim(0, max(steps) * 1.18)
    for idx, value in enumerate(steps):
        axes[1].text(idx, value + max(steps) * 0.035, f"{value:.0f}", ha="center", va="bottom")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=0)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_PREFIX.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_PREFIX.with_suffix(".png"), bbox_inches="tight", dpi=220)


def main() -> None:
    rows = load_rows()
    summary_rows = summarize(rows)
    write_csv(OUT_PREFIX.with_suffix(".csv"), rows)
    write_md(OUT_PREFIX.with_suffix(".md"), rows, summary_rows)
    plot(summary_rows)
    print(json.dumps({"rows": len(rows), "output_prefix": str(OUT_PREFIX)}, indent=2))


if __name__ == "__main__":
    main()
