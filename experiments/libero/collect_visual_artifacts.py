#!/usr/bin/env python3
"""Collect experiment visual artifacts into one flat folder."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISUAL_ROOT = Path(os.environ.get("HIBT_VISUAL_ROOT", PROJECT_ROOT / "visual"))
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTS = {".gif", ".mp4"}
ANALYSIS_VISUAL_EXTS = {".png", ".pdf", ".svg"}
INFO_FILES = [
    PROJECT_ROOT / "analysis" / "final_paper_update_packet.md",
    PROJECT_ROOT / "analysis" / "libero_task8_recovery_case.md",
    PROJECT_ROOT / "analysis" / "libero_task8_multiinit_recovery.md",
    PROJECT_ROOT / "analysis" / "libero_task8_multiinit_recovery.csv",
    PROJECT_ROOT / "analysis" / "libero_vla_init0_comparison.md",
    PROJECT_ROOT / "analysis" / "libero_vla_init0_comparison.csv",
    PROJECT_ROOT / "analysis" / "libero_planning_only_latest.md",
    PROJECT_ROOT / "analysis" / "libero_planning_only_latest.csv",
    PROJECT_ROOT / "design" / "vla_guided_execution_checklist_20260525.md",
    PROJECT_ROOT / "design" / "libero_experiment_progress_20260525.md",
    PROJECT_ROOT / "experiments" / "libero" / "README.md",
]


def safe_token(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metric = summary.get("summary_metric")
    return metric if isinstance(metric, dict) else {}


def run_prefix(run_dir: Path) -> str:
    summary = read_summary(run_dir)
    metric = metric_from_summary(summary)
    method = safe_token(str(metric.get("method") or summary.get("method") or summary.get("kind") or "run"))
    suite = safe_token(str(metric.get("suite") or summary.get("suite") or "suite"))
    task = metric.get("task_id", summary.get("task_id", "x"))
    init = metric.get("init_state_id", summary.get("init_state_id", "x"))
    success = metric.get("success", summary.get("success", "na"))
    success_token = "success" if success is True else "fail" if success is False else "na"
    return f"{run_dir.name}__{method}__{suite}_task{task}_init{init}__{success_token}"


def media_kind(path: Path) -> str:
    parts = set(path.parts)
    if "keyframes" in parts:
        return "single_frame"
    if "contact_sheets" in parts:
        return "contact_sheet"
    if "videos" in parts:
        return "video"
    return "media"


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def collect_run_media() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((PROJECT_ROOT / "runs").glob("*")):
        media_dir = run_dir / "media"
        if not media_dir.is_dir():
            continue
        prefix = run_prefix(run_dir)
        for src in sorted(media_dir.rglob("*")):
            if not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                continue
            rel = src.relative_to(media_dir)
            kind = media_kind(src)
            dst_name = safe_token(f"run__{prefix}__{kind}__{rel.with_suffix('').as_posix()}") + ext
            dst = VISUAL_ROOT / dst_name
            size = copy_file(src, dst)
            rows.append({"kind": kind, "source": str(src), "file": dst.name, "bytes": size})
    return rows


def collect_analysis_visuals() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in sorted((PROJECT_ROOT / "analysis").glob("*")):
        if not src.is_file() or src.suffix.lower() not in ANALYSIS_VISUAL_EXTS:
            continue
        dst_name = safe_token(f"analysis__figure__{src.stem}") + src.suffix.lower()
        size = copy_file(src, VISUAL_ROOT / dst_name)
        rows.append({"kind": "analysis_figure", "source": str(src), "file": dst_name, "bytes": size})
    return rows


def collect_info_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in INFO_FILES:
        if not src.is_file():
            continue
        rel = src.relative_to(PROJECT_ROOT)
        dst_name = safe_token(f"info__{rel.with_suffix('').as_posix()}") + src.suffix.lower()
        size = copy_file(src, VISUAL_ROOT / dst_name)
        rows.append({"kind": "info", "source": str(src), "file": dst_name, "bytes": size})
    return rows


def write_index(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["kind"] for row in rows)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    files = {str(row["file"]): row for row in rows}
    key_patterns = [
        "analysis_figure_libero_task8_multiinit_recovery.pdf",
        "analysis_figure_libero_task8_multiinit_recovery.svg",
        "analysis_figure_libero_task8_multiinit_recovery.png",
        "run_20260525_072900_pi05_openpi_libero_smoke_flat_vla_libero_10_task8_init0_fail_video_videos_flat_vla_task8_failure.mp4",
        "run_20260525_072658_pi05_guided_actionseq_libero_actionseq_vla_libero_10_task8_init0_success_video_videos_actionseq_vla_task8_success.mp4",
        "run_20260525_074630_pi05_guided_concept_repaired_libero_concept_repaired_vla_libero_10_task8_init0_success_video_videos_concept_repaired_vla_task8_success.mp4",
        "info_analysis_final_paper_update_packet.md",
        "info_analysis_libero_task8_recovery_case.md",
    ]
    lines = [
        "# Visual Artifact Index",
        "",
        "Flat collection folder for LIBERO experiment visuals. Files are copied from the experiment runpack; original run directories are unchanged.",
        "",
        "## Counts",
        "",
    ]
    for kind, count in sorted(counts.items()):
        lines.append(f"- `{kind}`: {count}")
    lines.extend(
        [
            f"- `total_files`: {len(rows)}",
            f"- `total_size_mb`: {total_bytes / (1024 * 1024):.2f}",
            "",
            "## Naming",
            "",
            "- `run_<run_id>_<method>_<suite_task_init>_<success>_single_frame_...png`: single-frame rollout screenshots.",
            "- `run_..._contact_sheet_...png`: rollout contact sheets.",
            "- `run_..._video_...gif/mp4`: rollout videos or GIF previews.",
            "- `analysis_figure_...pdf/svg/png`: paper-facing result figures.",
            "- `info_...md/csv`: related explanations and result tables.",
            "",
            "## Key Files",
            "",
        ]
    )
    for name in key_patterns:
        if name in files:
            lines.append(f"- `{name}`")
    lines.extend(["", "## Full File List", ""])
    for row in sorted(rows, key=lambda item: (str(item["kind"]), str(item["file"]))):
        lines.append(f"- `{row['file']}`  \n  source: `{row['source']}`")
    (VISUAL_ROOT / "VISUAL_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(collect_run_media())
    rows.extend(collect_analysis_visuals())
    rows.extend(collect_info_files())
    write_index(rows)
    print(json.dumps({"visual_root": str(VISUAL_ROOT), "files_copied": len(rows), "index": str(VISUAL_ROOT / "VISUAL_INDEX.md")}, indent=2))


if __name__ == "__main__":
    main()
