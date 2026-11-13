"""Local JSONL dataset for planner training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class PlanLocalDataset(Dataset):
    """Local JSONL dataset for planner predictor/builder experiments."""

    def __init__(
        self,
        split: str = "train",
        hf_dataname: str | None = None,
        config: dict | None = None,
    ):
        super().__init__()
        self.split = split
        self.hf_dataname = hf_dataname
        self.config = config if config is not None else {}
        self.samples = self._load_samples()

    def _resolve_path(self) -> Path:
        file_map = self.config.get("data_files", {})
        if self.split in file_map:
            return Path(file_map[self.split]).expanduser().resolve()

        if "jsonl_path" in self.config:
            return Path(self.config["jsonl_path"]).expanduser().resolve()

        if "data_path" not in self.config:
            raise KeyError(
                "PlanLocalDataset requires either 'data_path', 'jsonl_path', or "
                "'data_files' in the dataset config."
            )

        data_dir = Path(self.config["data_path"]).expanduser().resolve()
        return data_dir / f"{self.split}.jsonl"

    def _load_samples(self) -> list[dict[str, Any]]:
        path = self._resolve_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"PlanLocalDataset could not find split '{self.split}' at {path}"
            )

        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(self._normalize_row(payload, line_idx))
        return rows

    def _normalize_row(self, payload: dict[str, Any], line_idx: int) -> dict[str, Any]:
        main_id = payload.get("main_id", f"{self.split}_{line_idx}")
        question = payload.get("question", "")
        cot_answer = payload.get("cot_answer", "")
        groundtruth = payload.get("groundtruth", "")

        extra = {
            key: value
            for key, value in payload.items()
            if key not in {"main_id", "question", "cot_answer", "groundtruth"}
        }

        return {
            "main_id": str(main_id),
            "split": self.split,
            "question": str(question),
            "cot_answer": str(cot_answer),
            "groundtruth": str(groundtruth),
            "sample_info": extra,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]
