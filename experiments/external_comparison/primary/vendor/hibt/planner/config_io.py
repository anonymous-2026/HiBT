"""YAML configuration loading utilities for the planner package.

This is the minimal self-owned replacement for the upstream
configuration helpers used by the current training/inference mainline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


class IncludeLoader(yaml.SafeLoader):
    """YAML loader with ``!include`` support."""

    def __init__(self, stream):
        self._root = (
            os.path.dirname(stream.name) if hasattr(stream, "name") else os.getcwd()
        )
        super().__init__(stream)


def _include_constructor(loader: IncludeLoader, node: yaml.Node) -> Any:
    value = loader.construct_scalar(node)

    if ":" in value and not value.startswith("/"):
        parts = value.rsplit(":", 1)
        if len(parts) == 2 and not parts[0].endswith("\\"):
            filepath, key = parts
        else:
            filepath, key = value, None
    else:
        filepath, key = value, None

    if not os.path.isabs(filepath):
        filepath = os.path.join(loader._root, filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        content = yaml.load(f, IncludeLoader)

    if key is not None:
        for k in key.split("."):
            content = content[k]

    return content


IncludeLoader.add_constructor("!include", _include_constructor)


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_in_string(value: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        if var_name in os.environ:
            return os.environ[var_name]
        if default is not None:
            return default
        raise KeyError(
            f"Missing required environment variable '{var_name}' while loading config"
        )

    return _ENV_VAR_PATTERN.sub(replacer, value)


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _expand_env(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, str):
        return _expand_env_in_string(obj)
    return obj


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return _expand_env(yaml.load(f, IncludeLoader))


_STORAGE_ROOT_KEYS = ("save_folder", "checkpoint_path", "log_path")


def apply_storage_root(config: dict, storage_root: str | os.PathLike) -> dict:
    root = Path(storage_root)
    log_cfg = config["log"]
    for key in _STORAGE_ROOT_KEYS:
        raw = log_cfg[key]
        p = Path(raw)
        if p.is_absolute():
            continue
        log_cfg[key] = str(root / p)
    return config


def print_storage_paths(config: dict, storage_root: str | os.PathLike) -> None:
    shown = str(storage_root)
    cwd = Path.cwd().resolve()
    print(f"[STORAGE] storage_root = {shown!r} (cwd={cwd})")
    log_cfg = config["log"]
    width = max(len(k) for k in _STORAGE_ROOT_KEYS)
    for key in _STORAGE_ROOT_KEYS:
        val = log_cfg[key]
        abs_path = Path(val).expanduser()
        if not abs_path.is_absolute():
            abs_path = (cwd / abs_path).resolve()
        print(f"[STORAGE]   {key:<{width}s} = {val}")
        print(f"[STORAGE]   {' ' * width}   (absolute: {abs_path})")
