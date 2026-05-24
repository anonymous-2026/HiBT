"""Minimal environment and device helpers for planner runtime."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_best_gpu(min_memory_mb: int = 1024) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    if num_gpus == 1:
        return "cuda:0"

    free_memory: list[tuple[int, float]] = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        try:
            torch.cuda.reset_peak_memory_stats(i)
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**2)
        except Exception:
            free_mem = props.total_memory / (1024**2)
        free_memory.append((i, free_mem))

    free_memory.sort(key=lambda x: x[1], reverse=True)
    best_gpu, best_free = free_memory[0]
    if best_free < min_memory_mb:
        raise RuntimeError(
            f"No GPU has sufficient free memory. Best GPU {best_gpu} has "
            f"{best_free:.1f} MB free, but {min_memory_mb} MB required."
        )
    return f"cuda:{best_gpu}"


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device(select_best_gpu())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(select_best_gpu())
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs)"
            )
        return torch.device(device)
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device(device)
