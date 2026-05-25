# Experiments

This directory contains reusable experiment entry points and lightweight input
assets. It intentionally does not contain generated run records, large result
folders, checkpoints, videos, GIFs, or local environment caches.

## Included

- `libero/`: optional LIBERO planning-only and VLA rollout harnesses
- `libero/requests/`: extracted LIBERO planning requests with relative BDDL
  paths
- `libero/hier10/`: lightweight scaffold assets for hierarchical LIBERO runs

## Main Artifact Inputs

The symbolic artifact benchmark inputs live under:

- `artifact/data/requests/benchmark_main.json`
- `artifact/data/requests/benchmark_main_metadata.json`
- `artifact/data/requests/benchmark_heldout.json`
- `artifact/data/requests/benchmark_heldout_metadata.json`
- `artifact/data/requests/artifact_request_pool.json`

## Generated Outputs

Generated outputs should be written to ignored directories such as:

- `outputs/`
- `runs/`
- `artifact/data/runtime/`
- `visual/`

Use `analysis/` scripts to aggregate local run directories after reproducing
experiments.
