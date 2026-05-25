# LIBERO Experiment Entry Points

This directory contains reusable LIBERO planning and rollout harnesses. The
scripts write generated artifacts under `runs/`, which is ignored by Git.

## Environment

Set paths for your local LIBERO and VLA installation:

```bash
export LIBERO_CONFIG_PATH=/path/to/libero_config
export LIBERO_PYTHON=/path/to/python
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

Optional VLA carrier paths:

```bash
export OPENPI_ROOT=/path/to/openpi
export OPENPI_PYTHON=/path/to/openpi/.venv/bin/python
export OPENVLA_ROOT=/path/to/openvla-oft
export LIBERO_SRC=/path/to/LIBERO
export PI05_LIBERO_CHECKPOINT=/path/to/pi05_libero
export OPENVLA_OFT_LIBERO10_CHECKPOINT=/path/to/openvla-oft-libero-10
```

## Planning-Only

```bash
${LIBERO_PYTHON:-python3} experiments/libero/check_libero_preflight.py \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --reset-smoke libero_10:0 \
  --output-root runs

${LIBERO_PYTHON:-python3} experiments/libero/build_libero_planning_requests.py \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --output-dir experiments/libero/requests \
  --output-root runs

${LIBERO_PYTHON:-python3} experiments/libero/run_libero_planning_only.py \
  --requests experiments/libero/requests/all_10task_suites.json \
  --output-root runs \
  --seed 7

python3 analysis/aggregate_libero_planning_only.py \
  --output-prefix analysis/libero_planning_only_latest
```

## VLA Rollouts

Carrier preflight:

```bash
${LIBERO_PYTHON:-python3} experiments/libero/check_vla_carriers.py \
  --output-root runs
```

Run a rollout-harness smoke test without a VLA policy:

```bash
MUJOCO_EGL_DEVICE_ID=0 ${LIBERO_PYTHON:-python3} \
  experiments/libero/run_rollout_harness_smoke.py \
  --suite libero_10 \
  --task-id 0 \
  --steps 8 \
  --export-frames \
  --camera-transform rotate180 \
  --output-root runs
```

Run an OpenPI pi0.5 client against a separately started policy server:

```bash
PYTHONPATH="$OPENPI_ROOT/packages/openpi-client/src:${PYTHONPATH:-}" \
MUJOCO_EGL_DEVICE_ID=0 ${LIBERO_PYTHON:-python3} \
  experiments/libero/run_pi05_openpi_smoke.py \
  --openpi-root "$OPENPI_ROOT" \
  --host 127.0.0.1 \
  --port 8105 \
  --suite libero_10 \
  --task-id 0 \
  --init-state-id 0 \
  --max-steps 520 \
  --num-steps-wait 10 \
  --replan-steps 5 \
  --camera-transform rotate180 \
  --output-root runs
```

Run an OpenVLA-OFT smoke rollout:

```bash
PYTHONPATH="$OPENVLA_ROOT:$LIBERO_SRC:${PYTHONPATH:-}" \
${OPENPI_PYTHON:-python3} experiments/libero/run_openvla_oft_smoke.py \
  --checkpoint "$OPENVLA_OFT_LIBERO10_CHECKPOINT" \
  --suite libero_10 \
  --task-id 0 \
  --init-state-id 0 \
  --max-steps 520 \
  --num-steps-wait 10 \
  --replan-steps 8 \
  --camera-transform rotate180 \
  --output-root runs
```

Batch VLA matrix:

```bash
${LIBERO_PYTHON:-python3} experiments/libero/run_vla_matrix.py \
  --carrier pi05_openpi \
  --suite libero_10 \
  --task-ids 0-2 \
  --init-state-ids 0 \
  --gpu-id 0 \
  --host 127.0.0.1 \
  --port 8105 \
  --max-steps 520 \
  --replan-steps 5
```

Aggregate rollout results:

```bash
python3 analysis/aggregate_libero_rollouts.py \
  --runs-root runs \
  --output-prefix analysis/libero_rollouts_latest
```

The repository includes request files and harness code, not generated run
records, model checkpoints, or rollout media.
