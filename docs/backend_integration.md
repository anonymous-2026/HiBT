# Backend Integration

HiBT separates symbolic planning from backend execution.

## Symbolic Backend

The symbolic backend validates Behavior Trees with a lightweight simulator. It
is used for the 60-instance assembly benchmark and reports Exec, LC, SR, and
generation time.

## Language-Model Backends

The repository supports local and API-backed planners for baseline generation
and concept-interface experiments. See `configuration.md` for model and
checkpoint options.

## Embodied Rollout Backend

For LIBERO-style embodied experiments, HiBT supplies dependency-aware subgoals
and BT structure while a low-level VLA carrier executes actions from visual
observations. Symbolic planning success and physical rollout success are
reported separately.
