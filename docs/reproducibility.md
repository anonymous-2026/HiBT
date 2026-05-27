# Reproducibility Notes

## Repository-Local Checks

Run the quick compile/evaluation path:

```bash
bash examples/quick_compile_and_eval.sh
```

This validates a compact plan artifact with the lightweight symbolic runtime.

## Concept Components

Concept checkpoints are not committed. To rebuild them from repository data,
use the training command in `quickstart.md` and write runtime artifacts under
`artifact/data/runtime/`, which is ignored by Git.

## Evaluation Scope

The paper reports:

- a 60-instance symbolic robotic-assembly benchmark
- realization ablations and interface perturbations
- LIBERO planning and embodied rollout summaries

Symbolic BT metrics and physical rollout metrics should not be merged into a
single success metric because the latter also depends on perception, low-level
control, simulator horizon, and carrier execution.
