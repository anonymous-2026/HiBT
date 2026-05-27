# Data Formats

## Planning Request

A planning request contains the natural-language task, symbolic initial state,
target predicate, and domain references needed by the planner.

Typical fields:

- `request_id`
- `instruction`
- `initial_state`
- `target`
- `domain`

## Plan Pyramid

HiBT represents a repaired symbolic plan as:

```text
P = (target, stable_subgoals, method_nodes, action_sequence)
```

Method nodes bind an achieved condition to a grounded action and the stable
subgoals that should hold before executing it.

## Compiled Behavior Tree

Compiled BT artifacts are JSON trees with condition and action leaves. The
evaluator checks structural executability, leaf/action consistency, and whether
symbolic execution reaches the requested target.

## Evaluation Reports

Reports include:

- `Exec`: whether the BT can be parsed, compiled, and stepped
- `LC`: logical coherence between reported actions and BT leaves
- `SR`: symbolic or rollout success rate, depending on the backend
- latency or rollout-cost summaries when available
