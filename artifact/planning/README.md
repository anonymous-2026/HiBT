# Planning Bridge

This directory defines the intermediate planning contract between the predictor
and the behavior-tree compiler.

`plan_schema.json` is the domain-level schema that the predictor should emit
for assembly tasks. A deterministic compiler converts that structured plan into
an executable behavior-tree skeleton.

## Purpose

The schema separates three concerns that were previously mixed into a single
LLM output:

1. Hierarchical planning: what stable subgoals are needed.
2. Method selection: which grounded action achieves each subgoal.
3. BT compilation: how those plans become `selector/target/sequence/...`.

The predictor should emit levels 0-3 of the pyramid. The final behavior tree
is produced by the deterministic compiler, not by direct text generation.

## Levels

The schema fixes exactly four levels:

1. `goal`
   - Exactly one final grounded predicate.
2. `stable_subgoals`
   - Only subgoals that should remain true under later execution.
3. `methods`
   - One grounded action method per achieved goal/subgoal.
4. `actions`
   - Final ordered executable action sequence.

## Stable target rule

Stable subgoals are essential for repeated-tick execution semantics. Examples:

- `put_down(hand, tool, part)` -> stable target: `is_empty(tool)`
- `unload_tool(hand, tool)` -> stable target: `is_equippable(tool)`
- `load_tool(hand, tool)` -> stable target: `hold(hand, tool)`

Do not place transient conditions such as `is_empty(hand)` for
`unload_tool(hand, tool)` if later actions will invalidate them.

## Next step

After this schema is fixed, the next concrete step is:

1. create hand-labeled plan examples
2. implement `compile_plan_to_bt.py`
3. validate `hand-labeled plan -> BT -> execution`

The first batch of hand-labeled examples lives in:

- [plan_examples_v1.json](./plan_examples_v1.json)

An expanded validated batch can be regenerated with:

- [expand_plan_examples.py](./expand_plan_examples.py)

and is written to:

- [plan_examples_v2.json](./plan_examples_v2.json)

## Candidate-pool scan

For artifact-internal benchmark inspection, the repository keeps a
self-contained scanner:

- [scan_full_task_candidates.py](./scan_full_task_candidates.py)

It analyzes the current local `plan_examples_v2.json` pool only.

External scans that depended on the old private KIOS repository are intentionally
not part of the portable artifact workflow.
