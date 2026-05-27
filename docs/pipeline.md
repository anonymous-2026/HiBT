# HiBT Pipeline

## Inputs

HiBT consumes language-conditioned robot planning instances:

- a task instruction
- an initial symbolic world state
- a target predicate
- a symbolic domain with grounded actions, preconditions, and effects

## Stage 1: Hierarchical Concept Prediction

The inference-time Predictor maps the task instruction and symbolic state to a
fixed-geometry latent concept pyramid. The final paper uses four concept levels
with slot counts `(1, 4, 5, 5)`.

## Stage 2: Prototype-Guided Interface Decoding

A prototype bank retrieves a structural template and slot-level symbolic
candidates. The decoded interface provides the scaffold for downstream symbolic
realization instead of directly emitting a final Behavior Tree.

## Stage 3: Deterministic Realization and Dependency Repair

The realization operator instantiates a grounded plan from the decoded
interface. Dependency closure inserts missing support actions for unmet tool,
grasp, and ordering preconditions before execution.

## Stage 4: Behavior Tree Compilation

The compiler recursively maps repaired method nodes and stable subgoals into an
executable Behavior Tree. The auxiliary action sequence is retained for
reporting and consistency checks, but BT compilation operates on the repaired
hierarchical plan structure.

## Outputs

The code produces compiled BT JSON artifacts, evaluation reports, and optional
planning or rollout summaries. Large runtime outputs and checkpoints should
remain outside Git-tracked source files.
