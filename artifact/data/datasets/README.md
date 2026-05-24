# Predictor Datasets

This directory stores predictor training data derived from structured plan
examples.

The first exported dataset is:

- `plan_predictor_v1/`

Each JSONL record contains the four fields required by the current training
loader:

- `main_id`
- `question`
- `cot_answer`
- `groundtruth`

Extra metadata such as `problem_id`, `source_path`, `target`, and
`pyramid_json` is also included for debugging and later custom loaders.
