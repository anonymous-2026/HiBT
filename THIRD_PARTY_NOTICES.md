# Third-Party Notices

This repository accompanies:

**HiBT: From Hierarchical Concepts to Behavior Trees for Reliable Robot Task
Planning**

It is organized as a compact, reusable research artifact. This file documents
the third-party dependency boundary and what is not redistributed in Git.

## Redistribution Scope

This repository does not vendor full external research codebases, foundation
model weights, large runtime checkpoints, generated run directories, or rollout
media. The released source tree contains project-owned code for:

- planning and behavior-tree compilation under `artifact/`
- Builder/Predictor model code under `planner/`
- command-line entry points under `scripts/`
- optional LIBERO experiment harnesses under `experiments/libero/`
- result aggregation scripts under `analysis/`

Model weights, VLA checkpoints, LIBERO assets, and large training outputs must
be obtained or reproduced separately.

## Python Dependencies

Runtime dependencies are listed in [requirements.txt](requirements.txt). Users
are responsible for complying with the licenses of packages they install.

The core planning artifact uses packages such as:

- `torch`
- `transformers`
- `accelerate`
- `sentencepiece`
- `safetensors`
- `peft`
- `pyyaml`
- `python-dotenv`
- `py_trees`

Optional LIBERO/VLA harnesses may require additional packages and external
installations that are intentionally not bundled in this repository.

## Model Families

Configs and examples reference several model families, including:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

No model weights are distributed in this repository. Users must obtain model
weights from their official distribution channels and comply with the
corresponding licenses and usage terms.

## Benchmark And Simulator Assets

The repository includes compact symbolic requests, schemas, prompts, and small
prototype-bank artifacts used by the planning code. It does not include:

- external simulator installations
- LIBERO source or assets
- OpenPI or OpenVLA source trees
- pi0.5/OpenVLA-OFT checkpoints
- generated videos, GIFs, or rollout frames
- experiment run logs or large result folders

## Attribution Boundary

The artifact implements a paper-specific pipeline that connects:

- language-conditioned task inputs
- hierarchical concept planning
- symbolic plan repair and realization
- deterministic behavior-tree compilation
- executable symbolic evaluation
- optional LIBERO/VLA harness entry points

If you reuse or extend this repository, preserve relevant third-party license
notices for any dependencies, model weights, benchmark assets, or external code
you add.
