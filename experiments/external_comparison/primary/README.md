# Complete-method comparison: fresh runs

Evaluation code for HiBT, LLM-BT, BTGenBot, LLM-as-BT-Planner (KIOS), and LLM-OBTEA. All methods receive the same instruction, state, supplied final goal, and complete domain interface. The fixed matrix is **600 assembly evaluations** (2 backbones × 60 tasks × 5 methods), **50 LIBERO initial plans**, and **500 physical episodes** (10 Long tasks × 10 initial states × 5 methods). This directory contains code and task/training inputs; no experiment outcomes or model weights are included.

## Setup

Use Python 3.11; install `requirements-cpu.txt` for offline checks and `requirements-neural.txt` for training/services. Obtain local copies of `Qwen/Qwen2.5-7B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`. Copy `configs/services.example.json` to `services.json` and fill model/checkpoint paths, service URLs, and policy connection fields. Paths must be accessible to the relevant service process.

Physics requires [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), its assets/initial states and compatible robosuite/MuJoCo, plus [OpenPI](https://github.com/Physical-Intelligence/openpi) at `fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc`, its `openpi-client`, and public `pi05_libero` policy/server. Follow those projects' installation instructions. The archived extracted LIBERO source did not retain a Git commit; a fully pinned one-command simulator installation is therefore not claimed.

```bash
python verify_release.py                        # offline CPU checks; no model calls
python prepare.py --settings services.json --output work/evaluation
```

## Training

`train.py` writes commands/configs without training by default. Add `--execute` to train. All supervision and original stage-specific formatters are included. Fresh training requires compute; byte-identical checkpoints across environments are not promised.

```bash
python fetch_ner.py --output models/llmbt_original_ner
# Set models.ner_original in services.json to the downloaded checkpoint.
python train.py --component ner --settings services.json --output work/ner --execute
python train.py --component hibt --domain assembly --backbone qwen --settings services.json --output work/hibt_qwen --execute
python train.py --component hibt --domain assembly --backbone mistral --settings services.json --output work/hibt_mistral --execute
python train.py --component hibt --domain libero --backbone qwen --settings services.json --output work/hibt_libero --execute
python train.py --component btgen --domain assembly --backbone qwen --settings services.json --output work/btgen_qwen --execute
python train.py --component btgen --domain assembly --backbone mistral --settings services.json --output work/btgen_mistral --execute
python train.py --component btgen --domain libero --backbone qwen --settings services.json --output work/btgen_libero --execute
```

HiBT trains Builder → prototype bank → Predictor on 360/72 assembly and 31/6 non-Long LIBERO rows. NER adapts the pinned original LLM-BT Parser for 4 epochs using the same pool, retaining actual token-classification inference; its checkpoint is `work/ner/training/ner_model`. BTGenBot assembly uses 3 shared-pool epochs + 1 explicit-goal epoch. LIBERO uses assembly240/48 initialization (3 epochs), LIBERO31/6 adaptation (3 epochs), then explicit-goal continuation (1 epoch). The initialization formatters infer tasks from instruction/state; final training/evaluation include the supplied goal. Final adapters are `goal1/adapter`; evaluation answers never enter training.

## Services and evaluation

Start each base model using `src/serve_llm.py --model MODEL_DIRECTORY --port PORT --log-dir OUTPUT`, and NER using `src/serve_ner.py` with the same flags. Ports are user-selected. For HiBT:

```bash
python hibt_service_config.py --qwen work/hibt_qwen --mistral work/hibt_mistral --libero-qwen work/hibt_libero --output work/hibt_service.json
python src/serve_hibt.py --source vendor/hibt --config work/hibt_service.json --port YOUR_PORT --log-dir work/services/hibt
```

After checking the settings and starting your model and `pi05_libero` servers:

```bash
python prepare.py --settings services.json --output work/evaluation --activate
python src/run_assembly_comparison.py --run-dir work/evaluation/assembly --model-tag qwen
python src/run_assembly_comparison.py --run-dir work/evaluation/assembly --model-tag mistral
python src/plan_libero_comparison.py --run-dir work/evaluation/libero --phase formal --model-tag qwen
python export_plans.py --run-dir work/evaluation/libero
python src/run_native_libero.py --run-dir work/evaluation/libero --preflight
python src/run_native_libero.py --run-dir work/evaluation/libero
```

Drivers preserve native generation/update procedures through the common benchmark runtime; they do not reproduce each baseline's original robot hardware stack. Assembly applies the identical original state canonicalization/transition implementation to every method. LIBERO reuses each task's initial symbolic plan across 10 physical initial states, retaining semantic failures. Its official goal checker determines success. Limits: 650 environment steps, 180 per skill attempt, 3 attempts, chunk5, settling10, policyseed0; planning uses 36 model calls, output4096, input12000. Policy images rotate180; its prefix uses the original instruction plus selected primitive, without the formal goal. Physical initial-state hashes are recorded for pairing.

Use `--worker-count N --worker-index I` for disjoint CPU partitions. An activated source/config/input protocol is immutable; use a new output directory for changes. Infrastructure errors require diagnosis and explicit `--retry-infrastructure`; semantic failures remain outcomes. `--help` documents all entry points. See `PORTABILITY.md`, `configs/upstream_sources.json`, and upstream licenses for export changes and attribution.
