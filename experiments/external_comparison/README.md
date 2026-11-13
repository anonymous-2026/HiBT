# External-method evaluation scripts

Benchmark adapters for HiBT, LLM-BT, BTGenBot, LLM-as-BT-Planner/KIOS,
LLM-OBTEA, MAESTRO and CABTO on common assembly and LIBERO tasks.

- [Primary methods: setup, training and evaluation](primary/README.md)
- [MAESTRO and CABTO: setup and evaluation](extensions/README.md)

The package contains source, configurations and task/training inputs.
Install the dependencies and provide the model/checkpoint paths described
in the guides, then run the evaluation commands in fresh output directories.
Results, logs, model weights and paper PDFs are not included.

```bash
python3 experiments/external_comparison/verify_release.py
python3 experiments/external_comparison/summarize.py --help
python3 experiments/external_comparison/summarize.py \
  --primary-assembly NEW_RUN/assembly --primary-libero NEW_RUN/libero
```

`summarize.py` analyzes your generated run records, including paired execution
costs. It does not contain saved experiment outcomes.
