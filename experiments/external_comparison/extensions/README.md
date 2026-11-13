# MAESTRO and CABTO evaluation

Benchmark adapters for fresh model generation and execution on assembly and
LIBERO-Long. MAESTRO retains its coding-agent/feedback loop; CABTO uses the
assembly-v2 and complete-skill physical-v3 implementations. Both use
Qwen3.5-9B and the common `pi05_libero` policy. No experiment results are included.

## Setup

Run from this directory:

```bash
python3 -m pip install -r requirements-client.txt
python3 fetch_cabto.py
python3 check_environment.py --cabto-core
```

The fetcher obtains SHA-verified planner files from CABTO commit
`4cc89d501080015334f7e331471724b7cf6da7bc`; see
[third-party notices](THIRD_PARTY_NOTICES.md) for licenses and source attribution.

Use a separate VLM environment with Transformers 5.3.0 and PyTorch 2.7.1
(the recorded CUDA build was cu126):

```bash
python3 -m pip install -r requirements-vlm.txt
python3 download_vlm.py --output MODEL_DIRECTORY
python3 services/serve_vlm.py --model MODEL_DIRECTORY --port 8000 --log-dir VLM_LOG_DIRECTORY
```

The downloader pins `Qwen/Qwen3.5-9B` revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Model weights remain external.

Physical evaluation additionally requires [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
its assets, MuJoCo/robosuite and `openpi-client`. Install the policy environment
from [OpenPI's LIBERO guide](https://github.com/Physical-Intelligence/openpi/tree/fdc03f527881cdfc8ae1a168ed6a20c60edbbbcc/examples/libero)
and obtain its public `gs://openpi-assets/checkpoints/pi05_libero/` checkpoint.
Keep the OpenPI environment separate from the Qwen VLM environment.

```bash
# In the OpenPI policy environment:
python3 services/serve_policy.py --checkpoint POLICY_CHECKPOINT_DIRECTORY --port 8001
# In the simulator/client environment:
python3 check_environment.py --simulation
```

## Evaluate

```bash
python3 run.py --method maestro --mode assembly --output outputs/maestro_assembly --vlm-url http://localhost:8000
python3 run.py --method cabto --mode assembly --output outputs/cabto_assembly --vlm-url http://localhost:8000
python3 run.py --method maestro --mode libero --output outputs/maestro_libero --vlm-url http://localhost:8000 --policy-port 8001
python3 run.py --method cabto --mode libero --output outputs/cabto_libero --vlm-url http://localhost:8000 --policy-port 8001
```

Each assembly command evaluates 60 tasks; each physical command evaluates ten
tasks at ten prescribed starts. Input-state and BDDL fingerprints are checked
before physical model calls. Use `--prepare-only` to prepare configurations
without inference, `--bddl-root` for relocated assets, and `--help` for worker
partitioning and service options. Outputs include generated programs, calls
and execution records. Keep semantic failures in the evaluation denominator;
use a new output directory when changing the protocol.
