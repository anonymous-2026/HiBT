"""CPU-only import and domain checks; never loads model weights or starts a simulator."""
from pathlib import Path
import argparse,importlib,json,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vlm',action='store_true');p.add_argument('--simulation',action='store_true');p.add_argument('--policy',action='store_true');p.add_argument('--cabto-core',action='store_true');a=p.parse_args()
sys.path[:0]=[str(ROOT/'common'),str(ROOT/'maestro')]
for name in ['numpy','PIL','native_runtime','assembly_registry','run_native_libero','maestro_benchmark.cli']:importlib.import_module(name)
from native_runtime import _world_class
from assembly_registry import make_registry
rows=json.loads((ROOT/'inputs/assembly_formal_requests.json').read_text());assert len(rows)==60
for row in rows:
 assert make_registry(row)==row['registry'];_world_class()(row['world_state'])
if a.vlm:
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 import torch
if a.simulation:
 from libero.libero import benchmark
 from libero.libero.envs import OffScreenRenderEnv
 from openpi_client import image_tools,websocket_client_policy
if a.policy:
 import jax
 from openpi.models import pi0_config,model,tokenizer
 from openpi.policies import policy,libero_policy
if a.cabto_core:
 sys.path.insert(0,str(ROOT/'cabto_libero'))
 from planner_bridge import official_core
 official_core()
print('PASS: CPU imports and60 unchanged assembly registries; no model/simulator calls')
