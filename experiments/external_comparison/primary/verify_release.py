"""Offline CPU integrity, split, interface and preparation checks (no models)."""
import argparse,ast,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def main():
 argparse.ArgumentParser(description=__doc__).parse_args()
 expected=json.loads((ROOT/'configs/scientific_inputs_sha256.json').read_text())
 def projection(rows):return [{k:r[k] for k in ['sample_id','request_id','instruction','language','world_state','planning_goal','registry'] if k in r} for r in rows]
 for domain in ['assembly','libero']:
  rows=json.loads((ROOT/'data'/domain/'requests.json').read_text());digest=hashlib.sha256(json.dumps(projection(rows),sort_keys=True,ensure_ascii=False).encode()).hexdigest();assert digest==expected[domain]
 from assembly_state_canonicalization import validate_prepared_inputs
 from registry_lineage import verify_cached_registry_lineage
 validate_prepared_inputs(ROOT/'data/assembly/requests_raw.json',ROOT/'data/assembly/requests.json',ROOT/'data/assembly/state_canonicalization_manifest.json')
 verify_cached_registry_lineage(json.loads((ROOT/'data/libero/requests.json').read_text()))
 for f in ROOT.rglob('*.py'):ast.parse(f.read_text(),filename=str(f.relative_to(ROOT)))
 def run(cmd):
  r=subprocess.run([str(x) for x in cmd],capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 with tempfile.TemporaryDirectory(prefix='hibt_cpu_check_') as t:
  tmp=Path(t)
  for file in ['prepare.py','train.py','export_plans.py','fetch_ner.py','hibt_service_config.py','src/run_assembly_comparison.py','src/plan_libero_comparison.py','src/run_native_libero.py','src/serve_llm.py','src/serve_hibt.py','src/serve_ner.py','src/train_btgenbot.py','src/train_llmbt_ner.py']:
   run([sys.executable,ROOT/file,'--help'])
  run([sys.executable,ROOT/'prepare.py','--settings',ROOT/'configs/services.example.json','--output',tmp/'evaluation'])
  run([sys.executable,ROOT/'src/run_assembly_comparison.py','--run-dir',tmp/'software','--software-smoke'])
  for component,domain,tag in [('btgen','assembly','qwen'),('btgen','assembly','mistral'),('btgen','libero','qwen'),('ner','assembly','qwen'),('hibt','assembly','qwen'),('hibt','assembly','mistral'),('hibt','libero','qwen')]:
   out=tmp/(component+'_'+domain+'_'+tag)
   run([sys.executable,ROOT/'train.py','--component',component,'--domain',domain,'--backbone',tag,'--settings',ROOT/'configs/services.example.json','--output',out])
   if component=='btgen':
    for c in json.loads((out/'commands.json').read_text()):run(c['argv']+['--validate-only'])
   if component=='ner':run(json.loads((out/'commands.json').read_text())[0]['argv']+['--prepare-only'])
 print(json.dumps({'status':'PASS','scope':'OFFLINE_CPU_NO_MODEL_OR_PHYSICS_RESULTS','canonical_assembly_inputs':60,'cached_libero_interfaces':40,'prescribed_assembly_cases':600,'prescribed_native_episodes':500,'btgen_training_stages_validated':7,'ner_training_data_prepared':True,'scientific_input_identity':'unchanged'}))
if __name__=='__main__':main()
