"""Prepare new immutable evaluation configs; never starts model services."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=['hibt','llmbt','btgenbot','kios','obtea']
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--settings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--activate',action='store_true',help='Authorize actual evaluation after reviewing services/checkpoints');a=p.parse_args();s=json.loads(a.settings.read_text());out=a.output.resolve()
 if any((out/x/'protocol.json').exists() or (out/x/'protocol_formal.json').exists() for x in ['assembly','libero']):raise ValueError('An existing protocol cannot be reconfigured; choose a new output')
 for domain in ['assembly','libero']:
  v=json.loads((ROOT/'configs'/f'{domain}.json').read_text());v['protocol_frozen']=a.activate;v['model_settings']={}
  for tag in v['model_tags']:
   v['model_settings'][tag]={'llm_url':s['services'][tag],'hibt_url':s['services']['hibt'],'ner_url':s['services']['ner'],'expected_model_name':{'qwen':'Qwen2.5-7B-Instruct','mistral':'Mistral-7B-Instruct-v0.3'}[tag],'hibt_model_tag':tag if domain=='assembly' else 'libero_qwen','btgenbot_adapter_checkpoint':s['checkpoints'][f'btgen_{tag}_{domain}'],'ner_checkpoint':s['checkpoints']['ner']}
  for key,filename in [('requests_file','requests.json'),('evaluators_file','evaluators.json'),('demonstrations_file','demonstrations.json'),('training_pool_manifest','btgen_split_manifest.json')]:v[key]=str(ROOT/'data'/domain/filename)
  if domain=='assembly':
   v.update(assembly_state_contract='original_kios_state_canonicalization_v1',n_requests=60,raw_requests_file=str(ROOT/'data/assembly/requests_raw.json'),state_canonicalization_manifest=str(ROOT/'data/assembly/state_canonicalization_manifest.json'))
  else:
   v.update(input_manifest=str(ROOT/'data/libero/manifest.json'),plans_file=str(out/'libero/planning_outputs.jsonl'),llm_url=s['services']['qwen'],ner_url=s['services']['ner'],ner_checkpoint=s['checkpoints']['ner'],expected_model_name='Qwen2.5-7B-Instruct',policy_host=s['policy']['host'],policy_port=s['policy']['port'],bddl_root=s['bddl_root'])
  if a.activate:
   text=json.dumps(v)
   if 'YOUR_' in text or 'PATH_TO_' in text:raise ValueError('Replace every settings placeholder before activation')
   if domain=='libero' and not isinstance(v['policy_port'],int):raise ValueError('policy.port must be an integer')
  write(out/domain/'config.json',v)
 print(json.dumps({'prepared':True,'activated':a.activate,'assembly_cases':600,'libero_initial_plans':50,'libero_episodes':500}))
if __name__=='__main__':main()
