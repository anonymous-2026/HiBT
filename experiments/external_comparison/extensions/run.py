"""Run fresh MAESTRO/CABTO generation and assembly or physical evaluation."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--method',choices=['maestro','cabto'],required=True)
 p.add_argument('--mode',choices=['assembly','libero'],required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--vlm-url',required=True,help='Base URL for the supplied VLM service')
 p.add_argument('--model',default='Qwen/Qwen3.5-9B')
 p.add_argument('--policy-host',default='localhost');p.add_argument('--policy-port',type=int,default=8001)
 p.add_argument('--openpi-client-source',type=Path,help='Optional source path; omit when openpi-client is installed')
 p.add_argument('--bddl-root',type=Path,help='Optional relocated LIBERO BDDL root')
 p.add_argument('--worker-count',type=int,default=1);p.add_argument('--worker-index',type=int,default=0)
 p.add_argument('--prepare-only',action='store_true',help='Write/verify configs and invocation without contacting services')
 a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if not 0<=a.worker_index<a.worker_count:raise ValueError('Invalid worker partition')
 physical=json.loads((ROOT/'configs/native_physics.json').read_text());physical['openpi_client_source']=str(a.openpi_client_source.resolve()) if a.openpi_client_source else None
 if a.bddl_root:physical['bddl_root']=str(a.bddl_root.resolve())
 native_config=out/'native_physics.json'
 def immutable(path,value):
  if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Config changed: use a fresh output directory')
  if not path.exists():path.write_text(json.dumps(value,indent=2)+'\n')
 immutable(native_config,physical)
 if a.method=='maestro':
  cmd=[sys.executable,'-m','maestro_benchmark.cli','--mode',a.mode,'--phase','formal','--requests-file',str(ROOT/'inputs'/('assembly_formal_requests.json' if a.mode=='assembly' else 'libero_formal_requests.json')),'--config',str(ROOT/'configs/maestro.json'),'--output-dir',str(out),'--endpoint',a.vlm_url,'--model',a.model,'--native-source',str(ROOT/'common'),'--worker-count',str(a.worker_count),'--worker-index',str(a.worker_index)]
  if a.mode=='assembly':cmd+=['--evaluators-file',str(ROOT/'inputs/assembly_formal_evaluators.json')]
  else:cmd+=['--native-config',str(native_config),'--policy-host',a.policy_host,'--policy-port',str(a.policy_port),'--init-ids',*map(str,range(10))]
 else:
  which='cabto_assembly' if a.mode=='assembly' else 'cabto_libero';config=json.loads((ROOT/'configs'/f'{which}.json').read_text())
  config.update(shared_source=str(ROOT/'common'),vlm_url=a.vlm_url,expected_model_name=a.model,input_manifest=str(ROOT/'inputs/manifest.json'),requests_file=str(ROOT/'inputs'/('assembly_formal_requests.json' if a.mode=='assembly' else 'libero_formal_requests.json')))
  if a.mode=='assembly':config['evaluators_file']=str(ROOT/'inputs/assembly_formal_evaluators.json')
  else:config.update(physical,policy_host=a.policy_host,policy_port=a.policy_port)
  immutable(out/'config.json',config)
  cmd=[sys.executable,str(ROOT/which/'run_cabto.py'),'--run-dir',str(out),'--mode',a.mode,'--worker-count',str(a.worker_count),'--worker-index',str(a.worker_index)]
 env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'maestro'),str(ROOT/'common'),env.get('PYTHONPATH','')]);env['PYTHONDONTWRITEBYTECODE']='1'
 immutable(out/f'invocation_{a.worker_index}.json',{'argv':cmd,'method':a.method,'mode':a.mode,'fresh_generation_required':True,'expected_cases':60 if a.mode=='assembly' else 100})
 if a.prepare_only:print(json.dumps({'status':'PREPARED_NO_MODEL_CALLS','argv':cmd},indent=2));return
 if a.mode=='libero':
  from check_assets import verify
  verify(a.bddl_root)
 subprocess.run(cmd,env=env,check=True)
if __name__=='__main__':main()
