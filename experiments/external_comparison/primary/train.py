"""Generate or execute the documented learned-component training chains."""
import argparse,json,os,shlex,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component',choices=['btgen','ner','hibt'],required=True);p.add_argument('--domain',choices=['assembly','libero'],default='assembly');p.add_argument('--backbone',choices=['qwen','mistral'],default='qwen');p.add_argument('--settings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true',help='Run actual training; otherwise only write exact commands/configs');a=p.parse_args();s=json.loads(a.settings.read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);commands=[];py=sys.executable
 if a.domain=='libero' and a.backbone!='qwen':raise ValueError('The retained LIBERO protocol uses Qwen only')
 def add(cmd,cwd=ROOT):commands.append({'argv':[str(x) for x in cmd],'cwd':str(cwd)})
 if a.component=='btgen':
  base=s['models'][a.backbone];previous=None
  stages=[]
  if a.domain=='assembly':stages=[('joint3','btgen_joint_v1','assembly_full',3,4,8,None)]
  else:stages=[('assembly240_3','btgen_assembly_v3','assembly',3,4,8,None),('libero3','btgen_libero_v1','libero_shared',3,2,4,'previous')]
  for label,snapshot,data,epochs,batch,grad,initial in stages:
   root=ROOT/'training_sources/frozen'/snapshot;d=root/'data'/data;prefix='btgen_' if data=='libero_shared' else ''
   cmd=[py,root/'src/train_btgenbot.py','--model',base,'--train-data',d/(prefix+'train.jsonl'),'--validation-data',d/(prefix+'validation.jsonl'),'--split-manifest',d/'split_manifest.json','--output-dir',out/label,'--epochs',str(epochs),'--micro-batch-size',str(batch),'--gradient-accumulation',str(grad),'--seed','41714','--max-length','8192']
   if initial:cmd+=['--initial-adapter',previous]
   add(cmd);previous=out/label/'adapter'
  d=ROOT/'data'/a.domain;cmd=[py,ROOT/'src/train_btgenbot.py','--model',base,'--initial-adapter',previous,'--train-data',d/'btgen_train.jsonl','--validation-data',d/'btgen_validation.jsonl','--split-manifest',d/'btgen_split_manifest.json','--output-dir',out/'goal1','--epochs','1','--learning-rate','0.0001','--micro-batch-size','4' if a.domain=='assembly' else '2','--gradient-accumulation','8' if a.domain=='assembly' else '4','--seed','41714','--max-length','8192'];add(cmd)
 elif a.component=='ner':
  d=ROOT/'data/teachers';add([py,ROOT/'src/train_llmbt_ner.py','--model',s['models']['ner_original'],'--train-data',d/'assembly/teacher_train.jsonl','--validation-data',d/'assembly/teacher_validation.jsonl','--split-manifest',d/'assembly/split_manifest.json','--aux-train-data',ROOT/'data/libero/btgen_train.jsonl','--aux-validation-data',ROOT/'data/libero/btgen_validation.jsonl','--aux-split-manifest',ROOT/'data/libero/source_split_manifest.json','--output-dir',out/'training','--epochs','4','--batch-size','32','--learning-rate','0.00003','--max-length','256','--seed','41714','--device','cpu'])
 else:
  import yaml
  source=ROOT/'vendor/hibt';data=ROOT/'training_sources/data'/('hibt_full_verified_v2' if a.domain=='assembly' else 'libero_shared');cfgout=out/'configs';cfgout.mkdir(exist_ok=True)
  for stage in ['builder','predictor']:
   cfg=yaml.safe_load((data/'configs'/f'{a.backbone}_{stage}.yml').read_text());cfg['data']['data_path']=str(data);cfg['evaluation']['data']['data_path']=str(data)
   for key,suffix in [('save_folder',''),('checkpoint_path','checkpoints'),('log_path','logs')]:cfg['log'][key]=str(out/stage/suffix)
   if stage=='builder':cfg['model']['reason_model']['reason_model_name']=s['models'][a.backbone]
   else:cfg['model']['builder'].update(config_path=str(cfgout/'builder.yml'),checkpoint_path=str(out/'builder_best.pt'))
   (cfgout/f'{stage}.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
  add([py,source/'planner/train_builder.py','-c',cfgout/'builder.yml'],source)
  add([py,Path(__file__).resolve(),'--select-checkpoint',out/'builder/checkpoints','--destination',out/'builder_best.pt'])
  add([py,source/'artifact/planning/build_plan_bank.py','--builder-config',cfgout/'builder.yml','--builder-checkpoint',out/'builder_best.pt','--dataset-jsonl',data/'train.jsonl','--output',out/'bank.pt','--device','cuda:0'],source)
  add([py,source/'planner/train_predictor.py','-c',cfgout/'predictor.yml'],source)
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print('\n'.join(' '.join(shlex.quote(x) for x in c['argv']) for c in commands))
 if a.execute:
  for i,c in enumerate(commands):
   env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'vendor/hibt')+os.pathsep+str(ROOT/'src')
   with (out/f'stage_{i}.log').open('w') as log:subprocess.run(c['argv'],cwd=c['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  (out/'training_complete.json').write_text(json.dumps({'status':'TRAINING_COMPLETE','component':a.component,'domain':a.domain,'backbone':a.backbone})+'\n')
if __name__=='__main__':
 if '--select-checkpoint' in sys.argv:
  p=argparse.ArgumentParser();p.add_argument('--select-checkpoint',type=Path);p.add_argument('--destination',type=Path);a=p.parse_args();files=list(a.select_checkpoint.glob('checkpoint_best_eval-*.pt')) or list(a.select_checkpoint.glob('checkpoint_best-*.pt'))
  if not files:raise FileNotFoundError('Training produced no best checkpoint')
  shutil.copy2(max(files,key=lambda x:x.stat().st_mtime),a.destination)
 else:main()
