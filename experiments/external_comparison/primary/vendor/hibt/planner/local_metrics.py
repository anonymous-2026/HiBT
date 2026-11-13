"""Local JSONL metric sink replacing cloud-only logging for reconstruction."""
import json,time,uuid
from pathlib import Path
from types import SimpleNamespace
_run=None
_path=None
def init(*,project=None,experiment_name=None,config=None,**kwargs):
 global _run,_path
 _run=SimpleNamespace(id='local-'+uuid.uuid4().hex[:12]);cfg=config or {};d=Path(cfg.get('log',{}).get('log_path','rebuild_local_logs'));d.mkdir(parents=True,exist_ok=True);_path=d/'metrics.jsonl'
 _path.write_text(json.dumps({'event':'init','id':_run.id,'project':project,'config':cfg},default=str)+'\n');return _run
def get_run():return _run
def log(metrics,step=None,**kwargs):
 if _path:
  with _path.open('a') as f:f.write(json.dumps({'event':'metrics','step':step,'time':time.time(),'metrics':metrics},default=str)+'\n')
def finish(*args,**kwargs):
 if _path:
  with _path.open('a') as f:f.write(json.dumps({'event':'finish','time':time.time()})+'\n')
