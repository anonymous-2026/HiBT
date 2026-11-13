"""Bind newly trained HiBT Builder/Predictor/banks to the actual model server."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
for tag in ['qwen','mistral','libero-qwen']:p.add_argument('--'+tag,type=Path,required=True,help='Output folder of corresponding train.py --component hibt run')
p.add_argument('--output',type=Path,required=True);a=p.parse_args();backends={}
for tag,d in [('qwen',a.qwen),('mistral',a.mistral),('libero_qwen',a.libero_qwen)]:
 d=d.resolve();c=list((d/'predictor/checkpoints').glob('checkpoint_best_eval-*.pt')) or list((d/'predictor/checkpoints').glob('checkpoint_best-*.pt'))
 if not c:raise FileNotFoundError('Missing trained Predictor checkpoint for '+tag)
 backends[tag]={'predictor_config':str(d/'configs/predictor.yml'),'predictor_checkpoint':str(max(c,key=lambda x:x.stat().st_mtime)),'builder_checkpoint':str(d/'builder_best.pt'),'bank':str(d/'bank.pt')}
requests=json.loads((ROOT/'data/libero/requests.json').read_text())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'backends':backends,'libero_requests':{x['request_id']:x for x in requests}},indent=2)+'\n')
