"""Actual trained HiBT assembly backend and learned LIBERO-domain realization."""
from __future__ import annotations
import argparse,copy,gc,importlib.util,json,os,sys,threading,time,traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--host',default='localhost');p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--log-dir',type=Path,required=True);a=p.parse_args();
import torch
a.log_dir.mkdir(parents=True,exist_ok=True);cfg=json.loads(a.config.read_text());sys.path[:0]=[str(a.source),str(a.source/'artifact'),str(a.source/'artifact/planning')]
from artifact.planning.concept_backend import ConceptBehaviorTreeGenerator
from methods.obtea import formula_to_dnf
from assembly_registry import parse_call,call
from torch.nn import functional as F
backends={};lock=threading.Lock();last=time.monotonic()
def backend(tag):
 if tag not in backends:
  backends.clear();gc.collect();torch.cuda.empty_cache()
  c=cfg['backends'][tag];checkpoint=Path(c['predictor_checkpoint'])
  if not checkpoint.is_file():raise FileNotFoundError('An exact frozen predictor checkpoint is required: '+str(checkpoint))
  backends[tag]=ConceptBehaviorTreeGenerator(predictor_config_path=c['predictor_config'],prototype_bank_path=c['bank'],predictor_checkpoint_path=str(checkpoint),builder_checkpoint_path=c['builder_checkpoint'],device='cuda:0',compare_to_normal=False)
 return backends[tag]
def compile_libero(b,target,state,req):
 registry=req['registry'];dnf=formula_to_dnf(target,registry);question=json.dumps({'language':req['instruction'],'initial_state':state,'target':'; '.join('('+' '.join([parse_call(x)[0]]+parse_call(x)[1])+')' for x in sorted(dnf[0]))},sort_keys=True)
 levels=b._predict_concepts(question);bank=b.prototype_bank;allactions=registry['ground_actions'];byname={x['name']:x for x in allactions};facts=set(registry['facts']);chosen={}
 def choose(literal):
  matches=[x for x in allactions if literal in x['add']]
  if not matches:raise ValueError('No registered realization for interpreted goal '+literal)
  return sorted(matches,key=lambda x:x['name'])[0]
 def add_action(action,ancestors=()):
  if action['name'] in ancestors:raise ValueError('Cyclic domain closure')
  chosen[action['name']]=action
  for fact in action['pre']:
   if fact.startswith('!'):continue
   if fact not in facts:add_action(choose(fact),ancestors+(action['name'],))
 for clause in dnf:
  for literal in clause:
   if literal.startswith('!'):raise ValueError('LIBERO realization currently requires positive achievement goals')
   add_action(choose(literal))
 def score(signature):
  name,_=parse_call(signature);vectors=[]
  for entry in bank['entries']:
   for i,item in enumerate(entry['pyramid_json'][3]['items'][:entry['gt_concepts'][3].shape[0]]):
    if item['name']==name:vectors.append(entry['gt_concepts'][3][i].float())
  if not vectors:return float(levels[3].shape[0])
  mat=torch.stack(vectors);ss=F.cosine_similarity(levels[3][:,None,:],mat[None,:,:],dim=-1).max(dim=1).values;weights=torch.softmax(ss*10,dim=0)
  return float((weights*torch.arange(len(weights))).sum())
 scores={sig:score(sig) for sig in chosen}
 def node(literal,chain=()):
  if literal in chain:raise ValueError('Cyclic support condition')
  action=choose(literal);children=[node(x,chain+(literal,)) for x in sorted(action['pre']) if not x.startswith('!')]
  children += [{'name':'action: '+action['name']}]
  return {'name':'selector: achieve '+literal,'children':[{'name':'target: '+literal},{'name':'sequence: support '+literal,'children':children}]}
 clauses=[]
 for clause in dnf:
  goals=sorted(clause,key=lambda lit:(scores[choose(lit)['name']],lit));clauses.append({'name':'sequence: interpreted goals','children':[node(lit) for lit in goals]})
 tree=clauses[0] if len(clauses)==1 else {'name':'selector: goal alternatives','children':clauses}
 return tree,{'predicted_concept_shapes':[list(x.shape) for x in levels],'learned_option_priority':scores,'interpreted_goal_dnf':[sorted(clause) for clause in dnf],'question':question,'domain_closure':'Goal-to-skill binding and preconditions from shared complete registry; native dependency subtrees retained'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,value,code=200):
  data=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_GET(self):self.send({'status':'READY_FOR_LAZY_MODEL_LOAD','model_tags':list(cfg['backends'])})
 def do_POST(self):
  global last
  try:
   data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));beg=time.monotonic();last=beg
   with lock:
    suite=data.get('suite','assembly');tag=data.get('model_tag','qwen') if suite=='assembly' else 'libero_qwen';b=backend(tag)
    if suite=='assembly':
     try:
      generation,elapsed=b.generate(data['target'],data['world_state']);tree=generation['behavior_tree'];report=generation.get('decode_report',{});report['decoded_plan']=generation.get('decoded_plan')
     except ValueError as exc:
      self.send({'method':'HiBT','model_executed':True,'tree':None,'errors':[repr(exc)],'failure_kind':'semantic_planning_failure'});return
    else:
     req=cfg['libero_requests'][data['request_id']]
     try:tree,report=compile_libero(b,data['target'],data['world_state'],req)
     except ValueError as exc:
      self.send({'method':'HiBT','model_executed':True,'tree':None,'errors':[repr(exc)],'failure_kind':'semantic_planning_failure'});return
    result={'method':'HiBT','model_executed':True,'tree':tree,'planning_ms':1000*(time.monotonic()-beg),'decode_report':report,'provenance':{'backend':cfg['backends'][tag],'model_tag':tag,'native_control_flow':True,'neural_latents_executed':True}}
    with (a.log_dir/'calls.jsonl').open('a') as f:f.write(json.dumps({'request':data,'response':result})+'\n')
    last=time.monotonic();self.send(result)
  except Exception as exc:traceback.print_exc();self.send({'method':'HiBT','model_executed':False,'errors':[repr(exc)]},500)
server=ThreadingHTTPServer((a.host,a.port),Handler);server.timeout=10
while time.monotonic()-last<1800:server.handle_request()
