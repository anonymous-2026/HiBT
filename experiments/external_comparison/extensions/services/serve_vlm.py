"""Image-capable local endpoint; each request and response is durably recorded."""
import argparse,base64,hashlib,io,json,os,threading,time,uuid
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
class InputBudgetExhausted(ValueError):
 pass
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--port',type=int,default=8000);p.add_argument('--host',default='localhost');p.add_argument('--device',default='cuda');p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--idle-seconds',type=int,default=14400);a=p.parse_args()
identity_path=Path(a.model)/'download_identity.json'
if not identity_path.is_file():raise ValueError('Download the pinned model with download_vlm.py first')
artifact_identity=json.loads(identity_path.read_text())
if artifact_identity.get('model')!='Qwen/Qwen3.5-9B' or artifact_identity.get('revision')!='c202236235762e1c871ad0ccb60c8ee5ba337b9a':raise ValueError('Wrong model artifact identity')
for relative,digest in artifact_identity['files'].items():
 if hashlib.file_digest((Path(a.model)/relative).open('rb'),'sha256').hexdigest()!=digest:raise ValueError('Model file SHA mismatch: '+relative)
a.log_dir.mkdir(parents=True,exist_ok=True)
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(a.device).eval()
lock=threading.Lock();last=[time.time()];busy=[False]
identity={'model':'Qwen/Qwen3.5-9B','revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','pid':os.getpid(),'ready':True,'supports_images':True,'max_input_tokens':32768,'max_output_tokens':4096,'supports_constrained_decoding':False}
(a.log_dir/'ready.local.json').write_text(json.dumps(identity,indent=2))
def convert(messages):
 out=[]
 for m in messages:
  content=m['content']
  if isinstance(content,str):content=[{'type':'text','text':content}]
  dest=[]
  for c in content:
   if c['type']=='text':dest.append({'type':'text','text':c['text']})
   elif c['type']=='image_url':
    url=c['image_url']['url'];assert url.startswith('data:image/') and ';base64,' in url,'Only inline image data supported'
    raw=base64.b64decode(url.split(';base64,',1)[1],validate=True)
    im=Image.open(io.BytesIO(raw));im.load();assert im.width*im.height<=4000000
    dest.append({'type':'image','image':im.convert('RGB')})
   else:raise ValueError('Unsupported content type '+str(c['type']))
  out.append({'role':m['role'],'content':dest})
 return out
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,code,obj):
  data=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_GET(self):
  if self.path in ('/health','/v1/models'):self.reply(200,identity)
  else:self.reply(404,{'error':'unknown path'})
 def do_POST(self):
  if self.path!='/v1/chat/completions':self.reply(404,{'error':'unknown path'});return
  n=int(self.headers.get('Content-Length',0))
  if not 0<n<=30000000:self.reply(400,{'error':'body size'});return
  raw=self.rfile.read(n);rid=str(uuid.uuid4());d=a.log_dir/rid;d.mkdir();(d/'request.json').write_bytes(raw)
  start=time.time()
  try:
   req=json.loads(raw);messages=convert(req['messages']);limit=int(req.get('max_tokens',2048));assert 1<=limit<=4096
   with lock:
    busy[0]=True;last[0]=time.time()
    try:
     inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
     count=inp['input_ids'].shape[-1]
     if count>32768:raise InputBudgetExhausted(f'Input token limit: {count} > 32768; no silent truncation')
     temperature=float(req.get('temperature',0.0));torch.manual_seed(int(req.get('seed',0)))
     opts={'max_new_tokens':limit,'do_sample':temperature>0}
     if temperature>0:opts.update(temperature=temperature,top_p=float(req.get('top_p',0.9)))
     with torch.inference_mode():tokens=model.generate(**inp,**opts)
     new=tokens[0,count:];content=processor.decode(new,skip_special_tokens=True)
     response={'id':rid,'model':identity['model'],'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'length' if len(new)>=limit else 'stop'}],'usage':{'prompt_tokens':count,'completion_tokens':len(new),'total_tokens':count+len(new)},'metadata':{'request_sha256':hashlib.sha256(raw).hexdigest(),'seconds':time.time()-start,'revision':identity['revision'],'images':sum(c['type']=='image' for m in messages for c in m['content'])}}
    finally:busy[0]=False;last[0]=time.time()
   (d/'response.json').write_text(json.dumps(response,ensure_ascii=False));self.reply(200,response)
  except Exception as exc:
   err={'error':type(exc).__name__+': '+str(exc),'id':rid,'code':'input_budget_exhausted' if isinstance(exc,InputBudgetExhausted) else 'inference_error'};(d/'error.json').write_text(json.dumps(err));self.reply(400 if isinstance(exc,InputBudgetExhausted) else 500,err)
server=ThreadingHTTPServer((a.host,a.port),Handler)
def idle():
 while True:
  time.sleep(30)
  if not busy[0] and time.time()-last[0]>a.idle_seconds:server.shutdown();return
threading.Thread(target=idle,daemon=True).start();print('READY',json.dumps(identity),flush=True);server.serve_forever()
