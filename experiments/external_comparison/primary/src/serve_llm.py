"""Actual frozen-backbone/LoRA inference and original LLM-BT NER service."""
import argparse,json,os,threading,time,traceback
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--host',default='localhost');p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--ner-model');p.add_argument('--idle-exit',type=int,default=1800);a=p.parse_args();
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
a.log_dir.mkdir(exist_ok=True,parents=True)
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True);model.eval();ner=None;current_adapter=None;lock=threading.Lock();last=time.monotonic()
meta={'model':a.model,'torch':torch.__version__,'pid':os.getpid(),'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'port':a.port,'status':'READY'};(a.log_dir/'server_meta.local.json').write_text(json.dumps(meta,indent=2)+'\n')
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,data,code=200):
  raw=json.dumps(data).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):self.send({'status':'READY','model':Path(a.model).name})
 def do_POST(self):
  global last,ner,model,current_adapter
  try:
   data=json.loads(self.rfile.read(int(self.headers['Content-Length'])));last=time.monotonic()
   with lock:
    if self.path=='/ner':
     if not a.ner_model or not Path(a.ner_model,'pytorch_model.bin').exists():raise RuntimeError('Original Parser checkpoint not downloaded yet')
     if ner is None:ner=pipeline('ner',model=a.ner_model,tokenizer=a.ner_model,device=-1)
     ent=ner(data['text']);self.send({'entities':[{k:(v.item() if hasattr(v,'item') else v) for k,v in x.items()} for x in ent]});return
    assert self.path=='/complete'
    adapter=data.get('adapter_checkpoint');start=time.monotonic();torch.manual_seed(int(data.get('seed',0)))
    if adapter:
     from peft import PeftModel
     if current_adapter!=adapter:
      if isinstance(model,PeftModel):model=model.unload()
      model=PeftModel.from_pretrained(model,adapter);model.eval();current_adapter=adapter
     context=None
    else:
     from peft import PeftModel
     context=model.disable_adapter() if isinstance(model,PeftModel) else None
    inputs=tok.apply_chat_template(data['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda:0')
    if inputs.shape[1]>12000:raise ValueError('Input exceeds fixed12000-token ceiling')
    kwargs={'max_new_tokens':min(int(data.get('max_new_tokens',2048)),4096),'do_sample':float(data.get('temperature',0))>0,'pad_token_id':tok.eos_token_id}
    if kwargs['do_sample']:kwargs['temperature']=float(data['temperature'])
    with torch.inference_mode():
     if context:
      with context:generated=model.generate(inputs,**kwargs)
     else:generated=model.generate(inputs,**kwargs)
    n=inputs.shape[1];reply={'text':tok.decode(generated[0,n:],skip_special_tokens=True),'prompt_tokens':n,'completion_tokens':generated.shape[1]-n,'wall_s':time.monotonic()-start,'model':Path(a.model).name,'adapter_checkpoint':adapter}
    with (a.log_dir/'calls.jsonl').open('a') as f:f.write(json.dumps({'messages':data['messages'],'reply':reply})+'\n')
    last=time.monotonic();self.send(reply)
  except Exception as exc:traceback.print_exc();self.send({'error':repr(exc)},500)
server=ThreadingHTTPServer((a.host,a.port),Handler);server.timeout=10
while time.monotonic()-last<a.idle_exit:server.handle_request()
(a.log_dir/'idle_exit.txt').write_text('Actual inference service idle timeout; GPU process exiting.\n')
