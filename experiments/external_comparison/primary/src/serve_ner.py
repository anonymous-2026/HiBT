"""Genuine frozen token-classification Parser inference (CPU)."""
import argparse,json,time,threading,os
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--host',default='localhost');p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--log-dir',type=Path,required=True);a=p.parse_args();
import torch
from transformers import pipeline
a.log_dir.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);ner=pipeline('ner',model=a.model,tokenizer=a.model,device=-1);lock=threading.Lock();last=time.monotonic()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send(self,x,code=200):
  b=json.dumps(x).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):self.send({'status':'READY','model':Path(a.model).name})
 def do_POST(self):
  global last
  try:
   x=json.loads(self.rfile.read(int(self.headers['Content-Length'])));beg=time.monotonic()
   if x.get('checkpoint') and Path(x['checkpoint']).resolve()!=Path(a.model).resolve():raise ValueError('Requested NER checkpoint differs from loaded model')
   with lock:entities=ner(x['text'])
   result={'entities':[{k:(v.item() if hasattr(v,'item') else v) for k,v in row.items()} for row in entities],'wall_s':time.monotonic()-beg,'model':a.model};last=time.monotonic()
   with (a.log_dir/'calls.jsonl').open('a') as f:f.write(json.dumps({'text':x['text'],'response':result})+'\n')
   self.send(result)
  except Exception as e:self.send({'error':repr(e)},500)
server=ThreadingHTTPServer((a.host,a.port),Handler);server.timeout=10
while time.monotonic()-last<1800:server.handle_request()
