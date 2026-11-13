"""Loopback model transport with complete per-call provenance."""
import json,time,urllib.request
from pathlib import Path
class TransportProvenanceError(OSError):
 pass

class LLMClient:
 def __init__(self,url,*,adapter_method=None,adapter_checkpoint=None,log_path=None,hibt_url=None,model_tag="qwen",ner_checkpoint=None,ner_url=None,max_calls=None):
  self.url=url.rstrip('/');self.hibt_url=hibt_url;self.model_tag=model_tag;self.ner_checkpoint=ner_checkpoint;self.ner_url=ner_url;self.max_calls=max_calls;self.attempted_calls=0;self.ner_calls=[];self.adapter_method=adapter_method;self.adapter_checkpoint=adapter_checkpoint;self.log_path=log_path;self.metadata={'adapter_method':adapter_method,'adapter_checkpoint':adapter_checkpoint};self.calls=[]
 def _post(self,path,data):
  req=urllib.request.Request(self.url+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(req,timeout=600) as response:return json.load(response)
 def complete(self,messages,max_new_tokens=2048,temperature=0.0,seed=0):
  if self.max_calls is not None and self.attempted_calls>=self.max_calls:raise RuntimeError('Global per-case LLM call budget exhausted')
  self.attempted_calls+=1
  payload={'messages':messages,'max_new_tokens':max_new_tokens,'temperature':temperature,'seed':seed,'adapter_checkpoint':self.adapter_checkpoint};result=self._post('/complete',payload);row={'request':payload,'response':result};self.calls.append(row)
  if self.log_path:
   with open(self.log_path,'a') as f:f.write(json.dumps(row)+'\n')
  return result
 def ner(self,text):
  original=self.url
  try:
   self.url=self.ner_url or original
   value=self._post('/ner',{'text':text,'checkpoint':self.ner_checkpoint})
   if self.ner_checkpoint and Path(value.get('model','')).as_posix()!=Path(self.ner_checkpoint).as_posix():raise TransportProvenanceError('NER service checkpoint provenance mismatch')
   self.ner_calls.append({'text':text,'response':value});return value['entities']
  finally:self.url=original
 def hibt(self,target,world_state,**kw):
  original=self.url
  try:
   self.url=self.hibt_url or original
   return self._post('/hibt',{'target':target,'world_state':world_state,'model_tag':self.model_tag,**kw})
  finally:self.url=original
