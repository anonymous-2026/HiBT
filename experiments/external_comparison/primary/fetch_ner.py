"""Fetch the pinned original LLM-BT Parser checkpoint (no API credentials)."""
import argparse,urllib.request,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base='https://raw.githubusercontent.com/henryhaotian/LLM-BT/c69c18d0cf4b78f166ed352fc0fa8470823b32f6/LLMBT/Parser/keywords_extraction/'
for name in ['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','pytorch_model.bin']:
 url=base+name;dest=a.output/name
 with urllib.request.urlopen(url,timeout=180) as r,dest.open('wb') as out:
  while chunk:=r.read(1024*1024):out.write(chunk)
 if name.endswith('.bin') and dest.stat().st_size<1024 and dest.read_bytes().startswith(b'version https://git-lfs'):
  with urllib.request.urlopen(url.replace('raw.githubusercontent.com/','media.githubusercontent.com/media/'),timeout=180) as r,dest.open('wb') as out:
   while chunk:=r.read(1024*1024):out.write(chunk)
assert (a.output/'pytorch_model.bin').stat().st_size>100000000
(a.output/'download_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file() and p.name!='download_manifest.json'},indent=2)+'\n')
