"""Fetch the five pinned CABTO core files and upstream licensing README, verifying every SHA256."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',default='https://raw.githubusercontent.com/DIDS-EI/CABTO',help='Optional mirror retaining commit/path layout');a=p.parse_args()
 for kind in ['cabto_assembly','cabto_libero']:
  manifest=json.loads((ROOT/kind/'source_audit/source_manifest.json').read_text())
  for relative,digest in manifest['files'].items():
   path=ROOT/kind/'source_audit/pinned'/relative
   if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:continue
   url=a.base_url.rstrip('/')+'/'+manifest['commit']+'/'+relative
   with urllib.request.urlopen(url,timeout=120) as response:data=response.read()
   if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Official source SHA mismatch: '+relative)
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 print('PASS: pinned CABTO source available; no model calls')
if __name__=='__main__':main()
