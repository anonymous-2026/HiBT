"""Download the exact public Qwen3.5-9B revision; no inference is performed."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
from huggingface_hub import snapshot_download
repo='Qwen/Qwen3.5-9B';revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'
snapshot_download(repo_id=repo,revision=revision,local_dir=a.output)
files={str(f.relative_to(a.output)):hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in a.output.rglob('*') if f.is_file() and '.cache' not in f.parts and f.name!='download_identity.json'}
(a.output/'download_identity.json').write_text(json.dumps({'model':repo,'revision':revision,'files':files},indent=2)+'\n')
print('Downloaded and hashed',repo,revision)
