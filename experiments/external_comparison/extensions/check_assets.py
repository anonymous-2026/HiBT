"""Verify the100 prescribed LIBERO starts/BDDL files without rendering or policy calls."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parent

def verify(bddl_root=None):
 import numpy as np
 from libero.libero import benchmark,get_libero_path
 expected=json.loads((ROOT/'inputs/expected_libero_identities.json').read_text())
 suites={}
 for r in expected:
  if r['suite'] not in suites:suites[r['suite']]=benchmark.get_benchmark_dict()[r['suite']]()
  suite=suites[r['suite']];arr=np.asarray(suite.get_task_init_states(r['task_id'])[r['init']])
  h=hashlib.sha256();h.update(arr.dtype.str.encode());h.update(str(arr.shape).encode());h.update(arr.tobytes())
  # Use the exact original runtime hash function, including its serialization.
  import sys
  sys.path.insert(0,str(ROOT/'common'));from run_native_libero import initial_state_sha256
  if initial_state_sha256(arr)!=r['initial_state_sha256'] or arr.dtype.str!=r['initial_state_dtype'] or list(arr.shape)!=r['initial_state_shape']:raise ValueError('Initial-state fingerprint mismatch: '+str((r['task_id'],r['init'])))
  task=suite.get_task(r['task_id']);path=Path(bddl_root or get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
  if hashlib.sha256(path.read_bytes()).hexdigest()!=r['actual_bddl_sha256']:raise ValueError('BDDL fingerprint mismatch: '+str(r['task_id']))
 return {'status':'PASS','paired_starts':len(expected),'policy_calls':0,'physical_steps':0}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bddl-root',type=Path);a=p.parse_args();print(json.dumps(verify(a.bddl_root),indent=2))
if __name__=='__main__':main()
