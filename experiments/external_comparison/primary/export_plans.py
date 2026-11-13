"""Export all 50 completed initial plan attempts, retaining semantic failures."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
rows=[json.loads(x.read_text()) for x in sorted((a.run_dir/'results').glob('*.json'))]
expected={(i,m) for i in range(10) for m in ['hibt','llmbt','btgenbot','kios','obtea']}
assert len(rows)==50 and {(x['task_id'],x['method']) for x in rows}==expected,'Need exactly all50 prescribed plans'
assert all(x['status']=='COMPLETED' for x in rows),'Resolve explicit infrastructure errors before exporting'
assert len({x['protocol_id'] for x in rows})==1,'Mixed planning protocol'
(a.run_dir/'planning_outputs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
print('Exported all 50 attempts; null/failed semantic plans are preserved.')
