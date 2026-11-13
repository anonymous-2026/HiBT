"""Validate preserved cached scene registries separately from runtime source."""
import hashlib,json
from pathlib import Path
from goal_contract import VERSION

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def verify_cached_registry_lineage(requests,lineage_path=None):
    path=Path(lineage_path or Path(__file__).with_name('cached_registry_lineage.json'))
    lineage=json.loads(path.read_text())
    if lineage.get('goal_input_contract')!=VERSION or lineage.get('registry_rebuilt_for_goal_track') is not False:
        raise ValueError('Unexpected cached-registry task-interface lineage')
    actual={row['request_id']:digest(row['registry']) for row in requests}
    if len(actual)!=len(requests) or actual!=lineage['per_task_cached_registry_sha256']:
        raise ValueError('Cached registry values/membership differ from the preserved scene interface')
    runtime_sha=hashlib.sha256(Path(__file__).with_name('libero_registry.py').read_bytes()).hexdigest()
    if runtime_sha!=lineage['runtime_registry_implementation_sha256']:
        raise ValueError('Active runtime registry implementation changed since lineage freeze')
    return {'cached_registry_source_sha256':lineage['cached_registry_source_sha256'],
            'runtime_registry_implementation_sha256':runtime_sha,
            'lineage_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'cached_registry_values_preserved':True,'n_requests':len(requests)}
