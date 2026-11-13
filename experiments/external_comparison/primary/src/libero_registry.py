"""Complete scene-dependent skill registry, independent of benchmark goal answers."""
from __future__ import annotations
import copy,itertools,re
from pathlib import Path

def call(name,args):return name+'('+', '.join(args)+')'

def declared_containment_regions(registry, parent):
 """All declared interior regions of a parent, independent of task goals.

 A parent-level containment condition denotes the union of these regions.
 Exterior/support and initialization regions are not container interiors.
 This also works with the already-frozen registries (no input regeneration).
 """
 exterior = ('init_region', 'top_side', 'top_surface', 'surface', 'cook_region')
 objects = set(registry.get('objects', []))
 return sorted(region for region, owner in registry.get('region_parents', {}).items()
               if owner == parent and region in objects and region.endswith('_region')
               and not any(word in region.lower() for word in exterior))

def make_registry(request, bddl_root=None):
 s=copy.deepcopy(request['world_state']);rows=s.get('objects',[])+s.get('fixtures',[]);types={o['name']:o.get('type','') for o in rows};objects=set(types)
 for f in s.get('init_predicates',[]):objects.update(f['args'])
 # Region names come from the scene declaration (not the target goal).
 regions={x for x in objects if x.endswith('_region')};bounds={};declared_entities=set(objects)
 if bddl_root and request.get('bddl_path'):
  text=(Path(bddl_root)/request['bddl_path']).read_text()
  tokens=re.findall(r'\(|\)|[^\s()]+',text);stack=[];tree=None
  for token in tokens:
   if token=='(':
    new=[]
    if stack:stack[-1].append(new)
    stack.append(new)
   elif token==')':
    closed=stack.pop()
    if not stack:tree=closed
   else:stack[-1].append(token)
  declarations=next((x[1:] for x in tree if isinstance(x,list) and x and x[0]==':regions'),[])
  for declaration in declarations:
   name=declaration[0];target=next((x[1] for x in declaration[1:] if isinstance(x,list) and x and x[0]==':target'),None)
   if target and target in declared_entities:
    region=target+'_'+name;regions.add(region);bounds[region]=target
 objects|=regions
 movable=[x['name'] for x in s.get('objects',[]) if not any(k in x.get('type','') for k in ['cabinet','microwave','stove','table'])]
 # Some LIBERO versions omit fixtures classified as movable bowls from objects;
 # all nonfixture entity sources in initial facts are legitimate movable objects.
 fixtures={x['name'] for x in s.get('fixtures',[])}
 for row in s.get('init_predicates',[]):
  if row['args'] and row['args'][0] not in fixtures and not row['args'][0].endswith('_region') and row['args'][0] not in movable:movable.append(row['args'][0])
 closable=[x for x in objects if not x.endswith('_region') and any(k in x for k in ['cabinet','microwave'])]
 access_units=sorted(set(closable)|{x for x in regions if any(c in x for c in closable) and not any(k in x for k in ['init_region','top_side','top_surface'])})
 switchable=[x for x in objects if not x.endswith('_region') and any(k in x for k in ['stove'])]
 contain=[x for x in objects if any(k in x for k in ['contain_region','basket','bowl','cabinet','microwave']) and not any(k in x for k in ['init_region','top_region','bottom_region'])]
 # Include all declared cabinet/microwave regions, not only goal regions.
 contain+= [x for x in regions if any(c in x for c in closable) and x not in contain]
 abstract_fixtures={x['name'] for x in s.get('fixtures',[]) if x.get('type') in ['table','living_room_table','kitchen_table','study_table','coffee_table'] or x['name'].endswith('_table')}
 predicate_locations=objects-abstract_fixtures
 supports=[x for x in objects if x not in abstract_fixtures and (x.endswith('_region') or any(k in x for k in ['plate','bowl','stove','table','cabinet'])) and not any(k in x for k in ['contain_region','init_region'])]
 predicates={n:{'arity':arity,'arguments':['arg'+str(i) for i in range(arity)],'static':False,'kind':'relation'} for n,arity in [('On',2),('In',2),('Open',1),('Close',1),('Turnon',1),('Turnoff',1)]}
 schemas={
  'place_on':{'arguments':['object','target'],'precondition':[],'effect':['On ?object ?target'],'template':'place_on(?object, ?target)'},
  'place_in':{'arguments':['object','target'],'precondition':[],'conditional_precondition':'For interior cabinet drawer target require Open(target); for interior microwave target require Open(parent(target)); region parents given in region_parents. Exterior top surfaces require no opening.','effect':['In ?object ?target'],'template':'place_in(?object, ?target)'},
  'open':{'arguments':['object'],'precondition':[],'effect':['Open ?object','(not (Close ?object))'],'template':'open(?object)'},
  'close':{'arguments':['object'],'precondition':[],'effect':['Close ?object','(not (Open ?object))'],'template':'close(?object)'},
  'turnon':{'arguments':['object'],'precondition':[],'effect':['Turnon ?object','(not (Turnoff ?object))'],'template':'turnon(?object)'},
  'turnoff':{'arguments':['object'],'precondition':[],'effect':['Turnoff ?object','(not (Turnon ?object))'],'template':'turnoff(?object)'}}
 facts={call(x['predicate'],x['args']) for x in s.get('init_predicates',[]) if x['predicate'] in predicates}
 for obj in access_units:
  if call('Open',[obj]) not in facts:facts.add(call('Close',[obj]))
 for obj in switchable:
  if call('Turnon',[obj]) not in facts:facts.add(call('Turnoff',[obj]))
 grounded=[]
 for op,targets in [('place_on',supports),('place_in',contain)]:
  for obj,target in itertools.product(sorted(movable),sorted(set(targets))):
   if obj==target or target.startswith(obj+'_'):continue
   parent=next((c for c in closable if target==c or target.startswith(c+'_')),None)
   interior=parent and not any(k in target for k in ['top_side','surface'])
   access=target if interior and 'cabinet' in target and target in access_units else parent
   pre=[call('Open',[access])] if interior else []
   grounded.append({'name':call(op,[obj,target]),'pre':pre,'add':[call('On' if op=='place_on' else 'In',[obj,target])],'delete':[call(pred,[obj,old]) for pred in ['On','In'] for old in sorted(predicate_locations) if (pred,old)!=(('On' if op=='place_on' else 'In'),target)],'cost':1})
 for op,objs,pred,neg in [('open',access_units,'Open','Close'),('close',access_units,'Close','Open'),('turnon',switchable,'Turnon','Turnoff'),('turnoff',switchable,'Turnoff','Turnon')]:
  for obj in sorted(objs):grounded.append({'name':call(op,[obj]),'pre':[],'add':[call(pred,[obj])],'delete':[call(neg,[obj])],'cost':1})
 return {'domain':'libero','actions':schemas,'predicates':predicates,'objects':sorted(objects),'world_state':s,'facts':sorted(facts),'ground_actions':grounded,'grounding_policy':'All scene-compatible registered skills; scene declarations only; no target-conditioned option pruning','region_parents':{**{x:next((c for c in closable if x==c or x.startswith(c+'_')),bounds.get(x)) for x in sorted(regions)},**bounds},'role_objects':{'movable':sorted(set(movable)),'placement_supports':sorted(set(supports)),'containment_targets':sorted(set(contain)),'closable':sorted(access_units),'switchable':sorted(switchable)},'domain_notes':['place_on/place_in remove previous On/In location relations for the moved object; only the new location remains.','Interior cabinet placement requires Open of the addressed drawer region; interior microwave placement requires Open of the microwave object. Exterior top surfaces require no opening.','Registered objects and region names come from complete scene declarations, not the instruction target. Abstract table fixtures are not direct On/In predicate targets; use their registered regions.']}

def action_prompt(name,args,registry):
 clean=lambda value: re.sub(r'\s+',' ',re.sub(r'(?<=[A-Za-z])_[0-9]+(?=_|$)','',value).replace('_',' ')).strip()
 obj=clean(args[0])
 if name in ['place_on','place_in']:
  target=args[1];target=clean(re.sub(r'_(contain|heating|cooking|cook)_region$','',target))
  return 'put the '+obj+(' on the ' if name=='place_on' else ' in the ')+target
 if name in ['open','close'] and 'cabinet' in args[0] and args[0].endswith('_region'):
  match=re.match(r'(.*)_(top|middle|bottom)_region$',args[0])
  if match:return name+' the '+match[2]+' drawer of the '+clean(match[1])
 return {'open':'open the ','close':'close the ','turnon':'turn on the ','turnoff':'turn off the '}[name]+obj

def effect_predicate(name,args):
 return {'place_on':'On','place_in':'In','open':'Open','close':'Close','turnon':'Turnon','turnoff':'Turnoff'}[name],list(args)
