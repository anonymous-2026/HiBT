"""CPU-only software tests; canned chats are not experiment/model results."""
from __future__ import annotations

import ast
import copy
import importlib
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
E6 = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E6 / 'common'))
sys.dont_write_bytecode = True

from assembly_registry import make_registry
from native_runtime import _world_class
from maestro_benchmark.runner import run_episode, RunConfig, SCHEMAS, public_request
from maestro_benchmark.sandbox import Interpreter, SandboxViolation
from maestro_benchmark.tools import AssemblyBackend, LiberoBackend, InfrastructureError, PlannerBudgetExhausted
from maestro_benchmark.transport import OpenAIChat


class CannedChat:
    def __init__(self, responses): self.responses, self.messages = list(responses), []
    def send(self, parts, schema):
        self.messages.append((parts, schema))
        value = self.responses.pop(0)
        if isinstance(value, Exception): raise value
        return value, json.dumps(value)


def plan(code): return {'describe_scene': 'Synthetic CPU test scene', 'reason_and_plan': 'Test program', 'actor_code': code}
def replan(code='', complete=False, success=True):
    return {'reason_and_plan':'Test feedback','success':success,'task_complete':complete,'actor_code':code}


def assembly():
    state = {'objects': [{'name':'cpu_hand','properties':['is_empty']},
                         {'name':'cpu_tool','properties':['is_empty','is_equippable']},
                         {'name':'cpu_pin','properties':[]},{'name':'cpu_socket','properties':[]}],
             'constraints':[{'source':'cpu_tool','name':'can_manipulate','target':'cpu_pin'},
                            {'source':'cpu_pin','name':'can_insert_to','target':'cpu_socket'}], 'relations':[]}
    request = {'sample_id':'maestro_cpu_only','instruction':'Insert cpu_pin into cpu_socket.',
               'planning_goal':'is_inserted_to(cpu_pin, cpu_socket)', 'world_state':state}
    request['registry'] = make_registry(request)
    world = _world_class()(state)
    return request, AssemblyBackend(world, request['registry']), world


class SandboxTests(unittest.TestCase):
    def test_control_functions_comprehensions_and_persistent_variables(self):
        calls=[];x=Interpreter({'robot':{'execute_skill':lambda name,args:calls.append((name,args)) or {'status':'success'}}})
        code='''
import math
def run(name, item):
    result = robot.execute_skill(name, [item])
    return result['status'] == 'success'
items = [str(i) for i in range(3) if i != 1]
done = []
for item in items:
    if run('skill', item):
        done.append(item)
print(done, math.sqrt(9))
'''
        self.assertIn("['0', '2'] 3.0",x.execute(code))
        self.assertEqual(x.execute('print(len(done))'),'2\n');self.assertEqual(len(calls),2)

    def test_network_files_reflection_and_code_execution_unavailable(self):
        sources=['import os\nos.system("id")','open("/etc/passwd").read()',
                 '__import__("socket")','robot.execute_skill.__globals__',
                 '(1).__class__.__mro__','eval("1+1")','exec("1+1")',
                 'import urllib.request','from pathlib import Path','getattr(robot,"execute_skill")']
        for source in sources:
            with self.subTest(source=source),self.assertRaises((SandboxViolation,NameError)):
                Interpreter({'robot':{'execute_skill':lambda *x:None}}).execute(source)

    def test_full_validation_before_forbidden_import_side_effect(self):
        calls=[];x=Interpreter({'robot':{'execute_skill':lambda *a:calls.append(a)}})
        with self.assertRaises(SandboxViolation):x.execute('robot.execute_skill("x", [])\nimport os')
        self.assertEqual(calls,[])

    def test_infinite_loop_and_memory_bounded(self):
        for code in ['while True:\n    pass','x="x"*1000000000','x=10**1000000','x=list(range(1000000000))']:
            with self.subTest(code=code),self.assertRaises(SandboxViolation):Interpreter({},max_nodes=100).execute(code)


class HarnessTests(unittest.TestCase):
    def test_exact_upstream_schemas(self):
        tree=ast.parse((ROOT/'upstream/maestro/runner.py').read_text())
        for name,fields in SCHEMAS.items():
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            upstream={n.target.id:n.annotation.id for n in node.body if isinstance(n,ast.AnnAssign)}
            self.assertEqual(upstream,{k:v.__name__ for k,v in fields.items()})

    def test_real_worldstate_failure_then_feedback_replan_and_execution(self):
        request,backend,world=assembly()
        code='''
print(robot.execute_skill('load_tool', ['cpu_hand', 'cpu_tool']))
print(robot.execute_skill('pick_up', ['cpu_hand', 'cpu_tool', 'cpu_pin']))
print(robot.execute_skill('insert', ['cpu_hand', 'cpu_tool', 'cpu_pin', 'cpu_socket']))
'''
        chat=CannedChat([plan("saved = 'persistent'\nprint(robot.execute_skill('insert', ['cpu_hand','cpu_tool','cpu_pin','cpu_socket']))"),
                         replan("print(saved)\n"+code,success=False),replan(complete=True)])
        result=run_episode(request,backend,chat,lambda:world.check('is_inserted_to',['cpu_pin','cpu_socket']))
        self.assertTrue(result['success']);self.assertEqual(result['attempts'],2);self.assertEqual(result['planner_calls'],3)
        self.assertEqual(result['tool_trace'][0]['status'],'failure')
        self.assertIn('persistent',result['events'][-2]['stdout'] if result['events'][-2]['event']=='execution_feedback' else str(result['events']))

    def test_planner_claim_is_separate_from_final_success_and_next_code_not_run(self):
        request,backend,world=assembly()
        chat=CannedChat([plan('print("Nothing executed")'),replan("robot.execute_skill('change_tool',['cpu_hand','cpu_tool'])",complete=True)])
        r=run_episode(request,backend,chat,lambda:False)
        self.assertTrue(r['planner_reported_complete']);self.assertFalse(r['success']);self.assertEqual(r['skill_calls'],0)

    def test_code_error_is_returned_to_the_next_model_turn(self):
        request,backend,_=assembly();chat=CannedChat([plan('open("/etc/passwd")'),replan(complete=True)])
        r=run_episode(request,backend,chat,lambda:False)
        self.assertEqual(r['status'],'COMPLETED');self.assertIn('Unavailable name: open',str(chat.messages[1]))

    def test_private_evaluator_and_teacher_fields_not_in_model_or_tools(self):
        request,backend,_=assembly();request.update(target='PRIVATE_EVAL_SENTINEL',teacher_actions=['PRIVATE_TEACHER_SENTINEL'])
        chat=CannedChat([plan('print("ok")'),replan(complete=True)])
        run_episode(request,backend,chat,lambda:False)
        text=str(chat.messages)+str(backend.snapshot())
        self.assertNotIn('PRIVATE_EVAL_SENTINEL',text);self.assertNotIn('PRIVATE_TEACHER_SENTINEL',text)

    def test_transport_failure_is_infrastructure(self):
        request,backend,_=assembly();chat=CannedChat([InfrastructureError('Unavailable VLM')])
        result=run_episode(request,backend,chat,lambda:False)
        self.assertEqual(result['status'],'INFRASTRUCTURE_ERROR')

    def test_input_resource_limit_is_retained_semantic_outcome(self):
        request,backend,_=assembly();chat=CannedChat([PlannerBudgetExhausted('32768 input tokens')])
        result=run_episode(request,backend,chat,lambda:False)
        self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['termination_reason'],'planner_resource_budget_exhausted')

    def test_invalid_json_receives_format_only_feedback_and_is_budgeted(self):
        request,backend,_=assembly();chat=CannedChat(['INVALID_JSON',plan('print("ok")'),replan(complete=True)])
        result=run_episode(request,backend,chat,lambda:False)
        self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['planner_calls'],3)
        self.assertTrue(any(e['event']=='semantic_format_error' for e in result['events']))
        self.assertIn('same intended response',str(chat.messages[1]))


class PhysicalAdapterTests(unittest.TestCase):
    def fake_controller(self):
        class Controller:
            def __init__(self):
                self.registry={'actions':{'move':{'arguments':['object']}},'predicates':{'Done':{'arity':1}},'objects':['cube']}
                self.steps=0;self.policy_calls=0;self.retries=0;self.reason=None;self.cache={};self.semantic_errors=[]
                self.config={'max_steps':650,'camera_transform':'rotate180'};self.obs={};self.request={};self.catalog=[]
                self.env=object()
            def execute(self,name,args):
                self.steps+=5;self.policy_calls+=1
                return 'success' if self.steps==15 else 'running'
            def check(self,name,args):return self.steps==15
        return Controller()

    def test_running_skill_is_actually_advanced_until_completion(self):
        c=self.fake_controller();b=LiberoBackend(c,lambda *x:{'facts':[]})
        result=b.execute_skill('move',['cube'])
        self.assertEqual(result['status'],'success');self.assertEqual(result['physical_steps'],15);self.assertEqual(c.policy_calls,3)

    def test_actual_two_camera_images_rotated_and_delivered_to_model(self):
        import numpy as np
        c=self.fake_controller();a=np.zeros((4,4,3),dtype=np.uint8);a[0,0]=[255,0,0]
        c.obs={'agentview_image':a,'robot0_eye_in_hand_image':a.copy()}
        b=LiberoBackend(c,lambda *x:{'facts':[]});images=b.images()
        self.assertEqual(len(images),2);self.assertEqual(images[0].getpixel((3,3)),(255,0,0))
        request={'instruction':'Move the cube','planning_goal':'Done(cube)'}
        chat=CannedChat([plan("print(robot.execute_skill('move',['cube']))"),replan(complete=True)])
        r=run_episode(request,b,chat,lambda:c.steps==15)
        self.assertTrue(r['success']);self.assertEqual(sum(hasattr(x,'save') for x in chat.messages[0][0]),2)
        self.assertEqual(sum(hasattr(x,'save') for x in chat.messages[1][0]),2)


class TransportAndCliTests(unittest.TestCase):
    def test_real_cpu_http_cli_and_original_worldstate(self):
        """A synthetic HTTP stub exercises the actual CLI without any model."""
        requests=[];counter=[0]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(request)
                code="print(robot.execute_skill('load_tool',['cpu_hand','cpu_tool']))\nprint(robot.execute_skill('pick_up',['cpu_hand','cpu_tool','cpu_pin']))\nprint(robot.execute_skill('insert',['cpu_hand','cpu_tool','cpu_pin','cpu_socket']))"
                value=plan(code) if counter[0]==0 else replan(complete=True);counter[0]+=1
                response={'model':'cpu-software-stub','choices':[{'message':{'content':json.dumps(value)}}]}
                data=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);request,_,_=assembly()
                (root/'requests.json').write_text(json.dumps([request]))
                (root/'eval.json').write_text(json.dumps([{'sample_id':request['sample_id'],'target':request['planning_goal']}]))
                cmd=[sys.executable,'-B','-m','maestro_benchmark.cli','--mode','assembly','--phase','development',
                     '--requests-file',str(root/'requests.json'),'--evaluators-file',str(root/'eval.json'),
                     '--config',str(ROOT/'config.example.json'),'--output-dir',str(root/'run'),
                     '--endpoint',f'http://127.0.0.1:{server.server_port}','--model','cpu-software-stub',
                     '--native-source',str(E6/'common')]
                run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                result=json.loads(next((root/'run/cases').glob('*/result.json')).read_text())
                self.assertTrue(result['success']);self.assertEqual(result['planner_calls'],2)
                self.assertEqual(len(requests),2);self.assertNotIn('response_format',requests[0])
                self.assertEqual(requests[0]['max_tokens'],4096)
                # Resume must retain bytes and make zero additional requests.
                saved=next((root/'run/cases').glob('*/result.json')).read_bytes()
                again=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=30)
                self.assertEqual(again.returncode,0,again.stdout+again.stderr);self.assertEqual(len(requests),2)
                self.assertEqual(saved,next((root/'run/cases').glob('*/result.json')).read_bytes())
        finally:server.shutdown();server.server_close();thread.join()

    def test_two_actual_images_encoded_and_old_images_not_retained(self):
        from PIL import Image
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def read(self):return json.dumps({'model':'cpu-stub','choices':[{'message':{'content':json.dumps(plan('print(1)'))}}]}).encode()
        class Opener:
            def __init__(self):self.payloads=[]
            def open(self,req,timeout):self.payloads.append(json.loads(req.data));return Response()
        chat=OpenAIChat('http://127.0.0.1:1','cpu-stub');opener=Opener();chat.opener=opener
        image=Image.new('RGB',(4,4),'red')
        chat.send(['initial',image,image],'FirstStepPlan');chat.send(['feedback',image],'Replan')
        first,second=opener.payloads
        self.assertEqual(sum(p['type']=='image_url' for p in first['messages'][0]['content']),2)
        self.assertTrue(all(p['type']=='text' for p in second['messages'][0]['content']))
        self.assertEqual(sum(p['type']=='image_url' for p in second['messages'][-1]['content']),1)

    def test_http400_declared_context_budget_is_not_infrastructure(self):
        class Opener:
            def open(self,req,timeout):raise urllib.error.HTTPError(req.full_url,400,'Bad Request',{},io.BytesIO(json.dumps({'code':'input_budget_exhausted','error':'limit32768'}).encode()))
        chat=OpenAIChat('http://127.0.0.1:1','cpu-stub');chat.opener=Opener()
        with self.assertRaises(PlannerBudgetExhausted):chat.send(['test'],'FirstStepPlan')
        self.assertTrue(chat.calls[0]['semantic_resource_budget_exhausted'])


if __name__=='__main__':unittest.main()
