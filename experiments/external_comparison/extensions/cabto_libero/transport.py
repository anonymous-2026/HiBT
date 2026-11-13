"""Audited OpenAI-compatible VLM transport; no service/model process startup."""
from __future__ import annotations
import copy
import hashlib
import json
import time
import urllib.request
import urllib.error


class InfrastructureError(OSError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class VLMClient:
    def __init__(self, url, model, maximum, log_path, timeout_s=600, expected_revision=None):
        self.url, self.model, self.maximum = url.rstrip('/'), model, int(maximum)
        self.log_path, self.timeout_s = log_path, float(timeout_s)
        self.used = 0
        self.events = []
        self.infrastructure_errors = []
        self.expected_revision = expected_revision

    def complete(self, messages, max_new_tokens=4096, temperature=0.0, seed=0, stage=None, images=None):
        if self.used >= self.maximum:
            raise BudgetExhausted('CABTO global per-case generation-call allowance exhausted')
        self.used += 1
        prepared = copy.deepcopy(messages)
        if images:
            last = next(i for i in reversed(range(len(prepared))) if prepared[i]['role'] == 'user')
            prepared[last]['content'] = [{'type': 'text', 'text': prepared[last]['content']}] + [
                {'type': 'image_url', 'image_url': {'url': value}} for value in images]
        payload = {'model': self.model, 'messages': prepared, 'max_tokens': int(max_new_tokens),
                   'temperature': float(temperature), 'seed': int(seed), 'stream': False}
        endpoint = self.url + ('/chat/completions' if self.url.endswith('/v1') else '/v1/chat/completions')
        event = {'call_index': self.used - 1, 'stage': stage, 'request': payload,
                 'request_sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
        begin = time.monotonic()
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with opener.open(req, timeout=self.timeout_s) as response:
                value = json.load(response)
            event['response'] = value
            if value.get('model') != self.model:
                raise InfrastructureError('Actual VLM backbone differs from the frozen model name')
            metadata = value.get('metadata', {})
            if self.expected_revision and metadata.get('revision') != self.expected_revision:
                raise InfrastructureError('Actual VLM revision differs from the frozen configuration')
            if metadata.get('request_sha256') != event['request_sha256']:
                raise InfrastructureError('VLM service did not bind the exact outgoing request body')
            if metadata.get('images') != len(images or []):
                raise InfrastructureError('Actual VLM image count differs from the submitted observation context')
            text = value['choices'][0]['message']['content']
            if not isinstance(text, str):
                raise InfrastructureError('Malformed OpenAI-compatible VLM response')
            usage = value.get('usage', {})
            return {'text': text, 'model': value['model'], 'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0), 'wall_s': time.monotonic() - begin}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            try:
                error_response = json.loads(body)
            except ValueError:
                error_response = {'body': body}
            event['response'] = error_response
            code = error_response.get('code', error_response.get('error', {}).get('code') if isinstance(error_response.get('error'), dict) else None)
            if exc.code == 400 and code == 'input_budget_exhausted':
                event['declared_budget_failure'] = True
                event['error'] = 'Declared VLM input allowance exhausted; prompt was not truncated'
                raise BudgetExhausted(event['error']) from exc
            event['error'] = repr(exc)
            self.infrastructure_errors.append(event['error'])
            raise InfrastructureError(str(exc)) from exc
        except (OSError, TimeoutError, ConnectionError) as exc:
            event['error'] = repr(exc)
            self.infrastructure_errors.append(event['error'])
            raise InfrastructureError(str(exc)) from exc
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            event['error'] = repr(exc)
            self.infrastructure_errors.append(event['error'])
            raise InfrastructureError('VLM transport/response error: ' + str(exc)) from exc
        finally:
            event['wall_s'] = time.monotonic() - begin
            self.events.append(event)
            with open(self.log_path, 'a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def metrics(self):
        return {'generation_calls': self.used,
                'prompt_tokens': sum(x.get('response', {}).get('usage', {}).get('prompt_tokens', 0) for x in self.events),
                'completion_tokens': sum(x.get('response', {}).get('usage', {}).get('completion_tokens', 0) for x in self.events),
                'infrastructure_errors': self.infrastructure_errors}
