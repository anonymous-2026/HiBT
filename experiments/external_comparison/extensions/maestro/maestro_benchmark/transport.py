"""OpenAI-compatible multimodal chat; explicit backbone adaptation, no retries."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import urllib.error
import urllib.request

from .runner import schema_json
from .tools import InfrastructureError, PlannerBudgetExhausted


class OpenAIChat:
    def __init__(self, endpoint, model, *, seed=0, temperature=0.1,
                 max_tokens=4096, timeout=600, log=None, structured=False):
        self.endpoint = endpoint.rstrip('/')
        if not self.endpoint.endswith('/chat/completions'):
            self.endpoint += '/chat/completions' if self.endpoint.endswith('/v1') else '/v1/chat/completions'
        self.model, self.seed, self.temperature = model, seed, temperature
        self.max_tokens, self.timeout, self.log, self.structured = max_tokens, timeout, log, structured
        self.history, self.calls = [], []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def send(self, parts, schema):
        content = []
        for part in parts:
            if isinstance(part, str): content.append({'type': 'text', 'text': part})
            elif hasattr(part, 'save'):
                buffer = io.BytesIO();part.convert('RGB').save(buffer, format='PNG')
                content.append({'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), 'detail': 'high'}})
            else: raise TypeError('Planner accepts only text and actual PIL images')
        content.append({'type': 'text', 'text': 'Return exactly one JSON object, without Markdown, conforming to this schema: ' + json.dumps(schema_json(schema))})
        payload = {'model': self.model, 'messages': self.history + [{'role': 'user', 'content': content}],
                   'temperature': self.temperature, 'max_tokens': self.max_tokens, 'seed': self.seed}
        if self.structured: payload['response_format'] = {'type': 'json_schema', 'json_schema': {'name': schema, 'strict': True, 'schema': schema_json(schema)}}
        data = json.dumps(payload).encode();record = {'schema': schema, 'request': payload,
            'request_sha256': hashlib.sha256(data).hexdigest(), 'endpoint': self.endpoint}
        try:
            request = urllib.request.Request(self.endpoint, data=data, headers={'Content-Type': 'application/json'})
            with self.opener.open(request, timeout=self.timeout) as response: raw = response.read()
            value = json.loads(raw)
            record.update(response=value, response_sha256=hashlib.sha256(raw).hexdigest())
            if value.get('model') != self.model:
                raise InfrastructureError('VLM model identity mismatch: ' + repr(value.get('model')))
            text = value['choices'][0]['message']['content']
            if not isinstance(text, str): raise InfrastructureError('VLM has no textual structured output')
        except urllib.error.HTTPError as exc:
            error_bytes = exc.read()
            try: error_body = json.loads(error_bytes)
            except Exception: error_body = {'error': error_bytes.decode(errors='replace')}
            record.update(http_status=exc.code, error_response=error_body,
                          response_sha256=hashlib.sha256(error_bytes).hexdigest())
            if exc.code == 400 and error_body.get('code') == 'input_budget_exhausted':
                record['semantic_resource_budget_exhausted'] = True
                raise PlannerBudgetExhausted(str(error_body.get('error'))) from exc
            record['error'] = 'HTTP ' + str(exc.code) + ': ' + str(error_body)
            raise InfrastructureError('VLM transport unavailable: ' + record['error']) from exc
        except (InfrastructureError, PlannerBudgetExhausted): raise
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
            raise InfrastructureError('VLM transport/response unavailable: ' + record['error']) from exc
        finally:
            self.calls.append(record)
            if self.log: self.log(record)
        # Match the released Gemini/Astra chat: retain past text, never old images.
        self.history.extend([{'role': 'user', 'content': [x for x in content if x['type'] == 'text']},
                             {'role': 'assistant', 'content': text}])
        return text, text
