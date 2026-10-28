# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only input, response and failure regressions for the VLM evidence runner."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/bench_batched_decode_vlm.py'
spec = importlib.util.spec_from_file_location('vlm_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def results():
    return [{'index': index, 'response': {'choices': [{'message': {
        'role': 'assistant', 'content': f'answer {index}', 'reasoning_content': 'reasoning'}}]}}
            for index in range(3)]


class InputsAndResponses(unittest.TestCase):
    def test_input_manifest_roundtrip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'image.png'
            image.write_bytes(b'PNG fixture bytes')
            manifest = runner.request_manifest(image)
            self.assertEqual(manifest['workload'], 'original_freeform')
            self.assertNotIn('data:image/png;base64,', json.dumps(manifest['payloads']))
            requests = runner.expand_requests(manifest, image)
            self.assertEqual(requests[0]['messages'][0]['content'], 'Write a short description of a mountain.')
            self.assertEqual(requests[1]['messages'][0]['content'], 'Count from one to ten in words.')
            image_url = requests[2]['messages'][0]['content'][0]['image_url']['url']
            self.assertEqual(base64.b64decode(image_url.split(',', 1)[1]), image.read_bytes())
            for payload in requests:
                self.assertEqual(payload['temperature'], 0)
                self.assertEqual(payload['max_tokens'], 48)
            image.write_bytes(b'changed image')
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner.expand_requests(manifest, image)

    def test_constrained_mode_is_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'image.png';image.write_bytes(b'fixture')
            manifest = runner.request_manifest(image, constrained=True)
            self.assertEqual(manifest['workload'], 'constrained_diagnostic')
            self.assertTrue(manifest['constrained_prompts'])
            self.assertEqual(manifest['payloads'][0]['messages'][0]['content'], 'Reply with exactly this word: mountain')

    def test_equality_includes_reasoning(self):
        solo = results();concurrent = copy.deepcopy(solo)
        self.assertTrue(runner.compare_messages(solo, concurrent)['all_match'])
        concurrent[1]['response']['choices'][0]['message']['reasoning_content'] = 'changed'
        self.assertEqual(runner.compare_messages(solo, concurrent)['exact_message_matches'], [True, False, True])

    def test_partial_and_error_responses_rejected(self):
        with self.assertRaises(ValueError):
            runner.compare_messages(results()[:2], results())
        invalid = [{'error': {'message': 'HTTP500'}}, {'choices': []},
                   {'choices': [{'message': {'role': 'assistant', 'content': ''}}]}]
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):runner.response_message(response)

    def test_backend_contract(self):
        with self.assertRaises(ValueError):runner.validate_backend('metal', None)
        for backend in ('cpu', 'cuda', 'rocm'):
            self.assertIsNone(runner.validate_backend(backend, None)['loaded_metallib'])


class FailureEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'model').mkdir();(self.root / 'model/config.json').write_text('{}')
        (self.root / 'server').write_text('fixture server binary')
        (self.root / 'image.png').write_bytes(b'fixture image')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0));port = probe.getsockname()[1]
        self.output = self.root / 'output'
        self.argv = [str(SCRIPT), '--repo', str(self.root), '--binary', str(self.root/'server'),
                     '--model', str(self.root/'model'), '--image', str(self.root/'image.png'),
                     '--output', str(self.output), '--source-revision', 'fixture', '--backend', 'cpu',
                     '--port', str(port)]

    def test_existing_output_not_overwritten(self):
        self.output.mkdir();marker=self.output/'keep';marker.write_text('preserve')
        with patch.object(sys, 'argv', self.argv), patch.object(sys, 'stderr', io.StringIO()), patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaises(SystemExit):runner.main()
            launch.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserve')

    def test_startup_failure_records_error_and_cleanup(self):
        process = Mock();process.poll.return_value = 7;process.returncode = 7
        with patch.object(sys, 'argv', self.argv), patch.object(runner.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'before readiness'):runner.main()
        process.terminate.assert_called_once();process.wait.assert_called_once_with(timeout=15)
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['returncode'], 1)
        self.assertEqual(manifest['error_type'], 'RuntimeError')
        self.assertEqual(manifest['phases'][0]['returncode'], 1)
        self.assertIn('requests.json', manifest['artifacts'])
        self.assertIn('input-image.png', manifest['artifacts'])


class PortProbeTests(unittest.TestCase):
    def test_live_listener_rejected_and_closed_connection_reusable(self):
        server = socket.socket()
        server.bind(('127.0.0.1', 0))
        server.listen(2)
        port = server.getsockname()[1]
        try:
            with self.assertRaises(ValueError):
                runner.ensure_no_listener(port)
            first, _ = server.accept()
            first.close()
            client = socket.create_connection(('127.0.0.1', port))
            accepted, _ = server.accept()
            accepted.close()
            server.close()
            client.close()
            runner.ensure_no_listener(port)
        finally:
            server.close()


if __name__ == '__main__':
    unittest.main()
