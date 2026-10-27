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

"""CPU-only regression fixtures for the two standalone evidence tools."""
import importlib.util
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import socket
import json

VALID = '''conc ok fail ttft_ms(mean) ttft_ms(p95) decode_tok_s(mean) aggregate_tok_s
1 1 0 238.5 238.5 134.6 108.3
2 2 0 517.4 517.5 112.5 155.5
4 4 0 874.8 875.2 80.7 209.0
'''


class LadderValidation(unittest.TestCase):
    def test_valid_and_failed_ladders(self):
        for filename in ('bench_batched_decode.py', 'archive_batched_decode.py'):
            spec = importlib.util.spec_from_file_location('tool', Path(__file__).resolve().parents[1] / 'scripts' / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.subTest(tool=filename, case='valid'):
                module.validate_ladder(VALID)
            invalid = {
                'http500': 'request error: HTTP Error 500: Internal Server Error\n' + VALID,
                'failed_count': VALID.replace('4 4 0', '4 3 1'),
                'missing_row': '\n'.join(line for line in VALID.splitlines() if not line.startswith('2 ')),
                'duplicate_row': VALID + '1 1 0 1 1 1 1\n',
                'nonfinite': VALID.replace('209.0', 'nan'),
            }
            for case, text in invalid.items():
                with self.subTest(tool=filename, case=case):
                    with self.assertRaises(ValueError):
                        module.validate_ladder(text)


def load_tool(filename):
    path = Path(__file__).resolve().parents[1] / 'scripts' / filename
    spec = importlib.util.spec_from_file_location('evidence_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveRequirements(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source' / 'candidate'
        self.source.mkdir(parents=True)
        for phase in ('greedy', 'ladder', 'trace-b1'):
            (self.source / (phase + '.out')).write_text(VALID if phase == 'ladder' else 'fixture\n')
            (self.source / (phase + '.err')).write_text('diagnostic fixture\n')
            (self.source / (phase + '.status.json')).write_text('{"returncode": 0}')
        (self.source / 'server-route.log').write_text('decode path: native batched forward\n')
        for name in ('config.json', 'corpus.txt'):
            (self.root / name).write_text('fixture\n')
        self.run = {'label': 'candidate', 'destination': 'model/candidate',
                    'verified': True, 'source_revision': 'fixture',
                    'config': str(self.root / 'config.json'),
                    'corpus': str(self.root / 'corpus.txt')}
        self.tool = load_tool('archive_batched_decode.py')

    def plan(self):
        return self.tool.plan_run(self.root / 'source', self.root / 'archive', self.run)

    def test_expected_trace_rows_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, 'expected_trace_rows'):
            self.plan()

    def test_candidate_missing_expected_b4_is_rejected(self):
        self.run['expected_trace_rows'] = [1, 4]
        with self.assertRaisesRegex(ValueError, 'trace-b4'):
            self.plan()

    def test_expected_phases_include_stderr_and_metadata(self):
        self.run['expected_trace_rows'] = [1]
        _, files, _, metadata = self.plan()
        self.assertEqual(metadata['expected_trace_rows'], [1])
        self.assertTrue({'greedy.err', 'ladder.err', 'trace-b1.err'} <= {p.name for p in files})
        (self.source / 'trace-b1.err').unlink()
        with self.assertRaisesRegex(ValueError, 'stderr'):
            self.plan()


    def test_original_run_manifest_survives_archiving(self):
        self.run['expected_trace_rows'] = [1]
        original = b'{"source_revision":"fixture","binary_sha256":{"server":"recorded"}}\n'
        (self.source / 'manifest.json').write_bytes(original)
        mapping = self.root / 'mapping.json'
        mapping.write_text(json.dumps({'runs': [self.run]}))
        destination = self.root / 'archive'
        result = subprocess.run([sys.executable, str(Path(self.tool.__file__)),
                                 '--source-root', str(self.root / 'source'),
                                 '--destination', str(destination), '--mapping', str(mapping)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        target = destination / 'model/candidate'
        self.assertEqual((target / 'run-metadata.json').read_bytes(), original)
        archived = json.loads((target / 'manifest.json').read_text())
        self.assertEqual(archived['artifacts']['run-metadata.json']['sha256'],
                         hashlib.sha256(original).hexdigest())


    def test_original_run_manifest_survives_rearchiving(self):
        self.run['expected_trace_rows'] = [1]
        original = b'{"binary_sha256":{"server":"original-runtime"}}\n'
        (self.source / 'run-metadata.json').write_bytes(original)
        (self.source / 'manifest.json').write_text('{"artifacts":{},"verified":true}')
        mapping = self.root / 'mapping.json'
        mapping.write_text(json.dumps({'runs': [self.run]}))
        destination = self.root / 'rearchive'
        result = subprocess.run([sys.executable, str(Path(self.tool.__file__)),
                                 '--source-root', str(self.root / 'source'),
                                 '--destination', str(destination), '--mapping', str(mapping)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        target = destination / 'model/candidate'
        self.assertEqual((target / 'run-metadata.json').read_bytes(), original)
        archived = json.loads((target / 'manifest.json').read_text())
        self.assertEqual(archived['artifacts']['run-metadata.json']['sha256'],
                         hashlib.sha256(original).hexdigest())


class PortProbeRequirements(unittest.TestCase):
    def test_live_listener_rejected_and_closed_connection_reusable(self):
        tool = load_tool('bench_batched_decode.py')
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0))
            server.listen(2)
            port = server.getsockname()[1]
            with self.assertRaises(ValueError):
                tool.ensure_no_listener(port)
            first, _ = server.accept()
            first.close()
            with socket.create_connection(('127.0.0.1', port)) as client:
                accepted, _ = server.accept()
                accepted.close()
                server.close()
            tool.ensure_no_listener(port)


class BackendRequirements(unittest.TestCase):
    def test_metal_requires_actual_library_and_hashes_it(self):
        tool = load_tool('bench_batched_decode.py')
        with self.assertRaisesRegex(ValueError, '--loaded-metallib'):
            tool.validate_backend('metal', None)
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder) / 'actual.metallib'
            with self.assertRaisesRegex(ValueError, 'missing'):
                tool.validate_backend('metal', library)
            library.write_bytes(b'actual selected library fixture')
            metadata = tool.validate_backend('metal', library)
            self.assertEqual(metadata['loaded_metallib']['path'], str(library.resolve()))
            self.assertEqual(metadata['loaded_metallib']['sha256'], hashlib.sha256(library.read_bytes()).hexdigest())

    def test_nonmetal_needs_no_library(self):
        tool = load_tool('bench_batched_decode.py')
        for backend in ('cpu', 'cuda', 'rocm'):
            self.assertEqual(tool.validate_backend(backend, None), {'backend': backend, 'loaded_metallib': None})
        with self.assertRaises(ValueError):
            tool.validate_backend('cpu', Path('unrelated.metallib'))

    def test_cli_requires_explicit_backend(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/bench_batched_decode.py'
        result = subprocess.run([sys.executable, str(script), '--repo', '.', '--binary-dir', '.',
                                 '--model', '.', '--corpus', 'unused', '--output', 'unused',
                                 '--source-revision', 'fixture'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--backend', result.stderr)


if __name__ == '__main__':
    unittest.main()
