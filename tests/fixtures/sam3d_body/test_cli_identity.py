# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure driver evidence tests; never starts a CLI, build, model or GPU process."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

M = runpy.run_path(str(Path(__file__).with_name('cli_identity.py')))


class CliIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='body-cli-evidence-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        provenance = self.root / 'provenance.json'
        provenance.write_text(json.dumps({'cases': {'default': {'bbox': None, 'cam_int': None}, 'boxed': {'bbox': [732,114,1523,1451], 'cam_int': None}}}))
        self.image = self.root / 'image.png'
        self.image.write_bytes(b'synthetic input; no image decoding')
        self.plan = {'capture_kind': 'cli', 'case': 'default', 'run_dir': str(self.root),
                     'source': {'root': '/source'}, 'checkpoint_dir': '/checkpoint', 'image': str(self.image),
                     'executable': {'path': '/binary', 'sha256': 'a' * 64},
                     'helpers': {'provenance.json': {'path': str(provenance)}}}
        for name in M['ARTIFACTS']:
            (self.output / name).write_bytes(b'synthetic process-artifact fixture')
        (self.output / 'meta.json').write_text('{"family":"sam3d_body"}')
        for name in ('child.exit', 'wrapper.exit'):
            (self.root / name).write_text('0\n')
        self.log = 'Runtime device: GPU\n' + ''.join('saved ' + str(self.output / n) + '\n' for n in sorted(M['ARTIFACTS']))
        (self.root / 'child.log').write_text(self.log)
        command, environment = M['command_environment'](self.plan)
        (self.root / 'invocation.json').write_text(json.dumps({'command': command, 'environment': environment, 'cwd': '/source'}))

    def test_actual_cli_syntax_and_default_omission(self):
        command, environment = M['command_environment'](self.plan)
        self.assertEqual(command, ['/binary', 'predict', '--model', '/checkpoint', '--image', str(self.image), '--task', 'body_mesh', '--output', str(self.output)])
        self.assertNotIn('--opts', command)
        self.assertEqual(environment, {'MLX_ENABLE_TF32': '0', 'MLXCEL_DEVICE': 'gpu'})
        self.plan['case'] = 'boxed'
        command, _ = M['command_environment'](self.plan)
        self.assertEqual(command[-2:], ['--opts', '{"bbox":[732,114,1523,1451]}'])

    def test_receipt_truthfully_labels_cli_and_refuses_overwrite(self):
        M['write_process_receipt'](self.plan)
        receipt = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(receipt, M['process_receipt'](self.plan))
        self.assertEqual(receipt['receipt_author'], 'capture_wrapper')
        self.assertEqual(receipt['execution_api'], 'mlxcel predict')
        self.assertEqual(receipt['predict_options'], {'task': 'body_mesh'})
        self.assertNotIn('DensePredictor.predict', json.dumps(receipt))
        with self.assertRaises(ValueError):
            M['write_process_receipt'](self.plan)
        (self.output / 'mesh.obj').write_text('tampered')
        self.assertNotEqual(receipt, M['process_receipt'](self.plan))

    def test_rejects_wrong_device_test_log_saved_set_and_exit(self):
        for text in (self.log.replace('GPU', 'CPU'), self.log + 'test result: ok\n', self.log + 'Runtime device: GPU\n', self.log.replace('mesh.obj', 'other.obj')):
            (self.root / 'child.log').write_text(text)
            with self.assertRaises(ValueError):
                M['process_receipt'](self.plan)
        (self.root / 'child.log').write_text(self.log)
        (self.root / 'child.exit').write_text('1\n')
        with self.assertRaises(ValueError):
            M['write_process_receipt'](self.plan)
        self.assertFalse((self.output / 'run.json').exists())

    def test_extra_product_rejected_before_receipt(self):
        (self.output / 'unexpected.txt').write_text('extra')
        with self.assertRaises(ValueError):
            M['write_process_receipt'](self.plan)
        self.assertFalse((self.output / 'run.json').exists())


if __name__ == '__main__':
    unittest.main()
