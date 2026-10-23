"""Negative checks for acceptance metrics; these are not model references."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_reference import compare, load_file


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            'coords': np.array([[0, 1, 2, 3]], np.int32),
            'latents': np.ones((1, 8), np.float32),
            'mesh.vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        }

    def test_identical_complete_sets(self):
        metrics, failures = compare(self.reference, self.reference)
        self.assertEqual(failures, [])
        self.assertEqual(metrics['normalized_symmetric_chamfer'], 0)

    def test_coordinate_change_rejected(self):
        with self.assertRaises(ValueError):
            compare(self.reference, {**self.reference, 'coords': self.reference['coords'] + 1})

    def test_latent_threshold_enforced(self):
        _, failures = compare(self.reference, {**self.reference, 'latents': self.reference['latents'] * 1.02})
        self.assertIn('latents relative RMS >=1e-2', failures)

    def test_mesh_translation_rejected(self):
        _, failures = compare(self.reference, {**self.reference, 'mesh.vertices': self.reference['mesh.vertices'] + .1})
        self.assertIn('normalized Chamfer >=1e-3', failures)

    def test_degenerate_scale_rejected(self):
        reference = {**self.reference, 'mesh.vertices': np.zeros((3, 3), np.float32)}
        with self.assertRaises(ValueError):
            compare(reference, reference)

    def test_tensor_payload_overlap_rejected(self):
        header = json.dumps({name: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]} for name in ('a', 'b')}).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.safetensors'
            path.write_bytes(len(header).to_bytes(8, 'little') + header + bytes(4))
            with self.assertRaises(ValueError):
                load_file(path)


if __name__ == '__main__':
    unittest.main()
