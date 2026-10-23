# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Positive and adversarial checks for the strict numerical comparator."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors.numpy import load_file
from compare_reference import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mesh = Path(self.temp.name)/'mesh.obj'
        self.reference = load_file(str(Path(__file__).with_name('reference-boxed.safetensors')))
        self.actual = {k:self.reference['pred_'+k].copy() for k in ('vertices','keypoints_3d','joint_coords','camera')}
        self.write_mesh()

    def tearDown(self):
        self.temp.cleanup()

    def write_mesh(self):
        with self.mesh.open('w') as f:
            for row in self.actual['vertices']: f.write('v '+' '.join(f'{v:.6f}' for v in row)+'\n')
            for row in self.reference['faces']: f.write('f '+' '.join(str(int(v)+1) for v in row)+'\n')

    def test_identical(self):
        _, failures = compare(self.reference,self.actual,self.mesh)
        self.assertEqual(failures,[])

    def test_all_vertices_checked(self):
        self.actual['vertices'][-1,2] += .002
        self.write_mesh()
        _, failures = compare(self.reference,self.actual,self.mesh)
        self.assertTrue(any('vertices' in x for x in failures))

    def test_all_keypoints_checked(self):
        self.actual['keypoints_3d'][-1,0] += .002
        _, failures = compare(self.reference,self.actual,self.mesh)
        self.assertTrue(any('keypoints' in x for x in failures))

    def test_camera_relative(self):
        self.actual['camera'] *= 1.002
        _, failures = compare(self.reference,self.actual,self.mesh)
        self.assertTrue(any('camera' in x for x in failures))

    def test_nonfinite_rejected(self):
        for value in (np.nan,np.inf,-np.inf):
            self.actual['joint_coords'][0,0] = value
            with self.assertRaises(ValueError): compare(self.reference,self.actual,self.mesh)

    def test_missing_tensor_rejected(self):
        del self.actual['joint_coords']
        with self.assertRaises(ValueError): compare(self.reference,self.actual,self.mesh)

    def test_colored_obj_rejected(self):
        self.mesh.write_text('v 0 0 0 1 0 0\n')
        with self.assertRaises(ValueError): compare(self.reference,self.actual,self.mesh)

    def test_wrong_decimal_precision_rejected(self):
        for coordinate in ('0.00000','0.0000000','0e0'):
            self.write_mesh()
            lines = self.mesh.read_text().splitlines()
            fields = lines[0].split(); fields[1] = coordinate
            lines[0] = ' '.join(fields)
            self.mesh.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(ValueError): compare(self.reference,self.actual,self.mesh)

    def test_topology_rejected(self):
        self.reference['faces'][0,0] = (self.reference['faces'][0,0]+1)%18439
        with self.assertRaises(ValueError): compare(self.reference,self.actual,self.mesh)


if __name__ == '__main__': unittest.main()
