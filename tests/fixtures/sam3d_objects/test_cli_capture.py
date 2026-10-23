"""Bounded synthetic parser/negative tests; no model or capture is executed."""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import cli_identity as identity
import run_cli_capture as driver
import capture_identity as shared
import run_rust_capture as monitor


class CliIdentityTests(unittest.TestCase):
    def test_literal_default_command_has_no_inference_overrides(self) -> None:
        plan = {"input_dir": "/inputs", "run_dir": "/run", "checkpoint_dir": "/checkpoint", "executable": {"path": "/bin/pinned"}}
        command, env = identity.command_environment(plan)
        self.assertEqual(command, ["/bin/pinned", "predict", "--model", "/checkpoint", "--image", "/inputs/object.png", "--mask", "/inputs/mask.png", "--output", "/run/output"])
        self.assertFalse({"--opts", "--task", "--pointmap", "--ignored"} & set(command))
        self.assertEqual(env, {"MLX_ENABLE_TF32": "0", "MLXCEL_DEVICE": "gpu"})

    def test_cpu_duplicate_runtime_and_missing_saved_files_rejected(self) -> None:
        root = Path("/run/output")
        log = "Runtime device: GPU\n" + "".join(f"saved {root / name}\n" for name in sorted(identity.OUTPUT_FILES))
        identity.validate_log(log, root)
        for changed in (log.replace("GPU", "CPU"), log + "Runtime device: GPU\n", log.replace("saved /run/output/mesh.obj\n", ""), log + "saved /unexpected\n"):
            with self.subTest(log=changed), self.assertRaises(ValueError):
                identity.validate_log(changed, root)

    def test_instances_sidecar_requires_exact_empty_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / identity.INSTANCE_SIDECAR
            path.write_bytes(b"[]")
            self.assertEqual(identity.empty_instances_sha(path), identity.sha(path))
            for payload in (b"[{}]", b"{", b"[]\n"):
                path.write_bytes(payload)
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "canonical empty"):
                    identity.empty_instances_sha(path)

    def test_obj_complete_parser_rejects_nonfinite_bad_faces_and_hidden_rows(self) -> None:
        valid = "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-unit.obj"
            path.write_text(valid)
            vertices, faces = identity.read_obj(path)
            self.assertEqual(vertices.shape, (3, 6))
            np.testing.assert_array_equal(faces, [[0, 1, 2]])
            for invalid in (valid.replace("v 0 0 0", "v nan 0 0", 1), valid.replace("f 1 2 3", "f 1 2 4"),
                            valid + "ignored hidden data\n", valid.replace("f 1 2 3", "f 0 2 3"),
                            valid + "v 0 0 0 0 0 0\n", valid.replace("v 0 0 0", "v 1e300 0 0", 1)):
                path.write_text(invalid)
                with self.subTest(text=invalid), self.assertRaises(ValueError):
                    identity.read_obj(path)

    def test_ply_validates_all_properties_and_exact_binary_payload(self) -> None:
        header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n" +
                  "".join(f"property float {name}\n" for name in identity.FIELDS) + "end_header\n").encode()
        values = [0.0] * 17
        values[-4] = 1.0
        payload = struct.pack("<17f", *values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-unit.ply"
            path.write_bytes(header + payload)
            self.assertEqual(identity.read_ply(path).shape, (1, 17))
            cases = [header + payload[:-1], header + payload + b"extra", header.replace(b"f_dc_1", b"f_dc_0") + payload]
            values[8] = float("nan")
            cases.append(header + struct.pack("<17f", *values))
            values[8], values[3] = 0, 1
            cases.append(header + struct.pack("<17f", *values))
            for invalid in cases:
                path.write_bytes(invalid)
                with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                    identity.read_ply(path)

    def test_compact_requires_conditioning_schema_and_finite_full_arrays(self) -> None:
        values = {name: np.ones(shape, dtype=np.float32) for name, shape in identity.POSE.items()}
        values.update(coords=np.zeros((2, 4), dtype=np.int32), latents=np.zeros((2, 8), dtype=np.float32), pointmap_conditioned=np.array([True]))
        identity.validate_compact(values)
        changes = [("coords", values["coords"].astype(np.float32)), ("latents", np.zeros((2, 7), dtype=np.float32)),
                   ("pointmap_conditioned", np.array([False])), ("pointmap_conditioned", np.array([1], dtype=np.int32)),
                   ("pose.scale", np.zeros(3, dtype=np.float32)), ("latents", np.full((2, 8), np.nan, dtype=np.float32))]
        for name, changed in changes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                identity.validate_compact({**values, name: changed})

    def test_pinned_import_rejects_changed_code_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "helper.py"
            marker = Path(directory) / "executed"
            path.write_text(f"open({str(marker)!r}, 'w').write('bad')")
            with self.assertRaisesRegex(ValueError, "before import"):
                identity.load_module(path, "never_imported", "0" * 64)
            self.assertFalse(marker.exists())

    def test_all_helpers_must_authenticate_before_any_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helpers = {}
            for name in identity.HELPERS:
                path = root / name
                path.write_text("unit helper placeholder")
                helpers[name] = {"path": str(path), "sha256": identity.sha(path)}
            helpers["provenance.py"]["sha256"] = "0" * 64
            plan = root / "plan.json"
            plan.write_text(json.dumps({"helpers": helpers}))
            with patch.object(identity, "load_module") as load, self.assertRaisesRegex(ValueError, "Helper changed"):
                identity.authenticate_helpers(plan, identity.sha(plan))
            load.assert_not_called()

    def test_existing_full_array_thresholds_reject_changed_data(self) -> None:
        from compare_reference import compare
        reference = {"coords": np.zeros((2, 4), dtype=np.int32),
                     "latents": np.ones((2, 8), dtype=np.float32),
                     "mesh.vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)}
        changed = {**reference, "latents": reference["latents"] * 1.1}
        self.assertIn("latents relative RMS >=1e-2", compare(reference, changed)[1])
        changed = {**reference, "mesh.vertices": reference["mesh.vertices"] + 1}
        self.assertIn("normalized Chamfer >=1e-3", compare(reference, changed)[1])
        changed = {**reference, "mesh.vertices": reference["mesh.vertices"][:3]}
        self.assertIn("mesh vertex count error >1%", compare(reference, changed)[1])
        changed = {**reference, "coords": reference["coords"] + 1}
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            compare(reference, changed)

    def test_no_metrics_on_failed_capture_authentication(self) -> None:
        with patch.object(identity, "authenticate_plan", side_effect=ValueError("unapproved actual/reference")), \
             patch.object(identity, "read_obj") as obj, patch.object(identity, "_difference") as difference:
            with self.assertRaisesRegex(ValueError, "unapproved"):
                identity.validate_capture(Path("/absent-plan"), "0" * 64)
            obj.assert_not_called()
            difference.assert_not_called()


class CliDriverTests(unittest.TestCase):
    def test_reused_monitor_policy_is_the_approved_bounded_policy(self) -> None:
        self.assertEqual(shared.POLICY, {"interval_seconds": .1, "reserve_bytes": 3 * 2**30, "low_samples": 2,
                                       "swap_loss_bytes": 256 * 2**20, "swap_pressure_available_bytes": 8 * 2**30, "child_swap_bytes": 64 * 2**20, "term_grace_seconds": 5})
        self.assertTrue(callable(monitor.monitor))

    def test_unapproved_plan_is_rejected_before_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"run_dir": str(Path(directory) / "never-created")}))
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                driver.bootstrap(plan, "0" * 64)
            self.assertFalse((Path(directory) / "never-created").exists())

    def test_live_child_attestation_rejects_wrong_environment(self) -> None:
        child = Mock(pid=9876)
        plan = {"executable": {"path": "/approved", "sha256": "0" * 64}}
        with patch.object(Path, "resolve", return_value=Path("/approved")), \
             patch.object(Path, "read_bytes", return_value=b"MLX_ENABLE_TF32=1\0MLXCEL_DEVICE=gpu\0"), \
             patch.object(driver, "digest") as digest:
            with self.assertRaisesRegex(ValueError, "environment differs"):
                driver.attest_child(child, plan, {"MLX_ENABLE_TF32": "0", "MLXCEL_DEVICE": "gpu"})
            digest.assert_not_called()

    def test_failed_child_attestation_terminates_and_never_writes_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "frozen").mkdir()
            for name in driver.HELPERS:
                (run / "frozen" / name).write_text("unit helper")
            plan = {"input_dir": "/inputs", "run_dir": str(run), "checkpoint_dir": "/checkpoint",
                    "executable": {"path": "/never-executed"}, "source": {"root": str(run)}}
            mon = SimpleNamespace(memory=Mock(return_value={"MemAvailable": shared.RESERVE + 1, "SwapFree": 0, "MemTotal": 16 * 2**30}),
                                  terminate=Mock(return_value=-15), monitor=Mock())
            validator = SimpleNamespace(authenticate_plan=Mock(return_value=(plan, {}, {"run_rust_capture": mon, "capture_identity": shared})),
                                        command_environment=identity.command_environment, validate_capture=Mock())
            child = Mock(pid=9876)
            with patch.object(driver, "bootstrap", return_value=(plan, run)), \
                 patch.object(driver, "freeze", return_value=validator), \
                 patch.object(driver.subprocess, "Popen", return_value=child) as spawn, \
                 patch.object(driver, "attest_child", side_effect=ValueError("mock wrong executable")), \
                 patch.object(driver.sys, "stderr", io.StringIO()):
                self.assertEqual(driver.run_capture(run / "unused", "a" * 64), 1)
            mon.terminate.assert_called_once_with(child)
            mon.monitor.assert_not_called()
            validator.validate_capture.assert_not_called()
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual((run / "child.exit").read_text(), "-15\n")
            self.assertEqual((run / "wrapper.exit").read_text(), "1\n")
            self.assertFalse((run / "capture-receipt.json").exists())
            self.assertFalse((run / "comparison.json").exists())
            self.assertFalse((run / "output").exists())


if __name__ == "__main__":
    unittest.main()
