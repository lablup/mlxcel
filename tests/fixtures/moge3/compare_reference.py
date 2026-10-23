"""Compare complete MoGe3 outputs without hiding mask/non-finite mismatches."""

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np
from provenance import canonical, require, validate_outputs, verify_receipt


def compare(actual_path, reference_path, device):
    mx.set_default_device(mx.cpu)
    actual_receipt = json.loads(pathlib.Path(actual_path).with_suffix(".json").read_text())
    verified = pathlib.Path(reference_path).with_suffix(".verified.json")
    reference_receipt = json.loads((verified if verified.exists() else pathlib.Path(reference_path).with_suffix(".json")).read_text())
    actual_opts = verify_receipt(actual_receipt, actual_path, reference=False)
    reference_opts = verify_receipt(reference_receipt, reference_path, reference=True)
    require(actual_opts == reference_opts, "Actual/reference inference options differ")
    for receipt in (actual_receipt, reference_receipt):
        require(receipt["device"] in (device, f"Device({device}, 0)"), "Tolerance device does not match actual/reference receipts")
    require(actual_receipt["checkpoint_sha256"] == reference_receipt["checkpoint_sha256"], "Actual/reference checkpoints differ")
    actual = mx.load(str(actual_path))
    reference = mx.load(str(reference_path))
    validate_outputs(actual, actual_opts, canonical()["image_shape"])
    validate_outputs(reference, reference_opts, canonical()["image_shape"], reference=True)
    if actual.keys() != reference.keys():
        raise AssertionError(f"Output keys differ: {sorted(actual)} != {sorted(reference)}")
    required = {"points", "depth", "mask", "normal", "intrinsics"}
    if not required.issubset(reference):
        raise AssertionError(f"Missing required outputs: {sorted(required - reference.keys())}")
    metrics = {}
    failures = []
    for key in sorted(reference):
        got, expected = np.asarray(actual[key]), np.asarray(reference[key])
        if got.shape != expected.shape:
            raise AssertionError(f"{key}: shapes differ {got.shape} != {expected.shape}")
        if key == "mask":
            if not np.array_equal(got, expected):
                failures.append(f"{key}: {np.count_nonzero(got != expected)} mask mismatches")
            metrics[key] = {"shape": list(got.shape), "equal": bool(np.array_equal(got, expected))}
            continue
        finite = np.isfinite(expected)
        same_support = np.array_equal(np.isfinite(got), finite)
        same_nan = np.array_equal(np.isnan(got), np.isnan(expected))
        same_inf = np.array_equal(np.isposinf(got), np.isposinf(expected)) and np.array_equal(
            np.isneginf(got), np.isneginf(expected)
        )
        if not (same_support and same_nan and same_inf):
            failures.append(f"{key}: finite/NaN/infinity support differs")
            metrics[key] = {"support_equal": False}
            continue
        if not finite.any():
            failures.append(f"{key}: no finite values for numerical comparison")
            continue
        lhs = got[finite].astype(np.float64)
        rhs = expected[finite].astype(np.float64)
        delta = lhs - rhs
        rms = float(np.sqrt(np.mean(delta * delta)))
        reference_rms = float(np.sqrt(np.mean(rhs * rhs)))
        relative_rms = rms / max(reference_rms, 1e-12)
        max_abs = float(np.max(np.abs(delta)))
        entry = {"shape": list(got.shape), "finite": int(finite.sum()), "elements": got.size,
                 "rms": rms, "relative_rms": relative_rms, "max_abs": max_abs}
        metrics[key] = entry
        if key.startswith("intrinsics"):
            passed = max_abs <= 1e-3
        else:
            passed = relative_rms <= (1e-4 if device == "cpu" else 5e-3)
        if not passed:
            failures.append(f"{key}: {entry}")
    return metrics, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=pathlib.Path)
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    metrics, failures = compare(args.actual, args.reference, args.device)
    report = {"device": args.device, "metrics": metrics, "failures": failures}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
