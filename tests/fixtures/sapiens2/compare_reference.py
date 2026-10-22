#!/usr/bin/env python3
"""Compare mlxcel result.safetensors with a Sapiens2 upstream reference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file


def relative_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    error = np.sqrt(np.mean(np.square(actual.astype(np.float64) - expected)))
    scale = np.sqrt(np.mean(np.square(expected.astype(np.float64))))
    return float(error / max(scale, np.finfo(np.float64).tiny))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("pose", "seg", "normal", "pointmap", "matting"))
    parser.add_argument("result", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    actual = load_file(args.result)

    if args.task == "pose":
        reference = json.loads(args.reference.read_text())
        expected = np.asarray(reference["keypoints"], dtype=np.float32)
        distances = np.linalg.norm(actual["keypoints"] - expected, axis=-1)
        metric = float(distances.max(initial=0.0))
        print(f"maximum keypoint error: {metric:.6f} px")
        if metric > 1.5:
            raise SystemExit("pose parity exceeds 1.5 px")
    elif args.task == "seg":
        expected = np.asarray(Image.open(args.reference), dtype=np.int32)
        agreement = float(np.mean(actual["segmentation"].astype(np.int32) == expected))
        print(f"segmentation label agreement: {agreement:.8%}")
        if agreement < 0.995:
            raise SystemExit("segmentation agreement is below 99.5%")
    else:
        with np.load(args.reference) as reference:
            key = {
                "normal": "normals",
                "pointmap": "pointmaps",
                "matting": "foregrounds",
            }[args.task]
            metric = relative_rms(actual[key], reference[key])
            print(f"{key} relative RMS: {metric:.8g}")
            if args.task in {"normal", "pointmap"} and metric >= 1e-2:
                raise SystemExit(f"{args.task} relative RMS is not below 1e-2")


if __name__ == "__main__":
    main()
