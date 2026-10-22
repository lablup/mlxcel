#!/usr/bin/env python3
"""Fetch an isolated, revision-pinned upstream Sapiens2 reference package."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "abd3b2d7b728cb5372d4e092260da325e5ded209"
BASE = f"https://raw.githubusercontent.com/Blaizzy/mlx-vlm/{REVISION}/mlx_vlm/models/"


def legacy_seg_config(config: dict) -> dict:
    """Map only the metadata names absent from the cited upstream revision."""
    backbone, head = config["backbone_config"], config["head_config"]
    assert config["task"] == "seg"
    result = {
        "model_type": "sapiens2",
        "architectures": ["Sapiens2ForSemanticSegmentation"],
        "num_labels": head["num_classes"],
        "head_config": {
            "upsample_out_channels": head["deconv_out_channels"],
            "upsample_kernel_sizes": head["deconv_kernel_sizes"],
            "use_pixel_shuffle": False,
            "conv_out_channels": head["conv_out_channels"],
            "conv_kernel_sizes": head["conv_kernel_sizes"],
        },
    }
    fields = {
        "embed_dims": "hidden_size",
        "num_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads",
        "feedforward_channels": "intermediate_size",
        "image_size": "image_size",
        "patch_size": "patch_size",
        "n_storage_tokens": "num_register_tokens",
        "rope_base": "rope_theta",
        "mhsa_early": "num_first_full_attention_layers",
        "mhsa_late": "num_last_full_attention_layers",
        "final_norm": "normalize_backbone_outputs",
    }
    result.update({new: backbone[old] for old, new in fields.items()})
    return result


def prepare(destination: Path) -> None:
    root = destination / "reference/mlx_vlm/models"
    (root / "sapiens2").mkdir(parents=True, exist_ok=True)
    (root.parent / "__init__.py").write_text("")
    (root / "__init__.py").write_text("")
    hashes = {}
    files = ["base.py", "kernels.py", "interpolate.py"] + [
        "sapiens2/" + name + ".py"
        for name in [
            "__init__",
            "config",
            "backbone",
            "heads",
            "image",
            "pose",
            "sapiens2",
            "generate",
        ]
    ]
    for relative in files:
        data = urllib.request.urlopen(BASE + relative).read()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        source = data.decode()
        if relative == "base.py":
            nodes = [
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name == "BaseModelConfig"
            ]
            source = "import inspect\nfrom dataclasses import dataclass\n" + ast.unparse(
                nodes[0]
            )
        elif relative == "sapiens2/image.py":
            source = source.replace(
                "from ..base import install_auto_processor_patch\n", ""
            )
            source = source.replace(
                'install_auto_processor_patch(["sapiens2"], Sapiens2Processor)', ""
            )
        (root / relative).write_text(source)
    (destination / "reference-source-hashes.json").write_text(
        json.dumps({"revision": REVISION, "sha256": hashes}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--legacy-seg-config", type=Path)
    args = parser.parse_args()
    prepare(args.destination)
    if args.legacy_seg_config:
        output = args.destination / "seg-reference-config.json"
        output.write_text(
            json.dumps(
                legacy_seg_config(json.loads(args.legacy_seg_config.read_text())),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
