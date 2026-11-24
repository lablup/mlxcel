#!/usr/bin/env python3
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
"""Load the mlx-community Inkling 2 4-bit export through mlx-vlm (#3378).

mlx-vlm (466069f5) reads only the native Inkling 2 layout (`model.llm.*`,
`model.visual.*`) plus its own converted layout. The mlx-community export
(`mlx-community/Inkling-2-Mini-4bit`) uses the Transformers layout
(`model.language_model.*`, `model.vision_tower.*`, top-level `lm_head.*`,
`*_message_gate`, `post_attention_convolution`, a fused `gate_up_proj`), so
mlx-vlm cannot load it directly. This module rewrites the export into
mlx-vlm's converted layout, the names `Model.sanitize` passes through
unchanged, and writes a directory `mlx_vlm.load` accepts. It touches names and
expert metadata only: no dense tensor is requantized or recomputed.

What the rewrite does:

* renames every tensor to mlx-vlm's parameter name (the same table
  `src/models/inkling2/sanitize.rs` applies on the mlxcel side);
* splits the fused `gate_up_proj` planes into `gate_proj` / `up_proj` with
  mlx-vlm's own `_split_gate_up` (rows are interleaved, packed codes and
  scales alike);
* sets `weight_scale_2 = global_scale / 2688` for both halves (the export
  stores MLX's native scale, `2688 * weight_scale_2`);
* declares the per-expert quantization modes in `config.json`: NVFP4
  (group 16, 4 bit) or MXFP8 (group 32, 8 bit), inferred per projection from
  the packed width and the scale width, over the export's affine group-64
  base for dense layers.

The export ships no `processor_config.json`; mlx-vlm needs one to preprocess
images, so files missing from the export are linked from a native checkpoint
directory (`--native-dir`, default: a sibling `inkling-2-mini`).

Usage:
    inkling2_mlx_community_remap.py MODEL_DIR [OUT_DIR] [--native-dir DIR]

prints the directory to hand to `mlx_vlm.load`. Importable as
`prepare_model_dir(model_dir, ...)`; the two logit-trace scripts call it for
`--inkling2-mlx-community`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

#: MLX's native NVFP4 global scale is `GLOBAL_SCALE_DENOMINATOR * weight_scale_2`.
GLOBAL_SCALE_DENOMINATOR = 2688.0
DONE_MARKER = ".inkling2-remap-done"

_CONTROLLER = {"q": "attn", "k": "k", "v": "v", "mlp": "mlp"}
_CONTROLLER_LEAF = {
    "down_proj": "down_weight",
    "forget_gate_proj": "forget_weight",
    "message_gate_proj": "gain_weight",
}
_LAYER_RENAMES = {
    "mlp.gate.weight": "mlp.gate_weight",
    "mlp.gate.e_score_correction_bias": "mlp.e_score_correction_bias",
    "post_attention_convolution.conv1d.weight": "attn_sconv.conv.weight",
    "post_mlp_convolution.conv1d.weight": "mlp_sconv.conv.weight",
    "self_attn.k_sconv.conv1d.weight": "self_attn.k_sconv.conv.weight",
    "self_attn.v_sconv.conv1d.weight": "self_attn.v_sconv.conv.weight",
    "self_attn.q_l2_clip.scale": "self_attn.q_scale",
    "self_attn.rel_l2_clip.scale": "self_attn.rel_scale",
    "self_attn.rel_logits_proj.proj": "self_attn.rel_proj",
}
_TEXT = "language_model.model."
_EXPERT = re.compile(r"^layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)\.(.+)$")
_MESSAGE_GATE = re.compile(r"^(q|k|v|mlp)_message_gate\.(.+)$")


def map_key(key: str) -> list[tuple[str, str]]:
    """Map one export tensor name to `[(mlx-vlm name, role)]`.

    `role` is `"copy"` for a plain rename, `"split"` for a `gate_up_proj`
    plane that becomes two tensors (gate first, then up) and `"scale2"` for a
    `global_scale`, whose value becomes `weight_scale_2` (`/ 2688`).
    """
    if key.startswith("lm_head."):
        return [("language_model." + key, "copy")]
    if key.startswith("model.vision_tower."):
        sub = key[len("model.vision_tower.") :]
        sub = sub.replace("merger.linear_1.", "proj_in.").replace(
            "merger.linear_2.", "proj_out."
        )
        sub = sub.replace(
            "embeddings.position_embedding.weight", "embeddings.position_embedding"
        )
        return [("vision_tower." + sub, "copy")]
    if not key.startswith("model.language_model."):
        raise ValueError(f"unmapped Inkling 2 mlx-community tensor: {key}")
    sub = key[len("model.language_model.") :]
    if sub == "message_input.query":
        return [(_TEXT + "message_queries", "copy")]
    expert = _EXPERT.match(sub)
    if expert:
        layer, fused, leaf = expert.groups()
        base = f"{_TEXT}layers.{layer}.mlp.switch_mlp."
        projections = ("gate_proj", "up_proj") if fused == "gate_up_proj" else ("down_proj",)
        if leaf == "global_scale":
            return [(f"{base}{p}.weight_scale_2", "scale2") for p in projections]
        role = "split" if fused == "gate_up_proj" else "copy"
        return [(f"{base}{p}.{leaf}", role) for p in projections]
    if not sub.startswith("layers."):
        return [(_TEXT + sub, "copy")]
    layer, _, leaf = sub[len("layers.") :].partition(".")
    base = f"{_TEXT}layers.{layer}."
    gate = _MESSAGE_GATE.match(leaf)
    if gate:
        controller, proj = gate.groups()
        name, _, suffix = proj.partition(".")
        if name not in _CONTROLLER_LEAF or suffix != "weight":
            raise ValueError(f"unmapped Inkling 2 mlx-community tensor: {key}")
        return [(f"{base}controllers.{_CONTROLLER[controller]}.{_CONTROLLER_LEAF[name]}", "copy")]
    return [(base + _LAYER_RENAMES.get(leaf, leaf), "copy")]


def expert_quantization(weight_width: int, scales_width: int) -> dict:
    """NVFP4 or MXFP8 settings from one expert plane's packed and scale widths.

    The packed width is `in_features * bits / 32` words and the scale width is
    `in_features / group_size`, so `weight / scales` is 2 for NVFP4 (4 bit,
    group 16) and 8 for MXFP8 (8 bit, group 32).
    """
    ratio = weight_width / scales_width
    if ratio == 2:
        return {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    if ratio == 8:
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    raise ValueError(
        f"expert plane of width {weight_width} with {scales_width} scale columns "
        "is neither NVFP4 nor MXFP8"
    )


def remap_config(config: dict, weights: dict) -> dict:
    """The mlx-vlm `config.json`: the export's own, plus per-expert modes.

    `weights` maps each already-remapped expert name to its array (only
    `.shape` is read).
    """
    base = dict(config.get("quantization") or {"group_size": 64, "bits": 4, "mode": "affine"})
    quantization = {k: v for k, v in base.items() if k in ("group_size", "bits", "mode")}
    for name, array in weights.items():
        if not (name.endswith(".weight") and ".mlp.switch_mlp." in name):
            continue
        module = name[: -len(".weight")]
        scales = weights[module + ".scales"]
        quantization[module] = expert_quantization(array.shape[-1], scales.shape[-1])
    out = dict(config)
    out["quantization"] = quantization
    out["quantization_config"] = dict(quantization)
    return out


def remap_weights(weights: dict) -> dict:
    """Rename and split one shard's tensors (mx arrays) into mlx-vlm's layout."""
    from mlx_vlm.models.inkling.inkling import _split_gate_up

    out: dict = {}
    for key, value in weights.items():
        targets = map_key(key)
        for index, (name, role) in enumerate(targets):
            if role == "scale2":
                part = value / GLOBAL_SCALE_DENOMINATOR
            elif role == "split":
                part = _split_gate_up(value)[index]
            else:
                part = value
            if name in out:
                raise ValueError(f"two tensors map onto {name}")
            out[name] = part
    return out


def _link_missing(src: Path, dst: Path) -> None:
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        if entry.suffix == ".safetensors" or entry.name in (
            "model.safetensors.index.json",
            "config.json",
            # The native sidecar would redeclare the experts as native-layout.
            "hf_quant_config.json",
        ):
            continue
        if not target.exists() and not target.is_symlink():
            target.symlink_to(entry.resolve())


def default_out_dir(model_dir: Path) -> Path:
    cache = Path(os.environ.get("INKLING2_REMAP_DIR", tempfile.gettempdir()))
    return cache / f"inkling2-mlx-vlm-{model_dir.resolve().name}"


def prepare_model_dir(
    model_dir: str | Path,
    out_dir: str | Path | None = None,
    native_dir: str | Path | None = None,
) -> Path:
    """Write (or reuse) the mlx-vlm view of an mlx-community export."""
    import mlx.core as mx

    model_dir = Path(model_dir)
    out = Path(out_dir) if out_dir else default_out_dir(model_dir)
    if (out / DONE_MARKER).exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weight_map: dict[str, str] = {}
    shapes: dict = {}
    for number, shard in enumerate(shards, start=1):
        remapped = remap_weights(mx.load(str(model_dir / shard)))
        name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), remapped, {"format": "mlx"})
        for tensor, array in remapped.items():
            weight_map[tensor] = name
            if ".mlp.switch_mlp." in tensor:
                shapes[tensor] = array
        print(f"remapped {shard} -> {name} ({len(remapped)} tensors)", file=sys.stderr)
        del remapped
    config = json.loads((model_dir / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(remap_config(config, shapes), indent=2))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )
    _link_missing(model_dir, out)
    native = Path(native_dir) if native_dir else model_dir.parent / "inkling-2-mini"
    if native.is_dir():
        _link_missing(native, out)
    (out / DONE_MARKER).write_text("ok\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--native-dir")
    args = ap.parse_args()
    print(prepare_model_dir(args.model_dir, args.out_dir, args.native_dir))


if __name__ == "__main__":
    main()
