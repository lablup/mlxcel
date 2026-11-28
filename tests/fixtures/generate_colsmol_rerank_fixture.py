#!/usr/bin/env python3
"""Build the merged colSmol-256M checkpoint and the two rendered pages the
late-interaction rerank gate (#1426) scores.

``vidore/colSmol-256M`` is a LoRA adapter over
``vidore/ColSmolVLM-Instruct-256M-base``, and mlxcel deliberately does not
merge adapters at load time, so the gate needs a merged directory. This script
merges the adapter the way PEFT does (``W + (lora_alpha / r) * B @ A`` on every
targeted text-model projection, computed in float32 and stored back in the
base dtype), copies the base checkpoint's other files, and copies the adapter's
trained ``1_Dense/`` projection, which the ColIdefics3 loader promotes over the
base's untrained ``linear.*`` head.

It then renders a page that answers ``What was the total revenue in 2023?``
(a revenue table) and an unrelated page (a recipe) into ``rerank_pages/`` in
the merged directory. PR #1414's measurement used pages that were not
committed, so these are stand-ins: the gate asserts the ordering and a margin,
not PR #1414's absolute numbers.

Nothing here is committed output; the gate soft-skips when the directory is
absent.

Usage (from the repository root):
    uv run --python 3.12 --with safetensors --with numpy --with pillow \\
        python tests/fixtures/generate_colsmol_rerank_fixture.py \\
        models/colsmolvlm-instruct-256m-base models/colsmol-256m \\
        models/colsmol-256m-merged
"""

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open

ADAPTER_PREFIX = "base_model.model."


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    # Round to nearest even, the same rounding torch applies.
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def read_tensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read every tensor, keeping bf16 as raw uint16 with its dtype noted."""
    tensors: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    header_len = int.from_bytes(path.read_bytes()[:8], "little")
    header = json.loads(path.read_bytes()[8 : 8 + header_len])
    blob = path.read_bytes()[8 + header_len :]
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype = meta["dtype"]
        np_dtype = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}[dtype]
        tensors[name] = np.frombuffer(blob[start:end], dtype=np_dtype).reshape(meta["shape"])
        dtypes[name] = dtype
    return tensors, dtypes


def merge(base: Path, adapter: Path, out: Path) -> None:
    config = json.loads((adapter / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    tensors, dtypes = read_tensors(base / "model.safetensors")
    lora = safe_open(str(adapter / "adapter_model.safetensors"), "np")
    merged = 0
    for key in lora.keys():
        match = re.fullmatch(re.escape(ADAPTER_PREFIX) + r"(.*)\.lora_A\.weight", key)
        if not match:
            continue
        target = f"{match.group(1)}.weight"
        a = lora.get_tensor(key).astype(np.float32)
        b = lora.get_tensor(key.replace("lora_A", "lora_B")).astype(np.float32)
        weight = tensors[target]
        as_f32 = bf16_to_f32(weight) if dtypes[target] == "BF16" else weight.astype(np.float32)
        updated = as_f32 + scale * (b @ a)
        tensors[target] = f32_to_bf16(updated) if dtypes[target] == "BF16" else updated
        merged += 1
    print(f"merged {merged} LoRA projections at scale {scale}")

    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.name in {"model.safetensors", ".cache"} or item.name.startswith("."):
            continue
        if item.is_file():
            shutil.copy2(item, out / item.name)
    shutil.copytree(adapter / "1_Dense", out / "1_Dense", dirs_exist_ok=True)

    # numpy has no bf16, so bf16 tensors travel as raw uint16 and the
    # safetensors header is written by hand.
    header: dict[str, object] = {}
    offset = 0
    chunks = []
    for name in sorted(tensors):
        data = np.ascontiguousarray(tensors[name]).tobytes()
        header[name] = {
            "dtype": dtypes[name],
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        chunks.append(data)
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(out / "model.safetensors", "wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for chunk in chunks:
            handle.write(chunk)


def render(lines: list[str], path: Path) -> None:
    image = Image.new("RGB", (768, 1024), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=40)
    body = ImageFont.load_default(size=28)
    y = 60
    for index, line in enumerate(lines):
        draw.text((60, y), line, fill="black", font=title if index == 0 else body)
        y += 70 if index == 0 else 52
    image.save(path)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    base, adapter, out = (Path(arg) for arg in sys.argv[1:])
    merge(base, adapter, out)
    pages = out / "rerank_pages"
    pages.mkdir(exist_ok=True)
    render(
        [
            "Annual Report 2023",
            "Revenue by quarter (USD millions)",
            "Q1 2023        1,020",
            "Q2 2023        1,085",
            "Q3 2023        1,140",
            "Q4 2023        1,210",
            "Total revenue 2023   4,455",
            "Total revenue 2022   3,980",
            "Growth year over year   11.9%",
        ],
        pages / "matching.png",
    )
    render(
        [
            "Grandmother's Apple Pie",
            "Ingredients",
            "6 apples, peeled and sliced",
            "2 cups of flour",
            "1 cup of butter, chilled",
            "3/4 cup of sugar",
            "1 teaspoon of cinnamon",
            "Bake at 190 C for 50 minutes",
            "until the crust is golden.",
        ],
        pages / "unrelated.png",
    )
    print(f"wrote {out} and {pages}")


if __name__ == "__main__":
    main()
