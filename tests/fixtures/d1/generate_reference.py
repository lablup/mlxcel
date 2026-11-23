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
"""Regenerate the LiquidAI d1-3B decision reference (`reference.json`).

Out-of-band tooling for issue #2367: nothing in mlxcel's request path runs
Python. mlx-vlm's own `D1Model` (`mlx_vlm/models/d1`) answers each case on the
checkpoint, and the fixture records its answers, its trunk and branch token
ids, and its usage, so the Rust gates in `src/decision/d1_tests.rs` compare
both the prompt and the probabilities.

Cases:

- `text_all`: the README text example (refund, team, urgency) asked together,
  which runs the trunk/branch path.
- `text_refund`, `text_team`, `text_urgency`: each question asked alone, the
  single-pass path.
- `cats`: the README COCO image example (`tests/fixtures/sam3/cats.png`, the
  pixels of COCO val2017 000000039769) with a null state.
- `cats_2x`: the same image upscaled 2x by pixel replication (1280 x 960,
  over the 1024 x 1024 area cap), so d1's bicubic pre-downscale runs ahead of
  the LFM2-VL processor's own resize. Pixel replication is exact, so the Rust
  gate rebuilds the identical input.
- `cats_multi`: the image with two questions, so the trunk with expanded
  image tokens is tiled across branches.
- `options_27`: a 27-label choice, the two-digit code branch.
- `bool_criteria`: a bool question with `true` / `false` descriptions.

Every case also records the token ids each option is read from
(`readout_ids`).

Usage (from the repository root, in a venv with mlx-vlm installed):

    python -I tests/fixtures/d1/generate_reference.py models/d1-3b-mlx-8bit \
        > tests/fixtures/d1/reference.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

from mlx_vlm import load

ROOT = Path(__file__).resolve().parents[3]
CATS = ROOT / "tests/fixtures/sam3/cats.png"

STATE = "I was charged twice, please refund one."
REFUND = {"type": "bool", "instructions": "Is the customer asking for a refund?"}
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges, refunds", "technical": "App faults"},
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["Can wait", "Today", "Blocking the customer now"],
}
CATS_Q = {
    "type": "choice",
    "instructions": "How many cats are there?",
    "criteria": ["one", "two", "more"],
}
BOOL_CRITERIA = {
    "type": "bool",
    "instructions": "Does this need a reply today?",
    "criteria": {"true": "the customer is blocked", "false": "it can wait"},
}
CATS_INDOORS = {
    "type": "bool",
    "instructions": "Is this photo taken indoors?",
}
OPTIONS_27 = {
    "type": "choice",
    "instructions": "Which word comes first in the alphabet?",
    "criteria": [f"word_{i}" for i in range(27)],
}


def doubled(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"))
    return Image.fromarray(pixels.repeat(2, axis=0).repeat(2, axis=1))


def run(model, processor, name, state, questions, images=None):
    questions = dict(questions)
    from mlx_vlm.models.d1.prompt import as_question, readout_ids

    normalized = [as_question(q) for q in questions.values()]
    tokenizer = getattr(processor, "tokenizer", processor)
    trunk, branches, _ = model.decision_inputs(processor, state, normalized, images)
    result = model.predict(processor, state, questions, images=images)
    result.pop("model", None)
    return {
        "name": name,
        "state": state,
        "questions": questions,
        "trunk_ids": trunk,
        "branch_ids": branches,
        "readout_ids": [readout_ids(tokenizer, q) for q in normalized],
        "result": result,
    }


def main() -> None:
    model_dir = sys.argv[1]
    model, processor = load(model_dir)
    cats = Image.open(CATS).convert("RGB")
    cases = [
        run(
            model,
            processor,
            "text_all",
            STATE,
            {"refund": REFUND, "team": TEAM, "urgency": URGENCY},
        ),
        run(model, processor, "text_refund", STATE, {"refund": REFUND}),
        run(model, processor, "text_team", STATE, {"team": TEAM}),
        run(model, processor, "text_urgency", STATE, {"urgency": URGENCY}),
        run(model, processor, "cats", None, {"cats": CATS_Q}, images=[cats]),
        run(model, processor, "cats_2x", None, {"cats": CATS_Q}, images=[doubled(cats)]),
        run(
            model,
            processor,
            "cats_multi",
            None,
            {"cats": CATS_Q, "indoors": CATS_INDOORS},
            images=[cats],
        ),
        run(model, processor, "options_27", STATE, {"first": OPTIONS_27}),
        run(model, processor, "bool_criteria", STATE, {"today": BOOL_CRITERIA}),
    ]
    revision = subprocess.run(
        ["git", "-C", str(ROOT / "references/mlx-vlm"), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    out = {
        "checkpoint": "LiquidAI/d1-3B-MLX-8bit",
        "mlx_vlm": metadata.version("mlx-vlm"),
        "mlx_vlm_revision": revision,
        "mlx": metadata.version("mlx"),
        "cases": cases,
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
