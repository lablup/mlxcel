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
"""Generate the Clef sequence and parity fixture from mlx-vlm (#2365).

Out-of-band tooling: nothing in mlxcel's request path runs Python. Run it
under the GPU lock from the repository root:

    /usr/bin/lockf -k <gpu lock> env PYTHONPATH=references/mlx-vlm \
        uv run --python 3.12 --with mlx --with transformers --with numpy \
        --with pillow --with torch --with torchvision \
        python tests/fixtures/decision/generate_clef_fixture.py \
        models/clef-flash-mlx-8bit tests/fixtures/decision/clef_parity.json

The output pins upstream's ``Clef._sequence`` (token ids, question and option
spans, option labels, media grids) for a two-field schema with and without one
image and for a synthetic 48-frame video, and the full ``predict`` answers of
a text-only and an image request. ``src/decision/clef_tests.rs`` reads it.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mlx_vlm.decision import predict
from mlx_vlm.models.clef.clef import Clef
from mlx_vlm.utils import load

ROOT = Path(__file__).resolve().parents[3]
CATS = ROOT / "tests/fixtures/sam3/cats.png"

model_dir, out_path = sys.argv[1], sys.argv[2]
model, processor = load(model_dir)
clef = Clef(model, processor)
cats = Image.open(CATS).convert("RGB")

TICKET = "Please refund my duplicate charge"
department = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": ["billing", "technical", "sales"],
}
team = {
    "type": "choice",
    "instructions": "Which team owns the follow-up?",
    "criteria": {
        "technical": "Bugs and outages",
        "billing": "Charges and refunds",
        "account": None,
    },
}
urgency = {
    "type": "score",
    "instructions": "How urgent is this ticket?",
    "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"],
}
refund = {
    "type": "noul",
    "instructions": "Does the customer ask for money back?",
    "criteria": {"true": "A refund is requested"},
}
cats_count = {
    "type": "choice",
    "instructions": "How many cats are in the picture?",
    "criteria": ["one", "two", "three or more"],
}
remote = {"type": "bool", "instructions": "Is there a remote control in the picture?"}


def video_frames(count=48, width=320, height=256):
    """A deterministic synthetic clip: a square moving over a gradient."""
    frames = []
    for index in range(count):
        y, x = np.mgrid[0:height, 0:width]
        pixels = np.stack(
            [(x * 255 // width), (y * 255 // height), np.full_like(x, index * 5)], -1
        ).astype(np.uint8)
        left = (index * 5) % (width - 64)
        pixels[96:160, left : left + 64] = (255, 255, 255)
        frames.append(Image.fromarray(pixels))
    return frames


def sequence_case(name, state, questions, **media):
    normalized = {
        key: {**spec, "type": "noul" if spec["type"] == "bool" else spec["type"]}
        for key, spec in questions.items()
    }
    ids, rows, tensors = clef._sequence(state, normalized, **media)
    return {
        "name": name,
        "state": state,
        "questions": questions,
        "ids": [int(i) for i in ids],
        "rows": [
            {
                "name": row[0],
                "question_span": [int(row[2][0]), int(row[2][1])],
                "option_spans": [[int(s), int(e)] for s, e in row[3]],
                "labels": row[4],
            }
            for row in rows
        ],
        "image_grid_thw": np.asarray(tensors["image_grid_thw"]).tolist()
        if "image_grid_thw" in tensors
        else None,
        "video_grid_thw": np.asarray(tensors["video_grid_thw"]).tolist()
        if "video_grid_thw" in tensors
        else None,
    }


two_fields = {"team": team, "refund": refund}
sequences = [
    sequence_case("two_fields_text", TICKET, two_fields),
    sequence_case("two_fields_image", TICKET, two_fields, images=[cats]),
    sequence_case("two_fields_null_state_image", None, two_fields, images=[cats]),
    sequence_case("one_field_video", TICKET, {"department": department}, videos=[video_frames()]),
]

requests = [
    {
        "name": "ticket",
        "state": TICKET,
        "questions": {
            "department": department,
            "team": team,
            "urgency": urgency,
            "refund": refund,
        },
        "image": False,
    },
    {
        "name": "cats",
        "state": None,
        "questions": {"cats": cats_count, "remote": remote},
        "image": True,
    },
]
for request in requests:
    media = {"images": [cats]} if request["image"] else {}
    request["expected"] = predict(
        model, processor, request["state"], request["questions"], **media
    )

fixture = {
    "generator": "mlx-vlm d998a620 mlx_vlm.models.clef (Clef._sequence and mlx_vlm.decision.predict)",
    "checkpoint": "nativ-community/clef-flash-MLX-8bit",
    "image": "tests/fixtures/sam3/cats.png",
    "sequences": sequences,
    "requests": requests,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(json.dumps([r["expected"] for r in requests], ensure_ascii=False, indent=1))
