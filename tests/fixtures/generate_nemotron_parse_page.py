#!/usr/bin/env python3
"""Generate the Nemotron-Parse parity fixture page.

A rendered text page with a title and a paragraph gives a known answer: the
model has to emit the title and the text lines with box tokens and class tags,
so a broken tower, neck, or decoder produces visibly wrong output rather than
plausible markdown.

The page is 1240x1754, inside Nemotron-Parse's 1664x2048 box, so the
reference processor only pads it and never resamples it. That keeps the parity
comparison in ``tests/nemotron_parse_real_model.rs`` about the model, not about
PIL-versus-``image`` bilinear taps.

The fixture is committed, so this script documents how it was made rather than
being run by the tests. Regenerating it changes the reference token ids pinned
in the parity test; re-derive them with the checkpoint's own
``hf_nemotron_parse_modeling.py`` on CPU before updating the fixture.

Rendered with Pillow 12 and macOS Arial / Arial Bold.

Usage:
    python3 tests/fixtures/generate_nemotron_parse_page.py <out.png>
"""

import sys

from PIL import Image, ImageDraw, ImageFont

BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
LINES = [
    "This page checks the mlxcel port of Nemotron-Parse.",
    "The encoder is a C-RADIO ViT-H tower with a neck,",
    "and the decoder is a pre-norm mBART stack.",
    "",
    "Invoice total: 1,234.56 USD",
]


def main() -> None:
    img = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 150), "Hello Nemotron", font=ImageFont.truetype(BOLD, 56), fill="black")
    body = ImageFont.truetype(REGULAR, 32)
    y = 300
    for line in LINES:
        draw.text((100, y), line, font=body, fill="black")
        y += 60
    img.save(sys.argv[1])


if __name__ == "__main__":
    main()
