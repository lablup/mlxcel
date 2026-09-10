#!/usr/bin/env python3
"""Generate the GOT-OCR 2.0 parity fixture.

A rendered text page is what an OCR parity test needs: unlike a colored shape,
the correct answer is a string known in advance, so a run that drops the vision
features or mis-orders the 256 feature rows produces visibly wrong text rather
than a plausible caption.

The wording is deliberately ASCII, single-case-mixed, and free of anything a
language model could guess from context, so a transcription that matches did so
by reading the page.

The fixture is committed, so this script documents how it was made rather than
being run by the tests. Regenerating it changes the reference token ids pinned
in ``tests/got_ocr_real_model.rs``; re-derive them with the checkpoint's own
``got_vision_b.py`` before updating the fixture.

Rendered with Pillow 10.2.0 and DejaVuSans-Bold 2.37 (Debian
``fonts-dejavu-core``).

Usage:
    python3 tests/fixtures/generate_got_ocr_page.py <out.png>
"""

import sys

from PIL import Image, ImageDraw, ImageFont

PILLOW_VERSION = "10.2.0"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SIZE = (1024, 512)
LINES = ["GOT OCR two point zero", "renders this page"]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = sys.argv[1]

    image = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 48)
    for index, line in enumerate(LINES):
        draw.text((80, 160 + index * 100), line, fill="black", font=font)
    image.save(out, optimize=True)
    print(f"wrote {out} {image.size}")


if __name__ == "__main__":
    main()
