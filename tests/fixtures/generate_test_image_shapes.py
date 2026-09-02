#!/usr/bin/env python3
"""Generate the three-shape VLM fixture with Pillow 12.3.0.

The repository's other image fixture, ``test_image.png``, is a solid orange
square: a VLM test built on it can only catch an empty or non-finite
generation, never a description that quietly drops an object. This image
carries three unambiguous objects of three unambiguous colors, so a parity test
can assert that every one of them is named.

The default 448x448 size is the one issue #1596 was measured at. It is a
multiple of the Qwen2-VL patch factor (28), so ``smart_resize`` leaves it
untouched and the processor stage introduces no resampling of its own. It also
gives a 16x16 merged grid, where the vision tower's window permutation is its
own inverse.

Pass a second argument to render the same scene at another size. 336 is the
useful one: it gives a 12x12 merged grid whose window permutation is not an
involution, which is what separates a correct window un-reorder from one that
merely returns the permutation it was given.

Usage:
    python3 tests/fixtures/generate_test_image_shapes.py <out.png> [size]
"""

import sys

from PIL import Image, ImageDraw

PILLOW_VERSION = "12.3.0"
BASE_SIZE = 448
BACKGROUND = (245, 245, 245)
SQUARE_BOX = (40, 60, 200, 220)
SQUARE_FILL = (220, 30, 30)
CIRCLE_BOX = (250, 80, 410, 240)
CIRCLE_FILL = (30, 60, 220)
TRIANGLE_POINTS = ((90, 400), (230, 260), (370, 400))
TRIANGLE_FILL = (30, 170, 60)


def scaled(values: tuple[int, ...], size: int) -> list[int]:
    return [round(v * size / BASE_SIZE) for v in values]


def render(size: int = BASE_SIZE) -> Image.Image:
    img = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.rectangle(scaled(SQUARE_BOX, size), fill=SQUARE_FILL)
    draw.ellipse(scaled(CIRCLE_BOX, size), fill=CIRCLE_FILL)
    draw.polygon(
        [tuple(scaled(point, size)) for point in TRIANGLE_POINTS],
        fill=TRIANGLE_FILL,
    )
    return img


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    size = int(argv[2]) if len(argv) == 3 else BASE_SIZE
    render(size).save(argv[1], format="PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
