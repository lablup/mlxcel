#!/usr/bin/env python3
"""Require every tracked binary file to be declared, with a size budget.

Rationale
---------
Binary blobs enter a git repository silently. Nothing about ``git add`` warns
that a file will never delta-compress, that every later revision of it is stored
whole, and that removing it later does not shrink the history. The cost only
becomes visible as clone time, long after the commit that caused it.

That is not hypothetical here. ``webui/tests/screenshots/`` accumulated 24 PNG
baselines totalling ~1.9 MB across two platform directories without any review
step noticing, because each individual commit looked reasonable and no gate
looked at the aggregate. They were pixel baselines: renderer-, font- and
platform-specific, regenerated on a hosted runner and manually reviewed on every
font, Chromium or UI change. The browser suite's accessibility and geometry
assertions covered the same ground, so the images were dropped and the directory
is now ignored.

This check makes the next one impossible to add quietly.

The rule
--------
Every tracked file that git would treat as binary must match one ``Rule`` below.
A binary file matching nothing fails the job, and so does one that exceeds its
rule's ``max_bytes`` or pushes the declared set past ``TOTAL_BUDGET_BYTES``.

The list is inverted on purpose, the same way ``check_crate_versions.py``
inverts the crate list: a *new* binary path is the failure mode, so the default
for an undeclared path has to be "fail", not "ignore". An allowlist of paths to
*skip* would have stayed silent on exactly the directory that prompted this.

Adding a binary asset is therefore a deliberate act: add a ``Rule`` with a
reason, and the reason is what review reads. Generated artifacts, profiling
captures and test output belong in ``.gitignore`` instead.

Text bloat is out of scope. Large text files delta-compress and stay diffable,
so they do not carry the same one-way cost, and a ceiling on them would fire on
the generated WebUI bundle and the ``.mlir`` fixtures without telling anyone
anything they did not already know.

Usage
-----
    scripts/ci/check_binary_assets.py

Exits non-zero and names every undeclared or oversize file.
"""
import dataclasses
import fnmatch
import pathlib
import subprocess
import sys

# git's own binary heuristic: a NUL byte anywhere in the first 8000 bytes.
# See `buffer_is_binary()` in git's xdiff-interface.c.
BINARY_SNIFF_BYTES = 8000

# Ceiling on the declared set as a whole, so growth inside an existing rule is
# also a review event rather than a slow drift.
# Retain the original allowance plus exact DINOv2, Sapiens2, SAM 3, Video
# Depth Anything, SAM 3.1, Objects, PP-DocLayoutV3, SAM 3D Body,
# RF-DETR, FLUX.2, Z-Image, Bonsai, and ERNIE-Image fixture bytes.
TOTAL_BUDGET_BYTES = 41_690_560


@dataclasses.dataclass(frozen=True)
class Rule:
    """One declared binary asset path, its per-file ceiling, and why it exists."""

    pattern: str
    max_bytes: int
    why: str


DECLARED = (
    Rule(
        pattern="benchmarks/epic-2574/raw/vlm/qwen35/baseline/input-image.png",
        max_bytes=679,
        why="Pinned Qwen3.5-VL benchmark input image for hash-verifiable request replay.",
    ),
    Rule(
        pattern="benchmarks/epic-2574/raw/vlm/qwen35/candidate/input-image.png",
        max_bytes=679,
        why="Pinned Qwen3.5-VL benchmark input image for hash-verifiable request replay.",
    ),
    Rule(
        pattern="tests/fixtures/bonsai/generate_512_seed9909.safetensors",
        max_bytes=524_480,
        why="Pinned Bonsai FLUX.2 generation latent trajectory for full-model parity.",
    ),
    Rule(
        pattern="tests/fixtures/bonsai/transformer_step0_f32.safetensors",
        max_bytes=557_232,
        why="Pinned upstream Bonsai FLUX.2 first-step F32 transformer output.",
    ),
    Rule(
        pattern="tests/fixtures/bonsai/generate_512_seed9909.png",
        max_bytes=370_849,
        why="Pinned decoded Bonsai generation output compared pixelwise in parity tests.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/edit_512_seed123.safetensors",
        max_bytes=524_480,
        why="Pinned FLUX.2 edit latent trajectory for upstream full-model parity.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/generate_512_seed9909.safetensors",
        max_bytes=524_480,
        why="Pinned FLUX.2 generation latent trajectory for upstream full-model parity.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/transformer_step0_f32.safetensors",
        max_bytes=557_232,
        why="Pinned upstream FLUX.2 first-step F32 transformer output.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/edit_512_seed123.png",
        max_bytes=338_348,
        why="Pinned decoded FLUX.2 edit output compared pixelwise by parity tests.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/edit_reference.png",
        max_bytes=348_745,
        why="Fixed upstream FLUX.2 edit input preserved at native 512-pixel resolution.",
    ),
    Rule(
        pattern="tests/fixtures/flux2/generate_512_seed9909.png",
        max_bytes=340_928,
        why="Pinned decoded FLUX.2 generation output compared pixelwise by parity tests.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/z_image_parity.safetensors",
        max_bytes=198_396,
        why="Compact upstream Z-Image full-model numerical parity references.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/ernie_image_parity.safetensors",
        max_bytes=247_627,
        why="Compact upstream ERNIE-Image full-model numerical parity references.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/z_image_tiny_refs.safetensors",
        max_bytes=1_638,
        why="Tiny upstream Z-Image transformer normalization and RoPE references.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/base/input.safetensors",
        max_bytes=3_763_296,
        why="Exact authenticated RF-DETR Base processor output for full-model parity.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/base/raw.safetensors",
        max_bytes=114_160,
        why="Authentic RF-DETR Base F32 logits and boxes from the released checkpoint.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/input.safetensors",
        max_bytes=1_769_568,
        why="Exact authenticated RF-DETR Seg Small processor output for full-model parity.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/raw.safetensors",
        max_bytes=3_724_640,
        why="Authentic RF-DETR Seg Small F32 logits, boxes, and mask logits.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/mask-000.png",
        max_bytes=2_898,
        why="Source-resolution upstream RF-DETR Seg Small mask for selected detection 0.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/mask-001.png",
        max_bytes=646,
        why="Source-resolution upstream RF-DETR Seg Small mask for selected detection 1.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/mask-002.png",
        max_bytes=2_527,
        why="Source-resolution upstream RF-DETR Seg Small mask for selected detection 2.",
    ),
    Rule(
        pattern="tests/fixtures/rfdetr/seg/mask-003.png",
        max_bytes=776,
        why="Source-resolution upstream RF-DETR Seg Small mask for selected detection 3.",
    ),
    Rule(
        pattern="tests/fixtures/pp_doclayout_v3/input.safetensors",
        max_bytes=13_697_116,
        why="Exact decoded RGB and Torchvision uint8-bicubic PP-DocLayoutV3 processor input.",
    ),
    Rule(
        pattern="tests/fixtures/pp_doclayout_v3/paddle_raw.safetensors",
        max_bytes=395_168,
        why="Authentic raw F32 outputs from the released Paddle PP-DocLayoutV3 checkpoint.",
    ),
    Rule(
        pattern="tests/fixtures/pp_doclayout_v3/hashnuke_raw.safetensors",
        max_bytes=395_168,
        why="Authentic raw F32 outputs from the released converted PP-DocLayoutV3 checkpoint.",
    ),
    Rule(
        pattern="tests/fixtures/pp_doclayout_v3/two_column_page.png",
        max_bytes=590_803,
        why="Licensed ACL 2019 BERT page used for PP reading-order and Falcon-OCR integration.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_body/reference-boxed.safetensors",
        max_bytes=666_588,
        why="Authentic complete SAM 3D Body boxed mesh, joints, camera and topology reference.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_body/reference-default.safetensors",
        max_bytes=666_588,
        why="Authentic complete SAM 3D Body default mesh, joints, camera and topology reference.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_body/geometry.safetensors",
        max_bytes=6_632,
        why="Authentic tiny CPU rotation, FK and ordered scatter outputs from pinned SAM 3D Body.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_body/crop-tiny.safetensors",
        max_bytes=2_824,
        why="Authentic OpenCV full and partial crop outputs from a synthetic tiny RGB image.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/focal-solver-states-cpu.safetensors",
        max_bytes=1_035,
        why="Authentic CPU control for backend-specific compiled focal solver rounding, using the same sampled UV/XYZ inputs.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/focal-solver-inputs.safetensors",
        max_bytes=82_084,
        why="Exact sampled UV/XYZ from authenticated MoGe affine output for the Objects compiled focal solver oracle.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/focal-solver-states.safetensors",
        max_bytes=1_035,
        why="Pinned upstream focal solver initial and thirty compiled iteration states plus final focal/shift.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/gelu-tanh-power-cpu.safetensors",
        max_bytes=197_600,
        why="Authentic CPU control for backend-specific F32 tanh GELU rounding; regenerated by generate_gelu_tanh_power_oracle.py --device cpu.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/gelu-tanh-power.safetensors",
        max_bytes=197_600,
        why="Pinned exact-native BF16/F32 Objects tanh GELU Power graph oracle; regenerated by generate_gelu_tanh_power_oracle.py.",
    ),
    Rule(
        pattern="tests/fixtures/sam3d_objects/layers.safetensors",
        max_bytes=67_631,
        why="Pinned exact-native BF16 Objects graph helper and GELU reference literals; regenerated by generate_helper_oracles.py.",
    ),
    Rule(
        pattern="tests/fixtures/dinov2/small_reference.safetensors",
        max_bytes=1_392_288,
        why="Complete pinned HF DINOv2 small CPU f32 reference outputs and deterministic input; regenerated by generate_reference.py.",
    ),
    Rule(
        pattern="tests/fixtures/dinov2/register_small_reference.safetensors",
        max_bytes=1_398_448,
        why="Complete pinned HF DINOv2 register-small CPU f32 reference outputs and deterministic input; regenerated by generate_reference.py.",
    ),
    Rule(
        pattern="tests/fixtures/video_depth_anything/davis_64.mp4",
        max_bytes=69_065,
        why="Pinned 64-frame DAVIS car-turn clip for consecutive-frame Video Depth Anything parity and CLI validation.",
    ),
    Rule(
        pattern="tests/fixtures/video_depth_anything/davis_40_frames.safetensors",
        max_bytes=1_474_656,
        why="Exact decoded uint8 RGB frames shared by the authentic Video Depth Anything CPU references.",
    ),
    Rule(
        pattern="tests/fixtures/video_depth_anything/small_relative_40.safetensors",
        max_bytes=1_966_288,
        why="Complete pinned upstream Small relative Video Depth Anything CPU float32 output for 40 frames.",
    ),
    Rule(
        pattern="tests/fixtures/video_depth_anything/small_metric_40.safetensors",
        max_bytes=1_966_288,
        why="Complete pinned upstream Small metric Video Depth Anything CPU float32 output for 40 frames.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/cats.png",
        max_bytes=672 * 1024,
        why=(
            "Losslessly recompressed COCO two-cat image used for real-checkpoint "
            "SAM 3 image and box-prompt parity."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sam3/car-clip.mp4",
        max_bytes=192 * 1024,
        why=(
            "Three-frame DAVIS car clip used for real-checkpoint SAM 3 memory "
            "tracking parity; codec-decoded pixels are part of the fixture."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sam3/text-reference.safetensors",
        max_bytes=84 * 1024,
        why=(
            "Pinned mlx-vlm token IDs, attention mask, CLIP hidden states and "
            "projected tokens for stage-by-stage SAM 3 checkpoint parity."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sam3/pixel-reference.safetensors",
        max_bytes=24 * 1024,
        why=(
            "Sampled pixels from the pinned Pillow bilinear SAM 3 processor, "
            "covering resize and normalization before model inference."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sam3/cats-mask-0.png",
        max_bytes=2 * 1024,
        why="One-bit upstream SAM 3 mask for the right cat.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/cats-mask-1.png",
        max_bytes=2 * 1024,
        why="One-bit upstream SAM 3 mask for the left cat.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/box-mask-0.png",
        max_bytes=2 * 1024,
        why="One-bit upstream SAM 3 box-prompt mask for the right cat.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/box-mask-1.png",
        max_bytes=5 * 1024,
        why="One-bit upstream SAM 3 box-prompt mask for the left cat.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-native-mask-0.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 native-resolution tracking mask, frame 0.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-native-mask-1.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 native-resolution tracking mask, frame 1.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-native-mask-2.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 native-resolution tracking mask, frame 2.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-source-mask-0.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 source-resolution tracking mask, frame 0.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-source-mask-1.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 source-resolution tracking mask, frame 1.",
    ),
    Rule(
        pattern="tests/fixtures/sam3/video-source-mask-2.png",
        max_bytes=1024,
        why="One-bit pinned upstream SAM 3 source-resolution tracking mask, frame 2.",
    ),
    Rule(
        pattern="tests/fixtures/sam3_1/three-cars.mp4",
        max_bytes=48 * 1024,
        why=(
            "Three consecutive frames from the genuine OpenGeos traffic video, "
            "used for nine-object SAM 3.1 multiplex tracking parity."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sam3_1/*.png",
        max_bytes=2 * 1024,
        why=(
            "One-bit upstream SAM 3.1 cat and nine-object traffic masks. Exact "
            "checkpoint, source-video, and runtime provenance is stored beside them."
        ),
    ),
    Rule(
        pattern="tests/fixtures/imagegen/dit_blocks_refs.safetensors",
        max_bytes=6 * 1024,
        why="Tiny upstream DiT numerical references; generated by generate_imagegen_fixtures.py.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/qwen3_encoder_refs.safetensors",
        max_bytes=10 * 1024,
        why="Tiny upstream Qwen3 hidden-state references; generated by generate_imagegen_fixtures.py.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/vae_ernie_refs.safetensors",
        max_bytes=170 * 1024,
        why="Fixed-latent upstream VAE outputs for numerical parity; generated by generate_imagegen_fixtures.py.",
    ),
    Rule(
        pattern="tests/fixtures/imagegen/vae_tiny.safetensors",
        max_bytes=156 * 1024,
        why="Seeded tiny VAE weights and reference outputs; generated by generate_imagegen_fixtures.py.",
    ),
    Rule(
        pattern="tests/fixtures/sapiens2/person.jpg",
        max_bytes=50_427,
        why=(
            "Fixed upstream Sapiens2 pose and dense-prediction parity image. "
            "Its source URL and SHA-256 are pinned beside the numerical fixtures."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sapiens2/person.png",
        max_bytes=766_288,
        why=(
            "Lossless canonical RGB input for Sapiens2 numerical parity. It is "
            "the Pillow-decoded form of the pinned JPEG and removes JPEG decoder "
            "rounding differences from cross-runtime comparisons."
        ),
    ),
    Rule(
        pattern="tests/fixtures/sapiens2/seg-bf16.png",
        max_bytes=8_481,
        why="Pinned Sapiens2 BF16 segmentation reference with provenance beside the fixture.",
    ),
    Rule(
        pattern="tests/fixtures/sapiens2/seg-a7d-rocm-bf16.png",
        max_bytes=8_487,
        why="Pinned Sapiens2 BF16 segmentation reference with provenance beside the fixture.",
    ),
    Rule(
        pattern="tests/fixtures/sapiens2/seg-rocm-bf16.png",
        max_bytes=8_510,
        why="Pinned Sapiens2 BF16 segmentation reference with provenance beside the fixture.",
    ),
    Rule(
        pattern="tests/fixtures/*.png",
        max_bytes=32 * 1024,
        why=(
            "Synthetic images for the vision preprocessing tests: a solid probe, "
            "two aspect-ratio shape fixtures and one page of rendered text for "
            "the GOT-OCR path. Checked in because the expected patch grids are "
            "asserted against exact pixel content."
        ),
    ),
    Rule(
        pattern="tests/fixtures/kimi_k3_vision/*.png",
        max_bytes=32 * 1024,
        why=(
            "NaViT dynamic-resolution probe for the Kimi K3 vision tower. Its "
            "dimensions pick the tile grid under test, so it cannot be generated "
            "at an arbitrary size at run time."
        ),
    ),
    Rule(
        pattern="tests/fixtures/audio/tone.*",
        max_bytes=32 * 1024,
        why=(
            "One second of a pure tone in WAV, MP3 and FLAC. The three encodings "
            "are the point: they cover the decoder branches behind /v1/audio, "
            "and the MP3 and FLAC framing cannot be synthesized in the test."
        ),
    ),
    Rule(
        pattern="docs/benchmark_results/data/*.tar.gz",
        max_bytes=32 * 1024,
        why=(
            "Raw per-run logs behind a committed benchmark report, archived "
            "because nobody reads them in normal review; the report's tables "
            "and CSVs carry the numbers, and the archive is there to audit a "
            "run. Keep each under 32 KB compressed; larger log sets belong "
            "outside the repository."
        ),
    ),
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def tracked_files(root: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [path for path in out.split("\0") if path]


def is_binary(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False  # A submodule or broken symlink is not ours to judge.


def main() -> int:
    root = repo_root()
    undeclared: list[str] = []
    oversize: list[str] = []
    declared_bytes = 0
    declared_count = 0

    for rel in sorted(tracked_files(root)):
        path = root / rel
        if not path.is_file() or path.is_symlink() or not is_binary(path):
            continue

        size = path.stat().st_size
        rule = next(
            (r for r in DECLARED if fnmatch.fnmatch(rel, r.pattern)),
            None,
        )
        if rule is None:
            undeclared.append(f"{rel} ({size / 1024:.1f} KB)")
            continue

        declared_count += 1
        declared_bytes += size
        if size > rule.max_bytes:
            oversize.append(
                f"{rel} is {size / 1024:.1f} KB, over the "
                f"{rule.max_bytes / 1024:.0f} KB ceiling on '{rule.pattern}'"
            )

    over_budget = declared_bytes > TOTAL_BUDGET_BYTES

    if undeclared or oversize or over_budget:
        print("binary-assets: FAIL")
        for entry in undeclared:
            print(f"  undeclared: {entry}")
        for entry in oversize:
            print(f"  oversize:   {entry}")
        if over_budget:
            print(
                f"  budget:     declared binaries total "
                f"{declared_bytes / 1024:.1f} KB, over the "
                f"{TOTAL_BUDGET_BYTES / 1024:.0f} KB ceiling"
            )
        print()
        if undeclared:
            print(
                "A tracked binary file has to be declared in DECLARED in\n"
                "scripts/ci/check_binary_assets.py, with a per-file ceiling and a\n"
                "reason review can read. Binary blobs never delta-compress and\n"
                "deleting one later does not shrink the history, so adding one is\n"
                "a decision, not a detail.\n"
                "\n"
                "If the file is generated output, a profiling capture or test\n"
                "debris, add it to .gitignore and `git rm --cached` it instead."
            )
        if oversize or over_budget:
            print(
                "A ceiling was raised by the file, not by a reviewer. Confirm the\n"
                "asset still needs to be this large, then raise the number in the\n"
                "rule (or TOTAL_BUDGET_BYTES) in the same commit."
            )
        return 1

    print(
        f"binary-assets: OK — {declared_count} declared binary files, "
        f"{declared_bytes / 1024:.1f} KB of "
        f"{TOTAL_BUDGET_BYTES / 1024:.0f} KB budget."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
