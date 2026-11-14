// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use image::metadata::Orientation;
use image::{DynamicImage, GenericImageView, ImageFormat, RgbImage};

use super::test_pictures::{
    colour_blocks, distinct_pixels, jpeg_with_orientation, png_with_orientation, stored_for,
    webp_with_orientation,
};
use super::{
    exif_orientation, load_upright_from_memory, open_upright, swaps_dimensions, turn_upright,
    upright_dimensions,
};

const TAGS: std::ops::RangeInclusive<u8> = 1..=8;

fn plain_png(picture: &RgbImage) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(picture.clone())
        .write_to(&mut out, ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

#[test]
fn every_exif_orientation_comes_back_upright_pixel_for_pixel() {
    let upright = distinct_pixels();
    for tag in TAGS {
        let stored = stored_for(tag, &upright);
        let bytes = png_with_orientation(&stored, u16::from(tag));

        // What the `image` crate hands back on its own: the stored pixels.
        let untouched = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(untouched, stored, "tag {tag}: the decode alone");

        let loaded = load_upright_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(loaded.dimensions(), upright.dimensions(), "tag {tag}");
        assert_eq!(loaded, upright, "tag {tag}");
    }
}

#[test]
fn a_webp_exif_chunk_is_read_as_well() {
    let upright = distinct_pixels();
    for tag in TAGS {
        let stored = stored_for(tag, &upright);
        let bytes = webp_with_orientation(&stored, u16::from(tag));
        assert_eq!(
            exif_orientation(&bytes),
            Orientation::from_exif(tag).unwrap(),
            "tag {tag}"
        );
        assert_eq!(
            image::load_from_memory(&bytes).unwrap().to_rgb8(),
            stored,
            "tag {tag}: the decode alone"
        );
        assert_eq!(
            load_upright_from_memory(&bytes).unwrap().to_rgb8(),
            upright,
            "tag {tag}"
        );
    }
}

#[test]
fn a_jpeg_app1_orientation_is_read_as_well() {
    let upright = colour_blocks();
    for tag in TAGS {
        let stored = stored_for(tag, &upright);
        let bytes = jpeg_with_orientation(&stored, u16::from(tag));
        assert_eq!(
            exif_orientation(&bytes),
            Orientation::from_exif(tag).unwrap(),
            "tag {tag}"
        );

        let loaded = load_upright_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(loaded.dimensions(), (48, 32), "tag {tag}");
        // JPEG is lossy: the middle of each 16 x 16 block, within a tolerance.
        for block_y in 0..2 {
            for block_x in 0..3 {
                let (x, y) = (block_x * 16 + 8, block_y * 16 + 8);
                let (got, want) = (loaded.get_pixel(x, y).0, upright.get_pixel(x, y).0);
                for channel in 0..3 {
                    assert!(
                        got[channel].abs_diff(want[channel]) <= 24,
                        "tag {tag}, block ({block_x}, {block_y}): {got:?} for {want:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_picture_without_a_readable_tag_is_left_as_it_is() {
    let picture = distinct_pixels();
    let plain = plain_png(&picture);
    assert_eq!(exif_orientation(&plain), Orientation::NoTransforms);
    assert_eq!(load_upright_from_memory(&plain).unwrap().to_rgb8(), picture);

    // Values the tag does not define.
    for tag in [0u16, 9, 600] {
        let bytes = png_with_orientation(&picture, tag);
        assert_eq!(
            exif_orientation(&bytes),
            Orientation::NoTransforms,
            "tag {tag}"
        );
        assert_eq!(load_upright_from_memory(&bytes).unwrap().to_rgb8(), picture);
    }

    // Not a picture at all: no orientation, and the decode still reports it.
    assert_eq!(
        exif_orientation(b"not a picture"),
        Orientation::NoTransforms
    );
    assert_eq!(exif_orientation(&[]), Orientation::NoTransforms);
    assert!(load_upright_from_memory(b"not a picture").is_err());
}

#[test]
fn only_the_quarter_turns_swap_the_dimensions() {
    let upright = distinct_pixels();
    for tag in TAGS {
        let orientation = Orientation::from_exif(tag).unwrap();
        assert_eq!(swaps_dimensions(orientation), tag >= 5, "tag {tag}");

        let stored = stored_for(tag, &upright);
        let bytes = png_with_orientation(&stored, u16::from(tag));
        // The header says what is stored; the model is shown the upright size.
        let header = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        assert_eq!(header, stored.dimensions(), "tag {tag}");
        assert_eq!(upright_dimensions(&bytes, header), (6, 4), "tag {tag}");
    }
}

#[test]
fn turn_upright_changes_a_decoded_picture_in_place() {
    let upright = distinct_pixels();
    let stored = stored_for(6, &upright);
    let bytes = png_with_orientation(&stored, 6);
    let mut image = image::load_from_memory(&bytes).unwrap();
    assert_eq!(image.dimensions(), (4, 6));
    turn_upright(&mut image, &bytes);
    assert_eq!(image.to_rgb8(), upright);
}

#[test]
fn a_file_is_opened_upright() {
    let upright = distinct_pixels();
    let dir = tempfile::tempdir().unwrap();
    for tag in [1u8, 3, 6, 8] {
        let stored = stored_for(tag, &upright);
        let path = dir.path().join(format!("tag{tag}.png"));
        std::fs::write(&path, png_with_orientation(&stored, u16::from(tag))).unwrap();
        assert_eq!(open_upright(&path).unwrap().to_rgb8(), upright, "tag {tag}");
    }
    assert!(open_upright(&dir.path().join("missing.png")).is_err());
}
