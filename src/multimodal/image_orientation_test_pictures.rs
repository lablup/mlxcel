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

//! Pictures with an EXIF orientation tag, built in memory for the tests.
//!
//! No binary fixture: the stored pixels are computed from the upright picture
//! with the tag's own definition, and the tag is written into the container by
//! hand (a JPEG `APP1` segment, a PNG `eXIf` chunk).
//!
//! Used by: `image_orientation_tests.rs` and `server::model_worker_tests`.

use std::io::Cursor;

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

/// The pixels a camera stores for `upright` under EXIF orientation `tag`.
///
/// Written from the tag's definition (EXIF 2.32, tag 0x0112), the table
/// `PIL.ImageOps.exif_transpose` implements, and not from the `image` crate's
/// transforms: stored pixel `(x, y)` is the one a viewer shows at `(u, v)`.
pub(crate) fn stored_for(tag: u8, upright: &RgbImage) -> RgbImage {
    let (width, height) = upright.dimensions();
    let (stored_width, stored_height) = if (5..=8).contains(&tag) {
        (height, width)
    } else {
        (width, height)
    };
    let (last_x, last_y) = (stored_width - 1, stored_height - 1);
    RgbImage::from_fn(stored_width, stored_height, |x, y| {
        let (u, v) = match tag {
            1 => (x, y),
            // Mirrored left to right.
            2 => (last_x - x, y),
            // Turned by a half.
            3 => (last_x - x, last_y - y),
            // Mirrored top to bottom.
            4 => (x, last_y - y),
            // Mirrored along the main diagonal.
            5 => (y, x),
            // The viewer turns it a quarter clockwise.
            6 => (last_y - y, x),
            // Mirrored along the other diagonal.
            7 => (last_y - y, last_x - x),
            // The viewer turns it a quarter counterclockwise.
            8 => (y, last_x - x),
            other => panic!("EXIF orientation is 1 to 8, got {other}"),
        };
        *upright.get_pixel(u, v)
    })
}

/// A little-endian TIFF block whose only entry is `Orientation = tag`.
fn exif_block(tag: u16) -> Vec<u8> {
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II\x2a\x00");
    tiff.extend_from_slice(&8u32.to_le_bytes()); // offset of IFD0
    tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&tag.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]); // value padding
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    tiff
}

fn encoded(picture: &RgbImage, format: ImageFormat) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(picture.clone())
        .write_to(&mut out, format)
        .expect("encode a test picture");
    out.into_inner()
}

/// `stored` as a PNG, lossless, with an `eXIf` chunk that says `tag`.
pub(crate) fn png_with_orientation(stored: &RgbImage, tag: u16) -> Vec<u8> {
    let png = encoded(stored, ImageFormat::Png);
    // Signature (8 bytes) and IHDR (4 length + 4 type + 13 data + 4 CRC).
    let after_header = 8 + 25;
    assert_eq!(&png[12..16], b"IHDR");
    let mut chunk = b"eXIf".to_vec();
    chunk.extend_from_slice(&exif_block(tag));
    let mut out = png[..after_header].to_vec();
    out.extend_from_slice(&((chunk.len() - 4) as u32).to_be_bytes());
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&crc32(&chunk).to_be_bytes());
    out.extend_from_slice(&png[after_header..]);
    out
}

/// `stored` as a JPEG with an `APP1` EXIF segment that says `tag`. Lossy:
/// compare with a tolerance.
pub(crate) fn jpeg_with_orientation(stored: &RgbImage, tag: u16) -> Vec<u8> {
    let jpeg = encoded(stored, ImageFormat::Jpeg);
    assert_eq!(&jpeg[..2], [0xff, 0xd8], "a JPEG starts with SOI");
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(&exif_block(tag));
    let mut out = vec![0xff, 0xd8, 0xff, 0xe1];
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// CRC-32 as PNG chunks use it (ISO 3309), bit by bit.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A 6 x 4 picture in which every pixel is different, so a wrong transform
/// cannot pass.
pub(crate) fn distinct_pixels() -> RgbImage {
    RgbImage::from_fn(6, 4, |x, y| {
        Rgb([40 * x as u8 + 10, 60 * y as u8 + 5, (4 * x + y) as u8])
    })
}

/// A 48 x 32 picture of six 16 x 16 blocks, one colour each: what survives
/// JPEG compression well enough to tell the blocks apart.
pub(crate) fn colour_blocks() -> RgbImage {
    const COLOURS: [[u8; 3]; 6] = [
        [220, 30, 30],
        [30, 200, 30],
        [30, 30, 220],
        [230, 230, 40],
        [40, 220, 220],
        [240, 240, 240],
    ];
    RgbImage::from_fn(
        48,
        32,
        |x, y| Rgb(COLOURS[(x / 16 + 3 * (y / 16)) as usize]),
    )
}
