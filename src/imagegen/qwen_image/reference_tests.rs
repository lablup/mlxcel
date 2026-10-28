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

//! Byte parity of the reference resize and composite against Pillow 13.0.0.
//!
//! The expected checksums come from
//! `Image.fromarray(make(w, h), "RGBA").resize((tw, th), LANCZOS)` and
//! `Image.new("RGB", size, "white").paste(im, mask=im.getchannel("A"))`
//! where `make` fills pixel `(x, y)` channel `c` with
//! `(37x + 101y + 59c + 7 * (xy mod 13)) mod 256`, alpha included, so every
//! alpha level from 0 to 255 occurs.

use super::*;
use crate::imagegen::testing::fnv1a32_bytes;

fn make(width: u32, height: u32) -> RgbaImage {
    RgbaImage::from_fn(width, height, |x, y| {
        let v = |c: u32| ((x * 37 + y * 101 + c * 59 + (x * y) % 13 * 7) % 256) as u8;
        image::Rgba([v(0), v(1), v(2), v(3)])
    })
}

#[test]
fn lanczos_rgba_resize_matches_pillow() {
    // (w, h, target w, target h, FNV-1a of the bytes, first eight bytes)
    let cases: [(u32, u32, u32, u32, u32, [u8; 8]); 5] = [
        (
            37,
            23,
            64,
            32,
            0x62b4_dc52,
            [1, 53, 113, 189, 9, 65, 126, 196],
        ),
        (
            37,
            23,
            16,
            12,
            0xada4_3f67,
            [76, 108, 131, 143, 116, 134, 134, 133],
        ),
        (
            50,
            20,
            48,
            40,
            0x8ae1_8a21,
            [2, 55, 115, 201, 41, 93, 164, 248],
        ),
        (
            31,
            47,
            31,
            20,
            0x5845_cae6,
            [55, 47, 118, 101, 111, 111, 116, 142],
        ),
        (
            40,
            30,
            40,
            30,
            0x1df4_cfc1,
            [0, 59, 118, 177, 37, 96, 155, 214],
        ),
    ];
    for (w, h, tw, th, fnv, head) in cases {
        let out = resize_lanczos_rgba(&make(w, h), tw, th).unwrap();
        assert_eq!(out.dimensions(), (tw, th));
        assert_eq!(&out.as_raw()[..8], &head, "{w}x{h} -> {tw}x{th}");
        assert_eq!(fnv1a32_bytes(out.as_raw()), fnv, "{w}x{h} -> {tw}x{th}");
    }
}

#[test]
fn composite_on_white_matches_pillow_paste() {
    let out = composite_on_white(&make(37, 23));
    assert_eq!(
        &out.as_raw()[..9],
        &[78, 119, 160, 72, 122, 171, 77, 135, 193]
    );
    assert_eq!(fnv1a32_bytes(out.as_raw()), 0x7f0c_5847);
}

#[test]
fn opaque_and_transparent_pixels_composite_exactly() {
    let image = RgbaImage::from_raw(2, 1, vec![10, 20, 30, 255, 10, 20, 30, 0]).unwrap();
    assert_eq!(
        composite_on_white(&image).as_raw(),
        &[10, 20, 30, 255, 255, 255]
    );
}

#[test]
fn zero_sized_targets_are_rejected() {
    assert!(resize_lanczos_rgba(&make(4, 4), 0, 4).is_err());
}
