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

use super::*;
use image::GenericImageView;

fn image(channels: u8, width: u32, height: u32) -> GeneratedImage {
    let len = width as usize * height as usize * usize::from(channels);
    GeneratedImage {
        pixels: (0..len).map(|i| (i % 251) as u8).collect(),
        width,
        height,
        channels,
        seed: 7,
        revised_prompt: None,
    }
}

#[test]
fn rgb_round_trips_through_png() {
    let img = image(3, 5, 4);
    let png = encode_png(&img).unwrap();
    let decoded = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
    assert_eq!(decoded.dimensions(), (5, 4));
    assert_eq!(decoded.color(), image::ColorType::Rgb8);
    assert_eq!(decoded.to_rgb8().into_raw(), img.pixels);
}

#[test]
fn rgba_keeps_its_alpha_channel() {
    let mut img = image(4, 3, 3);
    for px in img.pixels.chunks_exact_mut(4) {
        px[3] = 77;
    }
    let png = encode_png(&img).unwrap();
    let decoded = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
    assert_eq!(decoded.color(), image::ColorType::Rgba8);
    let raw = decoded.to_rgba8().into_raw();
    assert!(raw.chunks_exact(4).all(|px| px[3] == 77));
    assert_eq!(raw, img.pixels);
}

#[test]
fn b64_is_the_standard_alphabet_of_the_png() {
    use base64::Engine as _;
    let img = image(3, 2, 2);
    let b64 = to_b64_png(&img).unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(bytes, encode_png(&img).unwrap());
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
}

#[test]
fn malformed_images_are_rejected() {
    let mut bad = image(3, 2, 2);
    bad.channels = 2;
    assert!(
        encode_png(&bad)
            .unwrap_err()
            .to_string()
            .contains("channel")
    );

    let mut short = image(3, 2, 2);
    short.pixels.pop();
    assert!(
        encode_png(&short)
            .unwrap_err()
            .to_string()
            .contains("pixel buffer")
    );

    let mut empty = image(3, 2, 2);
    empty.width = 0;
    empty.pixels.clear();
    assert!(encode_png(&empty).is_err());
}

#[test]
fn single_image_keeps_its_name_and_gets_a_png_extension() {
    let p = |s: &str| std::path::PathBuf::from(s);
    assert_eq!(indexed_output_path(&p("out.png"), 0, 1), p("out.png"));
    assert_eq!(indexed_output_path(&p("out.PNG"), 0, 1), p("out.PNG"));
    assert_eq!(indexed_output_path(&p("out.jpg"), 0, 1), p("out.png"));
    assert_eq!(indexed_output_path(&p("out"), 0, 1), p("out.png"));
}

#[test]
fn several_images_are_numbered_from_one_with_two_digits() {
    let p = std::path::PathBuf::from("dir/out.png");
    assert_eq!(
        indexed_output_path(&p, 0, 2),
        std::path::PathBuf::from("dir/out-01.png")
    );
    assert_eq!(
        indexed_output_path(&p, 1, 2),
        std::path::PathBuf::from("dir/out-02.png")
    );
    assert_eq!(
        indexed_output_path(&p, 9, 10),
        std::path::PathBuf::from("dir/out-10.png")
    );
    // The extension is forced to png before numbering.
    assert_eq!(
        indexed_output_path(std::path::Path::new("out.jpg"), 0, 2),
        std::path::PathBuf::from("out-01.png")
    );
}

#[test]
fn default_path_is_outputs_image_seed() {
    assert_eq!(
        default_output_path(1234),
        std::path::PathBuf::from("outputs/image-1234.png")
    );
}
