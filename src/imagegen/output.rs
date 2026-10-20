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

//! PNG output for generated images.
//!
//! RGB and RGBA are both encoded as they are; RGBA keeps its alpha channel.
//! File naming follows `_indexed_output_path` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/openai.py>:
//! the extension is forced to `.png`, and `count > 1` inserts `-01`, `-02`, ...
//! before it.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use image::ExtendedColorType;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;

use super::pipeline::GeneratedImage;

/// Encode `img` as PNG bytes.
pub fn encode_png(img: &GeneratedImage) -> Result<Vec<u8>> {
    let color = match img.channels {
        3 => ExtendedColorType::Rgb8,
        4 => ExtendedColorType::Rgba8,
        other => bail!("unsupported channel count {other}; expected 3 (RGB) or 4 (RGBA)"),
    };
    if img.width == 0 || img.height == 0 {
        bail!("cannot encode an empty {}x{} image", img.width, img.height);
    }
    if img.pixels.len() != img.expected_len() {
        bail!(
            "pixel buffer holds {} bytes but a {}x{}x{} image needs {}",
            img.pixels.len(),
            img.width,
            img.height,
            img.channels,
            img.expected_len()
        );
    }
    let mut out = Vec::new();
    PngEncoder::new(&mut out).write_image(&img.pixels, img.width, img.height, color)?;
    Ok(out)
}

/// Encode `img` as PNG and return it base64-encoded (standard alphabet).
pub fn to_b64_png(img: &GeneratedImage) -> Result<String> {
    Ok(STANDARD.encode(encode_png(img)?))
}

/// Path of output `index` (0-based) out of `count`.
///
/// The extension is forced to `.png`. One image keeps the name; several get
/// `-01`, `-02`, ... before the extension.
#[must_use]
pub fn indexed_output_path(path: &Path, index: u32, count: u32) -> PathBuf {
    let path = if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        path.to_path_buf()
    } else {
        path.with_extension("png")
    };
    if count <= 1 {
        return path;
    }
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}-{:02}.png", index + 1))
}

/// Default output path for a run: `outputs/image-<seed>.png`.
#[must_use]
pub fn default_output_path(seed: u64) -> PathBuf {
    PathBuf::from("outputs").join(format!("image-{seed}.png"))
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
