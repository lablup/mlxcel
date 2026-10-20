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

//! Request validation and output writing shared by the CLI and the server.
//!
//! Kept in the library so the same code path is exercised by the CLI tests,
//! the route tests and the binaries.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::output::{encode_png, indexed_output_path};
use super::pipeline::{GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults};

/// A random 32-bit seed, as upstream picks when `--seed` is omitted.
///
/// Mixes the clock, the process id and a per-process counter, so two calls in
/// the same clock tick still differ. It only has to differ between runs, not
/// resist guessing.
#[must_use]
pub fn random_seed() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tick = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mixed = (nanos as u64)
        ^ ((nanos >> 64) as u64)
        ^ (u64::from(std::process::id()) << 32)
        ^ tick.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    // SplitMix64 finalizer so adjacent inputs give unrelated seeds.
    let mut z = mixed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) & u64::from(u32::MAX)
}

/// Parse `WIDTHxHEIGHT` (case-insensitive `x`, or `×`).
pub fn parse_size(value: &str) -> Result<(u32, u32)> {
    let normalized = value.to_lowercase().replace('×', "x");
    let err = || anyhow::anyhow!("Size must be WIDTHxHEIGHT, got {value:?}");
    let (w, h) = normalized.split_once('x').ok_or_else(err)?;
    let width: u32 = w.trim().parse().map_err(|_| err())?;
    let height: u32 = h.trim().parse().map_err(|_| err())?;
    Ok((width, height))
}

/// Check `req` against `model` and return the defaults it will run with.
///
/// Nothing here loads weights. The request must use a task the family
/// implements and a size on the family's grid.
pub fn validate_request(
    model: &dyn ImageGenerationModel,
    req: &ImageRequest,
) -> Result<ImageSamplingDefaults> {
    if !model.supports(req.task) {
        bail!(
            "image model family {} does not support task {:?}",
            model.family(),
            req.task.as_str()
        );
    }
    req.validate(model.size_multiple())?;
    Ok(model.defaults(req.task))
}

/// Validate, run the model, and check what it returned.
pub fn run_request(
    model: &dyn ImageGenerationModel,
    req: &ImageRequest,
) -> Result<Vec<GeneratedImage>> {
    validate_request(model, req)?;
    let images = model.generate(req)?;
    if images.len() != req.n as usize {
        bail!(
            "image model returned {} image(s) for n={}",
            images.len(),
            req.n
        );
    }
    for (index, image) in images.iter().enumerate() {
        if image.pixels.len() != image.expected_len() {
            bail!(
                "image model returned a malformed image at index {index}: {} bytes for {}x{}x{}",
                image.pixels.len(),
                image.width,
                image.height,
                image.channels
            );
        }
    }
    Ok(images)
}

/// Create the output directory and confirm a file can be written there.
///
/// Run before any weight is loaded so an unwritable `--output` fails in
/// milliseconds instead of after a model load.
pub fn prepare_output_dir(output: &Path) -> Result<()> {
    let Some(dir) = output.parent().filter(|p| !p.as_os_str().is_empty()) else {
        return Ok(());
    };
    std::fs::create_dir_all(dir)
        .with_context(|| format!("cannot create output directory {}", dir.display()))?;
    let probe = dir.join(format!(".mlxcel-write-test-{}", std::process::id()));
    std::fs::write(&probe, b"")
        .with_context(|| format!("output directory {} is not writable", dir.display()))?;
    // Best effort: a leftover empty probe file is harmless.
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

/// Write each image as a PNG; with more than one, names get `-01`, `-02`, ...
pub fn write_images(images: &[GeneratedImage], output: &Path) -> Result<Vec<PathBuf>> {
    let count = images.len() as u32;
    let mut written = Vec::with_capacity(images.len());
    for (index, image) in images.iter().enumerate() {
        let path = indexed_output_path(output, index as u32, count);
        let png = encode_png(image)?;
        std::fs::write(&path, png).with_context(|| format!("cannot write {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

/// Run `req` and write the images to `output`.
///
/// The one call behind both `mlxcel generate-image` and the CLI tests: image
/// `i` of `req.n` lands at `output` (`n == 1`) or `output-0{i+1}.png`.
pub fn generate_to_files(
    model: &dyn ImageGenerationModel,
    req: &ImageRequest,
    output: &Path,
) -> Result<Vec<PathBuf>> {
    let images = run_request(model, req)?;
    write_images(&images, output)
}

#[cfg(test)]
#[path = "run_tests.rs"]
mod tests;
