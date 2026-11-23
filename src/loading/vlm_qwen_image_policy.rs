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

//! Checkpoint image-resize policy for the Qwen-VL processor family.
//!
//! The HF image processor resizes against the pixel-area bounds and the PIL
//! resampler the checkpoint ships in its sidecars. Qwen3-VL and its siblings
//! publish `size.shortest_edge` 65536, well above the Qwen2-VL family default
//! of 3136, so a loader that ignores the sidecar keeps a 224x224 image at
//! 224x224 (49 image tokens) where HF upscales it to 256x256 (64 tokens).
//! Every eager loader built on `qwen_vl_processor_with_norm` and the XLA
//! Qwen3-VL host preprocessor read the policy through [`QwenImageResizePolicy`]
//! so the two backends cannot drift apart again.
//!
//! Precedence follows HF: the `image_processor` block nested in
//! `processor_config.json` (what `AutoProcessor` builds from) wins over the
//! standalone `preprocessor_config.json`, key by key; within one file an
//! explicit `min_pixels` / `max_pixels` beats `size.shortest_edge` /
//! `size.longest_edge`.

use std::path::Path;

use image::imageops::FilterType;
use serde_json::Value;

use super::read_optional_model_json;
use crate::vision::processors::qwen2_vl::{
    DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, Qwen2VLProcessor,
};

/// PIL `Image.BICUBIC`, the HF default resampler for the Qwen-VL processors.
pub(crate) const PIL_BICUBIC: u64 = 3;

/// Largest pixel bound a checkpoint may state (16384 x 16384). Real
/// checkpoints use 16777216 at most; anything above this would let a crafted
/// sidecar force every image to an allocation-exhausting size.
pub(crate) const MAX_SIDECAR_PIXEL_BOUND: usize = 1 << 28;

/// Resize policy resolved from a checkpoint's processor sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenImageResizePolicy {
    /// Lower pixel-area bound the checkpoint states, if any.
    pub min_pixels: Option<usize>,
    /// Upper pixel-area bound the checkpoint states, if any.
    pub max_pixels: Option<usize>,
    /// PIL resample code the checkpoint states, if any.
    pub resample: Option<u64>,
    /// Whether any processor sidecar was found at all.
    pub has_sidecar: bool,
}

impl QwenImageResizePolicy {
    /// Read the policy from `model_path`. A missing sidecar or key is not an
    /// error; a present value that is not a positive integer, or bounds that
    /// end up inverted, are rejected so a malformed checkpoint cannot silently
    /// change the image token count.
    pub(crate) fn read(model_path: &Path) -> Result<Self, String> {
        let nested = read_optional_model_json(model_path, "processor_config.json")
            .and_then(|config| config.get("image_processor").cloned())
            .filter(Value::is_object);
        let standalone = read_optional_model_json(model_path, "preprocessor_config.json")
            .filter(Value::is_object);
        Self::from_sources(nested.as_ref(), standalone.as_ref())
    }

    fn from_sources(nested: Option<&Value>, standalone: Option<&Value>) -> Result<Self, String> {
        let sources = [
            nested.map(|value| ("processor_config.json image_processor", value)),
            standalone.map(|value| ("preprocessor_config.json", value)),
        ];
        let mut min_pixels = None;
        let mut max_pixels = None;
        let mut resample = None;
        for (name, source) in sources.into_iter().flatten() {
            let min = bound(name, source, "min_pixels", "shortest_edge")?;
            let max = bound(name, source, "max_pixels", "longest_edge")?;
            let code = positive_field(name, source.get("resample"), "resample", true)?;
            min_pixels = min_pixels.or(min);
            max_pixels = max_pixels.or(max);
            resample = resample.or(code);
        }
        let policy = Self {
            min_pixels,
            max_pixels,
            resample,
            has_sidecar: nested.is_some() || standalone.is_some(),
        };
        let (min, max) = policy.pixel_bounds();
        if min > MAX_SIDECAR_PIXEL_BOUND || max > MAX_SIDECAR_PIXEL_BOUND {
            return Err(format!(
                "Qwen-VL image processor pixel bounds exceed {MAX_SIDECAR_PIXEL_BOUND}: min={min}, max={max}"
            ));
        }
        if min > max {
            return Err(format!(
                "Qwen-VL image processor pixel bounds are inverted: min={min}, max={max}"
            ));
        }
        policy.resize_filter()?;
        Ok(policy)
    }

    /// The bounds `smart_resize` uses, filling unstated ones with the family
    /// defaults.
    #[must_use]
    pub(crate) fn pixel_bounds(&self) -> (usize, usize) {
        (
            self.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
            self.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
        )
    }

    /// The resampler matching the checkpoint's PIL `resample` code. A sidecar
    /// without one gets HF's bicubic default; a checkpoint with no sidecar at
    /// all returns `None` and keeps the processor's historical filter.
    pub(crate) fn resize_filter(&self) -> Result<Option<FilterType>, String> {
        let code = match self.resample {
            Some(code) => code,
            None if self.has_sidecar => PIL_BICUBIC,
            None => return Ok(None),
        };
        let filter = match code {
            0 => FilterType::Nearest,
            1 => FilterType::Lanczos3,
            2 => FilterType::Triangle,
            PIL_BICUBIC => FilterType::CatmullRom,
            other => {
                return Err(format!(
                    "Qwen-VL image processor resample={other} is not supported; expected a PIL \
                     code of 0 (NEAREST), 1 (LANCZOS), 2 (BILINEAR), or 3 (BICUBIC)"
                ));
            }
        };
        Ok(Some(filter))
    }

    /// Apply the bounds and resampler to `processor`.
    pub(crate) fn apply(&self, processor: Qwen2VLProcessor) -> Result<Qwen2VLProcessor, String> {
        let (min_pixels, max_pixels) = self.pixel_bounds();
        let processor = processor.with_pixel_bounds(min_pixels, max_pixels);
        Ok(match self.resize_filter()? {
            Some(filter) => processor.with_resize_filter(filter),
            None => processor,
        })
    }
}

/// One bound from one source: the explicit key wins over the `size` entry, but
/// both are validated when present.
fn bound(
    source_name: &str,
    source: &Value,
    explicit_key: &str,
    size_key: &str,
) -> Result<Option<usize>, String> {
    let size = match source.get("size") {
        None | Some(Value::Null) => None,
        Some(Value::Object(size)) => size.get(size_key),
        Some(_) => return Err(format!("{source_name} `size` must be an object")),
    };
    let from_size = positive_field(source_name, size, &format!("size.{size_key}"), false)?;
    let explicit = positive_field(source_name, source.get(explicit_key), explicit_key, false)?;
    explicit
        .or(from_size)
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                format!("{source_name} `{explicit_key}` is too large for this platform")
            })
        })
        .transpose()
}

fn positive_field(
    source_name: &str,
    value: Option<&Value>,
    field: &str,
    allow_zero: bool,
) -> Result<Option<u64>, String> {
    let kind = if allow_zero {
        "non-negative"
    } else {
        "positive"
    };
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|&value| allow_zero || value > 0)
            .map(Some)
            .ok_or_else(|| {
                format!("{source_name} `{field}` must be a {kind} integer, got {value}")
            }),
    }
}

#[cfg(test)]
#[path = "vlm_qwen_image_policy_tests.rs"]
mod tests;
