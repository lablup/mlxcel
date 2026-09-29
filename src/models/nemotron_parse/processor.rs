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

//! Nemotron-Parse page preprocessing and decoder-seed construction.
//!
//! The page is converted to RGB, shrunk (never enlarged) to fit inside
//! 2048x1664 with the aspect ratio kept, centre-padded with white to exactly
//! that size, scaled to `[0, 1]`, and normalized with the CLIP mean/std.
//!
//! Resizing uses bilinear (`FilterType::Triangle`) filtering. PIL's BILINEAR
//! downscale uses a slightly different antialias tap set, so a page that has
//! to be shrunk is close to, not bit-identical with, the reference; a page
//! already inside the target box only pads, which is exact.
//!
//! Reference: `NemotronParseImageProcessor` in
//! `hf_nemotron_parse_processor.py`.

use std::path::Path;

use anyhow::{Result, anyhow};
use image::DynamicImage;
use image::imageops::FilterType;
use serde_json::Value;

use mlxcel_core::{MlxArray, UniquePtr};

use crate::tokenizer::MlxcelTokenizer;

/// The task prompt the model card uses: bounding boxes, class tags, and
/// markdown text, without transcribing text inside pictures.
pub const DEFAULT_TASK_PROMPT: &str =
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>";

const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Page preprocessing parameters (`preprocessor_config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParseImageProcessor {
    /// `(height, width)` every page is padded to.
    pub final_size: (u32, u32),
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Default for NemotronParseImageProcessor {
    fn default() -> Self {
        Self {
            final_size: (2048, 1664),
            do_normalize: true,
            image_mean: CLIP_MEAN,
            image_std: CLIP_STD,
        }
    }
}

fn triple(v: Option<&Value>, key: &str) -> Result<Option<[f32; 3]>> {
    let Some(v) = v else { return Ok(None) };
    let arr = v
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| anyhow!("Nemotron-Parse preprocessor {key} must be a 3-element list"))?;
    let mut out = [0.0f32; 3];
    for (slot, x) in out.iter_mut().zip(arr) {
        *slot = x
            .as_f64()
            .ok_or_else(|| anyhow!("Nemotron-Parse preprocessor {key} has a non-number"))?
            as f32;
    }
    Ok(Some(out))
}

impl NemotronParseImageProcessor {
    /// Read `preprocessor_config.json`, falling back to the model-card
    /// defaults when the file or a field is absent.
    pub fn from_pretrained(model_path: &Path, default_size: (u32, u32)) -> Result<Self> {
        let mut out = Self {
            final_size: default_size,
            ..Self::default()
        };
        let path = model_path.join("preprocessor_config.json");
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Ok(out);
        };
        let v: Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("Failed to parse {path:?}: {e}"))?;
        if let Some(fs) = v.get("final_size").and_then(Value::as_array)
            && fs.len() == 2
        {
            let h = fs[0].as_u64().unwrap_or(default_size.0 as u64);
            let w = fs[1].as_u64().unwrap_or(default_size.1 as u64);
            if h == 0 || w == 0 || h > 8192 || w > 8192 {
                return Err(anyhow!(
                    "Nemotron-Parse preprocessor final_size {h}x{w} is out of range"
                ));
            }
            out.final_size = (h as u32, w as u32);
        }
        if let Some(b) = v.get("do_normalize").and_then(Value::as_bool) {
            out.do_normalize = b;
        }
        if let Some(m) = triple(v.get("image_mean"), "image_mean")? {
            out.image_mean = m;
        }
        if let Some(s) = triple(v.get("image_std"), "image_std")? {
            if s.contains(&0.0) {
                return Err(anyhow!("Nemotron-Parse preprocessor image_std has a zero"));
            }
            out.image_std = s;
        }
        Ok(out)
    }

    /// Size after the aspect-preserving shrink (`LongestMaxSizeHW`): clamp the
    /// height first and recompute the width, then clamp the width and
    /// recompute the height, truncating like Python `int()`.
    pub fn resized_size(&self, width: u32, height: u32) -> (u32, u32) {
        let (max_h, max_w) = self.final_size;
        let aspect = width as f64 / height as f64;
        let (mut new_w, mut new_h) = (width, height);
        if height > max_h {
            new_h = max_h;
            new_w = (new_h as f64 * aspect) as u32;
        }
        if new_w > max_w {
            new_w = max_w;
            new_h = (new_w as f64 / aspect) as u32;
        }
        (new_w.max(1), new_h.max(1))
    }

    /// Preprocess one page into `[1, 3, H, W]` f32 pixels.
    pub fn preprocess(&self, image: &DynamicImage) -> UniquePtr<MlxArray> {
        let (data, h, w) = self.preprocess_to_vec(image);
        mlxcel_core::from_slice_f32(&data, &[1, 3, h as i32, w as i32])
    }

    /// Host-side CHW pixel buffer plus `(height, width)`.
    pub fn preprocess_to_vec(&self, image: &DynamicImage) -> (Vec<f32>, u32, u32) {
        let rgb = image.to_rgb8();
        let (w0, h0) = rgb.dimensions();
        let (nw, nh) = self.resized_size(w0, h0);
        let rgb = if (nw, nh) != (w0, h0) {
            image::imageops::resize(&rgb, nw, nh, FilterType::Triangle)
        } else {
            rgb
        };
        let (th, tw) = self.final_size;
        let (ch, cw) = (nh.min(th), nw.min(tw));
        let pad_top = (th - ch) / 2;
        let pad_left = (tw - cw) / 2;

        let plane = (th * tw) as usize;
        let mut data = vec![0.0f32; 3 * plane];
        let norm = |c: usize, v: f32| -> f32 {
            let x = v / 255.0;
            if self.do_normalize {
                (x - self.image_mean[c]) / self.image_std[c]
            } else {
                x
            }
        };
        let white: [f32; 3] = [norm(0, 255.0), norm(1, 255.0), norm(2, 255.0)];
        for (c, value) in white.iter().enumerate() {
            data[c * plane..(c + 1) * plane].fill(*value);
        }
        for y in 0..ch {
            for x in 0..cw {
                let p = rgb.get_pixel(x, y);
                let idx = ((y + pad_top) * tw + (x + pad_left)) as usize;
                for c in 0..3 {
                    data[c * plane + idx] = norm(c, p[c] as f32);
                }
            }
        }
        (data, th, tw)
    }
}

/// Decoder seed for a task prompt, as `transformers` `generate` builds it:
/// the prompt is tokenized without the tokenizer's own `<s>`/`</s>` wrapper
/// (the reference processor call passes `add_special_tokens=False`), an empty
/// prompt becomes `[decoder_start]`, and a prompt that does not already start
/// with `decoder_start` gets it prepended.
pub fn seed_ids(tokenizer: &MlxcelTokenizer, prompt: &str, decoder_start: i32) -> Result<Vec<i32>> {
    let ids = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("Nemotron-Parse: failed to tokenize task prompt: {e}"))?;
    let ids: Vec<i32> = ids.into_iter().map(|id| id as i32).collect();
    Ok(seed_from_prompt_ids(ids, decoder_start))
}

/// The pure half of [`seed_ids`].
pub fn seed_from_prompt_ids(mut ids: Vec<i32>, decoder_start: i32) -> Vec<i32> {
    if ids.first() != Some(&decoder_start) {
        ids.insert(0, decoder_start);
    }
    ids
}
