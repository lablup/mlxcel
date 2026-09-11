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

//! `preprocessor_config.json` (`media_proc_cfg`) parsing for the Kimi K3
//! image processor.
//!
//! Used by: `processors::kimi_k3`.

use serde::Deserialize;
use std::path::Path;

use super::{
    ChessboardConfig, FillStage, KimiK3ImageProcessor, KimiK3NavitConfig, TransparentBackground,
};

#[derive(Debug, Deserialize)]
struct RawTransparentBg {
    #[serde(default = "default_pattern")]
    pattern: String,
    #[serde(default = "default_square")]
    chessboard_square_size: u32,
    #[serde(default = "default_true")]
    chessboard_square_on_top_left: bool,
    #[serde(default = "default_white")]
    chessboard_white_value: u8,
    #[serde(default = "default_gray")]
    chessboard_gray_value: u8,
}

fn default_pattern() -> String {
    "black".to_string()
}
fn default_square() -> u32 {
    16
}
fn default_true() -> bool {
    true
}
fn default_white() -> u8 {
    255
}
fn default_gray() -> u8 {
    200
}

#[derive(Debug, Deserialize)]
struct RawMediaProcCfg {
    #[serde(flatten)]
    navit: KimiK3NavitConfig,
    #[serde(default = "default_norm")]
    image_mean: [f32; 3],
    #[serde(default = "default_norm")]
    image_std: [f32; 3],
    /// `fixed_output_tokens` pins every image to one token count regardless
    /// of its size. The reference honours it; this port does not implement
    /// it, so a checkpoint that sets it is refused rather than silently
    /// resized by the ordinary rule.
    #[serde(default)]
    fixed_output_tokens: Option<u32>,
    #[serde(default)]
    transparent_bg_config: Option<RawTransparentBg>,
    #[serde(default = "default_fill_stage")]
    transparent_bg_fill_stage: String,
}

fn default_norm() -> [f32; 3] {
    [0.5; 3]
}
fn default_fill_stage() -> String {
    "before_resize".to_string()
}

impl KimiK3ImageProcessor {
    /// Parse a `media_proc_cfg` object.
    pub fn from_media_proc_cfg(cfg: &serde_json::Value) -> Result<Self, String> {
        let raw: RawMediaProcCfg = serde_json::from_value(cfg.clone())
            .map_err(|e| format!("Kimi K3 media_proc_cfg: {e}"))?;
        if let Some(fixed) = raw.fixed_output_tokens {
            return Err(format!(
                "Kimi K3 media_proc_cfg: fixed_output_tokens = {fixed} is not supported; this \
                 port sizes every image by the navit rule"
            ));
        }
        let background = match raw.transparent_bg_config {
            None => None,
            Some(bg) => Some(match bg.pattern.as_str() {
                "white" => TransparentBackground::White,
                "black" => TransparentBackground::Black,
                "gray" => TransparentBackground::Gray,
                "chessboard" => {
                    if bg.chessboard_square_size == 0 {
                        return Err(
                            "Kimi K3 media_proc_cfg: chessboard_square_size must be positive"
                                .into(),
                        );
                    }
                    TransparentBackground::Chessboard(ChessboardConfig {
                        square_size: bg.chessboard_square_size,
                        white_on_top_left: bg.chessboard_square_on_top_left,
                        white: bg.chessboard_white_value,
                        gray: bg.chessboard_gray_value,
                    })
                }
                other => {
                    return Err(format!(
                        "Kimi K3 media_proc_cfg: invalid background pattern {other:?}"
                    ));
                }
            }),
        };
        let fill_stage = match raw.transparent_bg_fill_stage.as_str() {
            "before_resize" => FillStage::BeforeResize,
            "after_resize" => FillStage::AfterResize,
            other => {
                return Err(format!(
                    "Kimi K3 media_proc_cfg: invalid transparent_bg_fill_stage {other:?}"
                ));
            }
        };
        if raw.image_std.contains(&0.0) {
            return Err("Kimi K3 media_proc_cfg: image_std must be non-zero".into());
        }
        Ok(Self {
            navit: raw.navit,
            mean: raw.image_mean,
            std: raw.image_std,
            background,
            fill_stage,
        })
    }

    /// Read `<model_dir>/preprocessor_config.json`; a missing file gives the
    /// published configuration.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("preprocessor_config.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("Failed to parse {}: {e}", path.display()))?;
        match value.get("media_proc_cfg") {
            Some(cfg) => Self::from_media_proc_cfg(cfg),
            None => Ok(Self::default()),
        }
    }
}
