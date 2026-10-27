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

//! The Bonsai checkpoint layout and `quantization_config.json`.
//!
//! Counterparts of `REQUIRED_FILES` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/bonsai/download.py>
//! and `_read_packed_quant_config` in `klein_fast/loader.py`. Upstream reads
//! only `bits` and `group_size` and treats any linear with a sibling `.scales`
//! as packed; this port also checks `format` and holds the file to its own
//! `quantized_modules` list (see `loader.rs`).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

use crate::imagegen::flux2::config::{Flux2Layout, QuantParams};

/// The packed ternary transformer (`Flux2Transformer2DModel` config).
pub const TRANSFORMER_DIR: &str = "transformer-packed-mflux";
/// The MLX 4-bit Qwen3 conditioner.
pub const TEXT_ENCODER_DIR: &str = "text_encoder-mlx-4bit";
pub const QUANT_CONFIG_FILE: &str = "quantization_config.json";
/// The only `format` this port (and upstream's loader) reads.
pub const PACKED_FORMAT: &str = "mlx-packed-affine";

/// The component directories of a Bonsai checkpoint, as a [`Flux2Layout`] so
/// the FLUX.2 pipeline loads them unchanged. Names the first missing one.
pub fn layout(root: &Path) -> Result<Flux2Layout> {
    if !root.is_dir() {
        bail!("Bonsai model directory does not exist: {}", root.display());
    }
    let layout = Flux2Layout {
        root: root.to_path_buf(),
        transformer: root.join(TRANSFORMER_DIR),
        text_encoder: root.join(TEXT_ENCODER_DIR),
        vae: root.join("vae"),
        tokenizer: root.join("tokenizer"),
    };
    for dir in [
        &layout.transformer,
        &layout.text_encoder,
        &layout.vae,
        &layout.tokenizer,
    ] {
        if !dir.is_dir() {
            bail!("Bonsai checkpoint is missing {}", dir.display());
        }
    }
    for file in [
        layout.transformer.join(QUANT_CONFIG_FILE),
        layout.tokenizer.join("tokenizer.json"),
    ] {
        if !file.is_file() {
            bail!("Bonsai checkpoint is missing {}", file.display());
        }
    }
    Ok(layout)
}

/// `quantization_config.json` of `transformer-packed-mflux/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedQuantConfig {
    pub quant: QuantParams,
    /// `quantized_modules`, under the checkpoint's own tensor names
    /// (`transformer_blocks.0.attn.to_out.0`, before any rename).
    pub modules: BTreeSet<String>,
}

impl PackedQuantConfig {
    /// Read `<dir>/quantization_config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join(QUANT_CONFIG_FILE);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", path.display()))?;
        Self::from_json(&value).with_context(|| format!("invalid {}", path.display()))
    }

    /// Validate `format`, `bits` and `group_size` against what mlxcel-core's
    /// affine path implements, and read `quantized_modules`.
    pub fn from_json(value: &Value) -> Result<Self> {
        let format = value
            .get("format")
            .and_then(Value::as_str)
            .context("quantization_config.json has no string `format`")?;
        if format != PACKED_FORMAT {
            bail!("unsupported Bonsai quantization format {format:?}, expected {PACKED_FORMAT:?}");
        }
        let int = |key: &str| -> Result<i32> {
            let v = value
                .get(key)
                .with_context(|| format!("quantization_config.json has no `{key}`"))?;
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .with_context(|| format!("`{key}` must be an integer, got {v}"))
        };
        let bits = int("bits")?;
        mlxcel_core::layers::validate_affine_quantization_bits(bits).map_err(|e| anyhow!(e))?;
        let group_size = int("group_size")?;
        mlxcel_core::layers::validate_affine_quantization_group_size(group_size)
            .map_err(|e| anyhow!(e))?;
        let list = value
            .get("quantized_modules")
            .and_then(Value::as_array)
            .context("quantization_config.json has no `quantized_modules` list")?;
        let modules = list
            .iter()
            .map(|m| {
                m.as_str().map(str::to_string).with_context(|| {
                    format!("`quantized_modules` entries must be strings, got {m}")
                })
            })
            .collect::<Result<BTreeSet<_>>>()?;
        Ok(Self {
            quant: QuantParams { bits, group_size },
            modules,
        })
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
