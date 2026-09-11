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

//! On-the-fly KV cache quantization for bandwidth-optimized transfer.
//!
//! Wraps the existing `tensor_quantize` module to provide cache-aware
//! quantization that operates on complete `SerializableCacheEntry` values.
//! Supports per-layer or global quantization with configurable levels.
//!
//! # Bandwidth Savings
//!
//! | Level | Reduction | Typical Perplexity Impact |
//! |-------|-----------|--------------------------|
//! | None  | 0%        | Baseline                 |
//! | Int8  | ~50%      | <0.5% increase           |
//! | Int4  | ~75%      | <2% increase             |
//!
//! Used by: KV cache transfer optimization (streamed, parallel transfers)

use std::collections::HashMap;

use anyhow::{Context, Result};

use super::CacheQuantizationLevel;
use crate::distributed::kv_cache_serde::types::{RawTensorData, SerializableCacheEntry};
use crate::distributed::tensor_quantize;

/// Per-layer quantization configuration.
///
/// Allows different quantization levels for different layers. This is
/// useful because early/late layers may be more sensitive to quantization
/// error than middle layers.
#[derive(Debug, Clone)]
pub struct CacheQuantizationConfig {
    /// Default quantization level for all layers.
    pub default_level: CacheQuantizationLevel,
    /// Per-layer overrides (layer_index -> level).
    /// Layers not in this map use `default_level`.
    pub layer_overrides: HashMap<usize, CacheQuantizationLevel>,
}

impl CacheQuantizationConfig {
    /// Create a uniform config that applies the same level to all layers.
    pub fn uniform(level: CacheQuantizationLevel) -> Self {
        Self {
            default_level: level,
            layer_overrides: HashMap::new(),
        }
    }

    /// Create a config that keeps first and last N layers unquantized
    /// (these tend to be most sensitive to precision loss).
    pub fn protect_boundary_layers(
        level: CacheQuantizationLevel,
        num_layers: usize,
        boundary: usize,
    ) -> Self {
        let mut overrides = HashMap::new();
        for i in 0..boundary.min(num_layers) {
            overrides.insert(i, CacheQuantizationLevel::None);
        }
        for i in num_layers.saturating_sub(boundary)..num_layers {
            overrides.insert(i, CacheQuantizationLevel::None);
        }
        Self {
            default_level: level,
            layer_overrides: overrides,
        }
    }

    /// Get the quantization level for a specific layer.
    pub fn level_for_layer(&self, layer_index: usize) -> CacheQuantizationLevel {
        self.layer_overrides
            .get(&layer_index)
            .copied()
            .unwrap_or(self.default_level)
    }
}

impl Default for CacheQuantizationConfig {
    fn default() -> Self {
        Self::uniform(CacheQuantizationLevel::None)
    }
}

/// On-the-fly quantization wrapper for KV cache transfer.
///
/// Applies quantization to cache entries before serialization and
/// dequantization after deserialization. Reuses the existing
/// `tensor_quantize` module for the actual quantization math.
pub struct QuantizedCacheTransfer {
    config: CacheQuantizationConfig,
}

impl QuantizedCacheTransfer {
    /// Create a new quantized transfer with the given config.
    pub fn new(config: CacheQuantizationConfig) -> Self {
        Self { config }
    }

    /// Quantize a single cache entry's tensor data in-place.
    ///
    /// Returns a new entry with quantized data, plus the original byte
    /// counts for bandwidth tracking.
    pub fn quantize_entry(
        &self,
        entry: &SerializableCacheEntry,
        layer_index: usize,
    ) -> Result<QuantizedEntry> {
        let level = self.config.level_for_layer(layer_index);

        if level == CacheQuantizationLevel::None {
            return Ok(QuantizedEntry {
                entry: entry.clone(),
                level,
                original_key_bytes: entry.keys.as_ref().map_or(0, |k| k.data.len()),
                original_value_bytes: entry.values.as_ref().map_or(0, |v| v.data.len()),
                quantized_key_bytes: entry.keys.as_ref().map_or(0, |k| k.data.len()),
                quantized_value_bytes: entry.values.as_ref().map_or(0, |v| v.data.len()),
            });
        }

        let (quantized_keys, orig_k_bytes, quant_k_bytes) =
            quantize_tensor_opt(entry.keys.as_ref(), level)
                .with_context(|| format!("quantizing layer {layer_index} keys"))?;

        let (quantized_values, orig_v_bytes, quant_v_bytes) =
            quantize_tensor_opt(entry.values.as_ref(), level)
                .with_context(|| format!("quantizing layer {layer_index} values"))?;

        Ok(QuantizedEntry {
            entry: SerializableCacheEntry {
                keys: quantized_keys,
                values: quantized_values,
            },
            level,
            original_key_bytes: orig_k_bytes,
            original_value_bytes: orig_v_bytes,
            quantized_key_bytes: quant_k_bytes,
            quantized_value_bytes: quant_v_bytes,
        })
    }

    /// Quantize all layers of a cache state.
    pub fn quantize_all(&self, entries: &[SerializableCacheEntry]) -> Result<Vec<QuantizedEntry>> {
        entries
            .iter()
            .enumerate()
            .map(|(i, entry)| self.quantize_entry(entry, i))
            .collect()
    }

    /// Dequantize a quantized entry back to the original dtype.
    pub fn dequantize_entry(&self, entry: &QuantizedEntry) -> Result<SerializableCacheEntry> {
        if entry.level == CacheQuantizationLevel::None {
            return Ok(entry.entry.clone());
        }

        let keys = dequantize_tensor_opt(
            entry.entry.keys.as_ref(),
            entry.level,
            entry.original_key_bytes / 2, // float16 = 2 bytes per element
        )
        .context("dequantizing keys")?;

        let values = dequantize_tensor_opt(
            entry.entry.values.as_ref(),
            entry.level,
            entry.original_value_bytes / 2,
        )
        .context("dequantizing values")?;

        Ok(SerializableCacheEntry { keys, values })
    }

    /// Return the config.
    pub fn config(&self) -> &CacheQuantizationConfig {
        &self.config
    }
}

/// Result of quantizing a single cache entry.
#[derive(Debug, Clone)]
pub struct QuantizedEntry {
    /// The (potentially quantized) cache entry.
    pub entry: SerializableCacheEntry,
    /// Quantization level applied.
    pub level: CacheQuantizationLevel,
    /// Original key tensor size in bytes.
    pub original_key_bytes: usize,
    /// Original value tensor size in bytes.
    pub original_value_bytes: usize,
    /// Quantized key tensor size in bytes.
    pub quantized_key_bytes: usize,
    /// Quantized value tensor size in bytes.
    pub quantized_value_bytes: usize,
}

impl QuantizedEntry {
    /// Total original size (keys + values).
    pub fn original_total_bytes(&self) -> usize {
        self.original_key_bytes + self.original_value_bytes
    }

    /// Total quantized size (keys + values).
    pub fn quantized_total_bytes(&self) -> usize {
        self.quantized_key_bytes + self.quantized_value_bytes
    }

    /// Compression ratio (quantized / original).
    pub fn compression_ratio(&self) -> f64 {
        let orig = self.original_total_bytes();
        if orig == 0 {
            return 1.0;
        }
        self.quantized_total_bytes() as f64 / orig as f64
    }
}

/// Quantize an optional tensor, returning the quantized version plus
/// original and quantized byte counts.
fn quantize_tensor_opt(
    tensor: Option<&RawTensorData>,
    level: CacheQuantizationLevel,
) -> Result<(Option<RawTensorData>, usize, usize)> {
    let Some(tensor) = tensor else {
        return Ok((None, 0, 0));
    };

    let original_bytes = tensor.data.len();

    let quantized_data = match level {
        CacheQuantizationLevel::None => {
            return Ok((Some(tensor.clone()), original_bytes, original_bytes));
        }
        CacheQuantizationLevel::Int8 => tensor_quantize::quantize_int8(&tensor.data),
        CacheQuantizationLevel::Int4 => tensor_quantize::quantize_int4(&tensor.data),
    };

    let quantized_bytes = quantized_data.len();

    // Store quantized data in a RawTensorData with the original shape
    // but a marker dtype. The actual dequantization uses the quantized
    // wire format (scales + packed data), not MLX dtype codes.
    let quantized_tensor = RawTensorData {
        data: quantized_data,
        shape: tensor.shape.clone(),
        // Use matching dtype code as marker (INT8=5, INT4=8).
        // Actual data is in quantized wire format (scales + packed data).
        dtype: if level == CacheQuantizationLevel::Int8 {
            5
        } else {
            8
        },
    };

    Ok((Some(quantized_tensor), original_bytes, quantized_bytes))
}

/// Dequantize an optional tensor back to float16.
fn dequantize_tensor_opt(
    tensor: Option<&RawTensorData>,
    level: CacheQuantizationLevel,
    num_elements: usize,
) -> Result<Option<RawTensorData>> {
    let Some(tensor) = tensor else {
        return Ok(None);
    };

    let dequantized_data = match level {
        CacheQuantizationLevel::None => return Ok(Some(tensor.clone())),
        CacheQuantizationLevel::Int8 => {
            tensor_quantize::dequantize_int8(&tensor.data, num_elements)
        }
        CacheQuantizationLevel::Int4 => {
            tensor_quantize::dequantize_int4(&tensor.data, num_elements)
        }
    };

    Ok(Some(RawTensorData {
        data: dequantized_data,
        shape: tensor.shape.clone(),
        dtype: 9, // FLOAT16
    }))
}

/// Estimate the bandwidth savings for quantizing a cache of the given size.
pub fn estimate_savings(
    original_bytes: usize,
    level: CacheQuantizationLevel,
) -> QuantizationSavings {
    let ratio = level.bandwidth_ratio();
    let quantized_bytes = (original_bytes as f64 * ratio) as usize;
    QuantizationSavings {
        original_bytes,
        quantized_bytes,
        saved_bytes: original_bytes.saturating_sub(quantized_bytes),
        ratio,
    }
}

/// Summary of bandwidth savings from quantization.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationSavings {
    /// Original data size in bytes.
    pub original_bytes: usize,
    /// Estimated quantized size in bytes.
    pub quantized_bytes: usize,
    /// Bytes saved.
    pub saved_bytes: usize,
    /// Ratio (quantized / original).
    pub ratio: f64,
}

#[cfg(test)]
#[path = "quantized_tests.rs"]
mod tests;
