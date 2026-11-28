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

//! Route selection between the FP8 checkpoint layouts the Qwen3.5 loaders
//! reconstruct at load time.
//!
//! Two layouts exist and they are disjoint by `quant_method`:
//!
//! * [`super::fp8_block`]: vendor fine-grained block FP8
//!   (`quant_method: "fp8"`), E4M3 bytes plus a `.weight_scale_inv` per 128x128
//!   block.
//! * [`super::fp8_ct`]: compressed-tensors `float-quantized`
//!   (`quant_method: "compressed-tensors"`), E4M3 bytes plus a `.weight_scale`
//!   per output channel or per `block_structure` tile.
//!
//! Both are requantized to MLX-native `mxfp8`, so the config merge and the
//! weight pre-pass are the same shape for either route. This module keeps the
//! three loader call sites from having to branch on the layout in two places
//! (config and weights) and hand-check that only one applies.
//!
//! Used by: the Qwen3.5 text loader (`Qwen35Model::load`), the Qwen3.5 VLM
//! loader (`load_qwen3_5_vlm_with_variant`) and the special loader
//! (`try_load_special_model_from_weights`, `qwen35_text_config`).

use mlxcel_core::weights::WeightMap;
use serde_json::Value;

use super::fp8_block::{
    Fp8BlockQuantization, merge_quantization_block, qwen_fp8_block_quantization,
    requantize_block_fp8_weights,
};
use super::fp8_ct::{
    Fp8CtQuantization, compressed_tensors_fp8_quantization,
    requantize_compressed_tensors_fp8_weights,
};

/// Any FP8 checkpoint layout these loaders reconstruct at load time.
///
/// The two are selected by different `quant_method` values (`fp8` with a
/// `weight_block_size` versus `compressed-tensors` with `float-quantized`), so
/// at most one applies to a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Checkpoint {
    /// Vendor fine-grained block FP8: `.weight_scale_inv` per 128x128 block.
    VendorBlock(Fp8BlockQuantization),
    /// compressed-tensors `float-quantized` FP8: `.weight_scale` per channel or
    /// per block.
    CompressedTensors(Fp8CtQuantization),
}

impl Fp8Checkpoint {
    /// The `quantization` block to merge into a model config so the model
    /// constructor builds quantized layers for the converted tensors.
    #[must_use]
    pub fn effective_config(&self) -> Value {
        match self {
            Self::VendorBlock(detected) => detected.effective_config(),
            Self::CompressedTensors(detected) => detected.effective_config(),
        }
    }
}

/// Detect whichever FP8 layout `full_config` declares, if any.
///
/// Runs both detections because they are mutually exclusive but neither can
/// tell the other's config apart on its own: an `fp8` declaration is a vendor
/// block checkpoint, a `compressed-tensors` `float-quantized` declaration is
/// the other layout, and everything else returns `Ok(None)`.
pub fn fp8_checkpoint_quantization(full_config: &Value) -> Result<Option<Fp8Checkpoint>, String> {
    if let Some(detected) = qwen_fp8_block_quantization(full_config)? {
        return Ok(Some(Fp8Checkpoint::VendorBlock(detected)));
    }
    Ok(compressed_tensors_fp8_quantization(full_config)?.map(Fp8Checkpoint::CompressedTensors))
}

/// Merge the detected route's `quantization` block into a model config.
///
/// Returns whether a block was inserted; an explicit `quantization` already
/// present always wins.
pub fn merge_fp8_quantization(config: &mut Value, detected: Fp8Checkpoint) -> Result<bool, String> {
    merge_quantization_block(config, detected.effective_config())
}

/// Run the detected route's weight pre-pass, or return `weights` untouched.
///
/// Both routes replace `<name>.weight` with a packed plane and add
/// `<name>.scales`, and both are one-tensor-at-a-time so the peak stays near
/// the checkpoint's own footprint.
pub fn requantize_fp8_weights(
    weights: WeightMap,
    detected: Option<Fp8Checkpoint>,
) -> Result<WeightMap, String> {
    match detected {
        None => Ok(weights),
        Some(Fp8Checkpoint::VendorBlock(detected)) => {
            requantize_block_fp8_weights(weights, detected.block_rows)
        }
        Some(Fp8Checkpoint::CompressedTensors(detected)) => {
            requantize_compressed_tensors_fp8_weights(weights, detected.layout)
        }
    }
}
