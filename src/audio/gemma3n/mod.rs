// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Token-exact Gemma 3n audio reference path.
//!
//! This is intentionally separate from the Gemma 4 audio implementation:
//! Gemma 3n has a different waveform front-end, cumulative SSCP group norm,
//! attention scaling, weight layout, and fixed 188-token merge contract.

mod attention;
mod config;
mod encoder;
mod feature_extractor;

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

/// Maintained implementation used for architecture and frontend fixtures.
pub const GEMMA3N_TRANSFORMERS_REFERENCE_REVISION: &str =
    "181beb3ba4c47098ed8cbc97ee250d1d45ae0107";
/// Official `google/gemma-3n-E4B` checkpoint revision used for weight/layout qualification.
pub const GEMMA3N_E4B_REFERENCE_REVISION: &str = "af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee";

/// Load a linear tensor while checking its declared logical dimensions.
/// Quantized layouts are additionally validated by `UnifiedLinear`; here we
/// pin the output width and all dense/sidecar ranks so malformed audio exports
/// fail during model loading rather than in request-time matmul.
pub(crate) fn checked_unified_linear(
    weights: &WeightMap,
    prefix: &str,
    input_size: usize,
    output_size: usize,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear, String> {
    let weight_key = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_key)
        .ok_or_else(|| format!("Gemma3n audio weight not found: {weight_key}"))?;
    let weight_shape = mlxcel_core::array_shape(weight);
    let quantized = weights.contains_key(&format!("{prefix}.scales"));
    let valid_weight = weight_shape.len() == 2
        && weight_shape[0] == output_size as i32
        && (quantized || weight_shape[1] == input_size as i32);
    if !valid_weight {
        return Err(format!(
            "Gemma3n audio linear {weight_key} has shape {weight_shape:?}; expected logical [{output_size}, {input_size}]"
        ));
    }
    if let Some(scales) = weights.get(&format!("{prefix}.scales")) {
        let scales_shape = mlxcel_core::array_shape(scales);
        if scales_shape.len() != 2 || scales_shape[0] != output_size as i32 {
            return Err(format!(
                "Gemma3n audio linear {prefix}.scales has invalid shape {scales_shape:?}"
            ));
        }
        if let Some(biases) = weights.get(&format!("{prefix}.biases"))
            && mlxcel_core::array_shape(biases) != scales_shape
        {
            return Err(format!(
                "Gemma3n audio linear {prefix}.biases must match scales shape {scales_shape:?}"
            ));
        }
    }
    if let Some(bias) = weights.get(&format!("{prefix}.bias"))
        && mlxcel_core::array_shape(bias) != [output_size as i32]
    {
        return Err(format!(
            "Gemma3n audio linear {prefix}.bias must have {output_size} elements"
        ));
    }
    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
}

pub use config::Gemma3nAudioConfig;
pub use encoder::Gemma3nAudioEncoder;
pub use feature_extractor::{
    GEMMA3N_AUDIO_SOFT_TOKENS, GEMMA3N_MAX_SAMPLES, GEMMA3N_SAMPLE_RATE, Gemma3nAudioFeatureBatch,
    Gemma3nAudioFeatureExtractor,
};
