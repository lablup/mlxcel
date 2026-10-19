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

//! Qwen3-VL vision-tower construction shared by every loader that builds one.
//!
//! Qwen3-VL, Qwen3-VL-MoE, Qwen3-Omni, Qwen 3.5 / 3.8 and Cohere Compass all
//! build [`Qwen3VLVisionEncoder`], and they all build it here so the PyTorch
//! Conv3d layout guard runs at every site (#2388).

use mlxcel_core::weights::WeightMap;

use crate::vision::encoders::qwen2_5_vl::normalize_patch_embed_layout;
use crate::vision::encoders::qwen3_vl::{Qwen3VLVisionConfig, Qwen3VLVisionEncoder};

/// Normalize the patch-embedding layout under `prefix`, then build the tower.
///
/// Hub exports of Qwen3.8 (`Qwen/Qwen3.8-27B-FP8` among them) store
/// `patch_embed.proj.weight` in PyTorch's `[out, in, kT, kH, kW]` order, while
/// `PatchEmbed::from_weights` reads the mlx-vlm `[out, kT, kH, kW, in]` order.
/// The element counts match, so without the guard the tower loads and reads
/// scrambled filters. [`normalize_patch_embed_layout`] (added for Qwen2.5-VL in
/// #1423) converts the hub layout in place, leaves an already-converted one
/// untouched, and refuses to guess when both axes equal `in_channels`, as
/// Blaizzy/mlx-vlm#2186 does upstream.
pub(super) fn load_qwen3_vl_vision_encoder(
    weights: &mut WeightMap,
    config: &Qwen3VLVisionConfig,
    prefix: &str,
) -> Result<Qwen3VLVisionEncoder, String> {
    if normalize_patch_embed_layout(weights, prefix, config.in_channels) {
        tracing::debug!(
            "Qwen3-VL: converted {prefix}.patch_embed.proj.weight from the PyTorch Conv3d layout"
        );
    }
    Qwen3VLVisionEncoder::from_weights(weights, config, prefix)
}

#[cfg(test)]
#[path = "vlm_qwen3_vision_tests.rs"]
mod tests;
