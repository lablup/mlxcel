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

//! Nemotron-Parse weight-key normalization.
//!
//! Two layouts exist in the wild:
//!
//! - the Hub original (`nvidia/NVIDIA-Nemotron-Parse-2.0`): `encoder.*` holds
//!   the C-RADIO wrapper (`encoder.model_encoder.radio_model.model.*`) plus
//!   the neck, `decoder.*` holds the mBART decoder, and the LM head is tied;
//! - the MLX conversion (`mlx-community/Nemotron-Parse-2.0-{4bit,8bit}`):
//!   `vision_tower.*` and `language_model.{model,lm_head}.*`.
//!
//! [`canonicalize_keys`] maps the first onto the second and brings the two
//! neck convolutions into the matmul layout the encoder consumes, so the
//! model builders only ever see one layout. Both conv rewrites are gated on
//! the tensor shape and are idempotent.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Prefix of the vision tower (C-RADIO ViT plus neck) in the canonical layout.
pub(crate) const VISION_PREFIX: &str = "vision_tower";
/// Prefix of the text half in the canonical layout.
pub(crate) const TEXT_PREFIX: &str = "language_model";

const HUB_RADIO: &str = "encoder.model_encoder.radio_model.";

/// Rename one Hub key into the canonical layout. `None` drops the tensor.
fn canonical_key(key: &str) -> Option<String> {
    if key.starts_with("vision_tower.") || key.starts_with("language_model.") {
        return Some(key.to_string());
    }
    if let Some(rest) = key.strip_prefix(HUB_RADIO) {
        // `summary_idxs` is a buffer the config already describes, and the
        // input conditioner's mean/std are applied by the image processor.
        if rest == "summary_idxs" || rest.starts_with("input_conditioner.") {
            return None;
        }
        let rest = rest.strip_prefix("model.")?;
        let mapped = match rest {
            "patch_generator.embedder.weight" => "patch_embed.weight".to_string(),
            "patch_generator.pos_embed" => "pos_embed".to_string(),
            "patch_generator.cls_token.token" => "cls_token".to_string(),
            other if other.starts_with("blocks.") => other.to_string(),
            _ => return None,
        };
        return Some(format!("{VISION_PREFIX}.{mapped}"));
    }
    if let Some(rest) = key.strip_prefix("encoder.") {
        return Some(format!("{VISION_PREFIX}.neck.{rest}"));
    }
    if let Some(rest) = key.strip_prefix("decoder.") {
        // Auxiliary multi-token prediction heads kept from training; the
        // inference path never reads them.
        if rest.starts_with("extra_heads.") || rest.starts_with("extra_proj.") {
            return None;
        }
        if let Some(tail) = rest.strip_prefix("embed_tokens.") {
            return Some(format!("{TEXT_PREFIX}.model.shared.{tail}"));
        }
        return Some(format!("{TEXT_PREFIX}.model.decoder.{rest}"));
    }
    if let Some(rest) = key.strip_prefix("lm_head.") {
        return Some(format!("{TEXT_PREFIX}.lm_head.{rest}"));
    }
    None
}

/// Whether a `conv1.weight` shape is the torch Conv1d layout
/// `[out, in, 1]` (as opposed to MLX `[out, 1, in]` or an already-squeezed
/// `[out, in]` matrix).
pub(crate) fn conv1_is_torch_layout(shape: &[i32]) -> bool {
    shape.len() == 3 && shape[2] == 1 && shape[1] != 1
}

/// Whether a `conv2.weight` shape is the torch Conv2d layout
/// `[out, in, kh, kw]` (as opposed to MLX `[out, kh, kw, in]`).
pub(crate) fn conv2_is_torch_layout(shape: &[i32]) -> bool {
    shape.len() == 4 && shape[1] == shape[0] && shape[3] != shape[0]
}

/// Squeeze the 1x1 `conv1` kernel into a `[out, in]` Linear weight. Accepts
/// the torch `[out, in, 1]`, the MLX `[out, 1, in]`, and the already
/// squeezed `[out, in]` layouts.
fn conv1_as_linear(w: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(w);
    match shape.as_slice() {
        [o, i, 1] if conv1_is_torch_layout(&shape) => mlxcel_core::reshape(w, &[*o, *i]),
        [o, 1, i] => mlxcel_core::reshape(w, &[*o, *i]),
        _ => mlxcel_core::copy(w),
    }
}

/// Bring `conv2` into MLX `[out, kh, kw, in]` layout.
fn conv2_to_mlx(w: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(w);
    if conv2_is_torch_layout(&shape) {
        mlxcel_core::transpose_axes(w, &[0, 2, 3, 1])
    } else {
        mlxcel_core::copy(w)
    }
}

/// Map a Hub or MLX-converted weight map onto the canonical
/// `vision_tower.*` / `language_model.*` layout, dropping tensors inference
/// does not use and normalizing the two neck conv kernels.
pub(crate) fn canonicalize_keys(weights: WeightMap) -> WeightMap {
    let conv1_key = format!("{VISION_PREFIX}.neck.conv1.weight");
    let conv2_key = format!("{VISION_PREFIX}.neck.conv2.weight");
    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        let Some(new_key) = canonical_key(&key) else {
            continue;
        };
        let value = if new_key == conv1_key {
            conv1_as_linear(&value)
        } else if new_key == conv2_key {
            conv2_to_mlx(&value)
        } else {
            value
        };
        out.insert(new_key, value);
    }
    out
}

/// Tensors the model consumes as raw dense arrays rather than through the
/// unified (possibly quantized) layers. A packed `uint32` plane reaching one
/// of them would abort inside MLX instead of failing, so a checkpoint that
/// quantizes any of them is refused at load with the tensor named.
pub(crate) fn reject_unsupported_quantized_tensors(weights: &WeightMap) -> Result<(), String> {
    for key in weights.keys() {
        let Some(base) = key.strip_suffix(".scales") else {
            continue;
        };
        let dense_only = base.ends_with("pos_embed")
            || base.ends_with("cls_token")
            || base.ends_with("neck.conv1")
            || base.ends_with("neck.conv2")
            || base.contains("norm");
        if dense_only {
            return Err(format!(
                "Nemotron-Parse checkpoint quantizes {base}, which this model reads as a dense \
                 tensor; re-export it with that tensor left unquantized"
            ));
        }
    }
    Ok(())
}
