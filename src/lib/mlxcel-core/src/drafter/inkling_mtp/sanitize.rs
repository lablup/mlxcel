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

use std::path::Path;

use crate::UniquePtr;
use crate::drafter::DrafterError;
use crate::ffi::{self, MlxArray};
use crate::weights::{WeightMap, dir_has_tensor_name, load_weights_from_dir_index_filtered};

pub(super) fn is_inkling_mtp_tensor_name(name: &str) -> bool {
    name.starts_with("model.mtp.layers.") || name.starts_with("mtp.layers.")
}

pub(super) fn sanitize_weights(weights: &mut WeightMap) -> Result<(), String> {
    let input = std::mem::take(weights);
    let mut output = WeightMap::new();
    for (key, value) in input {
        if key.starts_with("blocks.") {
            output.insert(key, value);
            continue;
        }
        let Some(rest) = key
            .strip_prefix("model.mtp.")
            .or_else(|| key.strip_prefix("mtp."))
        else {
            continue;
        };
        let Some(rest) = rest.strip_prefix("layers.") else {
            continue;
        };
        let Some((index, sub)) = rest.split_once('.') else {
            return Err(format!("malformed Inkling MTP tensor name: {key}"));
        };
        let base = format!("blocks.{index}");
        match sub {
            "embed_norm.weight" | "hidden_norm.weight" | "input_proj.weight" => {
                output.insert(format!("{base}.{sub}"), value);
            }
            _ => {
                let Some(transformer) = sub.strip_prefix("transformer_block.") else {
                    return Err(format!("unsupported Inkling MTP tensor: {key}"));
                };
                map_transformer_weight(&mut output, &base, transformer, value)?;
            }
        }
    }
    *weights = output;
    Ok(())
}

fn map_transformer_weight(
    output: &mut WeightMap,
    base: &str,
    sub: &str,
    value: UniquePtr<MlxArray>,
) -> Result<(), String> {
    let transformer = format!("{base}.transformer_block");
    if let Some(attention) = sub.strip_prefix("attn.") {
        let (name, _) = attention
            .rsplit_once('.')
            .ok_or_else(|| format!("malformed Inkling MTP attention key: {sub}"))?;
        let mapped = match name {
            "wq_du" => "q_proj.weight",
            "wk_dv" => "k_proj.weight",
            "wv_dv" => "v_proj.weight",
            "wr_du" => "r_proj.weight",
            "wo_ud" => "o_proj.weight",
            "q_norm" => "q_norm.weight",
            "k_norm" => "k_norm.weight",
            "rel_logits_proj" => "rel_proj",
            "k_sconv" | "v_sconv" => {
                let shape = ffi::array_shape(&value);
                if shape.len() != 3 {
                    return Err(format!("{sub}: expected rank-3 convolution, got {shape:?}"));
                }
                let transposed = ffi::transpose_axes(&value, &[0, 2, 1]);
                output.insert(
                    format!("{transformer}.self_attn.{name}.conv.weight"),
                    ffi::contiguous(&transposed, false),
                );
                return Ok(());
            }
            other => return Err(format!("unsupported Inkling MTP attention tensor: {other}")),
        };
        output.insert(format!("{transformer}.self_attn.{mapped}"), value);
        return Ok(());
    }
    match sub {
        "attn_norm.weight" => {
            output.insert(format!("{transformer}.input_layernorm.weight"), value);
        }
        "mlp_norm.weight" => {
            output.insert(
                format!("{transformer}.post_attention_layernorm.weight"),
                value,
            );
        }
        "attn_sconv.weight" | "mlp_sconv.weight" => {
            let name = sub.trim_end_matches(".weight");
            let shape = ffi::array_shape(&value);
            if shape.len() != 3 {
                return Err(format!("{sub}: expected rank-3 convolution, got {shape:?}"));
            }
            let transposed = ffi::transpose_axes(&value, &[0, 2, 1]);
            output.insert(
                format!("{transformer}.{name}.conv.weight"),
                ffi::contiguous(&transposed, false),
            );
        }
        "mlp.w13_dn.weight" => {
            let shape = ffi::array_shape(&value);
            if shape.len() != 2 || shape[0] % 2 != 0 {
                return Err(format!("{sub}: expected [2I,H], got {shape:?}"));
            }
            let paired = ffi::reshape(&value, &[shape[0] / 2, 2, shape[1]]);
            let gate = ffi::squeeze_axis(&crate::utils::slice_axis(&paired, 1, 0, 1), 1);
            let up = ffi::squeeze_axis(&crate::utils::slice_axis(&paired, 1, 1, 2), 1);
            output.insert(
                format!("{transformer}.mlp.gate_proj.weight"),
                ffi::contiguous(&gate, false),
            );
            output.insert(
                format!("{transformer}.mlp.up_proj.weight"),
                ffi::contiguous(&up, false),
            );
        }
        "mlp.w2_md.weight" => {
            output.insert(format!("{transformer}.mlp.down_proj.weight"), value);
        }
        "mlp.gate.global_scale" | "mlp.global_scale" => {
            output.insert(format!("{transformer}.mlp.global_scale"), value);
        }
        other => {
            return Err(format!(
                "unsupported Inkling MTP transformer tensor: {other}"
            ));
        }
    }
    Ok(())
}

/// Load and sanitize the `model.mtp.layers.*` tensors under `path`, keeping
/// their checkpoint dtype.
///
/// Deliberately skips the shared drafter bf16 -> f16 load policy
/// (`apply_drafter_load_dtype_policy`). Inkling targets stay in their
/// checkpoint dtype on every backend (the f16 cast breaks the native NVFP4
/// target, issue #1549), so an f16 drafter would put an f16/bf16 boundary at
/// every shared embedding, norm and LM-head call, and MLX promotes that mix to
/// float32. Measured on an M3 Ultra with `Inkling-Small-mlx-4bit` and the
/// published bf16 `mtp.safetensors`: 128 greedy tokens at
/// `--draft-block-size 2` took 11.6 s with an f16 drafter and 5.1 s with the
/// bf16 shard as published, at the same acceptance.
///
/// Used by: [`super::InklingMtpDraftModel::from_path`].
pub(super) fn load_drafter_weights(path: &Path) -> Result<WeightMap, DrafterError> {
    let mut weights = load_weights_from_dir_index_filtered(path, |name| {
        is_inkling_mtp_tensor_name(name) || name.starts_with("blocks.")
    })
    .map_err(|reason| DrafterError::WeightLoad { reason })?;
    if weights.is_empty() {
        return Err(DrafterError::WeightLoad {
            reason: format!(
                "{} has no model.mtp.layers.* tensors; an Inkling config flag alone is not a drafter",
                path.display()
            ),
        });
    }
    sanitize_weights(&mut weights).map_err(|reason| DrafterError::WeightLoad { reason })?;
    Ok(weights)
}

pub fn has_inkling_mtp_tensors(path: &Path) -> Result<bool, DrafterError> {
    dir_has_tensor_name(path, is_inkling_mtp_tensor_name)
        .map_err(|reason| DrafterError::WeightLoad { reason })
}
