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

//! One Inkling 2 decoder layer.
//!
//! Derived from mlx-vlm `DecoderLayer` (`mlx_vlm/models/inkling2/language.py`,
//! MIT).

use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::layers::RMSNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::attention::Inkling2Attention;
use super::config::Inkling2TextConfig;
use super::conv::ShortConv;
use super::moe::Inkling2Moe;
use super::quant::Inkling2QuantTable;
use super::residual::{AttentionResidual, MessageController};

/// Sliding layers share one input norm; global layers norm q, k, v apart.
enum InputNorms {
    Shared(RMSNorm),
    Split([RMSNorm; 3]),
}

pub struct Inkling2DecoderLayer {
    is_block_boundary: bool,
    attention_input: Option<AttentionResidual>,
    mlp_input: AttentionResidual,
    attn: Inkling2Attention,
    mlp: Inkling2Moe,
    input_norms: InputNorms,
    post_attention_layernorm: RMSNorm,
    attn_sconv: ShortConv,
    mlp_sconv: ShortConv,
    /// attn, k, v, mlp; loop-core layers only.
    controllers: Option<[MessageController; 4]>,
}

fn norm(weights: &WeightMap, key: &str, hidden: usize, eps: f32) -> Result<RMSNorm, String> {
    let value = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    if mlxcel_core::array_shape(value) != [hidden as i32] {
        return Err(format!(
            "{key} must be [{hidden}], got {:?}",
            mlxcel_core::array_shape(value)
        ));
    }
    Ok(RMSNorm::new(mlxcel_core::copy(value), eps))
}

fn slice_part(mixed: &MlxArray, index: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::squeeze_axis(
        &mlxcel_core::utils::slice_axis(mixed, -2, index, index + 1),
        -2,
    )
}

impl Inkling2DecoderLayer {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Inkling2TextConfig,
        layer: usize,
        quant: (i32, i32),
        table: Option<&Inkling2QuantTable>,
    ) -> Result<Self, String> {
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps;
        let is_sliding = config.layer_is_sliding(layer);
        let normalize = config.attn_res_normalize_values;
        let attention_input = (layer > 0)
            .then(|| {
                AttentionResidual::from_weights(
                    weights,
                    &format!("{prefix}.attention_input.query"),
                    if is_sliding { 1 } else { 3 },
                    hidden,
                    normalize,
                )
            })
            .transpose()?;
        let input_norms = if is_sliding {
            InputNorms::Shared(norm(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                eps,
            )?)
        } else {
            InputNorms::Split([
                norm(
                    weights,
                    &format!("{prefix}.q_input_layernorm.weight"),
                    hidden,
                    eps,
                )?,
                norm(
                    weights,
                    &format!("{prefix}.k_input_layernorm.weight"),
                    hidden,
                    eps,
                )?,
                norm(
                    weights,
                    &format!("{prefix}.v_input_layernorm.weight"),
                    hidden,
                    eps,
                )?,
            ])
        };
        let controllers = match config.loop_config {
            Some(lc) if config.in_loop_core(layer) => {
                let load = |name: &str| {
                    MessageController::from_weights(
                        weights,
                        &format!("{prefix}.controllers.{name}"),
                        hidden,
                        lc.gate_rank,
                        eps,
                    )
                };
                Some([load("attn")?, load("k")?, load("v")?, load("mlp")?])
            }
            _ => None,
        };
        Ok(Self {
            is_block_boundary: layer.is_multiple_of(config.n_layers_per_attn_res_block),
            attention_input,
            mlp_input: AttentionResidual::from_weights(
                weights,
                &format!("{prefix}.mlp_input.query"),
                1,
                hidden,
                normalize,
            )?,
            attn: Inkling2Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                layer,
                quant,
            )?,
            mlp: Inkling2Moe::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                config,
                layer,
                quant,
                table,
            )?,
            input_norms,
            post_attention_layernorm: norm(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                eps,
            )?,
            attn_sconv: ShortConv::from_weights(weights, &format!("{prefix}.attn_sconv"), hidden)?,
            mlp_sconv: ShortConv::from_weights(weights, &format!("{prefix}.mlp_sconv"), hidden)?,
            controllers,
        })
    }

    pub fn attention(&self) -> &Inkling2Attention {
        &self.attn
    }

    pub fn conv_kernel(&self) -> i32 {
        self.attn_sconv.kernel()
    }

    /// `x` is the partial block entering the layer; `blocks` the completed
    /// blocks of this forward (appended to at a block boundary); `messages`
    /// the `[B, L, 4, hidden]` previous-pass messages inside a looped core.
    pub fn forward(
        &self,
        x: &MlxArray,
        blocks: &mut Vec<UniquePtr<MlxArray>>,
        cache: &mut InklingLayerCache,
        messages: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let x_dtype = mlxcel_core::array_dtype(x);
        let mixed = match &self.attention_input {
            Some(residual) => residual.forward(blocks, x),
            None => mlxcel_core::expand_dims(x, -2),
        };
        let attn = match (&self.input_norms, messages) {
            (InputNorms::Shared(norm), None) => {
                let input = mlxcel_core::astype(&norm.forward(&slice_part(&mixed, 0)), x_dtype);
                self.attn.forward(&input, &input, &input, cache)
            }
            (norms, messages) => {
                let count =
                    mlxcel_core::array_shape(&mixed)[mlxcel_core::array_shape(&mixed).len() - 2];
                let mut parts: Vec<_> = (0..3)
                    .map(|i| slice_part(&mixed, i.min(count - 1)))
                    .collect();
                if let (Some(messages), Some(controllers)) = (messages, &self.controllers) {
                    parts = parts
                        .iter()
                        .enumerate()
                        .map(|(i, part)| {
                            controllers[i].forward(part, &slice_part(messages, i as i32))
                        })
                        .collect();
                }
                let normed: Vec<_> = parts
                    .iter()
                    .enumerate()
                    .map(|(i, part)| {
                        let norm = match norms {
                            InputNorms::Shared(norm) => norm,
                            InputNorms::Split(norms) => &norms[i],
                        };
                        mlxcel_core::astype(&norm.forward(part), x_dtype)
                    })
                    .collect();
                self.attn.forward(&normed[0], &normed[1], &normed[2], cache)
            }
        };
        let residual = self.attn_sconv.forward(
            &attn,
            &mut cache.conv[2],
            (!self.is_block_boundary).then_some(x),
            dtype::FLOAT32,
        );
        if self.is_block_boundary {
            blocks.push(mlxcel_core::copy(x));
        }
        let mut hidden = slice_part(&self.mlp_input.forward(blocks, &residual), 0);
        if let (Some(messages), Some(controllers)) = (messages, &self.controllers) {
            hidden = controllers[3].forward(&hidden, &slice_part(messages, 3));
        }
        let hidden = mlxcel_core::astype(&self.post_attention_layernorm.forward(&hidden), x_dtype);
        let hidden = self.mlp.forward(&hidden);
        self.mlp_sconv
            .forward(&hidden, &mut cache.conv[3], Some(&residual), x_dtype)
    }
}
