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

//! Inkling 2 text decoder (`inkling2` Small, `inkling2_loop` Mini).
//!
//! Every layer mixes its input from the completed blocks through a learned
//! attention residual, convolves k/v and both sublayer outputs causally, and
//! routes a sigmoid MoE with softcapped SwiGLU experts. Mini runs layers
//! `core_start..core_end` `num_passes` times; from the second pass on, every
//! core layer blends in a message read from the previous pass's block outputs.
//! Each layer VISIT owns its KV cache and four convolution states.
//!
//! Derived from mlx-vlm `mlx_vlm/models/inkling2/language.py` (MIT). Its
//! three Mini Metal kernels (`kernels.py`, `grouped_experts.py`; issue
//! #3359) are gated in [`kernels`] and fall back to the ops path, which they
//! match bitwise; `MLXCEL_INKLING2_KERNELS=0` keeps every forward on it.

mod attention;
pub mod config;
mod conv;
pub(crate) mod kernels;
mod layer;
mod moe;
pub mod quant;
mod residual;
mod runtime;
pub(crate) mod sanitize;

use std::path::Path;

use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use self::config::{Inkling2Config, Inkling2LoopConfig};
use self::layer::Inkling2DecoderLayer;
use self::quant::Inkling2QuantTable;
use self::residual::AttentionResidual;
use self::sanitize::{HEAD_KEY, TEXT_PREFIX};
use super::model_owned::ModelOwnedSequenceState;

/// Diagnostic override of Mini's `loop_config.num_passes` (the one-pass
/// ablation in issue #3228). Unset uses the checkpoint's value.
pub const LOOP_PASSES_ENV: &str = "MLXCEL_INKLING2_LOOP_PASSES";

enum Head {
    /// Unquantized head kept as f32 `[vocab, hidden]`: the checkpoint asks
    /// for an f32 product, and promoting the bf16 matrix per call would read
    /// and write the whole vocabulary every decode step.
    Dense(UniquePtr<MlxArray>),
    Quantized(UnifiedLinear),
}

pub struct Inkling2Model {
    config: Inkling2Config,
    embed_tokens: UnifiedEmbedding,
    embed_norm: RMSNorm,
    layers: Vec<Inkling2DecoderLayer>,
    layer_indices: Vec<usize>,
    loop_config: Option<Inkling2LoopConfig>,
    /// One f32 `[4, hidden]` (attn, k, v, mlp) query per core layer.
    message_queries: Vec<UniquePtr<MlxArray>>,
    final_input: AttentionResidual,
    norm: RMSNorm,
    head: Head,
    eos_token_ids: Vec<i32>,
    sequence_state: ModelOwnedSequenceState<InklingLayerCache>,
}

/// `(group_size, bits)` of the dense projections of an MLX conversion; the
/// native release has no quantized dense planes, so the pair is unused there.
pub(crate) fn dense_quantization(config: &Inkling2Config) -> (i32, i32) {
    let block = config.mlx_quantization();
    let read = |key: &str, default: i64| {
        block
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(default) as i32
    };
    (read("group_size", 64), read("bits", 4))
}

fn loop_passes_override() -> Result<Option<usize>, String> {
    match std::env::var(LOOP_PASSES_ENV) {
        Ok(raw) if !raw.trim().is_empty() => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|passes| (1..=64).contains(passes))
            .map(Some)
            .ok_or_else(|| format!("{LOOP_PASSES_ENV} must be an integer in 1..=64, got {raw:?}")),
        _ => Ok(None),
    }
}

impl Inkling2Model {
    pub fn load(model_path: &str) -> Result<(Self, Inkling2Config), String> {
        let path = Path::new(model_path);
        let raw = std::fs::read_to_string(path.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = Inkling2Config::from_json(&raw)?;
        let table = quant::read_native_quant_table(path)?;
        let weights = super::load_text_weights(path, None)?;
        let model = Self::from_weights(config.clone(), weights, table.as_ref())?;
        Ok((model, config))
    }

    /// `weights` may be native, Transformers-layout or mlx-vlm-converted; it
    /// is sanitized here (the remap is idempotent).
    pub fn from_weights(
        mut config: Inkling2Config,
        weights: WeightMap,
        table: Option<&Inkling2QuantTable>,
    ) -> Result<Self, String> {
        if let (Some(passes), Some(lc)) = (
            loop_passes_override()?,
            config.text_config.loop_config.as_mut(),
        ) {
            tracing::warn!(
                "{LOOP_PASSES_ENV}={passes} overrides Inkling 2 loop_config.num_passes {} (diagnostic only)",
                lc.num_passes
            );
            lc.num_passes = passes;
        }
        let text = config.text_config.clone();
        text.validate()?;
        let weights = sanitize::sanitize_weights(weights, text.hidden_size)?;
        let quant = dense_quantization(&config);
        let embed_tokens = UnifiedEmbedding::from_weights(
            &weights,
            &format!("{TEXT_PREFIX}embed_tokens"),
            quant.0,
            quant.1,
        )?;
        let weight = |key: &str| {
            weights
                .get(key)
                .map(|value| mlxcel_core::copy(value))
                .ok_or_else(|| format!("Weight not found: {key}"))
        };
        let embed_norm = RMSNorm::new(
            weight(&format!("{TEXT_PREFIX}embed_norm.weight"))?,
            text.rms_norm_eps,
        );
        let norm = RMSNorm::new(
            weight(&format!("{TEXT_PREFIX}norm.weight"))?,
            text.rms_norm_eps,
        );
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for index in 0..text.num_hidden_layers {
            layers.push(Inkling2DecoderLayer::from_weights(
                &weights,
                &format!("{TEXT_PREFIX}layers.{index}"),
                &text,
                index,
                quant,
                table,
            )?);
        }
        let hidden = text.hidden_size as i32;
        let message_queries = match text.loop_config {
            Some(lc) => {
                let core = (lc.core_end - lc.core_start) as i32;
                let key = format!("{TEXT_PREFIX}message_queries");
                let value = weights
                    .get(&key)
                    .ok_or_else(|| format!("Weight not found: {key}"))?;
                if mlxcel_core::array_shape(value) != [core, 4, hidden] {
                    return Err(format!(
                        "{key} must be [{core}, 4, {hidden}], got {:?}",
                        mlxcel_core::array_shape(value)
                    ));
                }
                let value = mlxcel_core::astype(value, dtype::FLOAT32);
                (0..core)
                    .map(|i| {
                        mlxcel_core::contiguous(
                            &mlxcel_core::squeeze_axis(
                                &mlxcel_core::utils::slice_axis(&value, 0, i, i + 1),
                                0,
                            ),
                            false,
                        )
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        let head =
            if weights.contains_key(&format!("{}scales", HEAD_KEY.trim_end_matches("weight"))) {
                Head::Quantized(UnifiedLinear::from_weights(
                    &weights,
                    HEAD_KEY.trim_end_matches(".weight"),
                    quant.0,
                    quant.1,
                )?)
            } else {
                let value = weight(HEAD_KEY)?;
                if mlxcel_core::array_shape(&value) != [text.vocab_size as i32, hidden] {
                    return Err(format!(
                        "{HEAD_KEY} must be [{}, {hidden}], got {:?}",
                        text.vocab_size,
                        mlxcel_core::array_shape(&value)
                    ));
                }
                Head::Dense(mlxcel_core::contiguous(
                    &mlxcel_core::astype(&value, dtype::FLOAT32),
                    false,
                ))
            };
        let final_input = AttentionResidual::from_weights(
            &weights,
            &format!("{TEXT_PREFIX}final_input.query"),
            1,
            text.hidden_size,
            text.attn_res_normalize_values,
        )?;
        let layer_indices = text.layer_indices();
        let state = (0..layer_indices.len())
            .map(|_| InklingLayerCache::new())
            .collect();
        let eos_token_ids = config.eos_token_ids();
        Ok(Self {
            loop_config: text.loop_config,
            config,
            embed_tokens,
            embed_norm,
            layers,
            layer_indices,
            message_queries,
            final_input,
            norm,
            head,
            eos_token_ids,
            sequence_state: ModelOwnedSequenceState::new(state),
        })
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    /// Number of layer visits, i.e. of caches.
    pub fn visit_count(&self) -> usize {
        self.layer_indices.len()
    }

    /// `embed_norm(embed_tokens(ids))`: the matrix image features are
    /// scattered into before the decoder.
    pub fn normalized_input_embeddings(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        self.embed_norm.forward(&self.embed_tokens.forward(input))
    }

    fn make_internal_caches(&self) -> Vec<InklingLayerCache> {
        (0..self.layer_indices.len())
            .map(|_| InklingLayerCache::new())
            .collect()
    }

    /// Pre-norm final state `[B, L, hidden]` (f32) of normalized embeddings.
    fn hidden(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        assert_eq!(
            caches.len(),
            self.layer_indices.len(),
            "Inkling 2 needs one cache per layer visit"
        );
        let mut x = mlxcel_core::copy(embeddings);
        let mut blocks: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut prelude: Option<(UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>)> = None;
        let mut bank: Option<UniquePtr<MlxArray>> = None;
        let hidden = self.config.text_config.hidden_size as i32;
        for (visit, &index) in self.layer_indices.iter().enumerate() {
            if let Some(lc) = self.loop_config
                && index == lc.core_start
            {
                match &prelude {
                    None => prelude = Some((mlxcel_core::copy(&x), copy_all(&blocks))),
                    Some((prelude_x, prelude_blocks)) => {
                        // The first block appended in the core is the prelude
                        // partial; completed core blocks and the last partial
                        // supply the messages.
                        let mut states: Vec<_> = blocks[prelude_blocks.len() + 1..]
                            .iter()
                            .map(|block| mlxcel_core::astype(block, dtype::FLOAT32))
                            .collect();
                        states.push(mlxcel_core::astype(&x, dtype::FLOAT32));
                        let stacked = mlxcel_core::stack_owned(&states, -2);
                        bank = Some(mlxcel_core::fast_rms_norm_no_weight(&stacked, 1e-6));
                        x = mlxcel_core::copy(prelude_x);
                        blocks = copy_all(prelude_blocks);
                    }
                }
            }
            let messages = match (self.loop_config, &bank) {
                (Some(lc), Some(bank)) if index < lc.core_end => {
                    let query = &self.message_queries[index - lc.core_start];
                    let scores = mlxcel_core::matmul(query, &mlxcel_core::swap_axes(bank, -1, -2));
                    let scores =
                        mlxcel_core::multiply_scalar(&scores, attention::inverse_sqrt(hidden));
                    Some(mlxcel_core::matmul(
                        &mlxcel_core::softmax(&scores, -1),
                        bank,
                    ))
                }
                _ => None,
            };
            x = self.layers[index].forward(
                &x,
                &mut blocks,
                &mut caches[visit],
                messages.as_deref(),
            );
        }
        let out = self.final_input.forward(&blocks, &x);
        mlxcel_core::squeeze_axis(&mlxcel_core::utils::slice_axis(&out, -2, 0, 1), -2)
    }

    /// Final norm, the bf16 rounding the checkpoint's f32 head expects, the
    /// f32 head, the tanh softcap, and the unpadded vocabulary.
    fn project(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let text = &self.config.text_config;
        let h = self.norm.forward(hidden);
        let h = mlxcel_core::astype(
            &mlxcel_core::astype(&h, mlxcel_core::array_dtype(&self.embed_norm.weight)),
            dtype::FLOAT32,
        );
        let mut logits = match &self.head {
            Head::Dense(weight) => mlxcel_core::matmul(&h, &mlxcel_core::transpose(weight)),
            Head::Quantized(head) => head.forward(&h),
        };
        let cap = text.final_logit_softcapping;
        if cap > 0.0 {
            logits = mlxcel_core::multiply_scalar(
                &mlxcel_core::tanh(&mlxcel_core::divide_scalar(&logits, cap)),
                cap,
            );
        }
        let width = text.logits_width() as i32;
        if *mlxcel_core::array_shape(&logits).last().unwrap_or(&0) > width {
            mlxcel_core::utils::slice_axis(&logits, -1, 0, width)
        } else {
            logits
        }
    }

    fn logits(
        &self,
        embeddings: &MlxArray,
        caches: &mut [InklingLayerCache],
        last_pos: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.hidden(embeddings, caches);
        match last_pos {
            Some(row) => self.project(&mlxcel_core::utils::slice_axis(
                &hidden,
                1,
                row as i32,
                row as i32 + 1,
            )),
            None => self.project(&hidden),
        }
    }

    fn token_logits(
        &self,
        input: &MlxArray,
        caches: &mut [InklingLayerCache],
        last_pos: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let embeddings = self.normalized_input_embeddings(input);
        self.logits(&embeddings, caches, last_pos)
    }

    /// Logits for already-normalized (and image-merged) embeddings.
    pub fn forward_prepared_embeddings_with_sequence_id(
        &self,
        embeddings: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        last_pos: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| self.logits(embeddings, state, last_pos),
        )
    }
}

fn copy_all(values: &[UniquePtr<MlxArray>]) -> Vec<UniquePtr<MlxArray>> {
    values
        .iter()
        .map(|value| mlxcel_core::copy(value))
        .collect()
}

#[cfg(test)]
mod tiny_tests;
#[cfg(test)]
mod verify_bench_tests;
