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

//! DeBERTa-v2 / v3 encoder (`model_type: deberta-v2`).
//!
//! A post-LayerNorm encoder whose attention is *disentangled*: besides the
//! content-to-content scores it adds content-to-position (`c2p`) and
//! position-to-content (`p2c`) terms read from a shared table of relative
//! position embeddings, indexed by the log-bucketed distance between query
//! and key. There are no absolute position embeddings
//! (`position_biased_input: false`) and no token types (`type_vocab_size: 0`)
//! in the published v3 checkpoints, which is the configuration this port
//! accepts.
//!
//! The trunk carries no task head and knows nothing about the heads built on
//! it: GLiNER2.5 (`src/token_classification/gliner2_5.rs`) is the first
//! consumer. Weight keys are the Hugging Face `DebertaV2Model` layout
//! (`embeddings.word_embeddings.weight`, `encoder.layer.N.attention.self.
//! query_proj`, `encoder.rel_embeddings.weight`, `encoder.LayerNorm`), so a
//! head loader strips its own prefix before handing the map over.
//!
//! Relative positions follow Hugging Face `build_relative_position`: with
//! `position_buckets` set, a distance up to half the bucket count is kept and
//! larger ones are log-bucketed up to `max_relative_positions`, which falls
//! back to `max_position_embeddings` when the config sets it below 1. mlx-vlm's
//! `gliner2_5/deberta.py` leaves distances unbucketed when
//! `max_relative_positions <= 0` and clips them to the table, so the two agree
//! only on inputs shorter than the half bucket count (128 tokens for the
//! published geometry); this port follows Hugging Face.

use anyhow::{Result, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::bert::Activation;

pub use super::deberta_v2_config::{DebertaV2Args, relative_position_buckets};

/// Score every masked key gets before the softmax, as upstream.
const MASK_LOGIT: f32 = -1e4;

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm> {
    let weight_key = format!("{prefix}.weight");
    let Some(weight) = weights.get(&weight_key).map(|w| mlxcel_core::copy(w)) else {
        bail!("deberta-v2 weight not found: {weight_key}");
    };
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

fn load_linear(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear> {
    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
        .map_err(|e| anyhow::anyhow!("deberta-v2: {e}"))
}

/// Everything a layer needs besides the hidden states, built once per
/// forward pass and shared by every layer.
struct RelativeContext {
    /// `[1, 2 * buckets, D]` normalized relative embeddings.
    rel_embeddings: UniquePtr<MlxArray>,
    /// `[1, 1, L, L]` int32 gather indices for the `c2p` term.
    c2p_index: Option<UniquePtr<MlxArray>>,
    /// `[1, 1, L, L]` int32 gather indices for the `p2c` term.
    p2c_index: Option<UniquePtr<MlxArray>>,
    /// `[B, 1, L, L]` bool `keep` mask, `None` when every token is real.
    keep: Option<UniquePtr<MlxArray>>,
}

/// One post-LayerNorm block with disentangled attention.
pub struct DebertaV2Layer {
    query: UnifiedLinear,
    key: UnifiedLinear,
    value: UnifiedLinear,
    attn_out: UnifiedLinear,
    attn_norm: LayerNorm,
    intermediate: UnifiedLinear,
    output: UnifiedLinear,
    out_norm: LayerNorm,
    num_heads: i32,
    scale: f32,
}

impl DebertaV2Layer {
    fn from_weights(
        weights: &WeightMap,
        index: usize,
        args: &DebertaV2Args,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let base = format!("encoder.layer.{index}");
        let eps = args.layer_norm_eps;
        let linear =
            |suffix: &str| load_linear(weights, &format!("{base}.{suffix}"), group_size, bits);
        Ok(Self {
            query: linear("attention.self.query_proj")?,
            key: linear("attention.self.key_proj")?,
            value: linear("attention.self.value_proj")?,
            attn_out: linear("attention.output.dense")?,
            attn_norm: load_layer_norm(
                weights,
                &format!("{base}.attention.output.LayerNorm"),
                eps,
            )?,
            intermediate: linear("intermediate.dense")?,
            output: linear("output.dense")?,
            out_norm: load_layer_norm(weights, &format!("{base}.output.LayerNorm"), eps)?,
            num_heads: args.num_attention_heads as i32,
            scale: args.attention_scale(),
        })
    }

    /// `[B, H, L, head_dim]` view of one `[B, L, D]` projection.
    fn split_heads(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let reshaped = mlxcel_core::reshape(x, &[b, l, self.num_heads, d / self.num_heads]);
        mlxcel_core::transpose_axes(&reshaped, &[0, 2, 1, 3])
    }

    fn scaled(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::divide(
            x,
            &mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(x)),
        )
    }

    fn forward(
        &self,
        hidden: &MlxArray,
        context: &RelativeContext,
        activation: Activation,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let (b, l, d) = (shape[0], shape[1], shape[2]);

        let q = self.split_heads(&self.query.forward(hidden));
        let k = self.split_heads(&self.key.forward(hidden));
        let v = self.split_heads(&self.value.forward(hidden));
        let mut scores = self.scaled(&mlxcel_core::matmul(
            &q,
            &mlxcel_core::swap_axes(&k, -1, -2),
        ));

        // `share_att_key`: the content projections double as the position
        // projections. `[1, H, 2 * buckets, head_dim]` each.
        let position_key = self.split_heads(&self.key.forward(&context.rel_embeddings));
        let position_query = self.split_heads(&self.query.forward(&context.rel_embeddings));
        let gather_shape = [b, self.num_heads, l, l];
        if let Some(index) = &context.c2p_index {
            let c2p = mlxcel_core::matmul(&q, &mlxcel_core::swap_axes(&position_key, -1, -2));
            let index = mlxcel_core::broadcast_to(index, &gather_shape);
            let gathered = mlxcel_core::take_along_axis(&c2p, &index, -1);
            scores = mlxcel_core::add(&scores, &self.scaled(&gathered));
        }
        if let Some(index) = &context.p2c_index {
            let p2c = mlxcel_core::matmul(&k, &mlxcel_core::swap_axes(&position_query, -1, -2));
            let index = mlxcel_core::broadcast_to(index, &gather_shape);
            let gathered = mlxcel_core::take_along_axis(&p2c, &index, -1);
            scores = mlxcel_core::add(
                &scores,
                &self.scaled(&mlxcel_core::swap_axes(&gathered, -1, -2)),
            );
        }
        if let Some(keep) = &context.keep {
            let masked = mlxcel_core::full_f32(&[1], MASK_LOGIT, mlxcel_core::array_dtype(&scores));
            scores = mlxcel_core::where_cond(keep, &scores, &masked);
        }
        let probabilities = mlxcel_core::softmax(&scores, -1);
        let attended = mlxcel_core::matmul(&probabilities, &v);
        let attended = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]),
            &[b, l, d],
        );

        let attn_state = self
            .attn_norm
            .forward(&mlxcel_core::add(&self.attn_out.forward(&attended), hidden));
        let expanded = activation.apply(&self.intermediate.forward(&attn_state));
        let out = self.out_norm.forward(&mlxcel_core::add(
            &self.output.forward(&expanded),
            &attn_state,
        ));
        // The whole trunk runs through MLX's own CUDA graph chunking. The
        // first run of a fresh process used to return wrong hidden states in
        // about 40% of the processes on GB10 (a cuBLASLt GEMM racing with
        // concurrent kernel nodes of the same graph); the CUDA overlay now
        // makes captured child graphs barriers (issue #2462), and
        // `deberta_v2_tests::real_encoder_first_run_equals_replay` runs the
        // trunk in fresh processes to keep it that way.
        out
    }
}

/// The encoder trunk: word embeddings, the relative table and
/// `num_hidden_layers` blocks.
pub struct DebertaV2Encoder {
    word_embeddings: UnifiedEmbedding,
    embedding_norm: LayerNorm,
    rel_embeddings: UnifiedEmbedding,
    rel_norm: Option<LayerNorm>,
    layers: Vec<DebertaV2Layer>,
    activation: Activation,
    args: DebertaV2Args,
}

impl DebertaV2Encoder {
    /// Build the trunk from weights in the Hugging Face `DebertaV2Model`
    /// layout.
    pub fn from_weights(
        weights: &WeightMap,
        args: DebertaV2Args,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let activation = Activation::from_name(&args.hidden_act).map_err(|e| {
            anyhow::anyhow!(
                "deberta-v2 config: {}",
                e.to_string().trim_start_matches("bert config: ")
            )
        })?;
        let embedding = |prefix: &str| {
            UnifiedEmbedding::from_weights(weights, prefix, group_size, bits)
                .map_err(|e| anyhow::anyhow!("deberta-v2: {e}"))
        };
        let word_embeddings = embedding("embeddings.word_embeddings")?;
        let rel_embeddings = embedding("encoder.rel_embeddings")?;
        let rows = mlxcel_core::array_shape(rel_embeddings.weight())[0] as usize;
        if rows < 2 * args.position_buckets {
            bail!(
                "deberta-v2: encoder.rel_embeddings has {rows} rows, fewer than the 2 * {} the \
                 position buckets address",
                args.position_buckets
            );
        }
        if !rel_embeddings.is_quantized() {
            let width = mlxcel_core::array_shape(rel_embeddings.weight())[1] as usize;
            if width != args.hidden_size {
                bail!(
                    "deberta-v2: encoder.rel_embeddings is {width} wide, not hidden_size {}",
                    args.hidden_size
                );
            }
        }
        let rel_norm = args
            .norm_rel_ebd
            .then(|| load_layer_norm(weights, "encoder.LayerNorm", args.layer_norm_eps))
            .transpose()?;
        let layers = (0..args.num_hidden_layers)
            .map(|index| DebertaV2Layer::from_weights(weights, index, &args, group_size, bits))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            word_embeddings,
            embedding_norm: load_layer_norm(weights, "embeddings.LayerNorm", args.layer_norm_eps)?,
            rel_embeddings,
            rel_norm,
            layers,
            activation,
            args,
        })
    }

    pub fn args(&self) -> &DebertaV2Args {
        &self.args
    }

    /// Rows of the word-embedding table, for validating ids before a lookup.
    pub fn vocab_rows(&self) -> usize {
        mlxcel_core::array_shape(self.word_embeddings.weight())[0] as usize
    }

    /// `[L, L]` int32 bucketed `i - j` distances.
    fn relative_matrix(&self, length: i32) -> UniquePtr<MlxArray> {
        let buckets = relative_position_buckets(
            length as usize,
            self.args.position_buckets,
            self.args.max_relative_positions,
        );
        let table = mlxcel_core::from_slice_i32(&buckets, &[buckets.len() as i32]);
        let positions = mlxcel_core::arange_i32(0, length, 1);
        let rows = mlxcel_core::reshape(&positions, &[length, 1]);
        let cols = mlxcel_core::reshape(&positions, &[1, length]);
        let offset = mlxcel_core::from_slice_i32(&[length - 1], &[1]);
        let index = mlxcel_core::add(&mlxcel_core::subtract(&rows, &cols), &offset);
        mlxcel_core::take(&table, &index, 0)
    }

    /// `clip(relative + span, 0, 2 * span - 1)` as a `[1, 1, L, L]` index.
    fn gather_index(&self, relative: &MlxArray, negate: bool) -> UniquePtr<MlxArray> {
        let span = self.args.position_buckets as i32;
        let signed = if negate {
            mlxcel_core::negative(relative)
        } else {
            mlxcel_core::copy(relative)
        };
        let shifted = mlxcel_core::add(&signed, &mlxcel_core::from_slice_i32(&[span], &[1]));
        let clipped = mlxcel_core::clip(
            &shifted,
            &mlxcel_core::from_slice_i32(&[0], &[1]),
            &mlxcel_core::from_slice_i32(&[2 * span - 1], &[1]),
        );
        let shape = mlxcel_core::array_shape(relative);
        mlxcel_core::reshape(&clipped, &[1, 1, shape[0], shape[1]])
    }

    fn relative_context(
        &self,
        batch: i32,
        length: i32,
        attention_mask: Option<&MlxArray>,
    ) -> RelativeContext {
        let span = 2 * self.args.position_buckets as i32;
        // Read the table through the layer rather than off `.weight`: a
        // quantized table holds packed uint32 with a narrower last dimension.
        let full = self.rel_embeddings.dequantized_weight();
        let table = mlxcel_core::slice(&full, &[0, 0], &[span, self.args.hidden_size as i32]);
        let table = match &self.rel_norm {
            Some(norm) => norm.forward(&table),
            None => table,
        };
        let rel_embeddings = mlxcel_core::expand_dims(&table, 0);
        let relative = self.relative_matrix(length);
        let keep = attention_mask.map(|mask| {
            let real = mlxcel_core::astype(mask, dtype::BOOL);
            let rows = mlxcel_core::reshape(&real, &[batch, 1, length, 1]);
            let cols = mlxcel_core::reshape(&real, &[batch, 1, 1, length]);
            mlxcel_core::logical_and(&rows, &cols)
        });
        RelativeContext {
            rel_embeddings,
            c2p_index: self.args.c2p.then(|| self.gather_index(&relative, false)),
            p2c_index: self.args.p2c.then(|| self.gather_index(&relative, true)),
            keep,
        }
    }

    /// Run the encoder over a right-padded `[B, L]` int32 batch and return the
    /// `[B, L, D]` last hidden state.
    ///
    /// `attention_mask` is `[B, L]` int32 (`1` = real token). `None` treats
    /// every position as real, which is how GLiNER2.5 encodes its single
    /// sequence; with a mask, padded positions are zeroed after the embedding
    /// and excluded from every attention row and column, as upstream does.
    pub fn encode(
        &self,
        input_ids: &MlxArray,
        attention_mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        if shape.len() != 2 {
            bail!("deberta-v2 encoder: input_ids must be [B, L], got {shape:?}");
        }
        let (batch, length) = (shape[0], shape[1]);
        if let Some(mask) = attention_mask
            && mlxcel_core::array_shape(mask) != shape
        {
            bail!(
                "deberta-v2 encoder: attention_mask {:?} does not match input_ids {shape:?}",
                mlxcel_core::array_shape(mask)
            );
        }
        let context = self.relative_context(batch, length, attention_mask);
        let mut hidden = self
            .embedding_norm
            .forward(&self.word_embeddings.forward(input_ids));
        if let Some(mask) = attention_mask {
            let mask = mlxcel_core::astype(
                &mlxcel_core::expand_dims(mask, -1),
                mlxcel_core::array_dtype(&hidden),
            );
            hidden = mlxcel_core::multiply(&hidden, &mask);
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &context, self.activation);
        }
        Ok(hidden)
    }
}

#[cfg(test)]
#[path = "deberta_v2_tests.rs"]
pub(crate) mod deberta_v2_tests;
