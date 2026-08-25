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

//! BERT and XLM-RoBERTa encoder (`model_type: bert` / `xlm-roberta`).
//!
//! Both families are the same post-LayerNorm encoder over absolute position
//! embeddings; they differ only in how position ids are built, in the
//! `type_vocab_size` / `layer_norm_eps` / `pad_token_id` defaults and in the
//! checkpoint's weight-key prefix. One [`BertVariant`] switch therefore
//! covers `sentence-transformers/all-MiniLM-L6-v2`,
//! `intfloat/multilingual-e5-small` (both `BertModel`) and `BAAI/bge-m3`,
//! `BAAI/bge-reranker-v2-m3` (both XLM-RoBERTa).
//!
//! The heads that sit on top of this encoder (the `/v1/embeddings` model and
//! the `ForSequenceClassification` reranker head) live in
//! [`super::bert_heads`].

use anyhow::{Result, bail};
use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

pub use super::bert_config::{BertArgs, BertVariant};

/// Strip the task-head prefix and drop the tensors this port never reads.
///
/// - a leading `bert.` or `roberta.` is removed, so a
///   `ForSequenceClassification` checkpoint and a bare `BertModel` export
///   land on the same keys;
/// - `*position_ids*` buffers (registered non-parameters older transformers
///   exports still ship) are dropped;
/// - `cls.*` and `lm_head.*` masked-LM heads are dropped;
/// - `pooler.*` is dropped unless the caller is building a classifier head,
///   which is the only consumer of the BERT pooler.
///
/// Idempotent: the prefixes are gone after one pass, so a second pass is a
/// no-op, matching the convention in [`super::sanitize`].
pub fn sanitize(weights: WeightMap, keep_pooler: bool) -> WeightMap {
    weights
        .into_iter()
        .filter_map(|(key, value)| {
            let stripped = key
                .strip_prefix("bert.")
                .or_else(|| key.strip_prefix("roberta."))
                .map(str::to_string);
            let key = stripped.unwrap_or(key);
            let drop = key.contains("position_ids")
                || key.starts_with("cls.")
                || key.starts_with("lm_head.")
                || (!keep_pooler && key.starts_with("pooler."));
            (!drop).then_some((key, value))
        })
        .collect()
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm> {
    let weight_key = format!("{prefix}.weight");
    let Some(weight) = weights.get(&weight_key).map(|w| mlxcel_core::copy(w)) else {
        bail!("bert weight not found: {weight_key}");
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
        .map_err(|e| anyhow::anyhow!("bert: {e}"))
}

/// XLM-RoBERTa position ids: `cumsum(input_ids != pad, axis=1) * mask + pad`.
///
/// The result starts at `pad_token_id + 1` for the first real token and stays
/// at `pad_token_id` on padding, matching
/// `transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids`.
/// It keys off the token ids rather than the attention mask, exactly as
/// upstream does.
pub fn xlm_roberta_position_ids(input_ids: &MlxArray, pad_token_id: i32) -> UniquePtr<MlxArray> {
    let pad = mlxcel_core::from_slice_i32(&[pad_token_id], &[1, 1]);
    let real = mlxcel_core::astype(&mlxcel_core::not_equal(input_ids, &pad), dtype::INT32);
    let running = mlxcel_core::cumsum(&real, 1, false, true);
    mlxcel_core::add(&mlxcel_core::multiply(&running, &real), &pad)
}

/// Word, position and token-type tables plus the embedding LayerNorm.
pub struct BertEmbeddings {
    word: UnifiedEmbedding,
    position: UnifiedEmbedding,
    token_type: UnifiedEmbedding,
    layer_norm: LayerNorm,
    variant: BertVariant,
    pad_token_id: i32,
}

impl BertEmbeddings {
    fn from_weights(
        weights: &WeightMap,
        args: &BertArgs,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let embedding = |name: &str| -> Result<UnifiedEmbedding> {
            UnifiedEmbedding::from_weights(weights, &format!("embeddings.{name}"), group_size, bits)
                .map_err(|e| anyhow::anyhow!("bert: {e}"))
        };
        Ok(Self {
            word: embedding("word_embeddings")?,
            position: embedding("position_embeddings")?,
            token_type: embedding("token_type_embeddings")?,
            layer_norm: load_layer_norm(weights, "embeddings.LayerNorm", args.layer_norm_eps)?,
            variant: args.variant,
            pad_token_id: args.pad_token_id,
        })
    }

    /// `[B, L]` (XLM-RoBERTa) or `[1, L]` (BERT) int32 position ids. BERT's
    /// row is identical for every batch member, so it is left unbroadcast and
    /// the addition below broadcasts it.
    fn position_ids(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        match self.variant {
            BertVariant::Bert => {
                let length = mlxcel_core::array_shape(input_ids)[1];
                mlxcel_core::reshape(&mlxcel_core::arange_i32(0, length, 1), &[1, length])
            }
            BertVariant::XlmRoberta => xlm_roberta_position_ids(input_ids, self.pad_token_id),
        }
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        token_type_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let words = self.word.forward(input_ids);
        let positions = self.position.forward(&self.position_ids(input_ids));
        // Absent segment ids mean "everything is segment 0"; a `[1, 1]` index
        // produces a `[1, 1, D]` row that broadcasts over `[B, L, D]`.
        let zero = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
        let segment_ids: &MlxArray = token_type_ids.unwrap_or(&zero);
        let segments = self.token_type.forward(segment_ids);
        let sum = mlxcel_core::add(&mlxcel_core::add(&words, &positions), &segments);
        self.layer_norm.forward(&sum)
    }
}

/// One post-LayerNorm encoder block.
pub struct BertLayer {
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

impl BertLayer {
    fn from_weights(
        weights: &WeightMap,
        index: usize,
        args: &BertArgs,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let base = format!("encoder.layer.{index}");
        let eps = args.layer_norm_eps;
        Ok(Self {
            query: load_linear(
                weights,
                &format!("{base}.attention.self.query"),
                group_size,
                bits,
            )?,
            key: load_linear(
                weights,
                &format!("{base}.attention.self.key"),
                group_size,
                bits,
            )?,
            value: load_linear(
                weights,
                &format!("{base}.attention.self.value"),
                group_size,
                bits,
            )?,
            attn_out: load_linear(
                weights,
                &format!("{base}.attention.output.dense"),
                group_size,
                bits,
            )?,
            attn_norm: load_layer_norm(
                weights,
                &format!("{base}.attention.output.LayerNorm"),
                eps,
            )?,
            intermediate: load_linear(
                weights,
                &format!("{base}.intermediate.dense"),
                group_size,
                bits,
            )?,
            output: load_linear(weights, &format!("{base}.output.dense"), group_size, bits)?,
            out_norm: load_layer_norm(weights, &format!("{base}.output.LayerNorm"), eps)?,
            num_heads: args.num_attention_heads as i32,
            scale: (args.head_dim() as f32).powf(-0.5),
        })
    }

    /// `[B, H, L, head_dim]` view of one `[B, L, D]` projection.
    fn split_heads(&self, x: &MlxArray, b: i32, l: i32, head_dim: i32) -> UniquePtr<MlxArray> {
        let reshaped = mlxcel_core::reshape(x, &[b, l, self.num_heads, head_dim]);
        mlxcel_core::transpose_axes(&reshaped, &[0, 2, 1, 3])
    }

    fn forward(
        &self,
        hidden: &MlxArray,
        mask: &MlxArray,
        activation: Activation,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let head_dim = d / self.num_heads;

        let q = self.split_heads(&self.query.forward(hidden), b, l, head_dim);
        let k = self.split_heads(&self.key.forward(hidden), b, l, head_dim);
        let v = self.split_heads(&self.value.forward(hidden), b, l, head_dim);
        let attended = mlxcel_core::layers::attention(&q, &k, &v, self.scale, Some(mask), 0.0, 0);
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(&attended, &[b, l, d]);

        let projected = self.attn_out.forward(&attended);
        let attn_state = self
            .attn_norm
            .forward(&mlxcel_core::add(&projected, hidden));

        let expanded = activation.apply(&self.intermediate.forward(&attn_state));
        let contracted = self.output.forward(&expanded);
        self.out_norm
            .forward(&mlxcel_core::add(&contracted, &attn_state))
    }
}

/// Feed-forward activation named by `config.json` `hidden_act`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Exact erf GELU, the default for every published BERT / XLM-R export.
    Gelu,
    /// The tanh approximation (`gelu_new`, `gelu_pytorch_tanh`).
    GeluTanh,
    Relu,
}

impl Activation {
    fn from_name(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "gelu" | "gelu_python" => Ok(Self::Gelu),
            "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => Ok(Self::GeluTanh),
            "relu" => Ok(Self::Relu),
            other => bail!(
                "bert config: hidden_act `{other}` is not supported; expected gelu, \
                 gelu_new / gelu_pytorch_tanh or relu"
            ),
        }
    }

    fn apply(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Gelu => mlxcel_core::gelu(x),
            Self::GeluTanh => gelu_tanh(x),
            Self::Relu => mlxcel_core::relu(x),
        }
    }
}

/// HuggingFace `gelu_new` / `gelu_pytorch_tanh`:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// `mlxcel_core::gelu` and `gelu_approx` are both the exact erf form, so the
/// tanh variant is spelled out here. The cube is built with multiplications
/// rather than a generic power, which is undefined for negative inputs.
fn gelu_tanh(x: &MlxArray) -> UniquePtr<MlxArray> {
    let constant = |value: f32| mlxcel_core::from_slice_f32(&[value], &[1]);
    let cubed = mlxcel_core::multiply(&mlxcel_core::multiply(x, x), x);
    let inner = mlxcel_core::multiply(
        &constant(0.797_884_6),
        &mlxcel_core::add(x, &mlxcel_core::multiply(&constant(0.044_715), &cubed)),
    );
    let cdf = mlxcel_core::multiply(
        &constant(0.5),
        &mlxcel_core::add(&constant(1.0), &mlxcel_core::tanh(&inner)),
    );
    mlxcel_core::multiply(x, &cdf)
}

/// The shared encoder trunk: embeddings plus `num_hidden_layers` blocks.
pub struct BertEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    activation: Activation,
    args: BertArgs,
}

impl BertEncoder {
    /// Build the trunk from already-sanitized weights.
    pub fn from_weights(
        weights: &WeightMap,
        args: BertArgs,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let activation = Activation::from_name(&args.hidden_act)?;
        let embeddings = BertEmbeddings::from_weights(weights, &args, group_size, bits)?;
        let layers = (0..args.num_hidden_layers)
            .map(|index| BertLayer::from_weights(weights, index, &args, group_size, bits))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            layers,
            activation,
            args,
        })
    }

    /// Resolved config of this trunk.
    pub fn args(&self) -> &BertArgs {
        &self.args
    }

    /// Run the encoder over a right-padded `[B, L]` batch and return the
    /// `[B, L, D]` last hidden state.
    ///
    /// `attention_mask` is `[B, L]` int32 (`1` = real token) and becomes the
    /// additive `[B, 1, 1, L]` bidirectional mask every block shares.
    pub fn encode(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
        token_type_ids: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(input_ids);
        if shape.len() != 2 {
            bail!("bert encoder: input_ids must be [B, L], got {shape:?}");
        }
        let length = shape[1] as usize;
        let usable = self.args.max_sequence_length();
        if length > usable {
            bail!(
                "bert encoder: {length} tokens exceed the {usable} positions this checkpoint's \
                 absolute position table addresses (max_position_embeddings = {})",
                self.args.max_position_embeddings
            );
        }
        let mask = mlxcel_core::utils::create_bidirectional_padding_mask(attention_mask);
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask, self.activation);
        }
        Ok(hidden)
    }
}

#[cfg(test)]
#[path = "bert_tests.rs"]
pub(crate) mod bert_tests;
