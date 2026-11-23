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

//! The Clef joint schema head: `JointSchemaHead`, `EvidenceRoutingLayer`
//! and `FieldDecoderLayer` of mlx-vlm's `mlx_vlm/models/clef/clef.py`.
//!
//! The head runs in the dtype its tensors are stored in (bf16 on the
//! published checkpoints) and is never quantized. Every op mirrors upstream's
//! MLX graph, including where upstream promotes to f32: the span means are
//! an f32 `[spans, L] @ hidden` matmul cast back to the hidden dtype, and the
//! per-question option softmax is `precise`.

use anyhow::{Result, anyhow, bail};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde::Deserialize;

use super::clef_layers::{EvidenceRoutingLayer, FieldDecoderLayer, Linear, Weights, gelu};

/// `nn.LayerNorm`'s default epsilon.
pub(super) const LAYER_NORM_EPS: f32 = 1e-5;

/// `config.json` `head_config`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) struct HeadConfig {
    /// Backbone hidden size the head reads.
    pub(crate) hidden_size: usize,
    /// Head width.
    pub(crate) width: usize,
    /// Evidence routing layers.
    pub(crate) routing_layers: usize,
    /// Field decoder layers.
    pub(crate) layers: usize,
    /// Attention heads of every head MHA.
    pub(crate) heads: usize,
    /// Feed-forward width.
    pub(crate) feedforward: usize,
}

impl HeadConfig {
    /// Read `head_config` from a parsed `config.json`, naming a missing key.
    pub(crate) fn from_config(config: &serde_json::Value) -> Result<Self> {
        let section = config
            .get("head_config")
            .ok_or_else(|| anyhow!("clef config.json has no head_config"))?;
        for key in [
            "hidden_size",
            "width",
            "routing_layers",
            "layers",
            "heads",
            "feedforward",
        ] {
            if section
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .is_none()
            {
                bail!("clef head_config is missing the integer `{key}`");
            }
        }
        let parsed: Self = serde_json::from_value(section.clone())
            .map_err(|error| anyhow!("parsing clef head_config: {error}"))?;
        if parsed.heads == 0 || !parsed.width.is_multiple_of(parsed.heads) {
            bail!(
                "clef head_config width {} is not divisible by heads {}",
                parsed.width,
                parsed.heads
            );
        }
        Ok(parsed)
    }
}

/// Upstream's `sanitize` for the head tensors: the PyTorch `in_proj_*`
/// packing split into query, key and value, and the PyTorch `Sequential`
/// index `.3.` renamed to `.1.`. Keys arrive with or without the `head.`
/// prefix and leave with it.
pub(crate) fn sanitize_head_weights(raw: WeightMap) -> Result<WeightMap> {
    let mut out = WeightMap::new();
    for (key, value) in raw {
        let key = format!("head.{}", key.strip_prefix("head.").unwrap_or(&key));
        if let Some((prefix, suffix)) = key.rsplit_once(".in_proj_")
            && (suffix == "weight" || suffix == "bias")
        {
            let rows = mlxcel_core::array_shape(&value)
                .first()
                .copied()
                .unwrap_or(0);
            if rows <= 0 || rows % 3 != 0 {
                bail!("clef head tensor {key} has {rows} rows, not a multiple of 3");
            }
            let part = rows / 3;
            let ndim = mlxcel_core::array_shape(&value).len();
            for (index, name) in ["query_proj", "key_proj", "value_proj"].iter().enumerate() {
                let mut start = vec![0; ndim];
                let mut stop = mlxcel_core::array_shape(&value);
                start[0] = part * index as i32;
                stop[0] = part * (index as i32 + 1);
                let piece = mlxcel_core::slice(&value, &start, &stop);
                out.insert(format!("{prefix}.{name}.{suffix}"), piece);
            }
            continue;
        }
        let key = key
            .replace("feedforward.3.", "feedforward.1.")
            .replace("residual_scorer.3.", "residual_scorer.1.");
        out.insert(key, value);
    }
    Ok(out)
}

/// One request's span layout, in sequence positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HeadInputs {
    /// `[start, end)` of each question's instruction.
    pub(crate) question_spans: Vec<(usize, usize)>,
    /// `[start, end)` of every option, questions in order.
    pub(crate) option_spans: Vec<(usize, usize)>,
    /// Option count of each question.
    pub(crate) counts: Vec<usize>,
    /// Type id of each question (`noul` 0, `choice` 1, `score` 2).
    pub(crate) types: Vec<i32>,
}

/// The loaded joint schema head.
pub(crate) struct JointSchemaHead {
    hidden_norm: LayerNorm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: UniquePtr<MlxArray>,
    evidence_layers: Vec<EvidenceRoutingLayer>,
    option_summary_norm: LayerNorm,
    layers: Vec<FieldDecoderLayer>,
    field_norm: LayerNorm,
    option_norm: LayerNorm,
    residual_scorer: [Linear; 2],
    prior_logit_scale: UniquePtr<MlxArray>,
    joint_logit_scale: UniquePtr<MlxArray>,
    residual_gate: UniquePtr<MlxArray>,
}

/// `[rows, length]` f32 averaging matrix of `spans` (upstream `_span_means`).
pub(crate) fn span_mean_weights(spans: &[(usize, usize)], length: usize) -> Vec<f32> {
    let mut weights = vec![0f32; spans.len() * length];
    for (row, &(start, end)) in spans.iter().enumerate() {
        let value = 1.0 / (end - start) as f32;
        weights[row * length + start..row * length + end].fill(value);
    }
    weights
}

/// `span_mean_weights(spans, L) @ x` in f32, cast to `out_dtype`.
pub(crate) fn span_means(
    x: &MlxArray,
    spans: &[(usize, usize)],
    out_dtype: i32,
) -> UniquePtr<MlxArray> {
    let length = mlxcel_core::array_shape(x)[0] as usize;
    let weights = mlxcel_core::from_slice_f32(
        &span_mean_weights(spans, length),
        &[spans.len() as i32, length as i32],
    );
    let product = mlxcel_core::matmul(&weights, &mlxcel_core::astype(x, dtype::FLOAT32));
    mlxcel_core::astype(&product, out_dtype)
}

pub(super) fn scalar_like(value: f32, dtype: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[value], &[]), dtype)
}

/// `x / max(||x||, eps)` along the last axis, in `x`'s dtype.
fn normalize(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let norm = mlxcel_core::linalg_norm(x, -1, true);
    let floor = scalar_like(eps, mlxcel_core::array_dtype(x));
    mlxcel_core::divide(x, &mlxcel_core::maximum(&norm, &floor))
}

/// `exp(min(scale, ln 100))`.
fn clamped_exp(scale: &MlxArray) -> UniquePtr<MlxArray> {
    let cap = scalar_like(100f32.ln(), mlxcel_core::array_dtype(scale));
    mlxcel_core::exp(&mlxcel_core::minimum(scale, &cap))
}

fn rows(x: &MlxArray, start: usize, end: usize) -> UniquePtr<MlxArray> {
    let mut lo = vec![0; mlxcel_core::array_shape(x).len()];
    let mut hi = mlxcel_core::array_shape(x);
    lo[0] = start as i32;
    hi[0] = end as i32;
    mlxcel_core::slice(x, &lo, &hi)
}

fn expand(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::expand_dims(x, 0)
}

fn first(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::squeeze_axis(x, 0)
}

impl JointSchemaHead {
    /// Build the head from sanitized `head.*` tensors.
    pub(crate) fn from_weights(weights: &WeightMap, config: &HeadConfig) -> Result<Self> {
        let w = Weights { map: weights };
        let p = |name: &str| format!("head.{name}");
        let evidence_layers = (0..config.routing_layers)
            .map(|i| {
                let prefix = p(&format!("evidence_layers.{i}"));
                Ok(EvidenceRoutingLayer {
                    query_norm: w.norm(&format!("{prefix}.query_norm"))?,
                    memory_norm: w.norm(&format!("{prefix}.memory_norm"))?,
                    attention: w.attention(&format!("{prefix}.attention"), config.heads)?,
                    feedforward_norm: w.norm(&format!("{prefix}.feedforward_norm"))?,
                    feedforward: [
                        w.linear(&format!("{prefix}.feedforward.0"), true)?,
                        w.linear(&format!("{prefix}.feedforward.1"), true)?,
                    ],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..config.layers)
            .map(|i| {
                let prefix = p(&format!("layers.{i}"));
                Ok(FieldDecoderLayer {
                    self_attn: w.attention(&format!("{prefix}.self_attn"), config.heads)?,
                    multihead_attn: w
                        .attention(&format!("{prefix}.multihead_attn"), config.heads)?,
                    linear1: w.linear(&format!("{prefix}.linear1"), true)?,
                    linear2: w.linear(&format!("{prefix}.linear2"), true)?,
                    norm1: w.norm(&format!("{prefix}.norm1"))?,
                    norm2: w.norm(&format!("{prefix}.norm2"))?,
                    norm3: w.norm(&format!("{prefix}.norm3"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let hidden_norm = w.norm(&p("hidden_norm"))?;
        let width = mlxcel_core::array_shape(&hidden_norm.weight)
            .first()
            .copied()
            .unwrap_or(0);
        if width as usize != config.hidden_size {
            bail!(
                "clef head hidden_norm has {width} features, head_config.hidden_size is {}",
                config.hidden_size
            );
        }
        Ok(Self {
            hidden_norm,
            memory_projection: w.linear(&p("memory_projection"), false)?,
            question_projection: w.linear(&p("question_projection"), false)?,
            option_question_projection: w.linear(&p("option_question_projection"), false)?,
            global_projection: w.linear(&p("global_projection"), false)?,
            option_context_projection: w.linear(&p("option_context_projection"), false)?,
            option_lexical_projection: w.linear(&p("option_lexical_projection"), false)?,
            type_embedding: w.tensor(&p("type_embedding.weight"))?,
            evidence_layers,
            option_summary_norm: w.norm(&p("option_summary_norm"))?,
            layers,
            field_norm: w.norm(&p("field_norm"))?,
            option_norm: w.norm(&p("option_norm"))?,
            residual_scorer: [
                w.linear(&p("residual_scorer.0"), true)?,
                w.linear(&p("residual_scorer.1"), true)?,
            ],
            prior_logit_scale: w.tensor(&p("prior_logit_scale"))?,
            joint_logit_scale: w.tensor(&p("joint_logit_scale"))?,
            residual_gate: w.tensor(&p("residual_gate"))?,
        })
    }

    /// `hidden_norm` applied to the backbone's post-norm hidden `[L, H]`.
    pub(crate) fn normalize_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.hidden_norm.forward(hidden)
    }

    /// The option logits `[options]` for `hidden` (already through
    /// [`Self::normalize_hidden`]) and the lexical vectors `[options, H]` in
    /// the hidden dtype.
    pub(crate) fn forward(
        &self,
        hidden: &MlxArray,
        inputs: &HeadInputs,
        lexical: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(hidden);
        let length = mlxcel_core::array_shape(hidden)[0];
        let memory = expand(&self.memory_projection.forward(hidden));
        let global_vector = first(&rows(hidden, length as usize - 1, length as usize));
        let questions = span_means(hidden, &inputs.question_spans, dtype);
        let contexts = span_means(hidden, &inputs.option_spans, dtype);
        let owner_ids: Vec<i32> = inputs
            .counts
            .iter()
            .enumerate()
            .flat_map(|(question, &count)| std::iter::repeat_n(question as i32, count))
            .collect();
        let owner = mlxcel_core::from_slice_i32(&owner_ids, &[owner_ids.len() as i32]);

        let routed = mlxcel_core::add(
            &mlxcel_core::add(
                &self.option_context_projection.forward(&contexts),
                &self.option_lexical_projection.forward(lexical),
            ),
            &mlxcel_core::take(
                &self.option_question_projection.forward(&questions),
                &owner,
                0,
            ),
        );
        let mut routed = expand(&routed);
        for layer in &self.evidence_layers {
            routed = layer.forward(&routed, &memory);
        }
        let routed = first(&routed);

        let base = self.question_projection.forward(&questions);
        let width = *mlxcel_core::array_shape(&routed).last().unwrap_or(&1);
        let scores = mlxcel_core::divide(
            &mlxcel_core::sum_axis(
                &mlxcel_core::multiply(&routed, &mlxcel_core::take(&base, &owner, 0)),
                -1,
                false,
            ),
            &scalar_like((width as f32).sqrt(), dtype),
        );
        let mut summaries = Vec::with_capacity(inputs.counts.len());
        let mut start = 0usize;
        for &count in &inputs.counts {
            let end = start + count;
            let weights = mlxcel_core::softmax_precise(&rows(&scores, start, end), -1);
            let weighted = mlxcel_core::multiply(
                &mlxcel_core::expand_dims(&weights, 1),
                &rows(&routed, start, end),
            );
            summaries.push(mlxcel_core::sum_axis(&weighted, 0, false));
            start = end;
        }
        let summaries = mlxcel_core::stack_owned(&summaries, 0);
        let types = mlxcel_core::from_slice_i32(&inputs.types, &[inputs.types.len() as i32]);
        let fields = mlxcel_core::add(
            &mlxcel_core::add(
                &mlxcel_core::add(&base, &self.option_summary_norm.forward(&summaries)),
                &self.global_projection.forward(&global_vector),
            ),
            &mlxcel_core::take(&self.type_embedding, &types, 0),
        );
        let mut fields = expand(&fields);
        for layer in &self.layers {
            fields = layer.forward(&fields, &memory);
        }
        let fields = mlxcel_core::take(&self.field_norm.forward(&first(&fields)), &owner, 0);

        let anchor = normalize(&mlxcel_core::add(&questions, &global_vector), 1e-12);
        let lexical = normalize(lexical, 1e-12);
        let prior = mlxcel_core::multiply(
            &clamped_exp(&self.prior_logit_scale),
            &mlxcel_core::sum_axis(
                &mlxcel_core::multiply(&lexical, &mlxcel_core::take(&anchor, &owner, 0)),
                -1,
                false,
            ),
        );
        let options = self.option_norm.forward(&routed);
        let cosine = mlxcel_core::divide(
            &mlxcel_core::sum_axis(&mlxcel_core::multiply(&fields, &options), -1, false),
            &mlxcel_core::maximum(
                &mlxcel_core::multiply(
                    &mlxcel_core::linalg_norm(&fields, -1, false),
                    &mlxcel_core::linalg_norm(&options, -1, false),
                ),
                &scalar_like(1e-8, dtype),
            ),
        );
        let product = mlxcel_core::multiply(&fields, &options);
        let gap = mlxcel_core::abs(&mlxcel_core::subtract(&fields, &options));
        let features = mlxcel_core::concatenate_many(&[&fields, &options, &product, &gap], -1);
        let residual = mlxcel_core::squeeze_axis(
            &self.residual_scorer[1].forward(&gelu(&self.residual_scorer[0].forward(&features))),
            -1,
        );
        let joint = mlxcel_core::add(
            &mlxcel_core::multiply(&clamped_exp(&self.joint_logit_scale), &cosine),
            &residual,
        );
        mlxcel_core::add(
            &prior,
            &mlxcel_core::multiply(&mlxcel_core::sigmoid(&self.residual_gate), &joint),
        )
    }
}
