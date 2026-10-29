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

//! The Ming-Image conditioner: prompt to `cap_feats` and `cap_feats_2`.
//!
//! A port of `MingImageTextEncoder` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ming_image/text_encoder.py>
//! and the prompt handling of `pipeline.py` (mlx-vlm `73d366ec`):
//!
//! 1. The prompt is rendered as one user turn with the checkpoint's chat
//!    template (`add_generation_prompt`), tokenized without special tokens,
//!    and followed by `<image_start>`, `query_token_count` `<image_patch>`
//!    tokens and `<image_end>`.
//! 2. The patch span's embeddings are replaced by the learned
//!    `query_tokens`, and the BailingMoeV2 backbone runs causally with each
//!    MoE layer routing the patch tokens through its `image_gate`
//!    ([`crate::models::bailing_moe::image_router`]).
//! 3. `cap_feats`: the final normed state at the patch span, through
//!    `proj_in`, the bidirectional Qwen2 connector (`mask = None`) and
//!    `proj_out`.
//! 4. `cap_feats_2`: the hidden states of `selected_hidden_states_layers`
//!    concatenated at the text positions (every token outside the block),
//!    through `directvlm_norm` and `directvlm_proj`.
//!
//! Dtypes follow upstream: the backbone computes in its bf16, the bridge and
//! connector in the f32 they are stored in (the bf16 states are widened
//! explicitly, since the CUDA build of mlxcel's MLX would otherwise round a
//! bf16 by f32 product to bf16), and both outputs are returned in bf16, as
//! the pipeline casts them.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::json;

use super::config::{BlockTokens, BridgeConfig, MingImageConfig};
use super::weights::TextEncoderWeights;
use crate::imagegen::pipeline::InvalidImageRequest;
use crate::models::bailing_moe::image_router::ImageRouters;
use crate::models::bailing_moe::{BailingMoeModel, Quantization};
use crate::models::llama3::TransformerBlock;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// `nn.RMSNorm` default epsilon of `directvlm_norm`.
const DIRECTVLM_EPS: f32 = 1e-5;

fn eval(x: &MlxArray) -> Result<()> {
    mlxcel_core::try_eval(x).map_err(|e| anyhow::anyhow!("Ming-Image conditioner: {e}"))
}

/// Tokenizer and chat template of `<model>/mllm/`.
pub struct PromptFormatter {
    tokenizer: MlxcelTokenizer,
    template: ChatTemplateProcessor,
    tokens: BlockTokens,
    query_tokens: usize,
}

impl PromptFormatter {
    pub fn load(mllm_dir: &Path, config: &MingImageConfig) -> Result<Self> {
        let tokenizer = load_tokenizer(mllm_dir)
            .with_context(|| format!("cannot load tokenizer from {}", mllm_dir.display()))?;
        let template = ChatTemplateProcessor::from_model_path(mllm_dir)?
            .with_context(|| format!("{} ships no chat template", mllm_dir.display()))?;
        Ok(Self {
            tokenizer,
            template,
            tokens: config.tokens,
            query_tokens: config.bridge.query_token_count as usize,
        })
    }

    /// Upstream `_prompt_ids`: the templated prompt, tokenized without
    /// special tokens. A prompt whose ids already contain one of the image
    /// block tokens is rejected, since upstream would locate the query span
    /// there instead.
    pub fn prompt_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let messages = json!([{ "role": "user", "content": prompt }]);
        let text =
            self.template
                .apply_raw_with_kwargs(&messages, None, &ChatTemplateKwargs::new())?;
        let ids = self
            .tokenizer
            .encode(&text, false)?
            .into_iter()
            .map(|id| i32::try_from(id).context("token id exceeds i32"))
            .collect::<Result<Vec<_>>>()?;
        let t = self.tokens;
        if ids
            .iter()
            .any(|&id| id == t.patch || id == t.start || id == t.end)
        {
            return Err(InvalidImageRequest(
                "the Ming-Image prompt must not contain the image block tokens".into(),
            )
            .into());
        }
        Ok(ids)
    }

    /// Upstream `_tokenize`: the prompt ids plus the image block.
    pub fn input_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        let mut ids = self.prompt_ids(prompt)?;
        ids.push(self.tokens.start);
        ids.extend(std::iter::repeat_n(self.tokens.patch, self.query_tokens));
        ids.push(self.tokens.end);
        Ok(ids)
    }
}

/// Where the query span starts and which positions are text, as upstream's
/// `encode` derives them; an id list without exactly one contiguous span of
/// `count` patch tokens is an error.
pub fn query_layout(ids: &[i32], tokens: BlockTokens, count: usize) -> Result<(usize, Vec<i32>)> {
    let Some(start) = ids.iter().position(|&id| id == tokens.patch) else {
        bail!("the Ming-Image input ids have no image-patch span after templating");
    };
    let patches = ids.iter().filter(|&&id| id == tokens.patch).count();
    ensure!(
        patches == count
            && ids[start..start + count]
                .iter()
                .all(|&id| id == tokens.patch),
        "the Ming-Image input ids must hold one contiguous span of {count} image-patch tokens, found {patches}"
    );
    let text = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| ![tokens.patch, tokens.start, tokens.end].contains(id))
        .map(|(i, _)| i as i32)
        .collect();
    Ok((start, text))
}

/// The two caption streams, bf16.
pub struct Encoded {
    /// `[1, query_token_count, cap_feat_dim]`.
    pub cap_feats: UniquePtr<MlxArray>,
    /// `[1, text_tokens, directvlm_dim]`.
    pub cap_feats_2: UniquePtr<MlxArray>,
}

/// The conditioner's intermediates, for the parity tests.
pub struct EncodedStates {
    pub encoded: Encoded,
    /// The final normed state at the query span, `[1, count, hidden]`.
    pub query_hidden: UniquePtr<MlxArray>,
    /// The concatenated selected states at the text positions.
    pub text_states: UniquePtr<MlxArray>,
}

/// The loaded conditioner.
pub struct TextEncoder {
    formatter: Option<PromptFormatter>,
    mllm: BailingMoeModel,
    routers: ImageRouters,
    connector: Vec<TransformerBlock>,
    connector_norm: RMSNorm,
    query_tokens: UniquePtr<MlxArray>,
    proj_in: UnifiedLinear,
    proj_out: UnifiedLinear,
    directvlm_norm: RMSNorm,
    directvlm_proj: UnifiedLinear,
    bridge: BridgeConfig,
    tokens: BlockTokens,
}

fn tensor(w: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    w.get(key)
        .map(|t| mlxcel_core::copy(t))
        .with_context(|| format!("Ming-Image conditioner weight not found: {key}"))
}

fn linear(w: &WeightMap, prefix: &str, quant: (i32, i32)) -> Result<UnifiedLinear> {
    UnifiedLinear::from_weights(w, prefix, quant.0, quant.1)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("Ming-Image conditioner weight {prefix}"))
}

impl TextEncoder {
    /// Build from the weights [`super::weights::load_text_encoder`] read.
    pub fn new(root: &Path, config: &MingImageConfig, weights: TextEncoderWeights) -> Result<Self> {
        let formatter = PromptFormatter::load(&root.join("mllm"), config)?;
        Self::build(Some(formatter), config, weights)
    }

    /// Build from in-memory weights; without a formatter only
    /// [`Self::encode_ids`] works.
    pub fn build(
        formatter: Option<PromptFormatter>,
        config: &MingImageConfig,
        weights: TextEncoderWeights,
    ) -> Result<Self> {
        let TextEncoderWeights {
            mllm,
            connector,
            bridge,
            quant,
        } = weights;
        let q = quant.as_ref().map_or((64, 4), |s| (s.group_size, s.bits));
        let mut mllm_args = config.mllm.clone();
        let mut connector_args = config.connector.clone();
        if let Some(spec) = &quant {
            mllm_args.quantization = Some(Quantization {
                group_size: spec.group_size,
                bits: spec.bits,
            });
            connector_args.quantization = Some(crate::models::llama3::Quantization {
                group_size: spec.group_size,
                bits: spec.bits,
            });
        }
        let routers = ImageRouters::from_weights(&mllm, &mllm_args).map_err(anyhow::Error::msg)?;
        let backbone = BailingMoeModel::from_weights(&mllm, &mllm_args)
            .map_err(anyhow::Error::msg)
            .context("cannot build the Ming-Image backbone")?;
        drop(mllm);
        let connector_blocks = (0..connector_args.num_hidden_layers)
            .map(|i| {
                TransformerBlock::from_weights(&connector, &connector_args, i)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("Ming-Image connector layer {i}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let query_tokens = tensor(&bridge, "query_tokens")?;
        let qshape = mlxcel_core::array_shape(&query_tokens);
        ensure!(
            qshape
                == [
                    config.bridge.query_token_count,
                    config.mllm.hidden_size as i32
                ],
            "Ming-Image query_tokens are {qshape:?}, expected [{}, {}]",
            config.bridge.query_token_count,
            config.mllm.hidden_size
        );
        Ok(Self {
            formatter,
            mllm: backbone,
            routers,
            connector: connector_blocks,
            connector_norm: RMSNorm::new(
                tensor(&connector, "model.norm.weight")?,
                connector_args.rms_norm_eps,
            ),
            query_tokens,
            proj_in: linear(&bridge, "proj_in", q)?,
            proj_out: linear(&bridge, "proj_out", q)?,
            directvlm_norm: RMSNorm::new(tensor(&bridge, "directvlm_norm.weight")?, DIRECTVLM_EPS),
            directvlm_proj: linear(&bridge, "directvlm_proj", q)?,
            bridge: config.bridge.clone(),
            tokens: config.tokens,
        })
    }

    /// The tokenizer and template, `None` for an encoder built without one.
    #[must_use]
    pub fn formatter(&self) -> Option<&PromptFormatter> {
        self.formatter.as_ref()
    }

    /// Run the bidirectional connector over `x` (`[1, L, D]`).
    fn connector(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // `mask = None` upstream is unmasked attention; the shared Qwen2 block
        // reads a missing mask on a multi-token step as causal, so an all-zero
        // additive mask (`x + 0 == x`) carries the bidirectional semantics.
        let l = mlxcel_core::array_shape(x)[1];
        let zeros = mlxcel_core::full_f32(&[1, 1, l, l], 0.0, mlxcel_core::array_dtype(x));
        let mut h = mlxcel_core::copy(x);
        for block in &self.connector {
            let mut cache = KVCache::new();
            h = block.forward(&h, &mut cache, Some(&zeros));
        }
        self.connector_norm.forward(&h)
    }

    /// Encode `prompt` (upstream `encode` on `_tokenize(prompt)`).
    pub fn encode(&self, prompt: &str) -> Result<Encoded> {
        let formatter = self
            .formatter
            .as_ref()
            .context("this Ming-Image text encoder has no tokenizer")?;
        Ok(self.encode_ids(&formatter.input_ids(prompt)?)?.encoded)
    }

    /// Every backbone hidden state (`num_layers + 1`) of `ids`, the query
    /// span replaced by the learned tokens; `image_routing = false` routes the
    /// query tokens with the text router too (a diagnostic arm).
    pub fn backbone_states(
        &self,
        ids: &[i32],
        image_routing: bool,
    ) -> Result<Vec<UniquePtr<MlxArray>>> {
        let count = self.bridge.query_token_count as usize;
        let (start, _) = query_layout(ids, self.tokens, count)?;
        let len = ids.len() as i32;
        let input_ids = mlxcel_core::from_slice_i32(ids, &[1, len]);
        let embeds = self.mllm.word_embeddings.forward(&input_ids);
        let hidden = *mlxcel_core::array_shape(&embeds).last().context("rank")?;
        let (s, c) = (start as i32, count as i32);
        let query = mlxcel_core::astype(&self.query_tokens, mlxcel_core::array_dtype(&embeds));
        let query = mlxcel_core::reshape(&query, &[1, c, hidden]);
        let head = mlxcel_core::slice(&embeds, &[0, 0, 0], &[1, s, hidden]);
        let tail = mlxcel_core::slice(&embeds, &[0, s + c, 0], &[1, len, hidden]);
        let embeds = mlxcel_core::concatenate_many(&[&head, &query, &tail], 1);
        let mask: Vec<i32> = ids
            .iter()
            .map(|&id| i32::from(id == self.tokens.patch))
            .collect();
        let image_mask =
            mlxcel_core::astype(&mlxcel_core::from_slice_i32(&mask, &[1, len]), dtype::BOOL);
        let routing = image_routing.then_some((&self.routers, &*image_mask));
        self.mllm
            .hidden_states_from_embeds(&embeds, routing)
            .map_err(anyhow::Error::msg)
    }

    /// Backbone layer `layer` alone on hidden state `h` (`[1, L, hidden]`) of
    /// `ids`, routed as [`Self::backbone_states`] routes it: a per-layer
    /// parity hook that feeds upstream's own state in.
    pub fn backbone_layer(
        &self,
        layer: usize,
        h: &MlxArray,
        ids: &[i32],
    ) -> Result<UniquePtr<MlxArray>> {
        let block = self.mllm.layers.get(layer).context("layer out of range")?;
        let len = ids.len() as i32;
        let mask: Vec<i32> = ids
            .iter()
            .map(|&id| i32::from(id == self.tokens.patch))
            .collect();
        let image_mask =
            mlxcel_core::astype(&mlxcel_core::from_slice_i32(&mask, &[1, len]), dtype::BOOL);
        let mut cache = KVCache::new();
        Ok(block.forward_image_routed(h, &mut cache, None, self.routers.get(layer), &image_mask))
    }

    /// The backbone's final RMSNorm alone (a per-stage parity hook).
    pub fn backbone_norm(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        self.mllm.norm.forward(h)
    }

    /// [`Self::encode`] on explicit input ids, keeping the intermediates.
    pub fn encode_ids(&self, ids: &[i32]) -> Result<EncodedStates> {
        self.encode_ids_routed(ids, true)
    }

    /// [`Self::encode_ids`] with the query tokens routed by the image router
    /// (`true`, the model) or by the text router (`false`, the broken arm the
    /// parity tests check the bounds against).
    pub fn encode_ids_routed(&self, ids: &[i32], image_routing: bool) -> Result<EncodedStates> {
        let count = self.bridge.query_token_count as usize;
        let (start, text_positions) = query_layout(ids, self.tokens, count)?;
        let states = self.backbone_states(ids, image_routing)?;
        let hidden = *mlxcel_core::array_shape(&states[0])
            .last()
            .context("rank")?;
        let (s, c) = (start as i32, count as i32);
        let last = states.last().context("the backbone returned no states")?;
        let query_hidden = mlxcel_core::slice(last, &[0, s, 0], &[1, s + c, hidden]);
        let widened = mlxcel_core::astype(&query_hidden, dtype::FLOAT32);
        let cap_feats = self
            .proj_out
            .forward(&self.connector(&self.proj_in.forward(&widened)));

        let selected = self
            .bridge
            .selected_layers
            .iter()
            .map(|&l| {
                states
                    .get(l)
                    .map(|s| &**s)
                    .context("selected layer out of range")
            })
            .collect::<Result<Vec<&MlxArray>>>()?;
        let stacked = mlxcel_core::concatenate_many(&selected, -1);
        let positions =
            mlxcel_core::from_slice_i32(&text_positions, &[text_positions.len() as i32]);
        let text_states = mlxcel_core::take(&stacked, &positions, 1);
        let cap_feats_2 = self.directvlm_proj.forward(
            &self
                .directvlm_norm
                .forward(&mlxcel_core::astype(&text_states, dtype::FLOAT32)),
        );
        let encoded = Encoded {
            cap_feats: mlxcel_core::astype(&cap_feats, dtype::BFLOAT16),
            cap_feats_2: mlxcel_core::astype(&cap_feats_2, dtype::BFLOAT16),
        };
        for a in [
            &encoded.cap_feats,
            &encoded.cap_feats_2,
            &query_hidden,
            &text_states,
        ] {
            eval(a)?;
        }
        Ok(EncodedStates {
            encoded,
            query_hidden,
            text_states,
        })
    }
}

/// Upstream's `count_prompt_tokens`: the templated prompt's length.
pub fn count_prompt_tokens(formatter: &PromptFormatter, prompt: &str) -> Result<usize> {
    Ok(formatter.prompt_ids(prompt)?.len())
}

#[cfg(test)]
#[path = "text_encoder_tests.rs"]
mod tests;
