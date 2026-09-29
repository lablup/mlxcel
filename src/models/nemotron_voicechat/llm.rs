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

//! Duplex language model of Nemotron VoiceChat: a Nemotron-H backbone that
//! consumes one fused embedding row per 80 ms timeline position and emits a
//! text token and a function-channel token from two heads.
//!
//! Port of `DuplexSTTModel` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/model.py
//! reusing [`NemotronHModel`]: the checkpoint's `stt_model.llm.*`,
//! `stt_model.embed_tokens.*` and `stt_model.lm_head.*` are renamed onto the
//! Nemotron-H loader's `backbone.*` / `lm_head.*` layout, and
//! `stt_model.function_head.*` is loaded beside it as a second head.
//!
//! Used by: Nemotron VoiceChat offline session and streaming session

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::NemotronHModel;
use crate::models::nemotron_h::{BlockType, NemotronHConfig, NemotronLayerCache, Quantization};

/// Prefix renames from the VoiceChat checkpoint onto the Nemotron-H loader.
const LLM_RENAMES: &[(&str, &str)] = &[
    ("stt_model.llm.layers.", "backbone.layers."),
    ("stt_model.llm.norm_f.", "backbone.norm_f."),
    ("stt_model.embed_tokens.", "backbone.embeddings."),
    ("stt_model.lm_head.", "lm_head."),
];

/// Nemotron-H with a fused input and two greedy heads.
pub struct VoiceChatLanguageModel {
    llm: NemotronHModel,
    function_head: UnifiedLinear,
    function_channel_weight: f32,
}

/// Greedy outputs of one timeline position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DuplexTokens {
    pub text: i32,
    pub function: i32,
}

/// Rename the language-model keys of a VoiceChat checkpoint onto the
/// Nemotron-H layout, moving them out of `weights`. The function head keys
/// keep their original `stt_model.function_head.` prefix.
pub(crate) fn take_llm_weights(weights: &mut WeightMap) -> WeightMap {
    let keys: Vec<String> = weights.keys().cloned().collect();
    let mut out = WeightMap::new();
    for key in keys {
        let renamed = LLM_RENAMES
            .iter()
            .find_map(|(from, to)| key.strip_prefix(from).map(|rest| format!("{to}{rest}")))
            .or_else(|| {
                key.starts_with("stt_model.function_head.")
                    .then(|| key.clone())
            });
        if let Some(new_key) = renamed
            && let Some(value) = weights.remove(&key)
        {
            out.insert(new_key, value);
        }
    }
    out
}

impl VoiceChatLanguageModel {
    /// Build from the VoiceChat `text_config` and the checkpoint weights.
    ///
    /// `quantization` is the checkpoint's top-level default `(group_size,
    /// bits)`; quantized versus dense is still resolved per module from the
    /// presence of `.scales`, so dense checkpoints load unchanged.
    pub fn from_weights(
        mut text_config: NemotronHConfig,
        quantization: Option<(i32, i32)>,
        weights: &mut WeightMap,
        function_channel_weight: f32,
    ) -> Result<Self, String> {
        text_config.post_init()?;
        if text_config.quantization.is_none()
            && let Some((group_size, bits)) = quantization
        {
            text_config.quantization = Some(Quantization { group_size, bits });
        }
        let pattern = text_config
            .hybrid_override_pattern
            .clone()
            .ok_or("nemotron_voicechat: text_config.hybrid_override_pattern is required")?;
        let block_types: Vec<BlockType> = pattern
            .iter()
            .map(|name| BlockType::from_str(name))
            .collect();
        let group_size = text_config.group_size();
        let bits = text_config.bits();

        let mut llm_weights = take_llm_weights(weights);
        let function_head =
            UnifiedLinear::from_weights(&llm_weights, "stt_model.function_head", group_size, bits)
                .map_err(|e| format!("nemotron_voicechat: function_head: {e}"))?;
        llm_weights.retain(|k, _| !k.starts_with("stt_model.function_head."));

        let llm_weights = NemotronHModel::sanitize_weights(llm_weights, &text_config);
        let llm = NemotronHModel::from_weights(text_config, llm_weights, block_types)
            .map_err(|e| format!("nemotron_voicechat: language model: {e}"))?;
        Ok(Self {
            llm,
            function_head,
            function_channel_weight,
        })
    }

    /// Fresh per-session Nemotron-H caches (Mamba state + attention KV).
    pub fn make_caches(&self) -> Vec<NemotronLayerCache> {
        self.llm.make_caches()
    }

    /// The wrapped Nemotron-H model (text head only), for the generic
    /// `LanguageModel` delegation.
    pub fn backbone(&self) -> &NemotronHModel {
        &self.llm
    }

    /// Hidden size of the backbone (4480 for the released checkpoints).
    pub fn hidden_size(&self) -> usize {
        self.llm.hidden_size()
    }

    /// Token embeddings `[1, ids.len(), hidden]` from the shared table.
    pub fn embed(&self, ids: &[i32]) -> UniquePtr<MlxArray> {
        let arr = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
        self.llm.input_embeddings(&arr)
    }

    /// `E(prev_text) + audio + w * E(prev_function)` for one position.
    ///
    /// The scalar weight is applied in the embedding dtype before the sum,
    /// matching the reference's Python-scalar multiply, so the result dtype
    /// follows MLX promotion with `audio` (float32 for the perception path).
    pub fn fused_input(
        &self,
        prev_text: i32,
        audio: &MlxArray,
        prev_function: i32,
    ) -> UniquePtr<MlxArray> {
        let text = self.embed(&[prev_text]);
        let function = self.embed(&[prev_function]);
        let weight = mlxcel_core::full_like(&function, self.function_channel_weight);
        let function = mlxcel_core::multiply(&function, &weight);
        let sum = mlxcel_core::add(&text, audio);
        mlxcel_core::add(&sum, &function)
    }

    /// Run the backbone on `inputs_embeds [1, L, hidden]` with `caches` and
    /// return the greedy text and function ids of the last position.
    pub fn step(
        &self,
        inputs_embeds: &MlxArray,
        caches: &mut [NemotronLayerCache],
    ) -> DuplexTokens {
        let hidden = self.llm.forward_embeds_to_hidden(inputs_embeds, caches);
        crate::audio::stage_probe::mark("language.backbone", &[&hidden]);
        let shape = mlxcel_core::array_shape(&hidden);
        let last = if shape[1] > 1 {
            mlxcel_core::slice(
                &hidden,
                &[0, shape[1] - 1, 0],
                &[shape[0], shape[1], shape[2]],
            )
        } else {
            hidden
        };
        let text_logits = self.llm.apply_lm_head(&last);
        let function_logits = self.function_head.forward(&last);
        let text = mlxcel_core::argmax(&text_logits, -1, false);
        let function = mlxcel_core::argmax(&function_logits, -1, false);
        crate::audio::stage_probe::mark("language.heads", &[&text, &function]);
        let tokens = DuplexTokens {
            text: mlxcel_core::item_i32(&text),
            function: mlxcel_core::item_i32(&function),
        };
        crate::audio::stage_probe::count_sync(2);
        tokens
    }
}
