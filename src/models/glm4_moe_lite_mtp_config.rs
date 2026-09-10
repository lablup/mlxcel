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

//! Configuration for the GLM-4.7-Flash MTP drafter
//! (`model_type: "glm4_moe_lite_mtp"`), the directory `mlxcel split-mtp`
//! writes (issue #1326).
//!
//! Same shape as the Qwen 3.5 MTP drafter config: a small top level
//! (`block_size`, `tie_word_embeddings`, the optional `quantization` block)
//! plus a `text_config` that is the source checkpoint's own config minus its
//! quantization keys. Because the drafter reuses the `glm4_moe_lite` decoder
//! block verbatim, `text_config` deserializes straight into the target's
//! [`ModelArgs`].

use serde::Deserialize;

use super::glm4_moe_lite::ModelArgs;

/// `model_type` that selects this drafter, shared with the core registry.
pub const GLM4_MOE_LITE_MTP_MODEL_TYPE: &str = mlxcel_core::drafter::GLM4_MOE_LITE_MTP_MODEL_TYPE;

/// `model_type` the nested `text_config` must declare.
pub const TARGET_MODEL_TYPE: &str = "glm4_moe_lite";

/// Affine quantization block written by `split-mtp --q-bits`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Glm4MoeLiteMtpQuantization {
    pub group_size: i32,
    pub bits: i32,
}

/// Drafter config for the GLM-4.7-Flash MTP block.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm4MoeLiteMtpConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Total verify-round budget including the bonus token. `None` resolves
    /// to `num_nextn_predict_layers + 1` in [`Self::normalize`] (2 on
    /// GLM-4.7-Flash: one draft token plus the bonus).
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Always false for this drafter: the nextn block carries its own
    /// `embed_tokens` and `shared_head.head`. Parsed for parity with the
    /// other MTP configs.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub text_config: Option<ModelArgs>,
    #[serde(default)]
    pub quantization: Option<Glm4MoeLiteMtpQuantization>,
}

fn default_model_type() -> String {
    GLM4_MOE_LITE_MTP_MODEL_TYPE.to_string()
}

impl Glm4MoeLiteMtpConfig {
    /// Validate and apply the post-parse fixups:
    ///
    /// - `text_config` must be present and declare `glm4_moe_lite`;
    /// - the head geometry the block reshapes on must be non-zero (a zero
    ///   reaches MLX as a reshape extent and aborts rather than errors);
    /// - the `quantization` block, when present, is copied into the
    ///   `text_config` `group_size` / `bits` the decoder-block loader reads;
    /// - `block_size` defaults to `num_nextn_predict_layers + 1` and must be
    ///   at least 2 (a block of 1 drafts nothing).
    pub fn normalize(mut self) -> Result<Self, String> {
        if self.model_type != GLM4_MOE_LITE_MTP_MODEL_TYPE {
            return Err(format!(
                "glm4_moe_lite_mtp drafter: config.json declares model_type {:?}, expected \
                 {GLM4_MOE_LITE_MTP_MODEL_TYPE:?}",
                self.model_type
            ));
        }
        let text = self
            .text_config
            .as_mut()
            .ok_or_else(|| "glm4_moe_lite_mtp drafter: text_config must be set".to_string())?;
        if text.model_type != TARGET_MODEL_TYPE {
            return Err(format!(
                "glm4_moe_lite_mtp drafter: text_config.model_type {:?} is not \
                 {TARGET_MODEL_TYPE:?}; the drafter reuses that family's decoder block and \
                 pairs only with a {TARGET_MODEL_TYPE} target",
                text.model_type
            ));
        }
        for (name, value) in [
            ("hidden_size", text.hidden_size),
            ("vocab_size", text.vocab_size),
            ("num_attention_heads", text.num_attention_heads),
            ("kv_lora_rank", text.kv_lora_rank),
            ("qk_nope_head_dim", text.qk_nope_head_dim),
            ("qk_rope_head_dim", text.qk_rope_head_dim),
            ("v_head_dim", text.v_head_dim),
        ] {
            if value == 0 {
                return Err(format!(
                    "glm4_moe_lite_mtp drafter: text_config.{name} must be > 0; it is a reshape \
                     extent or a projection width in the MTP block, so zero is an uncatchable \
                     abort at first draft rather than a load error"
                ));
            }
        }
        if text.n_routed_experts.is_some_and(|n| n > 0) && text.num_experts_per_tok == 0 {
            return Err(
                "glm4_moe_lite_mtp drafter: text_config.num_experts_per_tok must be > 0 when \
                 n_routed_experts is set; the router selects that many experts per token"
                    .to_string(),
            );
        }
        if let Some(q) = &self.quantization {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("glm4_moe_lite_mtp drafter: quantization: {e}"))?;
            text.group_size = Some(q.group_size);
            text.bits = Some(q.bits);
        }
        let default_block = text.num_nextn_predict_layers.saturating_add(1);
        let block_size = self.block_size.unwrap_or(default_block);
        if block_size < 2 {
            // Naming the source keeps a hand-written config actionable: with
            // neither key present the default lands at 1 and the message
            // would otherwise blame a `block_size` nobody wrote.
            let source = if self.block_size.is_some() {
                "block_size"
            } else {
                "text_config.num_nextn_predict_layers + 1 (no top-level block_size)"
            };
            return Err(format!(
                "glm4_moe_lite_mtp drafter: block_size {block_size} from {source} drafts \
                 nothing; the verify block must hold at least one draft token plus the bonus (2)"
            ));
        }
        self.block_size = Some(block_size);
        Ok(self)
    }

    /// Nested text config accessor. Call after [`Self::normalize`].
    pub fn text_config(&self) -> &ModelArgs {
        self.text_config
            .as_ref()
            .expect("Glm4MoeLiteMtpConfig.text_config must be set (call normalize first)")
    }

    /// Resolved total round budget (bonus token included). Call after
    /// [`Self::normalize`].
    pub fn block_size(&self) -> usize {
        self.block_size
            .expect("Glm4MoeLiteMtpConfig.block_size resolved by normalize")
    }

    /// The block width the trained depth supports:
    /// `min(block_size, num_nextn_predict_layers + 1)`, floored at 2.
    ///
    /// This is what the drafter reports as its configured depth. A larger
    /// `--draft-block-size` is still honored (the drafter prefers the
    /// requested width), at the cost of drafting past the depth the head was
    /// trained for.
    pub fn runtime_block_size(&self) -> usize {
        let trained = self
            .text_config()
            .num_nextn_predict_layers
            .saturating_add(1);
        self.block_size().min(trained).max(2)
    }

    /// Whether the drafter directory declares affine quantization.
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }
}
