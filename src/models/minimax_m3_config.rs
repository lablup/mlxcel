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

//! MiniMax-M3 serde config (`ModelArgs`) and the derived layer-plan helpers.
//!
//! Parses a FLAT text config (`model_type: "minimax_m3"`), the shape a
//! text-only export / community conversion uses. A future VL wrapper (#764)
//! constructs the same [`ModelArgs`] from the checkpoint's nested `text_config`
//! block, so every field has a serde default keyed to the real checkpoint value.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SparseAttentionConfig {
    #[serde(default)]
    pub use_sparse_attention: bool,
    #[serde(default = "default_sparse_index_dim")]
    pub sparse_index_dim: usize,
    #[serde(default = "default_sparse_num_index_heads")]
    pub sparse_num_index_heads: usize,
    #[serde(default = "default_sparse_topk_blocks")]
    pub sparse_topk_blocks: usize,
    #[serde(default = "default_sparse_block_size")]
    pub sparse_block_size: usize,
    #[serde(default = "default_sparse_score_type")]
    pub sparse_score_type: String,
    #[serde(default)]
    pub sparse_init_block: usize,
    #[serde(default = "default_sparse_local_block")]
    pub sparse_local_block: usize,
    #[serde(default)]
    pub sparse_attention_freq: Vec<usize>,
    #[serde(default)]
    pub sparse_disable_index_value: Vec<usize>,
}

fn default_sparse_index_dim() -> usize {
    128
}
fn default_sparse_num_index_heads() -> usize {
    4
}
fn default_sparse_topk_blocks() -> usize {
    16
}
fn default_sparse_block_size() -> usize {
    128
}
fn default_sparse_score_type() -> String {
    "max".to_string()
}
fn default_sparse_local_block() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub use_gemma_norm: bool,
    #[serde(default)]
    pub attention_output_gate: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rotary_dim")]
    pub rotary_dim: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_qk_norm_type")]
    pub qk_norm_type: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // MLP widths.
    #[serde(default)]
    pub dense_intermediate_size: usize,
    #[serde(default)]
    pub shared_intermediate_size: usize,

    // MoE routing.
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_true")]
    pub use_routing_bias: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    /// Per-layer dense(0)/MoE(1) plan. The real config ships a 60-entry array;
    /// an empty vec is treated as all-MoE.
    #[serde(default)]
    pub moe_layer_freq: Vec<usize>,

    // Clamp-SwiGLU activation.
    #[serde(default = "default_swiglu_alpha")]
    pub swiglu_alpha: f32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,

    #[serde(default)]
    pub sparse_attention_config: Option<SparseAttentionConfig>,

    // MTP metadata is parsed but unused; the MTP head is out of scope.
    #[serde(default)]
    pub num_mtp_modules: Option<usize>,
    #[serde(default)]
    pub num_nextn_predict_layers: Option<usize>,

    #[serde(default)]
    pub quantization: Option<Quantization>,
}

fn default_model_type() -> String {
    "minimax_m3".to_string()
}
fn default_head_dim() -> usize {
    128
}
fn default_max_position_embeddings() -> usize {
    1_048_576
}
fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f32 {
    5_000_000.0
}
fn default_rotary_dim() -> usize {
    64
}
fn default_partial_rotary_factor() -> f32 {
    0.5
}
fn default_hidden_act() -> String {
    "swigluoai".to_string()
}
fn default_qk_norm_type() -> String {
    "per_head".to_string()
}
fn default_n_shared_experts() -> usize {
    1
}
fn default_scoring_func() -> String {
    "sigmoid".to_string()
}
fn default_routed_scaling_factor() -> f32 {
    2.0
}
fn default_swiglu_alpha() -> f32 {
    1.702
}
fn default_swiglu_limit() -> f32 {
    7.0
}

impl ModelArgs {
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// MoE routers are commonly kept at 8-bit even in 4-bit checkpoints; the
    /// `UnifiedLinear` loader only quantizes when scales are actually present,
    /// so this is a no-op for an unquantized (bf16) gate.
    pub fn gate_bits(&self) -> i32 {
        if self.quantization.is_some() {
            8
        } else {
            self.bits()
        }
    }

    /// Whether layer `idx` uses the MoE MLP (vs the dense MLP). Driven by
    /// `moe_layer_freq`; an out-of-range or empty plan defaults to MoE.
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        self.moe_layer_freq.get(idx).copied().unwrap_or(1) != 0
    }

    /// Whether layer `idx` runs the block-sparse indexer.
    pub fn is_sparse_layer(&self, idx: usize) -> bool {
        match &self.sparse_attention_config {
            Some(cfg) if cfg.use_sparse_attention => {
                cfg.sparse_attention_freq.get(idx).copied().unwrap_or(0) != 0
            }
            _ => false,
        }
    }
}

impl SparseAttentionConfig {
    /// Reject a degenerate sparse-attention config before it reaches the
    /// indexer math, following the `validate_quantization_scheme` precedent in
    /// `gemma4.rs` (validate once at load time rather than guard ad hoc in the
    /// hot path).
    ///
    /// `sparse_block_size == 0` would divide by zero computing `num_blocks` in
    /// `build_block_drop_mask` and would always satisfy
    /// `should_apply_sparse`'s `kv_len > 2 * topk_blocks * block_size` check
    /// (degenerating it to "always sparse"); `sparse_topk_blocks == 0` would
    /// call `argpartition` with `kth = topk_blocks - 1 = -1`, an invalid
    /// partition index.
    pub fn validate(&self) -> Result<(), String> {
        if self.sparse_block_size == 0 {
            return Err(
                "minimax_m3: sparse_attention_config.sparse_block_size must be non-zero \
                 (zero divides by zero when computing the block-sparse mask)"
                    .to_string(),
            );
        }
        if self.sparse_topk_blocks == 0 {
            return Err(
                "minimax_m3: sparse_attention_config.sparse_topk_blocks must be non-zero \
                 (zero is an invalid argpartition kth index)"
                    .to_string(),
            );
        }
        Ok(())
    }
}
