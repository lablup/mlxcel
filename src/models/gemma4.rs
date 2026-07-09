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

//! Gemma 4 text model implementation using mlxcel-core.
//!
//! Key features:
//! - Layer-type driven sliding/full attention
//! - Dual head dimensions (`head_dim` + `global_head_dim`)
//! - Partial RoPE on full-attention layers
//! - K-eq-V full-attention path for 26B/31B variants
//! - KV sharing for E-series models
//! - Per-layer input gating for E-series models
//! - Dense + MoE feed-forward paths
//! - Final logit softcapping

use crate::distributed::pipeline::LayerFilter;
use crate::distributed::pipeline::StageExecutionOutput;
use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::model_owned::ModelOwnedSequenceState;
use crate::models::recurrent_snapshot::{push_i32, push_optional, restore_i32, restore_optional};
use crate::models::switch_layers::{SwitchLinear, gather_sort};
use mlxcel_core::cache::{
    KVCacheMode, RotatingKVCacheSnapshotState, SequenceId, SequenceStateLayout,
};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::{
    FusedQKVLinear, KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear,
    compiled_gelu_mlp_fp16,
};
use mlxcel_core::utils::{
    create_causal_mask, create_causal_mask_with_left_padding, create_causal_mask_with_window,
    create_causal_mask_with_window_and_left_padding, create_sliding_window_prefill_mask,
    mask_stale_key_gap, pipeline_hint, slice_axis,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuantizationArgs {
    #[serde(default = "default_quant_group_size")]
    pub group_size: usize,
    #[serde(default = "default_quant_bits")]
    pub bits: usize,
    #[serde(default, flatten)]
    pub overrides: HashMap<String, serde_json::Value>,
}

fn default_quant_group_size() -> usize {
    64
}

fn default_quant_bits() -> usize {
    4
}

/// Bit widths mlxcel's affine quantization understands, mirroring
/// `mlxcel_core::layers::infer_quantization_bits`'s `{2,3,4,5,6,8}` set.
const SUPPORTED_OVERRIDE_BITS: &[u64] = &[2, 3, 4, 5, 6, 8];

/// Validates a per-module override's `group_size` value, returning an
/// actionable reason string when it is not a positive integer.
fn validate_override_group_size(raw: &serde_json::Value) -> Result<i32, String> {
    let Some(group_size) = raw.as_u64() else {
        return Err(format!("must be a positive integer, got {raw}"));
    };
    if group_size == 0 {
        return Err("must be a positive integer, got 0".to_string());
    }
    i32::try_from(group_size).map_err(|_| format!("overflows a 32-bit group size ({group_size})"))
}

/// Validates a per-module override's `bits` value, returning an actionable
/// reason string when it is not an integer or not a supported bit width.
fn validate_override_bits(raw: &serde_json::Value) -> Result<i32, String> {
    let Some(bits) = raw.as_u64() else {
        return Err(format!("must be a positive integer, got {raw}"));
    };
    if !SUPPORTED_OVERRIDE_BITS.contains(&bits) {
        return Err(format!(
            "must be one of {{2, 3, 4, 5, 6, 8}} bits, got {bits}"
        ));
    }
    i32::try_from(bits).map_err(|_| format!("overflows a 32-bit bit width ({bits})"))
}

impl QuantizationArgs {
    /// Resolves the quantization parameters for the module weight registered
    /// under `prefix`, applying a per-module override when one is present.
    ///
    /// `overrides` is the flattened remainder of the parent `quantization`
    /// object (everything besides the well-known root `group_size` / `bits`
    /// pair, see PR #690). Only the entry keyed by the module's own `prefix`
    /// is consulted here, so unrelated metadata keys that legitimately live
    /// alongside per-module overrides (e.g. `mode`, `quant_method`,
    /// `quant_algo`, `config_groups`, all validated separately by
    /// `validate_quantization_scheme`) never collide with a real per-module
    /// lookup and are left untouched.
    ///
    /// A malformed override for an exact-matching `prefix` (a non-object
    /// value, or a `group_size`/`bits` field with the wrong type, a negative
    /// value, a zero group size, or an unsupported bit width) is diagnosed
    /// with an actionable `tracing::warn!` naming the offending key before
    /// falling back to the root defaults for that field, rather than
    /// silently mis-routing the load (issue #691).
    fn quant_params_for(&self, prefix: &str) -> QuantizationParams {
        let mut params = QuantizationParams {
            group_size: self.group_size as i32,
            bits: self.bits as i32,
        };
        let Some(value) = self.overrides.get(prefix) else {
            return params;
        };
        let Some(override_obj) = value.as_object() else {
            tracing::warn!(
                override_key = prefix,
                value = %value,
                "gemma4 per-module quantization override '{prefix}' is not an object \
                 (found {value}); falling back to root defaults (group_size={}, bits={})",
                self.group_size,
                self.bits
            );
            return params;
        };
        if let Some(raw) = override_obj.get("group_size") {
            match validate_override_group_size(raw) {
                Ok(group_size) => params.group_size = group_size,
                Err(reason) => tracing::warn!(
                    override_key = prefix,
                    reason = %reason,
                    "gemma4 per-module quantization override '{prefix}.group_size' is \
                     malformed: {reason}; falling back to root group_size={}",
                    self.group_size
                ),
            }
        }
        if let Some(raw) = override_obj.get("bits") {
            match validate_override_bits(raw) {
                Ok(bits) => params.bits = bits,
                Err(reason) => tracing::warn!(
                    override_key = prefix,
                    reason = %reason,
                    "gemma4 per-module quantization override '{prefix}.bits' is malformed: \
                     {reason}; falling back to root bits={}",
                    self.bits
                ),
            }
        }
        params
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantizationParams {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    /// Upstream Gemma 4 (post-commit bb91e23 on mlx-vlm main) marks
    /// full-attention layers with `rope_type: "proportional"`, whose exponents
    /// are normalized by the FULL head dimension (`head_dim`) rather than the
    /// rotated-only slice. `"default"` means the usual
    /// `nn.RoPE(dims = head_dim * partial_rotary_factor)` path.
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub vocab_size_per_layer_input: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: usize,
    #[serde(default)]
    pub hidden_size_per_layer_input: usize,
    #[serde(default)]
    pub rope_traditional: bool,
    pub rope_parameters: HashMap<String, RopeParameters>,
    pub sliding_window: usize,
    #[serde(default)]
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    /// Gemma 4 Unified blockwise bidirectional attention selector.
    ///
    /// When set to `"vision"` (the only value emitted by current
    /// `gemma4_unified` checkpoints), image/video token spans attend
    /// bidirectionally *within each contiguous span* during prefill, while
    /// everything else stays causal/windowed. `None`/absent (the value on
    /// every `gemma4`/`gemma4_text` checkpoint) preserves the standard
    /// fully-causal behaviour. The blockwise overlay itself is driven by the
    /// `bidirectional_block_ids` argument passed into
    /// [`Gemma4TextModel::forward_with_speculative_sinks`]; this flag only
    /// records the checkpoint's intent.
    #[serde(default)]
    pub use_bidirectional_attention: Option<String>,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub top_k_experts: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationArgs>,
}

impl TextConfig {
    fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size as i32)
            .unwrap_or(64)
    }

    fn bits(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.bits as i32)
            .unwrap_or(4)
    }

    pub(crate) fn quant_params_for(&self, prefix: &str) -> QuantizationParams {
        self.quantization
            .as_ref()
            .map(|q| q.quant_params_for(prefix))
            .unwrap_or(QuantizationParams {
                group_size: self.group_size(),
                bits: self.bits(),
            })
    }

    fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        self.num_kv_shared_layers > 0 && layer_idx >= self.first_kv_shared_layer_idx()
    }

    fn layer_type(&self, layer_idx: usize) -> &str {
        self.layer_types[layer_idx].as_str()
    }

    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    /// Whether the checkpoint requests blockwise bidirectional attention over
    /// vision token spans (the `gemma4_unified` `"vision"` mode). Used by the
    /// Gemma 4 Unified runtime to decide whether to build the `same_block`
    /// overlay during image/video prefill.
    pub fn uses_bidirectional_vision_attention(&self) -> bool {
        self.use_bidirectional_attention.as_deref() == Some("vision")
    }

    fn head_dim_for_layer(&self, layer_idx: usize) -> i32 {
        if self.layer_type(layer_idx) == "full_attention" {
            self.global_head_dim.unwrap_or(self.head_dim) as i32
        } else {
            self.head_dim as i32
        }
    }

    fn num_kv_heads_for_layer(&self, layer_idx: usize) -> i32 {
        if self.attention_k_eq_v && !self.is_sliding_layer(layer_idx) {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads) as i32
        } else {
            self.num_key_value_heads as i32
        }
    }

    fn rope_params_for_layer(&self, layer_idx: usize) -> RopeParameters {
        let key = if self.is_sliding_layer(layer_idx) {
            "sliding_attention"
        } else {
            "full_attention"
        };
        self.rope_parameters
            .get(key)
            .cloned()
            .unwrap_or(RopeParameters {
                rope_theta: default_rope_theta(),
                partial_rotary_factor: default_partial_rotary_factor(),
                rope_type: default_rope_type(),
            })
    }

    fn mlp_intermediate_size(&self, layer_idx: usize) -> usize {
        let is_shared = self.is_kv_shared_layer(layer_idx);
        if self.use_double_wide_mlp && is_shared {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }
}

pub type RootQuantization = QuantizationArgs;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub text_config: serde_json::Value,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<RootQuantization>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

impl ModelArgs {
    pub fn text_args(&self) -> TextConfig {
        let mut config: TextConfig =
            serde_json::from_value(self.text_config.clone()).expect("Failed to parse text_config");
        if config.quantization.is_none()
            && let Some(ref q) = self.quantization
        {
            config.quantization = Some(q.clone());
        }
        if config.quantization.is_none()
            && let Some(ref quantization_config) = self.quantization_config
        {
            let root = serde_json::json!({ "quantization_config": quantization_config });
            config.quantization = Some(QuantizationArgs {
                group_size: super::sanitize::gemma4_configured_group_size(Some(&root)) as usize,
                bits: super::sanitize::gemma4_configured_bits(Some(&root)) as usize,
                overrides: HashMap::new(),
            });
        }
        config
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        parse_eos_ids(self.eos_token_id.as_ref())
    }
}

pub(crate) fn parse_eos_ids(value: Option<&serde_json::Value>) -> Vec<i32> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            n.as_i64().map(|v| vec![v as i32]).unwrap_or_default()
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64().map(|n| n as i32))
            .collect(),
        _ => Vec::new(),
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Overlay a blockwise bidirectional span onto an additive attention mask.
///
/// `base` is an additive f32 mask (`0.0` = attend, `-inf` = masked) of shape
/// `[q, k]` (= `[seq_len, seq_len + offset]`). `block_ids` is an int32 `[seq_len]`
/// tensor assigning each query/key position a *block id*: positions inside the
/// same contiguous image/video span share a non-negative id, and every other
/// position is `-1`. The returned mask additionally allows attention (sets the
/// additive value to `0`) wherever the query and key positions share the same
/// non-negative block id — making each vision span bidirectional in both
/// directions while leaving text↔text, text↔vision and cross-block pairs at the
/// base causal/windowed value.
///
/// This mirrors the upstream `gemma4_unified` mask construction
/// (`new_mask = base_mask | same_block`): for the Gemma 4 Unified prefill the
/// vision span is never split across a chunk (the loader sets
/// `no_chunked_prefill`), so `offset == 0` and `block_ids` aligns with both the
/// query and key axes.
///
/// Used by: Gemma 4 Unified prefill masks, DiffusionGemma image prefill
/// (`DiffusionGemmaModel::forward_encoder_embeds`, issue #217 phase 2).
pub(crate) fn overlay_block_bidirectional(
    base: &MlxArray,
    block_ids: &MlxArray,
) -> UniquePtr<MlxArray> {
    let base_shape = mlxcel_core::array_shape(base);
    debug_assert!(
        base_shape.len() >= 2,
        "base attention mask must be at least 2-D, got shape {base_shape:?}",
    );
    let q = base_shape[base_shape.len() - 2];
    let k = base_shape[base_shape.len() - 1];

    // Align the per-position block ids to the query (rows) and key (cols)
    // axes. For single-chunk prefill q == k == seq_len; two key-axis mismatches
    // are handled:
    // * `k > id_len` (cached prefix, offset > 0): the leading `k - id_len` key
    //   columns are the already-cached prefix and get id -1 (left pad) so they
    //   never participate in a same-block match.
    // * `k < id_len` (sliding-window cap): when `seq_len > sliding_window`,
    //   `create_causal_mask_with_window` caps the key axis to the last `k`
    //   logical positions (the rotating window holds only the most recent `k`
    //   keys). Column `k_c` then maps to logical key position
    //   `(id_len - k) + k_c`, so align the key-side ids to the trailing `k`
    //   block ids. Every vision span is at most one frame/image (≤ a few dozen
    //   soft tokens) and so always fits inside the window; aligning to the tail
    //   keeps each span's same-block match intact without leaking attention
    //   outside the windowed key axis (issue #164).
    let ids_shape = mlxcel_core::array_shape(block_ids);
    let id_len = if ids_shape.is_empty() {
        1
    } else {
        ids_shape[0]
    };
    let q_ids = mlxcel_core::reshape(block_ids, &[q, 1]);
    let k_ids = if k == id_len {
        mlxcel_core::reshape(block_ids, &[1, k])
    } else if k < id_len {
        // Sliding-window cap: take the trailing `k` block ids (the keys the
        // rotating window retains).
        let tail = mlxcel_core::slice(block_ids, &[id_len - k], &[id_len]);
        mlxcel_core::reshape(&tail, &[1, k])
    } else {
        // Cached prefix (offset > 0): pad the key-side ids on the left with -1
        // so cached-prefix columns are non-matching. pad_width is
        // [(before, after)] per axis.
        let pad_before = (k - id_len).max(0);
        let padded = mlxcel_core::pad(block_ids, &[pad_before, 0], -1.0);
        mlxcel_core::reshape(&padded, &[1, k])
    };

    // same_block[q, k] = (q_ids >= 0) && (q_ids == k_ids).
    let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
    let q_non_neg = mlxcel_core::greater_equal(&q_ids, &zero);
    let eq = mlxcel_core::equal(&q_ids, &k_ids);
    let same_block = mlxcel_core::logical_and(&q_non_neg, &eq);

    // new_mask = where(same_block, 0.0, base).
    let zero_f32 = mlxcel_core::full_f32(&base_shape, 0.0, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::where_cond(&same_block, &zero_f32, base)
}

pub struct RMSNormNoScale {
    eps: f32,
}

impl RMSNormNoScale {
    pub fn new(dim: i32, eps: f32) -> Self {
        let _ = dim;
        Self { eps }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rms_norm_no_weight(x, self.eps)
    }
}

struct SwitchGeGLU {
    gate_proj: SwitchLinear,
    up_proj: SwitchLinear,
    down_proj: SwitchLinear,
}

impl SwitchGeGLU {
    fn forward(&self, x: &MlxArray, indices: &MlxArray, hidden_size: i32) -> UniquePtr<MlxArray> {
        // `MLXCEL_PROFILE_MOE_INNER=1` emits `[MOE] name=... ms=...` per
        // sub-sub-op so the `experts` band in the sub-op profiler can be
        // drilled one level further. Like the outer profilers this forces a
        // sync at every step and distorts absolute throughput; use it only
        // for relative distribution analysis.
        let profile_inner = std::env::var("MLXCEL_PROFILE_MOE_INNER").is_ok();
        let inner_tick = |name: &str, arr: &MlxArray, last: &mut std::time::Instant| {
            if !profile_inner {
                return;
            }
            mlxcel_core::eval(arr);
            let dt = last.elapsed();
            eprintln!("[MOE] name={} ms={:.4}", name, dt.as_secs_f64() * 1000.0);
            *last = std::time::Instant::now();
        };
        let mut last = std::time::Instant::now();

        let indices_shape = mlxcel_core::array_shape(indices);
        let n_tokens = indices_shape[0];
        let top_k = indices_shape[1];
        let do_sort = n_tokens * top_k >= 64;

        // `Experts::forward` always passes a flattened `[tokens, hidden]`
        // input here. Python writes this as `expand_dims((-2, -3))`, which
        // gives `[tokens, 1, 1, hidden]`; a reshape is equivalent for this
        // rank-2 internal path and avoids an extra shape primitive in the
        // decode-hot SwitchGeGLU graph.
        let x_exp = mlxcel_core::reshape(x, &[n_tokens, 1, 1, hidden_size]);
        inner_tick("expand_dims", &x_exp, &mut last);

        if do_sort {
            let (sorted_x, sorted_idx, inv_order) = gather_sort(&x_exp, indices);
            inner_tick("gather_sort", &sorted_x, &mut last);
            let up = self.up_proj.forward(&sorted_x, &sorted_idx, true);
            inner_tick("up_proj", &up, &mut last);
            let gate = self.gate_proj.forward(&sorted_x, &sorted_idx, true);
            inner_tick("gate_proj", &gate, &mut last);
            let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
            inner_tick("geglu", &activated, &mut last);
            let output = self.down_proj.forward(&activated, &sorted_idx, true);
            inner_tick("down_proj", &output, &mut last);
            let unsorted = scatter_unsort(&output, &inv_order, &indices_shape);
            inner_tick("scatter_unsort", &unsorted, &mut last);
            unsorted
        } else {
            // Decode defaults to a wider compiled SwitchGeGLU window when the
            // quantized expert weights match the supported affine 4-bit shape.
            // The separate gather_qmm path remains available as an opt-out
            // diagnostic if backend scheduling regresses on a future MLX build.
            let disable_compiled_switch =
                std::env::var_os("MLXCEL_DISABLE_COMPILED_SWITCH_QGEGLU").is_some();
            let output = if !disable_compiled_switch
                && let (Some(gate_q), Some(up_q), Some(down_q)) = (
                    self.gate_proj.quantized_parts(),
                    self.up_proj.quantized_parts(),
                    self.down_proj.quantized_parts(),
                ) {
                let output = unsafe {
                    mlxcel_core::compiled_switch_qgeglu_forward(
                        &x_exp,
                        gate_q.weight,
                        gate_q.scales,
                        gate_q.biases.as_ref().unwrap() as *const _,
                        up_q.weight,
                        up_q.scales,
                        up_q.biases.as_ref().unwrap() as *const _,
                        down_q.weight,
                        down_q.scales,
                        down_q.biases.as_ref().unwrap() as *const _,
                        indices,
                        gate_q.group_size,
                        gate_q.bits,
                        gate_q.mode,
                    )
                };
                inner_tick("switch_geglu_fused", &output, &mut last);
                output
            } else {
                let up = self.up_proj.forward(&x_exp, indices, false);
                inner_tick("up_proj", &up, &mut last);
                let gate = self.gate_proj.forward(&x_exp, indices, false);
                inner_tick("gate_proj", &gate, &mut last);
                let activated = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
                inner_tick("geglu", &activated, &mut last);
                let output = self.down_proj.forward(&activated, indices, false);
                inner_tick("down_proj", &output, &mut last);
                output
            };
            let squeezed = mlxcel_core::squeeze_axis(&output, -2);
            inner_tick("squeeze", &squeezed, &mut last);
            squeezed
        }
    }

    /// Single-token decode via the fused GeGLU MoE kernel (#268). Returns the
    /// score-weighted expert sum `[hidden]`, or `None` (caller falls back to
    /// `forward` + combine) for any unsupported config: non-affine, gate/up not
    /// 4/8-bit or down not 4/6/8-bit, gate/up bits mismatch, group_size
    /// mismatch, the Regular variant, or a non-single-token `x`.
    fn forward_fused_kernel(
        &self,
        x: &MlxArray,
        indices: &MlxArray,
        scores: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let gate = self.gate_proj.quantized_parts()?;
        let up = self.up_proj.quantized_parts()?;
        let down = self.down_proj.quantized_parts()?;
        if gate.bits != 4 && gate.bits != 8 {
            return None;
        }
        if down.bits != 4 && down.bits != 8 && down.bits != 6 {
            return None;
        }
        if gate.bits != up.bits
            || gate.group_size != up.group_size
            || gate.group_size != down.group_size
        {
            return None;
        }
        if gate.mode != "affine" || up.mode != "affine" || down.mode != "affine" {
            return None;
        }
        let gw_shape = mlxcel_core::array_shape(gate.weight.as_ref().unwrap());
        if gw_shape.len() != 3 {
            return None;
        }
        let dff = gw_shape[1];
        let din = gw_shape[2] * (32 / gate.bits);
        if down.bits == 6 && dff % 16 != 0 {
            return None;
        }
        let k = *mlxcel_core::array_shape(indices).last()?;
        let x_elems: i32 = mlxcel_core::array_shape(x).iter().product();
        if x_elems != din {
            return None;
        }
        let x_flat = mlxcel_core::reshape(x, &[din]);
        let idx_flat = mlxcel_core::reshape(indices, &[k]);
        let sc_flat = mlxcel_core::reshape(scores, &[k]);
        Some(mlxcel_core::fused_moe_geglu_kernel(
            &x_flat,
            &idx_flat,
            gate.weight.as_ref().unwrap(),
            gate.scales.as_ref().unwrap(),
            gate.biases.as_ref().unwrap(),
            up.weight.as_ref().unwrap(),
            up.scales.as_ref().unwrap(),
            up.biases.as_ref().unwrap(),
            down.weight.as_ref().unwrap(),
            down.scales.as_ref().unwrap(),
            down.biases.as_ref().unwrap(),
            &sc_flat,
            din,
            dff,
            k,
            gate.bits,
            down.bits,
            gate.group_size,
        ))
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gate_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                group_size,
                bits,
            )?,
            up_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                group_size,
                bits,
            )?,
            down_proj: SwitchLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                group_size,
                bits,
            )?,
        })
    }
}

fn scatter_unsort(x: &MlxArray, inv_order: &MlxArray, orig_shape: &[i32]) -> UniquePtr<MlxArray> {
    let unsorted = mlxcel_core::take(x, inv_order, 0);
    let x_shape = mlxcel_core::array_shape(&unsorted);
    let n_tokens = orig_shape[0];
    let top_k = orig_shape[1];
    let reshaped = mlxcel_core::reshape(&unsorted, &[n_tokens, top_k, x_shape[1], x_shape[2]]);
    mlxcel_core::squeeze_axis(&reshaped, 2)
}

pub struct MLP {
    pub(crate) gate_proj: UnifiedLinear,
    pub(crate) up_proj: UnifiedLinear,
    pub(crate) down_proj: UnifiedLinear,
}

fn dense_mlp_shared_quant_layout(
    gate: QuantizationParams,
    gate_mode: &str,
    up: QuantizationParams,
    up_mode: &str,
    down: QuantizationParams,
    down_mode: &str,
) -> bool {
    gate == up && gate == down && gate_mode == up_mode && gate_mode == down_mode
}

/// Whether the NVFP4 fused global-scale fold is disabled via
/// `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE` (issue #698). When disabled, gemma4 MLP
/// and per-layer-input-gate paths that carry a `weight_scale_2` sidecar restore
/// the op-at-a-time bypass, where `UnifiedLinear::forward` applies the scalar.
/// Read once; accepts `1`/`true`/`on`/`yes` (case-insensitive) as truthy,
/// matching the documented disable-var value convention.
pub(crate) fn fused_global_scale_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("MLXCEL_DISABLE_FUSED_GLOBAL_SCALE")
            .ok()
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
            .unwrap_or(false)
    })
}

/// Whether the compiled per-layer-input-gate fused path
/// (`compiled_per_layer_input_gate`) can be used for a given gate/projection
/// pair. The fused C++ path now folds the native-NVFP4 `weight_scale_2`
/// sidecar (issue #698), so a sidecar-carrying pair is eligible when the fold
/// is enabled (`fused_scale_enabled`) AND the call is single-token decode
/// (`single_token`). For multi-token prefill, the C++ bridge itself falls back
/// to an uncompiled eager fold once the sidecar is present, which regresses
/// prefill throughput versus the compiled op-at-a-time activation used below,
/// so a sidecar-carrying pair must bypass the fused path for any multi-token
/// call regardless of the fold toggle (issue #698 follow-up). When the kill
/// switch disables the fold, or the call is multi-token, a sidecar-carrying
/// pair falls through to the op-at-a-time path, where `UnifiedLinear::forward`
/// applies the scalar. Pairs with no sidecar are always eligible, exactly as
/// before.
fn per_layer_input_gate_fused_path_eligible(
    gate_qw: &mlxcel_core::layers::QuantizedWeight,
    proj_qw: &mlxcel_core::layers::QuantizedWeight,
    fused_scale_enabled: bool,
    single_token: bool,
) -> bool {
    if gate_qw.global_scale.is_some() || proj_qw.global_scale.is_some() {
        fused_scale_enabled && single_token
    } else {
        true
    }
}

/// Whether the compiled scaled dense-MLP fused path
/// (`compiled_gelu_approx_mlp_forward_global_scale`) can be used for a given
/// sidecar / kill-switch / query-length / quant-layout combination. A
/// sidecar-carrying MLP is eligible when the fold is enabled and either:
///
/// - the call is single-token decode, which uses the existing shapeless
///   compiled graph from issue #698; or
/// - the call is native ModelOpt NVFP4 prefill (group_size=16/bits=4/mode=nvfp4),
///   where issue #705 added a shape-specific compiled graph so MLX keeps the
///   prefill qmm kernel while fusing the sidecar multiplies and GeGLU
///   activation.
///
/// Other multi-token sidecar layouts still bypass the fused path because the
/// bridge would otherwise fall back to an eager C++ fold; the op-at-a-time
/// path below keeps the compiled activation and remains the safer default for
/// unknown sidecar layouts. An MLP with no sidecar is unaffected by this gate;
/// `MLP::forward` takes the separate unscaled fused path for that case
/// regardless of `single_token`.
fn dense_mlp_scaled_fused_path_eligible(
    any_sidecar: bool,
    fused_scale_enabled: bool,
    single_token: bool,
    group_size: i32,
    bits: i32,
    mode: &str,
) -> bool {
    let native_nvfp4_prefill = !single_token && group_size == 16 && bits == 4 && mode == "nvfp4";
    any_sidecar && fused_scale_enabled && (single_token || native_nvfp4_prefill)
}

impl MLP {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Dense quantized MLP with a shared gate/up/down quant layout uses the
        // fused gate/up/gelu/down C++ path. When the projections carry a native
        // NVFP4 `weight_scale_2` sidecar (issue #693/#697), the fused scaled
        // variant folds each per-projection global scale in at the
        // mathematically correct point (issue #698): the gate scale before the
        // GeGLU activation (nonlinear), the up scale on the up product, and the
        // down scale on the fused output. Each fold reproduces
        // `apply_global_scale` byte-for-byte, so the result is bit-identical to
        // the op-at-a-time bypass. `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE` restores
        // that bypass below, where each `UnifiedLinear::forward` applies its own
        // scalar.
        if let (Some(gate_qw), Some(up_qw), Some(down_qw)) = (
            self.gate_proj.quantized_weight(),
            self.up_proj.quantized_weight(),
            self.down_proj.quantized_weight(),
        ) && dense_mlp_shared_quant_layout(
            QuantizationParams {
                group_size: gate_qw.group_size,
                bits: gate_qw.bits,
            },
            &gate_qw.mode,
            QuantizationParams {
                group_size: up_qw.group_size,
                bits: up_qw.bits,
            },
            &up_qw.mode,
            QuantizationParams {
                group_size: down_qw.group_size,
                bits: down_qw.bits,
            },
            &down_qw.mode,
        ) {
            let any_sidecar = gate_qw.global_scale.is_some()
                || up_qw.global_scale.is_some()
                || down_qw.global_scale.is_some();

            // The scaled fused C++ path is now used for native NVFP4 prefill
            // as well as decode. Issue #705 adds a shape-specific compiled
            // graph for the group_size=16/bits=4/nvfp4 path so prefill avoids
            // both the old C++ eager fold and the per-projection Rust/FFI
            // sidecar dispatch. Unknown multi-token sidecar layouts still
            // bypass the fused path via `dense_mlp_scaled_fused_path_eligible`
            // because their bridge fallback remains eager rather than compiled.
            let single_token = mlxcel_core::array_shape(x)[1] == 1;

            if dense_mlp_scaled_fused_path_eligible(
                any_sidecar,
                !fused_global_scale_disabled(),
                single_token,
                gate_qw.group_size,
                gate_qw.bits,
                &gate_qw.mode,
            ) {
                return unsafe {
                    mlxcel_core::compiled_gelu_approx_mlp_forward_global_scale(
                        x,
                        &gate_qw.weight,
                        &gate_qw.scales,
                        &up_qw.weight,
                        &up_qw.scales,
                        &down_qw.weight,
                        &down_qw.scales,
                        gate_qw.global_scale_ptr(),
                        up_qw.global_scale_ptr(),
                        down_qw.global_scale_ptr(),
                        gate_qw.group_size,
                        gate_qw.bits,
                        &gate_qw.mode,
                    )
                };
            }

            if !any_sidecar {
                return unsafe {
                    mlxcel_core::compiled_gelu_approx_mlp_forward(
                        x,
                        &gate_qw.weight,
                        &gate_qw.scales,
                        gate_qw.biases_ptr(),
                        &up_qw.weight,
                        &up_qw.scales,
                        up_qw.biases_ptr(),
                        &down_qw.weight,
                        &down_qw.scales,
                        down_qw.biases_ptr(),
                        gate_qw.group_size,
                        gate_qw.bits,
                        &gate_qw.mode,
                    )
                };
            }
            // any_sidecar && (kill switch set OR multi-token prefill): fall
            // through to op-at-a-time, where each `UnifiedLinear::forward`
            // applies its own scalar and the compiled activation below
            // handles the GeGLU.
        }

        if let Some(out) =
            compiled_gelu_mlp_fp16(x, &self.gate_proj, &self.up_proj, &self.down_proj)
        {
            return out;
        }

        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let hidden = mlxcel_core::compiled_geglu_approx_activation(&gate, &up);
        self.down_proj.forward(&hidden)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let _ = config.mlp_intermediate_size(layer_idx);
        let gate_prefix = format!("{}.gate_proj", prefix);
        let up_prefix = format!("{}.up_proj", prefix);
        let down_prefix = format!("{}.down_proj", prefix);
        let gate_quant = config.quant_params_for(&gate_prefix);
        let up_quant = config.quant_params_for(&up_prefix);
        let down_quant = config.quant_params_for(&down_prefix);
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &gate_prefix,
                gate_quant.group_size,
                gate_quant.bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &up_prefix,
                up_quant.group_size,
                up_quant.bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &down_prefix,
                down_quant.group_size,
                down_quant.bits,
            )?,
        })
    }
}

pub struct Router {
    /// Precomputed `scale * hidden_size^-0.5` fed as the weight to
    /// `fast_rms_norm`, collapsing the old
    /// `rms_norm_no_weight → multiply_scalar → multiply` trio into
    /// one fused Metal kernel to match Python's
    /// `mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)`.
    pub(crate) scale_with_root: UniquePtr<MlxArray>,
    pub(crate) rms_eps: f32,
    pub(crate) proj: UnifiedLinear,
    pub(crate) per_expert_scale: UniquePtr<MlxArray>,
    pub(crate) top_k_experts: i32,
}

impl Router {
    pub fn forward(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let x = mlxcel_core::fast_rms_norm(x, &self.scale_with_root, self.rms_eps);
        let expert_scores = self.proj.forward(&x);

        // Pick top-k before softmax so softmax runs over k=8 values,
        // not the full num_experts=128. Use the same negative-kth
        // argpartition shape as mlx-lm to avoid an extra negation graph.
        let top_k_indices = mlxcel_core::argpartition(&expert_scores, -self.top_k_experts, -1);
        let top_k_indices = slice_axis(&top_k_indices, -1, -self.top_k_experts, -1);

        let top_k_weights = mlxcel_core::take_along_axis(&expert_scores, &top_k_indices, -1);
        let top_k_weights = mlxcel_core::softmax(&top_k_weights, -1);
        let expert_scale = mlxcel_core::take(&self.per_expert_scale, &top_k_indices, 0);
        let top_k_weights = mlxcel_core::multiply(&top_k_weights, &expert_scale);

        (top_k_indices, top_k_weights)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let raw_scale = get_weight_copy(weights, &format!("{}.scale", prefix))?;
        let root = (config.hidden_size as f32).powf(-0.5);
        let scale_with_root = mlxcel_core::multiply_scalar(&raw_scale, root);
        mlxcel_core::eval(&scale_with_root);

        Ok(Self {
            scale_with_root,
            rms_eps: config.rms_norm_eps,
            proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            per_expert_scale: get_weight_copy(weights, &format!("{}.per_expert_scale", prefix))?,
            top_k_experts: config
                .top_k_experts
                .ok_or_else(|| "Missing top_k_experts for Gemma4 MoE router".to_string())?
                as i32,
        })
    }
}

pub struct Experts {
    switch_geglu: SwitchGeGLU,
}

impl Experts {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        top_k_indices: &MlxArray,
        top_k_weights: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let s = shape[1];
        let h = shape[2];
        let k = mlxcel_core::array_shape(top_k_indices)[2];

        let x_flat = mlxcel_core::reshape(x, &[b * s, h]);
        let indices_flat = mlxcel_core::reshape(top_k_indices, &[b * s, k]);

        // Fused single-token decode GeGLU kernel (#268) on by default
        // (MLXCEL_FUSED_MOE=0 disables); otherwise SwitchGeGLU + weighted combine
        // (also the kernel's automatic fallback).
        if b * s == 1
            && crate::models::switch_layers::fused_moe_enabled()
            && let Some(out) =
                self.switch_geglu
                    .forward_fused_kernel(&x_flat, &indices_flat, top_k_weights)
        {
            return mlxcel_core::reshape(&out, &[b, s, h]);
        }

        let expert_out = self.switch_geglu.forward(&x_flat, &indices_flat, h);
        let weights = mlxcel_core::reshape(top_k_weights, &[b * s, k, 1]);
        let weighted = mlxcel_core::multiply(&expert_out, &weights);
        let reduced = mlxcel_core::sum_axis(&weighted, -2, false);
        mlxcel_core::reshape(&reduced, &[b, s, h])
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        Ok(Self {
            switch_geglu: SwitchGeGLU::from_weights(
                weights,
                &format!("{}.switch_glu", prefix),
                config.group_size(),
                config.bits(),
            )?,
        })
    }
}

enum AttentionProjection {
    Fused(FusedQKVLinear),
    Separate {
        q_proj: UnifiedLinear,
        k_proj: UnifiedLinear,
        v_proj: Option<UnifiedLinear>,
    },
    /// KV-shared layers only compute Q; K/V come from a prior non-shared layer
    /// via the `shared_kv` argument to `forward`. No k_proj, v_proj, or k_norm
    /// are constructed for these layers. Mirrors the upstream `has_kv = False`
    /// gate in Gemma4Attention.__init__ (mlx-lm PR #1158).
    KvShared {
        q_proj: UnifiedLinear,
    },
}

pub(crate) trait CacheInterface {
    fn offset(&self) -> i32;
    fn set_offset(&mut self, offset: i32);
    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);
}

impl CacheInterface for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn set_offset(&mut self, offset: i32) {
        self.offset = offset;
    }

    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(keys, values)
    }
}

impl CacheInterface for RotatingKVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn set_offset(&mut self, offset: i32) {
        self.offset = offset;
    }

    fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.update_and_fetch(keys, values)
    }
}

fn kv_cache_mode_to_i32(mode: KVCacheMode) -> i32 {
    match mode {
        KVCacheMode::Fp16 => 0,
        KVCacheMode::Int8 => 1,
        KVCacheMode::Turbo4Asym => 2,
        KVCacheMode::Turbo3Asym => 3,
        KVCacheMode::Turbo4 => 4,
        KVCacheMode::Turbo4Delegated => 5,
    }
}

fn kv_cache_mode_from_i32(value: i32) -> Result<KVCacheMode, String> {
    match value {
        0 => Ok(KVCacheMode::Fp16),
        1 => Ok(KVCacheMode::Int8),
        2 => Ok(KVCacheMode::Turbo4Asym),
        3 => Ok(KVCacheMode::Turbo3Asym),
        4 => Ok(KVCacheMode::Turbo4),
        5 => Ok(KVCacheMode::Turbo4Delegated),
        other => Err(format!("unknown Gemma 4 cache snapshot mode tag {other}")),
    }
}

pub enum Cache {
    Standard(KVCache),
    Rotating(RotatingKVCache),
}

impl Cache {
    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Standard(cache) => cache.offset,
            Self::Rotating(cache) => cache.offset,
        }
    }

    pub(crate) fn as_interface(&mut self) -> &mut dyn CacheInterface {
        match self {
            Self::Standard(cache) => cache,
            Self::Rotating(cache) => cache,
        }
    }

    pub(crate) fn snapshot_into(
        &self,
        snapshot: &mut ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Self::Standard(cache) => {
                if cache.keys.is_none() && cache.values.is_none() {
                    return Ok(());
                }
                if cache.keys.is_some() != cache.values.is_some() {
                    return Err(format!(
                        "Gemma 4 snapshot {prefix}: standard cache has only one of keys/values"
                    ));
                }
                if cache.mode != KVCacheMode::Fp16 {
                    return Err(format!(
                        "Gemma 4 snapshot {prefix}: standard cache mode {:?} is not supported by model-state snapshots",
                        cache.mode
                    ));
                }
                push_optional(snapshot, format!("{prefix}.standard.keys"), &cache.keys);
                push_optional(snapshot, format!("{prefix}.standard.values"), &cache.values);
                push_i32(snapshot, format!("{prefix}.standard.offset"), cache.offset);
                push_i32(
                    snapshot,
                    format!("{prefix}.standard.mode"),
                    kv_cache_mode_to_i32(cache.mode),
                );
            }
            Self::Rotating(cache) => {
                if cache.keys.is_none() && cache.values.is_none() {
                    return Ok(());
                }
                if cache.keys.is_some() != cache.values.is_some() {
                    return Err(format!(
                        "Gemma 4 snapshot {prefix}: rotating cache has only one of keys/values"
                    ));
                }
                let state = cache.snapshot_state();
                if state.mode != KVCacheMode::Fp16 {
                    return Err(format!(
                        "Gemma 4 snapshot {prefix}: rotating cache mode {:?} is not supported by model-state snapshots",
                        state.mode
                    ));
                }
                push_optional(snapshot, format!("{prefix}.rotating.keys"), &cache.keys);
                push_optional(snapshot, format!("{prefix}.rotating.values"), &cache.values);
                push_i32(
                    snapshot,
                    format!("{prefix}.rotating.max_size"),
                    state.max_size,
                );
                push_i32(
                    snapshot,
                    format!("{prefix}.rotating.buffer_size"),
                    state.buffer_size,
                );
                push_i32(snapshot, format!("{prefix}.rotating.offset"), state.offset);
                push_i32(
                    snapshot,
                    format!("{prefix}.rotating.start_position"),
                    state.start_position,
                );
                push_i32(snapshot, format!("{prefix}.rotating.idx"), state.idx);
                push_i32(snapshot, format!("{prefix}.rotating.step"), state.step);
                push_i32(
                    snapshot,
                    format!("{prefix}.rotating.mode"),
                    kv_cache_mode_to_i32(state.mode),
                );
                push_i32(
                    snapshot,
                    format!("{prefix}.rotating.turbo_seed"),
                    state.turbo_seed as i32,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn restore_from(
        &mut self,
        snapshot: &ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        match self {
            Self::Standard(cache) => {
                let keys = restore_optional(snapshot, format!("{prefix}.standard.keys"));
                let values = restore_optional(snapshot, format!("{prefix}.standard.values"));
                if keys.is_none() && values.is_none() {
                    return Ok(());
                }
                if keys.is_some() != values.is_some() {
                    return Err(format!(
                        "Gemma 4 restore {prefix}: standard snapshot has only one of keys/values"
                    ));
                }
                let mode = restore_i32(snapshot, format!("{prefix}.standard.mode"))
                    .map(kv_cache_mode_from_i32)
                    .transpose()?
                    .unwrap_or(KVCacheMode::Fp16);
                if mode != KVCacheMode::Fp16 {
                    return Err(format!(
                        "Gemma 4 restore {prefix}: standard snapshot mode {:?} is not supported",
                        mode
                    ));
                }
                cache.keys = keys;
                cache.values = values;
                cache.offset = restore_i32(snapshot, format!("{prefix}.standard.offset"))
                    .unwrap_or(snapshot.token_len() as i32);
                cache.mode = KVCacheMode::Fp16;
            }
            Self::Rotating(cache) => {
                let keys = restore_optional(snapshot, format!("{prefix}.rotating.keys"));
                let values = restore_optional(snapshot, format!("{prefix}.rotating.values"));
                if keys.is_none() && values.is_none() {
                    return Ok(());
                }
                if keys.is_some() != values.is_some() {
                    return Err(format!(
                        "Gemma 4 restore {prefix}: rotating snapshot has only one of keys/values"
                    ));
                }
                let current = cache.snapshot_state();
                let mode = restore_i32(snapshot, format!("{prefix}.rotating.mode"))
                    .map(kv_cache_mode_from_i32)
                    .transpose()?
                    .unwrap_or(KVCacheMode::Fp16);
                let state = RotatingKVCacheSnapshotState {
                    max_size: restore_i32(snapshot, format!("{prefix}.rotating.max_size"))
                        .unwrap_or(current.max_size),
                    buffer_size: restore_i32(snapshot, format!("{prefix}.rotating.buffer_size"))
                        .unwrap_or(0),
                    offset: restore_i32(snapshot, format!("{prefix}.rotating.offset"))
                        .unwrap_or(snapshot.token_len() as i32),
                    start_position: restore_i32(
                        snapshot,
                        format!("{prefix}.rotating.start_position"),
                    )
                    .unwrap_or(0),
                    idx: restore_i32(snapshot, format!("{prefix}.rotating.idx"))
                        .unwrap_or(snapshot.token_len() as i32),
                    step: restore_i32(snapshot, format!("{prefix}.rotating.step"))
                        .unwrap_or(current.step),
                    mode,
                    turbo_seed: restore_i32(snapshot, format!("{prefix}.rotating.turbo_seed"))
                        .map(|seed| seed as u32)
                        .unwrap_or(current.turbo_seed),
                };
                cache.restore_fp16_snapshot_state(state, keys, values)?;
            }
        }
        Ok(())
    }

    /// Trim the last `n` entries from this cache, dispatching to the
    /// underlying `KVCache::trim` or `RotatingKVCache::trim`. Mirrors the
    /// upstream Python `hasattr(c, "trim")` dispatch in
    /// `Gemma4 LanguageModel.rollback_speculative_cache`.
    ///
    /// Used by: Gemma 4 MTP `rollback_speculative_cache`.
    pub(crate) fn trim_speculative(&mut self, n: i32) -> i32 {
        match self {
            Self::Standard(cache) => cache.trim(n),
            Self::Rotating(cache) => cache.trim(n),
        }
    }

    /// Add upstream-style rollback slack to sliding-window target caches used
    /// by Gemma 4 MTP. The logical attention window is unchanged; only the
    /// rotating cache's temporary storage grows so verify-block append +
    /// rollback cannot overwrite still-visible window entries.
    ///
    /// Used by: Gemma 4 MTP B=1/B>1 target adapters.
    pub(crate) fn enable_mtp_rotating_buffer(&mut self, buffer_size: i32) -> Result<(), String> {
        match self {
            Self::Standard(_) => Ok(()),
            Self::Rotating(cache) => cache.enable_speculative_buffer(buffer_size),
        }
    }

    /// Per-row tail-zero of the rotating cache for partial-accept rows in a
    /// batched verify pass. Mirrors the Python `hasattr(c, "_idx")` branch in
    /// `rollback_speculative_cache` (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4
    /// /language.py lines 637-645):
    ///
    /// ```text
    /// kv_len = c._idx                       # post-trim write index
    /// n = max(valid_ends)                   # == max(accepted) + 1
    /// verify_start = kv_len - n
    /// for bi in range(accepted.shape[0]):
    ///     start = verify_start + valid_ends[bi]
    ///     if start < kv_len:
    ///         keys[bi, :, start:kv_len, :] = 0
    ///         values[bi, :, start:kv_len, :] = 0
    /// ```
    ///
    /// `valid_ends[bi]` is `accepted[bi] + 1` and `kv_len` is the current
    /// rotating-buffer write index (`buffer_write_idx`). The `block_size`
    /// parameter is retained for API parity with the Python signature; the
    /// effective span to zero is derived from `n = max(valid_ends)`, exactly
    /// as Python does. This step is a no-op for `Cache::Standard` (dense
    /// `KVCache`) because Python's hook gates per-row zeroing on `_idx`,
    /// which only `RotatingKVCache` exposes — the dense cache simply trims
    /// its monotonic offset and the next decode write overwrites the
    /// trimmed slot.
    ///
    /// Used by: Gemma 4 MTP `rollback_speculative_cache`.
    pub(crate) fn zero_partial_accept_tail(
        &mut self,
        valid_ends: &[i32],
        block_size: i32,
    ) -> Result<(), String> {
        let cache = match self {
            Self::Standard(_) => return Ok(()),
            Self::Rotating(cache) => cache,
        };
        if cache.keys.is_none() || cache.values.is_none() {
            return Ok(());
        }
        if valid_ends.is_empty() {
            return Ok(());
        }
        let kv_len = cache.buffer_write_idx();
        // n = max(valid_ends) — bit-identical to Python's `n = max_a + 1`
        // where `max_a = accepted.max()` and `valid_ends = accepted + 1`.
        let n = *valid_ends.iter().max().unwrap();
        if n <= 0 {
            // All rows rejected everything; nothing to zero (matches Python's
            // `max_a > 0` guard).
            return Ok(());
        }
        let verify_start = kv_len - n;
        if verify_start < 0 {
            return Err(format!(
                "zero_partial_accept_tail: n ({n}) exceeds rotating-cache write index \
                 ({kv_len}); refusing to zero a range that predates the verify pass \
                 (block_size = {block_size})"
            ));
        }
        let keys = cache.keys.as_ref().unwrap();
        let values = cache.values.as_ref().unwrap();
        let k_shape = mlxcel_core::array_shape(keys);
        let v_shape = mlxcel_core::array_shape(values);
        let batch = k_shape[0];
        if batch != valid_ends.len() as i32 {
            return Err(format!(
                "zero_partial_accept_tail: rotating-cache batch ({batch}) does not match \
                 accepted-row count ({})",
                valid_ends.len()
            ));
        }

        let k_dtype = mlxcel_core::array_dtype(keys);
        let v_dtype = mlxcel_core::array_dtype(values);
        // Walk rows individually so we can skip the no-op row (start == kv_len)
        // and apply slice_update only where the row needs it.
        let mut new_keys = mlxcel_core::copy(keys);
        let mut new_values = mlxcel_core::copy(values);
        for (bi, ve) in valid_ends.iter().copied().enumerate() {
            let start = verify_start + ve;
            if start >= kv_len {
                continue;
            }
            let span = kv_len - start;
            let bi_i = bi as i32;
            let k_zero = mlxcel_core::zeros(&[1, k_shape[1], span, k_shape[3]], k_dtype);
            let v_zero = mlxcel_core::zeros(&[1, v_shape[1], span, v_shape[3]], v_dtype);
            new_keys = mlxcel_core::slice_update(
                &new_keys,
                &k_zero,
                &[bi_i, 0, start, 0],
                &[bi_i + 1, k_shape[1], kv_len, k_shape[3]],
            );
            new_values = mlxcel_core::slice_update(
                &new_values,
                &v_zero,
                &[bi_i, 0, start, 0],
                &[bi_i + 1, v_shape[1], kv_len, v_shape[3]],
            );
        }
        cache.keys = Some(new_keys);
        cache.values = Some(new_values);
        Ok(())
    }

    /// Issue #203: close each row's post-rollback position hole by moving the
    /// row's accepted verify-window K/V down to the row's logical valid end,
    /// so every row's cache content stays a contiguous prefix (physical slot
    /// == logical position), exactly like its standalone B = 1 run.
    ///
    /// Coordinates: `ve_pre[r]` is row `r`'s logical valid end BEFORE this
    /// round's verify window (`left_padding[r] + kv_valid_len[r]`); the
    /// window of `width` tokens was appended at the shared physical offset
    /// `o_pre = self.offset() - width`. Row `r` accepted `accepted[r] + 1`
    /// window tokens (bonus + accepted drafts); they move from
    /// `[o_pre, o_pre + accepted[r] + 1)` to `[ve_pre[r], ...)` (a no-op for
    /// rows already at the shared offset) and the vacated region up to
    /// `o_post = max(ve_pre[r] + accepted[r] + 1)` is zeroed. The caller
    /// trims the cache offset down to `o_post` afterwards.
    ///
    /// Both cache kinds are only ever Fp16 on this path
    /// (`make_speculative_caches` constructs plain `KVCache::new` /
    /// `RotatingKVCache::new`), so no quantization sidecars need moving.
    /// Refuses front-compacted caches (buffer slot != logical position),
    /// the eligible batched MTP regime never compacts.
    pub(crate) fn compact_partial_accept_rows(
        &mut self,
        ve_pre: &[i32],
        accepted: &[i32],
        width: i32,
    ) -> Result<(), String> {
        if ve_pre.len() != accepted.len() {
            return Err(format!(
                "compact_partial_accept_rows: ve_pre rows ({}) != accepted rows ({})",
                ve_pre.len(),
                accepted.len()
            ));
        }
        if ve_pre.is_empty() {
            return Ok(());
        }
        let offset = self.offset();
        let o_pre = offset - width;
        if o_pre < 0 {
            return Err(format!(
                "compact_partial_accept_rows: width ({width}) exceeds cache offset ({offset})"
            ));
        }
        // Validate every row BEFORE computing the global o_post or touching
        // any buffer: a single malformed row must not let an earlier row's
        // zeroing slice_update run against an inflated o_post.
        for (r, (&ve, &a)) in ve_pre.iter().zip(accepted).enumerate() {
            if ve > o_pre || a < 0 || a > width - 1 {
                return Err(format!(
                    "compact_partial_accept_rows: row {r} out of bounds \
                     (ve_pre {ve}, accepted {a}, o_pre {o_pre}, width {width})"
                ));
            }
        }
        let o_post = ve_pre
            .iter()
            .zip(accepted)
            .map(|(&v, &a)| v + a + 1)
            .max()
            .unwrap();
        // Buffer slot of logical position p is `p - slot_base`; the batched
        // MTP regime never front-compacts either cache kind, so require a
        // zero base rather than silently corrupting slots.
        let slot_base = match self {
            Self::Standard(c) => c.offset - c.live_len(),
            Self::Rotating(c) => c.offset - c.buffer_write_idx(),
        };
        if slot_base != 0 {
            return Err(format!(
                "compact_partial_accept_rows: cache buffer is front-compacted \
                 (slot base {slot_base}); the divergent batched MTP path requires \
                 the uncompacted regime"
            ));
        }
        let (keys_slot, values_slot) = match self {
            Self::Standard(c) => (&mut c.keys, &mut c.values),
            Self::Rotating(c) => (&mut c.keys, &mut c.values),
        };
        let (Some(keys), Some(values)) = (keys_slot.as_ref(), values_slot.as_ref()) else {
            return Ok(());
        };
        let k_shape = mlxcel_core::array_shape(keys);
        let v_shape = mlxcel_core::array_shape(values);
        let batch = k_shape[0];
        if batch != ve_pre.len() as i32 {
            return Err(format!(
                "compact_partial_accept_rows: cache batch ({batch}) does not match \
                 row count ({})",
                ve_pre.len()
            ));
        }
        let k_dtype = mlxcel_core::array_dtype(keys);
        let v_dtype = mlxcel_core::array_dtype(values);

        let mut new_keys = mlxcel_core::copy(keys);
        let mut new_values = mlxcel_core::copy(values);
        for (r, (&ve, &a)) in ve_pre.iter().zip(accepted).enumerate() {
            let n = a + 1;
            let bi = r as i32;
            if ve < o_pre {
                // Materialize the source slice with an explicit copy BEFORE
                // the update: a bare slice is a lazy view of the same buffer,
                // and `slice_update` may donate that buffer to its output, so
                // an overlapping move (`ve + n > o_pre`) would read
                // already-overwritten rows without the copy.
                let src_k = mlxcel_core::copy(&mlxcel_core::slice(
                    &new_keys,
                    &[bi, 0, o_pre, 0],
                    &[bi + 1, k_shape[1], o_pre + n, k_shape[3]],
                ));
                new_keys = mlxcel_core::slice_update(
                    &new_keys,
                    &src_k,
                    &[bi, 0, ve, 0],
                    &[bi + 1, k_shape[1], ve + n, k_shape[3]],
                );
                let src_v = mlxcel_core::copy(&mlxcel_core::slice(
                    &new_values,
                    &[bi, 0, o_pre, 0],
                    &[bi + 1, v_shape[1], o_pre + n, v_shape[3]],
                ));
                new_values = mlxcel_core::slice_update(
                    &new_values,
                    &src_v,
                    &[bi, 0, ve, 0],
                    &[bi + 1, v_shape[1], ve + n, v_shape[3]],
                );
            }
            // Zero the vacated region between the row's new valid end and the
            // post-trim global end. Masked out next round anyway; zeroing
            // keeps the buffers hygienic for any maskless fallback.
            let z_start = ve + n;
            if z_start < o_post {
                let span = o_post - z_start;
                let k_zero = mlxcel_core::zeros(&[1, k_shape[1], span, k_shape[3]], k_dtype);
                let v_zero = mlxcel_core::zeros(&[1, v_shape[1], span, v_shape[3]], v_dtype);
                new_keys = mlxcel_core::slice_update(
                    &new_keys,
                    &k_zero,
                    &[bi, 0, z_start, 0],
                    &[bi + 1, k_shape[1], o_post, k_shape[3]],
                );
                new_values = mlxcel_core::slice_update(
                    &new_values,
                    &v_zero,
                    &[bi, 0, z_start, 0],
                    &[bi + 1, v_shape[1], o_post, v_shape[3]],
                );
            }
        }
        *keys_slot = Some(new_keys);
        *values_slot = Some(new_values);
        Ok(())
    }
}

/// Per-row context for a DIVERGENT batched MTP verify round (issue #203):
/// some row's logical valid end lags the shared physical cache offset after
/// mixed speculative accepts. Threaded from the model forward into every
/// attention layer so queries and keys rotate at per-row logical positions
/// (`ve[r]`), while the model-level [`build_divergent_verify_mask`] excludes
/// each row's leading padding and stale gap and applies the window causality
/// at the row's logical positions.
pub(crate) struct DivergentVerifyRows<'a> {
    /// Per-row logical valid end (`left_padding[r] + kv_valid_len[r]`): the
    /// RoPE offset for the row's window tokens and the end of the row's
    /// contiguous valid prefix in the shared cache.
    pub(crate) ve: &'a [i32],
    /// Per-row resident leading prompt padding (all zero for equal-length
    /// bursts).
    pub(crate) lp: Vec<i32>,
}

pub struct Attention {
    projection: AttentionProjection,
    pub(crate) o_proj: UnifiedLinear,
    pub(crate) q_norm: RMSNorm,
    /// `None` for KV-shared layers — those layers never call `project_kv` and
    /// therefore never need k_norm.  See `AttentionProjection::KvShared`.
    pub(crate) k_norm: Option<RMSNorm>,
    pub(crate) v_norm: RMSNormNoScale,
    pub(crate) n_heads: i32,
    pub(crate) n_kv_heads: i32,
    pub(crate) head_dim: i32,
    /// For `rope_type == "default"`, this is `head_dim * partial_rotary_factor`
    /// (the historical mlxcel / pre-bb91e23 mlx-vlm behavior).
    /// For `rope_type == "proportional"`, this is unused; proportional RoPE
    /// is driven by the precomputed full-head `proportional_rope_freqs` table
    /// below.
    pub(crate) rope_dims: i32,
    pub(crate) rope_theta: f32,
    /// If `Some`, the layer uses proportional RoPE (Gemma 4 full-attention
    /// layers). Holds the length-`head_dim / 2` frequency table consumed by
    /// `mlxcel_core::rope_proportional::apply_proportional_rope`; entries past
    /// the rotated prefix are `inf` so MLX's RoPE applies zero phase there.
    pub(crate) proportional_rope_freqs: Option<UniquePtr<MlxArray>>,
    /// Only meaningful when `proportional_rope_freqs.is_some()`. Matches the
    /// `partial_rotary_factor` passed in to
    /// `compute_proportional_rope_freqs`, so `apply_proportional_rope` can
    /// recompute `rotated_dims` without serializing it as a separate field.
    pub(crate) proportional_partial_rotary_factor: f32,
    pub(crate) scale: f32,
    pub(crate) is_kv_shared_layer: bool,
    pub(crate) kv_shared_layer_index: Option<usize>,
    pub(crate) store_full_length_kv: bool,
    pub(crate) use_k_eq_v: bool,
    pub(crate) window_size: i32,
}

impl Attention {
    /// Apply the layer's configured RoPE variant.
    ///
    /// * Proportional RoPE (Gemma 4 full-attention layers, `rope_type ==
    ///   "proportional"`): call `mx.fast.rope` across the full `head_dim`
    ///   with exponents normalized by the full dimension and an `inf`
    ///   frequency tail. This matches `mlx_lm.models.rope_utils.ProportionalRoPE`.
    /// * Default RoPE (sliding-attention layers and legacy configs): rotate
    ///   only the first `rope_dims = head_dim * partial_rotary_factor` slots
    ///   with the standard `fast_rope` kernel.
    fn apply_rope(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        if let Some(ref freqs) = self.proportional_rope_freqs {
            mlxcel_core::rope_proportional::apply_proportional_rope(
                x,
                self.head_dim,
                self.proportional_partial_rotary_factor,
                offset,
                Some(freqs),
            )
        } else {
            mlxcel_core::fast_rope(x, self.rope_dims, false, self.rope_theta, 1.0, offset)
        }
    }

    /// Rotate a `[B, H, L, D]` tensor row-by-row, applying each batch row's
    /// own RoPE offset.
    ///
    /// Used by the divergent batched MTP verify forward (issue #203): after
    /// mixed speculative accepts each row's logical position lags the shared
    /// physical cache offset, so the row's new window tokens must rotate at
    /// the row's own logical positions to keep B = 1 relative RoPE distances.
    /// B and L are tiny on that path (opt-in verify blocks), so the per-row
    /// slice/rotate/concat loop is cheap relative to the layer matmuls.
    fn apply_rope_per_row(&self, x: &MlxArray, offsets: &[i32]) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        debug_assert_eq!(shape.len(), 4, "rope input must be [B, H, L, D]");
        debug_assert_eq!(
            shape[0] as usize,
            offsets.len(),
            "apply_rope_per_row needs exactly one offset per batch row"
        );
        let mut out: Option<UniquePtr<MlxArray>> = None;
        for (r, &offset) in offsets.iter().enumerate() {
            let r_i = r as i32;
            let row =
                mlxcel_core::slice(x, &[r_i, 0, 0, 0], &[r_i + 1, shape[1], shape[2], shape[3]]);
            let rotated = self.apply_rope(&row, offset);
            out = Some(match out {
                None => rotated,
                Some(acc) => {
                    mlxcel_core::concatenate(acc.as_ref().unwrap(), rotated.as_ref().unwrap(), 0)
                }
            });
        }
        out.expect("apply_rope_per_row requires at least one batch row")
    }

    /// Per-row variant of the compiled proportional Q/K path: slice the
    /// `[B, L, n_heads_or_kv * head_dim]` projection output row by row and
    /// run each row through the SAME `compiled_q_path_proportional` fused
    /// kernel the uniform path uses, with that row's own RoPE offset.
    ///
    /// Used by the divergent batched MTP verify forward (issue #203). The
    /// offset flows into the compile window as a scalar array, so per-row
    /// offsets reuse the cached graph (one extra compile per `[1, L, ...]`
    /// shape, not per offset).
    #[allow(clippy::too_many_arguments)]
    fn compiled_q_path_proportional_per_row(
        &self,
        proj_out: &MlxArray,
        norm_weight: &MlxArray,
        norm_eps: f32,
        freqs: &MlxArray,
        n_heads: i32,
        rotated_dims: i32,
        l: i32,
        offsets: &[i32],
    ) -> UniquePtr<MlxArray> {
        let width = n_heads * self.head_dim;
        let mut out: Option<UniquePtr<MlxArray>> = None;
        for (r, &offset) in offsets.iter().enumerate() {
            let r_i = r as i32;
            let row = mlxcel_core::slice(proj_out, &[r_i, 0, 0], &[r_i + 1, l, width]);
            let rotated = mlxcel_core::compiled_q_path_proportional(
                &row,
                norm_weight,
                freqs,
                norm_eps,
                n_heads,
                self.head_dim,
                rotated_dims,
                offset,
            );
            out = Some(match out {
                None => rotated,
                Some(acc) => {
                    mlxcel_core::concatenate(acc.as_ref().unwrap(), rotated.as_ref().unwrap(), 0)
                }
            });
        }
        out.expect("compiled_q_path_proportional_per_row requires at least one batch row")
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
        divergent_rows: Option<&DivergentVerifyRows<'_>>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];
        let offset = cache.offset();
        let rope_offsets: Option<&[i32]> = divergent_rows.map(|ctx| ctx.ve);

        let q_proj_out = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (q, _, _) = proj.forward(x);
                q
            }
            AttentionProjection::Separate { q_proj, .. }
            | AttentionProjection::KvShared { q_proj } => q_proj.forward(x),
        };

        // Fast path: full-attention Gemma 4 layers run
        // `reshape -> q_norm -> transpose -> full-head ProportionalRoPE`
        // inside a single `mx::core::compile` window. Sliding layers and
        // layers with non-proportional RoPE stay on the op-at-a-time chain.
        //
        // Divergent batched MTP verify rounds (issue #203) instead rotate
        // each row at its own logical position; `rope_offsets` is `Some` only
        // on that path. Each row goes through the SAME kernels the uniform
        // path uses (the compiled fused chain for proportional layers, the
        // op-at-a-time chain otherwise), just sliced per row, so lockstep
        // rows stay bitwise-identical to the uniform rounds and near-tie
        // argmaxes cannot flip from a kernel-path change.
        let queries = if let Some(offsets) = rope_offsets {
            if let Some(ref freqs) = self.proportional_rope_freqs {
                let rotated_dims = 2 * ((self.proportional_partial_rotary_factor as f64
                    * self.head_dim as f64
                    / 2.0)
                    .floor() as i32)
                    .max(0);
                self.compiled_q_path_proportional_per_row(
                    &q_proj_out,
                    &self.q_norm.weight,
                    self.q_norm.eps,
                    freqs,
                    self.n_heads,
                    rotated_dims,
                    l,
                    offsets,
                )
            } else {
                let queries =
                    mlxcel_core::reshape(&q_proj_out, &[b, l, self.n_heads, self.head_dim]);
                let queries = self.q_norm.forward(&queries);
                let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
                self.apply_rope_per_row(&queries, offsets)
            }
        } else if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2
                * ((self.proportional_partial_rotary_factor as f64 * self.head_dim as f64 / 2.0)
                    .floor() as i32)
                    .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &q_proj_out,
                &self.q_norm.weight,
                freqs,
                self.q_norm.eps,
                self.n_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let queries = mlxcel_core::reshape(&q_proj_out, &[b, l, self.n_heads, self.head_dim]);
            let queries = self.q_norm.forward(&queries);
            let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
            self.apply_rope(&queries, offset)
        };

        if self.is_kv_shared_layer
            && let Some((keys, values)) = shared_kv
        {
            let attn_out = self.attend(&queries, keys, values, mask);
            return (self.project_output(&attn_out, b, l), None);
        }

        let (keys, values) = self.project_kv(x, b, l, offset, cache, rope_offsets);
        let attn_out = self.attend(&queries, &keys, &values, mask);
        let stored = if self.store_full_length_kv {
            Some((keys, values))
        } else {
            None
        };

        (self.project_output(&attn_out, b, l), stored)
    }

    fn attend(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let local_mask = trim_mask_to_keys(mask, keys);

        // When mask was discarded (undersized) or originally None,
        // use causal attention if possible.
        if local_mask.is_none() {
            return mlxcel_core::causal_attention(
                queries,
                keys,
                values,
                self.scale,
                0.0,
                self.window_size,
            );
        }

        let mask_ptr = local_mask
            .as_ref()
            .map(|m| m.as_ref().unwrap() as *const MlxArray)
            .unwrap_or(std::ptr::null());

        unsafe {
            mlxcel_core::layers::attention_from_ptr(
                queries,
                keys,
                values,
                self.scale,
                mask_ptr,
                0.0,
                self.window_size,
            )
        }
    }

    fn project_output(&self, attn_out: &MlxArray, b: i32, l: i32) -> UniquePtr<MlxArray> {
        let attn_out = mlxcel_core::transpose_axes(attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.n_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    fn project_kv(
        &self,
        x: &MlxArray,
        b: i32,
        l: i32,
        offset: i32,
        cache: &mut dyn CacheInterface,
        rope_offsets: Option<&[i32]>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let (raw_keys, raw_values) = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (_, k, v) = proj.forward(x);
                (k, Some(v))
            }
            AttentionProjection::Separate { k_proj, v_proj, .. } => {
                let raw_keys = k_proj.forward(x);
                let raw_values = if self.use_k_eq_v {
                    None
                } else {
                    Some(
                        v_proj
                            .as_ref()
                            .expect("Gemma4 attention expected v_proj for non-k_eq_v layer")
                            .forward(x),
                    )
                };
                (raw_keys, raw_values)
            }
            AttentionProjection::KvShared { .. } => {
                unreachable!(
                    "project_kv must not be called on a KV-shared layer; \
                     forward() should have taken the early-return shared_kv path"
                )
            }
        };

        // Fast path: on full-attention layers the K branch is the same
        // `reshape -> norm -> transpose -> full-head ProportionalRoPE` shape
        // as the Q branch, so it reuses `compiled_q_path_proportional` with
        // `n_kv_heads` and the k_norm weight.
        let k_norm = self
            .k_norm
            .as_ref()
            .expect("k_norm must be Some for non-KV-shared layers");
        let keys = if let Some(offsets) = rope_offsets {
            // Divergent batched MTP verify (issue #203): rotate each row's
            // new keys at the row's own logical positions, mirroring the
            // per-row query rotation in `forward`, through the same kernels
            // the uniform path uses, sliced per row.
            if let Some(ref freqs) = self.proportional_rope_freqs {
                let rotated_dims = 2 * ((self.proportional_partial_rotary_factor as f64
                    * self.head_dim as f64
                    / 2.0)
                    .floor() as i32)
                    .max(0);
                self.compiled_q_path_proportional_per_row(
                    &raw_keys,
                    &k_norm.weight,
                    k_norm.eps,
                    freqs,
                    self.n_kv_heads,
                    rotated_dims,
                    l,
                    offsets,
                )
            } else {
                let keys = mlxcel_core::reshape(&raw_keys, &[b, l, self.n_kv_heads, self.head_dim]);
                let keys = k_norm.forward(&keys);
                let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
                self.apply_rope_per_row(&keys, offsets)
            }
        } else if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2
                * ((self.proportional_partial_rotary_factor as f64 * self.head_dim as f64 / 2.0)
                    .floor() as i32)
                    .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &raw_keys,
                &k_norm.weight,
                freqs,
                k_norm.eps,
                self.n_kv_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let keys = mlxcel_core::reshape(&raw_keys, &[b, l, self.n_kv_heads, self.head_dim]);
            let keys = k_norm.forward(&keys);
            let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
            self.apply_rope(&keys, offset)
        };

        let raw_values_ref = raw_values
            .as_ref()
            .map(|values| values.as_ref().unwrap())
            .unwrap_or_else(|| raw_keys.as_ref().unwrap());
        let values = mlxcel_core::reshape(raw_values_ref, &[b, l, self.n_kv_heads, self.head_dim]);
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        cache.update_and_fetch(keys, values)
    }

    /// DiffusionGemma canvas (decoder-mode) attention (issue #217, additive
    /// seam). Mirrors `diffusion_gemma.language.Attention.__call__` with
    /// `decoder=True`, line by line:
    ///
    /// * Q/K are projected, normed, and rotated at `offset` (the encoder
    ///   length) through the SAME kernels the encoder path uses (the compiled
    ///   proportional fused chain on full-attention layers, the op-at-a-time
    ///   chain on sliding layers), so canvas numerics match the reference.
    /// * `values_raw = v_proj(x)` when the layer has a v_proj (sliding
    ///   layers), otherwise the raw K projection (`use_k_eq_v` full-attention
    ///   layers); `values = v_norm(values_raw)`.
    /// * The read-only `encoder_kv` prefix is concatenated in front of the
    ///   canvas K/V. On sliding layers the prefix is first trimmed to the
    ///   last `sliding_window - 1` positions (the upstream O(window) trim);
    ///   with the trim applied, the no-padding decoder mask is `None`
    ///   (canvas positions attend bidirectionally to everything that
    ///   remains), so SDPA runs maskless with `self.scale` (1.0).
    /// * The cache is NEVER updated; the canvas changes every denoising step.
    pub(crate) fn forward_canvas(
        &self,
        x: &MlxArray,
        encoder_kv: Option<(&MlxArray, &MlxArray)>,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let l = shape[1];

        let q_proj_out = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (q, _, _) = proj.forward(x);
                q
            }
            AttentionProjection::Separate { q_proj, .. }
            | AttentionProjection::KvShared { q_proj } => q_proj.forward(x),
        };

        let queries = if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2
                * ((self.proportional_partial_rotary_factor as f64 * self.head_dim as f64 / 2.0)
                    .floor() as i32)
                    .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &q_proj_out,
                &self.q_norm.weight,
                freqs,
                self.q_norm.eps,
                self.n_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let queries = mlxcel_core::reshape(&q_proj_out, &[b, l, self.n_heads, self.head_dim]);
            let queries = self.q_norm.forward(&queries);
            let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
            self.apply_rope(&queries, offset)
        };

        let (raw_keys, raw_values) = match &self.projection {
            AttentionProjection::Fused(proj) => {
                let (_, k, v) = proj.forward(x);
                (k, Some(v))
            }
            AttentionProjection::Separate { k_proj, v_proj, .. } => {
                let raw_keys = k_proj.forward(x);
                let raw_values = if self.use_k_eq_v {
                    None
                } else {
                    Some(
                        v_proj
                            .as_ref()
                            .expect("Gemma4 attention expected v_proj for non-k_eq_v layer")
                            .forward(x),
                    )
                };
                (raw_keys, raw_values)
            }
            AttentionProjection::KvShared { .. } => {
                unreachable!(
                    "forward_canvas must not be called on a KV-shared layer; \
                     DiffusionGemma forces num_kv_shared_layers == 0"
                )
            }
        };

        let k_norm = self
            .k_norm
            .as_ref()
            .expect("k_norm must be Some for non-KV-shared layers");
        let keys = if let Some(ref freqs) = self.proportional_rope_freqs {
            let rotated_dims = 2
                * ((self.proportional_partial_rotary_factor as f64 * self.head_dim as f64 / 2.0)
                    .floor() as i32)
                    .max(0);
            mlxcel_core::compiled_q_path_proportional(
                &raw_keys,
                &k_norm.weight,
                freqs,
                k_norm.eps,
                self.n_kv_heads,
                self.head_dim,
                rotated_dims,
                offset,
            )
        } else {
            let keys = mlxcel_core::reshape(&raw_keys, &[b, l, self.n_kv_heads, self.head_dim]);
            let keys = k_norm.forward(&keys);
            let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
            self.apply_rope(&keys, offset)
        };

        let raw_values_ref = raw_values
            .as_ref()
            .map(|values| values.as_ref().unwrap())
            .unwrap_or_else(|| raw_keys.as_ref().unwrap());
        let values = mlxcel_core::reshape(raw_values_ref, &[b, l, self.n_kv_heads, self.head_dim]);
        let values = self.v_norm.forward(&values);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let (keys, values) = if let Some((encoder_keys, encoder_values)) = encoder_kv {
            // Sliding layers: the canvas only sees the last
            // `sliding_window - 1` encoder positions, so drop the
            // out-of-window prefix before SDPA instead of scoring thousands
            // of positions the (implicit) mask would zero anyway. Safe for
            // the dense KVCache because offset == encoder_len always holds
            // (no trailing-invalid slots).
            let window_prefix = (self.window_size - 1).max(0);
            let encoder_len = mlxcel_core::array_shape(encoder_keys)[2];
            let (encoder_keys, encoder_values) =
                if self.window_size > 0 && encoder_len > window_prefix && offset >= encoder_len {
                    (
                        slice_axis(encoder_keys, 2, encoder_len - window_prefix, encoder_len),
                        slice_axis(encoder_values, 2, encoder_len - window_prefix, encoder_len),
                    )
                } else {
                    (
                        mlxcel_core::copy(encoder_keys),
                        mlxcel_core::copy(encoder_values),
                    )
                };
            (
                mlxcel_core::concatenate(&encoder_keys, &keys, 2),
                mlxcel_core::concatenate(&encoder_values, &values, 2),
            )
        } else {
            (keys, values)
        };

        // Full bidirectional attention over [trimmed encoder prefix, canvas]:
        // batch-1 with no padding means mask == None in the reference decoder
        // mask builder. `window_size` 0 keeps the dispatch on the plain
        // (non-windowed) SDPA path.
        let attn_out =
            mlxcel_core::layers::attention(&queries, &keys, &values, self.scale, None, 0.0, 0);
        self.project_output(&attn_out, b, l)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim_for_layer(layer_idx);
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_kv_heads_for_layer(layer_idx);
        let rope_params = config.rope_params_for_layer(layer_idx);
        let rope_dims = (head_dim as f32 * rope_params.partial_rotary_factor) as i32;
        // For `rope_type == "proportional"`, precompute the proportional
        // frequency table once per layer. The exponents are normalized by the
        // full head_dim, and the non-rotated tail is filled with `inf`,
        // matching upstream `ProportionalRoPE` semantics.
        // Other rope_types fall through to the historical partial-dim
        // `fast_rope` path (no precomputed freqs).
        let proportional_rope_freqs = if rope_params.rope_type == "proportional" {
            mlxcel_core::rope_proportional::compute_proportional_rope_freqs(
                head_dim,
                rope_params.partial_rotary_factor,
                rope_params.rope_theta,
                1.0,
            )
        } else {
            None
        };
        let proportional_partial_rotary_factor = rope_params.partial_rotary_factor;
        let use_k_eq_v = config.attention_k_eq_v && !config.is_sliding_layer(layer_idx);
        let first_kv_shared_idx = config.first_kv_shared_layer_idx();
        let is_kv_shared_layer = config.is_kv_shared_layer(layer_idx);

        let kv_shared_layer_index = if is_kv_shared_layer {
            let prev_layers = &config.layer_types[..first_kv_shared_idx];
            Some(
                prev_layers
                    .iter()
                    .rposition(|layer_type| layer_type == config.layer_types[layer_idx].as_str())
                    .ok_or_else(|| {
                        format!(
                            "Failed to locate KV-sharing source layer for Gemma4 layer {}",
                            layer_idx
                        )
                    })?,
            )
        } else {
            None
        };

        let store_full_length_kv = if !is_kv_shared_layer && first_kv_shared_idx > 0 {
            let prev_layers = &config.layer_types[..first_kv_shared_idx];
            prev_layers
                .iter()
                .rposition(|layer_type| layer_type == config.layer_types[layer_idx].as_str())
                .map(|idx| idx == layer_idx)
                .unwrap_or(false)
        } else {
            false
        };

        // Gemma 4 matches mlx-lm's separate Q/K/V projection path by default.
        // A concatenated QKV projection reduces QuantizedMatmul count, but on
        // 26B/31B decode it adds slice-heavy graphs and measures slower.
        //
        // KV-shared layers never compute their own K/V — they reuse the
        // full-length cache entries from an earlier non-shared layer (passed
        // in via the `shared_kv` argument to `forward`).  Constructing
        // k_proj/v_proj/k_norm for these layers wastes VRAM and mirrors what
        // upstream mlx-lm fixed in PR #1158 (commit 4f5cbd2).
        let enable_fused_qkv = std::env::var_os("MLXCEL_GEMMA4_ENABLE_FUSED_QKV").is_some();
        let projection = if is_kv_shared_layer {
            // Only q_proj is needed; K/V come from the shared KV cache.
            AttentionProjection::KvShared {
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
            }
        } else if use_k_eq_v {
            AttentionProjection::Separate {
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.k_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                v_proj: None,
            }
        } else if enable_fused_qkv {
            AttentionProjection::Fused(FusedQKVLinear::from_weights_separate(
                weights,
                prefix,
                config.group_size(),
                config.bits(),
                n_heads,
                n_kv_heads,
                head_dim,
            )?)
        } else {
            AttentionProjection::Separate {
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.q_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.k_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?,
                v_proj: Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.v_proj", prefix),
                    config.group_size(),
                    config.bits(),
                )?),
            }
        };

        // k_norm is not needed for KV-shared layers: those layers never call
        // project_kv() and use the shared K from an earlier layer's cache.
        let k_norm = if is_kv_shared_layer {
            None
        } else {
            Some(RMSNorm::new(
                get_weight_copy(weights, &format!("{}.k_norm.weight", prefix))?,
                config.rms_norm_eps,
            ))
        };

        Ok(Self {
            projection,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.o_proj", prefix),
                config.group_size(),
                config.bits(),
            )?,
            q_norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.q_norm.weight", prefix))?,
                config.rms_norm_eps,
            ),
            k_norm,
            v_norm: RMSNormNoScale::new(head_dim, config.rms_norm_eps),
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            rope_theta: rope_params.rope_theta,
            proportional_rope_freqs,
            proportional_partial_rotary_factor,
            scale: 1.0,
            is_kv_shared_layer,
            kv_shared_layer_index,
            store_full_length_kv,
            use_k_eq_v,
            window_size: if config.is_sliding_layer(layer_idx) {
                config.sliding_window as i32
            } else {
                0
            },
        })
    }
}

fn trim_mask_to_keys(mask: Option<&MlxArray>, keys: &MlxArray) -> Option<UniquePtr<MlxArray>> {
    let mask = mask?;
    let mask_shape = mlxcel_core::array_shape(mask);
    let key_shape = mlxcel_core::array_shape(keys);
    let key_len = key_shape[2];
    let mask_len = *mask_shape.last().unwrap_or(&0);
    if mask_len == key_len {
        Some(mlxcel_core::copy(mask))
    } else if mask_len > key_len {
        let start = mask_len - key_len;
        Some(slice_axis(mask, -1, start, mask_len))
    } else {
        // mask is smaller than key length — this happens when an external
        // mask was created with stale offset (e.g. during chunked prefill on
        // non-batching models).  Discard the undersized caller mask and let
        // the attention kernel fall back to its internal causal handling by
        // returning None, which will cause the caller to pass a null mask.
        tracing::warn!(
            mask_len,
            key_len,
            "trim_mask_to_keys: mask shorter than key length, discarding caller mask"
        );
        None
    }
}

pub struct DecoderLayer {
    pub(crate) self_attn: Attention,
    pub(crate) mlp: MLP,
    pub(crate) input_layernorm: RMSNorm,
    pub(crate) post_attention_layernorm: RMSNorm,
    pub(crate) pre_feedforward_layernorm: RMSNorm,
    pub(crate) post_feedforward_layernorm: RMSNorm,
    pub(crate) router: Option<Router>,
    pub(crate) experts: Option<Experts>,
    pub(crate) post_feedforward_layernorm_1: Option<RMSNorm>,
    pub(crate) pre_feedforward_layernorm_2: Option<RMSNorm>,
    pub(crate) post_feedforward_layernorm_2: Option<RMSNorm>,
    pub(crate) per_layer_input_gate: Option<UnifiedLinear>,
    pub(crate) per_layer_projection: Option<UnifiedLinear>,
    pub(crate) post_per_layer_input_norm: Option<RMSNorm>,
    pub(crate) layer_scalar: UniquePtr<MlxArray>,
    pub(crate) layer_type: String,
}

/// Optional sub-op timer used by `MLXCEL_PROFILE_LAYER_SUBOPS=1` inside
/// `DecoderLayer::forward`. Each `tick` evaluates the tensor (forcing a sync)
/// and prints `[SUBOP ...] layer=.. name=.. ms=..` to stderr.
struct SubopTimer {
    enabled: bool,
    layer_idx: usize,
    last: std::time::Instant,
}

impl SubopTimer {
    fn new(enabled: bool, layer_idx: usize) -> Self {
        Self {
            enabled,
            layer_idx,
            last: std::time::Instant::now(),
        }
    }

    fn tick(&mut self, name: &str, tensor: &MlxArray) {
        if !self.enabled {
            return;
        }
        mlxcel_core::eval(tensor);
        let dt = self.last.elapsed();
        eprintln!(
            "[SUBOP] layer={:02} name={} ms={:.4}",
            self.layer_idx,
            name,
            dt.as_secs_f64() * 1000.0
        );
        self.last = std::time::Instant::now();
    }
}

impl DecoderLayer {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        self.forward_with_profile(
            x,
            mask,
            cache,
            per_layer_input,
            shared_kv,
            usize::MAX,
            false,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_profile(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        per_layer_input: Option<&MlxArray>,
        shared_kv: Option<(&MlxArray, &MlxArray)>,
        layer_idx: usize,
        profile_subops: bool,
        divergent_rows: Option<&DivergentVerifyRows<'_>>,
    ) -> (
        UniquePtr<MlxArray>,
        Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    ) {
        let mut timer = SubopTimer::new(profile_subops, layer_idx);
        // Mirror mlx-lm's `residual = x` aliasing by holding a reference to
        // the prior stage's tensor instead of copying it. The earlier
        // `mlxcel_core::copy` calls produced a MLX Copy primitive and a fresh
        // UniquePtr<MlxArray> per layer; both are wasted work since the
        // residual is only *read* by the final `add`.
        let h_attn = self.input_layernorm.forward(x);
        timer.tick("input_layernorm", &h_attn);
        let (h_attn, stored_kv) =
            self.self_attn
                .forward(&h_attn, mask, cache, shared_kv, divergent_rows);
        timer.tick("self_attn", &h_attn);
        let h_attn = self.post_attention_layernorm.forward(&h_attn);
        timer.tick("post_attention_layernorm", &h_attn);
        let after_attn = mlxcel_core::add(x, &h_attn);
        timer.tick("attn_residual_add", &after_attn);

        let ffn_out = self.ffn_branch(&after_attn, &mut timer);

        let ffn_out = self.post_feedforward_layernorm.forward(&ffn_out);
        timer.tick("post_ffn_ln", &ffn_out);
        let after_ffn = mlxcel_core::add(&after_attn, &ffn_out);
        timer.tick("ffn_residual_add", &after_ffn);

        let h = if let (Some(gate_proj), Some(proj), Some(post_norm), Some(per_layer_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            // Fast path: when both gate and proj are quantized,
            // collapse the whole chain
            //   gate_proj → gelu_approx → mul(per_layer) → proj →
            //   post_norm → add(after_ffn)
            // into a single fused C++ call. The native-NVFP4 `global_scale`
            // sidecar (issue #693) is now folded into that call (issue #698):
            // the gate scale before the GeGLU activation and the proj scale on
            // the projected output, byte-identical to `apply_global_scale`.
            // `MLXCEL_DISABLE_FUSED_GLOBAL_SCALE` disables the fold, sending a
            // sidecar-carrying pair to the op-at-a-time chain below. So does a
            // multi-token (prefill) call: the C++ bridge falls back to an
            // uncompiled eager fold once a sidecar is present, which regresses
            // prefill versus the compiled op-at-a-time activation, so
            // multi-token inputs bypass the fused path regardless of the kill
            // switch (issue #698 follow-up).
            let combined = if let (Some(gate_qw), Some(proj_qw)) =
                (gate_proj.quantized_weight(), proj.quantized_weight())
                && per_layer_input_gate_fused_path_eligible(
                    gate_qw,
                    proj_qw,
                    !fused_global_scale_disabled(),
                    mlxcel_core::array_shape(x)[1] == 1,
                ) {
                unsafe {
                    mlxcel_core::compiled_per_layer_input_gate(
                        &after_ffn,
                        per_layer_input,
                        &gate_qw.weight,
                        &gate_qw.scales,
                        gate_qw.biases_ptr(),
                        &proj_qw.weight,
                        &proj_qw.scales,
                        proj_qw.biases_ptr(),
                        &post_norm.weight,
                        gate_qw.global_scale_ptr(),
                        proj_qw.global_scale_ptr(),
                        post_norm.eps,
                        gate_qw.group_size,
                        gate_qw.bits,
                        &gate_qw.mode,
                    )
                }
            } else {
                let gate = gate_proj.forward(&after_ffn);
                let gated = mlxcel_core::compiled_geglu_approx_activation(&gate, per_layer_input);
                let gate = proj.forward(&gated);
                let gate = post_norm.forward(&gate);
                mlxcel_core::add(&after_ffn, &gate)
            };
            timer.tick("per_layer_input_gate", &combined);
            combined
        } else {
            after_ffn
        };

        let h = mlxcel_core::multiply(&h, &self.layer_scalar);
        timer.tick("layer_scalar", &h);
        (h, stored_kv)
    }

    /// Shared feed-forward stage: dense MLP branch plus (when present) the
    /// MoE branch, combined exactly like the upstream Gemma 4 layer. Extracted
    /// from [`Self::forward_with_profile`] verbatim so the DiffusionGemma
    /// encoder/canvas forwards (issue #217) reuse the identical op sequence;
    /// the hot autoregressive path calls this with its own timer and is
    /// byte-identical to the pre-extraction code.
    fn ffn_branch(&self, after_attn: &MlxArray, timer: &mut SubopTimer) -> UniquePtr<MlxArray> {
        if let (Some(router), Some(experts)) = (&self.router, &self.experts) {
            let h1 = self.pre_feedforward_layernorm.forward(after_attn);
            timer.tick("pre_ffn_ln_shared_mlp", &h1);
            let h1 = self.mlp.forward(&h1);
            timer.tick("shared_mlp", &h1);
            let h1 = self
                .post_feedforward_layernorm_1
                .as_ref()
                .expect("Missing Gemma4 MoE post_feedforward_layernorm_1")
                .forward(&h1);
            timer.tick("post_shared_mlp_ln", &h1);

            let (top_k_indices, top_k_weights) = router.forward(after_attn);
            timer.tick("router", &top_k_indices);
            let h2 = self
                .pre_feedforward_layernorm_2
                .as_ref()
                .expect("Missing Gemma4 MoE pre_feedforward_layernorm_2")
                .forward(after_attn);
            timer.tick("pre_moe_ln", &h2);
            let h2 = experts.forward(&h2, &top_k_indices, &top_k_weights);
            timer.tick("experts", &h2);
            let h2 = self
                .post_feedforward_layernorm_2
                .as_ref()
                .expect("Missing Gemma4 MoE post_feedforward_layernorm_2")
                .forward(&h2);
            timer.tick("post_moe_ln", &h2);
            let combined = mlxcel_core::add(&h1, &h2);
            timer.tick("moe_shared_add", &combined);
            combined
        } else {
            let h_norm = self.pre_feedforward_layernorm.forward(after_attn);
            timer.tick("pre_ffn_ln", &h_norm);
            let out = self.mlp.forward(&h_norm);
            timer.tick("mlp", &out);
            out
        }
    }

    /// DiffusionGemma encoder-mode layer forward (issue #217, additive seam).
    ///
    /// Identical to [`Self::forward`] for a layer without per-layer inputs or
    /// KV sharing, except the final output scalar is the caller-provided
    /// `layer_scalar` (the checkpoint's per-layer ENCODER scalar at
    /// `model.encoder.language_model.layers.N.layer_scalar`) instead of the
    /// layer's stored decoder `layer_scalar`. Mirrors the
    /// `layer_scalar` override parameter on the upstream
    /// `diffusion_gemma.language.DecoderLayer.__call__`.
    pub(crate) fn forward_encoder_with_scalar(
        &self,
        x: &MlxArray,
        mask: Option<&MlxArray>,
        cache: &mut dyn CacheInterface,
        layer_scalar: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mut timer = SubopTimer::new(false, usize::MAX);
        let h_attn = self.input_layernorm.forward(x);
        let (h_attn, _stored_kv) = self.self_attn.forward(&h_attn, mask, cache, None, None);
        let h_attn = self.post_attention_layernorm.forward(&h_attn);
        let after_attn = mlxcel_core::add(x, &h_attn);

        let ffn_out = self.ffn_branch(&after_attn, &mut timer);
        let ffn_out = self.post_feedforward_layernorm.forward(&ffn_out);
        let after_ffn = mlxcel_core::add(&after_attn, &ffn_out);

        mlxcel_core::multiply(&after_ffn, layer_scalar)
    }

    /// DiffusionGemma canvas (decoder-mode) layer forward (issue #217,
    /// additive seam).
    ///
    /// The canvas hidden states attend bidirectionally within themselves and
    /// to the read-only `encoder_kv` prefix (the cache is never written);
    /// everything after attention is the standard Gemma 4 layer with the
    /// layer's stored DECODER `layer_scalar`. Mirrors the upstream
    /// `decoder=True` path of `diffusion_gemma.language.DecoderLayer`.
    pub(crate) fn forward_canvas(
        &self,
        x: &MlxArray,
        encoder_kv: Option<(&MlxArray, &MlxArray)>,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let mut timer = SubopTimer::new(false, usize::MAX);
        let h_attn = self.input_layernorm.forward(x);
        let h_attn = self.self_attn.forward_canvas(&h_attn, encoder_kv, offset);
        let h_attn = self.post_attention_layernorm.forward(&h_attn);
        let after_attn = mlxcel_core::add(x, &h_attn);

        let ffn_out = self.ffn_branch(&after_attn, &mut timer);
        let ffn_out = self.post_feedforward_layernorm.forward(&ffn_out);
        let after_ffn = mlxcel_core::add(&after_attn, &ffn_out);

        mlxcel_core::multiply(&after_ffn, &self.layer_scalar)
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self, String> {
        let enable_moe = config.enable_moe_block;
        let has_per_layer_input = config.hidden_size_per_layer_input > 0;

        Ok(Self {
            self_attn: Attention::from_weights(
                weights,
                config,
                layer_idx,
                &format!("{}.self_attn", prefix),
            )?,
            mlp: MLP::from_weights(weights, config, layer_idx, &format!("{}.mlp", prefix))?,
            input_layernorm: RMSNorm::new(
                get_weight_copy(weights, &format!("{}.input_layernorm.weight", prefix))?,
                config.rms_norm_eps,
            ),
            post_attention_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            pre_feedforward_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.pre_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            post_feedforward_layernorm: RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.post_feedforward_layernorm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ),
            router: if enable_moe {
                Some(Router::from_weights(
                    weights,
                    config,
                    &format!("{}.router", prefix),
                )?)
            } else {
                None
            },
            experts: if enable_moe {
                Some(Experts::from_weights(
                    weights,
                    config,
                    &format!("{}.experts", prefix),
                )?)
            } else {
                None
            },
            post_feedforward_layernorm_1: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_feedforward_layernorm_1.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            pre_feedforward_layernorm_2: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.pre_feedforward_layernorm_2.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            post_feedforward_layernorm_2: if enable_moe {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_feedforward_layernorm_2.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            per_layer_input_gate: if has_per_layer_input {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_input_gate", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            },
            per_layer_projection: if has_per_layer_input {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_projection", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            },
            post_per_layer_input_norm: if has_per_layer_input {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.post_per_layer_input_norm.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            },
            layer_scalar: get_weight_copy(weights, &format!("{}.layer_scalar", prefix))?,
            layer_type: config.layer_type(layer_idx).to_string(),
        })
    }
}

pub struct Gemma4TextModel {
    pub(crate) embed_tokens: UnifiedEmbedding,
    pub(crate) embed_tokens_per_layer: Option<UnifiedEmbedding>,
    pub(crate) per_layer_model_projection: Option<UnifiedLinear>,
    pub(crate) per_layer_projection_scale: f32,
    pub(crate) per_layer_projection_norm: Option<RMSNorm>,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: RMSNorm,
    pub(crate) config: TextConfig,
}

/// Output sinks for the Gemma 4 MTP speculative-decoding target hooks
///
/// All fields are `Option` so callers that do not need a hook pay zero cost
/// on the hot path — the `forward_with_speculative_sinks` consumer only
/// allocates and assigns when the corresponding sink is requested.
///
/// - [`hidden_sink`](Self::hidden_sink): When `Some`, the target appends the
///   last decoder layer's hidden state **before the final RMSNorm**
///   (matching upstream HF `_can_record_outputs={"hidden_states":
///   Gemma4TextDecoderLayer}` semantics). When the caller also supplies a
///   non-empty `capture_layer_ids` list, the sink instead receives the
///   captured layer outputs in iteration order (one entry per matching
///   layer index). Shape per entry: `[B, L, hidden_size]`, preserving the
///   model's native dtype (bf16/f16).
///
/// - [`shared_kv_sink`](Self::shared_kv_sink): When `Some`, the target
///   inserts the K/V slabs of the **last** non-KV-shared full-attention
///   layer (`"full_attention"` key) and the **last** non-KV-shared
///   sliding-attention layer (`"sliding_attention"` key). These are the
///   exact slabs the Gemma 4 MTP drafter binds against — its 4-layer
///   assistant transformer reads them in lieu of a private KV cache. Shape
///   per K and V: `[B, num_kv_heads, kv_len, head_dim]` in the model's
///   native dtype.
///
/// Used by: future `Gemma4AssistantDraftModel` and
/// `MtpGenerator`.
#[derive(Default)]
pub struct Gemma4SpeculativeSinks {
    pub hidden_sink: Option<Vec<UniquePtr<MlxArray>>>,
    pub shared_kv_sink: Option<HashMap<String, (UniquePtr<MlxArray>, UniquePtr<MlxArray>)>>,
}

impl Gemma4SpeculativeSinks {
    /// Sink set that captures the target's last layer hidden state (matches
    /// `return_hidden=True` in the upstream Python signature).
    #[must_use]
    pub fn with_hidden() -> Self {
        Self {
            hidden_sink: Some(Vec::new()),
            shared_kv_sink: None,
        }
    }

    /// Sink set that captures the last full-attention + last sliding-
    /// attention K/V slabs (matches `return_shared_kv=True` upstream).
    #[must_use]
    pub fn with_shared_kv() -> Self {
        Self {
            hidden_sink: None,
            shared_kv_sink: Some(HashMap::new()),
        }
    }

    /// Sink set that captures both hidden and shared K/V — the common
    /// `Gemma4AssistantDraftModel.bind` call pattern.
    #[must_use]
    pub fn with_hidden_and_shared_kv() -> Self {
        Self {
            hidden_sink: Some(Vec::new()),
            shared_kv_sink: Some(HashMap::new()),
        }
    }
}

impl Gemma4TextModel {
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_speculative_sinks(
            input_ids,
            input_embeddings,
            caches,
            mask,
            per_layer_inputs,
            None,
            None,
            false,
            None,
            None,
            None,
        )
    }

    /// Forward pass with optional speculative-decoding hooks for the Gemma 4
    /// MTP target path.
    ///
    /// When `sinks` is `None`, behaves exactly like [`Self::forward`] (zero
    /// overhead beyond the wrapping struct).
    ///
    /// When `sinks` is `Some(&mut sinks)`:
    /// - If `sinks.hidden_sink` is `Some` and `capture_layer_ids` is empty
    ///   or `None`, the LAST decoder layer's pre-norm hidden state is
    ///   appended exactly once (matching upstream Python semantics:
    ///   `hidden_sink.append(h)` after the layer loop, before `self.norm`).
    /// - If `sinks.hidden_sink` is `Some` and `capture_layer_ids` is
    ///   non-empty, hidden states at the listed layer indices are appended
    ///   in the order they are produced (the layer loop's natural ascending
    ///   index order).
    /// - If `sinks.shared_kv_sink` is `Some`, the K and V slabs returned by
    ///   `Attention::forward` on the last non-KV-shared full-attention layer
    ///   and the last non-KV-shared sliding-attention layer are inserted
    ///   under the keys `"full_attention"` and `"sliding_attention"`. Each
    ///   value is the `(K, V)` tuple in `[B, num_kv_heads, kv_len, head_dim]`
    ///   shape and the model's native dtype.
    ///
    /// The hidden-state and shared-K/V captures intentionally preserve the
    /// model's native bf16/f16 dtype — no f32 promotion (per
    /// `docs/apple-silicon-precision.md`).
    ///
    /// `per_row_valid_end` (issue #163) is the per-row logical valid key end
    /// (`left_padding[r] + kv_valid_len[r]`) supplied only by the batched MTP
    /// verify forward. After divergent accepts a shorter row's valid end lags
    /// the physical cache offset; the keys in `[per_row_valid_end[r], offset)`
    /// are that row's stale rejected-draft / zeroed tail. When `Some`, those gap
    /// columns are excluded from the derived masks via [`mask_stale_key_gap`] so
    /// each row's logits match its standalone B = 1 run (whose exact cache trim
    /// has no such gap); the current window's keys `[offset, offset + l)` keep
    /// base causal semantics. `None` (every non-batched-verify caller) is a
    /// byte-identical no-op, as is a uniform round where every `ve[r] == offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        capture_layer_ids: Option<&[usize]>,
        mut sinks: Option<&mut Gemma4SpeculativeSinks>,
        skip_final_norm: bool,
        bidirectional_block_ids: Option<&MlxArray>,
        left_padding: Option<&[i32]>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        // When `input_embeddings` is supplied (e.g. from the VLM path where
        // vision/audio features have already been merged into the embedding
        // stream), the caller is responsible for applying the
        // `sqrt(hidden_size)` embed scale to the text portion *before*
        // merging. Scaling here would double-scale the text tokens and
        // incorrectly scale image/audio features that are already in the
        // language-model embedding space..
        let mut h = if let Some(embeddings) = input_embeddings {
            mlxcel_core::copy(embeddings)
        } else {
            let embeds = self.embed_tokens.forward(input_ids);
            mlxcel_core::multiply_scalar(&embeds, (self.config.hidden_size as f32).sqrt())
        };

        let shape = mlxcel_core::array_shape(&h);
        let b = shape[0];
        let l = shape[1];

        let per_layer_inputs = if self.config.hidden_size_per_layer_input > 0 {
            let inputs = if let Some(inputs) = per_layer_inputs {
                mlxcel_core::copy(inputs)
            } else {
                self.get_per_layer_inputs(input_ids)
            };
            Some(self.project_per_layer_inputs(&h, Some(&inputs)))
        } else {
            None
        };

        // Ragged batched MTP (variable-length B>1 burst) threads the per-row
        // leading-padding column count so EVERY verify step masks each row's
        // `[0, left_padding[r])` resident prompt padding. Without this the
        // verify query attends the prompt's padding K/V (token 0) and breaks
        // greedy parity for the most-left-padded row (the error scales with
        // `left_padding[r]`). This MUST be honoured for single-token decode
        // steps (`l == 1`) too, not only multi-token verify blocks — the
        // padding stays resident in both the unbounded full-attention cache and
        // the MTP-buffered sliding cache, so an `l == 1` step that took the
        // `mask == None` fast path would silently attend it.
        let has_padding = left_padding
            .map(|lp| lp.iter().any(|&p| p > 0))
            .unwrap_or(false);

        // Divergent batched MTP verify round (issue #203): some row's logical
        // valid end lags the shared physical cache offset after mixed
        // speculative accepts. The finalize-side compaction
        // (`rollback_speculative_cache_divergent`) keeps each row's cache
        // content a contiguous prefix `[0, ve[r])`, so the exact per-row mask
        // is: padding `[0, lp[r])` and the stale gap `[ve[r], offset)` are
        // blocked, history `[lp[r], ve[r])` and the causal window prefix are
        // visible, with the sliding band evaluated at each row's LOGICAL
        // positions. Queries/keys rotate at per-row logical positions via
        // `rope_offsets` below. Uniform rounds (`ve[r] == offset` for every
        // row, always true until the first divergent accept) skip this branch
        // entirely and stay byte-identical to the pre-#203 path.
        let global_offset_pre = first_present_cache_offset(caches);
        let divergent_verify = mask.is_none()
            && !mtp_divergent_fix_disabled()
            && per_row_valid_end
                .map(|ve| ve.iter().any(|&v| v != global_offset_pre))
                .unwrap_or(false);
        // Per-row context for the divergent verify path (issue #203): the
        // attention layers rotate queries/keys at per-row logical positions
        // and run attention per row over each row's exact contiguous K/V, so
        // no batch-level masks are needed (and the SDPA reduction matches the
        // row's standalone B = 1 run bitwise).
        let divergent_rows: Option<DivergentVerifyRows<'_>> = if divergent_verify {
            let ve = per_row_valid_end.expect("divergent_verify implies Some(per_row_valid_end)");
            tracing::debug!(
                l,
                o_pre = global_offset_pre,
                ?ve,
                ?left_padding,
                "gemma4 batched MTP divergent verify round"
            );
            let lp = match left_padding {
                Some(lp) => lp.to_vec(),
                None => vec![0; ve.len()],
            };
            Some(DivergentVerifyRows { ve, lp })
        } else {
            None
        };

        let (global_mask, sliding_mask) = if let Some(ctx) = divergent_rows.as_ref() {
            let sliding_offset = first_cache_offset(caches, "sliding_attention");
            let window = self.config.sliding_window as i32;
            (
                Some(build_divergent_verify_mask(
                    l,
                    global_offset_pre,
                    ctx.ve,
                    &ctx.lp,
                    None,
                )),
                Some(build_divergent_verify_mask(
                    l,
                    sliding_offset,
                    ctx.ve,
                    &ctx.lp,
                    Some(window),
                )),
            )
        } else if let Some(mask) = mask {
            (Some(mlxcel_core::copy(mask)), Some(mlxcel_core::copy(mask)))
        } else if has_padding {
            // Resident prompt padding present (ragged burst). Build per-row
            // left-padding-aware masks for ANY query width, including `l == 1`.
            //
            // These offsets stay monotonic (NOT `live_len`): this is the
            // batched-MTP regime, which never front-compacts either cache kind
            // (`slot_base == 0`, so `live_len == offset` here), and the same
            // offset value is the column-coordinate base that the
            // `per_row_valid_end` stale-gap masking below compares against
            // (`mask_stale_key_gap`). The `--max-kv-size` trim cannot reach
            // these model-owned caches, so there is no live/monotonic split to
            // reconcile on this path. See issue #430.
            let global_offset = first_cache_offset(caches, "full_attention");
            let sliding_offset = first_cache_offset(caches, "sliding_attention");
            let window = self.config.sliding_window as i32;
            let lp = left_padding.expect("has_padding implies Some");

            // Full attention: unbounded KVCache keeps the padding at columns
            // `[0, lp)` for the whole run; mask it with the plain left-padding
            // causal mask sized to the full key axis (`l + global_offset`).
            let global_mask = create_causal_mask_with_left_padding(l, global_offset, lp);

            // Sliding attention: the MTP rollback buffer (`buffer_size`) keeps
            // the cache *uncompacted* far past the bare `sliding_window` — its
            // logical capacity is `sliding_window + buffer_size` — so the
            // resident prompt padding survives at columns `[0, lp)` long after
            // `sliding_offset + l > sliding_window`. The previous gate
            // (`sliding_offset + l <= window` -> windowed-left-padding mask,
            // else a padding-UNAWARE plain windowed mask) therefore stopped
            // masking the padding exactly when it was still resident, leaking
            // `lp` padding keys into the most-padded row's window every verify
            // step. While the buffer has not compacted, the key axis is the full
            // `[0, sliding_offset + l)` (no eviction), so the windowed
            // left-padding mask — which enforces both the sliding-window band
            // (`tril(offset)` ∩ `triu(offset - window + 1)`) AND the `[0, lp)`
            // padding filter — is the correct mask for `sliding_offset + l >
            // window` too. The eligible regime (`max_prompt_len <=
            // sliding_window`, realistic output lengths) never reaches the
            // buffer-compaction point, so the full key axis always holds; if the
            // cache ever did compact, the oldest keys (padding first) would be
            // evicted and `trim_mask_to_keys` would crop the mask to the
            // surviving (padding-free) suffix.
            let sliding_mask = create_causal_mask_with_window_and_left_padding(
                l,
                sliding_offset,
                Some(window),
                lp,
            );
            (Some(global_mask), Some(sliding_mask))
        } else if l > DENSE_PREFILL_MASK_MAX_TOKENS
            && bidirectional_block_ids.is_none()
            && per_row_valid_end.is_none()
        {
            // Long text-only prefill: leave both masks `None` so `attend`
            // routes through `causal_attention`'s flag paths instead of
            // shipping dense `[l, l + offset]` f32 masks through every layer.
            // At 32k tokens the two retained masks alone are ~8 GiB and their
            // construction transients more, on top of the O(heads*L^2) score
            // materialization they used to feed (issue #672). Semantics are
            // unchanged: full-attention layers (`window_size == 0`) take the
            // bottom-right-aligned causal path (row `q` attends `k <= q +
            // offset`, exactly `create_causal_mask(l, live_len)`), and
            // sliding layers take the over-window prefill path, which builds
            // the same `create_causal_mask_with_window_full(l, offset,
            // window)` mask per layer as a transient instead of retaining it
            // for the whole forward. Short prefills and every verify /
            // padding / vision-overlay shape keep the dense masks below,
            // byte-identical to the previous behavior.
            (None, None)
        } else if l > 1 {
            // Non-ragged prefill / multi-token verify: derive both masks from
            // the cache's live window (`live_len`), not the monotonic
            // `offset`. Under `--max-kv-size`, `trim_front` advances the
            // full-attention `KVCache`'s `live_start` while `offset` keeps
            // growing for RoPE, so `update_and_fetch` returns only `live_len`
            // keys; an `offset`-sized mask would be wider than the returned
            // K/V and trip `broadcast_shapes`. With no trim, `live_len ==
            // offset`, so this stays byte-identical to the pre-ragged
            // behaviour. See issue #430 (mirrors #419/#420, #421/#422).
            let global_live_len = first_cache_live_len(caches, "full_attention");
            let sliding_live_len = first_cache_live_len(caches, "sliding_attention");
            let window = self.config.sliding_window as i32;
            // Shared helper (hoisted in #410) gates the full mask on
            // `l > window && sliding_offset == 0`; the old local copy gated on
            // `l > window` alone. The gates differ only when
            // `l > window && sliding_offset > 0`. In that case RotatingKVCache
            // trims to exactly `window` keys, and `trim_mask_to_keys` crops the
            // full `[l, l+offset]` mask to its trailing `window` columns -- the
            // same band as the new clamped output (`q-l+1 <= k <= q-l+window`,
            // independent of offset). Old-trimmed equals new for every input,
            // so the migration is behaviour-preserving.
            (
                Some(create_causal_mask(l, global_live_len)),
                Some(create_sliding_window_prefill_mask(
                    l,
                    sliding_live_len,
                    window,
                )),
            )
        } else {
            // Ordinary single-token decode with no resident padding: the
            // attention kernels derive their own causal / windowed masks.
            (None, None)
        };

        // Per-row valid-length tail exclusion (issue #163). After divergent
        // accepts in a batched verify round, row `r`'s logical valid key end
        // `per_row_valid_end[r]` lags the physical cache offset (the global max),
        // so the offset-derived mask would let it attend the stale rejected-draft
        // K/V (full-attention `Cache::Standard`, never zeroed) and the zeroed
        // phantom tail (sliding `Cache::Rotating`) in `[ve[r], offset)`. The
        // B = 1 reference trims its cache exactly and has no such gap, so masking
        // the gap moves the batched logits onto the B = 1 semantics; it can only
        // improve parity. A uniform round (every `ve[r] == offset`, always true
        // for the first verify after prefill) is a byte-identical no-op.
        // (Skipped when the divergent branch above already built exact
        // per-row masks; the gap is part of those masks.)
        let (global_mask, sliding_mask) = if let Some(ve) = per_row_valid_end
            && !divergent_verify
        {
            // Full-attention family: the unbounded `Cache::Standard` keeps each
            // row's rejected-draft K/V resident at `[ve[r], global_offset)`.
            let global_offset = first_cache_offset(caches, "full_attention");
            let global_mask = if ve.iter().any(|&v| v < global_offset) {
                // Materialize the base when the fast branch left it `None`
                // (covers the `l == 1` (None, None) decode branch).
                let base = global_mask.unwrap_or_else(|| create_causal_mask(l, global_offset));
                Some(mask_stale_key_gap(&base, ve, global_offset))
            } else {
                global_mask
            };

            // Sliding family: only safe when the sliding mask's key axis is the
            // FULL uncompacted `[0, sliding_offset + l)` so column `k` maps 1:1
            // to logical key position `k`. That holds in (1) the `has_padding`
            // branch (the MTP rotating buffer never compacts in the eligible
            // regime) and (2) any branch with `sliding_offset + l <= window`
            // (the windowed mask is uncapped: `sliding_effective_offset ==
            // sliding_offset`, plus the materialize-from-None path). If the axis
            // may be capped we skip the sliding gap penalty and leave the
            // rotating cache's `zero_partial_accept_tail` to approximate it (the
            // eligible regime `max_prompt_len <= sliding_window` never caps).
            let sliding_offset = first_cache_offset(caches, "sliding_attention");
            let window = self.config.sliding_window as i32;
            let sliding_axis_full = has_padding || sliding_offset + l <= window;
            let sliding_mask = if sliding_axis_full && ve.iter().any(|&v| v < sliding_offset) {
                let base = sliding_mask.unwrap_or_else(|| {
                    create_causal_mask_with_window(l, sliding_offset, Some(window))
                });
                Some(mask_stale_key_gap(&base, ve, sliding_offset))
            } else {
                sliding_mask
            };

            (global_mask, sliding_mask)
        } else {
            (global_mask, sliding_mask)
        };

        // Gemma 4 Unified blockwise bidirectional overlay. When the caller
        // (the `gemma4_unified` runtime) supplies per-position vision block
        // ids during prefill, allow bidirectional attention *within* each
        // image/video span on both the full-attention and sliding-window base
        // masks. Text↔text, text↔vision and cross-block pairs keep the causal
        // (and, for sliding layers, windowed) base value. Decode (`l == 1`)
        // leaves the masks `None` and is unaffected.
        let (global_mask, sliding_mask) = match bidirectional_block_ids {
            Some(block_ids) if l > 1 => (
                global_mask
                    .as_ref()
                    .map(|m| overlay_block_bidirectional(m.as_ref().unwrap(), block_ids)),
                sliding_mask
                    .as_ref()
                    .map(|m| overlay_block_bidirectional(m.as_ref().unwrap(), block_ids)),
            ),
            _ => (global_mask, sliding_mask),
        };

        let mut shared_kv_store: HashMap<usize, (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32)> =
            HashMap::new();
        let n_layers = self.layers.len();

        // Per-layer profiling: set `MLXCEL_PROFILE_PER_LAYER=1` to dump a
        // `[LAYER xx] type=... ms=...` line to stderr for every layer on
        // every call. Forces `eval` before and after each layer so the
        // measured times include all kernels queued inside that layer.
        // Distorts absolute decode throughput (short-circuits graph
        // fusion), so use it for relative per-layer breakdowns only.
        //
        // `MLXCEL_PROFILE_LAYER_SUBOPS=1` extends this by emitting a
        // `[SUBOP] layer=xx name=.. ms=..` line per sub-step inside the
        // layer (input_ln, self_attn, post_ln, MoE router/experts, etc.).
        let profile_per_layer = std::env::var("MLXCEL_PROFILE_PER_LAYER").is_ok();
        let profile_layer_build = std::env::var("MLXCEL_PROFILE_LAYER_BUILD").is_ok();
        let profile_subops = std::env::var("MLXCEL_PROFILE_LAYER_SUBOPS").is_ok();

        // Speculative-decoding capture. The `capture_set` mirrors
        // upstream's `set(capture_layer_ids)`; when non-empty the
        // `hidden_sink` is appended to inside the loop at each matching idx,
        // otherwise the sink receives the final pre-norm `h` after the loop.
        let capture_set: Option<std::collections::HashSet<usize>> =
            capture_layer_ids.map(|ids| ids.iter().copied().collect());
        let has_capture_layers = capture_set.as_ref().is_some_and(|s| !s.is_empty());

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = caches[i].as_interface();
            let mut shared_kv = None;

            if layer.self_attn.is_kv_shared_layer
                && let Some(ref_idx) = layer.self_attn.kv_shared_layer_index
                && let Some((keys, values, ref_offset)) = shared_kv_store.get(&ref_idx)
            {
                cache.set_offset(*ref_offset);
                shared_kv = Some((keys.as_ref().unwrap(), values.as_ref().unwrap()));
            }

            let local_mask = match layer.layer_type.as_str() {
                "full_attention" => global_mask.as_ref().map(|m| m.as_ref().unwrap()),
                _ => sliding_mask.as_ref().map(|m| m.as_ref().unwrap()),
            };

            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                slice_layer_input(
                    inputs,
                    i as i32,
                    b,
                    l,
                    self.config.hidden_size_per_layer_input as i32,
                )
            });

            let pre_offset = cache.offset();
            let layer_start = if profile_per_layer {
                mlxcel_core::eval(&h);
                Some(std::time::Instant::now())
            } else {
                None
            };
            let layer_build_start = if profile_layer_build {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let (next_h, stored_kv) = layer.forward_with_profile(
                &h,
                local_mask,
                cache,
                layer_input.as_ref().map(|arr| arr.as_ref().unwrap()),
                shared_kv,
                i,
                profile_subops,
                divergent_rows.as_ref(),
            );
            h = next_h;
            if let Some(start) = layer_build_start {
                eprintln!(
                    "[LAYER_BUILD {:02}] type={} seq_len={} ms={:.3}",
                    i,
                    layer.layer_type,
                    l,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            if let Some(start) = layer_start {
                mlxcel_core::eval(&h);
                eprintln!(
                    "[LAYER {:02}] type={} seq_len={} ms={:.3}",
                    i,
                    layer.layer_type,
                    l,
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            if let Some((keys, values)) = stored_kv {
                shared_kv_store.insert(i, (keys, values, pre_offset));
            }

            // Per-layer hidden capture (`return_hidden` with explicit
            // `capture_layer_ids`). Mirrors upstream Python's
            // `if hidden_sink is not None and idx in capture_set:
            //     hidden_sink.append(h)`. `mlxcel_core::copy` is a shallow
            // MLX-array clone — it does not allocate device memory or
            // promote dtype.
            if has_capture_layers
                && let Some(s) = sinks.as_mut()
                && let Some(sink) = s.hidden_sink.as_mut()
                && let Some(set) = capture_set.as_ref()
                && set.contains(&i)
            {
                sink.push(mlxcel_core::copy(&h));
            }

            pipeline_hint(&h, i, n_layers);
        }

        // Final hidden capture (`return_hidden=True` without
        // `capture_layer_ids`). Matches HF's
        // `_can_record_outputs={"hidden_states": Gemma4TextDecoderLayer}`
        // contract — last decoder layer output captured BEFORE the final
        // RMSNorm, in the model's native dtype.
        if !has_capture_layers
            && let Some(s) = sinks.as_mut()
            && let Some(sink) = s.hidden_sink.as_mut()
        {
            sink.push(mlxcel_core::copy(&h));
        }

        // Shared K/V capture (`return_shared_kv=True`). The Rust path stores
        // K/V only on the LAST non-KV-shared layer of each type (i.e. the
        // sources that all KV-shared layers reference) — exactly the slabs
        // the Gemma 4 MTP drafter binds against. Iterate `shared_kv_store`
        // (small: at most two entries) and key the sink by the source
        // layer's type string (`"full_attention"` / `"sliding_attention"`).
        // The store is drained so we move ownership of the K/V handles into
        // the sink rather than paying for a redundant `mlxcel_core::copy`.
        if let Some(s) = sinks.as_mut()
            && let Some(sink) = s.shared_kv_sink.as_mut()
        {
            for (idx, (keys, values, _ref_offset)) in shared_kv_store.drain() {
                let layer_type = self.config.layer_type(idx).to_string();
                sink.insert(layer_type, (keys, values));
            }
        }

        if skip_final_norm {
            h
        } else {
            self.norm.forward(&h)
        }
    }

    pub(crate) fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let embedded = self
            .embed_tokens_per_layer
            .as_ref()
            .expect("Gemma4 per-layer embeddings missing")
            .forward(input_ids);
        let embedded = mlxcel_core::multiply_scalar(
            &embedded,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );

        let shape = mlxcel_core::array_shape(input_ids);
        mlxcel_core::reshape(
            &embedded,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    pub(crate) fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let projected = self
            .per_layer_model_projection
            .as_ref()
            .expect("Gemma4 per_layer_model_projection missing")
            .forward(inputs_embeds);
        let projected = mlxcel_core::multiply_scalar(&projected, self.per_layer_projection_scale);
        let shape = mlxcel_core::array_shape(inputs_embeds);
        let projected = mlxcel_core::reshape(
            &projected,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .expect("Gemma4 per_layer_projection_norm missing")
            .forward(&projected);

        if let Some(per_layer_inputs) = per_layer_inputs {
            let sum = mlxcel_core::add(&projected, per_layer_inputs);
            mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2)
        } else {
            projected
        }
    }

    pub fn from_weights(
        weights: &WeightMap,
        config: &TextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            &format!("{}.embed_tokens", prefix),
            config.group_size(),
            config.bits(),
        )?;

        let embed_tokens_per_layer = if config.hidden_size_per_layer_input > 0 {
            Some(UnifiedEmbedding::from_weights(
                weights,
                &format!("{}.embed_tokens_per_layer", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let per_layer_projection_scale = (config.hidden_size as f32).powf(-0.5);
        let per_layer_model_projection = if config.hidden_size_per_layer_input > 0 {
            Some(UnifiedLinear::from_weights(
                weights,
                &format!("{}.per_layer_model_projection", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let per_layer_projection_norm = if config.hidden_size_per_layer_input > 0 {
            Some(RMSNorm::new(
                get_weight_copy(
                    weights,
                    &format!("{}.per_layer_projection_norm.weight", prefix),
                )?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                config,
                layer_idx,
                &format!("{}.layers.{}", prefix, layer_idx),
            )?);
        }

        let norm = RMSNorm::new(
            get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
            config.rms_norm_eps,
        );

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_scale,
            per_layer_projection_norm,
            layers,
            norm,
            config: config.clone(),
        })
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.config
            .layer_types
            .iter()
            .map(|layer_type| {
                if layer_type == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.config.sliding_window as i32))
                }
            })
            .collect()
    }
}

/// Safety valve for the issue #203 divergent-round batched MTP machinery
/// (per-row logical RoPE + exact per-row masks + compacting rollback).
/// `MLXCEL_MTP_DISABLE_DIVERGENT_FIX=1` restores the pre-#203 behavior
/// (shared-physical-offset rotation + #163 stale-gap masks + global-max
/// trim). Used by the parity gates to validate that a structural positional
/// defect fails the jitter-aware parity check, and as an operational escape
/// hatch for the opt-in batched MTP paths.
pub(crate) fn mtp_divergent_fix_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("MLXCEL_MTP_DISABLE_DIVERGENT_FIX")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Build the exact per-row additive attention mask for a DIVERGENT batched
/// MTP verify round (issue #203).
///
/// Layout model (maintained by `rollback_speculative_cache_divergent`): row
/// `r`'s cache content is a contiguous valid prefix `[0, ve[r])` (physical
/// slot == logical position, with `[0, lp[r])` being resident prompt
/// padding), followed by a stale gap `[ve[r], offset)`, followed by the
/// current verify window at physical `[offset, offset + l)` whose token `j`
/// sits at the row's LOGICAL position `ve[r] + j`.
///
/// Query `j` of row `r` (logical position `q_pos = ve[r] + j`) may attend:
/// - history keys `k in [lp[r], ve[r])` (logical position == `k`), subject to
///   the sliding band `k > q_pos - window` when `window` is `Some`;
/// - window keys `k in [offset, offset + j]` (logical position
///   `ve[r] + (k - offset)`), always inside the band for `l <= window`.
///
/// Everything else (leading padding, the stale gap, future window keys) is
/// `-inf`. Returns a `[B, 1, l, offset + l]` f32 additive mask.
pub(crate) fn build_divergent_verify_mask(
    l: i32,
    offset: i32,
    per_row_valid_end: &[i32],
    left_padding: &[i32],
    window: Option<i32>,
) -> UniquePtr<MlxArray> {
    assert_eq!(
        per_row_valid_end.len(),
        left_padding.len(),
        "build_divergent_verify_mask: one left_padding entry per row"
    );
    let b = per_row_valid_end.len();
    let k_len = offset + l;
    let mut data = vec![f32::NEG_INFINITY; b * l as usize * k_len as usize];
    for (r, (&ve, &lp)) in per_row_valid_end.iter().zip(left_padding).enumerate() {
        for j in 0..l {
            let q_pos = ve + j;
            let row_base = (r * l as usize + j as usize) * k_len as usize;
            for k in 0..k_len {
                let visible = if k < lp {
                    false
                } else if k < ve {
                    window.map(|w| k > q_pos - w).unwrap_or(true)
                } else if k < offset {
                    false
                } else {
                    let k_pos = ve + (k - offset);
                    k_pos <= q_pos && window.map(|w| k_pos > q_pos - w).unwrap_or(true)
                };
                if visible {
                    data[row_base + k as usize] = 0.0;
                }
            }
        }
    }
    mlxcel_core::from_slice_f32(&data, &[b as i32, 1, l, k_len])
}

/// Offset of the first per-layer cache regardless of attention family.
///
/// The batched MTP regime keeps every family's offset in lockstep (one
/// logical token count), but a model may lack one family entirely (a
/// sliding-only synthetic fixture has no `full_attention` cache, for which
/// [`first_cache_offset`] would return a spurious `0`). Layer 0 is never a
/// KV-shared placeholder, so its cache carries the real offset.
pub(crate) fn first_present_cache_offset(caches: &[Cache]) -> i32 {
    match caches.first() {
        Some(Cache::Standard(c)) => c.offset,
        Some(Cache::Rotating(c)) => c.offset,
        None => 0,
    }
}

pub(crate) fn first_cache_offset(caches: &mut [Cache], layer_type: &str) -> i32 {
    for cache in caches.iter_mut() {
        match (layer_type, cache) {
            ("full_attention", Cache::Standard(c)) => return c.offset,
            ("sliding_attention", Cache::Rotating(c)) => return c.offset,
            _ => {}
        }
    }
    0
}

/// Live-window length of the first cache of `layer_type`, the count of keys
/// `update_and_fetch` will actually return.
///
/// This is the mask-sizing companion to [`first_cache_offset`]: prefill masks
/// must be sized from the live window, not the monotonic `offset`. Under
/// `--max-kv-size`, `trim_front` advances a full-attention `KVCache`'s
/// `live_start` while `offset` keeps growing for RoPE, so the cache returns
/// only `live_len = offset - live_start` keys; a mask sized from `offset`
/// would be wider than the returned K/V and trip `broadcast_shapes`. For a
/// `Rotating` (sliding) cache `seq_len()` already reports the live window.
/// With no trim (`live_start == 0`), this equals [`first_cache_offset`], so
/// the prefill path stays byte-identical. RoPE/position bookkeeping and the
/// batched-MTP `per_row_valid_end` coordinate math keep using
/// [`first_cache_offset`] (the monotonic value). See issue #430.
pub(crate) fn first_cache_live_len(caches: &mut [Cache], layer_type: &str) -> i32 {
    for cache in caches.iter_mut() {
        match (layer_type, cache) {
            ("full_attention", Cache::Standard(c)) => return c.live_len(),
            ("sliding_attention", Cache::Rotating(c)) => return c.seq_len(),
            _ => {}
        }
    }
    0
}

pub(crate) fn slice_layer_input(
    layer_inputs: &MlxArray,
    layer_idx: i32,
    batch: i32,
    seq_len: i32,
    hidden_size: i32,
) -> UniquePtr<MlxArray> {
    let sliced = mlxcel_core::slice(
        layer_inputs,
        &[0, 0, layer_idx, 0],
        &[batch, seq_len, layer_idx + 1, hidden_size],
    );
    mlxcel_core::squeeze_axis(&sliced, 2)
}

/// Quantization schemes mlxcel can actually dequantize: MLX-native affine plus
/// the block-float families. Anything else uses a packing mlxcel does not
/// implement.
const SUPPORTED_QUANT_SCHEMES: &[&str] = &["affine", "mxfp4", "nvfp4", "mxfp8"];

/// Longest text-only prefill that still builds the dense `[l, l + offset]`
/// f32 prefill masks. Above this, the mask pair is left `None` and `attend`
/// routes through `causal_attention`'s causal / sliding-window flag paths,
/// avoiding the O(L^2) retained masks entirely (issue #672). 4096 keeps every
/// dense mask under ~64 MiB and, like the LM-head gate in
/// `forward_last_logits`, leaves short-context behavior byte-identical.
const DENSE_PREFILL_MASK_MAX_TOKENS: i32 = 4096;

/// Validate that a model's declared quantization scheme is one mlxcel supports,
/// before any weights are loaded (issue #467).
///
/// MLX-native quantization records only `{group_size, bits}` (optionally a
/// `mode` naming a block-float family); it never carries a `quant_method`.
/// External / QAT formats such as OptiQ, AWQ, or GPTQ tag themselves with a
/// `quant_method` (or a non-affine `mode`) whose on-disk packing does not match
/// the affine layout the loader assumes. Dequantizing one as affine collapses
/// the logits and produces the degenerate, repeating output reported in issue
/// #467, so reject it here with an actionable message that names the format
/// instead of serving garbage. The narrow exception is NVIDIA ModelOpt NVFP4:
/// Gemma 4 checkpoints store explicit `{weight, weight_scale, weight_scale_2}`
/// triplets that the sanitize layer repacks to MLX affine 4-bit at load time.
///
/// Inspects the top-level and `text_config`-nested `quantization` /
/// `quantization_config` objects. Pure over the parsed config JSON so the
/// policy is unit-testable without a model on disk.
pub(crate) fn validate_quantization_scheme(config: &serde_json::Value) -> Result<(), String> {
    fn is_modelopt_nvfp4(obj: &serde_json::Value) -> bool {
        let method = obj
            .get("quant_method")
            .and_then(|v| v.as_str())
            .map(|v| v.trim().to_ascii_lowercase());
        if method.as_deref() != Some("modelopt") {
            return false;
        }

        let algo = obj
            .get("quant_algo")
            .and_then(|v| v.as_str())
            .map(|v| v.trim().to_ascii_lowercase());
        if algo.as_deref() != Some("nvfp4") {
            return false;
        }

        let Some(groups) = obj.get("config_groups").and_then(|v| v.as_object()) else {
            return false;
        };
        !groups.is_empty()
            && groups.values().all(|group| {
                group
                    .get("weights")
                    .and_then(|weights| {
                        let bits = weights.get("num_bits").and_then(|v| v.as_i64())?;
                        let group_size = weights.get("group_size").and_then(|v| v.as_i64())?;
                        let ty = weights
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .trim()
                            .to_ascii_lowercase();
                        Some(bits == 4 && group_size == 16 && ty == "float")
                    })
                    .unwrap_or(false)
            })
    }

    fn check_obj(obj: &serde_json::Value, location: &str) -> Result<(), String> {
        for key in ["quant_method", "mode"] {
            let Some(raw) = obj.get(key).and_then(|v| v.as_str()) else {
                continue;
            };
            let norm = raw.trim().to_ascii_lowercase();
            if key == "quant_method" && norm == "modelopt" && is_modelopt_nvfp4(obj) {
                continue;
            }
            if norm.is_empty() || SUPPORTED_QUANT_SCHEMES.contains(&norm.as_str()) {
                continue;
            }
            return Err(format!(
                "Unsupported quantization scheme '{raw}' declared at {location}.{key}. \
                 mlxcel supports MLX-native affine and block-float (mxfp4 / nvfp4 / mxfp8) \
                 quantization only; external formats such as OptiQ / AWQ / GPTQ use a \
                 different packing that would dequantize to degenerate output. Re-export the \
                 model to an MLX affine quantization (mlx_lm.convert / mlx-vlm) before serving."
            ));
        }
        Ok(())
    }

    let null = serde_json::Value::Null;
    let text_config = config.get("text_config").unwrap_or(&null);
    for (root, root_name) in [(config, "config"), (text_config, "text_config")] {
        for qkey in ["quantization", "quantization_config"] {
            if let Some(obj) = root.get(qkey)
                && obj.is_object()
            {
                check_obj(obj, &format!("{root_name}.{qkey}"))?;
            }
        }
    }
    Ok(())
}

pub struct Gemma4Model {
    pub(crate) text_model: Gemma4TextModel,
    pub(crate) config: TextConfig,
    pub(crate) eos_token_ids: Vec<i32>,
    _weight_backing: super::sanitize::Gemma4WeightBacking,
}

impl Gemma4Model {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse sanitized config.json: {}", e))?;

        // Refuse unsupported external quantization formats (OptiQ / AWQ / GPTQ)
        // up front rather than dequantizing them as affine and serving
        // degenerate output (issue #467).
        validate_quantization_scheme(&config_value)?;

        let is_quantized = super::sanitize::config_has_quantization_metadata(&config_value);
        let (mut weights, weight_backing) = if is_quantized {
            super::sanitize::load_gemma4_text_weights_with_backing(model_dir)?
        } else {
            (
                crate::models::load_text_weights(model_dir, None)?,
                super::sanitize::Gemma4WeightBacking::default(),
            )
        };
        if is_quantized {
            super::sanitize::sanitize_gemma4_nvfp4_weights(&mut weights, Some(&config_value));
        }
        // Strip k_proj/v_proj/k_norm entries for KV-shared layers so the
        // model constructor does not attempt to allocate them.  This applies
        // on all loader paths including the quantized text-only path above,
        // which bypasses load_text_weights and therefore does not
        // benefit from the strip already embedded in that function.
        crate::models::strip_gemma4_kv_shared_weights(&mut weights, &config_value);
        crate::models::sanitize_tied_embeddings(&mut weights, &config_value);
        let mut model = Self::from_weights(&weights, &args)?;
        model._weight_backing = weight_backing;

        Ok((model, args))
    }

    pub fn from_weights(weights: &WeightMap, args: &ModelArgs) -> Result<Self, String> {
        let config = args.text_args();
        let text_model = Gemma4TextModel::from_weights(weights, &config, "language_model.model")?;
        let eos_token_ids = {
            let eos = args.eos_token_ids();
            if eos.is_empty() {
                vec![1, 106, 50]
            } else {
                eos
            }
        };

        Ok(Self {
            text_model,
            config,
            eos_token_ids,
            _weight_backing: super::sanitize::Gemma4WeightBacking::default(),
        })
    }

    fn forward_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden =
            self.text_model
                .forward(input_ids, input_embeddings, caches, mask, per_layer_inputs);
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    /// [`Self::forward_with_caches_and_embeddings`] variant that projects only
    /// the `last_pos` hidden row through the LM head, for prefill callers that
    /// sample a single position.
    ///
    /// With `vocab_size = 262144`, full-sequence logits at a 32k-token prefill
    /// are ~17 GiB in f16 and `final_logit_softcapping` materializes a second
    /// copy; slicing the hidden state first bounds both to `[batch, 1, vocab]`
    /// (issue #672). The KV caches are updated by the full text forward
    /// exactly as before; only the LM-head projection width changes.
    fn forward_last_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        let hidden =
            self.text_model
                .forward(input_ids, input_embeddings, caches, mask, per_layer_inputs);
        let shape = mlxcel_core::array_shape(&hidden);
        let last = mlxcel_core::slice(
            &hidden,
            &[0, last_pos as i32, 0],
            &[shape[0], last_pos as i32 + 1, shape[2]],
        );
        let mut logits = self.text_model.embed_tokens.as_linear(&last);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    /// Gemma 4 Unified forward with the optional blockwise bidirectional
    /// vision overlay. Identical to [`Self::forward_with_caches_and_embeddings`]
    /// except it forwards `bidirectional_block_ids` (per-position image/video
    /// span ids; `None` for fully-causal prefill) into the text model so a
    /// vision span attends bidirectionally within itself during prefill.
    fn forward_unified_with_caches_and_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        bidirectional_block_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.text_model.forward_with_speculative_sinks(
            input_ids,
            input_embeddings,
            caches,
            mask,
            per_layer_inputs,
            None,
            None,
            false,
            bidirectional_block_ids,
            None,
            None,
        );
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    /// Sink-aware variant of [`Self::forward_with_caches_and_embeddings`]
    /// used by the Gemma 4 MTP target path. Delegates the
    /// transformer pass to
    /// [`Gemma4TextModel::forward_with_speculative_sinks`] then applies
    /// the embedding tied LM head + optional final-logit softcap exactly
    /// like the non-sink path — so callers get bit-identical logits when
    /// `sinks` is `None`.
    ///
    /// Used by: [`Gemma4Wrapper::forward_with_speculative_sinks`].
    #[allow(clippy::too_many_arguments)]
    fn forward_with_caches_and_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut Gemma4SpeculativeSinks>,
        left_padding: Option<&[i32]>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.text_model.forward_with_speculative_sinks(
            input_ids,
            input_embeddings,
            caches,
            mask,
            per_layer_inputs,
            capture_layer_ids,
            sinks,
            false,
            None,
            left_padding,
            per_row_valid_end,
        );
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    /// Run the transformer with speculative sinks but skip the tied LM head.
    ///
    /// Used by: Gemma 4 MTP deferred greedy verification, which needs the
    /// pre-norm hidden states and shared K/V slabs but can project only the
    /// positions required by the speculative walk.
    #[allow(clippy::too_many_arguments)]
    fn forward_hidden_with_caches_and_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [Cache],
        mask: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut Gemma4SpeculativeSinks>,
        skip_final_norm: bool,
        left_padding: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_with_speculative_sinks(
            input_ids,
            input_embeddings,
            caches,
            mask,
            per_layer_inputs,
            capture_layer_ids,
            sinks,
            skip_final_norm,
            None,
            left_padding,
            None,
        )
    }

    /// Apply the Gemma 4 final norm to a pre-norm decoder hidden state.
    ///
    /// Used by: Gemma 4 MTP drafter hidden preparation and deferred
    /// hidden-to-logits verification.
    fn speculative_draft_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.text_model.norm.forward(hidden)
    }

    /// Project a pre-norm decoder hidden state to logits using the tied LM
    /// head and optional final-logit softcap.
    ///
    /// Used by: Gemma 4 MTP deferred greedy verification.
    fn speculative_logits_from_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.speculative_draft_hidden(hidden);
        let mut logits = self.text_model.embed_tokens.as_linear(&hidden);
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = mlxcel_core::compiled_softcap(&logits, cap);
        }
        logits
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.text_model.make_caches()
    }
}

pub(crate) struct Gemma4StageModel {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    embed_tokens_per_layer: Option<UnifiedEmbedding>,
    per_layer_model_projection: Option<UnifiedLinear>,
    per_layer_projection_scale: f32,
    per_layer_projection_norm: Option<RMSNorm>,
    layers: Vec<(usize, DecoderLayer)>,
    norm: Option<RMSNorm>,
    config: TextConfig,
    final_logit_softcapping: Option<f32>,
    _weight_backing: super::sanitize::Gemma4WeightBacking,
}

impl Gemma4StageModel {
    pub(crate) fn load(
        model_dir: &Path,
        filter: &LayerFilter,
        stage_index: usize,
    ) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;
        let config_value: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse sanitized config.json: {}", e))?;

        // Refuse unsupported external quantization formats (OptiQ / AWQ / GPTQ)
        // up front rather than dequantizing them as affine and serving
        // degenerate output (issue #467).
        validate_quantization_scheme(&config_value)?;

        let is_quantized = super::sanitize::config_has_quantization_metadata(&config_value);
        let (mut weights, weight_backing) = if is_quantized {
            super::sanitize::load_gemma4_text_weights_with_backing(model_dir)?
        } else {
            (
                crate::models::load_text_weights(model_dir, None)?,
                super::sanitize::Gemma4WeightBacking::default(),
            )
        };
        if is_quantized {
            super::sanitize::sanitize_gemma4_nvfp4_weights(&mut weights, Some(&config_value));
        }
        // Strip k_proj/v_proj/k_norm entries for KV-shared layers so the
        // model constructor does not attempt to allocate them.  Required on
        // the quantized path because load_gemma4_text_weights_with_backing
        // does not run the strip internally.
        crate::models::strip_gemma4_kv_shared_weights(&mut weights, &config_value);
        crate::models::sanitize_tied_embeddings(&mut weights, &config_value);
        let mut effective_filter = filter.clone();
        if filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);
        Self::from_filtered_weights(&weights, &args, filter, stage_index, weight_backing)
    }

    fn from_filtered_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        filter: &LayerFilter,
        stage_index: usize,
        weight_backing: super::sanitize::Gemma4WeightBacking,
    ) -> Result<Self, String> {
        let config = args.text_args();
        let prefix = "language_model.model";

        let embed_tokens = if filter.has_embedding || filter.has_lm_head {
            Some(UnifiedEmbedding::from_weights(
                weights,
                &format!("{}.embed_tokens", prefix),
                config.group_size(),
                config.bits(),
            )?)
        } else {
            None
        };

        let embed_tokens_per_layer =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(UnifiedEmbedding::from_weights(
                    weights,
                    &format!("{}.embed_tokens_per_layer", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            };

        let per_layer_projection_scale = (config.hidden_size as f32).powf(-0.5);
        let per_layer_model_projection =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(UnifiedLinear::from_weights(
                    weights,
                    &format!("{}.per_layer_model_projection", prefix),
                    config.group_size(),
                    config.bits(),
                )?)
            } else {
                None
            };

        let per_layer_projection_norm =
            if filter.has_embedding && config.hidden_size_per_layer_input > 0 {
                Some(RMSNorm::new(
                    get_weight_copy(
                        weights,
                        &format!("{}.per_layer_projection_norm.weight", prefix),
                    )?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = DecoderLayer::from_weights(
                weights,
                &config,
                layer_idx,
                &format!("{}.layers.{}", prefix, layer_idx),
            )?;
            if layer.self_attn.is_kv_shared_layer
                && let Some(shared_idx) = layer.self_attn.kv_shared_layer_index
                && !filter.layer_range.contains(&shared_idx)
            {
                return Err(format!(
                    "stage {} cannot host Gemma4 layer {} because it shares KV with layer {} outside range {}..{}",
                    stage_index,
                    layer_idx,
                    shared_idx,
                    filter.layer_range.start,
                    filter.layer_range.end
                ));
            }
            layers.push((layer_idx, layer));
        }

        if layers.is_empty() {
            return Err(format!(
                "stage {} did not load any layers from range {}..{}",
                stage_index, filter.layer_range.start, filter.layer_range.end
            ));
        }

        let norm = if filter.has_lm_head {
            Some(RMSNorm::new(
                get_weight_copy(weights, &format!("{}.norm.weight", prefix))?,
                config.rms_norm_eps,
            ))
        } else {
            None
        };

        Ok(Self {
            filter: filter.clone(),
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_scale,
            per_layer_projection_norm,
            layers,
            norm,
            final_logit_softcapping: config.final_logit_softcapping,
            config,
            _weight_backing: weight_backing,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn make_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .map(|(layer_idx, _)| {
                if self.config.layer_type(*layer_idx) == "full_attention" {
                    Cache::Standard(KVCache::new())
                } else {
                    Cache::Rotating(RotatingKVCache::new(self.config.sliding_window as i32))
                }
            })
            .collect()
    }

    pub(crate) fn execute_from_token_ids(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        let mut hidden = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| {
                "stage does not host embeddings; hidden-state input required".to_string()
            })?
            .forward(input_ids);
        hidden = mlxcel_core::multiply_scalar(&hidden, (self.config.hidden_size as f32).sqrt());

        let stage_inputs = if self.config.hidden_size_per_layer_input > 0 {
            let raw_inputs = self.get_per_layer_inputs(input_ids)?;
            let projected = self.project_per_layer_inputs(
                hidden.as_ref().unwrap(),
                Some(raw_inputs.as_ref().unwrap()),
            )?;
            Some(projected)
        } else {
            None
        };
        let local_inputs = stage_inputs.as_ref().map(|inputs| {
            self.slice_layer_input_range(inputs.as_ref().unwrap(), self.filter.layer_range.clone())
        });
        let downstream_inputs = stage_inputs.as_ref().and_then(|inputs| {
            if self.filter.layer_range.end >= self.config.num_hidden_layers {
                None
            } else {
                Some(self.slice_layer_input_range(
                    inputs.as_ref().unwrap(),
                    self.filter.layer_range.end..self.config.num_hidden_layers,
                ))
            }
        });

        self.execute_hidden(hidden, local_inputs.as_ref(), downstream_inputs, caches)
    }

    pub(crate) fn execute_from_hidden_states(
        &self,
        packed_hidden: UniquePtr<MlxArray>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if self.filter.has_embedding {
            return Err("entry stage expects token IDs, not hidden states".to_string());
        }

        let (hidden, local_inputs, downstream_inputs) =
            self.unpack_stage_hidden(packed_hidden.as_ref().unwrap())?;
        self.execute_hidden(hidden, local_inputs.as_ref(), downstream_inputs, caches)
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        local_per_layer_inputs: Option<&UniquePtr<MlxArray>>,
        downstream_inputs: Option<UniquePtr<MlxArray>>,
        caches: &mut [Cache],
    ) -> Result<StageExecutionOutput, String> {
        if caches.len() != self.layers.len() {
            return Err(format!(
                "stage cache count mismatch: expected {}, got {}",
                self.layers.len(),
                caches.len()
            ));
        }

        let shape = mlxcel_core::array_shape(hidden.as_ref().unwrap());
        let batch = shape[0];
        let seq_len = shape[1];

        let (global_mask, sliding_mask) = if seq_len > 1 {
            // Size the prefill masks from the cache's live window
            // (`live_len`), not the monotonic `offset`, so the mask key axis
            // matches the K/V `update_and_fetch` returns after a
            // `--max-kv-size` trim (`live_start` advances while `offset` keeps
            // growing for RoPE). Byte-identical when untrimmed (`live_len ==
            // offset`). See issue #430 (mirrors #419/#420, #421/#422).
            let global_live_len = self.first_cache_live_len(caches, "full_attention");
            let sliding_live_len = self.first_cache_live_len(caches, "sliding_attention");
            (
                Some(create_causal_mask(seq_len, global_live_len)),
                // Shared helper hoisted in #410; behaviour-preserving for Gemma 4
                // (see the gate-divergence note in `forward_with_speculative_sinks`).
                Some(create_sliding_window_prefill_mask(
                    seq_len,
                    sliding_live_len,
                    self.config.sliding_window as i32,
                )),
            )
        } else {
            (None, None)
        };

        let mut shared_kv_store: HashMap<usize, (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32)> =
            HashMap::new();
        let n_layers = self.layers.len();

        for (local_idx, (global_idx, layer)) in self.layers.iter().enumerate() {
            let cache = caches[local_idx].as_interface();
            let mut shared_kv = None;

            if layer.self_attn.is_kv_shared_layer
                && let Some(ref_idx) = layer.self_attn.kv_shared_layer_index
            {
                let (keys, values, ref_offset) =
                    shared_kv_store.get(&ref_idx).ok_or_else(|| {
                        format!(
                            "stage missing shared KV source layer {} for Gemma4 layer {}",
                            ref_idx, global_idx
                        )
                    })?;
                cache.set_offset(*ref_offset);
                shared_kv = Some((keys.as_ref().unwrap(), values.as_ref().unwrap()));
            }

            let local_mask = match layer.layer_type.as_str() {
                "full_attention" => global_mask.as_ref().map(|m| m.as_ref().unwrap()),
                _ => sliding_mask.as_ref().map(|m| m.as_ref().unwrap()),
            };

            let layer_input = local_per_layer_inputs.as_ref().map(|inputs| {
                slice_layer_input(
                    inputs.as_ref().unwrap(),
                    local_idx as i32,
                    batch,
                    seq_len,
                    self.config.hidden_size_per_layer_input as i32,
                )
            });

            let pre_offset = cache.offset();
            let (next_hidden, stored_kv) = layer.forward(
                hidden.as_ref().unwrap(),
                local_mask,
                cache,
                layer_input.as_ref().map(|arr| arr.as_ref().unwrap()),
                shared_kv,
            );
            hidden = next_hidden;

            if let Some((keys, values)) = stored_kv {
                shared_kv_store.insert(*global_idx, (keys, values, pre_offset));
            }

            pipeline_hint(&hidden, local_idx, n_layers);
        }

        if let Some(norm) = &self.norm {
            let hidden = norm.forward(hidden.as_ref().unwrap());
            let mut logits = self
                .embed_tokens
                .as_ref()
                .ok_or_else(|| "final Gemma4 stage missing embeddings".to_string())?
                .as_linear(&hidden);
            if let Some(cap) = self.final_logit_softcapping {
                logits = mlxcel_core::compiled_softcap(&logits, cap);
            }
            return Ok(StageExecutionOutput::Logits(logits));
        }

        Ok(StageExecutionOutput::HiddenStates(
            self.pack_hidden_for_downstream(hidden.as_ref().unwrap(), downstream_inputs.as_ref())?,
        ))
    }

    fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let embedded = self
            .embed_tokens_per_layer
            .as_ref()
            .ok_or_else(|| "Gemma4 per-layer embeddings missing".to_string())?
            .forward(input_ids);
        let embedded = mlxcel_core::multiply_scalar(
            &embedded,
            (self.config.hidden_size_per_layer_input as f32).sqrt(),
        );
        let shape = mlxcel_core::array_shape(input_ids);
        Ok(mlxcel_core::reshape(
            &embedded,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        ))
    }

    fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let projected = self
            .per_layer_model_projection
            .as_ref()
            .ok_or_else(|| "Gemma4 per_layer_model_projection missing".to_string())?
            .forward(inputs_embeds);
        let projected = mlxcel_core::multiply_scalar(&projected, self.per_layer_projection_scale);
        let shape = mlxcel_core::array_shape(inputs_embeds);
        let projected = mlxcel_core::reshape(
            &projected,
            &[
                shape[0],
                shape[1],
                self.config.num_hidden_layers as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let projected = self
            .per_layer_projection_norm
            .as_ref()
            .ok_or_else(|| "Gemma4 per_layer_projection_norm missing".to_string())?
            .forward(&projected);

        Ok(if let Some(per_layer_inputs) = per_layer_inputs {
            let sum = mlxcel_core::add(&projected, per_layer_inputs);
            mlxcel_core::multiply_scalar(&sum, std::f32::consts::FRAC_1_SQRT_2)
        } else {
            projected
        })
    }

    fn pack_hidden_for_downstream(
        &self,
        hidden: &MlxArray,
        downstream_inputs: Option<&UniquePtr<MlxArray>>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let Some(downstream_inputs) = downstream_inputs else {
            return Ok(mlxcel_core::copy(hidden));
        };

        let aux_shape = mlxcel_core::array_shape(downstream_inputs.as_ref().unwrap());
        let flat_aux = mlxcel_core::reshape(
            downstream_inputs.as_ref().unwrap(),
            &[aux_shape[0], aux_shape[1], aux_shape[2] * aux_shape[3]],
        );
        Ok(mlxcel_core::concatenate(hidden, &flat_aux, -1))
    }

    fn unpack_stage_hidden(
        &self,
        packed_hidden: &MlxArray,
    ) -> Result<
        (
            UniquePtr<MlxArray>,
            Option<UniquePtr<MlxArray>>,
            Option<UniquePtr<MlxArray>>,
        ),
        String,
    > {
        if self.config.hidden_size_per_layer_input == 0 {
            return Ok((mlxcel_core::copy(packed_hidden), None, None));
        }

        let shape = mlxcel_core::array_shape(packed_hidden);
        let batch = shape[0];
        let seq_len = shape[1];
        let hidden_size = self.config.hidden_size as i32;
        let remaining_layers =
            (self.config.num_hidden_layers - self.filter.layer_range.start) as i32;
        let hidden = slice_axis(packed_hidden, -1, 0, hidden_size);
        let flat_aux = slice_axis(
            packed_hidden,
            -1,
            hidden_size,
            hidden_size + remaining_layers * self.config.hidden_size_per_layer_input as i32,
        );
        let aux = mlxcel_core::reshape(
            &flat_aux,
            &[
                batch,
                seq_len,
                remaining_layers,
                self.config.hidden_size_per_layer_input as i32,
            ],
        );
        let local_layers = self.filter.num_layers() as i32;
        let local_inputs = self.slice_layer_input_range(&aux, 0..local_layers as usize);
        let downstream_inputs =
            if local_layers < remaining_layers {
                Some(self.slice_layer_input_range(
                    &aux,
                    local_layers as usize..remaining_layers as usize,
                ))
            } else {
                None
            };
        Ok((hidden, Some(local_inputs), downstream_inputs))
    }

    fn slice_layer_input_range(
        &self,
        inputs: &MlxArray,
        range: std::ops::Range<usize>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs);
        mlxcel_core::slice(
            inputs,
            &[0, 0, range.start as i32, 0],
            &[
                shape[0],
                shape[1],
                range.end as i32,
                self.config.hidden_size_per_layer_input as i32,
            ],
        )
    }

    /// Live-window length of the first cache of `layer_type` (mask-sizing
    /// companion to the free-function [`first_cache_offset`]). See
    /// [`first_cache_live_len`] for the full rationale; prefill masks size
    /// from this so the mask key axis matches the K/V `update_and_fetch`
    /// returns after a `--max-kv-size` trim, while RoPE keeps the monotonic
    /// offset. See issue #430.
    fn first_cache_live_len(&self, caches: &[Cache], layer_type: &str) -> i32 {
        for cache in caches {
            match (layer_type, cache) {
                ("full_attention", Cache::Standard(cache)) => return cache.live_len(),
                ("sliding_attention", Cache::Rotating(cache)) => return cache.seq_len(),
                _ => {}
            }
        }
        0
    }
}

/// Wrapper for [`Gemma4Model`] that implements [`LanguageModel`] with
/// per-`SequenceId` cache isolation so the server scheduler can run
/// mixed-length batches correctly.
///
/// Gemma 4's `Cache` enum (`KVCache | RotatingKVCache`) is a sliding-
/// window-aware cache that the model owns internally — it cannot be
/// wired to scheduler-managed `&mut [KVCache]` slices the way standard
/// transformer text models can. Instead, the wrapper opts into the
/// `SequenceStateBackend::ModelOwned` layout and stores a per-sequence
/// `Vec<Cache>` keyed on [`SequenceId`] in [`ModelOwnedSequenceState`].
/// `forward_with_sequence_id` resolves the right slot per row, and a
/// fallback `internal` slot preserves the legacy single-row CLI / VLM-
/// prefill path that does not plumb a `SequenceId`.
///
/// This mirrors the approach Gemma 3 (issue analogous, see `gemma3.rs`)
/// already uses; that family has been running batched decode in
/// production this way since its own enable-batching change. Gemma 4
/// inherits the same correctness guarantees here.
pub struct Gemma4Wrapper {
    model: Gemma4Model,
    sequence_state: ModelOwnedSequenceState<Cache>,
}

impl Gemma4Wrapper {
    pub fn new(model: Gemma4Model) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            sequence_state: ModelOwnedSequenceState::new(caches),
        }
    }

    /// Reset the wrapper's fallback cache slot (the `internal` slot used
    /// by the CLI / single-row VLM-prefill path) to a fresh, empty set
    /// of caches. Per-sequence cache slots in
    /// [`ModelOwnedSequenceState`] are unaffected — those are owned by
    /// the scheduler and dropped via `release_sequence_state_by_id`.
    ///
    /// Used by: legacy CLI generate path (`mlxcel generate`) when starting
    /// a fresh request that does not flow through the server scheduler.
    pub fn reset_caches(&self) {
        self.sequence_state
            .replace_internal(self.model.make_caches());
    }

    pub(crate) fn input_embeddings(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.model.text_model.embed_tokens.forward(input_ids)
    }

    pub(crate) fn get_per_layer_inputs(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        if self.model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            Some(self.model.text_model.get_per_layer_inputs(input_ids))
        }
    }

    pub(crate) fn project_per_layer_inputs(
        &self,
        inputs_embeds: &MlxArray,
        per_layer_inputs: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if self.model.config.hidden_size_per_layer_input == 0 {
            None
        } else {
            Some(
                self.model
                    .text_model
                    .project_per_layer_inputs(inputs_embeds, per_layer_inputs),
            )
        }
    }

    /// VLM-prefill / VLM-step path: forward with optional pre-merged
    /// input embeddings and per-layer-inputs, routing to the per-
    /// `SequenceId` cache slot when one is provided (the scheduler
    /// allocates a `SequenceId` for every server-side VLM request) and
    /// to the wrapper's fallback `internal` slot when `seq_id` is `None`
    /// (CLI generate path / single-row tests).
    ///
    /// Used by: [`crate::vision::Gemma4VLModel::forward_with_embeddings_and_sequence_id`]
    /// for VLM prefill and decode.
    pub(crate) fn forward_with_inputs_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_with_caches_and_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
                    per_layer_inputs,
                )
            },
        )
    }

    /// Gemma 4 Unified VLM-prefill / step forward.
    ///
    /// Mirrors [`Self::forward_with_inputs_and_sequence_id`] but additionally
    /// forwards `bidirectional_block_ids` so image/video token spans attend
    /// bidirectionally within themselves during prefill (the `gemma4_unified`
    /// `use_bidirectional_attention == "vision"` behaviour). `None` block ids
    /// (audio present, decode, or a non-vision prompt) keep the standard
    /// causal/windowed masks. Used by
    /// [`crate::vision::Gemma4UnifiedModel::forward_with_embeddings_and_sequence_id`].
    pub(crate) fn forward_unified_with_inputs_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        bidirectional_block_ids: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_unified_with_caches_and_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
                    per_layer_inputs,
                    bidirectional_block_ids,
                )
            },
        )
    }

    /// Read-only access to the wrapped text config (for the Gemma 4 Unified
    /// runtime to inspect `use_bidirectional_attention`, `sliding_window`,
    /// etc. without re-parsing the checkpoint).
    pub(crate) fn text_config(&self) -> &TextConfig {
        &self.model.config
    }

    pub(crate) fn num_layers_value(&self) -> usize {
        self.model.text_model.layers.len()
    }

    /// Sliding-window size of the sliding-attention layers.
    ///
    /// Used by the ragged batched MTP target adapter to gate variable-length
    /// prefill eligibility: ragged left-padded prefill is only admitted when
    /// `max_prompt_len <= sliding_window` (the non-capped RotatingKVCache
    /// regime), where the windowed left-padding mask is well-defined.
    pub(crate) fn sliding_window_value(&self) -> usize {
        self.model.config.sliding_window
    }

    pub(crate) fn eos_token_ids_value(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    /// Read the current absolute KV-cache offset for the requested
    /// attention-layer family in a model-owned speculative cache slot.
    ///
    /// Used by: `Gemma4MtpTargetAdapter` to rebind the assistant drafter
    /// with the same post-rollback absolute offset that upstream exposes
    /// as `prompt_cache[0].offset`.
    pub(crate) fn speculative_cache_offset(
        &self,
        seq_id: Option<SequenceId>,
        layer_type: &str,
    ) -> i32 {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| first_cache_offset(sequence_caches, layer_type),
        )
    }

    /// Enable buffered rotating caches for a Gemma 4 MTP B=1 target slot.
    ///
    /// Mirrors upstream mlx-vlm's `BufferedRotatingKVCache` conversion after
    /// prompt prefill: full-attention caches are left unchanged, while
    /// sliding-attention caches keep a small temporary prefix buffer so MTP
    /// verify bursts can be rolled back without destructive ring wrap.
    ///
    /// Used by: `Gemma4MtpTargetAdapter::prefill_and_seed`.
    pub(crate) fn enable_mtp_rotating_cache_buffer(
        &self,
        seq_id: Option<SequenceId>,
        buffer_size: i32,
    ) {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                for cache in sequence_caches.iter_mut() {
                    if let Err(error) = cache.enable_mtp_rotating_buffer(buffer_size) {
                        tracing::warn!(
                            error,
                            buffer_size,
                            "Gemma4 MTP could not enable rotating-cache rollback buffer"
                        );
                    }
                }
            },
        );
    }

    /// Returns the text model's hidden size (embedding dimension).
    ///
    /// Used by: `Gemma4VLModel::get_input_embeddings_with_audio` to apply the
    /// `sqrt(hidden_size)` embed scale to text embeddings before merging in
    /// vision / audio features. Vision and audio features must NOT be scaled
    /// again since they are already in the language-model embedding space
    pub fn hidden_size(&self) -> usize {
        self.model.config.hidden_size
    }

    /// Sink-aware forward used by the Gemma 4 MTP target path.
    ///
    /// Mirrors [`Self::forward_with_inputs_and_sequence_id`] but additionally
    /// captures the LAST decoder hidden state (or per-layer hidden states
    /// matching `capture_layer_ids`) and / or the shared K/V slabs (last
    /// full-attention + last sliding-attention) into the caller-provided
    /// [`Gemma4SpeculativeSinks`]. Resolves to the per-`SequenceId` cache
    /// slot via [`ModelOwnedSequenceState::with_or_create_sequence_state`]
    /// — the same isolation the rest of Gemma 4 already uses for batched
    /// decode, so the speculative loop does not need to
    /// allocate a side cache.
    ///
    /// When `sinks` is `None` and `capture_layer_ids` is `None`, this is
    /// behaviorally equivalent to the non-sink wrapper path: logits with
    /// optional final-logit softcap, zero allocation overhead beyond the
    /// `Option`s.
    ///
    /// Used by: future `Gemma4AssistantDraftModel` consumer and
    /// `MtpGenerator`.
    ///
    /// `per_row_valid_end` (issue #163) is forwarded to the inner text model's
    /// batched-verify tail exclusion; the seq-id path is single-row so its only
    /// callers pass `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut Gemma4SpeculativeSinks>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_with_caches_and_speculative_sinks(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
                    per_layer_inputs,
                    capture_layer_ids,
                    sinks,
                    None,
                    per_row_valid_end,
                )
            },
        )
    }

    /// Sink-aware forward against a **caller-owned** `[B, ...]` cache
    /// vector (batched MTP dispatch).
    ///
    /// Unlike [`Self::forward_with_speculative_sinks`], this variant does
    /// NOT resolve the cache through [`ModelOwnedSequenceState`] /
    /// `SequenceId`. The batched MTP target adapter
    /// ([`crate::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter`])
    /// owns a single `Vec<Cache>` whose every per-layer cache carries a
    /// leading batch dim `B` and drives all `B` rows through one forward
    /// pass. The per-`SequenceId` slot model is single-row by
    /// construction (one `Vec<Cache>` per `SequenceId`), so it cannot
    /// express a `[B, ...]` verify forward — hence the explicit-cache
    /// entrypoint.
    ///
    /// Behaviourally identical to the seq-id variant once the cache is
    /// resolved: same masks, same sink capture, same per-layer-input
    /// projection.
    ///
    /// Used by: [`crate::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter`].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_speculative_sinks_explicit_cache(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        caches: &mut [Cache],
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut Gemma4SpeculativeSinks>,
        left_padding: Option<&[i32]>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        self.model.forward_with_caches_and_speculative_sinks(
            input_ids,
            input_embeddings,
            caches,
            mask,
            per_layer_inputs,
            capture_layer_ids,
            sinks,
            left_padding,
            per_row_valid_end,
        )
    }

    /// Sink-aware hidden forward used by Gemma 4 MTP deferred greedy
    /// verification.
    ///
    /// Used by: [`crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter`].
    pub(crate) fn forward_hidden_with_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut Gemma4SpeculativeSinks>,
        skip_final_norm: bool,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_hidden_with_caches_and_speculative_sinks(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
                    per_layer_inputs,
                    capture_layer_ids,
                    sinks,
                    skip_final_norm,
                    None,
                )
            },
        )
    }

    /// Normalize a pre-norm hidden state before handing it to the MTP
    /// assistant drafter.
    ///
    /// Used by: [`crate::models::gemma4_mtp_target::Gemma4MtpTargetAdapter`].
    pub(crate) fn speculative_draft_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.model.speculative_draft_hidden(hidden)
    }

    /// Project a pre-norm hidden state to logits without rerunning the
    /// transformer.
    ///
    /// Used by: Gemma 4 MTP deferred greedy verification.
    pub(crate) fn speculative_logits_from_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.model.speculative_logits_from_hidden(hidden)
    }

    /// Allocate a fresh per-layer cache vector for a batched MTP burst
    /// Every cache starts empty; the batched verify pass
    /// grows them with a leading batch dim `B` once the first `[B, L]`
    /// prefill flows through
    /// [`Self::forward_with_speculative_sinks_explicit_cache`].
    ///
    /// Distinct from the [`LanguageModel::make_caches`] trait method
    /// (which returns an empty `Vec<KVCache>` because Gemma 4 owns its
    /// caches internally) — this returns the heterogeneous
    /// `Vec<Cache>` (`KVCache | RotatingKVCache`) the speculative
    /// forward actually consumes.
    ///
    /// Used by: [`crate::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter`].
    pub fn make_speculative_caches(&self) -> Vec<Cache> {
        self.model.make_caches()
    }

    /// Rewind the per-sequence target KV caches after a Gemma 4 MTP
    /// speculative-decoding round. Mirrors the upstream Python
    /// hook
    /// (https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py lines
    /// 608-646) bit-for-bit:
    ///
    /// 1. `n = max(accepted) + 1`, `trim = block_size - n`.
    /// 2. If `trim > 0`, each cache is trimmed by `trim` via
    ///    [`Cache::trim_speculative`] (dispatches to `KVCache::trim` for
    ///    full-attention caches and `RotatingKVCache::trim` for
    ///    sliding-attention caches).
    /// 3. For batched verify with at least one row having `accepted > 0`,
    ///    each rotating cache additionally gets per-row tail-zeroing via
    ///    [`Cache::zero_partial_accept_tail`] — required because rows with
    ///    different accept counts share the same KV buffer.
    ///
    /// The full-accept edge case (`accepted == block_size - 1` for every
    /// row, equivalent to `n == block_size`) results in `trim == 0` and no
    /// per-row zeroing (because every row's `start == kv_len`).
    ///
    /// Routes to the per-`SequenceId` cache slot (or the wrapper's internal
    /// fallback slot when `seq_id` is `None`) — same isolation pattern as
    /// the rest of `Gemma4Wrapper`.
    ///
    /// `gdn_states` is intentionally elided from the Rust signature: Gemma
    /// 4 has no SSM/GDN state, and the Python API-parity placeholder is
    /// unnecessary in the typed Rust API surface (the future hybrid-model
    /// MTP wrappers will define their own analogue).
    ///
    /// Returns `Ok(())` on success, or an `Err` describing the failure
    /// (e.g. mismatched batch shape between cache buffer and `accepted`,
    /// or rotating-cache write index that predates the verify pass).
    ///
    /// Used by: future `MtpGenerator` Gemma 4 verify loop.
    pub fn rollback_speculative_cache(
        &self,
        seq_id: Option<SequenceId>,
        accepted: &[i32],
        block_size: i32,
    ) -> Result<(), String> {
        if accepted.is_empty() {
            return Err("rollback_speculative_cache: accepted slice must be non-empty".into());
        }
        if block_size <= 0 {
            return Err(format!(
                "rollback_speculative_cache: block_size must be positive (got {block_size})"
            ));
        }
        let max_a = *accepted.iter().max().unwrap();
        if max_a < 0 {
            return Err(format!(
                "rollback_speculative_cache: accepted values must be non-negative \
                 (got max = {max_a})"
            ));
        }
        if max_a > block_size - 1 {
            return Err(format!(
                "rollback_speculative_cache: max(accepted) ({max_a}) cannot exceed \
                 block_size - 1 ({})",
                block_size - 1
            ));
        }
        let n = max_a + 1;
        let trim = block_size - n;
        let is_batch = accepted.len() > 1;
        // valid_ends[i] = accepted[i] + 1
        let valid_ends: Vec<i32> = accepted.iter().map(|a| a + 1).collect();

        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| -> Result<(), String> {
                for cache in sequence_caches.iter_mut() {
                    if trim > 0 {
                        cache.trim_speculative(trim);
                    }
                    if is_batch && max_a > 0 {
                        cache.zero_partial_accept_tail(&valid_ends, block_size)?;
                    }
                }
                Ok(())
            },
        )
    }

    /// Per-row tail-zero rollback against a **caller-owned** `[B, ...]`
    /// cache vector (batched MTP dispatch).
    ///
    /// Identical trim + per-row tail-zero logic as
    /// [`Self::rollback_speculative_cache`], but operates on an explicit
    /// `&mut [Cache]` instead of resolving through the per-`SequenceId`
    /// slot. The batched MTP adapter owns its `[B, ...]` caches directly,
    /// so it bypasses [`ModelOwnedSequenceState`].
    ///
    /// `accepted` is the per-row accept slice from the batched
    /// speculative walk (length `B`). The global trim amount is
    /// `block_size - (max(accepted) + 1)`; rows whose accept count is
    /// below `max(accepted)` get their KV tail per-row zeroed via
    /// [`Cache::zero_partial_accept_tail`].
    ///
    /// Used by: [`crate::models::gemma4_mtp_target::Gemma4MtpBatchedTargetAdapter`].
    pub fn rollback_speculative_cache_explicit(
        &self,
        caches: &mut [Cache],
        accepted: &[i32],
        block_size: i32,
    ) -> Result<(), String> {
        if accepted.is_empty() {
            return Err(
                "rollback_speculative_cache_explicit: accepted slice must be non-empty".into(),
            );
        }
        if block_size <= 0 {
            return Err(format!(
                "rollback_speculative_cache_explicit: block_size must be positive \
                 (got {block_size})"
            ));
        }
        let max_a = *accepted.iter().max().unwrap();
        if max_a < 0 {
            return Err(format!(
                "rollback_speculative_cache_explicit: accepted values must be non-negative \
                 (got max = {max_a})"
            ));
        }
        if max_a > block_size - 1 {
            return Err(format!(
                "rollback_speculative_cache_explicit: max(accepted) ({max_a}) cannot exceed \
                 block_size - 1 ({})",
                block_size - 1
            ));
        }
        let n = max_a + 1;
        let trim = block_size - n;
        let is_batch = accepted.len() > 1;
        let valid_ends: Vec<i32> = accepted.iter().map(|a| a + 1).collect();

        for cache in caches.iter_mut() {
            if trim > 0 {
                cache.trim_speculative(trim);
            }
            if is_batch && max_a > 0 {
                cache.zero_partial_accept_tail(&valid_ends, block_size)?;
            }
        }
        Ok(())
    }

    /// Issue #203: divergent-round batched MTP rollback. Replaces the
    /// global-max trim + tail-zero contract of
    /// [`Self::rollback_speculative_cache_explicit`] with per-row compaction:
    /// each row's accepted verify-window K/V moves down to the row's logical
    /// valid end (`ve_pre[r]`), restoring the contiguous-prefix layout
    /// (physical slot == logical position) the per-row RoPE rotation and the
    /// divergent verify mask assume. The cache offset is then trimmed to
    /// `o_post = max(ve_pre[r] + accepted[r] + 1)`, which this returns.
    ///
    /// Called by the batched MTP adapter only when at least one row's
    /// `ve_pre[r]` lags the shared physical write base (`o_pre`); uniform
    /// rounds keep using `rollback_speculative_cache_explicit`, whose
    /// behaviour this exactly reduces to when `ve_pre[r] == o_pre` for all
    /// rows.
    pub fn rollback_speculative_cache_divergent(
        &self,
        caches: &mut [Cache],
        ve_pre: &[i32],
        accepted: &[i32],
        block_size: i32,
    ) -> Result<i32, String> {
        if accepted.is_empty() || ve_pre.len() != accepted.len() {
            return Err(format!(
                "rollback_speculative_cache_divergent: ve_pre rows ({}) and accepted rows \
                 ({}) must match and be non-empty",
                ve_pre.len(),
                accepted.len()
            ));
        }
        if block_size <= 0 {
            return Err(format!(
                "rollback_speculative_cache_divergent: block_size must be positive \
                 (got {block_size})"
            ));
        }
        let max_a = *accepted.iter().max().unwrap();
        if accepted.iter().any(|&a| a < 0) {
            return Err(
                "rollback_speculative_cache_divergent: accepted values must be non-negative".into(),
            );
        }
        if max_a > block_size - 1 {
            return Err(format!(
                "rollback_speculative_cache_divergent: max(accepted) ({max_a}) cannot \
                 exceed block_size - 1 ({})",
                block_size - 1
            ));
        }
        let o_post = ve_pre
            .iter()
            .zip(accepted)
            .map(|(&v, &a)| v + a + 1)
            .max()
            .unwrap();
        for cache in caches.iter_mut() {
            cache.compact_partial_accept_rows(ve_pre, accepted, block_size)?;
            let n = cache.offset() - o_post;
            if n > 0 {
                cache.trim_speculative(n);
            }
        }
        Ok(o_post)
    }
}

impl LanguageModel for Gemma4Wrapper {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // No `seq_id` plumbed through (legacy CLI / single-row tests).
        // Route to the fallback `internal` slot.
        self.forward_with_sequence_id(input_ids, None, _caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            _caches,
            mask,
        )
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_with_caches_and_embeddings(
                    input_ids,
                    None,
                    sequence_caches,
                    mask,
                    None,
                )
            },
        )
    }

    /// Long-prefill override: project only the last real position through the
    /// 262k-vocab LM head instead of materializing `[1, seq_len, vocab]`
    /// logits plus a `final_logit_softcapping` copy (issue #672).
    ///
    /// Short prefills keep the full-logits path: the LM head then runs the
    /// same batched kernel as before this override existed, so short-context
    /// greedy output stays byte-identical. The threshold only trades memory
    /// (full logits are ~0.5 GiB per 1k tokens) against that guarantee; at
    /// long lengths the single-row projection is mathematically the same
    /// `hidden[last] @ W` row, merely computed by the single-row kernel.
    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        const FULL_LOGITS_MAX_PREFILL: i32 = 4096;
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        if seq_len <= FULL_LOGITS_MAX_PREFILL {
            let logits = self.forward(input_ids, _caches, mask);
            let shape = mlxcel_core::array_shape(&logits);
            return mlxcel_core::slice(
                &logits,
                &[0, last_pos as i32, 0],
                &[shape[0], last_pos as i32 + 1, shape[2]],
            );
        }
        self.sequence_state.with_or_create_sequence_state(
            None,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_last_with_caches_and_embeddings(
                    input_ids,
                    None,
                    sequence_caches,
                    mask,
                    None,
                    last_pos,
                )
            },
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.model.make_caches(),
            |sequence_caches| {
                self.model.forward_with_caches_and_embeddings(
                    input_ids,
                    input_embeddings,
                    sequence_caches,
                    mask,
                    None,
                )
            },
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.input_embeddings(input_ids))
    }

    /// Hand out the target embedding table for Gemma 4 MTP assistants.
    ///
    /// The assistant drafter's own embedding width can differ from the
    /// target backbone width (for example 31B target 5376 vs assistant
    /// 1024). Binding the target table lets `draft_block` concatenate
    /// `[token_embed, target_hidden]` at `2 * backbone_hidden_size`,
    /// matching upstream Gemma 4 MTP.
    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        Some(self.model.text_model.embed_tokens.clone_shared())
    }

    /// Empty external caches: the wrapper owns all cache state internally
    /// and resolves it per `SequenceId` via [`ModelOwnedSequenceState`].
    /// The matching layout descriptor is
    /// [`SequenceStateLayout::model_owned`] returned by
    /// [`Self::sequence_state_layout`].
    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.model.text_model.layers.len())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.model.make_caches());
    }

    fn reset_runtime_state(&self) {
        // Used by: CxxGenerator single-row generation paths. Gemma 4 owns
        // its fallback cache slot inside `ModelOwnedSequenceState`; reset it
        // for fresh CLI / benchmark runs without touching scheduler-owned
        // per-sequence entries.
        self.reset_caches();
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot = ModelStateSnapshot::new("gemma4", token_len);
                for (idx, cache) in state.iter().enumerate() {
                    if let Err(error) = cache.snapshot_into(&mut snapshot, &format!("layer{idx}")) {
                        tracing::warn!(
                            error,
                            layer_idx = idx,
                            "Gemma4 snapshot prompt-cache donation skipped"
                        );
                        return None;
                    }
                }
                if snapshot.is_empty() {
                    None
                } else {
                    Some(snapshot)
                }
            })
            .flatten()
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "gemma4" {
            return Err(format!(
                "cannot restore {} snapshot into Gemma 4",
                snapshot.family()
            ));
        }
        let mut state = self.model.make_caches();
        for (idx, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{idx}"))?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    fn num_layers(&self) -> usize {
        self.model.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.model.eos_token_ids.clone()
    }

    /// Gemma 4 supports batched decode now that
    /// [`ModelOwnedSequenceState`] isolates per-`SequenceId` cache state.
    /// The vision wrapper's `forward_batched_with_context_and_ids`
    /// override (in `vision::Gemma4VLModel`) routes each row through
    /// [`Self::forward_with_sequence_id`] which resolves to a distinct
    /// per-sequence `Vec<Cache>` — no shared-RefCell leakage across rows.
    fn supports_batching(&self) -> bool {
        true
    }

    fn supports_padded_prefill(&self) -> bool {
        // Tried enabling NA tile-aligned padded prefill (both with explicit
        // mask and maskless-causal) for the per_layer_input-free variants
        // (26B-a4b, 31B) and measured no change vs the default bulk forward
        // (26B long prefill 1836 → 1832 tok/s, 31B 430 → 435 tok/s — all
        // within run-to-run noise). Gemma 4's attention mix (sliding window,
        // K=V for full-attention layers, proportional RoPE) does not benefit
        // from the tile-aligned path in the same way plain causal models do,
        // so we keep the conservative default until a profiling-driven
        // rewrite of the prefill path.
        false
    }
}

#[cfg(test)]
mod gemma4_unified_mask_tests {
    use super::*;

    /// Read one f32 from a 2-D additive mask at `[q, k]`.
    fn mask_at(mask: &MlxArray, q: i32, k: i32) -> f32 {
        let scalar = mlxcel_core::slice(mask, &[q, k], &[q + 1, k + 1]);
        mlxcel_core::item_f32(&scalar)
    }

    #[test]
    fn overlay_opens_intra_block_bidirectionally() {
        // Sequence layout (block ids): text(-1) img(0) img(0) text(-1).
        // Base mask is plain causal [4, 4]; the overlay must additionally allow
        // the two image positions (1, 2) to attend to each other in BOTH
        // directions while leaving everything else causal.
        let base = create_causal_mask(4, 0);
        let block_ids = mlxcel_core::from_slice_i32(&[-1, 0, 0, -1], &[4]);
        let out = overlay_block_bidirectional(&base, &block_ids);
        mlxcel_core::eval(&out);

        let neg_inf_is = |v: f32| v.is_infinite() && v < 0.0;

        // Image position 1 may now attend forward to image position 2 (causal
        // would have masked this future key).
        assert_eq!(mask_at(&out, 1, 2), 0.0, "img1 must attend to img2");
        // And img2 -> img1 stays allowed (already causal).
        assert_eq!(mask_at(&out, 2, 1), 0.0, "img2 must attend to img1");
        // Self-attention within the block is allowed.
        assert_eq!(mask_at(&out, 1, 1), 0.0);

        // Text token 0 cannot attend forward to image token 1 (cross-block /
        // text→vision stays causal).
        assert!(
            neg_inf_is(mask_at(&out, 0, 1)),
            "text0 must NOT attend forward to img1",
        );
        // Text token 3 attends backward to images (causal) — unchanged.
        assert_eq!(mask_at(&out, 3, 1), 0.0);
        // Image token 1 cannot attend forward to text token 3.
        assert!(
            neg_inf_is(mask_at(&out, 1, 3)),
            "img1 must NOT attend forward to text3",
        );
    }

    #[test]
    fn overlay_separate_blocks_do_not_cross_attend() {
        // text img(0) text img(1): the two images are in different blocks and
        // must NOT attend to each other bidirectionally.
        let base = create_causal_mask(4, 0);
        let block_ids = mlxcel_core::from_slice_i32(&[-1, 0, -1, 1], &[4]);
        let out = overlay_block_bidirectional(&base, &block_ids);
        mlxcel_core::eval(&out);

        // img at index 1 must NOT attend forward to img at index 3 (different
        // block) — stays causal (-inf).
        let v = mask_at(&out, 1, 3);
        assert!(
            v.is_infinite() && v < 0.0,
            "cross-block forward stays masked"
        );
        // Backward (3 -> 1) is allowed only by causality, not by same-block.
        assert_eq!(mask_at(&out, 3, 1), 0.0);
    }

    // [1, H, n, D] K/V tensor; only the key axis (n) matters here.
    fn make_kv(h: i32, n: i32, d: i32, base: f32) -> UniquePtr<MlxArray> {
        let count = (h * n * d) as usize;
        let data: Vec<f32> = (0..count).map(|i| base + i as f32).collect();
        mlxcel_core::from_slice_f32(&data, &[1, h, n, d])
    }

    /// Regression for #430: `first_cache_live_len` must return the
    /// full-attention `Cache::Standard` LIVE window (`live_len`), not the
    /// monotonic `offset`, so the prefill mask matches the K/V a continuation
    /// chunk's `update_and_fetch` returns after a `--max-kv-size` trim.
    #[test]
    fn first_cache_live_len_sizes_global_mask_from_live_window_after_trim() {
        const H: i32 = 2;
        const D: i32 = 4;
        let (n1, trimmed, m) = (8, 3, 4);
        let window = 6;

        // Layer 0 sliding (Rotating), layer 1 global (Standard): matches the
        // `first_cache_live_len` family lookups.
        let mut caches = vec![
            Cache::Rotating(RotatingKVCache::new(window)),
            Cache::Standard(KVCache::new()),
        ];
        if let Cache::Standard(ref mut c) = caches[1] {
            let _ = c.update_and_fetch(make_kv(H, n1, D, 0.0), make_kv(H, n1, D, 100.0));
            assert_eq!(c.trim_front(trimmed), trimmed);
        }

        let global_offset = first_cache_offset(&mut caches, "full_attention");
        let global_live_len = first_cache_live_len(&mut caches, "full_attention");
        assert_eq!(global_offset, n1, "first_cache_offset stays monotonic");
        assert_eq!(
            global_live_len,
            n1 - trimmed,
            "first_cache_live_len reports the trimmed live window"
        );
        assert!(global_live_len < global_offset);

        let (cont_k, _) = if let Cache::Standard(ref mut c) = caches[1] {
            c.update_and_fetch(make_kv(H, m, D, 200.0), make_kv(H, m, D, 300.0))
        } else {
            unreachable!()
        };
        let returned_klen = mlxcel_core::array_shape(&cont_k)[2];
        assert_eq!(returned_klen, global_live_len + m);

        let live_mask = create_causal_mask(m, global_live_len);
        mlxcel_core::eval(&live_mask);
        assert_eq!(
            *mlxcel_core::array_shape(&live_mask).last().unwrap(),
            returned_klen,
            "live_len-sized global mask must match the returned K/V"
        );
        let offset_mask = create_causal_mask(m, global_offset);
        mlxcel_core::eval(&offset_mask);
        assert!(
            *mlxcel_core::array_shape(&offset_mask).last().unwrap() > returned_klen,
            "offset-sized global mask must be wider than the returned K/V (the bug)"
        );
    }

    /// The sliding lookup of `first_cache_live_len` returns the
    /// `RotatingKVCache` live window (`seq_len`), which equals the keys it
    /// returns from `update_and_fetch`.
    #[test]
    fn first_cache_live_len_sliding_matches_returned_keys() {
        const H: i32 = 2;
        const D: i32 = 4;
        let window = 6;
        let m = 4;

        let mut caches = vec![
            Cache::Rotating(RotatingKVCache::new(window)),
            Cache::Standard(KVCache::new()),
        ];
        let (k, _) = if let Cache::Rotating(ref mut c) = caches[0] {
            c.update_and_fetch(make_kv(H, m, D, 0.0), make_kv(H, m, D, 100.0))
        } else {
            unreachable!()
        };
        let returned_klen = mlxcel_core::array_shape(&k)[2];
        assert_eq!(
            first_cache_live_len(&mut caches, "sliding_attention"),
            returned_klen,
            "sliding first_cache_live_len (== seq_len) must equal the returned key axis"
        );
    }

    // The Gemma 4 over-window (`sliding_prefill_mask_over_window_*`) and
    // within-window prefill-mask tests were hoisted to the shared
    // `mlxcel_core::utils::create_sliding_window_prefill_mask` tests in #410
    // (`sliding_window_prefill_mask_selects_full_when_fresh_over_window`,
    // `sliding_window_prefill_mask_within_window_matches_capped_builder`,
    // `full_windowed_mask_over_window_has_no_all_masked_row`). The Gemma 3
    // caller's no-trim invariant is additionally pinned in
    // `gemma3::gemma3_mask_tests`. Gemma 4's real-model over-window behaviour is
    // covered by the byte-identical regression on `gemma-4-12b-it-4bit`.
}

#[cfg(test)]
mod quant_scheme_tests {
    use super::{
        ModelArgs, QuantizationParams, dense_mlp_scaled_fused_path_eligible,
        dense_mlp_shared_quant_layout, per_layer_input_gate_fused_path_eligible,
        validate_quantization_scheme,
    };
    use mlxcel_core::dtype;
    use mlxcel_core::layers::QuantizedWeight;
    use mlxcel_core::{MlxArray, UniquePtr};
    use serde_json::json;

    /// Minimal quantized weight for guard-only tests: a small dummy
    /// weight/scale pair is enough since `per_layer_input_gate_fused_path_eligible`
    /// never touches the tensor payload, only the `global_scale` field.
    fn dummy_quantized_weight(global_scale: Option<UniquePtr<MlxArray>>) -> QuantizedWeight {
        QuantizedWeight {
            weight: mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
            scales: mlxcel_core::ones(&[2, 2], dtype::FLOAT32),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "nvfp4".to_string(),
            global_scale,
        }
    }

    #[test]
    fn accepts_mlx_native_affine_quantization() {
        // MLX-native quant records only group_size/bits (no quant_method).
        let cfg = json!({ "quantization": { "group_size": 64, "bits": 4 } });
        assert!(validate_quantization_scheme(&cfg).is_ok());
    }

    #[test]
    fn accepts_absent_quantization() {
        // Non-quantized model: nothing to validate.
        let cfg = json!({ "model_type": "gemma4" });
        assert!(validate_quantization_scheme(&cfg).is_ok());
    }

    #[test]
    fn accepts_supported_block_float_mode() {
        let cfg = json!({ "quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4" } });
        assert!(validate_quantization_scheme(&cfg).is_ok());
    }

    #[test]
    fn accepts_modelopt_nvfp4_repack_source() {
        let cfg = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "dynamic": false,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16
                        },
                        "targets": ["Linear"]
                    }
                }
            }
        });
        assert!(validate_quantization_scheme(&cfg).is_ok());
    }

    #[test]
    fn model_args_promotes_modelopt_nvfp4_quantization_config() {
        let args = ModelArgs {
            model_type: "gemma4".to_string(),
            text_config: json!({
                "model_type": "gemma4_text",
                "hidden_size": 5376,
                "num_hidden_layers": 1,
                "intermediate_size": 21504,
                "num_attention_heads": 32,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "num_key_value_heads": 16,
                "rope_parameters": {
                    "sliding_attention": { "rope_theta": 10000.0 },
                    "full_attention": { "rope_theta": 1000000.0 }
                },
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "layer_types": ["sliding_attention"]
            }),
            eos_token_id: None,
            quantization: None,
            quantization_config: Some(json!({
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "dynamic": false,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16
                        },
                        "targets": ["Linear"]
                    }
                }
            })),
        };

        let text = args.text_args();
        assert_eq!(text.group_size(), 16);
        assert_eq!(text.bits(), 4);
    }

    #[test]
    fn rejects_modelopt_without_nvfp4_repack_metadata() {
        for cfg in [
            json!({ "quantization_config": { "quant_method": "modelopt" } }),
            json!({
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "INT4",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "group_size": 128
                            }
                        }
                    }
                }
            }),
        ] {
            assert!(
                validate_quantization_scheme(&cfg).is_err(),
                "non-NVFP4 ModelOpt metadata must remain rejected: {cfg}"
            );
        }
    }

    #[test]
    fn rejects_optiq_quant_method_and_names_it() {
        // The issue #467 model: an OptiQ-tagged quantization must be rejected
        // with a message that names the offending scheme.
        let cfg = json!({
            "quantization": { "group_size": 64, "bits": 4, "quant_method": "optiq" }
        });
        let err = validate_quantization_scheme(&cfg).expect_err("OptiQ must be rejected");
        assert!(
            err.to_lowercase().contains("optiq"),
            "message must name the scheme: {err}"
        );
        assert!(
            err.contains("Unsupported quantization scheme"),
            "actionable message: {err}"
        );
    }

    #[test]
    fn rejects_unknown_mode() {
        let cfg = json!({ "quantization": { "group_size": 64, "bits": 4, "mode": "optiq" } });
        assert!(validate_quantization_scheme(&cfg).is_err());
    }

    #[test]
    fn rejects_awq_gptq_quant_method() {
        for method in ["awq", "gptq", "AWQ"] {
            let cfg = json!({ "quantization": { "quant_method": method } });
            assert!(
                validate_quantization_scheme(&cfg).is_err(),
                "external method {method} must be rejected"
            );
        }
    }

    #[test]
    fn detects_scheme_nested_under_text_config() {
        // VLM configs nest text quantization under text_config.
        let cfg = json!({
            "text_config": { "quantization": { "quant_method": "optiq" } }
        });
        assert!(validate_quantization_scheme(&cfg).is_err());
    }

    #[test]
    fn detects_hf_quantization_config_object() {
        // HF-style external formats use a `quantization_config` object.
        let cfg = json!({ "quantization_config": { "quant_method": "gptq", "bits": 4 } });
        assert!(validate_quantization_scheme(&cfg).is_err());
    }

    #[test]
    fn dense_mlp_shared_quant_layout_accepts_uniform_frontier_variant() {
        let q = QuantizationParams {
            group_size: 32,
            bits: 4,
        };
        assert!(
            dense_mlp_shared_quant_layout(q, "affine", q, "affine", q, "affine"),
            "uniform gs32 MLP variants can use the shared-layout fused helper"
        );
    }

    #[test]
    fn dense_mlp_shared_quant_layout_rejects_mixed_down8_frontier_variant() {
        let q4 = QuantizationParams {
            group_size: 64,
            bits: 4,
        };
        let down8 = QuantizationParams {
            group_size: 64,
            bits: 8,
        };
        assert!(
            !dense_mlp_shared_quant_layout(q4, "affine", q4, "affine", down8, "affine"),
            "mixed down-proj-8-bit variants must use projection-local forwards because \
             compiled_gelu_approx_mlp_forward accepts only one quant layout"
        );
    }

    /// `MLP::forward`'s dense-MLP scaled fused path (issue #698 follow-up)
    /// mirrors `per_layer_input_gate_fused_path_eligible`'s single-token +
    /// kill-switch gating: no sidecar is unaffected by either toggle, a
    /// sidecar-carrying MLP is eligible only for single-token decode with the
    /// fold enabled, and multi-token bypasses the fused path regardless of
    /// the kill switch.
    #[test]
    fn dense_mlp_scaled_fused_path_eligible_no_sidecar_always_false() {
        // `any_sidecar = false` means `MLP::forward` takes the separate
        // unscaled fused path instead, so this helper is never consulted for
        // that branch; it must still report `false` for every toggle
        // combination since the scaled-path call itself is guarded on
        // `any_sidecar` first.
        for fused_scale_enabled in [true, false] {
            for single_token in [true, false] {
                assert!(
                    !dense_mlp_scaled_fused_path_eligible(
                        false,
                        fused_scale_enabled,
                        single_token,
                        16,
                        4,
                        "nvfp4"
                    ),
                    "no sidecar must never route through the scaled fused path \
                     (fused_scale_enabled={fused_scale_enabled}, single_token={single_token})"
                );
            }
        }
    }

    #[test]
    fn dense_mlp_scaled_fused_path_eligible_sidecar_single_token_enabled() {
        assert!(
            dense_mlp_scaled_fused_path_eligible(true, true, true, 16, 4, "nvfp4"),
            "sidecar + single-token + fold enabled must take the scaled fused path"
        );
    }

    #[test]
    fn dense_mlp_scaled_fused_path_eligible_native_nvfp4_prefill_enabled() {
        assert!(
            dense_mlp_scaled_fused_path_eligible(true, true, false, 16, 4, "nvfp4"),
            "sidecar + native NVFP4 multi-token prefill must take the shape-specific scaled \
             fused path when the fold is enabled"
        );
    }

    #[test]
    fn dense_mlp_scaled_fused_path_eligible_unknown_prefill_layout_bypasses() {
        for (group_size, bits, mode) in [(64, 4, "affine"), (64, 8, "affine"), (32, 4, "mxfp4")] {
            assert!(
                !dense_mlp_scaled_fused_path_eligible(true, true, false, group_size, bits, mode),
                "sidecar + multi-token must bypass the scaled fused path for unknown layouts \
                 (group_size={group_size}, bits={bits}, mode={mode})"
            );
        }
    }

    #[test]
    fn dense_mlp_scaled_fused_path_eligible_sidecar_rejects_when_fold_disabled() {
        assert!(
            !dense_mlp_scaled_fused_path_eligible(true, false, true, 16, 4, "nvfp4"),
            "sidecar must fall back to op-at-a-time when the fold is disabled even for \
             single-token decode"
        );
        assert!(
            !dense_mlp_scaled_fused_path_eligible(true, false, false, 16, 4, "nvfp4"),
            "sidecar must fall back to op-at-a-time when the fold is disabled for native \
             NVFP4 prefill"
        );
    }

    /// A per-layer-input gate/projection pair with no native-NVFP4
    /// `global_scale` sidecar on either side is always eligible for the
    /// compiled `compiled_per_layer_input_gate` fused path, regardless of the
    /// fold toggle or the query length (the unscaled graph is unchanged by
    /// issue #698 and its single-token follow-up).
    #[test]
    fn per_layer_input_gate_fused_path_eligible_without_global_scale() {
        let gate_qw = dummy_quantized_weight(None);
        let proj_qw = dummy_quantized_weight(None);
        for fused_scale_enabled in [true, false] {
            for single_token in [true, false] {
                assert!(
                    per_layer_input_gate_fused_path_eligible(
                        &gate_qw,
                        &proj_qw,
                        fused_scale_enabled,
                        single_token
                    ),
                    "no global_scale on either projection must always take the compiled fused \
                     path (fused_scale_enabled={fused_scale_enabled}, single_token={single_token})"
                );
            }
        }
    }

    /// Issue #698: the fused `compiled_per_layer_input_gate` path now folds the
    /// native-NVFP4 `weight_scale_2` sidecar, so a gate carrying one is
    /// eligible when the fold is enabled AND the call is single-token decode.
    #[test]
    fn per_layer_input_gate_accepts_fused_path_when_gate_has_global_scale() {
        let gate_qw = dummy_quantized_weight(Some(mlxcel_core::ones(&[1], dtype::FLOAT32)));
        let proj_qw = dummy_quantized_weight(None);
        assert!(
            per_layer_input_gate_fused_path_eligible(&gate_qw, &proj_qw, true, true),
            "a gate global_scale sidecar must take the fused path for single-token decode when \
             the fold is enabled"
        );
    }

    /// Issue #698 kill switch: with the fold disabled
    /// (`MLXCEL_DISABLE_FUSED_GLOBAL_SCALE`), a sidecar-carrying pair must fall
    /// through to the op-at-a-time path, where `UnifiedLinear::forward` applies
    /// the scalar. Covers a sidecar on the gate, on the proj, and on both, for
    /// a single-token call (the fold's only otherwise-eligible shape).
    #[test]
    fn per_layer_input_gate_rejects_fused_path_when_fold_disabled() {
        let with = || dummy_quantized_weight(Some(mlxcel_core::ones(&[1], dtype::FLOAT32)));
        let without = || dummy_quantized_weight(None);
        for (gate_qw, proj_qw, label) in [
            (with(), without(), "gate-only sidecar"),
            (without(), with(), "proj-only sidecar"),
            (with(), with(), "both sidecars"),
        ] {
            assert!(
                !per_layer_input_gate_fused_path_eligible(&gate_qw, &proj_qw, false, true),
                "{label}: a sidecar-carrying pair must fall back to op-at-a-time when the fold \
                 is disabled"
            );
        }
    }

    /// Issue #698 follow-up: a sidecar-carrying pair must bypass the fused
    /// path for a multi-token (prefill) call regardless of the fold toggle.
    /// The C++ bridge falls back to an uncompiled eager fold for multi-token
    /// sidecar calls, which regressed 2048-token prefill throughput by 8.3%
    /// versus the compiled op-at-a-time activation the bypass below uses, so
    /// multi-token must never route into the fused path even with the fold
    /// enabled.
    #[test]
    fn per_layer_input_gate_rejects_fused_path_for_multi_token_regardless_of_fold() {
        let with = || dummy_quantized_weight(Some(mlxcel_core::ones(&[1], dtype::FLOAT32)));
        let without = || dummy_quantized_weight(None);
        for (gate_qw, proj_qw, label) in [
            (with(), without(), "gate-only sidecar"),
            (without(), with(), "proj-only sidecar"),
            (with(), with(), "both sidecars"),
        ] {
            for fused_scale_enabled in [true, false] {
                assert!(
                    !per_layer_input_gate_fused_path_eligible(
                        &gate_qw,
                        &proj_qw,
                        fused_scale_enabled,
                        false
                    ),
                    "{label}: multi-token must bypass the fused path even when the fold is \
                     enabled (fused_scale_enabled={fused_scale_enabled})"
                );
            }
        }
    }

    /// Issue #698 mixed case: only the projection side carries a sidecar. The
    /// pair is eligible for a single-token call when the fold is enabled (the
    /// fused path treats the absent gate scale as no multiply) and ineligible
    /// when the fold is disabled or the call is multi-token.
    #[test]
    fn per_layer_input_gate_mixed_proj_only_sidecar_follows_fold_toggle() {
        let gate_qw = dummy_quantized_weight(None);
        let proj_qw = dummy_quantized_weight(Some(mlxcel_core::ones(&[1], dtype::FLOAT32)));
        assert!(
            per_layer_input_gate_fused_path_eligible(&gate_qw, &proj_qw, true, true),
            "proj-only sidecar must take the fused path for single-token decode when the fold \
             is enabled"
        );
        assert!(
            !per_layer_input_gate_fused_path_eligible(&gate_qw, &proj_qw, false, true),
            "proj-only sidecar must fall back when the fold is disabled"
        );
        assert!(
            !per_layer_input_gate_fused_path_eligible(&gate_qw, &proj_qw, true, false),
            "proj-only sidecar must fall back for multi-token even when the fold is enabled"
        );
    }
}

/// Unit coverage for issue #691: `QuantizationArgs::quant_params_for` must
/// diagnose malformed per-module overrides (a non-object value, or a
/// `group_size` / `bits` field with the wrong type, a negative value, a zero
/// group size, or an unsupported bit width) while still resolving valid
/// overrides and root-only configs unchanged.
#[cfg(test)]
mod quant_override_diagnostics_tests {
    use super::{QuantizationArgs, QuantizationParams};
    use serde_json::json;
    use std::collections::HashMap;

    fn args_with_override(
        group_size: usize,
        bits: usize,
        key: &str,
        value: serde_json::Value,
    ) -> QuantizationArgs {
        let mut overrides = HashMap::new();
        overrides.insert(key.to_string(), value);
        QuantizationArgs {
            group_size,
            bits,
            overrides,
        }
    }

    #[test]
    fn valid_override_applies_both_fields() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "group_size": 32, "bits": 8 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj"),
            QuantizationParams {
                group_size: 32,
                bits: 8
            }
        );
    }

    #[test]
    fn valid_partial_override_keeps_root_default_for_the_omitted_field() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.down_proj",
            json!({ "bits": 8 }),
        );
        let params = args.quant_params_for("language_model.model.layers.0.mlp.down_proj");
        assert_eq!(
            params.group_size, 64,
            "group_size absent from the override must retain the root default"
        );
        assert_eq!(params.bits, 8);
    }

    #[test]
    fn absent_override_returns_root_defaults_unchanged() {
        let args = QuantizationArgs {
            group_size: 64,
            bits: 4,
            overrides: HashMap::new(),
        };
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj"),
            QuantizationParams {
                group_size: 64,
                bits: 4
            }
        );
    }

    #[test]
    fn non_object_override_value_falls_back_to_root_defaults() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!(4),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj"),
            QuantizationParams {
                group_size: 64,
                bits: 4
            },
            "a scalar override value must diagnose and fall back to root defaults instead of \
             panicking or being silently misinterpreted"
        );
    }

    #[test]
    fn string_typed_bits_falls_back_to_root_bits_but_sibling_group_size_still_applies() {
        // Regression for issue #691: "bits": "4" (string) must not be
        // silently treated as if it were the integer 4.
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "group_size": 32, "bits": "4" }),
        );
        let params = args.quant_params_for("language_model.model.layers.0.mlp.gate_proj");
        assert_eq!(
            params.group_size, 32,
            "the well-formed group_size sibling must still apply"
        );
        assert_eq!(
            params.bits, 4,
            "a string-typed bits value must fall back to the root default, not parse as 4"
        );
    }

    #[test]
    fn negative_group_size_falls_back_to_root_default() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "group_size": -32 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj")
                .group_size,
            64
        );
    }

    #[test]
    fn zero_group_size_falls_back_to_root_default() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "group_size": 0 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj")
                .group_size,
            64
        );
    }

    #[test]
    fn unsupported_bit_width_falls_back_to_root_default() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "bits": 7 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj")
                .bits,
            4
        );
    }

    #[test]
    fn float_typed_group_size_falls_back_to_root_default() {
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_proj",
            json!({ "group_size": 32.5 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj")
                .group_size,
            64
        );
    }

    #[test]
    fn sibling_metadata_keys_are_not_mistaken_for_the_queried_module_override() {
        // `mode` / `quant_method` / `quant_algo` / `config_groups` legitimately
        // live alongside per-module overrides in the flattened `overrides` map
        // (see `validate_quantization_scheme`); their presence must never
        // affect resolution of an unrelated, well-formed per-module override.
        let mut overrides = HashMap::new();
        overrides.insert("mode".to_string(), json!("mxfp4"));
        overrides.insert(
            "language_model.model.layers.0.mlp.gate_proj".to_string(),
            json!({ "group_size": 32, "bits": 4 }),
        );
        let args = QuantizationArgs {
            group_size: 64,
            bits: 4,
            overrides,
        };
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj"),
            QuantizationParams {
                group_size: 32,
                bits: 4
            }
        );
    }

    #[test]
    fn typo_prefix_never_matched_falls_back_to_root_defaults_for_the_real_prefix() {
        // A typo'd override key never collides with a differently-named real
        // prefix lookup; the queried (correctly spelled) module simply sees
        // no override and keeps the root defaults.
        let args = args_with_override(
            64,
            4,
            "language_model.model.layers.0.mlp.gate_prj",
            json!({ "group_size": 32, "bits": 8 }),
        );
        assert_eq!(
            args.quant_params_for("language_model.model.layers.0.mlp.gate_proj"),
            QuantizationParams {
                group_size: 64,
                bits: 4
            }
        );
    }
}
