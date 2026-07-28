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

//! Architecture-aware KV-cache memory estimation (epic #52, TIER 1).
//!
//! The flat formula `num_layers × 2 × num_kv_heads × head_dim × ctx × elem ×
//! batch` is only correct for plain full-context MHA/GQA (Llama, Qwen3). It
//! mis-estimates — sometimes by 100×+ — for the architectures mlxcel actually
//! runs:
//!
//! - **Sliding-window** (Gemma 2/3/3n/4, RecurrentGemma's attention layers):
//!   sliding layers cap their KV at the window, not the full context, so the
//!   flat formula *overcounts* at long context. Gemma 3 splits layers — one in
//!   every `sliding_window_pattern` is global (full context), the rest are
//!   windowed.
//! - **MLA / compressed KV** (DeepSeek): V3/V3.2 cache a low-rank latent
//!   (`kv_lora_rank + qk_rope_head_dim`) shared across heads — far smaller than
//!   the per-head formula (*overcount*). V2 caches the *decompressed* K/V
//!   (`num_heads × (qk_nope + qk_rope + v_head_dim)`), which can be *larger*
//!   than `2 × num_kv_heads × head_dim` (*undercount* — dangerous for a budget).
//! - **Hybrid attention+SSM** (Jamba, NemotronH, RecurrentGemma, Qwen3-Next):
//!   only the attention layers hold a context-proportional KV cache; the
//!   recurrent/Mamba layers carry an O(1) state, negligible for budgeting. The
//!   flat formula counts every layer as attention (*overcount*).
//! - **Pure SSM** (Mamba, Mamba2): no KV cache at all (*phantom* KV).
//!
//! This module parses `config.json` into a set of homogeneous [`KvGroup`]s and
//! sums their context-aware footprint. It degrades gracefully: any model whose
//! special fields are absent falls back to the standard full-context formula,
//! reproducing the previous behaviour exactly. The per-architecture field names
//! and cache shapes are taken from mlxcel's own model implementations (see the
//! comments on each branch of [`classify`]).

use std::path::Path;

use mlxcel_core::hardware::KV_CACHE_ALLOC_STEP;
use serde_json::Value;

use crate::execution::config_fields::{
    HEAD_DIM_KEYS, HIDDEN_SIZE_KEYS, LAYER_COUNT_KEYS, NUM_HEADS_KEYS, get_u64,
    resolve_num_kv_heads,
};

/// Which KV architecture [`estimate_kv_arch`] detected, for labelling output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvArchKind {
    /// Plain full-context MHA/GQA (Llama, Qwen3).
    Standard,
    /// Some or all attention layers bound their KV to a sliding window.
    SlidingWindow,
    /// DeepSeek V3/V3.2 — KV cached as a compressed low-rank latent.
    MlaCompressed,
    /// DeepSeek V2 — KV cached decompressed (per-head).
    MlaDecompressed,
    /// Attention + recurrent/SSM layers; only attention layers hold KV.
    Hybrid,
    /// Pure SSM (Mamba/Mamba2) — no context-proportional KV cache.
    PureSsm,
}

impl KvArchKind {
    /// Short label for `mlxcel inspect`.
    pub fn label(self) -> &'static str {
        match self {
            KvArchKind::Standard => "standard attention",
            KvArchKind::SlidingWindow => "sliding-window attention",
            KvArchKind::MlaCompressed => "MLA (compressed latent)",
            KvArchKind::MlaDecompressed => "MLA (decompressed)",
            KvArchKind::Hybrid => "hybrid attention + recurrent",
            KvArchKind::PureSsm => "pure SSM (no KV cache)",
        }
    }
}

/// One homogeneous group of KV-holding layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvGroup {
    /// Number of layers in the group.
    layers: u64,
    /// K+V elements stored per token per layer (K and V already summed).
    elems_per_token: u64,
    /// Sliding-window cap in tokens; `None` for full-context layers.
    window: Option<u64>,
}

/// Architecture-aware KV estimate produced by [`estimate_kv_arch`].
#[derive(Debug, Clone)]
pub struct KvArchEstimate {
    /// Total KV-cache bytes at the requested context/batch/dtype.
    pub total_bytes: u64,
    /// Steady-state marginal bytes per *additional* token: the sum over
    /// full-context (uncapped) layers only. Sliding/SSM layers stop growing
    /// past their window, so they contribute 0 to the long-context margin.
    /// This is the meaningful "cost per token" once the window is saturated.
    pub marginal_bytes_per_token: u64,
    /// Detected architecture class.
    pub kind: KvArchKind,
    /// One-line human description for `mlxcel inspect`.
    pub detail: String,
}

/// Round a token count up to the next [`KV_CACHE_ALLOC_STEP`] (256) so the
/// estimate matches the cache's actual block pre-allocation.
fn round_ctx(tokens: u64) -> u64 {
    tokens
        .saturating_add(KV_CACHE_ALLOC_STEP - 1)
        .checked_div(KV_CACHE_ALLOC_STEP)
        .unwrap_or(0)
        .saturating_mul(KV_CACHE_ALLOC_STEP)
}

fn elem_bytes(int8: bool) -> u64 {
    if int8 { 1 } else { 2 }
}

fn get_str<'a>(obj: &'a Value, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(|v| v.as_str())
}

/// Per-head attention dimensions for full-attention layers.
struct AttnDims {
    num_kv_heads: u64,
    head_dim: u64,
}

/// Parse the per-head dimensions. Returns `None` when `head_dim` is underivable
/// (no explicit field and no `hidden_size / num_heads`), which signals the
/// caller that a standard / sliding / hybrid KV figure cannot be computed. The
/// MLA and pure-SSM paths do not call this — they size their cache from
/// architecture-specific fields and need only the layer count.
///
/// Field names come from [`crate::execution::config_fields`], shared with the
/// activation estimator and the KV-cache-mode advisor so the three never
/// disagree about which spellings a config may use. The KV-head count goes
/// through `resolve_num_kv_heads`, which honours GPT-BigCode's boolean
/// `multi_query` as one shared head.
fn attn_dims(text: &Value) -> Option<AttnDims> {
    let num_heads = get_u64(text, NUM_HEADS_KEYS).unwrap_or(1).max(1);
    let num_kv_heads = resolve_num_kv_heads(text, num_heads);
    let head_dim = match get_u64(text, HEAD_DIM_KEYS) {
        Some(h) => h,
        None => get_u64(text, HIDDEN_SIZE_KEYS)?
            .checked_div(num_heads)
            .unwrap_or(64),
    };
    Some(AttnDims {
        num_kv_heads,
        head_dim,
    })
}

/// Standard per-layer K+V element count: `2 × num_kv_heads × head_dim`.
fn standard_elems(dims: &AttnDims) -> u64 {
    2u64.saturating_mul(dims.num_kv_heads)
        .saturating_mul(dims.head_dim)
}

/// Count the attention (KV-holding) layers of a hybrid model, trying each
/// layer-typing scheme mlxcel's hybrid models use. Returns `None` when no
/// hybrid scheme is present (the model is not hybrid).
fn count_attention_layers(text: &Value, num_layers: u64) -> Option<u64> {
    // Explicit per-layer arrays (Jamba `layers_block_type`, RecurrentGemma
    // `block_types`, Gemma3n/4 use `layer_types` but that is sliding, handled
    // elsewhere). An entry is attention when it names attention / "*".
    for key in ["layers_block_type", "block_types"] {
        if let Some(arr) = text.get(key).and_then(|v| v.as_array()) {
            let n = arr
                .iter()
                .filter(|e| {
                    e.as_str()
                        .map(|s| {
                            let s = s.to_ascii_lowercase();
                            s.contains("attention") || s == "attn" || s == "*"
                        })
                        .unwrap_or(false)
                })
                .count() as u64;
            return Some(n);
        }
    }
    // NemotronH `hybrid_override_pattern`: a string (or char array) where "*"
    // marks an attention layer (vs "M" mamba, "-" / "E" others).
    if let Some(pat) = get_str(text, "hybrid_override_pattern") {
        return Some(pat.chars().filter(|c| *c == '*').count() as u64);
    }
    if let Some(arr) = text
        .get("hybrid_override_pattern")
        .and_then(|v| v.as_array())
    {
        let n = arr
            .iter()
            .filter(|e| e.as_str().map(|s| s == "*").unwrap_or(false))
            .count() as u64;
        return Some(n);
    }
    // Jamba periodic schedule: layer i is attention when
    // `(i + attn_layer_offset) % attn_layer_period == 0`.
    if let Some(period) = get_u64(text, &["attn_layer_period"]).filter(|p| *p > 0) {
        let offset = get_u64(text, &["attn_layer_offset"]).unwrap_or(0);
        let n = (0..num_layers)
            .filter(|i| (i + offset).is_multiple_of(period))
            .count() as u64;
        return Some(n);
    }
    // Qwen3-Next: full attention when `(i + 1) % full_attention_interval == 0`,
    // excluding `mlp_only_layers`.
    if let Some(interval) = get_u64(text, &["full_attention_interval"]).filter(|p| *p > 0) {
        let mlp_only: std::collections::HashSet<u64> = text
            .get("mlp_only_layers")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|e| e.as_u64()).collect())
            .unwrap_or_default();
        let n = (0..num_layers)
            .filter(|i| (i + 1).is_multiple_of(interval) && !mlp_only.contains(i))
            .count() as u64;
        return Some(n);
    }
    None
}

/// Classify a model's config into KV layer groups and an architecture kind.
fn classify(text: &Value, model_type: &str) -> Option<(Vec<KvGroup>, KvArchKind)> {
    let num_layers = get_u64(text, LAYER_COUNT_KEYS)?;
    let mt = model_type.to_ascii_lowercase();

    // 1. Pure SSM — no context-proportional KV cache (needs only the layer count).
    if mt == "mamba" || mt == "mamba2" || mt == "falcon_mamba" {
        return Some((Vec::new(), KvArchKind::PureSsm));
    }

    // 2. MLA / compressed KV (DeepSeek). `kv_lora_rank` is the tell. MLA sizes
    //    its cache from explicit latent/rope/v dims, so it needs the head count
    //    but NOT `head_dim` (a config may omit it). V3/V3.2 cache the compressed
    //    latent + rope (shared across heads); V2 caches decompressed per-head.
    if let Some(kv_lora_rank) = get_u64(text, &["kv_lora_rank"]) {
        let num_heads = get_u64(text, NUM_HEADS_KEYS).unwrap_or(1).max(1);
        let rope = get_u64(text, &["qk_rope_head_dim"]).unwrap_or(64);
        if mt.contains("v3") || mt.contains("v32") {
            // Cached form: (kv_latent[kv_lora_rank] , k_pe[qk_rope_head_dim]).
            let elems = kv_lora_rank.saturating_add(rope);
            return Some((
                vec![KvGroup {
                    layers: num_layers,
                    elems_per_token: elems,
                    window: None,
                }],
                KvArchKind::MlaCompressed,
            ));
        }
        // V2: decompressed K (num_heads × (qk_nope + qk_rope)) + V
        // (num_heads × v_head_dim). Defaults match the DeepSeek-V2 config.
        let qk_nope = get_u64(text, &["qk_nope_head_dim"]).unwrap_or(128);
        let v_head_dim = get_u64(text, &["v_head_dim"]).unwrap_or(128);
        let per_head = qk_nope.saturating_add(rope).saturating_add(v_head_dim);
        let elems = num_heads.saturating_mul(per_head);
        return Some((
            vec![KvGroup {
                layers: num_layers,
                elems_per_token: elems,
                window: None,
            }],
            KvArchKind::MlaDecompressed,
        ));
    }

    // Full-attention dimensions (and a derivable `head_dim`) are required for
    // every remaining class. Bail to "unavailable" if they cannot be parsed.
    let dims = attn_dims(text)?;

    // 3. Hybrid attention + recurrent/SSM — only attention layers hold KV.
    if let Some(n_attn) = count_attention_layers(text, num_layers) {
        // RecurrentGemma bounds its attention layers to `attention_window_size`.
        let window = get_u64(text, &["attention_window_size"]);
        return Some((
            vec![KvGroup {
                layers: n_attn,
                elems_per_token: standard_elems(&dims),
                window,
            }],
            KvArchKind::Hybrid,
        ));
    }

    // 4. Sliding-window attention (Gemma family). The flat formula overcounts
    //    because windowed layers cap their KV at `sliding_window`.
    if let Some(window) = get_u64(text, &["sliding_window"]).filter(|w| *w > 0) {
        let elems = standard_elems(&dims);
        // 4a. Per-layer `layer_types` array (Gemma 3n / 4): each entry is
        //     "full_attention" (global) or "sliding_attention" (windowed).
        if let Some(arr) = text.get("layer_types").and_then(|v| v.as_array()) {
            let mut global = 0u64;
            let mut sliding = 0u64;
            for e in arr {
                match e.as_str() {
                    Some(s) if s.eq_ignore_ascii_case("full_attention") => global += 1,
                    Some(_) => sliding += 1,
                    None => {}
                }
            }
            let mut groups = Vec::new();
            if global > 0 {
                groups.push(KvGroup {
                    layers: global,
                    elems_per_token: elems,
                    window: None,
                });
            }
            if sliding > 0 {
                groups.push(KvGroup {
                    layers: sliding,
                    elems_per_token: elems,
                    window: Some(window),
                });
            }
            if !groups.is_empty() {
                return Some((groups, KvArchKind::SlidingWindow));
            }
        }
        // 4b. `sliding_window_pattern` (Gemma 3): layer i is global when
        //     `(i + 1) % pattern == 0`, the rest are windowed. A pattern of 1
        //     means every layer is global (i.e. no real sliding).
        let pattern = get_u64(text, &["sliding_window_pattern"])
            .unwrap_or(1)
            .max(1);
        let global = num_layers / pattern;
        let sliding = num_layers.saturating_sub(global);
        let mut groups = Vec::new();
        if global > 0 {
            groups.push(KvGroup {
                layers: global,
                elems_per_token: elems,
                window: None,
            });
        }
        if sliding > 0 {
            groups.push(KvGroup {
                layers: sliding,
                elems_per_token: elems,
                window: Some(window),
            });
        }
        let kind = if sliding > 0 {
            KvArchKind::SlidingWindow
        } else {
            KvArchKind::Standard
        };
        return Some((groups, kind));
    }

    // 5. Standard full-context attention — the original formula.
    Some((
        vec![KvGroup {
            layers: num_layers,
            elems_per_token: standard_elems(&dims),
            window: None,
        }],
        KvArchKind::Standard,
    ))
}

/// Compute the total bytes and steady-state marginal rate for a set of groups.
fn sum_groups(groups: &[KvGroup], ctx: u64, int8: bool, batch: u64) -> (u64, u64) {
    let eb = elem_bytes(int8);
    let rounded_ctx = round_ctx(ctx);
    let mut total = 0u64;
    let mut marginal = 0u64;
    for g in groups {
        let effective = match g.window {
            Some(w) => rounded_ctx.min(round_ctx(w)),
            None => rounded_ctx,
        };
        let per_layer_at_ctx = g
            .elems_per_token
            .saturating_mul(effective)
            .saturating_mul(eb)
            .saturating_mul(batch);
        total = total.saturating_add(per_layer_at_ctx.saturating_mul(g.layers));
        if g.window.is_none() {
            let per_layer_per_token = g.elems_per_token.saturating_mul(eb).saturating_mul(batch);
            marginal = marginal.saturating_add(per_layer_per_token.saturating_mul(g.layers));
        }
    }
    (total, marginal)
}

/// Build the human-readable architecture detail line.
fn detail_line(groups: &[KvGroup], kind: KvArchKind) -> String {
    match kind {
        KvArchKind::PureSsm => "pure SSM — no KV cache".to_string(),
        KvArchKind::MlaCompressed => {
            let elems = groups.first().map(|g| g.elems_per_token).unwrap_or(0);
            format!("MLA compressed latent ({elems} elems/token/layer, shared across heads)")
        }
        KvArchKind::MlaDecompressed => {
            "MLA decompressed per-head K/V (DeepSeek-V2 style)".to_string()
        }
        KvArchKind::Hybrid => {
            let attn: u64 = groups.iter().map(|g| g.layers).sum();
            let windowed = groups.iter().any(|g| g.window.is_some());
            if windowed {
                format!("hybrid: {attn} attention layer(s), windowed")
            } else {
                format!("hybrid: {attn} attention layer(s) hold KV (rest recurrent/SSM)")
            }
        }
        KvArchKind::SlidingWindow => {
            let global: u64 = groups
                .iter()
                .filter(|g| g.window.is_none())
                .map(|g| g.layers)
                .sum();
            let sliding: u64 = groups
                .iter()
                .filter(|g| g.window.is_some())
                .map(|g| g.layers)
                .sum();
            let window = groups.iter().find_map(|g| g.window).unwrap_or(0);
            format!("sliding-window: {sliding} layer(s) capped at {window} tokens, {global} global")
        }
        KvArchKind::Standard => {
            let layers: u64 = groups.iter().map(|g| g.layers).sum();
            format!("standard attention ({layers} layers, full context)")
        }
    }
}

/// Architecture-aware KV-cache estimate for `model_dir` at the given
/// context / dtype / batch. Returns `None` only when `config.json` is missing
/// the basic architecture fields (the caller then reports KV as unavailable).
///
/// Pure: reads `model_dir/config.json` only; no MLX state is touched.
#[must_use]
pub fn estimate_kv_arch(
    model_dir: &Path,
    ctx_len: u64,
    int8_kv: bool,
    batch: u64,
) -> Option<KvArchEstimate> {
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let config: Value = serde_json::from_str(&config_str).ok()?;
    estimate_kv_arch_from_config(&config, ctx_len, int8_kv, batch)
}

/// [`estimate_kv_arch`] against an already-parsed config value. Split out for
/// testability without touching the filesystem.
#[must_use]
pub fn estimate_kv_arch_from_config(
    config: &Value,
    ctx_len: u64,
    int8_kv: bool,
    batch: u64,
) -> Option<KvArchEstimate> {
    // VLMs nest the decoder under `text_config`; `model_type` may live at the
    // top level (the VLM type) or inside `text_config` (the decoder type).
    let text = config.get("text_config").unwrap_or(config);
    let model_type = get_str(text, "model_type")
        .or_else(|| get_str(config, "model_type"))
        .unwrap_or("");

    let (groups, kind) = classify(text, model_type)?;
    let (total_bytes, marginal_bytes_per_token) = sum_groups(&groups, ctx_len, int8_kv, batch);
    let detail = detail_line(&groups, kind);
    Some(KvArchEstimate {
        total_bytes,
        marginal_bytes_per_token,
        kind,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FP16: u64 = 2;

    fn est(cfg: &Value, ctx: u64) -> KvArchEstimate {
        estimate_kv_arch_from_config(cfg, ctx, false, 1).expect("estimate")
    }

    #[test]
    fn standard_matches_flat_formula() {
        let cfg = json!({
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::Standard);
        // 32 layers × 2 × 8 kv heads × 128 × 8192 × 2 bytes.
        assert_eq!(e.total_bytes, 32 * 2 * 8 * 128 * 8192 * FP16);
        assert_eq!(e.marginal_bytes_per_token, 32 * 2 * 8 * 128 * FP16);
    }

    #[test]
    fn gqa_head_dim_from_hidden_when_absent() {
        let cfg = json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "hidden_size": 2048, // head_dim = 128
        });
        let e = est(&cfg, 256);
        // num_kv_heads defaults to num_heads (16), head_dim 128.
        assert_eq!(e.total_bytes, 4 * 2 * 16 * 128 * 256 * FP16);
    }

    #[test]
    fn sliding_window_pattern_splits_global_and_windowed() {
        // Gemma3: 1 in every 6 layers is global, the rest windowed at 1024.
        let cfg = json!({
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::SlidingWindow);
        let elems = 2 * 256; // 2 (K+V) × 1 kv_head × 256 head_dim
        let global = 12 / 6; // 2 global layers, full ctx (8192)
        let sliding = 12 - global; // 10 windowed layers, capped at 1024
        let expected = global * elems * 8192 * FP16 + sliding * elems * 1024 * FP16;
        assert_eq!(e.total_bytes, expected);
        // Only the 2 global layers grow per token at steady state.
        assert_eq!(e.marginal_bytes_per_token, global * elems * FP16);
    }

    #[test]
    fn sliding_window_is_a_massive_saving_vs_flat() {
        let cfg = json!({
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
        });
        let arch = est(&cfg, 32768).total_bytes;
        let flat = 26u64 * 2 * 256 * 32768 * FP16; // naive full-ctx (1 kv_head)
        assert!(
            arch < flat / 4,
            "sliding-window estimate {arch} should be far below flat {flat}"
        );
    }

    #[test]
    fn layer_types_array_counts_global_vs_sliding() {
        let cfg = json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "sliding_window": 512,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
        });
        let e = est(&cfg, 4096);
        assert_eq!(e.kind, KvArchKind::SlidingWindow);
        let elems = 2 * 64; // 1 kv_head × 64 head_dim
        let expected = elems * 4096 * FP16 + 3 * elems * 512 * FP16;
        assert_eq!(e.total_bytes, expected);
    }

    #[test]
    fn mla_v3_uses_compressed_latent() {
        let cfg = json!({
            "model_type": "deepseek_v3",
            "num_hidden_layers": 4,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "head_dim": 128,
        });
        let e = est(&cfg, 4096);
        assert_eq!(e.kind, KvArchKind::MlaCompressed);
        // Compressed: (512 + 64) per token per layer, shared across heads.
        let elems = 512 + 64;
        assert_eq!(e.total_bytes, 4 * elems * 4096 * FP16);
        // Far below the naive per-head formula.
        let naive = 4u64 * 2 * 128 * 128 * 4096 * FP16;
        assert!(e.total_bytes < naive / 10);
    }

    #[test]
    fn mla_v2_uses_decompressed_per_head() {
        let cfg = json!({
            "model_type": "deepseek_v2",
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
        });
        let e = est(&cfg, 2048);
        assert_eq!(e.kind, KvArchKind::MlaDecompressed);
        // num_heads × (qk_nope + qk_rope + v_head_dim) = 16 × (128+64+128).
        let elems = 16 * (128 + 64 + 128);
        assert_eq!(e.total_bytes, 4 * elems * 2048 * FP16);
    }

    #[test]
    fn jamba_counts_only_attention_layers() {
        // attn_layer_period = 4 → attention at i = 0,4,8,12 of 16 layers.
        let cfg = json!({
            "model_type": "jamba",
            "num_hidden_layers": 16,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "attn_layer_period": 4,
            "attn_layer_offset": 0,
        });
        let e = est(&cfg, 4096);
        assert_eq!(e.kind, KvArchKind::Hybrid);
        let n_attn = 4; // i in {0,4,8,12}
        let elems = 2 * 8 * 64;
        assert_eq!(e.total_bytes, n_attn * elems * 4096 * FP16);
    }

    #[test]
    fn layers_block_type_array_counts_attention() {
        let cfg = json!({
            "model_type": "jamba",
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "layers_block_type": ["mamba", "attention", "mamba", "mamba"],
        });
        let e = est(&cfg, 1024);
        assert_eq!(e.kind, KvArchKind::Hybrid);
        let elems = 2 * 2 * 64;
        assert_eq!(e.total_bytes, elems * 1024 * FP16);
    }

    #[test]
    fn nemotron_h_pattern_counts_star_layers() {
        let cfg = json!({
            "model_type": "nemotron_h",
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "hybrid_override_pattern": "M*M-M*M-",
        });
        let e = est(&cfg, 2048);
        assert_eq!(e.kind, KvArchKind::Hybrid);
        let n_attn = 2; // two '*'
        let elems = 2 * 8 * 64;
        assert_eq!(e.total_bytes, n_attn * elems * 2048 * FP16);
    }

    #[test]
    fn recurrent_gemma_attention_layers_are_windowed() {
        let cfg = json!({
            "model_type": "recurrent_gemma",
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "block_types": ["recurrent", "recurrent", "attention",
                            "recurrent", "recurrent", "attention"],
            "attention_window_size": 2048,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::Hybrid);
        let n_attn = 2;
        let elems = 2 * 256; // 2 (K+V) × 1 kv_head × 256 head_dim
        // Windowed at 2048, not 8192.
        assert_eq!(e.total_bytes, n_attn * elems * 2048 * FP16);
        // Windowed attention contributes nothing to the steady-state margin.
        assert_eq!(e.marginal_bytes_per_token, 0);
    }

    #[test]
    fn qwen3_next_interval_with_mlp_only() {
        // full_attention_interval = 4 → attention at i where (i+1)%4==0:
        // i = 3, 7 of 8 layers; mlp_only excludes i=7.
        let cfg = json!({
            "model_type": "qwen3_next",
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "full_attention_interval": 4,
            "mlp_only_layers": [7],
        });
        let e = est(&cfg, 4096);
        assert_eq!(e.kind, KvArchKind::Hybrid);
        let n_attn = 1; // i=3 only (7 excluded)
        let elems = 2 * 2 * 128;
        assert_eq!(e.total_bytes, n_attn * elems * 4096 * FP16);
    }

    #[test]
    fn pure_ssm_has_zero_kv() {
        for mt in ["mamba", "mamba2"] {
            let cfg = json!({
                "model_type": mt,
                "num_hidden_layers": 48,
                "hidden_size": 2048,
                "state_size": 16,
            });
            let e = est(&cfg, 8192);
            assert_eq!(e.kind, KvArchKind::PureSsm);
            assert_eq!(e.total_bytes, 0);
            assert_eq!(e.marginal_bytes_per_token, 0);
        }
    }

    #[test]
    fn int8_halves_every_arch() {
        let cfg = json!({
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "sliding_window": 1024,
            "sliding_window_pattern": 4,
        });
        let fp16 = estimate_kv_arch_from_config(&cfg, 8192, false, 1).unwrap();
        let int8 = estimate_kv_arch_from_config(&cfg, 8192, true, 1).unwrap();
        assert_eq!(int8.total_bytes * 2, fp16.total_bytes);
    }

    #[test]
    fn batch_scales_linearly() {
        let cfg = json!({
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
        });
        let b1 = estimate_kv_arch_from_config(&cfg, 4096, false, 1).unwrap();
        let b4 = estimate_kv_arch_from_config(&cfg, 4096, false, 4).unwrap();
        assert_eq!(b4.total_bytes, b1.total_bytes * 4);
    }

    #[test]
    fn vlm_text_config_is_used() {
        let cfg = json!({
            "model_type": "some_vlm",
            "text_config": {
                "model_type": "llama",
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64,
            }
        });
        let e = est(&cfg, 1024);
        assert_eq!(e.kind, KvArchKind::Standard);
        assert_eq!(e.total_bytes, 4 * 2 * 2 * 64 * 1024 * FP16);
    }

    #[test]
    fn missing_architecture_fields_returns_none() {
        let cfg = json!({ "vocab_size": 32000 });
        assert!(estimate_kv_arch_from_config(&cfg, 4096, false, 1).is_none());
    }

    // ── OpenAI-era field naming: GPT-2 / GPT-BigCode (#927) ──────────────────

    /// `models/gpt2` verbatim: `n_layer` / `n_embd` / `n_head`, no
    /// `multi_query`, so plain MHA at head_dim 64.
    #[test]
    fn gpt2_n_layer_and_n_embd_classify_as_standard_mha() {
        let cfg = json!({
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size": 50257,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::Standard);
        // 12 layers × 2 (K+V) × 12 kv heads × 64 head_dim × 8192 × 2 bytes.
        assert_eq!(e.total_bytes, 12 * 2 * 12 * 64 * 8192 * FP16);
        assert_eq!(e.total_bytes, 301_989_888);
        assert_eq!(e.marginal_bytes_per_token, 36_864);
    }

    /// `models/gpt_bigcode-santacoder` verbatim. The boolean `multi_query`
    /// means ONE shared kv head; falling through to the `num_kv_heads =
    /// num_heads` MHA default would over-reserve by 16x (1.50 GiB).
    #[test]
    fn gpt_bigcode_multi_query_is_one_kv_head() {
        let cfg = json!({
            "model_type": "gpt_bigcode",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 2048,
            "n_inner": 8192,
            "multi_query": true,
            "vocab_size": 49280,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::Standard);
        // 24 layers × 2 (K+V) × 1 kv head × 128 head_dim × 8192 × 2 bytes.
        assert_eq!(e.total_bytes, 24 * 2 * 128 * 8192 * FP16);
        assert_eq!(e.total_bytes, 100_663_296);
        assert_eq!(e.marginal_bytes_per_token, 12_288);
        // Explicitly below the alias-only-fix figure, which is the trap.
        let mha_would_be = 24u64 * 2 * 16 * 128 * 8192 * FP16;
        assert_eq!(mha_would_be, 1_610_612_736);
        assert_eq!(e.total_bytes * 16, mha_would_be);
    }

    /// `multi_query: false` must degenerate to `n_head` kv heads, so the
    /// branch cannot silently shrink a non-MQA checkpoint.
    #[test]
    fn gpt_bigcode_multi_query_false_is_full_mha() {
        let cfg = json!({
            "model_type": "gpt_bigcode",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 2048,
            "multi_query": false,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.kind, KvArchKind::Standard);
        assert_eq!(e.total_bytes, 24 * 2 * 16 * 128 * 8192 * FP16);
    }

    /// A StarCoder-sized head count: the over-reservation the `multi_query`
    /// branch avoids scales with `n_head`, so it is 48x here, not 16x.
    #[test]
    fn gpt_bigcode_over_reservation_scales_with_n_head() {
        let cfg = json!({
            "model_type": "gpt_bigcode",
            "n_layer": 40,
            "n_head": 48,
            "n_embd": 6144,
            "multi_query": true,
        });
        let e = est(&cfg, 8192);
        assert_eq!(e.total_bytes, 40 * 2 * 128 * 8192 * FP16);
        let mha_would_be = 40u64 * 2 * 48 * 128 * 8192 * FP16;
        assert_eq!(e.total_bytes * 48, mha_would_be);
    }

    /// An explicit numeric kv-head field outranks the boolean, so a config
    /// carrying both is read the way it states.
    #[test]
    fn numeric_kv_heads_outrank_multi_query_flag() {
        let cfg = json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "multi_query": true,
        });
        let e = est(&cfg, 1024);
        assert_eq!(e.total_bytes, 4 * 2 * 8 * 128 * 1024 * FP16);
    }

    /// The three families verified on real checkpoints in #927 must keep
    /// byte-identical estimates: appending aliases must not move them.
    #[test]
    fn modern_naming_families_are_unchanged() {
        // models/pythia-1b (gpt_neox).
        let pythia = json!({
            "model_type": "gpt_neox",
            "num_hidden_layers": 16,
            "hidden_size": 2048,
            "num_attention_heads": 8,
        });
        assert_eq!(est(&pythia, 8192).total_bytes, 1_073_741_824);

        // models/helium-1-preview-2b-4bit.
        let helium = json!({
            "model_type": "helium",
            "num_hidden_layers": 24,
            "hidden_size": 2560,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "head_dim": 128,
        });
        assert_eq!(est(&helium, 8192).total_bytes, 2_013_265_920);

        // models/ling-lite-1.5 (bailing_moe): the sparse block changes only
        // the FFN, so the attention stack is ordinary GQA.
        let ling = json!({
            "model_type": "bailing_moe",
            "num_hidden_layers": 28,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
        });
        assert_eq!(est(&ling, 8192).total_bytes, 469_762_048);
    }

    #[test]
    fn ctx_rounds_up_to_256() {
        let cfg = json!({
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
        });
        // ctx 100 rounds to 256.
        let e = est(&cfg, 100);
        // 1 layer × 2 (K+V) × 1 kv_head × 1 head_dim × 256 ctx × FP16.
        assert_eq!(e.total_bytes, 2 * 256 * FP16);
    }
}
