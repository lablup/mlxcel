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

//! Checkpoint tensor names in the emitter's weight arg order (issue #449 M3 Stage
//! 2d; generalized to per-architecture naming schemes in #499).
//!
//! The IREE loader ([`iree`](crate::iree)) reads each weight the emitted graph
//! takes as an argument, in the emitter's exact arg order, from the checkpoint's
//! safetensors. [`weight_names`] produces that ordered name list from the model
//! [`Config`]: embed, final_norm, then — for an untied checkpoint
//! (`tie_word_embeddings = false`) — the LM head, then per layer the nine core
//! tensors (down, gate, in_ln, post_ln, up, wk, wo, wq, wv), then — for a
//! `qkv_bias` architecture — the k/q/v projection biases, then — for Gemma2 — the
//! two extra feed-forward norms. The order matches `take_lm_head` /
//! `take_layer_weights` in `emitter/model.rs` so the loaded buffers line up with
//! the graph args.
//!
//! Only the *names* vary by [`WeightScheme`] (issue #499): almost every checkpoint
//! uses the standard HF Llama layout, while ExaOne 3.x keeps GPT-2-style names.
//! The scheme never changes the emitted graph, so a family that differs only in
//! naming reuses the proven Llama / Qwen2 forward and its structural goldens
//! unchanged; this module is the one place the naming delta lives.
//!
//! Pure Rust (no IREE), so the ordering is unit-tested without the `iree` feature
//! (the loader that consumes it is `iree`-gated).

use crate::emitter::{Config, WeightScheme};

/// The tensor-name pieces for one [`WeightScheme`]: the top-level embed /
/// final-norm / (untied) LM-head names, the per-layer prefix stem
/// (`"{stem}{i}."`), and the per-layer suffixes in the emitter's arg order.
struct SchemeNames {
    /// Token-embedding weight name.
    embed: &'static str,
    /// Final RMSNorm weight name.
    final_norm: &'static str,
    /// Untied LM-head weight name (used only when `tie_word_embeddings = false`).
    lm_head: &'static str,
    /// Per-layer prefix stem; the full prefix is `format!("{stem}{i}.")`.
    layer_stem: &'static str,
    /// The nine core per-layer suffixes IN THE EMITTER'S ARG ORDER: down, gate,
    /// in_ln, post_ln, up, wk, wo, wq, wv.
    core: [&'static str; 9],
    /// The q/k/v projection bias suffixes (k, q, v order, matching
    /// `take_layer_weights`); used only for a `qkv_bias` architecture.
    qkv_bias: [&'static str; 3],
    /// The Gemma2 pre / post feed-forward norm suffixes; used only for `gemma2`.
    gemma2_norms: [&'static str; 2],
}

/// The name pieces for `scheme`. `Llama` reproduces the standard HF layout the
/// loader has always used (byte-for-byte, so Llama / Qwen2 / Gemma2 loading is
/// unchanged); `Exaone` is ExaOne 3.x's GPT-2-style layout, verified against the
/// checkpoint's `modeling_exaone.py` (gated MLP `c_proj(act(c_fc_0(x)) *
/// c_fc_1(x))`, so `c_fc_0` is the gate and `c_fc_1` the up projection; attention
/// under `attn.attention.*` with `out_proj` as o_proj).
fn scheme_names(scheme: WeightScheme) -> SchemeNames {
    match scheme {
        WeightScheme::Llama => SchemeNames {
            embed: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_stem: "model.layers.",
            core: [
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.up_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
            ],
            qkv_bias: [
                "self_attn.k_proj.bias",
                "self_attn.q_proj.bias",
                "self_attn.v_proj.bias",
            ],
            gemma2_norms: [
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ],
        },
        WeightScheme::Exaone => SchemeNames {
            embed: "transformer.wte.weight",
            final_norm: "transformer.ln_f.weight",
            lm_head: "lm_head.weight",
            layer_stem: "transformer.h.",
            core: [
                "mlp.c_proj.weight",              // down
                "mlp.c_fc_0.weight",              // gate (activation input)
                "ln_1.weight",                    // in_ln
                "ln_2.weight",                    // post_ln
                "mlp.c_fc_1.weight",              // up
                "attn.attention.k_proj.weight",   // wk
                "attn.attention.out_proj.weight", // wo
                "attn.attention.q_proj.weight",   // wq
                "attn.attention.v_proj.weight",   // wv
            ],
            // ExaOne 3.x carries neither q/k/v biases nor feed-forward norms
            // (qkv_bias / gemma2 are false for it), so these are never emitted;
            // they keep the mapping total. The bias names mirror the attention path.
            qkv_bias: [
                "attn.attention.k_proj.bias",
                "attn.attention.q_proj.bias",
                "attn.attention.v_proj.bias",
            ],
            gemma2_norms: [
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ],
        },
    }
}

/// The checkpoint tensor names in the emitter's exact arg order for `cfg`.
///
/// Order: embed, final_norm, then (untied) the LM head, then per layer the nine
/// core tensors, then (qkv_bias) the k/q/v biases, then (gemma2) the two extra
/// feed-forward norms — matching `take_lm_head` / `take_layer_weights` in
/// `emitter/model.rs`. The names follow `cfg.weight_scheme`.
pub(crate) fn weight_names(cfg: &Config) -> Vec<String> {
    let s = scheme_names(cfg.weight_scheme);
    let mut names = vec![s.embed.to_string(), s.final_norm.to_string()];
    // Untied LM head: a separate head weight follows `final_norm`, matching the
    // `params['lm_head']` arg the emitter takes in the same position.
    if !cfg.tie_word_embeddings {
        names.push(s.lm_head.to_string());
    }
    for i in 0..cfg.n_layers {
        let p = format!("{}{i}.", s.layer_stem);
        for suf in s.core {
            names.push(format!("{p}{suf}"));
        }
        // q/k/v projection biases, appended per layer in the same k/q/v order
        // `take_layer_weights` adds them to the emitted graph args.
        if cfg.qkv_bias {
            for suf in s.qkv_bias {
                names.push(format!("{p}{suf}"));
            }
        }
        // Gemma2's two extra per-layer norms (pre/post feed-forward), appended in
        // the same order `take_layer_weights` takes their graph args.
        if cfg.gemma2 {
            for suf in s.gemma2_norms {
                names.push(format!("{p}{suf}"));
            }
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::Config;

    /// A minimal Llama-scheme config; the fields the namer reads are `n_layers`,
    /// `tie_word_embeddings`, `qkv_bias`, `gemma2`, `weight_scheme`.
    fn cfg(
        n_layers: usize,
        tie: bool,
        qkv_bias: bool,
        gemma2: bool,
        scheme: WeightScheme,
    ) -> Config {
        let mut c = Config::llama_3_2_1b();
        c.n_layers = n_layers;
        c.tie_word_embeddings = tie;
        c.qkv_bias = qkv_bias;
        c.gemma2 = gemma2;
        c.weight_scheme = scheme;
        c
    }

    /// The Llama scheme reproduces the exact standard HF names the loader has
    /// always used, so Llama / Qwen2 / Gemma2 loading is byte-for-byte unchanged.
    #[test]
    fn llama_scheme_matches_the_standard_hf_layout() {
        let names = weight_names(&cfg(1, true, false, false, WeightScheme::Llama));
        assert_eq!(
            names,
            vec![
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ]
        );
    }

    /// Untied adds exactly the LM head, right after `final_norm`, before the layers.
    #[test]
    fn untied_inserts_lm_head_after_final_norm() {
        let names = weight_names(&cfg(1, false, false, false, WeightScheme::Llama));
        assert_eq!(names[2], "lm_head.weight");
        let fnorm = names.iter().position(|n| n == "model.norm.weight").unwrap();
        let lm = names.iter().position(|n| n == "lm_head.weight").unwrap();
        let l0 = names
            .iter()
            .position(|n| n.starts_with("model.layers.0."))
            .unwrap();
        assert!(fnorm < lm && lm < l0, "final_norm < lm_head < layer0");
    }

    /// A `qkv_bias` arch (Seed-OSS / MiMo / Qwen2) appends the three biases per
    /// layer in k/q/v order, after the nine core tensors.
    #[test]
    fn qkv_bias_appends_three_biases_in_k_q_v_order() {
        let names = weight_names(&cfg(1, false, true, false, WeightScheme::Llama));
        let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
        let (v, bk, bq, bv) = (
            idx("model.layers.0.self_attn.v_proj.weight"),
            idx("model.layers.0.self_attn.k_proj.bias"),
            idx("model.layers.0.self_attn.q_proj.bias"),
            idx("model.layers.0.self_attn.v_proj.bias"),
        );
        assert!(v < bk && bk < bq && bq < bv, "wv < bk < bq < bv");
    }

    /// The ExaOne scheme maps the emitter's arg order onto the GPT-2-style names,
    /// with the critical gated-MLP mapping (down←c_proj, gate←c_fc_0, up←c_fc_1)
    /// and `out_proj` as the attention output; it is tied (no LM head).
    #[test]
    fn exaone_scheme_maps_gpt2_style_names() {
        let names = weight_names(&cfg(1, true, false, false, WeightScheme::Exaone));
        assert_eq!(
            names,
            vec![
                "transformer.wte.weight",
                "transformer.ln_f.weight",
                "transformer.h.0.mlp.c_proj.weight", // down
                "transformer.h.0.mlp.c_fc_0.weight", // gate
                "transformer.h.0.ln_1.weight",       // in_ln
                "transformer.h.0.ln_2.weight",       // post_ln
                "transformer.h.0.mlp.c_fc_1.weight", // up
                "transformer.h.0.attn.attention.k_proj.weight", // wk
                "transformer.h.0.attn.attention.out_proj.weight", // wo
                "transformer.h.0.attn.attention.q_proj.weight", // wq
                "transformer.h.0.attn.attention.v_proj.weight", // wv
            ]
        );
    }

    /// The per-layer name count is stable: 9 core (+3 biases when qkv_bias, +2
    /// norms when gemma2), for every scheme.
    #[test]
    fn per_layer_counts_scale_with_the_deltas() {
        for scheme in [WeightScheme::Llama, WeightScheme::Exaone] {
            let base = weight_names(&cfg(2, true, false, false, scheme)).len();
            assert_eq!(base, 2 + 2 * 9, "embed+norm + 2 layers * 9 core");
            let biased = weight_names(&cfg(2, true, true, false, scheme)).len();
            assert_eq!(biased - base, 2 * 3, "qkv_bias adds 3 per layer");
        }
    }
}
