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
//! 2d; generalized to per-architecture naming schemes in #499, and to the qk-norm /
//! Gemma dense pack in #497).
//!
//! The IREE loader ([`iree`](crate::iree)) reads each weight the emitted graph
//! takes as an argument, in the emitter's exact arg order, from the checkpoint's
//! safetensors. [`weight_names`] produces that ordered name list from the model
//! [`Config`], mirroring `take_lm_head` / `take_layer_weights` in `emitter/model.rs`
//! exactly: embed, final_norm, then — for an untied checkpoint
//! (`tie_word_embeddings = false`) — the LM head, then per layer down, gate,
//! `input_layernorm` (unless the OLMo reordered post-norm drops it),
//! post_attention_layernorm, up, wk, wo, wq, wv, then the arch-conditional extras in
//! the emitter's order: the k/q/v projection biases (`qkv_bias`), the q/k norms
//! (`qk_norm`), and the feed-forward norms (`pre_feedforward_layernorm` for Gemma
//! 2/3, `post_feedforward_layernorm` for Gemma 2/3 and OLMo 2/3). Llama / Qwen2 /
//! Gemma2 have none of the new extras, so their name lists are byte-for-byte
//! unchanged.
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
/// (`"{stem}{i}."`), and each per-layer suffix, emitted in the emitter's arg order
/// by [`weight_names`] under the same config gates as `take_layer_weights`.
struct SchemeNames {
    /// Token-embedding weight name.
    embed: &'static str,
    /// Final RMSNorm weight name.
    final_norm: &'static str,
    /// Untied LM-head weight name (used only when `tie_word_embeddings = false`).
    lm_head: &'static str,
    /// Per-layer prefix stem; the full prefix is `format!("{stem}{i}.")`.
    layer_stem: &'static str,
    /// MLP down projection.
    down: &'static str,
    /// MLP gate projection (the activation input).
    gate: &'static str,
    /// `input_layernorm` (skipped for the OLMo reordered post-norm).
    input_layernorm: &'static str,
    /// `post_attention_layernorm`.
    post_attention_layernorm: &'static str,
    /// MLP up projection.
    up: &'static str,
    /// Attention key projection.
    k_proj: &'static str,
    /// Attention output projection.
    o_proj: &'static str,
    /// Attention query projection.
    q_proj: &'static str,
    /// Attention value projection.
    v_proj: &'static str,
    /// Attention key projection bias (used only for a `qkv_bias` arch).
    k_bias: &'static str,
    /// Attention query projection bias (used only for a `qkv_bias` arch).
    q_bias: &'static str,
    /// Attention value projection bias (used only for a `qkv_bias` arch).
    v_bias: &'static str,
    /// Query RMSNorm weight (used only for a `qk_norm` arch).
    q_norm: &'static str,
    /// Key RMSNorm weight (used only for a `qk_norm` arch).
    k_norm: &'static str,
    /// Pre-feed-forward norm (used only for the Gemma four-norm layer).
    pre_ff_norm: &'static str,
    /// Post-feed-forward norm (used only for Gemma 2/3 and OLMo 2/3).
    post_ff_norm: &'static str,
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
            down: "mlp.down_proj.weight",
            gate: "mlp.gate_proj.weight",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_attention_layernorm.weight",
            up: "mlp.up_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            o_proj: "self_attn.o_proj.weight",
            q_proj: "self_attn.q_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            k_bias: "self_attn.k_proj.bias",
            q_bias: "self_attn.q_proj.bias",
            v_bias: "self_attn.v_proj.bias",
            q_norm: "self_attn.q_norm.weight",
            k_norm: "self_attn.k_norm.weight",
            pre_ff_norm: "pre_feedforward_layernorm.weight",
            post_ff_norm: "post_feedforward_layernorm.weight",
        },
        WeightScheme::Exaone => SchemeNames {
            embed: "transformer.wte.weight",
            final_norm: "transformer.ln_f.weight",
            lm_head: "lm_head.weight",
            layer_stem: "transformer.h.",
            down: "mlp.c_proj.weight",               // down
            gate: "mlp.c_fc_0.weight",               // gate (activation input)
            input_layernorm: "ln_1.weight",          // in_ln
            post_attention_layernorm: "ln_2.weight", // post_ln
            up: "mlp.c_fc_1.weight",                 // up
            k_proj: "attn.attention.k_proj.weight",
            o_proj: "attn.attention.out_proj.weight",
            q_proj: "attn.attention.q_proj.weight",
            v_proj: "attn.attention.v_proj.weight",
            // ExaOne 3.x carries none of the conditional extras (qkv_bias, qk_norm,
            // and the feed-forward norms are all off for it), so the names below are
            // never emitted; they keep the scheme mapping total and mirror the
            // attention path where one exists.
            k_bias: "attn.attention.k_proj.bias",
            q_bias: "attn.attention.q_proj.bias",
            v_bias: "attn.attention.v_proj.bias",
            q_norm: "attn.attention.q_norm.weight",
            k_norm: "attn.attention.k_norm.weight",
            pre_ff_norm: "pre_feedforward_layernorm.weight",
            post_ff_norm: "post_feedforward_layernorm.weight",
        },
    }
}

/// The checkpoint tensor names in the emitter's exact arg order for `cfg`, matching
/// `take_lm_head` / `take_layer_weights` in `emitter/model.rs`. The names follow
/// `cfg.weight_scheme`, and every per-layer knob (the untied head, the conditional
/// `input_layernorm`, the q/k/v biases, the q/k norms, the feed-forward norms) is
/// read from `cfg`, so Llama / Qwen2 / Gemma2 stay byte-for-byte unchanged.
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
        let mut push = |suf: &str| names.push(format!("{p}{suf}"));
        push(s.down);
        push(s.gate);
        // input_layernorm: present unless the reordered (OLMo) post-norm drops it.
        if cfg.has_input_norm() {
            push(s.input_layernorm);
        }
        push(s.post_attention_layernorm);
        push(s.up);
        push(s.k_proj);
        push(s.o_proj);
        push(s.q_proj);
        push(s.v_proj);
        // q/k/v projection biases (Qwen2 / Seed-OSS / MiMo / InternLM3), in the same
        // k/q/v order `take_layer_weights` adds them to the emitted graph args.
        if cfg.qkv_bias {
            push(s.k_bias);
            push(s.q_bias);
            push(s.v_bias);
        }
        // q/k norms (Qwen3 / Gemma3 per-head, OLMo2/3 flat), q then k.
        if cfg.qk_norm.is_some() {
            push(s.q_norm);
            push(s.k_norm);
        }
        // Feed-forward norms: Gemma2/3 add pre AND post; OLMo2/3 add post only.
        if cfg.has_pre_ff_norm() {
            push(s.pre_ff_norm);
        }
        if cfg.has_post_ff_norm() {
            push(s.post_ff_norm);
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::{NormStyle, QkNorm};

    /// A minimal config for the namer. Starts from the Llama-3.2-1B defaults (Plain
    /// norm, no bias, no q/k norm) and overrides the fields `weight_names` reads:
    /// `n_layers`, `tie_word_embeddings`, `qkv_bias`, `weight_scheme`. Tests that
    /// exercise the norm placement / q-k norm set `norm_style` / `qk_norm` directly.
    fn cfg(n_layers: usize, tie: bool, qkv_bias: bool, scheme: WeightScheme) -> Config {
        let mut c = Config::llama_3_2_1b();
        c.n_layers = n_layers;
        c.tie_word_embeddings = tie;
        c.qkv_bias = qkv_bias;
        c.weight_scheme = scheme;
        c
    }

    /// The Llama scheme reproduces the exact standard HF names the loader has
    /// always used, so Llama / Qwen2 / Gemma2 loading is byte-for-byte unchanged.
    #[test]
    fn llama_scheme_matches_the_standard_hf_layout() {
        let names = weight_names(&cfg(1, true, false, WeightScheme::Llama));
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
        let names = weight_names(&cfg(1, false, false, WeightScheme::Llama));
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
        let names = weight_names(&cfg(1, false, true, WeightScheme::Llama));
        let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
        let (v, bk, bq, bv) = (
            idx("model.layers.0.self_attn.v_proj.weight"),
            idx("model.layers.0.self_attn.k_proj.bias"),
            idx("model.layers.0.self_attn.q_proj.bias"),
            idx("model.layers.0.self_attn.v_proj.bias"),
        );
        assert!(v < bk && bk < bq && bq < bv, "wv < bk < bq < bv");
    }

    /// A `qk_norm` arch (Qwen3 / Gemma3 / OLMo2/3) appends q_norm then k_norm per
    /// layer, after the projections and after any biases.
    #[test]
    fn qk_norm_appends_q_then_k_norm_after_projections() {
        let mut c = cfg(1, true, false, WeightScheme::Llama);
        c.qk_norm = Some(QkNorm {
            per_head: true,
            one_plus: false,
        });
        let names = weight_names(&c);
        let idx = |n: &str| names.iter().position(|x| x == n).unwrap();
        let (v, qn, kn) = (
            idx("model.layers.0.self_attn.v_proj.weight"),
            idx("model.layers.0.self_attn.q_norm.weight"),
            idx("model.layers.0.self_attn.k_norm.weight"),
        );
        assert!(v < qn && qn < kn, "wv < q_norm < k_norm");
    }

    /// The Gemma four-norm layer keeps `input_layernorm` and adds the pre AND post
    /// feed-forward norms (pre before post), matching `take_layer_weights`.
    #[test]
    fn gemma_ff_adds_pre_then_post_feedforward_norm() {
        let mut c = cfg(1, true, false, WeightScheme::Llama);
        c.norm_style = NormStyle::GemmaFf;
        let names = weight_names(&c);
        assert!(
            names.iter().any(|n| n.ends_with("input_layernorm.weight")),
            "Gemma keeps input_layernorm"
        );
        let idx = |suf: &str| names.iter().position(|x| x.ends_with(suf)).unwrap();
        let pre = idx("pre_feedforward_layernorm.weight");
        let post = idx("post_feedforward_layernorm.weight");
        assert!(
            pre < post,
            "pre-feedforward norm before post-feedforward norm"
        );
    }

    /// The OLMo reordered post-norm drops `input_layernorm` and adds only the
    /// `post_feedforward_layernorm` (no pre), while keeping the q/k norm.
    #[test]
    fn olmo_post_norm_drops_input_norm_and_pre_ff() {
        let mut c = cfg(1, false, false, WeightScheme::Llama);
        c.norm_style = NormStyle::OlmoPost;
        c.qk_norm = Some(QkNorm {
            per_head: false,
            one_plus: false,
        });
        let names = weight_names(&c);
        assert!(
            !names.iter().any(|n| n.ends_with("input_layernorm.weight")),
            "OLMo post-norm has no input_layernorm"
        );
        assert!(
            names
                .iter()
                .any(|n| n.ends_with("post_feedforward_layernorm.weight")),
            "OLMo post-norm keeps the post-feedforward norm"
        );
        assert!(
            !names
                .iter()
                .any(|n| n.ends_with("pre_feedforward_layernorm.weight")),
            "OLMo post-norm has no pre-feedforward norm"
        );
        assert!(
            names.iter().any(|n| n.ends_with("q_norm.weight")),
            "OLMo keeps the flat q/k norm"
        );
    }

    /// The ExaOne scheme maps the emitter's arg order onto the GPT-2-style names,
    /// with the critical gated-MLP mapping (down←c_proj, gate←c_fc_0, up←c_fc_1)
    /// and `out_proj` as the attention output; it is tied (no LM head).
    #[test]
    fn exaone_scheme_maps_gpt2_style_names() {
        let names = weight_names(&cfg(1, true, false, WeightScheme::Exaone));
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

    /// The per-layer name count is stable: 9 core (+3 biases when qkv_bias), for
    /// every scheme.
    #[test]
    fn per_layer_counts_scale_with_the_deltas() {
        for scheme in [WeightScheme::Llama, WeightScheme::Exaone] {
            let base = weight_names(&cfg(2, true, false, scheme)).len();
            assert_eq!(base, 2 + 2 * 9, "embed+norm + 2 layers * 9 core");
            let biased = weight_names(&cfg(2, true, true, scheme)).len();
            assert_eq!(biased - base, 2 * 3, "qkv_bias adds 3 per layer");
        }
    }
}
