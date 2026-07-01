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

//! Rust-native StableHLO text emitter (issue #449 M3 Stage 2d, config.json-driven
//! emit). Ported verbatim from the issue #451 spike `spike/rust-emitter`, then
//! given a [`Config::from_json`](config::Config::from_json) reader so the engine
//! generates its prefill / decode / ragged-decode graphs from a model's
//! `config.json` at load, instead of being pinned to the bundled Llama-3.2-1B
//! `.mlir` assets.
//!
//! Scope: the Llama, Qwen2, and Gemma2 architectures. The `Config` is
//! parameterized by dimensions, so any checkpoint of a supported architecture (any
//! size) emits correctly. The architecture switches the emitter branches on are
//! the RoPE kind (llama3 scaling for Llama, plain for Qwen2 / Gemma2), whether the
//! q/k/v projections carry a bias (Qwen2), whether the LM head is tied or a
//! separate `lm_head.weight` (untied, e.g. Llama-3.1-8B), and the `gemma2` switch
//! (embedding scale, `(1+w)` RMSNorm, GeGLU, a four-norm layer, attention / final
//! logit soft-cap, non-square `o_proj`). Gemma2 is single-sequence only so far;
//! the batched / ragged serve graphs are a follow-up.
//!
//! Pure Rust (no IREE), so it compiles and is unit-tested without the `iree`
//! feature; only the IREE engine consumes it. The bundled `.mlir` assets remain
//! as the byte-exact regression fixtures the test below checks the emitter against
//! (the spike that generated them produces identical output, proven here).

mod builder;
mod config;
mod model;
mod rope;

pub(crate) use config::Config;
pub(crate) use model::{emit_decode, emit_decode_ragged, emit_prefill};

#[cfg(test)]
mod tests {
    use super::config::RopeScaling;
    use super::*;

    const CONFIG_JSON: &str = include_str!("../../assets/llama-3.2-1b/config.json");
    const PREFILL: &str = include_str!("../../assets/llama-3.2-1b/prefill.mlir");
    const DECODE: &str = include_str!("../../assets/llama-3.2-1b/decode.mlir");
    const PREFILL_LOGITS: &str = include_str!("../../assets/llama-3.2-1b/prefill_logits.mlir");
    const RAGGED_B4: &str = include_str!("../../assets/llama-3.2-1b/decode_ragged_logits_b4.mlir");
    const RAGGED_B8: &str = include_str!("../../assets/llama-3.2-1b/decode_ragged_logits_b8.mlir");
    const QWEN_CONFIG_JSON: &str = include_str!("../../assets/qwen2.5-0.5b/config.json");

    fn occurs(haystack: &str, needle: &str) -> usize {
        haystack.matches(needle).count()
    }

    /// Arg count of an emitted module = the number of `loc("...")` markers, since
    /// the signature renders exactly one per func argument and the body carries
    /// none.
    fn arg_count(mlir: &str) -> usize {
        occurs(mlir, "loc(\"")
    }

    /// A tiny Qwen2-shaped config (plain RoPE + QKV bias) for the structural emit
    /// tests: small dims keep the emitted text tiny while exercising every
    /// bias-bearing path. `qkv_bias` is the only difference from the Llama-shaped
    /// counterpart, so a with/without diff isolates exactly the bias surface.
    fn qwen_like(qkv_bias: bool) -> Config {
        Config {
            hidden: 8,
            inter: 16,
            n_layers: 2,
            n_q: 2,
            n_kv: 1,
            head_dim: 4,
            eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab: 10,
            rope: RopeScaling::Plain,
            qkv_bias,
            tie_word_embeddings: true,
            quantization: None,
            gemma2: false,
            query_pre_attn_scalar: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
        }
    }

    /// A tiny Gemma2-shaped config (all the Gemma2 switches on: `(1+w)` norm,
    /// four per-layer norms, GeGLU, embedding scale, attention / final logit
    /// soft-cap, non-square `o_proj` since `n_q*head_dim = 12 != hidden = 8`) for
    /// the shared-attention-core coverage test. Small dims keep the emitted text
    /// tiny while exercising every Gemma2 path.
    fn gemma2_like() -> Config {
        Config {
            hidden: 8,
            inter: 16,
            n_layers: 2,
            n_q: 2,
            n_kv: 1,
            head_dim: 6,
            eps: 1e-6,
            rope_theta: 1e4,
            vocab: 10,
            rope: RopeScaling::Plain,
            qkv_bias: false,
            tie_word_embeddings: true,
            quantization: None,
            gemma2: true,
            query_pre_attn_scalar: Some(6.0),
            attn_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
        }
    }

    /// The whole point of Stage A: a `Config` parsed from the real
    /// Llama-3.2-1B-Instruct `config.json` emits every bundled graph
    /// byte-for-byte. This proves the load-time emit path reproduces the assets
    /// the engine shipped with, so switching from `include_str!` to emit-at-load
    /// cannot change the compiled graphs for this model.
    #[test]
    fn from_json_reproduces_bundled_assets_byte_for_byte() {
        let c = Config::from_json_str(CONFIG_JSON).expect("parse Llama-3.2-1B config.json");
        assert_eq!(emit_prefill(&c, true), PREFILL, "prefill.mlir (argmax)");
        assert_eq!(emit_decode(&c, true), DECODE, "decode.mlir (argmax)");
        assert_eq!(
            emit_prefill(&c, false),
            PREFILL_LOGITS,
            "prefill_logits.mlir"
        );
        assert_eq!(
            emit_decode_ragged(&c, 4, false),
            RAGGED_B4,
            "decode_ragged_logits_b4.mlir"
        );
        assert_eq!(
            emit_decode_ragged(&c, 8, false),
            RAGGED_B8,
            "decode_ragged_logits_b8.mlir"
        );
    }

    /// `from_json` reads the same values the spike hard-coded, so it emits
    /// identically to the in-code `llama_3_2_1b()` reference.
    #[test]
    fn from_json_matches_hardcoded_reference_config() {
        let from = Config::from_json_str(CONFIG_JSON).expect("parse");
        let hard = Config::llama_3_2_1b();
        assert_eq!(emit_prefill(&from, false), emit_prefill(&hard, false));
        assert_eq!(emit_decode(&from, true), emit_decode(&hard, true));
    }

    /// Stage B: the real Qwen2.5-0.5B `config.json` parses to the Qwen2
    /// architecture switches (plain RoPE + QKV bias) and the right dimensions.
    #[test]
    fn from_json_parses_qwen2_architecture() {
        let c = Config::from_json_str(QWEN_CONFIG_JSON).expect("parse Qwen2.5-0.5B config.json");
        assert!(c.qkv_bias, "Qwen2 carries a q/k/v projection bias");
        assert_eq!(c.rope, RopeScaling::Plain, "Qwen2.5-0.5B uses plain RoPE");
        assert_eq!(c.hidden, 896);
        assert_eq!(c.inter, 4864);
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.n_q, 14);
        assert_eq!(c.n_kv, 2);
        assert_eq!(c.head_dim, 64, "no explicit head_dim -> hidden / n_q");
        assert_eq!(c.vocab, 151936);
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert_eq!(c.eps, 1e-6);
    }

    /// A non-Llama/Qwen2/Gemma2 architecture and an unsupported `rope_scaling` are
    /// each rejected with a clear message rather than mis-emitted. (Untied
    /// embeddings and Gemma2 are no longer rejected; see
    /// `from_json_accepts_untied_embeddings` / `from_json_parses_gemma2`.)
    #[test]
    fn from_json_rejects_unsupported_configs() {
        let gemma3 = r#"{"model_type":"gemma3","tie_word_embeddings":true,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(gemma3)
                .unwrap_err()
                .contains("model_type")
        );

        let yarn = r#"{"model_type":"qwen2","tie_word_embeddings":true,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10,
            "rope_scaling":{"rope_type":"yarn","factor":4.0}}"#;
        assert!(
            Config::from_json_str(yarn)
                .unwrap_err()
                .contains("rope_type")
        );
    }

    /// Gemma2 parses to its architecture switches: the soft-caps, the query
    /// pre-attention scale, plain RoPE, and the `gemma2` structural flag. Uses the
    /// Gemma2-9B values, where `query_pre_attn_scalar` (224) differs from `head_dim`
    /// (256), so the scale must come from the former, not the latter.
    #[test]
    fn from_json_parses_gemma2() {
        let g = r#"{"model_type":"gemma2","hidden_size":3584,"num_attention_heads":16,
            "num_key_value_heads":8,"head_dim":256,"intermediate_size":14336,
            "num_hidden_layers":42,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":256000,
            "query_pre_attn_scalar":224,"attn_logit_softcapping":50.0,
            "final_logit_softcapping":30.0,"hidden_activation":"gelu_pytorch_tanh"}"#;
        let c = Config::from_json_str(g).expect("gemma2 parses");
        assert!(c.gemma2);
        assert_eq!(c.rope, RopeScaling::Plain);
        assert_eq!(c.query_pre_attn_scalar, Some(224.0));
        assert_eq!(c.attn_logit_softcap, Some(50.0));
        assert_eq!(c.final_logit_softcap, Some(30.0));
        assert_eq!(c.head_dim, 256, "explicit head_dim (!= hidden/n_q)");
        assert!(c.tie_word_embeddings, "Gemma2 ties embeddings by default");
        // The scale is query_pre_attn_scalar^-0.5 (224), NOT head_dim^-0.5 (256).
        assert_eq!(c.scale(), (224.0f64.powf(-0.5)) as f32);
        assert_ne!(c.scale(), (256.0f32).powf(-0.5), "must not use head_dim");
    }

    /// Untied embeddings are supported (issue #449 M3 Stage 2d): `from_json` reads
    /// `tie_word_embeddings = false`, and an absent field defaults to tied (the HF
    /// `PretrainedConfig` default).
    #[test]
    fn from_json_accepts_untied_embeddings() {
        let untied = r#"{"model_type":"qwen2","tie_word_embeddings":false,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        let c = Config::from_json_str(untied).expect("untied qwen2 parses");
        assert!(
            !c.tie_word_embeddings,
            "tie_word_embeddings=false -> untied"
        );

        let absent = r#"{"model_type":"qwen2","hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(absent)
                .expect("absent field parses")
                .tie_word_embeddings,
            "absent tie_word_embeddings defaults to tied"
        );
    }

    /// Turning on `qkv_bias` adds exactly the three q/k/v projection biases per
    /// layer to every graph kind, and the adds that consume them: the single-token
    /// decode adds one `stablehlo.add` per bias; the seq graphs (prefill / batched
    /// / ragged) add a broadcast + an add per bias. A with/without diff over an
    /// otherwise-identical config isolates exactly the bias surface, so nothing
    /// else in the graph shifted.
    #[test]
    fn qkv_bias_adds_three_biases_per_layer_to_every_graph() {
        let with = qwen_like(true);
        let without = qwen_like(false);
        let nl = with.n_layers;

        // single-token decode: +3*L args, +3*L adds, no extra broadcasts.
        let d_with = emit_decode(&with, false);
        let d_without = emit_decode(&without, false);
        assert_eq!(
            arg_count(&d_with) - arg_count(&d_without),
            3 * nl,
            "decode args"
        );
        assert_eq!(
            occurs(&d_with, "stablehlo.add ") - occurs(&d_without, "stablehlo.add "),
            3 * nl,
            "decode bias adds"
        );
        assert_eq!(
            occurs(&d_with, "broadcast_in_dim"),
            occurs(&d_without, "broadcast_in_dim"),
            "single-token bias is shape-matched, so no extra broadcast"
        );

        // seq graphs: +3*L args, +3*L adds, +3*L broadcasts.
        let seq = [
            (
                emit_prefill(&with, false),
                emit_prefill(&without, false),
                "prefill",
            ),
            (
                super::model::emit_decode_batched(&with, 4, false),
                super::model::emit_decode_batched(&without, 4, false),
                "batched",
            ),
            (
                emit_decode_ragged(&with, 4, false),
                emit_decode_ragged(&without, 4, false),
                "ragged",
            ),
        ];
        for (g_with, g_without, name) in seq {
            assert_eq!(
                arg_count(&g_with) - arg_count(&g_without),
                3 * nl,
                "{name} args"
            );
            assert_eq!(
                occurs(&g_with, "stablehlo.add ") - occurs(&g_without, "stablehlo.add "),
                3 * nl,
                "{name} bias adds"
            );
            assert_eq!(
                occurs(&g_with, "broadcast_in_dim") - occurs(&g_without, "broadcast_in_dim"),
                3 * nl,
                "{name} bias broadcasts"
            );
        }
    }

    /// The q/k/v bias args are appended after `wv` in k/q/v order for every layer,
    /// matching `weight_names` in `iree.rs` so the loaded weight buffers line up
    /// with the emitted graph args.
    #[test]
    fn qkv_bias_args_follow_wv_in_k_q_v_order() {
        let c = qwen_like(true);
        let mlir = emit_decode(&c, false);
        for li in 0..c.n_layers {
            let at = |k: &str| {
                mlir.find(&format!("params['layers'][{li}]['{k}']"))
                    .unwrap_or_else(|| panic!("layer {li} missing {k}"))
            };
            let (wv, bk, bq, bv) = (at("wv"), at("bk"), at("bq"), at("bv"));
            assert!(
                wv < bk && bk < bq && bq < bv,
                "layer {li}: expected wv < bk < bq < bv arg order"
            );
        }
    }

    /// The Llama path is bias-free: no bias args leak onto a `qkv_bias = false`
    /// architecture (the guard that keeps Llama byte-identical).
    #[test]
    fn llama_graph_has_no_qkv_bias_args() {
        let mlir = emit_decode(&Config::llama_3_2_1b(), true);
        assert!(!mlir.contains("['bq']"));
        assert!(!mlir.contains("['bk']"));
        assert!(!mlir.contains("['bv']"));
    }

    /// The shared attention core (issue #494) keeps a family's attention / MLP
    /// deltas reaching every graph kind from one authoring site. Gemma2's GeGLU
    /// and its attention / final soft-caps are the only source of `stablehlo.tanh`
    /// in these graphs, so a Gemma2 config emits `tanh` in single decode, ragged
    /// decode, and prefill alike, while a Llama config (SwiGLU + no soft-cap)
    /// emits none in any of them. This locks the "authored once, reaches all three
    /// paths" guarantee for the one in-scope architecture with no committed byte
    /// asset.
    #[test]
    fn gemma2_deltas_reach_every_shared_core_kind() {
        let gemma = gemma2_like();
        let llama = Config::llama_3_2_1b();
        for (c, want_tanh) in [(&gemma, true), (&llama, false)] {
            let kinds = [
                ("decode", emit_decode(c, false)),
                ("ragged", emit_decode_ragged(c, 4, false)),
                ("prefill", emit_prefill(c, false)),
            ];
            for (name, g) in kinds {
                assert_eq!(
                    g.contains("stablehlo.tanh"),
                    want_tanh,
                    "{name}: gemma2={want_tanh} tanh presence"
                );
            }
        }
    }

    /// Untied embeddings add exactly one weight arg — the `[V, H]`
    /// `params['lm_head']` — to every graph kind, positioned right after
    /// `final_norm` and before the layers (arg 2), and the final projection
    /// consumes it. A `tie_word_embeddings = true` config emits no such arg, so the
    /// arg counts differ by exactly one and a tied graph never names `lm_head` (the
    /// guard that keeps every shipped tied checkpoint byte-identical). Mirrors
    /// `weight_names` in `iree.rs`, which adds `lm_head.weight` in the same slot.
    #[test]
    fn untied_adds_one_lm_head_arg_after_final_norm() {
        let tied = qwen_like(true);
        let mut untied = tied.clone();
        untied.tie_word_embeddings = false;

        let graphs = [
            (
                emit_decode(&untied, false),
                emit_decode(&tied, false),
                "decode",
            ),
            (
                emit_prefill(&untied, false),
                emit_prefill(&tied, false),
                "prefill",
            ),
            (
                super::model::emit_decode_batched(&untied, 4, false),
                super::model::emit_decode_batched(&tied, 4, false),
                "batched",
            ),
            (
                emit_decode_ragged(&untied, 4, false),
                emit_decode_ragged(&tied, 4, false),
                "ragged",
            ),
        ];
        for (g_untied, g_tied, name) in graphs {
            assert_eq!(
                arg_count(&g_untied) - arg_count(&g_tied),
                1,
                "{name}: untied adds exactly the lm_head arg"
            );
            assert_eq!(
                occurs(&g_untied, "params['lm_head']"),
                1,
                "{name}: lm_head declared exactly once (signature only)"
            );
            assert_eq!(
                occurs(&g_tied, "params['lm_head']"),
                0,
                "{name}: tied graph never names lm_head"
            );
            // arg order: final_norm (arg 1) < lm_head (arg 2) < layer 0's weights.
            let fnorm = g_untied.find("params['final_norm']").unwrap();
            let lm = g_untied.find("params['lm_head']").unwrap();
            let l0 = g_untied.find("params['layers'][0]").unwrap();
            assert!(
                fnorm < lm && lm < l0,
                "{name}: expected final_norm < lm_head < layer0 arg order"
            );
        }
    }

    /// Plain RoPE base frequencies are the textbook `1 / theta^(2i/head_dim)`
    /// (Qwen2), distinct from the llama3-scaled table.
    #[test]
    fn plain_rope_inv_freq_is_theta_power_series() {
        let c = qwen_like(true); // theta 1e6, head_dim 4 -> half = 2
        let inv = super::rope::inv_freq(&c);
        assert_eq!(inv.len(), 2);
        assert!((inv[0] - 1.0).abs() < 1e-12, "i=0 -> theta^0 = 1");
        let theta = 1_000_000.0f64;
        assert!(
            (inv[1] - 1.0 / theta.powf(2.0 / 4.0)).abs() < 1e-12,
            "i=1 -> theta^(-1/2)"
        );
    }
}
