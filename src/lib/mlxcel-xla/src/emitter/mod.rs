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
//! Scope: the dense families Llama, Qwen2, Qwen3, Gemma1/2/3, SmolLM3, OLMo2/3,
//! Seed-OSS, MiMo, InternLM3, ExaOne (issues #497 / #499), and the parallel-block
//! and norm-variant pack Cohere/Cohere2, Phi3, StableLM, StarCoder2, Granite,
//! MiniCPM (issue #498). The `Config` is parameterized by dimensions AND by
//! orthogonal architecture flags, so any checkpoint of a supported family (any
//! size) emits correctly and a new family is a flag combination rather than a new
//! code path. The switches the emitter branches on are the RoPE kind (llama3 vs
//! plain, an optional per-layer local base for Gemma3, and an interleaved / partial
//! variant for Cohere / StableLM), the q/k/v / o_proj / MLP biases, the LM-head tie,
//! MLX quantization, the Gemma embedding scale / `(1+w)` RMSNorm / GeGLU MLP, the
//! per-layer norm placement ([`NormStyle`](config::NormStyle): pre-norm, Gemma
//! four-norm, or OLMo reordered post-norm) plus the mean-subtract LayerNorm (with an
//! optional bias) for Cohere / StableLM / StarCoder2, an optional q/k normalization
//! ([`QkNorm`](config::QkNorm): per-head for Qwen3 / Gemma3, flat for OLMo2/3), the
//! sliding-window schedule, Gemma2/3 soft-caps, a per-layer NoPE mask (SmolLM3), the
//! block structure (sequential or Cohere's parallel attention+MLP), the dense
//! (StarCoder2) MLP, and the Granite / MiniCPM / Cohere scalar multipliers. The
//! single-token decode, ragged (continuous-batching) decode, and prefill graphs
//! share one per-layer core (issues #494 / #498), so a family is authored once and
//! reaches all three. MiniCPM3 (MLA attention) is rejected pending a follow-up.
//!
//! Pure Rust (no IREE), so it compiles and is unit-tested without the `iree`
//! feature; only the IREE engine consumes it. The bundled `.mlir` assets remain
//! as the byte-exact regression fixtures the test below checks the emitter against
//! (the spike that generated them produces identical output, proven here).

mod builder;
mod config;
mod model;
mod rope;

pub(crate) use config::{Config, NormStyle, QkNorm, WeightScheme};
// `resolve_precision` and the precision-taking `*_with` emit variants are consumed
// by the IREE execution path; the f32-default `emit_*` wrappers by the byte-exact
// regression tests. Which set is live depends on the build cfg (`iree` vs `test`),
// so allow the re-export to be unused in the other. `emit_decode_batched` (the
// superseded uniform-B Stage-1 graph) is re-exported for the validation harness.
#[allow(unused_imports)]
pub(crate) use builder::resolve_precision;
#[allow(unused_imports)]
pub(crate) use model::{
    emit_decode, emit_decode_batched, emit_decode_ragged, emit_decode_ragged_with,
    emit_decode_with, emit_prefill, emit_prefill_with,
};

#[cfg(test)]
mod tests {
    use super::config::{NormStyle, QkNorm, RopeScaling};
    use super::*;

    const CONFIG_JSON: &str = include_str!("../../assets/llama-3.2-1b/config.json");
    const QWEN_CONFIG_JSON: &str = include_str!("../../assets/qwen2.5-0.5b/config.json");
    // Dense arch pack (#498) synthetic fixtures.
    const COHERE_CFG: &str = include_str!("../../assets/cohere/config.json");
    const COHERE2_CFG: &str = include_str!("../../assets/cohere2/config.json");
    const PHI3_CFG: &str = include_str!("../../assets/phi3/config.json");
    const STABLELM_CFG: &str = include_str!("../../assets/stablelm/config.json");
    const STARCODER2_CFG: &str = include_str!("../../assets/starcoder2/config.json");
    const GRANITE_CFG: &str = include_str!("../../assets/granite/config.json");
    const MINICPM_CFG: &str = include_str!("../../assets/minicpm/config.json");

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
            embed_scale: false,
            norm_one_plus: false,
            mlp_geglu: false,
            norm_style: NormStyle::Plain,
            qk_norm: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            sliding_window: None,
            // The #497 and #498 switches all default off for a Qwen-shaped config;
            // pull them (sliding_pattern, use_rope_layers, weight_scheme, and the
            // dense-arch-pack flags) from the reference Llama config.
            ..Config::llama_3_2_1b()
        }
    }

    /// A tiny Gemma2-shaped config with every Gemma2 switch on: `(1 + w)` norm,
    /// four per-layer norms, GeGLU, embedding scale, attention / final logit
    /// soft-cap, and a non-square `o_proj` (`n_q*head_dim = 12 != hidden = 8`, as
    /// in real Gemma2) for the shared-attention-core coverage test (issue #494).
    /// `n_layers = 4` gives two local (even) and two global (odd) layers so the
    /// sliding-window alternation (issue #495) is observable; `sliding_window` is
    /// the only knob the window tests vary (the coverage test passes `None`). Small
    /// dims keep the emitted text tiny while exercising every Gemma2 path.
    fn gemma2_like(sliding_window: Option<usize>) -> Config {
        Config {
            hidden: 8,
            inter: 16,
            n_layers: 4,
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
            embed_scale: true,
            norm_one_plus: true,
            mlp_geglu: true,
            norm_style: NormStyle::GemmaFf,
            qk_norm: None,
            rope_local_base: None,
            query_pre_attn_scalar: Some(6.0),
            attn_logit_softcap: Some(50.0),
            final_logit_softcap: Some(30.0),
            sliding_window,
            // sliding_pattern (2), use_rope_layers, weight_scheme, and the #498 flags
            // take their Llama defaults.
            ..Config::llama_3_2_1b()
        }
    }

    /// A `Config` parsed from the real Llama-3.2-1B-Instruct `config.json` emits
    /// every bundled graph byte-for-byte, so switching from `include_str!` to
    /// emit-at-load cannot change the compiled graphs for this model. Asserted
    /// through the reusable per-architecture validation harness (issue #496),
    /// which owns the golden fixtures (`crate::validation::LLAMA_3_2_1B`); adding
    /// a family is then a registry row rather than a copy of this test.
    #[test]
    fn from_json_reproduces_bundled_assets_byte_for_byte() {
        let report = crate::validation::check_arch(&crate::validation::LLAMA_3_2_1B)
            .expect("llama-3.2-1b fixture parses at the default precision");
        assert!(report.passed(), "{report}");
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

    /// An unsupported `model_type` and an unsupported `rope_scaling` are each
    /// rejected with a clear message rather than mis-emitted. (Untied embeddings and
    /// the Gemma / Qwen3 / SmolLM3 / OLMo2/3 families are now accepted; see their
    /// own parse tests.)
    #[test]
    fn from_json_rejects_unsupported_configs() {
        let mamba = r#"{"model_type":"mamba","tie_word_embeddings":true,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(mamba)
                .unwrap_err()
                .contains("model_type")
        );

        // yarn RoPE (e.g. OLMo3 at full size) is not supported yet.
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
        assert_eq!(c.norm_style, NormStyle::GemmaFf);
        assert!(c.embed_scale && c.norm_one_plus && c.mlp_geglu);
        assert!(c.qk_norm.is_none(), "Gemma2 has no q/k norm");
        assert!(c.rope_local_base.is_none(), "Gemma2 has a single RoPE base");
        assert_eq!(c.rope, RopeScaling::Plain);
        assert_eq!(c.query_pre_attn_scalar, Some(224.0));
        assert_eq!(c.attn_logit_softcap, Some(50.0));
        assert_eq!(c.final_logit_softcap, Some(30.0));
        assert_eq!(c.head_dim, 256, "explicit head_dim (!= hidden/n_q)");
        assert!(c.tie_word_embeddings, "Gemma2 ties embeddings by default");
        assert_eq!(
            c.sliding_window,
            Some(4096),
            "absent sliding_window defaults to the HF Gemma2 4096"
        );
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
    /// matching `weight_specs` (`weights.rs`) so the loaded weight buffers line up
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
        let gemma = gemma2_like(None);
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
    /// `weight_specs` (`weights.rs`), which adds `lm_head.weight` in the same slot.
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

    /// Issue #495: Gemma2 parses its sliding-window size, defaulting to the HF
    /// Gemma2 default (4096) when the field is absent; a non-Gemma2 architecture
    /// gets `None` even when its own config carries a `sliding_window` (Qwen2.5
    /// ships `sliding_window = 32768` but the emitter serves it globally).
    #[test]
    fn from_json_parses_sliding_window() {
        // Explicit window on a gemma2 checkpoint (gemma2-2b ships 4096).
        let explicit = r#"{"model_type":"gemma2","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10,"sliding_window":4096}"#;
        assert_eq!(
            Config::from_json_str(explicit)
                .expect("gemma2 parses")
                .sliding_window,
            Some(4096)
        );

        // Absent field -> HF Gemma2 default of 4096.
        let absent = r#"{"model_type":"gemma2","hidden_size":8,"num_attention_heads":2,
            "num_key_value_heads":1,"intermediate_size":16,"num_hidden_layers":4,
            "rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert_eq!(
            Config::from_json_str(absent)
                .expect("gemma2 parses")
                .sliding_window,
            Some(4096),
            "absent sliding_window defaults to 4096"
        );

        // Qwen2.5 ships a sliding_window (32768) but the emitter serves it with
        // global attention, so it must parse to None (not the config value).
        let qwen = Config::from_json_str(QWEN_CONFIG_JSON).expect("qwen parses");
        assert_eq!(qwen.sliding_window, None, "Qwen2 sliding_window is ignored");

        // Llama has no window.
        assert_eq!(Config::llama_3_2_1b().sliding_window, None);
    }

    /// The local/global schedule (issue #495): Gemma2 alternates starting local,
    /// so even layers are local (sliding) and odd layers global; a config with no
    /// window (Llama / Qwen2, or a Gemma2 control with the window off) has no local
    /// layer, so its graphs are unchanged.
    #[test]
    fn is_sliding_layer_alternates_even_local() {
        let g = gemma2_like(Some(4096));
        assert!(g.is_sliding_layer(0), "layer 0 local");
        assert!(!g.is_sliding_layer(1), "layer 1 global");
        assert!(g.is_sliding_layer(2), "layer 2 local");
        assert!(!g.is_sliding_layer(3), "layer 3 global");

        let none = gemma2_like(None);
        for li in 0..none.n_layers {
            assert!(!none.is_sliding_layer(li), "no window -> no local layer");
        }
        for li in 0..4 {
            assert!(!qwen_like(true).is_sliding_layer(li), "Qwen2 is global");
            assert!(
                !Config::llama_3_2_1b().is_sliding_layer(li),
                "Llama is global"
            );
        }
    }

    /// The sliding-window mask is built once per graph and only when a window is
    /// configured: turning the window on adds exactly one `subtract`, one
    /// `compare`, and one `select` (the one-time local-mask block) to each graph
    /// kind, and nothing when the window is `None`. A with/without diff over an
    /// otherwise-identical Gemma2 config isolates exactly the window surface, so
    /// the reused-across-local-layers design is confirmed and no stray op shifted.
    #[test]
    fn sliding_window_adds_one_local_mask_block_per_graph() {
        let with = gemma2_like(Some(4096));
        let without = gemma2_like(None);
        let graphs = [
            (
                emit_decode(&with, false),
                emit_decode(&without, false),
                "decode",
            ),
            (
                emit_prefill(&with, false),
                emit_prefill(&without, false),
                "prefill",
            ),
            (
                emit_decode_ragged(&with, 4, false),
                emit_decode_ragged(&without, 4, false),
                "ragged",
            ),
        ];
        for (g_with, g_without, name) in graphs {
            assert_eq!(
                occurs(&g_with, "stablehlo.subtract ") - occurs(&g_without, "stablehlo.subtract "),
                1,
                "{name}: one window age subtract"
            );
            assert_eq!(
                occurs(&g_with, "stablehlo.compare ") - occurs(&g_without, "stablehlo.compare "),
                1,
                "{name}: one window compare"
            );
            assert_eq!(
                occurs(&g_with, "stablehlo.select ") - occurs(&g_without, "stablehlo.select "),
                1,
                "{name}: one window select"
            );
        }
    }

    /// The configured window size is emitted into the graph (as the `i32` constant
    /// the key age is compared against), so a different window yields a different
    /// graph. Uses windows that do not collide with the layer-index constants.
    #[test]
    fn sliding_window_size_is_emitted_into_the_graph() {
        let g7 = emit_decode(&gemma2_like(Some(7)), false);
        let g9 = emit_decode(&gemma2_like(Some(9)), false);
        assert!(
            g7.contains("dense<7> : tensor<i32>"),
            "window 7 constant present"
        );
        assert!(!g7.contains("dense<9> : tensor<i32>"));
        assert!(
            g9.contains("dense<9> : tensor<i32>"),
            "window 9 constant present"
        );
        assert!(!g9.contains("dense<7> : tensor<i32>"));
    }

    /// The heart of #495: within one graph the even layers consume the LOCAL
    /// (sliding-window) mask and the odd layers the GLOBAL (causal) mask, proving
    /// the per-layer alternation is actually wired, not merely that a local mask
    /// exists. The score-mask broadcast is `[S] -> [nq, S]` (`dims = [1]`, here
    /// `tensor<256xf32> -> tensor<2x256xf32>` for the 2-query-head config), and it
    /// is the only broadcast with that signature; collecting its operand per layer
    /// in order shows even layers share one mask value, odd layers another, and the
    /// two differ.
    #[test]
    fn local_and_global_layers_use_distinct_masks_alternating() {
        let g = emit_decode(&gemma2_like(Some(4096)), false);
        let needle = ", dims = [1] : (tensor<256xf32>) -> tensor<2x256xf32>";
        let operands: Vec<&str> = g
            .lines()
            .filter(|l| l.contains("stablehlo.broadcast_in_dim") && l.contains(needle))
            .map(|l| {
                // "  %N = stablehlo.broadcast_in_dim %OP, dims = ..."
                let after = l
                    .split("stablehlo.broadcast_in_dim ")
                    .nth(1)
                    .expect("broadcast operand");
                after.split(',').next().expect("operand token").trim()
            })
            .collect();
        assert_eq!(operands.len(), 4, "one mask broadcast per layer");
        assert_eq!(operands[0], operands[2], "even layers share the local mask");
        assert_eq!(operands[1], operands[3], "odd layers share the global mask");
        assert_ne!(
            operands[0], operands[1],
            "local (even) and global (odd) masks are distinct values"
        );
    }

    /// Opt-in graph dump for the out-of-crate execution check (issues #495 / #498).
    /// Ignored by default; when run with `--ignored` and `MLXCEL_DUMP_CONFIG` /
    /// `MLXCEL_DUMP_OUT` set, it parses that `config.json` and writes the prefill
    /// (logits) StableHLO to `MLXCEL_DUMP_OUT`, so a `spike/openxla` harness
    /// (`gemma2_sliding_window_check.py`, `dense_arch_check.py`) can compile it with
    /// IREE and compare last-token logits to an HF fp32 oracle for that
    /// architecture. Arch-generic (any config `from_json_str` accepts), and a plain,
    /// scoped, pure-Rust entry point so the execution check never needs the
    /// heavyweight `iree` cargo feature.
    #[test]
    #[ignore = "opt-in: writes a graph to disk for the spike/openxla execution check"]
    fn dump_prefill_graph_for_execution_check() {
        let cfg_path = std::env::var("MLXCEL_DUMP_CONFIG")
            .expect("set MLXCEL_DUMP_CONFIG to a config.json path");
        let out_path =
            std::env::var("MLXCEL_DUMP_OUT").expect("set MLXCEL_DUMP_OUT to the target .mlir path");
        let text = std::fs::read_to_string(&cfg_path).expect("read MLXCEL_DUMP_CONFIG");
        let cfg = Config::from_json_str(&text).expect("parse config.json");
        std::fs::write(&out_path, emit_prefill(&cfg, false)).expect("write MLXCEL_DUMP_OUT");
    }

    /// Opt-in graph dump for the issue #499 dense-pack execution check. Ignored by
    /// default; with `MLXCEL_DUMP_CONFIG` (a checkpoint `config.json`) and
    /// `MLXCEL_DUMP_DIR` set, it writes the host-sampled prefill + decode logits
    /// graphs (`prefill_logits.mlir` / `decode_logits.mlir`) from the REAL config,
    /// so `spike/openxla/arch_execution_check.py` can compile them with IREE and
    /// drive a token-exact continuation against an HF fp32 oracle. Pure Rust /
    /// scoped, so the check never needs the `iree` cargo feature.
    #[test]
    #[ignore = "opt-in: writes prefill/decode graphs to disk for the spike/openxla execution check"]
    fn dump_dense_pack_graphs_for_execution_check() {
        let cfg_path =
            std::env::var("MLXCEL_DUMP_CONFIG").expect("set MLXCEL_DUMP_CONFIG to a config.json");
        let dir = std::env::var("MLXCEL_DUMP_DIR").expect("set MLXCEL_DUMP_DIR to an output dir");
        let text = std::fs::read_to_string(&cfg_path).expect("read MLXCEL_DUMP_CONFIG");
        let cfg = Config::from_json_str(&text).expect("parse config.json");
        let dir = std::path::Path::new(&dir);
        std::fs::write(dir.join("prefill_logits.mlir"), emit_prefill(&cfg, false))
            .expect("write prefill_logits.mlir");
        std::fs::write(dir.join("decode_logits.mlir"), emit_decode(&cfg, false))
            .expect("write decode_logits.mlir");
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

    // The precision accuracy gate (structural). Guards the "targeted, not blanket"
    // invariant that makes f16 token-exact: only matmuls are demoted; RMSNorm
    // (rsqrt) and softmax (exponential) stay f32. A blanket f32->f16 (which
    // regressed accuracy and was slower) would demote those and fail this.
    #[test]
    fn f16_precision_demotes_only_matmuls_not_norm_or_softmax() {
        use super::builder::Precision;
        let c = Config::llama_3_2_1b();

        let f16 = emit_decode_with(&c, true, Precision::F16);
        assert!(f16.contains("xf16"), "f16 mode emitted no f16");
        for line in f16.lines().filter(|l| l.contains("stablehlo.dot_general")) {
            assert!(line.contains("f16>"), "matmul not demoted to f16: {line}");
            assert!(
                line.trim_end().ends_with("f32>"),
                "matmul output not f32 (accumulate lost): {line}"
            );
        }
        for line in f16
            .lines()
            .filter(|l| l.contains("stablehlo.rsqrt") || l.contains("stablehlo.exponential"))
        {
            assert!(
                !line.contains("f16"),
                "norm/softmax wrongly demoted to f16 (blanket regression): {line}"
            );
        }

        // The f32 default carries no f16 at all (byte-exact path preserved).
        assert!(!emit_decode_with(&c, true, Precision::F32).contains("f16"));
    }

    // ======================================================================
    // dense arch pack (issue #498)
    // ======================================================================

    /// Each dense-pack family parses to its architecture switches. Cohere/Cohere2
    /// are LayerNorm + parallel-block + interleaved-RoPE with a logit multiply
    /// (Cohere2 adds sliding layers + NoPE on the full ones); Phi3 fuses q/k/v and
    /// gate/up; StableLM is LayerNorm-with-bias + partial RoPE + q/k/v bias;
    /// StarCoder2 is LayerNorm-with-bias + all biases + a dense GELU MLP; Granite is
    /// RMSNorm + four scalar multipliers.
    #[test]
    fn from_json_parses_dense_arch_pack() {
        let co = Config::from_json_str(COHERE_CFG).expect("cohere");
        assert!(co.layernorm && !co.norm_bias && co.parallel_block && co.rope_interleaved);
        assert_eq!(co.logit_mul, Some(0.25));
        assert!(co.tie_word_embeddings);

        let c2 = Config::from_json_str(COHERE2_CFG).expect("cohere2");
        assert!(c2.parallel_block && c2.rope_interleaved && c2.rope_on_sliding_only);
        assert_eq!(c2.sliding_pattern, 2);
        assert!(c2.sliding_window.is_some());

        let p3 = Config::from_json_str(PHI3_CFG).expect("phi3");
        assert!(p3.fused_qkv && p3.fused_gate_up && !p3.tie_word_embeddings);
        assert!(!p3.layernorm && p3.rotary_dim.is_none()); // RMSNorm, full RoPE

        let sl = Config::from_json_str(STABLELM_CFG).expect("stablelm");
        assert!(sl.layernorm && sl.norm_bias && sl.qkv_bias && !sl.tie_word_embeddings);
        assert_eq!(sl.rotary_dim, Some(2)); // head_dim 8 * 0.25

        let sc = Config::from_json_str(STARCODER2_CFG).expect("starcoder2");
        assert!(sc.layernorm && sc.norm_bias && sc.dense_mlp);
        assert!(sc.qkv_bias && sc.attn_o_bias && sc.mlp_bias && sc.tie_word_embeddings);

        let gr = Config::from_json_str(GRANITE_CFG).expect("granite");
        assert_eq!(gr.embedding_multiplier, Some(12.0));
        assert_eq!(gr.residual_multiplier, Some(0.22));
        assert_eq!(gr.attention_multiplier, Some(0.125));
        assert_eq!(gr.logit_div, Some(8.0));
        assert_eq!(
            gr.scale(),
            0.125,
            "granite uses attention_multiplier as the scale"
        );
    }

    /// A MiniCPM checkpoint ships as `model_type = "llama"` but keeps `scale_emb` /
    /// `scale_depth` / `dim_model_base`; detection keys on those fields (a plain
    /// Llama has none, so it is unaffected). The residual multiplier is
    /// `scale_depth / sqrt(num_layers)` and the logit divide is
    /// `hidden / dim_model_base`.
    #[test]
    fn minicpm_llama_config_detects_scalars() {
        let m = Config::from_json_str(MINICPM_CFG).expect("minicpm");
        assert_eq!(m.embedding_multiplier, Some(12.0)); // scale_emb
        // scale_depth 1.4 / sqrt(2 layers).
        let want = (1.4f64 / (m.n_layers as f64).sqrt()) as f32;
        assert_eq!(m.residual_multiplier, Some(want));
        assert_eq!(m.logit_div, Some(4.0)); // hidden 32 / dim_model_base 8
        assert!(!m.tie_word_embeddings);
        // A plain Llama (no scale_emb) gets none of these.
        let llama = Config::llama_3_2_1b();
        assert!(llama.embedding_multiplier.is_none() && llama.residual_multiplier.is_none());
        assert!(llama.logit_div.is_none());
    }

    /// MiniCPM3 (MLA) is rejected with a clear follow-up message rather than
    /// mis-emitted through the standard-attention core.
    #[test]
    fn from_json_rejects_minicpm3_mla() {
        let j = r#"{"model_type":"minicpm3","hidden_size":32,"num_attention_heads":4,
            "num_key_value_heads":4,"intermediate_size":64,"num_hidden_layers":2,
            "rms_norm_eps":1e-5,"rope_theta":1e4,"vocab_size":32}"#;
        let e = Config::from_json_str(j).unwrap_err();
        assert!(e.contains("MiniCPM3") && e.contains("MLA"), "got: {e}");
    }

    /// The parallel-block archs (Cohere/Cohere2) carry a single `input_layernorm`
    /// per layer and NO `post_attention_layernorm`; the sequential archs carry both.
    /// Holds across every shared-core graph kind.
    #[test]
    fn parallel_block_omits_post_attention_layernorm() {
        let co = Config::from_json_str(COHERE_CFG).expect("cohere");
        let ll = Config::llama_3_2_1b();
        for kind in ["decode", "prefill", "ragged"] {
            let g_co = match kind {
                "decode" => emit_decode(&co, false),
                "prefill" => emit_prefill(&co, false),
                _ => emit_decode_ragged(&co, 4, false),
            };
            assert!(
                g_co.contains("['in_ln']"),
                "{kind}: cohere has input_layernorm"
            );
            assert!(
                !g_co.contains("['post_ln']"),
                "{kind}: cohere has no post-attn norm"
            );
        }
        assert!(
            emit_decode(&ll, false).contains("['post_ln']"),
            "llama is sequential"
        );
    }

    /// The dense (StarCoder2) MLP has no gate projection and uses a `gelu_tanh`
    /// activation (`stablehlo.tanh`), unlike the gated SwiGLU of the Llama family.
    #[test]
    fn dense_mlp_has_no_gate_and_uses_gelu_tanh() {
        let sc = Config::from_json_str(STARCODER2_CFG).expect("starcoder2");
        for kind in ["decode", "prefill", "ragged"] {
            let g = match kind {
                "decode" => emit_decode(&sc, false),
                "prefill" => emit_prefill(&sc, false),
                _ => emit_decode_ragged(&sc, 4, false),
            };
            assert!(!g.contains("['gate']"), "{kind}: dense MLP has no gate");
            assert!(
                g.contains("stablehlo.tanh"),
                "{kind}: dense MLP is gelu_tanh"
            );
        }
        // A Llama SwiGLU MLP has a gate and no tanh.
        assert!(emit_decode(&Config::llama_3_2_1b(), false).contains("['gate']"));
    }

    /// Partial RoPE (StableLM) rotates only `rotary_dim` of each head: the baked
    /// cos/sin table is `[256, rotary_dim]`, narrower than the `head_dim`-wide
    /// Llama table, and a full-rope arch's is `[256, head_dim]`.
    #[test]
    fn partial_rope_shrinks_the_rope_table() {
        let sl = Config::from_json_str(STABLELM_CFG).expect("stablelm");
        assert_eq!(sl.rotary_width(), 2);
        let g = emit_decode(&sl, false);
        assert!(
            g.contains("tensor<256x2xf32>"),
            "partial rope table is [256, 2]"
        );
        assert!(
            !g.contains("tensor<256x8xf32>"),
            "not the full head_dim (8) table"
        );
    }

    /// Cohere2 applies RoPE only on the sliding (local) layers and leaves the
    /// full-attention layers position-free (NoPE): with `sliding_window_pattern = 2`
    /// the odd layers are full, so they skip the rotation.
    #[test]
    fn cohere2_ropes_only_sliding_layers() {
        let c2 = Config::from_json_str(COHERE2_CFG).expect("cohere2");
        // pattern 2: layer 0 sliding (RoPE), layer 1 full (NoPE). The merged RoPE
        // gate `layer_uses_rope` folds Cohere2's rope-on-sliding-only into the
        // SmolLM3 NoPE mask, so it captures the same per-layer decision.
        assert!(c2.is_sliding_layer(0) && c2.layer_uses_rope(0));
        assert!(!c2.is_sliding_layer(1) && !c2.layer_uses_rope(1));
        // Cohere v1 has no such gate: every layer is RoPE'd.
        let co = Config::from_json_str(COHERE_CFG).expect("cohere");
        assert!(co.layer_uses_rope(0) && co.layer_uses_rope(1));
    }

    /// Granite / MiniCPM scalars reach the graph: the embedding multiplier and the
    /// per-residual multiplier are baked constants, and the final logits are divided
    /// (Granite `logits_scaling`) rather than left unscaled (Llama).
    #[test]
    fn granite_scalars_are_emitted() {
        let gr = Config::from_json_str(GRANITE_CFG).expect("granite");
        let g = emit_decode(&gr, false);
        // The same dims with the scalars off isolates exactly the scalar surface.
        let mut plain = gr.clone();
        plain.embedding_multiplier = None;
        plain.residual_multiplier = None;
        plain.logit_div = None;
        plain.attention_multiplier = None;
        let base = emit_decode(&plain, false);
        // logits_scaling adds exactly one logit divide; the embed / residual
        // multipliers add multiplies (embed once, residual twice per layer).
        assert_eq!(
            g.matches("stablehlo.divide").count(),
            base.matches("stablehlo.divide").count() + 1,
            "granite divides the logits once (logits_scaling)"
        );
        assert_eq!(
            g.matches("stablehlo.multiply").count() - base.matches("stablehlo.multiply").count(),
            1 + 2 * gr.n_layers,
            "embed multiplier once + residual multiplier on each sublayer"
        );
        // The scalar constants are baked into the graph.
        assert!(
            g.contains(&super::builder::f32_hex(12.0)),
            "embedding multiplier 12.0"
        );
        assert!(
            g.contains(&super::builder::f32_hex(8.0)),
            "logits_scaling 8.0"
        );
    }

    // ===================================================================
    // issue #497: dense arch pack (Qwen3, Gemma1/3, SmolLM3, OLMo2/3)
    // ===================================================================

    /// A small Plain (Llama-shaped) config to derive the new families from. Tiny
    /// dims keep the emitted text small while exercising every shared-core path;
    /// `head_dim` (4) deliberately differs from `hidden / n_q`, and `n_q*head_dim`
    /// (12) from `hidden` (8), so the flat q-norm and non-square o_proj widths are
    /// genuinely distinct (as in real checkpoints).
    fn dense_base() -> Config {
        Config {
            hidden: 8,
            inter: 16,
            n_layers: 4,
            n_q: 3,
            n_kv: 1,
            head_dim: 4,
            eps: 1e-6,
            rope_theta: 1e4,
            vocab: 12,
            rope: RopeScaling::Plain,
            qkv_bias: false,
            tie_word_embeddings: true,
            quantization: None,
            embed_scale: false,
            norm_one_plus: false,
            mlp_geglu: false,
            norm_style: NormStyle::Plain,
            qk_norm: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            sliding_window: None,
            // sliding_pattern (2), use_rope_layers, weight_scheme, and the #498
            // dense-arch-pack flags all take their Llama defaults.
            ..Config::llama_3_2_1b()
        }
    }

    fn qwen3_like() -> Config {
        Config {
            qk_norm: Some(QkNorm {
                per_head: true,
                one_plus: false,
            }),
            ..dense_base()
        }
    }

    fn gemma1_like() -> Config {
        Config {
            embed_scale: true,
            norm_one_plus: true,
            mlp_geglu: true,
            ..dense_base()
        }
    }

    fn gemma3_like() -> Config {
        Config {
            embed_scale: true,
            norm_one_plus: true,
            mlp_geglu: true,
            norm_style: NormStyle::GemmaFf,
            qk_norm: Some(QkNorm {
                per_head: true,
                one_plus: true,
            }),
            rope_local_base: Some(1e3),
            query_pre_attn_scalar: Some(4.0),
            sliding_window: Some(2),
            sliding_pattern: 3,
            ..dense_base()
        }
    }

    fn olmo2_like() -> Config {
        Config {
            norm_style: NormStyle::OlmoPost,
            qk_norm: Some(QkNorm {
                per_head: false,
                one_plus: false,
            }),
            tie_word_embeddings: false,
            ..dense_base()
        }
    }

    /// The three shared-core graph kinds (single decode, ragged decode, prefill),
    /// as `(name, emitter)` pairs, so a family delta is checked reaching all of them.
    fn shared_core_kinds(c: &Config) -> [(&'static str, String); 3] {
        [
            ("decode", emit_decode(c, false)),
            ("ragged", emit_decode_ragged(c, 4, false)),
            ("prefill", emit_prefill(c, false)),
        ]
    }

    /// Qwen3's per-head q/k RMSNorm reaches every shared-core graph kind: turning it
    /// on adds exactly the two `[head_dim]` norm weights per layer and the two extra
    /// `rsqrt`s (one for q, one for k) that normalize each head before RoPE, and
    /// nothing else. A with/without diff over an otherwise-identical config isolates
    /// exactly the q/k-norm surface.
    #[test]
    fn qwen3_per_head_qk_norm_reaches_every_shared_core_kind() {
        let with = qwen3_like();
        let without = dense_base();
        let nl = with.n_layers;
        for ((_, g_with), (name, g_without)) in shared_core_kinds(&with)
            .iter()
            .zip(shared_core_kinds(&without))
        {
            assert_eq!(
                arg_count(g_with) - arg_count(&g_without),
                2 * nl,
                "{name}: q_norm + k_norm per layer"
            );
            assert_eq!(occurs(g_with, "['q_norm']"), nl, "{name}: one q_norm/layer");
            assert_eq!(occurs(g_with, "['k_norm']"), nl, "{name}: one k_norm/layer");
            assert_eq!(
                occurs(g_with, "stablehlo.rsqrt") - occurs(&g_without, "stablehlo.rsqrt"),
                2 * nl,
                "{name}: one rsqrt each for the q and k head-norm per layer"
            );
            // Per-head: the norm weight is [head_dim] (4), not the flat n_q*head_dim.
            assert!(
                g_with.contains("tensor<4xf32> loc(\"params['layers'][0]['q_norm']\")"),
                "{name}: qwen3 q_norm is per-head [head_dim]"
            );
        }
    }

    /// OLMo2 is the reordered post-norm structure with a FLAT q/k norm and an untied
    /// head: no `input_layernorm`, the q/k norm sized over the whole projection
    /// (`n_q*head_dim` = 12, `n_kv*head_dim` = 4), a `post_feedforward_layernorm` but
    /// no `pre_feedforward_layernorm`, and a separate `lm_head`. Asserted on every
    /// shared-core kind so the post-norm reaches all of them.
    #[test]
    fn olmo2_flat_qk_norm_and_post_norm_structure() {
        let c = olmo2_like();
        let nl = c.n_layers;
        for (name, g) in shared_core_kinds(&c) {
            assert_eq!(
                occurs(&g, "['in_ln']"),
                0,
                "{name}: OLMo2 has no input norm"
            );
            assert_eq!(occurs(&g, "['q_norm']"), nl, "{name}: one q_norm/layer");
            assert_eq!(occurs(&g, "['k_norm']"), nl, "{name}: one k_norm/layer");
            assert_eq!(
                occurs(&g, "['post_ff_ln']"),
                nl,
                "{name}: post-feedforward norm/layer"
            );
            assert_eq!(occurs(&g, "['pre_ff_ln']"), 0, "{name}: no pre-ff norm");
            assert_eq!(occurs(&g, "params['lm_head']"), 1, "{name}: untied head");
            // Flat: q_norm is [n_q*head_dim] (12), NOT [head_dim].
            assert!(
                g.contains("tensor<12xf32> loc(\"params['layers'][0]['q_norm']\")"),
                "{name}: olmo2 q_norm is flat [n_q*head_dim]"
            );
        }
    }

    /// Gemma1 has the Gemma activation/scale surface (GeGLU `tanh`, `(1+w)` norm,
    /// embedding scale) but the Llama TWO-norm layer (an `input_layernorm`, no
    /// pre/post feed-forward norms) and no q/k norm, distinguishing it from Gemma2/3.
    /// The embedding scale is isolated by a with/without diff.
    #[test]
    fn gemma1_is_two_norm_geglu_with_embed_scale() {
        let g1 = gemma1_like();
        let nl = g1.n_layers;
        for (name, g) in shared_core_kinds(&g1) {
            assert!(g.contains("stablehlo.tanh"), "{name}: GeGLU emits tanh");
            assert_eq!(
                occurs(&g, "['in_ln']"),
                nl,
                "{name}: Gemma1 keeps input norm"
            );
            assert_eq!(
                occurs(&g, "['pre_ff_ln']"),
                0,
                "{name}: no pre-ff norm (2-norm)"
            );
            assert_eq!(
                occurs(&g, "['post_ff_ln']"),
                0,
                "{name}: no post-ff norm (2-norm)"
            );
            assert_eq!(
                occurs(&g, "['q_norm']"),
                0,
                "{name}: Gemma1 has no q/k norm"
            );
        }
        // The embedding scale is one const + broadcast + multiply in the head.
        let no_scale = Config {
            embed_scale: false,
            ..gemma1_like()
        };
        let d_with = emit_decode(&g1, false);
        let d_without = emit_decode(&no_scale, false);
        assert_eq!(
            occurs(&d_with, "stablehlo.multiply") - occurs(&d_without, "stablehlo.multiply"),
            1,
            "embed scale adds exactly one head multiply"
        );
    }

    /// SmolLM3's NoPE mask skips RoPE on the marked layers: the rotate-half
    /// `concatenate` (two per rope'd layer, q and k) drops by exactly two per NoPE
    /// layer, and nothing else changes. A with/without diff over the NoPE mask
    /// isolates it on every shared-core kind.
    #[test]
    fn smollm3_nope_skips_rope_on_marked_layers() {
        // Layer 3 is NoPE (the SmolLM3 every-fourth-layer pattern at n_layers = 4).
        let with_nope = Config {
            use_rope_layers: Some(vec![true, true, true, false]),
            ..dense_base()
        };
        assert!(!with_nope.layer_uses_rope(3), "layer 3 is NoPE");
        assert!(with_nope.layer_uses_rope(0), "layer 0 keeps RoPE");
        let all_rope = dense_base();
        for ((_, g_nope), (name, g_all)) in shared_core_kinds(&with_nope)
            .iter()
            .zip(shared_core_kinds(&all_rope))
        {
            assert_eq!(
                occurs(&g_all, "stablehlo.concatenate") - occurs(g_nope, "stablehlo.concatenate"),
                2,
                "{name}: the one NoPE layer drops the q and k rotate-half concatenates"
            );
        }
    }

    /// Gemma3 pairs the Gemma2 four-norm layer and a per-head `(1+w)` q/k norm with a
    /// DUAL RoPE base: the sliding layers rotate on a local-base table distinct from
    /// the global one, so the graph carries two extra `[MAX_SEQ, head_dim]` constant
    /// tables (cos_local, sin_local) versus a single-RoPE twin.
    #[test]
    fn gemma3_dual_rope_and_qk_norm_reach_the_shared_core() {
        let g3 = gemma3_like();
        let nl = g3.n_layers;
        for (name, g) in shared_core_kinds(&g3) {
            assert_eq!(occurs(&g, "['q_norm']"), nl, "{name}: per-head q norm");
            assert_eq!(
                occurs(&g, "['pre_ff_ln']"),
                nl,
                "{name}: Gemma 4-norm (pre)"
            );
            assert_eq!(
                occurs(&g, "['post_ff_ln']"),
                nl,
                "{name}: Gemma 4-norm (post)"
            );
            assert!(g.contains("stablehlo.tanh"), "{name}: GeGLU tanh");
        }
        // Dual RoPE: two extra dense hex-blob constant tables (the local cos/sin).
        let single = Config {
            rope_local_base: None,
            ..gemma3_like()
        };
        for ((_, g_dual), (name, g_single)) in shared_core_kinds(&g3)
            .iter()
            .zip(shared_core_kinds(&single))
        {
            assert_eq!(
                occurs(g_dual, "stablehlo.constant dense<\"0x")
                    - occurs(&g_single, "stablehlo.constant dense<\"0x"),
                2,
                "{name}: dual-RoPE adds the local cos + sin tables"
            );
        }
    }

    /// The local (sliding) layers rotate on the local RoPE table and the global
    /// layers on the global table (Gemma3 dual-RoPE). With `sliding_pattern = 3` and
    /// `n_layers = 4`, layers 0/1/3 are sliding and layer 2 is global, so
    /// `local_rope_layer` selects the local table on exactly the sliding layers.
    #[test]
    fn gemma3_local_rope_selected_on_sliding_layers() {
        let g3 = gemma3_like(); // sliding_pattern = 3
        assert!(g3.local_rope_layer(0), "layer 0 sliding -> local rope");
        assert!(g3.local_rope_layer(1), "layer 1 sliding -> local rope");
        assert!(
            !g3.local_rope_layer(2),
            "layer 2 global (3rd) -> global rope"
        );
        assert!(g3.local_rope_layer(3), "layer 3 sliding -> local rope");
    }

    /// Each new family's real `config.json` shape parses to the expected flags.
    #[test]
    fn from_json_parses_new_dense_families() {
        // Qwen3: per-head q/k norm, no bias, explicit head_dim, plain RoPE.
        let qwen3 = r#"{"model_type":"qwen3","hidden_size":1024,"num_attention_heads":16,
            "num_key_value_heads":8,"head_dim":128,"intermediate_size":3072,
            "num_hidden_layers":28,"rms_norm_eps":1e-6,"rope_theta":1000000,"vocab_size":151936,
            "attention_bias":false,"tie_word_embeddings":true}"#;
        let c = Config::from_json_str(qwen3).expect("qwen3 parses");
        assert_eq!(
            c.qk_norm,
            Some(QkNorm {
                per_head: true,
                one_plus: false
            })
        );
        assert!(!c.qkv_bias, "Qwen3 drops the Qwen2 bias");
        assert_eq!(c.head_dim, 128, "explicit head_dim != hidden/heads");
        assert_eq!(c.norm_style, NormStyle::Plain);

        // Gemma1: Plain norm, embed scale + (1+w) + GeGLU, no q/k norm, no sliding.
        let gemma = r#"{"model_type":"gemma","hidden_size":2048,"num_attention_heads":8,
            "num_key_value_heads":1,"head_dim":256,"intermediate_size":16384,
            "num_hidden_layers":18,"rms_norm_eps":1e-6,"rope_theta":10000.0,"vocab_size":256000,
            "hidden_activation":"gelu_pytorch_tanh"}"#;
        let c = Config::from_json_str(gemma).expect("gemma parses");
        assert_eq!(
            c.norm_style,
            NormStyle::Plain,
            "Gemma1 is Llama-shaped 2-norm"
        );
        assert!(c.embed_scale && c.norm_one_plus && c.mlp_geglu);
        assert!(c.qk_norm.is_none() && c.sliding_window.is_none());

        // Gemma3: GemmaFf 4-norm, per-head (1+w) q/k norm, dual RoPE, 5:1 sliding.
        let gemma3 = r#"{"model_type":"gemma3_text","hidden_size":1152,"num_attention_heads":4,
            "num_key_value_heads":1,"head_dim":256,"intermediate_size":6912,
            "num_hidden_layers":26,"rms_norm_eps":1e-6,"rope_theta":1000000,
            "rope_local_base_freq":10000,"sliding_window":512,"sliding_window_pattern":6,
            "query_pre_attn_scalar":256,"attn_logit_softcapping":null,
            "final_logit_softcapping":null,"vocab_size":262144,
            "hidden_activation":"gelu_pytorch_tanh"}"#;
        let c = Config::from_json_str(gemma3).expect("gemma3 parses");
        assert_eq!(c.norm_style, NormStyle::GemmaFf);
        assert_eq!(
            c.qk_norm,
            Some(QkNorm {
                per_head: true,
                one_plus: true
            })
        );
        assert_eq!(c.rope_local_base, Some(10000.0), "distinct local RoPE base");
        assert_eq!(c.sliding_window, Some(512));
        assert_eq!(c.sliding_pattern, 6, "5 local : 1 global");
        assert!(c.attn_logit_softcap.is_none(), "Gemma3 drops the soft-caps");

        // SmolLM3: NoPE mask (every 4th layer), no q/k norm.
        let smollm3 = r#"{"model_type":"smollm3","hidden_size":2048,"num_attention_heads":16,
            "num_key_value_heads":4,"intermediate_size":11008,"num_hidden_layers":8,
            "rms_norm_eps":1e-6,"rope_theta":5000000.0,"vocab_size":128256,
            "no_rope_layers":[1,1,1,0,1,1,1,0]}"#;
        let c = Config::from_json_str(smollm3).expect("smollm3 parses");
        assert!(c.qk_norm.is_none());
        let rope = c.use_rope_layers.as_ref().expect("NoPE mask present");
        assert_eq!(rope, &[true, true, true, false, true, true, true, false]);
        assert!(!c.layer_uses_rope(3) && c.layer_uses_rope(0));

        // OLMo2: reordered post-norm, flat q/k norm, untied.
        let olmo2 = r#"{"model_type":"olmo2","hidden_size":4096,"num_attention_heads":32,
            "num_key_value_heads":32,"intermediate_size":11008,"num_hidden_layers":32,
            "rms_norm_eps":1e-6,"rope_theta":500000,"vocab_size":100352,
            "tie_word_embeddings":false}"#;
        let c = Config::from_json_str(olmo2).expect("olmo2 parses");
        assert_eq!(c.norm_style, NormStyle::OlmoPost);
        assert_eq!(
            c.qk_norm,
            Some(QkNorm {
                per_head: false,
                one_plus: false
            })
        );
        assert!(!c.tie_word_embeddings);

        // A plain-RoPE OLMo3 (structure only; the full checkpoint's yarn RoPE is a
        // documented follow-up rejected by the rope guard).
        let olmo3 = r#"{"model_type":"olmo3","hidden_size":5120,"num_attention_heads":40,
            "num_key_value_heads":8,"intermediate_size":27648,"num_hidden_layers":64,
            "rms_norm_eps":1e-6,"rope_theta":500000,"vocab_size":100278,
            "sliding_window":4096,"sliding_window_pattern":4,"tie_word_embeddings":false}"#;
        let c = Config::from_json_str(olmo3).expect("plain-rope olmo3 parses");
        assert_eq!(c.norm_style, NormStyle::OlmoPost);
        assert_eq!(
            c.qk_norm,
            Some(QkNorm {
                per_head: false,
                one_plus: false
            })
        );
        assert_eq!(c.sliding_window, Some(4096));
        assert_eq!(c.sliding_pattern, 4, "3 sliding : 1 global");
    }

    /// OLMo3 at full size uses yarn RoPE, which is rejected with a clear message
    /// (the documented follow-up), rather than silently mis-emitted.
    #[test]
    fn from_json_rejects_yarn_olmo3() {
        let olmo3_yarn = r#"{"model_type":"olmo3","hidden_size":5120,"num_attention_heads":40,
            "num_key_value_heads":8,"intermediate_size":27648,"num_hidden_layers":64,
            "rms_norm_eps":1e-6,"rope_theta":500000,"vocab_size":100278,"sliding_window":4096,
            "tie_word_embeddings":false,"rope_scaling":{"rope_type":"yarn","factor":8.0,
            "original_max_position_embeddings":8192}}"#;
        assert!(
            Config::from_json_str(olmo3_yarn)
                .unwrap_err()
                .contains("rope_type"),
            "yarn RoPE is rejected"
        );
    }
}
