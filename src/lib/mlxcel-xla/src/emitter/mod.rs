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
//! Scope: the Llama and Qwen2 architectures (RMSNorm, SwiGLU MLP, GQA, tied
//! embeddings). The `Config` is parameterized by dimensions, so any checkpoint of
//! a supported architecture (any size) emits correctly. The two architecture
//! switches the emitter branches on are the RoPE kind (llama3 scaling for Llama,
//! plain for Qwen2) and whether the q/k/v projections carry a bias (Qwen2, Stage
//! B); other architectures (e.g. Gemma GeGLU / softcap) are a follow-up that
//! extends the emitter and `from_json`.
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

    /// A non-Llama/Qwen2 architecture, untied embeddings, and an unsupported
    /// `rope_scaling` are each rejected with a clear message rather than
    /// mis-emitted.
    #[test]
    fn from_json_rejects_unsupported_configs() {
        let gemma = r#"{"model_type":"gemma2","tie_word_embeddings":true,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(gemma)
                .unwrap_err()
                .contains("model_type")
        );

        let untied = r#"{"model_type":"qwen2","tie_word_embeddings":false,"hidden_size":8,
            "num_attention_heads":2,"intermediate_size":16,"num_hidden_layers":2,
            "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1e4,"vocab_size":10}"#;
        assert!(
            Config::from_json_str(untied)
                .unwrap_err()
                .contains("tie_word_embeddings")
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
