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
//! Scope: the Llama architecture (RMSNorm, SwiGLU MLP, llama3 RoPE scaling, no
//! attention bias, tied embeddings). The `Config` is parameterized by dimensions,
//! so any Llama-architecture checkpoint (any size) emits correctly; genuinely
//! different architectures (e.g. Qwen2 QKV bias, Gemma GeGLU / softcap) are a
//! follow-up that extends the emitter and `from_json`.
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
    use super::*;

    const CONFIG_JSON: &str = include_str!("../../assets/llama-3.2-1b/config.json");
    const PREFILL: &str = include_str!("../../assets/llama-3.2-1b/prefill.mlir");
    const DECODE: &str = include_str!("../../assets/llama-3.2-1b/decode.mlir");
    const PREFILL_LOGITS: &str = include_str!("../../assets/llama-3.2-1b/prefill_logits.mlir");
    const RAGGED_B4: &str = include_str!("../../assets/llama-3.2-1b/decode_ragged_logits_b4.mlir");
    const RAGGED_B8: &str = include_str!("../../assets/llama-3.2-1b/decode_ragged_logits_b8.mlir");

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
}
