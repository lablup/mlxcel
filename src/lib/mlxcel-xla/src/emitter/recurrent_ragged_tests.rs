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

//! The ragged (continuous-batching) decode for recurrent configs (issue
//! #3340): per-slot `[B, ...dims]` state inputs and results, no KV for an
//! attention-free model. Hybrid coverage is in `nemotron_h_tests`.

use super::builder::Precision;
use super::config::Config;
use super::model::{emit_decode_ragged, emit_decode_ragged_with};

const TINY_MAMBA2: &str = r#"{
  "model_type": "mamba2", "hidden_size": 16, "num_hidden_layers": 2,
  "vocab_size": 40, "num_heads": 4, "head_dim": 8, "state_size": 16,
  "n_groups": 2, "conv_kernel": 4, "expand": 2, "use_conv_bias": true,
  "use_bias": false, "time_step_limit": [0.0, Infinity],
  "layer_norm_epsilon": 1e-05, "chunk_size": 256, "rms_norm": true,
  "hidden_act": "silu", "tie_word_embeddings": true
}"#;

fn tiny() -> Config {
    Config::from_json_str(TINY_MAMBA2).expect("tiny mamba2 parses")
}

fn main_signature(graph: &str) -> &str {
    graph
        .lines()
        .find(|line| line.contains("func.func public @main"))
        .expect("graph has a main")
}

#[test]
fn mamba2_ragged_decode_threads_per_slot_state_without_kv() {
    let c = tiny();
    for bsz in [4, 8] {
        let graph = emit_decode_ragged(&c, bsz, false).expect("ragged mamba2 emits");
        for banned in ["scatter", "stablehlo.while", "stablehlo.custom_call"] {
            assert!(!graph.contains(banned), "graph must not emit {banned}");
        }
        assert!(
            !graph.contains("kcache"),
            "no KV for an attention-free model"
        );
        assert!(
            !graph.contains("util.global"),
            "ragged state is threaded, not global"
        );
        let sig = main_signature(&graph);
        let cache_len = sig.find("loc(\"cache_len\")").expect("cache_len arg");
        let h_ty = format!("tensor<{bsz}x2x4x8x16xf32>");
        let conv_ty = format!("tensor<{bsz}x2x3x96xf32>");
        let h = sig
            .find(&format!("{h_ty} loc(\"ssm_h\")"))
            .expect("ssm_h arg");
        let conv = sig
            .find(&format!("{conv_ty} loc(\"ssm_conv\")"))
            .expect("ssm_conv arg");
        assert!(cache_len < h && h < conv, "state follows cache_len: {sig}");
        assert!(
            sig.ends_with(&format!("-> (tensor<{bsz}x40xf32>, {h_ty}, {conv_ty}) {{")),
            "{sig}"
        );
        // Only the token embedding gathers: no RoPE table, no key mask.
        assert_eq!(graph.matches("\"stablehlo.gather\"(").count(), 1);
    }
}

#[test]
fn mamba2_ragged_decode_samples_and_emits_at_every_precision() {
    let c = tiny();
    for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
        let graph = emit_decode_ragged_with(&c, 4, true, precision).expect("emits");
        let sig = main_signature(&graph);
        assert!(
            sig.ends_with("-> (tensor<4xi32>, tensor<4x2x4x8x16xf32>, tensor<4x2x3x96xf32>) {"),
            "the recurrence stays f32 at {precision:?}: {sig}"
        );
    }
}

/// Each slot's state row is written back at its own layer row with the batch
/// axis intact: one write per state tensor per recurrent layer.
#[test]
fn mamba2_ragged_decode_writes_every_layer_row_back() {
    let graph = emit_decode_ragged(&tiny(), 4, false).expect("emits");
    let writes = graph
        .lines()
        .filter(|line| line.contains("stablehlo.dynamic_update_slice"))
        .count();
    assert_eq!(writes, 2 * 2, "ssm_h and ssm_conv for each of two layers");
}

#[test]
fn attention_ragged_decode_keeps_its_kv_results() {
    let c = Config::llama_3_2_1b();
    let graph = emit_decode_ragged(&c, 4, false).expect("dense ragged emits");
    let sig = main_signature(&graph);
    let cache = format!(
        "tensor<4x{}x{}x{}x{}xf32>",
        c.n_layers,
        c.context_capacity,
        c.cache_heads(),
        c.cache_dim()
    );
    assert!(
        sig.ends_with(&format!(
            "-> (tensor<4x{}xf32>, {cache}, {cache}) {{",
            c.vocab
        )),
        "{sig}"
    );
    assert!(!graph.contains("ssm_"));
}
