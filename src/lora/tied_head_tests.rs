// Copyright 2025-2026 Lablup Inc.
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

//! Unit tests for the tied output-head verdict (issue #1577).
//!
//! These pin the verdict itself against each way a checkpoint can be tied.
//! That every load path reaches it is pinned by
//! `loading/adapter_load_tests.rs` and the pipeline Llama stage tests.

use std::path::PathBuf;

use super::*;
use crate::lora::test_support::{
    TINY_HIDDEN, TINY_VOCAB, TinyLlama, adapter_pair_tensors, temp_dir, write_adapter_dir,
    write_tiny_llama_checkpoint,
};

fn checkpoint(name: &str, tie_word_embeddings: Option<bool>, stores_lm_head: bool) -> PathBuf {
    let dir = temp_dir(name);
    write_tiny_llama_checkpoint(
        &dir,
        &TinyLlama {
            tie_word_embeddings,
            stores_lm_head,
            unread: Vec::new(),
        },
    );
    dir
}

/// An adapter holding one `[hidden] -> [vocab]` pair on `layer`.
fn head_adapter(name: &str, layer: &str) -> PathBuf {
    let dir = temp_dir(name);
    write_adapter_dir(
        &dir,
        "lora",
        2,
        adapter_pair_tensors(layer, TINY_HIDDEN, 2, TINY_VOCAB),
    );
    dir
}

fn cleanup(dirs: &[&Path]) {
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_head_pair_on_a_config_tied_checkpoint_is_refused() {
    let model = checkpoint("tied_cfg_model", Some(true), false);
    let adapter = head_adapter("tied_cfg_adapter", "lm_head");

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("an lm_head pair on a tied checkpoint must not load")
        .to_string();
    assert!(err.contains("lm_head"), "{err}");
    assert!(err.contains("ties its output projection"), "{err}");
    assert!(err.contains("tie_word_embeddings"), "{err}");
    assert!(err.contains(&adapter.display().to_string()), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn a_head_pair_is_refused_when_the_config_ties_even_if_a_head_is_stored() {
    // The Llama constructors read `model.embed_tokens` for the head whenever
    // the config ties it, so a stored `lm_head.weight` is never read and a
    // pair fused into it would be dropped without a word.
    let model = checkpoint("tied_stored_model", Some(true), true);
    let adapter = head_adapter("tied_stored_adapter", "lm_head");

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("the config's tie decides, not the stored tensor")
        .to_string();
    assert!(err.contains("ties its output projection"), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn a_head_pair_on_a_checkpoint_without_its_own_head_is_refused() {
    // No `tie_word_embeddings` key and no stored head: the loaders fill
    // `lm_head.weight` by copying the embeddings, which is a tie in effect.
    let model = checkpoint("tied_implicit_model", None, false);
    let adapter = head_adapter("tied_implicit_adapter", "lm_head");

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("an lm_head the checkpoint does not store is the embedding matrix")
        .to_string();
    assert!(err.contains("stores no lm_head.weight"), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn a_head_pair_on_an_untied_checkpoint_passes() {
    let model = checkpoint("untied_model", Some(false), true);
    let adapter = head_adapter("untied_adapter", "lm_head");

    reject_tied_output_head_adapter(&model, &adapter)
        .expect("a separately stored, untied head can take a LoRA pair");

    cleanup(&[&model, &adapter]);
}

#[test]
fn an_adapter_without_a_head_pair_passes_on_a_tied_checkpoint() {
    let model = checkpoint("tied_nohead_model", Some(true), false);
    let adapter = temp_dir("tied_nohead_adapter");
    write_adapter_dir(
        &adapter,
        "lora",
        2,
        adapter_pair_tensors(
            "model.layers.0.self_attn.q_proj",
            TINY_HIDDEN,
            2,
            TINY_HIDDEN,
        ),
    );

    reject_tied_output_head_adapter(&model, &adapter)
        .expect("a tie only matters to an adapter that targets the head");

    cleanup(&[&model, &adapter]);
}

#[test]
fn head_stems_cover_both_spellings_and_the_vlm_head() {
    assert_eq!(output_head_of("lm_head.lora_a"), Some("lm_head"));
    assert_eq!(output_head_of("lm_head.base_layer.lora_b"), Some("lm_head"));
    assert_eq!(
        output_head_of("language_model.lm_head.lora_a"),
        Some("language_model.lm_head")
    );
    assert_eq!(
        output_head_of("model.layers.0.self_attn.q_proj.lora_a"),
        None
    );
    assert_eq!(output_head_of("lm_head.weight"), None);
    assert_eq!(output_head_of("model.lm_head_norm.lora_a"), None);
}

#[test]
fn the_config_tie_is_read_from_text_config_too() {
    let nested = serde_json::json!({ "text_config": { "tie_word_embeddings": true } });
    assert!(declares_tied_embeddings(&nested));
    let top_false = serde_json::json!({
        "tie_word_embeddings": false,
        "text_config": { "tie_word_embeddings": true }
    });
    assert!(!declares_tied_embeddings(&top_false), "the top level wins");
    assert!(!declares_tied_embeddings(&serde_json::json!({})));
}
