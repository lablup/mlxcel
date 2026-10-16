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

//! Unit tests for the tied output-head verdict (issues #1577 and #2332).
//!
//! These pin the verdict itself against each way a checkpoint can be tied,
//! for a pair on the output head and for a pair on the input embedding.
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

/// An adapter holding one `model.embed_tokens` pair in each orientation a
/// file can carry: `linear` is the `[hidden] -> [vocab]` shape every load path
/// accepts for the embedding matrix, `!linear` is mlx-lm's `LoRAEmbedding`
/// layout, `lora_a = [vocab, rank]`, `lora_b = [rank, hidden]`.
fn embedding_adapter(name: &str, linear: bool) -> PathBuf {
    let dir = temp_dir(name);
    let tensors = if linear {
        adapter_pair_tensors("model.embed_tokens", TINY_HIDDEN, 2, TINY_VOCAB)
    } else {
        adapter_pair_tensors("model.embed_tokens", TINY_VOCAB, 2, TINY_HIDDEN)
    };
    write_adapter_dir(&dir, "lora", 2, tensors);
    dir
}

fn assert_embedding_refusal(err: &str, adapter: &Path) {
    assert!(err.contains("model.embed_tokens"), "{err}");
    assert!(err.contains("ties its output projection"), "{err}");
    assert!(err.contains("input-embedding LoRA"), "{err}");
    assert!(err.contains("mlx_lm.fuse"), "{err}");
    assert!(
        !err.contains("output-head LoRA"),
        "an embedding-only adapter is not told about a head pair it does not carry: {err}"
    );
    assert!(err.contains(&adapter.display().to_string()), "{err}");
}

#[test]
fn an_embedding_pair_on_a_config_tied_checkpoint_is_refused_in_either_orientation() {
    // #2332: the tied head reads the embedding matrix, so an embed_tokens pair
    // lands on whichever end the load path's constructors happen to claim it
    // for. Both on-disk orientations get the same verdict before any shape
    // check runs.
    let model = checkpoint("tied_embed_cfg_model", Some(true), false);
    for linear in [true, false] {
        let adapter = embedding_adapter("tied_embed_cfg_adapter", linear);
        let err = reject_tied_output_head_adapter(&model, &adapter)
            .expect_err("an embed_tokens pair on a tied checkpoint must not load")
            .to_string();
        assert_embedding_refusal(&err, &adapter);
        assert!(err.contains("tie_word_embeddings"), "{err}");
        cleanup(&[&adapter]);
    }
    cleanup(&[&model]);
}

#[test]
fn an_embedding_pair_on_a_checkpoint_without_its_own_head_is_refused() {
    let model = checkpoint("tied_embed_implicit_model", None, false);
    let adapter = embedding_adapter("tied_embed_implicit_adapter", true);

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("with no stored head the embedding matrix is the head")
        .to_string();
    assert_embedding_refusal(&err, &adapter);
    assert!(err.contains("stores no lm_head.weight"), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn an_embedding_pair_on_an_untied_checkpoint_passes_the_tie_verdict() {
    // The tie is the only thing this verdict is about; an untied embedding is
    // left to the loaders' own checks.
    let model = checkpoint("untied_embed_model", Some(false), true);
    let adapter = embedding_adapter("untied_embed_adapter", true);

    reject_tied_output_head_adapter(&model, &adapter)
        .expect("an untied embedding is not the output head");

    cleanup(&[&model, &adapter]);
}

#[test]
fn a_head_and_an_embedding_pair_are_refused_together() {
    let model = checkpoint("tied_both_model", Some(true), false);
    let adapter = temp_dir("tied_both_adapter");
    let mut tensors = adapter_pair_tensors("lm_head", TINY_HIDDEN, 2, TINY_VOCAB);
    tensors.extend(adapter_pair_tensors(
        "model.embed_tokens",
        TINY_HIDDEN,
        2,
        TINY_VOCAB,
    ));
    write_adapter_dir(&adapter, "lora", 2, tensors);

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("both ends of the tie are refused")
        .to_string();
    assert!(
        err.contains("a LoRA pair for lm_head, model.embed_tokens,"),
        "{err}"
    );
    assert!(err.contains("output-head LoRA"), "{err}");
    assert!(err.contains("input-embedding LoRA"), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn embedding_stems_cover_the_text_and_vlm_layouts() {
    assert_eq!(
        tied_end_of("model.embed_tokens.lora_a"),
        Some(TiedTarget::Embedding("model.embed_tokens", "lm_head"))
    );
    assert_eq!(
        tied_end_of("model.embed_tokens.base_layer.lora_b"),
        Some(TiedTarget::Embedding("model.embed_tokens", "lm_head"))
    );
    assert_eq!(
        tied_end_of("language_model.model.embed_tokens.lora_a"),
        Some(TiedTarget::Embedding(
            "language_model.model.embed_tokens",
            "language_model.lm_head"
        ))
    );
    assert_eq!(
        tied_end_of("model.language_model.embed_tokens.lora_a"),
        Some(TiedTarget::Embedding(
            "model.language_model.embed_tokens",
            "lm_head"
        ))
    );
    assert_eq!(
        tied_end_of("lm_head.lora_a"),
        Some(TiedTarget::Head("lm_head"))
    );
    // The resolver reads a stem that names the `.weight` itself as that
    // weight, so the verdict must too.
    assert_eq!(
        tied_end_of("model.embed_tokens.weight.lora_a"),
        Some(TiedTarget::Embedding("model.embed_tokens", "lm_head"))
    );
    assert_eq!(
        tied_end_of("lm_head.weight.lora_b"),
        Some(TiedTarget::Head("lm_head"))
    );
    assert_eq!(tied_end_of("model.embed_tokens.weight"), None);
    assert_eq!(tied_end_of("model.embed_tokens_norm.lora_a"), None);
    assert_eq!(tied_end_of("model.layers.0.self_attn.q_proj.lora_a"), None);
}

#[test]
fn without_a_declared_tie_only_the_ends_whose_head_is_missing_are_refused() {
    // No `tie_word_embeddings` key and a stored `lm_head.weight`: the text
    // head is untied, so its pair passes, while the nested VLM spelling of the
    // embedding belongs to a `language_model.lm_head` the checkpoint does not
    // store.
    let model = checkpoint("partial_tie_model", None, true);
    let adapter = temp_dir("partial_tie_adapter");
    let mut tensors = adapter_pair_tensors("lm_head", TINY_HIDDEN, 2, TINY_VOCAB);
    tensors.extend(adapter_pair_tensors(
        "language_model.model.embed_tokens",
        TINY_HIDDEN,
        2,
        TINY_VOCAB,
    ));
    write_adapter_dir(&adapter, "lora", 2, tensors);

    let err = reject_tied_output_head_adapter(&model, &adapter)
        .expect_err("the embedding whose head is not stored is refused")
        .to_string();
    assert!(
        err.contains("a LoRA pair for language_model.model.embed_tokens, but"),
        "{err}"
    );
    assert!(
        err.contains("stores no language_model.lm_head.weight of its own"),
        "{err}"
    );
    assert!(
        !err.contains("output-head LoRA"),
        "the stored lm_head is untied and its pair is not refused: {err}"
    );

    cleanup(&[&model, &adapter]);
}
