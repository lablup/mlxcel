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

//! Pipeline Llama stage adapter tests (issue #1577).
//!
//! The stage loader validates an adapter against `load_text_weights`, which
//! has already synthesized `lm_head.weight` on a tied checkpoint, so an
//! `lm_head` pair used to fuse here while the single-process loaders refused
//! it, and the stage then built its head from `model.embed_tokens` without
//! ever reading the fused copy. These pin that the stage now gives the
//! single-process answer, word for word, and still applies a head pair where
//! the head is genuinely separate.
//!
//! The `model.embed_tokens` parity test (#2332) runs all three load paths, the
//! fused single-process path, the unfused runtime path, and this stage
//! loader, against one tied checkpoint: before the fix the first and last
//! folded a linear-shaped pair into the matrix both ends read while the
//! runtime path applied it to the head alone, and an mlx-lm `LoRAEmbedding`
//! pair failed with a different shape message on each.

use std::path::{Path, PathBuf};

use super::*;
use crate::lora::test_support::{
    TINY_HIDDEN, TINY_VOCAB, TinyLlama, adapter_pair_tensors, temp_dir, write_adapter_dir,
    write_tiny_llama_checkpoint,
};

fn checkpoint(name: &str, tie_word_embeddings: bool, stores_lm_head: bool) -> PathBuf {
    let dir = temp_dir(name);
    write_tiny_llama_checkpoint(
        &dir,
        &TinyLlama {
            tie_word_embeddings: Some(tie_word_embeddings),
            stores_lm_head,
            unread: Vec::new(),
        },
    );
    dir
}

/// An adapter with an `lm_head` pair and a `q_proj` pair.
fn head_adapter(name: &str) -> PathBuf {
    let dir = temp_dir(name);
    let mut tensors = adapter_pair_tensors("lm_head", TINY_HIDDEN, 2, TINY_VOCAB);
    tensors.extend(adapter_pair_tensors(
        "model.layers.0.self_attn.q_proj",
        TINY_HIDDEN,
        2,
        TINY_HIDDEN,
    ));
    write_adapter_dir(&dir, "lora", 2, tensors);
    dir
}

/// The single (and therefore last) stage of a one-layer model.
fn whole_model_stage() -> LayerFilter {
    LayerFilter {
        layer_range: 0..1,
        has_embedding: true,
        has_lm_head: true,
    }
}

fn cleanup(dirs: &[&Path]) {
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn the_stage_loader_and_the_single_process_loader_agree_on_a_tied_head_pair() {
    let model = checkpoint("stage_tied_model", true, false);
    let adapter = head_adapter("stage_tied_adapter");

    let stage_err = LlamaStageExecutor::load(&model, &whole_model_stage(), 0, Some(&adapter))
        .err()
        .expect("a pipeline stage must refuse an lm_head pair on a tied checkpoint")
        .to_string();
    assert!(
        stage_err.contains("ties its output projection"),
        "{stage_err}"
    );

    let single_err = crate::loading::load_model_with_adapter(&model, &adapter)
        .err()
        .expect("the single-process loader refuses it too")
        .to_string();
    assert_eq!(
        stage_err, single_err,
        "both paths must give the same verdict in the same words"
    );

    cleanup(&[&model, &adapter]);
}

#[test]
fn a_stage_that_does_not_own_the_head_refuses_the_adapter_too() {
    // The verdict is about the adapter and the checkpoint, not the stage's
    // slice: a first stage must not accept an adapter the last stage refuses.
    let model = checkpoint("stage_tied_first_model", true, false);
    let adapter = head_adapter("stage_tied_first_adapter");
    let first_stage = LayerFilter {
        layer_range: 0..1,
        has_embedding: true,
        has_lm_head: false,
    };

    let err = LlamaStageExecutor::load(&model, &first_stage, 0, Some(&adapter))
        .err()
        .expect("every stage refuses the adapter")
        .to_string();
    assert!(err.contains("ties its output projection"), "{err}");

    cleanup(&[&model, &adapter]);
}

#[test]
fn the_stage_loader_applies_a_head_pair_on_an_untied_checkpoint() {
    let model = checkpoint("stage_untied_model", false, true);
    let adapter = head_adapter("stage_untied_adapter");

    LlamaStageExecutor::load(&model, &whole_model_stage(), 0, Some(&adapter))
        .expect("a separately stored head takes its pair on a pipeline stage");

    cleanup(&[&model, &adapter]);
}

/// An adapter with a `model.embed_tokens` pair and a `q_proj` pair. `linear`
/// picks the `[hidden] -> [vocab]` orientation every path reads as a fit for
/// the embedding matrix; otherwise the pair is in mlx-lm's `LoRAEmbedding`
/// layout, `lora_a = [vocab, rank]`, `lora_b = [rank, hidden]`.
fn embedding_adapter(name: &str, linear: bool) -> PathBuf {
    let dir = temp_dir(name);
    let mut tensors = if linear {
        adapter_pair_tensors("model.embed_tokens", TINY_HIDDEN, 2, TINY_VOCAB)
    } else {
        adapter_pair_tensors("model.embed_tokens", TINY_VOCAB, 2, TINY_HIDDEN)
    };
    tensors.extend(adapter_pair_tensors(
        "model.layers.0.self_attn.q_proj",
        TINY_HIDDEN,
        2,
        TINY_HIDDEN,
    ));
    write_adapter_dir(&dir, "lora", 2, tensors);
    dir
}

#[test]
fn every_load_path_gives_the_same_verdict_on_a_tied_embedding_pair() {
    use crate::lora::{LoraAdapterSpec, RuntimeLoraSet};

    let model = checkpoint("stage_tied_embed_model", true, false);
    for linear in [true, false] {
        let adapter = embedding_adapter("stage_tied_embed_adapter", linear);
        let spec = LoraAdapterSpec {
            path: adapter.clone(),
            scale: 1.0,
            apply: true,
        };

        let stage = LlamaStageExecutor::load(&model, &whole_model_stage(), 0, Some(&adapter))
            .err()
            .map(|err| err.to_string());
        let fused = crate::loading::load_model_with_adapter(&model, &adapter)
            .err()
            .map(|err| err.to_string());
        let set =
            RuntimeLoraSet::from_specs(std::slice::from_ref(&spec)).expect("a lora adapter config");
        let runtime = crate::loading::load_model_with_adapter_specs(
            &model,
            std::slice::from_ref(&spec),
            Some(&set),
        )
        .err()
        .map(|err| err.to_string());
        assert!(
            !mlxcel_core::runtime_lora::has_pending(),
            "a refused runtime load leaves no staged term behind"
        );

        let layout = if linear { "linear" } else { "mlx-lm embedding" };
        let stage = stage.unwrap_or_else(|| {
            panic!(
                "the pipeline stage must refuse a {layout} embed_tokens pair on a tied checkpoint"
            )
        });
        assert!(stage.contains("model.embed_tokens"), "{stage}");
        assert!(stage.contains("input-embedding LoRA"), "{stage}");
        assert_eq!(
            Some(&stage),
            fused.as_ref(),
            "the fused single-process path must give the stage's verdict ({layout} layout)"
        );
        assert_eq!(
            Some(&stage),
            runtime.as_ref(),
            "the unfused runtime path must give the stage's verdict ({layout} layout)"
        );

        cleanup(&[&adapter]);
    }
    cleanup(&[&model]);
}
