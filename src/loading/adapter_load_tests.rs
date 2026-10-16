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

//! Single-process adapter load tests against a real (tiny) Llama checkpoint
//! (issue #1577).
//!
//! Which staged runtime terms get claimed is a property of the family's real
//! constructors, so these build the model through `load_model_from_weights`
//! rather than a synthetic weight map: an unclaimed term must fail the load
//! with every layer named, a fully claimed adapter must still load, and the
//! staging slot must be empty afterwards on every outcome. The tied output-head
//! verdict, for a head pair (#1577) and for an embedding pair (#2332), is
//! pinned on each single-process entry point here; the pipeline stage's copy
//! of it, and the three-path parity for the embedding pair, are pinned in the
//! Llama stage executor tests.

use std::path::{Path, PathBuf};

use super::*;
use crate::lora::test_support::{
    TINY_HIDDEN, TINY_VOCAB, TinyLlama, adapter_pair_tensors, temp_dir, write_adapter_dir,
    write_tiny_llama_checkpoint,
};
use crate::lora::{LoraAdapterSpec, RuntimeLoraSet};

/// An untied checkpoint whose constructors claim every projection, plus the
/// extra `[rows, cols]` tensors in `unread`, which no constructor reads.
fn untied_checkpoint(name: &str, unread: Vec<(&'static str, usize, usize)>) -> PathBuf {
    let dir = temp_dir(name);
    write_tiny_llama_checkpoint(
        &dir,
        &TinyLlama {
            tie_word_embeddings: Some(false),
            stores_lm_head: true,
            unread,
        },
    );
    dir
}

/// An adapter with one pair per `(layer, in, out)`.
fn adapter(name: &str, layers: &[(&str, usize, usize)]) -> PathBuf {
    let dir = temp_dir(name);
    let mut tensors = std::collections::HashMap::new();
    for (layer, in_features, out_features) in layers {
        tensors.extend(adapter_pair_tensors(layer, *in_features, 2, *out_features));
    }
    write_adapter_dir(&dir, "lora", 2, tensors);
    dir
}

fn spec(path: &Path) -> LoraAdapterSpec {
    LoraAdapterSpec {
        path: path.to_path_buf(),
        scale: 1.0,
        apply: true,
    }
}

/// Run the unfused runtime load the server's `--lora` path runs, up to (not
/// including) the tokenizer, which these fixtures do not carry.
fn runtime_load(model: &Path, adapter_dir: &Path) -> Result<LoadedModel> {
    let set = RuntimeLoraSet::from_specs(&[spec(adapter_dir)]).expect("a lora adapter config");
    let mut weights = mlxcel_core::weights::load_weights_from_dir(model).expect("tiny weights");
    load_runtime_adapted_model(model, &mut weights, &set)
}

fn cleanup(dirs: &[&Path]) {
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

const H: usize = TINY_HIDDEN;

#[test]
fn an_unclaimed_runtime_term_fails_the_load_with_its_layer_named() {
    // `model.embed_tokens` is built by `UnifiedEmbedding`, which does not
    // claim: before #1577 this loaded, logged a warning, and served the base
    // embedding under the adapter's name.
    let model = untied_checkpoint("unclaimed_one_model", Vec::new());
    let adapter_dir = adapter(
        "unclaimed_one_adapter",
        &[
            ("model.embed_tokens", H, TINY_VOCAB),
            ("model.layers.0.self_attn.q_proj", H, H),
        ],
    );

    let err = runtime_load(&model, &adapter_dir)
        .err()
        .expect("an adapter term no constructor claims must fail the load")
        .to_string();
    assert!(err.contains("model.embed_tokens"), "{err}");
    assert!(err.contains("not claimed"), "{err}");
    assert!(
        !err.contains("q_proj"),
        "a claimed layer must not be reported: {err}"
    );
    assert!(
        !mlxcel_core::runtime_lora::has_pending(),
        "the staging slot must be drained after a refused load"
    );

    cleanup(&[&model, &adapter_dir]);
}

#[test]
fn several_unclaimed_runtime_terms_are_reported_together_in_sorted_order() {
    let model = untied_checkpoint(
        "unclaimed_many_model",
        vec![
            ("model.zz_unread.weight", H, H),
            ("model.aa_unread.weight", H, H),
        ],
    );
    let adapter_dir = adapter(
        "unclaimed_many_adapter",
        &[
            ("model.zz_unread", H, H),
            ("model.embed_tokens", H, TINY_VOCAB),
            ("model.aa_unread", H, H),
            ("model.layers.0.mlp.down_proj", 16, H),
        ],
    );

    let err = runtime_load(&model, &adapter_dir)
        .err()
        .expect("every unclaimed term fails the same load")
        .to_string();
    assert!(err.contains("for 3 layers"), "{err}");
    let listed = "  model.aa_unread\n  model.embed_tokens\n  model.zz_unread\n";
    assert!(
        err.contains(listed),
        "the unclaimed layers must be listed one per line, sorted: {err}"
    );
    assert!(!err.contains("down_proj"), "{err}");

    cleanup(&[&model, &adapter_dir]);
}

#[test]
fn a_fully_claimed_runtime_adapter_still_loads() {
    // Positive control: every projection the Llama block builds goes through
    // a claiming constructor, so an adapter over all of them loads and
    // leaves nothing staged.
    let model = untied_checkpoint("claimed_model", Vec::new());
    let adapter_dir = adapter(
        "claimed_adapter",
        &[
            ("model.layers.0.self_attn.q_proj", H, H),
            ("model.layers.0.self_attn.k_proj", H, 4),
            ("model.layers.0.self_attn.v_proj", H, 4),
            ("model.layers.0.self_attn.o_proj", H, H),
            ("model.layers.0.mlp.gate_proj", H, 16),
            ("model.layers.0.mlp.up_proj", H, 16),
            ("model.layers.0.mlp.down_proj", 16, H),
            ("lm_head", H, TINY_VOCAB),
        ],
    );

    runtime_load(&model, &adapter_dir).expect("a fully claimed adapter loads");
    assert!(!mlxcel_core::runtime_lora::has_pending());

    cleanup(&[&model, &adapter_dir]);
}

#[test]
fn a_failed_construction_still_drains_the_staged_terms() {
    // Untied and no stored head: construction fails at `lm_head`. The
    // construction error must win over the unclaimed report, and nothing may
    // be left staged for the next load on this thread.
    let model = temp_dir("construction_fail_model");
    write_tiny_llama_checkpoint(
        &model,
        &TinyLlama {
            tie_word_embeddings: Some(false),
            stores_lm_head: false,
            unread: Vec::new(),
        },
    );
    let adapter_dir = adapter(
        "construction_fail_adapter",
        &[("model.embed_tokens", H, TINY_VOCAB)],
    );

    let err = runtime_load(&model, &adapter_dir)
        .err()
        .expect("a checkpoint without its head cannot construct")
        .to_string();
    assert!(err.contains("lm_head"), "{err}");
    assert!(!err.contains("not claimed"), "{err}");
    assert!(!mlxcel_core::runtime_lora::has_pending());

    cleanup(&[&model, &adapter_dir]);
}

/// A tied checkpoint plus an adapter carrying an `lm_head` pair.
fn tied_head_fixture(name: &str) -> (PathBuf, PathBuf) {
    let model = temp_dir(&format!("{name}_model"));
    write_tiny_llama_checkpoint(
        &model,
        &TinyLlama {
            tie_word_embeddings: Some(true),
            stores_lm_head: false,
            unread: Vec::new(),
        },
    );
    let adapter_dir = adapter(
        &format!("{name}_adapter"),
        &[
            ("lm_head", H, TINY_VOCAB),
            ("model.layers.0.self_attn.q_proj", H, H),
        ],
    );
    (model, adapter_dir)
}

fn assert_tied_head_refusal(err: &str) {
    assert!(err.contains("ties its output projection"), "{err}");
    assert!(err.contains("lm_head"), "{err}");
}

#[test]
fn every_single_process_entry_refuses_a_head_pair_on_a_tied_checkpoint() {
    let (model, adapter_dir) = tied_head_fixture("tied_single");

    let err = load_model_with_adapter(&model, &adapter_dir)
        .err()
        .expect("--adapter refuses it")
        .to_string();
    assert_tied_head_refusal(&err);

    let err = load_model_with_adapter_specs(&model, &[spec(&adapter_dir)], None)
        .err()
        .expect("--lora --lora-fuse refuses it")
        .to_string();
    assert_tied_head_refusal(&err);

    let set = RuntimeLoraSet::from_specs(&[spec(&adapter_dir)]).expect("a lora adapter config");
    let err = load_model_with_adapter_specs(&model, &[spec(&adapter_dir)], Some(&set))
        .err()
        .expect("unfused --lora refuses it")
        .to_string();
    assert_tied_head_refusal(&err);
    assert!(!mlxcel_core::runtime_lora::has_pending());

    cleanup(&[&model, &adapter_dir]);
}

#[test]
fn a_tied_embedding_pair_never_reaches_the_head_on_a_single_process_entry() {
    // #2332: the tiny Llama builds its tied head with
    // `UnifiedLinear::from_weights(.., "model.embed_tokens", ..)`, which claims
    // the staged `model.embed_tokens` term. Before the fix the unfused load
    // therefore succeeded with the pair applied to the logits alone while the
    // embedding lookup served base weights; now every entry refuses it before
    // anything is staged.
    let model = temp_dir("tied_embed_single_model");
    write_tiny_llama_checkpoint(
        &model,
        &TinyLlama {
            tie_word_embeddings: Some(true),
            stores_lm_head: false,
            unread: Vec::new(),
        },
    );
    let adapter_dir = adapter(
        "tied_embed_single_adapter",
        &[
            ("model.embed_tokens", H, TINY_VOCAB),
            ("model.layers.0.self_attn.q_proj", H, H),
        ],
    );
    let assert_refusal = |err: &str| {
        assert!(err.contains("model.embed_tokens"), "{err}");
        assert!(err.contains("ties its output projection"), "{err}");
        assert!(err.contains("input-embedding LoRA"), "{err}");
    };

    let set = RuntimeLoraSet::from_specs(&[spec(&adapter_dir)]).expect("a lora adapter config");
    let err = load_model_with_adapter_specs(&model, &[spec(&adapter_dir)], Some(&set))
        .err()
        .expect("unfused --lora must not hand the pair to the tied head")
        .to_string();
    assert_refusal(&err);
    assert!(
        !mlxcel_core::runtime_lora::has_pending(),
        "the verdict runs before anything is staged"
    );

    let err = load_model_with_adapter(&model, &adapter_dir)
        .err()
        .expect("--adapter refuses it")
        .to_string();
    assert_refusal(&err);

    let err = load_model_with_adapter_specs(&model, &[spec(&adapter_dir)], None)
        .err()
        .expect("--lora --lora-fuse refuses it")
        .to_string();
    assert_refusal(&err);

    cleanup(&[&model, &adapter_dir]);
}
