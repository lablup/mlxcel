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

//! DSpark drafter model tests for the Speculators and OpenBMB layouts
//! (issue #3261): a head and embedding the checkpoint ships itself, a
//! reduced draft vocabulary with its `d2t` table, and the two anchor
//! layouts.
//!
//! The fixture is a one-layer drafter whose attention and MLP weights are
//! zero, so the residual stream is the input embedding and a logit row
//! reads `lm_head(norm(embed(token)))`: which token a row's logits pick is
//! then decided by the embedding of that row's INPUT token, which is what
//! makes the anchor layout observable.

use super::*;
use crate::dtype;

const HIDDEN: i32 = 4;
const VOCAB: i32 = 8;
const RANK: i32 = 2;
/// Token whose embedding is `e0`; the head's row for it is on `OUT_ANCHOR`.
const ANCHOR: i32 = 1;
/// Mask token, embedded as `e1`.
const MASK: i32 = 7;
/// Head row that scores `e0`.
const OUT_ANCHOR: i32 = 2;
/// Head row that scores `e1`.
const OUT_MASK: i32 = 3;

fn config() -> DFlashConfig {
    DFlashConfig {
        hidden_size: HIDDEN as usize,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        vocab_size: VOCAB as usize,
        mask_token_id: MASK,
        block_size: 3,
        markov_rank: RANK as usize,
        tie_word_embeddings: false,
        num_target_layers: None,
        target_layer_ids: vec![0, 1],
        ..DFlashConfig::default()
    }
}

/// `[rows, HIDDEN]` table whose row `hot.0` is `e0` and row `hot.1` is `e1`.
fn table(rows: i32, hot: (i32, i32)) -> UniquePtr<MlxArray> {
    let mut data = vec![0.0_f32; (rows * HIDDEN) as usize];
    data[(hot.0 * HIDDEN) as usize] = 1.0;
    data[(hot.1 * HIDDEN + 1) as usize] = 1.0;
    ffi::from_slice_f32(&data, &[rows, HIDDEN])
}

/// Backbone weights for [`config`]: zero attention and MLP (the residual
/// stream passes through), unit norms, a two-tap `fc`.
fn backbone() -> WeightMap {
    let mut w: WeightMap = std::collections::HashMap::new();
    let zeros = |shape: &[i32]| ffi::zeros(shape, dtype::FLOAT32);
    let ones = |shape: &[i32]| ffi::ones(shape, dtype::FLOAT32);
    w.insert("fc.weight".to_string(), zeros(&[HIDDEN, 2 * HIDDEN]));
    w.insert("hidden_norm.weight".to_string(), ones(&[HIDDEN]));
    w.insert("norm.weight".to_string(), ones(&[HIDDEN]));
    for (name, shape) in [
        ("self_attn.q_proj", [4, HIDDEN]),
        ("self_attn.k_proj", [2, HIDDEN]),
        ("self_attn.v_proj", [2, HIDDEN]),
        ("self_attn.o_proj", [HIDDEN, 4]),
        ("mlp.gate_proj", [8, HIDDEN]),
        ("mlp.up_proj", [8, HIDDEN]),
        ("mlp.down_proj", [HIDDEN, 8]),
    ] {
        w.insert(format!("layers.0.{name}.weight"), zeros(&shape));
    }
    for name in ["self_attn.q_norm", "self_attn.k_norm"] {
        w.insert(format!("layers.0.{name}.weight"), ones(&[2]));
    }
    for name in ["input_layernorm", "post_attention_layernorm"] {
        w.insert(format!("layers.0.{name}.weight"), ones(&[HIDDEN]));
    }
    w
}

/// Add an embedding table, a head of `head_rows` rows and the two Markov
/// factors (`markov_w2` at `head_rows`) to [`backbone`].
fn self_contained(head_rows: i32) -> WeightMap {
    let mut w = backbone();
    w.insert(
        "embed_tokens.weight".to_string(),
        table(VOCAB, (ANCHOR, MASK)),
    );
    w.insert(
        "lm_head.weight".to_string(),
        table(head_rows, (OUT_ANCHOR, OUT_MASK)),
    );
    w.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::zeros(&[VOCAB, RANK], dtype::FLOAT32),
    );
    w.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::zeros(&[head_rows, RANK], dtype::FLOAT32),
    );
    w
}

fn context(rows: i32) -> UniquePtr<MlxArray> {
    ffi::zeros(&[1, rows, 2 * HIDDEN], dtype::FLOAT32)
}

fn err_of(result: Result<DFlashDraftModel, String>) -> String {
    match result {
        Ok(_) => panic!("the checkpoint must be refused"),
        Err(e) => e,
    }
}

/// The Speculators shape: the checkpoint ships `embed_tokens` and `lm_head`
/// at full vocabulary and nothing is bound from the target.
#[test]
fn a_shipped_head_is_kept_at_full_vocabulary() {
    let model =
        DFlashDraftModel::from_weights(&self_contained(VOCAB), config()).expect("self-contained");
    assert!(model.is_dspark());
    assert!(!model.needs_embed_binding());
    assert!(!model.needs_lm_head_binding());
    assert!(!model.markov_head().unwrap().remaps_draft_ids());

    let mut cache = model.make_cache();
    let inputs = ffi::from_slice_i32(&[ANCHOR, MASK], &[1, 2]);
    let logits = model.forward(&inputs, &context(3), &mut cache);
    assert_eq!(ffi::array_shape(&logits), vec![1, 2, VOCAB]);
}

/// The OpenBMB shape: no `embed_tokens`, no `lm_head`, untied. Both are
/// tombstones for `bind` to resolve from the target.
#[test]
fn an_untied_checkpoint_without_a_head_waits_for_the_target() {
    let mut weights = backbone();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::zeros(&[VOCAB, RANK], dtype::FLOAT32),
    );
    weights.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::zeros(&[VOCAB, RANK], dtype::FLOAT32),
    );
    let mut model = DFlashDraftModel::from_weights(&weights, config()).expect("lazy-bind DSpark");
    assert!(model.needs_embed_binding());
    assert!(model.needs_lm_head_binding());

    // Bound the way `DFlashDrafter::bind` binds: the target's table and its
    // untied head.
    model.bind_target_embedding(crate::layers::UnifiedEmbedding::Regular(
        crate::layers::Embedding::new(table(VOCAB, (ANCHOR, MASK))),
    ));
    let mut head: WeightMap = std::collections::HashMap::new();
    head.insert(
        "lm_head.weight".to_string(),
        table(VOCAB, (OUT_ANCHOR, OUT_MASK)),
    );
    model.bind_target_lm_head(Some(
        crate::layers::UnifiedLinear::from_weights(&head, "lm_head", 64, 4).expect("target head"),
    ));
    assert!(!model.needs_embed_binding());
    assert!(!model.needs_lm_head_binding());
    let mut cache = model.make_cache();
    assert_eq!(
        model.draft_block_dspark(ANCHOR, &context(2), &mut cache, 4),
        vec![OUT_ANCHOR, OUT_MASK, OUT_MASK]
    );
}

/// `sample_from_anchor` decides whether the anchor row is a proposal. With
/// it (every published checkpoint) the first proposal is read off the anchor
/// row; without it the anchor row is scaffolding and every proposal comes
/// from a mask row.
#[test]
fn the_anchor_row_is_a_proposal_only_with_sample_from_anchor() {
    let model =
        DFlashDraftModel::from_weights(&self_contained(VOCAB), config()).expect("self-contained");
    let mut cache = model.make_cache();
    assert_eq!(
        model.draft_block_dspark(ANCHOR, &context(2), &mut cache, 4),
        vec![OUT_ANCHOR, OUT_MASK, OUT_MASK]
    );
    // Only the context rows entered the cache.
    assert_eq!(cache[0].offset, 2);

    let scaffold = DFlashConfig {
        sample_from_anchor: false,
        ..config()
    };
    let model =
        DFlashDraftModel::from_weights(&self_contained(VOCAB), scaffold).expect("no anchor sample");
    let mut cache = model.make_cache();
    // Same verify width, same proposal count, none of them from the anchor.
    assert_eq!(
        model.draft_block_dspark(ANCHOR, &context(2), &mut cache, 4),
        vec![OUT_MASK, OUT_MASK, OUT_MASK]
    );
    assert_eq!(cache[0].offset, 2);
}

/// Verify width 2 is one proposal: the narrowest block either layout runs.
#[test]
fn a_two_row_verify_block_proposes_one_token() {
    for sample_from_anchor in [true, false] {
        let cfg = DFlashConfig {
            sample_from_anchor,
            ..config()
        };
        let model = DFlashDraftModel::from_weights(&self_contained(VOCAB), cfg).expect("model");
        let mut cache = model.make_cache();
        let want = if sample_from_anchor {
            OUT_ANCHOR
        } else {
            OUT_MASK
        };
        assert_eq!(
            model.draft_block_dspark(ANCHOR, &context(1), &mut cache, 2),
            vec![want],
            "sample_from_anchor {sample_from_anchor}"
        );
    }
}

/// A reduced draft vocabulary: the head and `markov_w2` score 4 ids, and a
/// proposal leaves the drafter as `t + d2t[t]`, a target id.
#[test]
fn a_reduced_draft_vocabulary_maps_proposals_to_target_ids() {
    let draft_vocab = 4;
    let cfg = DFlashConfig {
        draft_vocab_size: Some(draft_vocab as usize),
        ..config()
    };
    let mut weights = self_contained(draft_vocab);
    // Draft ids 0..4 are target ids 1, 3, 5, 7.
    weights.insert("d2t".to_string(), ffi::from_slice_i32(&[1, 2, 3, 4], &[4]));
    // `t2d` is never read, whatever it holds.
    weights.insert("t2d".to_string(), ffi::from_slice_i32(&[9; 8], &[8]));
    let model = DFlashDraftModel::from_weights(&weights, cfg).expect("reduced draft vocabulary");
    assert!(model.markov_head().unwrap().remaps_draft_ids());

    let mut cache = model.make_cache();
    let inputs = ffi::from_slice_i32(&[ANCHOR, MASK, MASK], &[1, 3]);
    let logits = model.forward(&inputs, &context(2), &mut cache);
    assert_eq!(
        ffi::array_shape(&logits),
        vec![1, 3, draft_vocab],
        "the head scores the draft vocabulary"
    );

    let mut cache = model.make_cache();
    // Draft ids OUT_ANCHOR = 2 and OUT_MASK = 3 are target ids 5 and 7.
    assert_eq!(
        model.draft_block_dspark(ANCHOR, &context(2), &mut cache, 4),
        vec![5, 7, 7]
    );
}

/// Every way a reduced-vocabulary checkpoint can be inconsistent is a load
/// error that names the tensor or the field, never a gather past a table.
#[test]
fn an_inconsistent_reduced_vocabulary_checkpoint_is_refused_at_load() {
    let draft_vocab = 4;
    let reduced = || DFlashConfig {
        draft_vocab_size: Some(draft_vocab as usize),
        ..config()
    };
    let d2t = |offsets: &[i32]| ffi::from_slice_i32(offsets, &[offsets.len() as i32]);

    // No `d2t`.
    let err = err_of(DFlashDraftModel::from_weights(
        &self_contained(draft_vocab),
        reduced(),
    ));
    assert!(err.contains("d2t"), "{err}");

    // `d2t` maps past the target vocabulary: draft id 3 + 5 = 8, vocab 8.
    let mut weights = self_contained(draft_vocab);
    weights.insert("d2t".to_string(), d2t(&[1, 2, 3, 5]));
    let err = err_of(DFlashDraftModel::from_weights(&weights, reduced()));
    assert!(err.contains("draft id 3"), "{err}");
    assert!(err.contains("target id 8"), "{err}");

    // `d2t` of the wrong length.
    let mut weights = self_contained(draft_vocab);
    weights.insert("d2t".to_string(), d2t(&[1, 2, 3]));
    let err = err_of(DFlashDraftModel::from_weights(&weights, reduced()));
    assert!(err.contains("3 entries"), "{err}");

    // No head of its own: neither the tied projection nor the target's head
    // scores the draft vocabulary.
    let mut weights = self_contained(draft_vocab);
    weights.remove("lm_head.weight");
    weights.insert("d2t".to_string(), d2t(&[1, 2, 3, 4]));
    let err = err_of(DFlashDraftModel::from_weights(&weights, reduced()));
    assert!(err.contains("lm_head.weight"), "{err}");

    // A head at the target vocabulary under a reduced config.
    let mut weights = self_contained(draft_vocab);
    weights.insert(
        "lm_head.weight".to_string(),
        table(VOCAB, (OUT_ANCHOR, OUT_MASK)),
    );
    weights.insert("d2t".to_string(), d2t(&[1, 2, 3, 4]));
    let err = err_of(DFlashDraftModel::from_weights(&weights, reduced()));
    assert!(err.contains("draft_vocab_size 4"), "{err}");

    // A plain DFlash checkpoint (no Markov head) cannot carry a reduced
    // draft vocabulary: nothing would map its argmax back.
    let plain = DFlashConfig {
        markov_rank: 0,
        draft_vocab_size: Some(draft_vocab as usize),
        ..config()
    };
    let err = err_of(DFlashDraftModel::from_weights(
        &self_contained(draft_vocab),
        plain,
    ));
    assert!(err.contains("only supported on a DSpark drafter"), "{err}");
}

/// A draft vocabulary equal to the target one needs no `d2t`; one that is
/// shipped anyway is validated and, unless it is the identity, applied.
#[test]
fn a_full_draft_vocabulary_needs_no_d2t() {
    let cfg = DFlashConfig {
        draft_vocab_size: Some(VOCAB as usize),
        ..config()
    };
    let model = DFlashDraftModel::from_weights(&self_contained(VOCAB), cfg.clone())
        .expect("draft vocabulary equal to the target one");
    assert!(!model.markov_head().unwrap().remaps_draft_ids());

    let mut identity = self_contained(VOCAB);
    identity.insert("d2t".to_string(), ffi::zeros(&[VOCAB], dtype::INT32));
    let model =
        DFlashDraftModel::from_weights(&identity, cfg.clone()).expect("an identity d2t loads");
    assert!(!model.markov_head().unwrap().remaps_draft_ids());

    let mut bad = self_contained(VOCAB);
    bad.insert(
        "d2t".to_string(),
        ffi::from_slice_i32(&[0, 0, 0, 0, 0, 0, 0, 1], &[VOCAB]),
    );
    let err = err_of(DFlashDraftModel::from_weights(&bad, cfg));
    assert!(err.contains("draft id 7"), "{err}");
}

/// The sliding and causal settings reach the layer they belong to.
#[test]
fn layer_types_reach_the_attention_of_their_own_layer() {
    let mut weights = self_contained(VOCAB);
    // A second layer with the same shapes.
    let second: Vec<(String, UniquePtr<MlxArray>)> = weights
        .iter()
        .filter(|(k, _)| k.starts_with("layers.0."))
        .map(|(k, v)| (k.replace("layers.0.", "layers.1."), ffi::copy(v)))
        .collect();
    weights.extend(second);
    let cfg = DFlashConfig {
        num_hidden_layers: 2,
        layer_types: vec![
            "full_attention".to_string(),
            "sliding_attention".to_string(),
        ],
        sliding_window: Some(3),
        is_causal: None,
        ..config()
    };
    let model = DFlashDraftModel::from_weights(&weights, cfg).expect("two layers");
    assert_eq!(model.layers[0].self_attn.sliding_window, None);
    assert!(!model.layers[0].self_attn.is_causal);
    assert_eq!(model.layers[1].self_attn.sliding_window, Some(3));
    assert!(model.layers[1].self_attn.is_causal);

    // The sliding layer's cache is bounded, the full one's is not.
    let mut cache = model.make_cache();
    let _ = model.draft_block_dspark(ANCHOR, &context(10), &mut cache, 4);
    assert_eq!(cache[0].live_len(), 10);
    assert_eq!(cache[1].live_len(), 2);
    assert_eq!(cache[1].offset, 10);
}
