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

//! Unit tests of the Muse Glimmer assistant drafter (issue #1343), on a
//! checkpoint-free synthetic model.

use super::*;
use crate::cache::KVCache;
use crate::drafter::{Drafter, DrafterError};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::{Embedding, Linear, UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

const HIDDEN: usize = 4;
const VOCAB: usize = 8;
const TARGET_LAYERS: usize = 2;
const MASK_ID: i32 = 7;
const WINDOW: usize = 8;

fn config_json(window: usize) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
        "model_type": "muse_glimmer_assistant",
        "hidden_size": HIDDEN,
        "intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1e-5,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "sliding_window": window,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "block_size": 4,
        "mask_token_id": MASK_ID,
        "target_layer_ids": [0, 1],
    })
}

fn tiny_config(window: usize) -> MuseAssistantConfig {
    MuseAssistantConfig::from_json(&config_json(window)).expect("tiny config must validate")
}

/// Deterministic, non-constant fill so no two rows or columns coincide.
fn varied(shape: &[i32], seed: usize) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let values: Vec<f32> = (0..len)
        .map(|i| (((i * 7919 + seed * 104_729) % 211) as f32 / 211.0 - 0.5) * 0.4)
        .collect();
    ffi::from_slice_f32(&values, shape)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    ffi::from_slice_f32(&vec![1.0; len], shape)
}

fn tiny_weights(config: &MuseAssistantConfig) -> WeightMap {
    let h = HIDDEN as i32;
    let inter = config.intermediate_size as i32;
    let q_out = (config.num_attention_heads * config.head_dim) as i32;
    let kv_out = (config.num_key_value_heads * config.head_dim) as i32;
    let hd = config.head_dim as i32;
    let mut w: WeightMap = WeightMap::new();
    w.insert(
        "encoder.fc.weight".to_string(),
        varied(&[h, (config.target_layer_ids.len() as i32) * h], 1),
    );
    w.insert("encoder.output_norm_enc.weight".to_string(), ones(&[h]));
    w.insert("norm.weight".to_string(), ones(&[h]));
    for i in 0..config.num_hidden_layers {
        let p = format!("layers.{i}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[h]));
        w.insert(format!("{p}.post_attention_layernorm.weight"), ones(&[h]));
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            varied(&[q_out, h], 10 + i),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            varied(&[kv_out, h], 20 + i),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            varied(&[kv_out, h], 30 + i),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            varied(&[h, q_out], 40 + i),
        );
        w.insert(format!("{p}.self_attn.q_norm.weight"), ones(&[hd]));
        w.insert(format!("{p}.self_attn.k_norm.weight"), ones(&[hd]));
        w.insert(
            format!("{p}.mlp.gate_proj.weight"),
            varied(&[inter, h], 50 + i),
        );
        w.insert(
            format!("{p}.mlp.up_proj.weight"),
            varied(&[inter, h], 60 + i),
        );
        w.insert(
            format!("{p}.mlp.down_proj.weight"),
            varied(&[h, inter], 70 + i),
        );
    }
    w
}

fn tiny_model(window: usize) -> MuseAssistantModel {
    let config = tiny_config(window);
    let weights = tiny_weights(&config);
    MuseAssistantModel::from_weights(&weights, config).expect("synthetic drafter must build")
}

/// A target stand-in: a raw embedding table and an untied head of the
/// drafter's width, plus the layer count the pairing check reads.
struct StubTarget {
    hidden: usize,
    layers: usize,
    with_head: bool,
}

impl StubTarget {
    fn muse() -> Self {
        Self {
            hidden: HIDDEN,
            layers: TARGET_LAYERS,
            with_head: true,
        }
    }
}

impl LanguageModel for StubTarget {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ffi::zeros(&[1, 1, VOCAB as i32], crate::dtype::FLOAT32)
    }
    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }
    fn num_layers(&self) -> usize {
        self.layers
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.embed_tokens_module().map(|e| e.forward(input_ids))
    }
    fn embed_tokens_module(&self) -> Option<UnifiedEmbedding> {
        Some(UnifiedEmbedding::Regular(Embedding::new(varied(
            &[VOCAB as i32, self.hidden as i32],
            99,
        ))))
    }
    fn lm_head_module(&self) -> Option<UnifiedLinear> {
        self.with_head.then(|| {
            UnifiedLinear::Regular(Linear::new(
                varied(&[VOCAB as i32, self.hidden as i32], 98),
                None,
            ))
        })
    }
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

fn hidden_rows(rows: i32) -> UniquePtr<MlxArray> {
    varied(&[1, rows, (TARGET_LAYERS * HIDDEN) as i32], 5)
}

fn mask_rows(mask: &MlxArray) -> Vec<Vec<bool>> {
    let shape = ffi::array_shape(mask);
    let bytes = ffi::array_to_raw_bytes(mask);
    bytes
        .chunks_exact(shape[1] as usize)
        .map(|row| row.iter().map(|b| *b != 0).collect())
        .collect()
}

/// The 3x6 reference: queries at 10..13 over keys at 7..13 with window 2,
/// `|q - k| <= window`, bidirectional (a later key is visible).
#[test]
fn bidirectional_sliding_mask_matches_reference_rows() {
    let mask = bidirectional_sliding_mask_bool(10, 3, 7, 6, 2);
    assert_eq!(
        mask_rows(&mask),
        vec![
            vec![false, true, true, true, true, true],
            vec![false, false, true, true, true, true],
            vec![false, false, false, true, true, true],
        ]
    );
    // The additive form keeps the same shape, batched for SDPA.
    let bias = bidirectional_sliding_mask(10, 3, 7, 6, 2, crate::dtype::FLOAT32);
    assert_eq!(ffi::array_shape(&bias), vec![1, 1, 3, 6]);
}

#[test]
fn config_flattens_dflash_config_and_validates_layer_types() {
    let nested = serde_json::json!({
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "dflash_config": {
            "mask_token_id": 5, "target_layer_ids": [1, 3], "block_size": 6,
            "runtime_block_size": 4, "num_target_layers": 4,
        },
    });
    let config = MuseAssistantConfig::from_json(&nested).expect("nested form must parse");
    assert_eq!(config.mask_token_id, 5);
    assert_eq!(config.target_layer_ids, vec![1, 3]);
    assert_eq!(config.block_size, 6);
    assert_eq!(config.runtime_block_size, Some(4));
    assert_eq!(config.num_target_layers, Some(4));
    assert_eq!(config.runtime_verify_width(), 4);

    let mut full = nested.clone();
    full["layer_types"][1] = serde_json::Value::String("full_attention".to_string());
    let err = MuseAssistantConfig::from_json(&full).expect_err("a full layer must be refused");
    assert!(err.contains("sliding_attention"), "{err}");

    let mut past = nested.clone();
    past["dflash_config"]["num_target_layers"] = serde_json::json!(3);
    let err =
        MuseAssistantConfig::from_json(&past).expect_err("ids past the target must be refused");
    assert!(err.contains("reach past"), "{err}");

    let mut unordered = nested.clone();
    unordered["dflash_config"]["target_layer_ids"] = serde_json::json!([3, 1]);
    assert!(MuseAssistantConfig::from_json(&unordered).is_err());
}

/// The published checkpoint's layout: flat keys, no `num_target_layers`, no
/// `vocab_size`, no `dflash_config`. Defaulting the two absent keys to zero
/// and validating against them would reject this file.
#[test]
fn config_accepts_the_published_flat_layout() {
    let published = serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
        "block_size": 16, "head_dim": 128, "hidden_size": 6656,
        "intermediate_size": 19968,
        "layer_types": vec!["sliding_attention"; 5],
        "mask_token_id": 201818, "max_position_embeddings": 131072,
        "model_type": "muse_glimmer_assistant", "num_attention_heads": 32,
        "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "sliding_window": 2048, "target_layer_ids": [1, 13, 25, 37, 49],
    });
    let config = MuseAssistantConfig::from_json(&published).expect("published config must parse");
    assert_eq!(config.num_target_layers, None);
    assert_eq!(config.vocab_size, None);
    assert_eq!(config.runtime_verify_width(), 16);
    assert_eq!(config.rope_theta(), 500_000.0);
    assert!(is_muse_assistant_config(&published));
    assert!(is_muse_assistant_config(&serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
    })));
    assert!(!is_muse_assistant_config(&serde_json::json!({
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
        "dflash_config": {"mask_token_id": 1},
    })));
    assert!(!is_muse_assistant_config(&serde_json::json!({
        "model_type": "muse_glimmer",
    })));
}

#[test]
fn configured_block_size_peek_reads_the_runtime_minimum() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut json = config_json(WINDOW);
    json["block_size"] = serde_json::json!(16);
    json["runtime_block_size"] = serde_json::json!(8);
    std::fs::write(dir.path().join("config.json"), json.to_string()).unwrap();
    assert!(is_muse_assistant_dir(dir.path()));
    assert_eq!(
        peek_muse_assistant_configured_block_size(dir.path()),
        Some(8)
    );

    let other = tempfile::tempdir().expect("temp dir");
    std::fs::write(
        other.path().join("config.json"),
        r#"{"model_type": "qwen3", "dflash_config": {"mask_token_id": 1}}"#,
    )
    .unwrap();
    assert!(!is_muse_assistant_dir(other.path()));
    assert_eq!(
        peek_muse_assistant_configured_block_size(other.path()),
        None
    );
}

/// After one draft over a 3-row prompt hidden every drafter cache holds
/// exactly the 3 context rows: the proposal K/V never enters the cache.
#[test]
fn draft_block_caches_context_rows_only() {
    let mut drafter = MuseAssistantDrafter::from_model(tiny_model(WINDOW));
    let target = StubTarget::muse();
    drafter.bind(&target).expect("bind against the stub target");

    let proposals = drafter
        .draft_block(1, Some(&hidden_rows(3)), 4, &greedy())
        .expect("draft");
    assert_eq!(proposals.len(), 3, "block 4 proposes 3 tokens");
    assert!(proposals.iter().all(|t| (0..VOCAB as i32).contains(t)));
    for cache in drafter.caches() {
        assert_eq!(cache.offset, 3);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.key_start(), 0);
    }

    // A partial-accept round hands over `accepted + 1` rows; they append.
    let arr = drafter
        .draft_block_array(2, Some(&hidden_rows(2)), 4, &greedy())
        .expect("draft array");
    assert_eq!(ffi::array_shape(&arr), vec![1, 3]);
    for cache in drafter.caches() {
        assert_eq!(cache.offset, 5);
        assert_eq!(cache.len(), 5);
    }

    // Reset empties the windows without touching the binding.
    drafter.reset(&target).expect("reset");
    assert!(
        drafter
            .caches()
            .iter()
            .all(|c| c.offset == 0 && c.is_empty())
    );
    assert!(drafter.is_bound());
}

/// A prompt longer than the window keeps the last `window` rows: the caches
/// record every row consumed (`offset == S`) but hold `window` rows.
#[test]
fn prompt_longer_than_window_skips_oldest_context() {
    let window = 2048;
    let mut drafter = MuseAssistantDrafter::from_model(tiny_model(window));
    drafter.bind(&StubTarget::muse()).expect("bind");
    let s = 2050;
    let proposals = drafter
        .draft_block(1, Some(&hidden_rows(s)), 4, &greedy())
        .expect("draft over an over-window prompt");
    assert_eq!(proposals.len(), 3);
    for cache in drafter.caches() {
        assert_eq!(cache.offset, s);
        assert_eq!(cache.len(), window as i32);
        assert_eq!(cache.key_start(), s - window as i32);
    }
    // The next round appends past the window and the window slides.
    drafter
        .draft_block(1, Some(&hidden_rows(3)), 4, &greedy())
        .expect("draft after the window filled");
    for cache in drafter.caches() {
        assert_eq!(cache.offset, s + 3);
        assert_eq!(cache.len(), window as i32);
        assert_eq!(cache.key_start(), s + 3 - window as i32);
    }
}

/// The published checkpoint ships neither table: both are tombstones until
/// `bind`, and a target that cannot hand out an untied head is refused.
#[test]
fn weights_without_embed_or_lm_head_bind_from_target() {
    let model = tiny_model(WINDOW);
    assert!(model.needs_embed_binding());
    assert!(model.needs_lm_head_binding());
    assert_eq!(model.fc_in_features(), Some(TARGET_LAYERS * HIDDEN));

    let mut drafter = MuseAssistantDrafter::from_model(model);
    assert!(!drafter.is_bound());
    let unbound = drafter.draft_block(1, Some(&hidden_rows(2)), 4, &greedy());
    assert!(matches!(unbound, Err(DrafterError::DraftFailed { .. })));

    let headless = StubTarget {
        hidden: HIDDEN,
        layers: TARGET_LAYERS,
        with_head: false,
    };
    let err = drafter
        .validate_target_compat(&headless)
        .expect_err("a target with no untied head cannot be bound");
    assert!(matches!(err, DrafterError::BindFailed { .. }), "{err}");
    assert!(matches!(
        drafter.bind(&headless),
        Err(DrafterError::BindFailed { .. })
    ));

    let target = StubTarget::muse();
    drafter
        .validate_target_compat(&target)
        .expect("the stub pairs");
    drafter.bind(&target).expect("bind installs both tables");
    assert!(!drafter.model.needs_embed_binding());
    assert!(!drafter.model.needs_lm_head_binding());
    assert_eq!(drafter.dflash_target_layer_ids(), Some(&[0usize, 1][..]));
    assert_eq!(drafter.configured_block_size(), Some(4));
    assert!(drafter.is_muse_assistant());
    assert!(!drafter.is_dspark());
    assert!(!drafter.prefer_requested_block_size());
    assert_eq!(drafter.kind(), crate::drafter::DrafterKind::Dflash);
    let proposals = drafter
        .draft_block(1, Some(&hidden_rows(2)), 4, &greedy())
        .expect("bound drafter drafts");
    assert_eq!(proposals.len(), 3);
}

#[test]
fn validate_target_compat_rejects_a_target_of_another_width() {
    let drafter = MuseAssistantDrafter::from_model(tiny_model(WINDOW));
    let wide = StubTarget {
        hidden: 6,
        layers: TARGET_LAYERS,
        with_head: true,
    };
    let err = drafter
        .validate_target_compat(&wide)
        .expect_err("hidden 6 is not the drafter's 4");
    assert!(err.to_string().contains("hidden_size = 4"), "{err}");

    let shallow = StubTarget {
        hidden: HIDDEN,
        layers: 1,
        with_head: true,
    };
    let err = drafter
        .validate_target_compat(&shallow)
        .expect_err("target_layer_ids [0, 1] need two target layers");
    assert!(err.to_string().contains("reach past"), "{err}");
}

/// The block forward is one non-causal pass: the same block drafted twice
/// from the same state gives the same proposals, and a context row that
/// differs changes them (the drafter does read its context).
#[test]
fn draft_is_deterministic_and_reads_the_context() {
    let target = StubTarget::muse();
    let draft = |rows: &MlxArray| {
        let mut drafter = MuseAssistantDrafter::from_model(tiny_model(WINDOW));
        drafter.bind(&target).expect("bind");
        drafter
            .draft_block(1, Some(rows), 4, &greedy())
            .expect("draft")
    };
    let rows = hidden_rows(3);
    assert_eq!(draft(&rows), draft(&rows));
    // `sanitize` strips a `model.` prefix so a re-exported checkpoint loads.
    let mut weights: WeightMap = WeightMap::new();
    weights.insert("model.norm.weight".to_string(), ones(&[HIDDEN as i32]));
    MuseAssistantModel::sanitize(&mut weights);
    assert!(weights.contains_key("norm.weight"));
    assert!(!weights.contains_key("model.norm.weight"));
}
