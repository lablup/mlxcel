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

//! Falcon-OCR config and runtime-state tests.

use super::*;

/// The shipped `config.json`, verbatim.
const RAW_CONFIG: &str = r#"{
  "architectures": ["FalconOCRForCausalLM"],
  "model_type": "falcon_ocr",
  "torch_dtype": "float32",
  "dim": 768, "n_layers": 22, "n_heads": 16, "head_dim": 64, "n_kv_heads": 8,
  "vocab_size": 65536, "ffn_dim": 2304, "norm_eps": 1e-05,
  "max_seq_len": 8192, "rope_theta": 10000,
  "channel_size": 3, "spatial_patch_size": 16, "temporal_patch_size": 1,
  "eos_id": 11, "img_id": 227,
  "image_cls_token_id": 244,
  "image_reg_1_token_id": 245, "image_reg_2_token_id": 246,
  "image_reg_3_token_id": 247, "image_reg_4_token_id": 248,
  "img_end_id": 230
}"#;

#[test]
fn the_raw_key_scheme_parses() {
    let cfg: FalconOcrConfig = serde_json::from_str(RAW_CONFIG).expect("raw config parses");
    assert_eq!(cfg.dim, 768);
    assert_eq!(cfg.n_layers, 22);
    assert_eq!(cfg.n_heads, 16);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.n_kv_heads(), 8);
    assert_eq!(cfg.ffn_dim, 2304);
    assert_eq!(cfg.vocab_size, 65536);
    assert_eq!(cfg.max_seq_len, 8192);
    assert!((cfg.rope_theta - 10000.0).abs() < 1e-6);
    // 1 * 16 * 16 * 3 == the 768-wide `img_projector` input.
    assert_eq!(cfg.patch_dim(), 768);
}

/// mlx-vlm's `config.py` normalizes the same checkpoint into HF spellings, so a
/// converted checkpoint must load through the same struct.
#[test]
fn the_hf_key_scheme_parses_to_the_same_shape() {
    let hf = r#"{
      "model_type": "falcon_ocr",
      "hidden_size": 768, "num_hidden_layers": 22, "num_attention_heads": 16,
      "head_dim": 64, "num_key_value_heads": 8, "vocab_size": 65536,
      "intermediate_size": 2304, "rms_norm_eps": 1e-05,
      "max_position_embeddings": 8192, "rope_theta": 10000
    }"#;
    let cfg: FalconOcrConfig = serde_json::from_str(hf).expect("hf config parses");
    let raw: FalconOcrConfig = serde_json::from_str(RAW_CONFIG).expect("raw config parses");
    assert_eq!(cfg.dim, raw.dim);
    assert_eq!(cfg.n_layers, raw.n_layers);
    assert_eq!(cfg.ffn_dim, raw.ffn_dim);
    assert_eq!(cfg.n_kv_heads(), raw.n_kv_heads());
    // The image token ids are not in the HF block; the defaults must supply the
    // shipped values or the placeholder scatter silently targets nothing.
    assert_eq!(cfg.img_id, raw.img_id);
    assert_eq!(cfg.image_cls_token_id, raw.image_cls_token_id);
    assert_eq!(cfg.img_end_id, raw.img_end_id);
    assert_eq!(cfg.patch_dim(), raw.patch_dim());
}

#[test]
fn token_ids_carry_the_four_register_tokens_in_order() {
    let cfg: FalconOcrConfig = serde_json::from_str(RAW_CONFIG).unwrap();
    let ids = cfg.token_ids();
    assert_eq!(ids.image_reg_token_ids, [245, 246, 247, 248]);
    assert_eq!(ids.block_prefix(), [244, 245, 246, 247, 248]);
}

#[test]
fn the_sequence_map_wins_over_the_fallback_slot() {
    let state = FalconOcrRuntimeState::default();
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 0, 1],
        pos_hw: None,
        rope_delta: -1,
    });
    let seq = SequenceId::from_raw(7);
    state.bind_to_sequence(seq);

    // A second request writes the fallback slot while `seq` is still decoding.
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 1, 2, 3],
        pos_hw: None,
        rope_delta: 0,
    });

    let bound = state.resolve(Some(seq)).expect("bound entry");
    assert_eq!(
        bound.rope_delta, -1,
        "sequence entry must not see the newer fallback"
    );
    let unbound = state.resolve(None).expect("fallback entry");
    assert_eq!(unbound.rope_delta, 0);
}

#[test]
fn releasing_a_sequence_drops_only_that_entry() {
    let state = FalconOcrRuntimeState::default();
    let (a, b) = (SequenceId::from_raw(1), SequenceId::from_raw(2));

    state.set_current(FalconOcrPrefillState {
        positions: vec![0],
        pos_hw: None,
        rope_delta: -5,
    });
    state.bind_to_sequence(a);
    state.set_current(FalconOcrPrefillState {
        positions: vec![0],
        pos_hw: None,
        rope_delta: -9,
    });
    state.bind_to_sequence(b);

    state.release(a);
    // With no fallback pending and no entry for `a`, resolution yields nothing.
    assert!(state.resolve(Some(a)).is_none());
    assert_eq!(state.resolve(Some(b)).expect("b survives").rope_delta, -9);
}

/// `bind_to_sequence` must *drain* the fallback, otherwise the next text-only
/// request would inherit the image row's negative delta.
#[test]
fn binding_drains_the_fallback_slot() {
    let state = FalconOcrRuntimeState::default();
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 0],
        pos_hw: None,
        rope_delta: -3,
    });
    state.bind_to_sequence(SequenceId::from_raw(4));
    assert!(state.resolve(None).is_none());
}

/// A prefill whose length matches the stashed positions consumes them.
#[test]
fn a_matching_prefill_gets_the_stashed_positions() {
    let state = FalconOcrRuntimeState::default();
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 0, 0, 1],
        pos_hw: None,
        rope_delta: -2,
    });
    let got = state.take_for_prefill(None, 4).expect("matching entry");
    assert_eq!(got.positions, vec![0, 0, 0, 1]);
    // It stays available so the decode steps can read the delta.
    assert_eq!(state.resolve(None).expect("still present").rope_delta, -2);
}

/// The dangerous case: an image turn is followed by a text-only turn. Nothing
/// tells the model a new request began, so a length mismatch at the prefill
/// boundary has to evict the old entry or every decode position is shifted by
/// the previous image's block size.
#[test]
fn a_mismatched_prefill_evicts_the_previous_requests_state() {
    let state = FalconOcrRuntimeState::default();
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 0, 0, 0, 1],
        pos_hw: None,
        rope_delta: -3,
    });
    assert!(state.take_for_prefill(None, 9).is_none());
    assert!(
        state.resolve(None).is_none(),
        "the stale entry must not survive to drive decode positions"
    );
}

#[test]
fn a_mismatched_prefill_evicts_the_sequence_entry_too() {
    let state = FalconOcrRuntimeState::default();
    let seq = SequenceId::from_raw(3);
    state.set_current(FalconOcrPrefillState {
        positions: vec![0, 0, 1],
        pos_hw: None,
        rope_delta: -1,
    });
    state.bind_to_sequence(seq);
    assert!(state.take_for_prefill(Some(seq), 12).is_none());
    assert!(state.resolve(Some(seq)).is_none());
}
