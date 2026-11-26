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

//! The Raon-1 mini DFlash2 target on the tiny random model (#3436): the
//! captured features are the streams after each tap layer in layer then
//! stream order, a verify block equals the decode chain byte for byte
//! (steered and unsteered), and a rollback after a partial accept leaves
//! the caches a fresh decode of the accepted tokens would have.

use super::super::steering_tests::steered_model;
use super::super::tests::{HIDDEN, ids, tiny_model, values};
use super::super::{Raon1MiniLayerCache, StepMasks};
use super::Raon1MiniModel;
use mlxcel_core::drafter::dflash::SpeculativeTarget;
use mlxcel_core::{MlxArray, UniquePtr};

const STREAMS: usize = 4;
const PROMPT: &[i32] = &[3, 17, 5, 29, 8, 1, 22];
const BLOCK: &[i32] = &[9, 14, 2, 30, 11];

fn bytes(a: &MlxArray) -> Vec<u8> {
    mlxcel_core::eval(a);
    mlxcel_core::array_to_raw_bytes(a)
}

fn row(logits: &MlxArray, r: i32) -> Vec<u8> {
    let s = mlxcel_core::array_shape(logits);
    bytes(&mlxcel_core::slice(logits, &[0, r, 0], &[1, r + 1, s[2]]))
}

/// Logits of `tokens` decoded one at a time after `prompt` on plain caches.
fn decode_chain(model: &Raon1MiniModel, prompt: &[i32], tokens: &[i32]) -> Vec<Vec<u8>> {
    let mut caches = model.make_layer_caches();
    let _ = model.forward_with_caches(&ids(&[prompt]), &mut caches, None);
    tokens
        .iter()
        .map(|t| {
            row(
                &model.forward_with_caches(&ids(&[&[*t]]), &mut caches, None),
                0,
            )
        })
        .collect()
}

#[test]
fn captured_features_are_tap_layer_streams_in_layer_then_stream_order() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let taps = [0usize, 2];
    let mut caches = model.make_speculative_caches(None);
    let out = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut caches, &taps);
    let feats = model.concat_hidden_for_drafter(&out);
    let n = PROMPT.len();
    let width = taps.len() * STREAMS * HIDDEN;
    // Release geometry: 5 taps x 4 streams x 2560 = 51200 per position.
    assert_eq!(
        mlxcel_core::array_shape(&feats),
        vec![1, n as i32, width as i32]
    );
    assert_eq!(
        mlxcel_core::array_shape(&out.logits)[1],
        1,
        "prefill scores the last row only"
    );

    // The same streams, layer by layer, on plain caches.
    let mut plain = model.make_layer_caches();
    let mut streams = super::super::raon_mhc::expand_streams(
        &model.embed_tokens.forward(&ids(&[PROMPT])),
        STREAMS,
    );
    let steer = model.steer_rows(&[model.reasoning_effort_alpha(None)]);
    let mut expected: Vec<Vec<f32>> = Vec::new();
    for (idx, (layer, cache)) in model.layers.iter().zip(plain.iter_mut()).enumerate() {
        streams = layer.forward(&streams, cache, &StepMasks::default(), steer.as_deref());
        if taps.contains(&idx) {
            expected.push(values(&streams)); // [1, n, STREAMS, HIDDEN]
        }
    }
    let got = values(&feats);
    for t in 0..n {
        for (tap, layer_streams) in expected.iter().enumerate() {
            for s in 0..STREAMS {
                for d in 0..HIDDEN {
                    let want = layer_streams[(t * STREAMS + s) * HIDDEN + d];
                    let at = t * width + (tap * STREAMS + s) * HIDDEN + d;
                    assert_eq!(
                        got[at], want,
                        "position {t}, tap {tap}, stream {s}, dim {d}"
                    );
                }
            }
        }
    }
}

fn assert_block_equals_chain(model: &Raon1MiniModel, what: &str) {
    let chain = decode_chain(model, PROMPT, BLOCK);
    let mut caches = model.make_speculative_caches(None);
    let _ = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut caches, &[]);
    let out = model.verify_forward(&ids(&[BLOCK]), &mut caches);
    for (i, want) in chain.iter().enumerate() {
        assert_eq!(
            &row(&out.logits, i as i32),
            want,
            "{what}: verify row {i} vs decode step"
        );
    }
}

#[test]
fn a_verify_block_equals_the_decode_chain_byte_for_byte() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    assert_block_equals_chain(&tiny_model(), "unsteered");
}

#[test]
fn a_steered_verify_block_applies_the_alpha_exactly_as_decode_does() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = steered_model();
    for level in ["low", "high"] {
        model.set_reasoning_effort(None, Some(level)).unwrap();
        assert_block_equals_chain(&model, level);
    }
    // The alpha really is in the verify forward: a different level moves it.
    model.set_reasoning_effort(None, Some("low")).unwrap();
    let low = decode_chain(&model, PROMPT, BLOCK);
    model.set_reasoning_effort(None, Some("max")).unwrap();
    let max = decode_chain(&model, PROMPT, BLOCK);
    assert_ne!(
        low, max,
        "the fixture's steered layer must change the logits"
    );
}

#[test]
fn the_exactness_probe_passes_on_the_tiny_model() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    for width in [3, 5, 7] {
        let verdict = model.probe_block_chain_exactness_at(width, &[8, 40]);
        assert!(verdict.is_equal(), "width {width}: {}", verdict.reason());
    }
}

fn layer_state(caches: &[Raon1MiniLayerCache]) -> Vec<(i32, Vec<u8>, Vec<u8>)> {
    caches
        .iter()
        .map(|c| match c {
            Raon1MiniLayerCache::Delta(d) => (
                d.offset,
                d.conv.as_deref().map(bytes).unwrap_or_default(),
                d.ssm.as_deref().map(bytes).unwrap_or_default(),
            ),
            Raon1MiniLayerCache::Attn(kv) => {
                // The buffers may run past `offset`; only the live prefix counts.
                let live = |a: &Option<UniquePtr<MlxArray>>| {
                    a.as_deref()
                        .map(|a| bytes(&mlxcel_core::utils::slice_axis(a, 2, 0, kv.offset)))
                        .unwrap_or_default()
                };
                (kv.offset, live(&kv.keys), live(&kv.values))
            }
        })
        .collect()
}

#[test]
fn rollback_after_a_partial_accept_equals_a_fresh_decode_of_the_accepted_tokens() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    for accepted in [0usize, 2, BLOCK.len() - 1] {
        let kept = &BLOCK[..accepted + 1];
        let next = 6;

        let mut spec = model.make_speculative_caches(None);
        let _ = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut spec, &[]);
        let out = model.verify_forward(&ids(&[BLOCK]), &mut spec);
        model.rollback_partial(&mut spec, &out, accepted as i32, BLOCK.len() as i32);

        let mut fresh = model.make_layer_caches();
        let _ = model.forward_with_caches(&ids(&[PROMPT]), &mut fresh, None);
        for t in kept {
            let _ = model.forward_with_caches(&ids(&[&[*t]]), &mut fresh, None);
        }

        let spec_state: Vec<_> = spec
            .iter()
            .flat_map(|c| layer_state(std::slice::from_ref(&c.layer)))
            .collect();
        assert_eq!(
            spec_state,
            layer_state(&fresh),
            "accepted {accepted}: every layer's state after rollback"
        );

        // And the next step from the rolled-back state is the fresh one.
        let a = model.verify_forward(&ids(&[&[next, next]]), &mut spec);
        let want = row(
            &model.forward_with_caches(&ids(&[&[next]]), &mut fresh, None),
            0,
        );
        assert_eq!(row(&a.logits, 0), want, "accepted {accepted}: next logits");
    }
}
