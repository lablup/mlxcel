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
use mlxcel_core::generate::LanguageModel;
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

#[test]
fn a_limited_prefill_captures_the_newest_rows_and_keeps_the_logits() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let taps = [0usize, 2];
    let mut full = model.make_speculative_caches(None);
    let whole = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut full, &taps);
    let whole_feats = model.concat_hidden_for_drafter(&whole);

    let keep = 3usize;
    let mut limited = model.make_speculative_caches(None);
    model.limit_prefill_capture(&mut limited, keep);
    let tail = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut limited, &taps);
    let tail_feats = model.concat_hidden_for_drafter(&tail);

    let n = PROMPT.len() as i32;
    let width = mlxcel_core::array_shape(&whole_feats)[2];
    assert_eq!(
        mlxcel_core::array_shape(&tail_feats),
        vec![1, keep as i32, width]
    );
    let newest = mlxcel_core::slice(&whole_feats, &[0, n - keep as i32, 0], &[1, n, width]);
    assert_eq!(
        bytes(&tail_feats),
        bytes(&newest),
        "newest rows, same bytes"
    );
    assert_eq!(
        bytes(&tail.logits),
        bytes(&whole.logits),
        "logits unchanged"
    );

    // A window at least the prompt's length captures every row.
    let mut wide = model.make_speculative_caches(None);
    model.limit_prefill_capture(&mut wide, PROMPT.len());
    let all = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut wide, &taps);
    assert_eq!(
        bytes(&model.concat_hidden_for_drafter(&all)),
        bytes(&whole_feats)
    );
}

/// The DFlash2 exactness gate on the two NVFP4 exports (issue #3458), on the
/// tiny model quantized into each layout and loaded from disk: the probe
/// gives the same verdict every time it runs, and where it allows a width,
/// a verify block is the decode chain byte for byte (NVFP4 `gather_qmm`
/// experts and quantized dense planes included).
#[test]
fn nvfp4_exports_get_a_deterministic_gate_answer_and_an_allowed_block_is_exact() {
    use super::super::export_tests::{Export, load_pair};
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    for export in [Export::Mlx, Export::ModelOpt] {
        // `None`: a backend without native NVFP4, where the load is refused.
        let Some((model, _twin)) = load_pair(export) else {
            continue;
        };
        let mut allowed = 0;
        for width in [3usize, 5, 7] {
            let verdict = model.probe_block_chain_exactness_at(width, &[8, 40]);
            for _ in 0..2 {
                let again = model.probe_block_chain_exactness_at(width, &[8, 40]);
                assert_eq!(
                    (again.is_equal(), again.reason()),
                    (verdict.is_equal(), verdict.reason()),
                    "{export:?} width {width}: the gate answer must not change between runs"
                );
            }
            if verdict.is_equal() {
                allowed += 1;
            }
        }
        if allowed > 0 {
            assert_block_equals_chain(&model, &format!("{export:?} export"));
        }
        // On Metal MLX's quantized products are per row at these widths, so
        // the gate allows every width; a refusal here would be a regression.
        if mlxcel_core::metal_is_available() {
            assert_eq!(
                allowed, 3,
                "{export:?}: every probed width is exact on Metal"
            );
        }
    }
}

/// The server burst's prefill follows the classic partition (issue #3458):
/// split at a prompt-cache snapshot boundary, its last-row logits and its
/// layer states are those of the classic split prefill bit for bit, and the
/// captured rows are those of the pieces in order, with a capture window
/// still keeping only the newest rows of the whole prompt.
#[test]
fn a_planned_burst_prefill_equals_the_classic_split_prefill() {
    use mlxcel_core::prefill_plan::PrefillSplits;
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = tiny_model();
    let boundary = 4usize;
    let taps = [0usize, 2];

    // Classic decode's prefill of the same partition: the history piece,
    // then the rest with logits for its last row only.
    let id = mlxcel_core::cache::SequenceId::from_raw(34_584);
    model.prepare_sequence_state(id);
    let history = ids(&[&PROMPT[..boundary]]);
    let _ = model.forward_with_sequence_id(&history, Some(id), &mut model.make_caches(), None);
    let rest = model
        .last_row_logits(
            &ids(&[&PROMPT[boundary..]]),
            None,
            Some(id),
            &mut model.make_caches(),
            None,
            PROMPT.len() - boundary - 1,
        )
        .expect("raon1_mini computes last-row logits");
    let want_logits = row(&rest, 0);
    let classic = model
        .sequence_state
        .with_existing_sequence_state(id, |caches| layer_state(caches))
        .expect("the sequence is open");

    let planned = |window: Option<usize>| {
        let mut caches = model.make_speculative_caches(None);
        if let Some(rows) = window {
            model.limit_prefill_capture(&mut caches, rows);
        }
        let out = model.prefill_speculative_planned(
            &ids(&[PROMPT]),
            &mut caches,
            &taps,
            0,
            PrefillSplits::from_points(&[boundary]),
        );
        let state: Vec<_> = caches.into_iter().map(|c| c.layer).collect();
        (out, state)
    };
    let (out, state) = planned(None);
    assert_eq!(mlxcel_core::array_shape(&out.logits)[1], 1, "last row only");
    assert_eq!(row(&out.logits, 0), want_logits, "split prefill logits");
    assert_eq!(layer_state(&state), classic, "split prefill state");
    assert_eq!(out.hidden_states.len(), taps.len());
    for hidden in &out.hidden_states {
        assert_eq!(mlxcel_core::array_shape(hidden)[1], PROMPT.len() as i32);
    }

    // A three-row window spans the boundary's last row and the second piece.
    let (windowed, _) = planned(Some(3));
    for (tail, full) in windowed.hidden_states.iter().zip(&out.hidden_states) {
        assert_eq!(mlxcel_core::array_shape(tail)[1], 3);
        let from = PROMPT.len() as i32 - 3;
        let want = mlxcel_core::utils::slice_axis(full, 1, from, PROMPT.len() as i32);
        assert_eq!(
            bytes(tail),
            bytes(&want),
            "the newest rows of the whole prompt"
        );
    }
    assert_eq!(row(&windowed.logits, 0), want_logits);

    // Without a boundary the plan is one pass, the existing prefill.
    let mut one_pass = model.make_speculative_caches(None);
    let single = model.prefill_speculative_planned(
        &ids(&[PROMPT]),
        &mut one_pass,
        &taps,
        0,
        PrefillSplits::NONE,
    );
    let mut reference = model.make_speculative_caches(None);
    let direct = model.prefill_forward_with_capture_layers(&ids(&[PROMPT]), &mut reference, &taps);
    assert_eq!(bytes(&single.logits), bytes(&direct.logits));
}
