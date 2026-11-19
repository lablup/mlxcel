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

//! Parity gates for the families #3101 opted into
//! `LanguageModel::supports_maskless_padded_prefill` (GLM4V, GLM4V-MoE,
//! Nemotron-NAS, Cohere2-MoE, EXAONE-MoE).
//!
//! Each gate prefills a real prompt three ways and then decodes one token:
//!
//! * unpadded, the reference;
//! * padded to the tile with no mask, the opted-in path;
//! * padded with `create_padded_prefill_mask`, the path before the opt-in.
//!
//! The pad rows are trimmed from the cache between the prefill and the
//! decode, as the engine does. The maskless arm must agree with the masked
//! arm to f32 rounding on both the last real row and the decode row, and
//! both must agree with the unpadded reference.

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Which of the three prefills an arm runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arm {
    Unpadded,
    Maskless,
    Masked,
}

/// What an arm returns: the last real row of the prefill logits and the
/// logits of the decode step that follows the pad trim.
pub(crate) struct ArmRows {
    pub prefill: Vec<f32>,
    pub decode: Vec<f32>,
}

/// Serialize MLX forwards across the gates of this module.
pub(crate) fn guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A deterministic prompt of `len` tokens inside `vocab`.
pub(crate) fn prompt(len: usize, vocab: i32) -> Vec<i32> {
    (0..len as i32)
        .map(|i| 1 + (i * 7 + 3) % (vocab - 1))
        .collect()
}

/// The `[vocab]` logits at sequence position `pos` of a `[1, L, V]` output.
pub(crate) fn row_at(logits: &MlxArray, pos: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let row = mlxcel_core::slice(logits, &[0, pos, 0], &[1, pos + 1, shape[2]]);
    let row = mlxcel_core::astype(&row, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&row);
    mlxcel_core::array_to_raw_bytes(&row)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// RMS of `a - b` divided by the RMS of `b`.
pub(crate) fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "logit row length mismatch");
    let (mut diff, mut base) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        diff += (x - y) * (x - y);
        base += y * y;
    }
    if base == 0.0 {
        diff.sqrt()
    } else {
        (diff / base).sqrt()
    }
}

/// `tokens` followed by zeros up to `padded`.
fn padded_tokens(tokens: &[i32], padded: usize) -> Vec<i32> {
    let mut out = tokens.to_vec();
    out.resize(padded, 0);
    out
}

/// The input and mask an arm feeds the forward.
pub(crate) fn arm_input(
    arm: Arm,
    tokens: &[i32],
    padded: usize,
) -> (UniquePtr<MlxArray>, Option<UniquePtr<MlxArray>>) {
    match arm {
        Arm::Unpadded => (
            mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]),
            None,
        ),
        Arm::Maskless => {
            let ids = padded_tokens(tokens, padded);
            (mlxcel_core::from_slice_i32(&ids, &[1, padded as i32]), None)
        }
        Arm::Masked => {
            let ids = padded_tokens(tokens, padded);
            let mask = mlxcel_core::utils::create_padded_prefill_mask(
                tokens.len() as i32,
                padded as i32,
                0,
            );
            (
                mlxcel_core::from_slice_i32(&ids, &[1, padded as i32]),
                Some(mask),
            )
        }
    }
}

/// One arm over a model whose K/V lives in the caller's `KVCache` slice.
/// `embeddings` builds the pre-merged embedding input for a token slice (the
/// VLM prefill); `None` forwards token ids.
pub(crate) fn standard_arm<M: LanguageModel>(
    model: &M,
    arm: Arm,
    tokens: &[i32],
    padded: usize,
    decode_token: i32,
    embeddings: Option<&dyn Fn(&MlxArray) -> UniquePtr<MlxArray>>,
) -> ArmRows {
    let mut caches: Vec<KVCache> = model.make_caches();
    let (ids, mask) = arm_input(arm, tokens, padded);
    let embeds = embeddings.map(|build| build(&ids));
    let logits =
        model.forward_with_embeddings(&ids, embeds.as_deref(), &mut caches, mask.as_deref());
    let prefill = row_at(&logits, tokens.len() as i32 - 1);
    let excess = (padded - tokens.len()) as i32;
    if arm != Arm::Unpadded {
        for (layer, cache) in caches.iter_mut().enumerate() {
            let trimmed = cache.trim(excess);
            assert_eq!(trimmed, excess, "layer {layer} pad trim");
        }
    }
    let next = mlxcel_core::from_slice_i32(&[decode_token], &[1, 1]);
    let decode = model.forward(&next, &mut caches, None);
    ArmRows {
        prefill,
        decode: row_at(&decode, 0),
    }
}

/// Run every arm and assert the maskless one matches the masked one and the
/// unpadded reference, on the prefill row and the decode row.
pub(crate) fn assert_arms_agree(who: &str, run: impl FnMut(Arm) -> ArmRows) {
    assert_arms_agree_with(who, true, run);
}

/// [`assert_arms_agree`], optionally leaving the padding-mask arm out of the
/// assertions. `create_padded_prefill_mask` is plain causal, so a family
/// that applies a caller mask to its sliding layers gets no window from it;
/// past the window that arm differs from the unpadded reference, which the
/// maskless arm (the model's own windowed mask) matches exactly.
pub(crate) fn assert_arms_agree_with(
    who: &str,
    masked_arm_is_window_exact: bool,
    mut run: impl FnMut(Arm) -> ArmRows,
) {
    let reference = run(Arm::Unpadded);
    let maskless = run(Arm::Maskless);
    let masked = run(Arm::Masked);

    let against = |rows: &ArmRows| {
        (
            relative_rms(&rows.prefill, &reference.prefill),
            relative_rms(&rows.decode, &reference.decode),
        )
    };
    let (ml_p, ml_d) = against(&maskless);
    let (mk_p, mk_d) = against(&masked);
    let cross_p = relative_rms(&maskless.prefill, &masked.prefill);
    let cross_d = relative_rms(&maskless.decode, &masked.decode);
    eprintln!(
        "{who}: maskless vs masked prefill {cross_p:.3e} decode {cross_d:.3e}; \
         vs unpadded: maskless {ml_p:.3e}/{ml_d:.3e}, masked {mk_p:.3e}/{mk_d:.3e}"
    );
    let mut checks = vec![
        ("maskless vs unpadded, prefill row", ml_p),
        ("maskless vs unpadded, decode row", ml_d),
    ];
    if masked_arm_is_window_exact {
        checks.extend([
            ("maskless vs masked, prefill row", cross_p),
            ("maskless vs masked, decode row", cross_d),
            ("masked vs unpadded, prefill row", mk_p),
            ("masked vs unpadded, decode row", mk_d),
        ]);
    }
    for (label, value) in checks {
        assert!(value < 1e-4, "{who}: {label} relative RMS {value:.3e}");
    }
}

// GLM4V family: the MRoPE positions of a padded row.

/// MRoPE positions `[3, 1, len]` of a prompt whose tokens 4..8 are a 2x2
/// image (temporal 4, a 2x2 spatial grid) and whose text continues after it,
/// with the `rope_deltas` the text model stores beside them.
fn image_positions(len: usize) -> (UniquePtr<MlxArray>, i32) {
    let triples: Vec<(i32, i32, i32)> = (0..len)
        .map(|i| match i {
            4..=7 => (4, 4 + ((i - 4) / 2) as i32, 4 + ((i - 4) % 2) as i32),
            _ if i > 7 => (i as i32 - 2, i as i32 - 2, i as i32 - 2),
            _ => (i as i32, i as i32, i as i32),
        })
        .collect();
    let next = triples
        .iter()
        .map(|&(t, h, w)| t.max(h).max(w))
        .max()
        .unwrap_or(0)
        + 1;
    let mut flat: Vec<i32> = triples.iter().map(|p| p.0).collect();
    flat.extend(triples.iter().map(|p| p.1));
    flat.extend(triples.iter().map(|p| p.2));
    (
        mlxcel_core::from_slice_i32(&flat, &[3, 1, len as i32]),
        next - len as i32,
    )
}

fn glm_embeddings<'a, M: LanguageModel>(
    model: &'a M,
) -> impl Fn(&MlxArray) -> UniquePtr<MlxArray> + 'a {
    move |ids| model.embed_tokens(ids).expect("GLM4V embeds tokens")
}

#[test]
fn glm4v_maskless_padded_prefill_matches_masked_with_and_without_image_positions() {
    let _guard = guard();
    let model = crate::models::vlm_family_batched_decode_test_support::tiny_glm4v_model();
    assert!(model.supports_maskless_padded_prefill());
    let tokens = prompt(10, 32);
    for with_image in [false, true] {
        for embeds in [false, true] {
            let build = glm_embeddings(&model);
            assert_arms_agree(
                &format!("glm4v image={with_image} embeds={embeds}"),
                |arm| {
                    model.clear_mrope_state();
                    if with_image {
                        let (positions, delta) = image_positions(tokens.len());
                        model.set_mrope_state(positions, delta);
                    }
                    let embed: Option<&dyn Fn(&MlxArray) -> UniquePtr<MlxArray>> =
                        embeds.then_some(&build);
                    standard_arm(&model, arm, &tokens, 16, 5, embed)
                },
            );
        }
    }
}

#[test]
fn glm4v_moe_maskless_padded_prefill_matches_masked_with_and_without_image_positions() {
    let _guard = guard();
    let model = crate::models::vlm_family_batched_decode_test_support::tiny_glm4v_moe_model();
    assert!(model.supports_maskless_padded_prefill());
    let tokens = prompt(10, 32);
    for with_image in [false, true] {
        for embeds in [false, true] {
            let build = glm_embeddings(&model);
            assert_arms_agree(
                &format!("glm4v_moe image={with_image} embeds={embeds}"),
                |arm| {
                    model.clear_mrope_state();
                    if with_image {
                        let (positions, delta) = image_positions(tokens.len());
                        model.set_mrope_state(positions, delta);
                    }
                    let embed: Option<&dyn Fn(&MlxArray) -> UniquePtr<MlxArray>> =
                        embeds.then_some(&build);
                    standard_arm(&model, arm, &tokens, 16, 5, embed)
                },
            );
        }
    }
}

// Nemotron-NAS: the model-owned caches.

#[test]
fn nemotron_nas_maskless_padded_prefill_matches_masked_through_model_owned_trim() {
    use crate::models::nemotron_nas::engine_tests::tiny_model;
    let _guard = guard();
    let model = tiny_model();
    assert!(model.supports_padded_prefill());
    assert!(model.supports_maskless_padded_prefill());
    let tokens = prompt(10, 32);
    assert_arms_agree("nemotron_nas", |arm| {
        // `make_caches` resets the model-owned slot; the slice is never read.
        let _ = model.make_caches();
        let (ids, mask) = arm_input(arm, &tokens, 16);
        let logits = model.forward(&ids, &mut [], mask.as_deref());
        let prefill = row_at(&logits, tokens.len() as i32 - 1);
        if arm != Arm::Unpadded {
            model
                .trim_state(None, (16 - tokens.len()) as i32)
                .expect("trim the model-owned slot");
            assert_eq!(model.slot_live_len(), tokens.len() as i32);
        }
        let next = mlxcel_core::from_slice_i32(&[5], &[1, 1]);
        let decode = model.forward(&next, &mut [], None);
        ArmRows {
            prefill,
            decode: row_at(&decode, 0),
        }
    });
}
