// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::models::batched_decode_harness::{
    assert_batched_decode_matches_sequential_with_ids, assert_empty_batch_returns_empty_logits,
    assert_single_row_batch_is_single_sequence_forward_with_ids,
};
use crate::models::embedding_test_support::{Rng, mlx_test_guard};
use mlxcel_core::cache::{CachePool, PagedKvLayout};
use mlxcel_core::generate::LanguageModel;

pub(crate) fn tiny(yarn: bool) -> GptOssWrapper {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "gpt_oss", "vocab_size": 32, "hidden_size": 64,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
        "head_dim": 32, "num_key_value_heads": 1, "num_local_experts": 4,
        "num_experts_per_tok": 2, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention"],
        "rope_scaling": if yarn { serde_json::json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32, "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0}) } else { serde_json::Value::Null }
    })).unwrap();
    let mut weights = WeightMap::new();
    let mut rng = Rng::new(2568);
    rng.insert(&mut weights, "model.embed_tokens.weight", &[32, 64], 0.2);
    rng.insert(&mut weights, "lm_head.weight", &[32, 64], 0.1);
    weights.insert(
        "model.norm.weight".into(),
        mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            weights.insert(
                format!("{prefix}.{name}.weight"),
                mlxcel_core::ones(&[64], mlxcel_core::dtype::FLOAT32),
            );
        }
        for (name, output) in [
            ("q_proj", 64),
            ("k_proj", 32),
            ("v_proj", 32),
            ("o_proj", 64),
        ] {
            rng.insert(
                &mut weights,
                &format!("{prefix}.self_attn.{name}.weight"),
                &[output, 64],
                0.08,
            );
            rng.insert(
                &mut weights,
                &format!("{prefix}.self_attn.{name}.bias"),
                &[output],
                0.03,
            );
        }
        weights.insert(
            format!("{prefix}.self_attn.sinks"),
            mlxcel_core::from_slice_f32(&[1.5, -0.75], &[2]),
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.router.weight"),
            &[4, 64],
            0.1,
        );
        rng.insert(
            &mut weights,
            &format!("{prefix}.mlp.router.bias"),
            &[4],
            0.02,
        );
        for name in ["gate_proj", "up_proj", "down_proj"] {
            rng.insert(
                &mut weights,
                &format!("{prefix}.mlp.experts.{name}.weight"),
                &[4, 64, 64],
                0.06,
            );
            rng.insert(
                &mut weights,
                &format!("{prefix}.mlp.experts.{name}.bias"),
                &[4, 64],
                0.02,
            );
        }
    }
    GptOssWrapper::new(GptOssModel::from_weights(&weights, &args).unwrap())
}

fn input(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn close(a: &MlxArray, b: &MlxArray, tolerance: f32) {
    let difference = mlxcel_core::abs(&mlxcel_core::subtract(a, b));
    let maximum = mlxcel_core::max_all(&difference);
    let error = mlxcel_core::item_f32(&maximum);
    assert!(error <= tolerance, "logit error {error} > {tolerance}");
}

const PROMPTS: [&[i32]; 3] = [&[1], &[2, 3, 4], &[5, 6, 7, 8, 9, 10, 11]];

#[test]
fn native_decode_harness_preserves_wrapped_rows_and_yarn() {
    let _guard = mlx_test_guard();
    for yarn in [false, true] {
        let model = tiny(yarn);
        assert!(model.supports_batching());
        assert_eq!(model.batched_decode_kind(), BatchedDecodeKind::Native);
        assert_eq!(
            model.sequence_state_layout(),
            SequenceStateLayout::model_owned(2)
        );
        assert!(!model.supports_padded_prefill());
        assert_batched_decode_matches_sequential_with_ids(&model, &PROMPTS, &[12, 13, 14], 1e-5);
        assert_single_row_batch_is_single_sequence_forward_with_ids(&model, PROMPTS[2], 12);
        assert_empty_batch_returns_empty_logits(&model);
    }
}

#[test]
fn interleaved_sequences_and_native_decode_match_fresh_solo_models() {
    let _guard = mlx_test_guard();
    let model = tiny(true);
    let solos = [tiny(true), tiny(true), tiny(true)];
    let ids = [
        SequenceId::from_raw(1),
        SequenceId::from_raw(2),
        SequenceId::from_raw(3),
    ];
    let mut caches: Vec<_> = ids.iter().map(|_| model.make_caches()).collect();
    let mut solo_caches: Vec<_> = solos.iter().map(|model| model.make_caches()).collect();
    for row in 0..3 {
        model.prepare_sequence_state(ids[row]);
        close(
            &model.forward_with_sequence_id(
                &input(PROMPTS[row]),
                Some(ids[row]),
                &mut caches[row],
                None,
            ),
            &solos[row].forward(&input(PROMPTS[row]), &mut solo_caches[row], None),
            0.0,
        );
    }
    for step in 0..6 {
        let tokens = [12 + step, 13 + step, 14 + step];
        if step % 2 == 0 {
            for row in [2, 0, 1] {
                close(
                    &model.forward_with_sequence_id(
                        &input(&[tokens[row]]),
                        Some(ids[row]),
                        &mut caches[row],
                        None,
                    ),
                    &solos[row].forward(&input(&[tokens[row]]), &mut solo_caches[row], None),
                    1e-5,
                );
            }
        } else {
            let inputs = mlxcel_core::from_slice_i32(&tokens, &[3, 1]);
            let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
            let actual = model.forward_batched_with_ids(&inputs, Some(&ids), &mut refs, None);
            for row in 0..3 {
                let actual =
                    mlxcel_core::slice(&actual, &[row as i32, 0, 0], &[row as i32 + 1, 1, 32]);
                close(
                    &actual,
                    &solos[row].forward(&input(&[tokens[row]]), &mut solo_caches[row], None),
                    1e-5,
                );
            }
        }
    }
    for row in 0..3 {
        model
            .sequence_state
            .with_sequence_state_ref(ids[row], |caches| {
                for cache in caches {
                    assert_eq!(gpt_oss_cache_offset(cache), PROMPTS[row].len() as i32 + 6);
                }
            })
            .expect("live sequence retained");
    }
    model.release_sequence_state_by_id(ids[1]);
    assert!(!model.sequence_state.has_sequence_state(ids[1]));
    assert!(model.sequence_state.has_sequence_state(ids[0]));
    assert!(model.sequence_state.has_sequence_state(ids[2]));
}

#[test]
fn fallback_reset_and_new_admission_leave_existing_sequence_unchanged() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let solo = tiny(false);
    let id = SequenceId::from_raw(1);
    let mut caches = model.make_caches();
    let mut solo_caches = solo.make_caches();
    model.prepare_sequence_state(id);
    for tokens in [PROMPTS[2], &[12]] {
        close(
            &model.forward_with_sequence_id(&input(tokens), Some(id), &mut caches, None),
            &solo.forward(&input(tokens), &mut solo_caches, None),
            0.0,
        );
    }
    let mut other_caches = model.make_caches();
    let other = SequenceId::from_raw(2);
    model.prepare_sequence_state(other);
    mlxcel_core::eval(&model.forward_with_sequence_id(
        &input(PROMPTS[1]),
        Some(other),
        &mut other_caches,
        None,
    ));
    close(
        &model.forward_with_sequence_id(&input(&[13]), Some(id), &mut caches, None),
        &solo.forward(&input(&[13]), &mut solo_caches, None),
        0.0,
    );
}

#[test]
fn paged_shadow_storage_tracks_model_owned_visible_lengths() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut pool = CachePool::new(1);
    let layout = PagedKvLayout::uniform(2, 4, 128).expect("valid test layout");
    let id = pool
        .allocate_with_layout(&model, Some(SequenceStateLayout::paged_kv_cache(layout)))
        .expect("paged shadow allocation succeeds");
    model.prepare_sequence_state(id);
    let mut compatibility_caches = model.make_caches();

    for tokens in [PROMPTS[2], &[12]] {
        let logits = model.forward_with_sequence_id(
            &input(tokens),
            Some(id),
            &mut compatibility_caches,
            None,
        );
        mlxcel_core::eval(&logits);
    }
    model
        .sync_sequence_storage(id, &mut pool)
        .expect("model-owned lengths synchronize");

    let state = pool.get_paged_state(id).expect("paged shadow state exists");
    let visible_lens: Vec<_> = (0..2)
        .map(|layer| state.layer(layer).expect("layer exists").visible_len())
        .collect();
    assert_eq!(visible_lens, vec![4, 8]);
    assert!(
        state.layers.iter().all(|layer| !layer.block_ids.is_empty()),
        "non-empty model-owned caches must reserve shadow accounting blocks"
    );
}

fn batched_after_prefill(model: &GptOssWrapper) -> UniquePtr<MlxArray> {
    let ids = [
        SequenceId::from_raw(11),
        SequenceId::from_raw(12),
        SequenceId::from_raw(13),
    ];
    let mut caches: Vec<_> = ids.iter().map(|_| model.make_caches()).collect();
    for row in 0..3 {
        model.prepare_sequence_state(ids[row]);
        mlxcel_core::eval(&model.forward_with_sequence_id(
            &input(PROMPTS[row]),
            Some(ids[row]),
            &mut caches[row],
            None,
        ));
    }
    let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
    model.forward_batched_with_ids(
        &mlxcel_core::from_slice_i32(&[12, 13, 14], &[3, 1]),
        Some(&ids),
        &mut refs,
        None,
    )
}

#[test]
fn learned_sinks_affect_batched_logits() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut no_sinks = tiny(false);
    for layer in &mut no_sinks.model.layers {
        layer.self_attn.sinks = mlxcel_core::zeros(&[2], mlxcel_core::dtype::FLOAT32);
    }
    let actual = batched_after_prefill(&model);
    let changed = batched_after_prefill(&no_sinks);
    let difference =
        mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&actual, &changed)));
    assert!(mlxcel_core::item_f32(&difference) > 1e-5);
}

#[test]
#[should_panic(expected = "model-owned batched decode requires sequence ids")]
fn anonymous_multirow_decode_is_rejected() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    model.forward_batched(
        &mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]),
        &mut [&mut a, &mut b],
        None,
    );
}

#[test]
#[should_panic(expected = "missing model-owned sequence state for sequence seq-23")]
fn missing_sequence_state_is_rejected() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    let ids = [SequenceId::from_raw(22), SequenceId::from_raw(23)];
    model.prepare_sequence_state(ids[0]);
    model.forward_batched_with_ids(
        &mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]),
        Some(&ids),
        &mut [&mut a, &mut b],
        None,
    );
}

#[test]
fn missing_sequence_state_is_rejected_before_per_row_mutation() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    let ids = [SequenceId::from_raw(22), SequenceId::from_raw(23)];
    model.prepare_sequence_state(ids[0]);
    mlxcel_core::eval(&model.forward_with_sequence_id(&input(&[1, 2]), Some(ids[0]), &mut a, None));

    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward_batched_with_ids(
            &mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[2, 2]),
            Some(&ids),
            &mut [&mut a, &mut b],
            None,
        );
    }));
    assert!(rejected.is_err());
    model
        .sequence_state
        .with_sequence_state_ref(ids[0], |caches| {
            assert!(caches.iter().all(|cache| gpt_oss_cache_offset(cache) == 2));
        })
        .expect("validated row must remain live");
}

#[test]
fn duplicate_sequence_state_is_rejected_before_per_row_mutation() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    let id = SequenceId::from_raw(22);
    model.prepare_sequence_state(id);
    mlxcel_core::eval(&model.forward_with_sequence_id(&input(&[1, 2]), Some(id), &mut a, None));

    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward_batched_with_ids(
            &mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[2, 2]),
            Some(&[id, id]),
            &mut [&mut a, &mut b],
            None,
        );
    }));
    assert!(rejected.is_err());
    model
        .sequence_state
        .with_sequence_state_ref(id, |caches| {
            assert!(caches.iter().all(|cache| gpt_oss_cache_offset(cache) == 2));
        })
        .expect("validated row must remain live");
}

#[test]
#[should_panic(expected = "input batch must match cache rows")]
fn input_batch_must_match_cache_rows_before_dispatch() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    let ids = [SequenceId::from_raw(22), SequenceId::from_raw(23)];
    model.prepare_sequence_state(ids[0]);
    model.prepare_sequence_state(ids[1]);
    model.forward_batched_with_ids(
        &mlxcel_core::from_slice_i32(&[1], &[1, 1]),
        Some(&ids),
        &mut [&mut a, &mut b],
        None,
    );
}

#[test]
#[should_panic(expected = "requires one sequence id per row")]
fn sequence_ids_must_match_cache_rows_before_dispatch() {
    let _guard = mlx_test_guard();
    let model = tiny(false);
    let mut a = model.make_caches();
    let mut b = model.make_caches();
    let id = SequenceId::from_raw(22);
    model.prepare_sequence_state(id);
    model.forward_batched_with_ids(
        &mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]),
        Some(&[id]),
        &mut [&mut a, &mut b],
        None,
    );
}

#[test]
fn masked_and_multitoken_batches_keep_sequence_identity() {
    let _guard = mlx_test_guard();
    for width in [1, 2] {
        let model = tiny(false);
        let solos = [tiny(false), tiny(false)];
        let ids = [SequenceId::from_raw(31), SequenceId::from_raw(32)];
        let mut caches: Vec<_> = ids.iter().map(|_| model.make_caches()).collect();
        let mut solo_caches: Vec<_> = solos.iter().map(|model| model.make_caches()).collect();
        for row in 0..2 {
            model.prepare_sequence_state(ids[row]);
            mlxcel_core::eval(&model.forward_with_sequence_id(
                &input(PROMPTS[row]),
                Some(ids[row]),
                &mut caches[row],
                None,
            ));
            mlxcel_core::eval(&solos[row].forward(
                &input(PROMPTS[row]),
                &mut solo_caches[row],
                None,
            ));
        }
        let tokens = if width == 1 {
            vec![12, 14]
        } else {
            vec![12, 13, 14, 15]
        };
        let inputs = mlxcel_core::from_slice_i32(&tokens, &[2, width]);
        let mask = mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32);
        let mut refs: Vec<_> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        // A supplied mask forces the per-row path, even at decode width.
        let actual = model.forward_batched_with_ids(&inputs, Some(&ids), &mut refs, Some(&mask));
        for row in 0..2 {
            let row_input = mlxcel_core::slice(&inputs, &[row as i32, 0], &[row as i32 + 1, width]);
            let row_actual =
                mlxcel_core::slice(&actual, &[row as i32, 0, 0], &[row as i32 + 1, width, 32]);
            close(
                &row_actual,
                &solos[row].forward(&row_input, &mut solo_caches[row], Some(&mask)),
                0.0,
            );
        }
    }
}
