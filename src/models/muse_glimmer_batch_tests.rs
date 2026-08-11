use super::*;
use mlxcel_core::cache::{SequenceId, SequenceStateBackend};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;

fn build_wrapper(config: &MuseGlimmerTextConfig) -> MuseGlimmerTextWrapper {
    let weights = synthetic_weights(config);
    let model = MuseGlimmerTextModel::from_weights(
        &weights,
        config,
        "model.language_model",
        "lm_head",
        vec![200001, 200008],
        vec![200092, 200091, 200018],
    )
    .expect("synthetic Muse weights should build");
    MuseGlimmerTextWrapper::new(model)
}

fn input(tokens: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn batch_input(rows: &[[i32; 1]]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let tokens = rows.iter().map(|row| row[0]).collect::<Vec<_>>();
    mlxcel_core::from_slice_i32(&tokens, &[rows.len() as i32, 1])
}

fn kv(seq_len: i32, base: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = (seq_len * 2) as usize;
    let values = (0..len).map(|idx| base + idx as f32).collect::<Vec<_>>();
    mlxcel_core::from_slice_f32(&values, &[1, 1, seq_len, 2])
}

fn logits_row(logits: &mlxcel_core::MlxArray, row: i32) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(logits);
    let row_logits = mlxcel_core::slice(logits, &[row, 0, 0], &[row + 1, 1, shape[2]]);
    to_vec_f32(&row_logits)
}

fn assert_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len());
    for (idx, (&a, &b)) in left.iter().zip(right).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-4,
            "value {idx} differs: left={a}, right={b}, diff={diff}"
        );
    }
}

fn cache_offsets(wrapper: &MuseGlimmerTextWrapper, seq_id: SequenceId) -> Vec<(bool, i32, i32)> {
    wrapper
        .sequence_cache_summaries(seq_id)
        .expect("sequence cache summary should exist")
}

#[test]
fn muse_batching_advertises_model_owned_sequence_state() {
    let wrapper = build_wrapper(&test_config());
    let layout = wrapper.sequence_state_layout();
    assert!(wrapper.supports_batching());
    assert!(!wrapper.supports_padded_prefill());
    assert_eq!(layout.backend, SequenceStateBackend::ModelOwned);
    assert_eq!(layout.num_layers, 2);
    assert!(wrapper.make_caches().is_empty());
}

#[test]
fn mixed_cache_boundaries_keep_sliding_rotating_and_full_growing() {
    let mut config = test_config();
    config.sliding_window = 2048;
    let weights = synthetic_weights(&config);
    let model = MuseGlimmerTextModel::from_weights(
        &weights,
        &config,
        "model.language_model",
        "lm_head",
        vec![200001, 200008],
        vec![200092, 200091, 200018],
    )
    .expect("synthetic Muse weights should build");

    assert!(config.rope_theta_for_layer(0).is_some());
    assert_eq!(config.rope_theta_for_layer(1), None);

    for len in [2047, 2048, 2049] {
        let mut caches = model.make_muse_caches();
        assert!(caches[0].is_sliding());
        assert!(!caches[1].is_sliding());
        caches[0].update_and_fetch(kv(len, 0.0), kv(len, 10_000.0));
        caches[1].update_and_fetch(kv(len, 20_000.0), kv(len, 30_000.0));
        assert_eq!(caches[0].offset(), len);
        assert_eq!(caches[0].live_len(), len.min(2048));
        assert_eq!(caches[1].offset(), len);
        assert_eq!(caches[1].live_len(), len);
    }
}

#[test]
fn batched_decode_with_sequence_ids_matches_isolated_rows() {
    let config = test_config();
    let batch = build_wrapper(&config);
    let single_a = build_wrapper(&config);
    let single_b = build_wrapper(&config);
    let seq_a = SequenceId::from_raw(10);
    let seq_b = SequenceId::from_raw(11);
    let ref_a = SequenceId::from_raw(20);
    let ref_b = SequenceId::from_raw(21);
    let mut empty = Vec::<KVCache>::new();

    for (model, seq, tokens) in [
        (&batch, seq_a, [1, 2].as_slice()),
        (&batch, seq_b, [3].as_slice()),
        (&single_a, ref_a, [1, 2].as_slice()),
        (&single_b, ref_b, [3].as_slice()),
    ] {
        model.prepare_sequence_state(seq);
        model.forward_with_sequence_id(&input(tokens), Some(seq), &mut empty, None);
    }

    let mut row0 = Vec::<KVCache>::new();
    let mut row1 = Vec::<KVCache>::new();
    let mut row_caches: Vec<&mut [KVCache]> = vec![row0.as_mut_slice(), row1.as_mut_slice()];
    let ids = [seq_a, seq_b];
    let logits = batch.forward_batched_with_context_and_ids(
        &batch_input(&[[4], [5]]),
        Some(&ids),
        &mut row_caches,
        None,
        None,
    );
    mlxcel_core::eval(&logits);

    let ref_logits_a =
        single_a.forward_with_sequence_id(&input(&[4]), Some(ref_a), &mut empty, None);
    let ref_logits_b =
        single_b.forward_with_sequence_id(&input(&[5]), Some(ref_b), &mut empty, None);
    mlxcel_core::eval(&ref_logits_a);
    mlxcel_core::eval(&ref_logits_b);

    assert_close(&logits_row(&logits, 0), &logits_row(&ref_logits_a, 0));
    assert_close(&logits_row(&logits, 1), &logits_row(&ref_logits_b, 0));
    assert_eq!(cache_offsets(&batch, seq_a)[0], (true, 3, 3));
    assert_eq!(cache_offsets(&batch, seq_b)[0], (true, 2, 2));
}

#[test]
fn embedding_prefill_populates_the_same_sequence_cache() {
    let config = test_config();
    let with_ids = build_wrapper(&config);
    let with_embeds = build_wrapper(&config);
    let seq_ids = SequenceId::from_raw(30);
    let seq_embeds = SequenceId::from_raw(31);
    let mut empty = Vec::<KVCache>::new();
    let prompt = input(&[1, 2, 3]);

    with_ids.prepare_sequence_state(seq_ids);
    with_embeds.prepare_sequence_state(seq_embeds);
    let embeds = with_embeds
        .embed_tokens(&prompt)
        .expect("Muse exposes token embeddings");
    with_ids.forward_with_sequence_id(&prompt, Some(seq_ids), &mut empty, None);
    with_embeds.forward_with_embeddings_and_sequence_id(
        &prompt,
        Some(&embeds),
        Some(seq_embeds),
        &mut empty,
        None,
    );

    let ids_logits =
        with_ids.forward_with_sequence_id(&input(&[4]), Some(seq_ids), &mut empty, None);
    let embed_logits =
        with_embeds.forward_with_sequence_id(&input(&[4]), Some(seq_embeds), &mut empty, None);
    mlxcel_core::eval(&ids_logits);
    mlxcel_core::eval(&embed_logits);

    assert_close(&logits_row(&ids_logits, 0), &logits_row(&embed_logits, 0));
    assert_eq!(cache_offsets(&with_embeds, seq_embeds)[0], (true, 4, 4));
}

#[test]
fn concurrent_sequences_do_not_share_or_overwrite_state() {
    let wrapper = build_wrapper(&test_config());
    let seq_a = SequenceId::from_raw(40);
    let seq_b = SequenceId::from_raw(41);
    let mut empty = Vec::<KVCache>::new();

    wrapper.prepare_sequence_state(seq_a);
    wrapper.prepare_sequence_state(seq_b);
    wrapper.forward_with_sequence_id(&input(&[1, 2, 3]), Some(seq_a), &mut empty, None);
    wrapper.forward_with_sequence_id(&input(&[4]), Some(seq_b), &mut empty, None);
    wrapper.forward_with_sequence_id(&input(&[5]), Some(seq_b), &mut empty, None);

    assert_eq!(cache_offsets(&wrapper, seq_a)[0], (true, 3, 3));
    assert_eq!(cache_offsets(&wrapper, seq_b)[0], (true, 2, 2));

    wrapper.forward_with_sequence_id(&input(&[6]), Some(seq_a), &mut empty, None);
    assert_eq!(cache_offsets(&wrapper, seq_a)[0], (true, 4, 4));
    assert_eq!(cache_offsets(&wrapper, seq_b)[0], (true, 2, 2));
}

#[test]
fn release_reset_and_snapshot_restore_make_reuse_equivalent() {
    let config = test_config();
    let wrapper = build_wrapper(&config);
    let seq_a = SequenceId::from_raw(50);
    let seq_b = SequenceId::from_raw(51);
    let mut empty = Vec::<KVCache>::new();

    wrapper.prepare_sequence_state(seq_a);
    wrapper.forward_with_sequence_id(&input(&[1, 2, 3]), Some(seq_a), &mut empty, None);
    let snapshot = wrapper
        .snapshot_sequence_state(seq_a, 3)
        .expect("Muse sequence snapshot should include cache tensors");
    wrapper
        .restore_sequence_state(seq_b, &snapshot)
        .expect("Muse snapshot should restore into another sequence");

    let logits_a = wrapper.forward_with_sequence_id(&input(&[4]), Some(seq_a), &mut empty, None);
    let logits_b = wrapper.forward_with_sequence_id(&input(&[4]), Some(seq_b), &mut empty, None);
    mlxcel_core::eval(&logits_a);
    mlxcel_core::eval(&logits_b);
    assert_close(&logits_row(&logits_a, 0), &logits_row(&logits_b, 0));

    wrapper.release_sequence_state_by_id(seq_a);
    wrapper.prepare_sequence_state(seq_a);
    wrapper.forward_with_sequence_id(&input(&[7]), Some(seq_a), &mut empty, None);
    assert_eq!(cache_offsets(&wrapper, seq_a)[0], (true, 1, 1));

    let first = wrapper.forward(&input(&[1, 2]), &mut empty, None);
    wrapper.reset_runtime_state();
    let second = wrapper.forward(&input(&[1, 2]), &mut empty, None);
    mlxcel_core::eval(&first);
    mlxcel_core::eval(&second);
    assert_close(&logits_row(&first, 0), &logits_row(&second, 0));
}
