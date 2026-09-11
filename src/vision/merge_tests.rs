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

use super::{masked_scatter, merge_llava, prepare_inputs_for_multimodal};
use mlxcel_core::streams::{DefaultDeviceGuard, DefaultDeviceLock, lock_default_device};
use mlxcel_core::{self, MlxArray, dtype};

/// Run one test on the CPU and put the previous default device back when the
/// returned guards drop; bind the result to a named local for the test's
/// duration. The `Once` this replaces moved the process-wide default device
/// for good, so every test that ran after this module measured the CPU
/// backend (issue #1421).
///
/// The lock is `mlxcel_core::streams::lock_default_device`, the one
/// process-wide lock every default-device mover takes. libtest runs one
/// binary's tests in parallel unless told otherwise, and a guard records the
/// current default device when it is created, so unserialized movers
/// interleave into exactly the leak this module was converted to remove: the
/// second records the first's CPU default as its baseline, the first restores
/// the GPU, and the second then restores the CPU for good. A lock private to
/// this module would not have covered that, because the other movers are in
/// `multimodal::host_preprocessor` and behind `mlx_test_guard`.
///
/// The tuple order is load-bearing. Tuple fields drop in declaration order,
/// so the device guard must come first: releasing the lock before the device
/// is restored would let the next test take the lock and record the *moved*
/// device as its baseline.
fn cpu_device() -> (DefaultDeviceGuard, DefaultDeviceLock) {
    let lock = lock_default_device();
    let device = DefaultDeviceGuard::cpu();
    (device, lock)
}

fn assert_arrays_equal(actual: &MlxArray, expected: &MlxArray) {
    let equal = mlxcel_core::array_equal(actual, expected, false);
    assert!(mlxcel_core::item_bool(&equal));
}

#[test]
fn masked_scatter_replaces_only_masked_positions() {
    let _cpu = cpu_device();

    let base = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
    let input_ids = mlxcel_core::from_slice_i32(&[0, 7, 0], &[1, 3]);
    let image_token = mlxcel_core::full_f32(&[1], 7.0, dtype::INT32);
    let image_token = mlxcel_core::astype(&image_token, dtype::INT32);
    let mask = mlxcel_core::equal(&input_ids, &image_token);
    let mask = mlxcel_core::expand_dims(&mask, -1);
    let mask = mlxcel_core::repeat(&mask, 2, -1);
    let features = mlxcel_core::from_slice_f32(&[9.0, 8.0], &[1, 1, 2]);

    let merged = masked_scatter(&base, &mask, &features);
    let expected = mlxcel_core::from_slice_f32(&[1.0, 2.0, 9.0, 8.0, 5.0, 6.0], &[1, 3, 2]);

    assert_arrays_equal(&merged, &expected);
}

#[test]
fn prepare_inputs_for_multimodal_builds_additive_mask_and_preserves_dtype() {
    let _cpu = cpu_device();

    let image_features = mlxcel_core::from_slice_f32(&[4.0, 6.0], &[1, 1, 2]);
    let inputs_embeds =
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 10.0, 11.0, 20.0, 21.0], &[1, 3, 2]);
    let inputs_embeds = mlxcel_core::astype(&inputs_embeds, dtype::FLOAT16);
    let input_ids = mlxcel_core::from_slice_i32(&[10, 99, 0], &[1, 3]);
    let attention_mask = mlxcel_core::from_slice_i32(&[1, 1, 0], &[1, 3]);

    let merged = prepare_inputs_for_multimodal(
        4,
        0,
        99,
        &image_features,
        &inputs_embeds,
        &input_ids,
        &attention_mask,
    );

    assert_eq!(
        mlxcel_core::array_dtype(&merged.inputs_embeds),
        dtype::FLOAT16
    );

    let expected_embeds = mlxcel_core::from_slice_f32(&[1.0, 2.0, 2.0, 3.0, 0.0, 0.0], &[1, 3, 2]);
    let expected_embeds = mlxcel_core::astype(&expected_embeds, dtype::FLOAT16);
    assert_arrays_equal(&merged.inputs_embeds, &expected_embeds);

    let expected_mask = mlxcel_core::from_slice_f32(
        &[
            0.0,
            0.0,
            f32::MIN,
            0.0,
            0.0,
            f32::MIN,
            f32::MIN,
            f32::MIN,
            f32::MIN,
        ],
        &[1, 1, 3, 3],
    );
    let actual_mask = match merged
        .attention_mask_4d
        .as_ref()
        .and_then(|mask| mask.as_ref())
    {
        Some(mask) => mask,
        None => panic!("expected 4D attention mask"),
    };
    assert_arrays_equal(actual_mask, &expected_mask);
}

#[test]
fn prepare_inputs_for_multimodal_all_ones_mask_is_all_zeros() {
    let _cpu = cpu_device();

    // Sanity check for the common case: when attention_mask is all ones,
    // the additive 4D mask must be all zeros (attend everywhere).
    let image_features = mlxcel_core::from_slice_f32(&[4.0, 6.0], &[1, 1, 2]);
    let inputs_embeds =
        mlxcel_core::from_slice_f32(&[1.0, 2.0, 10.0, 11.0, 20.0, 21.0], &[1, 3, 2]);
    let inputs_embeds = mlxcel_core::astype(&inputs_embeds, dtype::FLOAT16);
    let input_ids = mlxcel_core::from_slice_i32(&[10, 99, 42], &[1, 3]);
    let attention_mask = mlxcel_core::from_slice_i32(&[1, 1, 1], &[1, 3]);

    let merged = prepare_inputs_for_multimodal(
        4,
        0,
        99,
        &image_features,
        &inputs_embeds,
        &input_ids,
        &attention_mask,
    );

    let expected_mask = mlxcel_core::from_slice_f32(
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1, 1, 3, 3],
    );
    let actual_mask = match merged
        .attention_mask_4d
        .as_ref()
        .and_then(|mask| mask.as_ref())
    {
        Some(mask) => mask,
        None => panic!("expected 4D attention mask"),
    };
    assert_eq!(mlxcel_core::array_dtype(actual_mask), dtype::FLOAT32);
    assert_arrays_equal(actual_mask, &expected_mask);
}

#[test]
fn merge_llava_flattens_projected_features_in_image_token_order() {
    let _cpu = cpu_device();

    let image_features = mlxcel_core::from_slice_f32(&[10.0, 11.0, 12.0, 13.0], &[1, 2, 2]);
    let inputs_embeds =
        mlxcel_core::from_slice_f32(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], &[1, 4, 2]);
    let input_ids = mlxcel_core::from_slice_i32(&[5, 42, 42, 6], &[1, 4]);

    let merged = merge_llava(42, &image_features, &inputs_embeds, &input_ids);
    let expected =
        mlxcel_core::from_slice_f32(&[1.0, 1.0, 10.0, 11.0, 12.0, 13.0, 4.0, 4.0], &[1, 4, 2]);

    assert_arrays_equal(&merged.inputs_embeds, &expected);
    assert!(merged.attention_mask_4d.is_none());
}

/// Pin the two-op composition `Gemma4UnifiedModel::merge_multimodal` runs when
/// a prompt carries video frames and audio at once (issue #1349): `merge_llava`
/// on `video_token_id`, then `masked_scatter` on `audio_token_id`.
///
/// This is the primitive-level stand-in for a synthetic-model scatter test. A
/// `Gemma4UnifiedModel` cannot be built without a full Gemma 4 backbone weight
/// map plus vision and multimodal projector weights, and instantiating one
/// would run a real forward on the GPU inside the unit-test gate. The two ops
/// below are literally the ones `merge_multimodal` calls, in the same order and
/// against the same running embeddings, so the property that matters is
/// testable here: the two runs address disjoint placeholder ids, each run
/// receives its own features, and neither clobbers the other or the text.
#[test]
fn video_then_audio_scatter_fills_each_run_without_clobbering_the_other() {
    let _cpu = cpu_device();

    const VIDEO_TOKEN: i32 = 42;
    const AUDIO_TOKEN: i32 = 77;

    // text, 4 video tokens, text, 3 audio tokens, text — the shape an expanded
    // Gemma 4 Unified prompt has once the frame runs and the audio run are in.
    let input_ids = mlxcel_core::from_slice_i32(
        &[
            5,
            VIDEO_TOKEN,
            VIDEO_TOKEN,
            VIDEO_TOKEN,
            VIDEO_TOKEN,
            6,
            AUDIO_TOKEN,
            AUDIO_TOKEN,
            AUDIO_TOKEN,
            7,
        ],
        &[1, 10],
    );
    // One text value per position, hidden = 2, so a clobbered text row is
    // visible in the assertion rather than hidden behind a broadcast.
    let text: Vec<f32> = (0..10).flat_map(|i| [i as f32, i as f32]).collect();
    let inputs_embeds = mlxcel_core::from_slice_f32(&text, &[1, 10, 2]);

    // Distinct constants per modality: every video row is -1.0, every audio row
    // is -2.0, so a run that received the other modality's features fails.
    let video_features = mlxcel_core::from_slice_f32(&[-1.0; 8], &[1, 4, 2]);
    let audio_features = mlxcel_core::from_slice_f32(&[-2.0; 6], &[1, 3, 2]);

    let after_video = merge_llava(VIDEO_TOKEN, &video_features, &inputs_embeds, &input_ids);

    // The audio half is the `masked_scatter` branch of `merge_multimodal`:
    // build the `input_ids == audio_token_id` mask, broadcast it over the
    // hidden axis, and scatter the projected audio rows into it.
    let audio_token_arr = mlxcel_core::from_slice_i32(&[AUDIO_TOKEN], &[1]);
    let is_audio = mlxcel_core::equal(&input_ids, &audio_token_arr);
    let audio_mask = mlxcel_core::expand_dims(&is_audio, -1);
    let audio_mask = mlxcel_core::broadcast_to(
        &audio_mask,
        &mlxcel_core::array_shape(&after_video.inputs_embeds),
    );
    let merged = masked_scatter(&after_video.inputs_embeds, &audio_mask, &audio_features);

    let expected = mlxcel_core::from_slice_f32(
        &[
            0.0, 0.0, // text
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, // 4 video rows
            5.0, 5.0, // text
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, // 3 audio rows
            9.0, 9.0, // text
        ],
        &[1, 10, 2],
    );
    assert_arrays_equal(&merged, &expected);
}
