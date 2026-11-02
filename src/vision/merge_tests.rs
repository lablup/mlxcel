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

use super::{
    MultimodalMask, masked_scatter, merge_llava, multimodal_attention_mask,
    prepare_inputs_for_multimodal,
};
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
        MultimodalMask::PrefixLm,
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
fn paligemma_mask_stays_prefix_bidirectional() {
    let _cpu = cpu_device();

    // PaliGemma is a prefix LM: with an all-ones attention_mask the additive
    // 4D mask is all zeros, so every prompt position attends everywhere,
    // text after the image included.
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
        MultimodalMask::PrefixLm,
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

/// Read an additive `[B, 1, S, S]` mask back as `allowed[b][q][k]`
/// (`0.0` is allowed, `f32::MIN` is masked; nothing else may appear).
fn allowed_matrix(mask: &MlxArray) -> Vec<Vec<Vec<bool>>> {
    let shape = mlxcel_core::array_shape(mask);
    assert_eq!(shape.len(), 4, "mask must be [B, 1, S, S]");
    assert_eq!(shape[1], 1, "mask must have one head axis");
    assert_eq!(shape[2], shape[3], "prefill mask must be square");
    assert_eq!(mlxcel_core::array_dtype(mask), dtype::FLOAT32);
    let (batch, seq) = (shape[0] as usize, shape[2] as usize);
    mlxcel_core::eval(mask);
    let values: Vec<f32> = mlxcel_core::array_to_raw_bytes(mask)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(values.len(), batch * seq * seq);
    (0..batch)
        .map(|b| {
            (0..seq)
                .map(|q| {
                    (0..seq)
                        .map(|k| {
                            let v = values[(b * seq + q) * seq + k];
                            assert!(
                                v == 0.0 || v == f32::MIN,
                                "mask[{b}][{q}][{k}] = {v} is neither 0.0 nor f32::MIN"
                            );
                            v == 0.0
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Parse rows written as strings of `1` (allowed) and `0` (masked).
fn rows(spec: &[&str]) -> Vec<Vec<bool>> {
    spec.iter()
        .map(|row| row.bytes().map(|c| c == b'1').collect())
        .collect()
}

const IMG: i32 = 99;
const BOS: i32 = 2;
const BOI: i32 = 7;
const EOI: i32 = 8;
const TXT: i32 = 5;

fn gemma3_merged_mask(ids: &[i32], attention_mask: &[i32]) -> Vec<Vec<Vec<bool>>> {
    let seq = ids.len() as i32;
    let image_tokens = ids.iter().filter(|&&id| id == IMG).count() as i32;
    let features: Vec<f32> = (0..image_tokens * 2).map(|i| i as f32).collect();
    let image_features = mlxcel_core::from_slice_f32(&features, &[1, image_tokens, 2]);
    let text: Vec<f32> = (0..seq * 2).map(|i| i as f32).collect();
    let inputs_embeds = mlxcel_core::from_slice_f32(&text, &[1, seq, 2]);
    let input_ids = mlxcel_core::from_slice_i32(ids, &[1, seq]);
    let attention_mask = mlxcel_core::from_slice_i32(attention_mask, &[1, seq]);
    let merged = prepare_inputs_for_multimodal(
        4,
        0,
        IMG,
        MultimodalMask::Gemma3ImageBlocks,
        &image_features,
        &inputs_embeds,
        &input_ids,
        &attention_mask,
    );
    let mask = match merged
        .attention_mask_4d
        .as_ref()
        .and_then(|mask| mask.as_ref())
    {
        Some(mask) => mask,
        None => panic!("expected 4D attention mask"),
    };
    allowed_matrix(mask)
}

/// Gemma 3 image prefill follows transformers (`create_masks_for_vision_model`):
/// causal attention, plus bidirectional attention inside each maximal run of
/// image soft tokens. `<start_of_image>` and `<end_of_image>` are text, and a
/// block never sees a later block (issue #2760).
#[test]
fn gemma3_mask_is_causal_with_bidirectional_image_blocks() {
    let _cpu = cpu_device();

    // Positions 0-12: bos, t, boi, img x3 (block 3..=5), eoi, t, boi,
    // img x2 (block 9..=10), eoi, t.
    let ids = [
        BOS, TXT, BOI, IMG, IMG, IMG, EOI, TXT, BOI, IMG, IMG, EOI, TXT,
    ];
    let allowed = gemma3_merged_mask(&ids, &[1; 13]);
    assert_eq!(allowed.len(), 1);
    let allowed = &allowed[0];

    let expected = rows(&[
        "1000000000000",
        "1100000000000",
        "1110000000000",
        "1111110000000",
        "1111110000000",
        "1111110000000",
        "1111111000000",
        "1111111100000",
        "1111111110000",
        "1111111111100",
        "1111111111100",
        "1111111111110",
        "1111111111111",
    ]);
    assert_eq!(allowed, &expected);

    // The cases the issue names, spelled out so a failure reads plainly.
    assert!(!allowed[1][2], "text must not see the next token");
    assert!(allowed[3][5], "an image token sees the rest of its block");
    assert!(!allowed[4][9], "an image block must not see a later block");
    assert!(
        allowed[9][3],
        "a later block sees an earlier block causally"
    );
    assert!(allowed[9][10], "an image token sees the rest of its block");
    assert!(!allowed[7][8], "text must not see the next token");
    assert!(
        allowed[12].iter().all(|&a| a),
        "the last row sees every key"
    );
}

#[test]
fn gemma3_mask_without_image_tokens_is_exactly_causal() {
    let _cpu = cpu_device();

    let input_ids = mlxcel_core::from_slice_i32(&[BOS, TXT, BOI, EOI, TXT], &[1, 5]);
    let attention_mask = mlxcel_core::from_slice_i32(&[1; 5], &[1, 5]);
    let mask = multimodal_attention_mask(
        MultimodalMask::Gemma3ImageBlocks,
        IMG,
        &input_ids,
        &attention_mask,
    );
    let expected = rows(&["10000", "11000", "11100", "11110", "11111"]);
    assert_eq!(allowed_matrix(&mask), vec![expected]);
}

#[test]
fn gemma3_mask_keeps_padding_keys_masked() {
    let _cpu = cpu_device();

    // Right padding: the pad key is masked for every query, the image block
    // stays bidirectional, and text stays causal.
    let allowed = gemma3_merged_mask(&[BOS, IMG, IMG, TXT, 0], &[1, 1, 1, 1, 0]);
    let expected = rows(&["10000", "11100", "11100", "11110", "11110"]);
    assert_eq!(allowed, vec![expected]);

    // Left padding: the pad key is masked even inside the image rows' reach,
    // and the pad query sees no key at all.
    let allowed = gemma3_merged_mask(&[0, BOS, IMG, IMG, TXT], &[0, 1, 1, 1, 1]);
    let expected = rows(&["00000", "01000", "01110", "01110", "01111"]);
    assert_eq!(allowed, vec![expected]);
}

/// Block ids are found per batch row: a run that starts at a different
/// position in each row, and a row with two runs, must not leak across rows.
#[test]
fn gemma3_mask_finds_image_blocks_per_batch_row() {
    let _cpu = cpu_device();

    let input_ids = mlxcel_core::from_slice_i32(
        &[
            BOS, IMG, IMG, TXT, TXT, TXT, //
            BOS, TXT, IMG, TXT, IMG, IMG,
        ],
        &[2, 6],
    );
    let attention_mask = mlxcel_core::from_slice_i32(&[1; 12], &[2, 6]);
    let mask = multimodal_attention_mask(
        MultimodalMask::Gemma3ImageBlocks,
        IMG,
        &input_ids,
        &attention_mask,
    );
    let expected = vec![
        rows(&["100000", "111000", "111000", "111100", "111110", "111111"]),
        rows(&["100000", "110000", "111000", "111100", "111111", "111111"]),
    ];
    assert_eq!(allowed_matrix(&mask), expected);
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
