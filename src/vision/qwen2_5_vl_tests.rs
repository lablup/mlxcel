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

//! Qwen2.5-VL video MRoPE timing (#3234, Blaizzy/mlx-vlm#2760): the temporal
//! axis of a video is spaced by `tokens_per_second * second_per_grid`, and
//! that timing reaches the text model's MRoPE state through the
//! `QwenVlRuntime` entry point the server and CLI video path calls.

use super::Qwen25VLModel;
use crate::models::embedding_test_support::{mlx_test_guard, to_vec};
use crate::models::qwen2_vl::batched_tests::tiny_qwen2_vl_model;
use crate::multimodal::qwen_vl::QwenVlRuntime;
use crate::vision::feature_cache::{CacheKey, ModelVisionCaches, SingleArrayFeatures};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

const IMAGE_TOKEN: i32 = 29;
const VIDEO_TOKEN: i32 = 30;
/// Hidden size of `tiny_qwen2_vl_model`.
const HIDDEN: i32 = 16;

/// A Qwen2.5-VL wrapper over the tiny text model, with no vision tower: the
/// tests below either stop at the positions or feed the tower's output from
/// the vision feature cache.
fn tiny_qwen25(tokens_per_second: f32) -> Qwen25VLModel {
    Qwen25VLModel {
        text_model: tiny_qwen2_vl_model(),
        vision_encoder: None,
        text_only_path: None,
        processor: Qwen2VLProcessor::new(14, 2, 2),
        image_token_id: IMAGE_TOKEN,
        video_token_id: VIDEO_TOKEN,
        vision_start_token_id: 28,
        spatial_merge_size: 2,
        video_tokens_per_second: tokens_per_second,
    }
}

fn input_ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

/// `[3, 1, len]` positions as the three axes `[T, H, W]`.
fn axes(positions: &MlxArray) -> [Vec<i32>; 3] {
    let flat: Vec<i32> = to_vec(positions)
        .into_iter()
        .map(|value| value as i32)
        .collect();
    let len = flat.len() / 3;
    [
        flat[..len].to_vec(),
        flat[len..2 * len].to_vec(),
        flat[2 * len..].to_vec(),
    ]
}

/// Three temporal groups at interval 2.0 end at position 5, so the trailing
/// text sits at 6 in a 5-token prompt: the decode steps that follow have to
/// add a rope delta of 2. Unit spacing (the pre-#3234 behaviour) gives 0.
#[test]
fn qwen25_rope_delta_reflects_scaled_video() {
    let _guard = mlx_test_guard();
    let model = tiny_qwen25(2.0);
    let ids = input_ids(&[1, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 5]);
    let grid = [(3, 2, 2)];

    let (positions, delta) = model.compute_rope_state(&ids, &grid, &[2.0]).unwrap();
    assert_eq!(delta, 2);
    assert_eq!(
        axes(&positions),
        [
            vec![0, 1, 3, 5, 6],
            vec![0, 1, 1, 1, 6],
            vec![0, 1, 1, 1, 6]
        ]
    );

    let (positions, delta) = model.compute_rope_state(&ids, &grid, &[]).unwrap();
    assert_eq!(delta, 0, "no timing keeps unit spacing");
    assert_eq!(axes(&positions)[0], vec![0, 1, 2, 3, 4]);
}

/// Run the runtime entry point with the vision tower's output served from the
/// feature cache, then read back the MRoPE state it left for the sequence.
fn mrope_state_after_runtime_call(
    model: &Qwen25VLModel,
    tokens: &[i32],
    grid: &[(i32, i32, i32)],
    video_second_per_grid: &[f32],
    visual_tokens: i32,
) -> ([Vec<i32>; 3], i32) {
    let caches = ModelVisionCaches::new(4);
    let key = CacheKey::from_hash([7; 32]);
    let features = SingleArrayFeatures::new(mlxcel_core::from_slice_f32(
        &vec![0.25; (visual_tokens * HIDDEN) as usize],
        &[visual_tokens, HIDDEN],
    ));
    caches.single.lock().unwrap().put(key.clone(), &features);

    let ids = input_ids(tokens);
    // Never read: the cache hit above replaces the vision tower.
    let pixel_values = mlxcel_core::from_slice_f32(&[0.0], &[1, 1]);
    let runtime: &dyn QwenVlRuntime = model;
    let embeddings = runtime
        .input_embeddings_with_video_timing(
            &ids,
            &pixel_values,
            grid,
            video_second_per_grid,
            Some(&key),
            Some(&caches),
        )
        .unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&embeddings.inputs_embeds),
        [1, tokens.len() as i32, HIDDEN]
    );

    let seq_id = SequenceId::from_raw(3234);
    runtime.bind_mrope_state_to_sequence(seq_id);
    let (entry, _) = runtime.take_mrope_entry_for_sequence(seq_id).into_parts();
    let entry = entry.expect("the runtime call must leave MRoPE state for the sequence");
    let positions = entry
        .position_ids
        .expect("a media prefill stores positions");
    (
        axes(&positions),
        entry.rope_deltas.expect("and its rope delta"),
    )
}

/// The Qwen2.5-VL runtime turns `second_per_grid` into positions: at 2 tokens
/// per second and 1.0 s per temporal grid (2 fps, temporal patch 2) the
/// interval is 2. The same call with no timing keeps unit spacing, which is
/// the images-only path.
#[test]
fn qwen25_runtime_video_timing_reaches_mrope_state() {
    let _guard = mlx_test_guard();
    let model = tiny_qwen25(2.0);
    let tokens = [1, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 5];
    let grid = [(3, 2, 2)];

    let (positions, delta) = mrope_state_after_runtime_call(&model, &tokens, &grid, &[1.0], 3);
    assert_eq!(positions[0], vec![0, 1, 3, 5, 6]);
    assert_eq!(positions[1], vec![0, 1, 1, 1, 6]);
    assert_eq!(positions[2], vec![0, 1, 1, 1, 6]);
    assert_eq!(delta, 2);

    let (positions, delta) = mrope_state_after_runtime_call(&model, &tokens, &grid, &[], 3);
    assert_eq!(positions[0], vec![0, 1, 2, 3, 4]);
    assert_eq!(delta, 0);
}

/// The interval is `tokens_per_second * second_per_grid`, so the config value
/// moves the positions: 4 tokens per second at 0.5 s per grid is interval 2
/// again, and 4 at 1.0 is interval 4.
#[test]
fn qwen25_runtime_video_timing_uses_config_tokens_per_second() {
    let _guard = mlx_test_guard();
    let model = tiny_qwen25(4.0);
    let tokens = [1, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 5];
    let grid = [(3, 2, 2)];

    let (positions, delta) = mrope_state_after_runtime_call(&model, &tokens, &grid, &[0.5], 3);
    assert_eq!(positions[0], vec![0, 1, 3, 5, 6]);
    assert_eq!(delta, 2);

    let (positions, delta) = mrope_state_after_runtime_call(&model, &tokens, &grid, &[1.0], 3);
    assert_eq!(positions[0], vec![0, 1, 5, 9, 10]);
    assert_eq!(delta, 6);
}

/// A timing slice that does not match the prompt's videos is an error from
/// the runtime entry point, not a fallback to unit spacing.
#[test]
fn qwen25_runtime_rejects_mismatched_video_timing() {
    let _guard = mlx_test_guard();
    let model = tiny_qwen25(2.0);
    let ids = input_ids(&[1, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, 5]);
    let grid = [(3, 2, 2)];

    assert!(model.compute_rope_state(&ids, &grid, &[2.0, 2.0]).is_err());
    assert!(model.compute_rope_state(&ids, &grid, &[0.0]).is_err());
}

/// `tokens_per_second` is read from `vision_config`, defaulting to the HF
/// value of 2 when the checkpoint does not carry it.
#[test]
fn qwen25_vision_config_tokens_per_second_defaults_to_two() {
    use crate::vision::encoders::qwen2_5_vl::Qwen25VLVisionConfig;

    let default: Qwen25VLVisionConfig =
        serde_json::from_value(serde_json::json!({"hidden_size": 1280})).unwrap();
    assert_eq!(default.tokens_per_second, 2.0);

    // The published configs write it as an integer.
    let explicit: Qwen25VLVisionConfig =
        serde_json::from_value(serde_json::json!({"hidden_size": 1280, "tokens_per_second": 4}))
            .unwrap();
    assert_eq!(explicit.tokens_per_second, 4.0);
}
