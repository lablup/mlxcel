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

//! The OpenXLA Gemma 3 image mask against the MLX one (issue #2781).
//!
//! `mlxcel_xla::prepare_gemma3_vlm_prefill` builds the host-side additive
//! mask the OpenXLA embeddings prefill consumes; the MLX path builds its mask
//! with `vision::merge::multimodal_attention_mask`. Both must be the
//! transformers mask (causal plus bidirectional image blocks, padding keys
//! dropped) cell for cell. No IREE runtime is needed.

use mlxcel_core::session::OwnedTensor;

use crate::vision::merge::{MultimodalMask, multimodal_attention_mask};

const PAD: i32 = 0;
const TEXT: i32 = 7;
const IMG: i32 = 9;

fn f32_values(tensor: &OwnedTensor) -> Vec<f32> {
    tensor
        .bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn mlx_mask(ids: &[i32], padding: &[i32]) -> Vec<f32> {
    let len = ids.len() as i32;
    let input_ids = mlxcel_core::from_slice_i32(ids, &[1, len]);
    let attention_mask = mlxcel_core::from_slice_i32(padding, &[1, len]);
    let mask = multimodal_attention_mask(
        MultimodalMask::Gemma3ImageBlocks,
        IMG,
        &input_ids,
        &attention_mask,
    );
    let mask = mlxcel_core::astype(&mask, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&mask);
    mlxcel_core::array_to_raw_bytes(&mask)
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn xla_mask(ids: &[i32], padding: &[i32], images: usize) -> Vec<f32> {
    let image_rows = ids.iter().filter(|&&id| id == IMG).count();
    let prepared = mlxcel_xla::prepare_gemma3_vlm_prefill(
        ids.to_vec(),
        &vec![0.5; ids.len()],
        &vec![0.25; image_rows],
        padding,
        1,
        ids.len(),
        PAD,
        IMG,
        images,
    )
    .expect("valid Gemma 3 prepared prefill");
    assert_eq!(
        prepared.attention_bias.tensor.shape,
        vec![1, 1, ids.len(), ids.len()]
    );
    assert!(!prepared.attention_bias.causal);
    f32_values(&prepared.attention_bias.tensor)
}

fn assert_masks_agree(ids: &[i32], images: usize) {
    let padding: Vec<i32> = ids.iter().map(|&id| i32::from(id != PAD)).collect();
    let mlx = mlx_mask(ids, &padding);
    let xla = xla_mask(ids, &padding, images);
    assert_eq!(mlx.len(), xla.len(), "{ids:?}: mask size");
    let len = ids.len();
    for (index, (&got, &want)) in xla.iter().zip(&mlx).enumerate() {
        assert!(
            want == 0.0 || want == f32::MIN,
            "{ids:?}: MLX cell {index} is {want}, not 0.0 or f32::MIN"
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{ids:?}: query {} key {}: OpenXLA {got}, MLX {want}",
            index / len,
            index % len
        );
    }
}

/// The issue's prompt: padding, text, a two-token image block, text, a
/// one-token image block, text.
#[test]
fn openxla_gemma3_mask_matches_mlx_for_two_image_blocks() {
    assert_masks_agree(&[PAD, TEXT, IMG, IMG, TEXT, IMG, TEXT], 1);
}

/// Edge cases from the issue: an image run at position 0, no image at all,
/// two adjacent-looking blocks split by one text token, and padding inside
/// the prompt.
#[test]
fn openxla_gemma3_mask_matches_mlx_on_edge_cases() {
    assert_masks_agree(&[IMG, IMG, IMG, TEXT, TEXT], 1);
    assert_masks_agree(&[TEXT, TEXT, TEXT, TEXT], 0);
    assert_masks_agree(&[TEXT, IMG, IMG, TEXT, IMG, IMG, TEXT], 2);
    assert_masks_agree(&[TEXT, IMG, PAD, IMG, TEXT, PAD, TEXT], 1);
}
