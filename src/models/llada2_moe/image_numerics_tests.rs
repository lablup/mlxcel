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

//! The LLaDA-Image arithmetic on the tiny f32 synthetic checkpoint. In f32 it
//! computes the same function as the fused text path (only rounding order
//! differs), so the two must agree closely; a wrong rotation half, routing
//! group or score placement would not.

use super::super::tests::{build_tiny_checkpoint, to_host_f32, unique_temp_dir};
use super::topk_sorted;
use crate::models::llada2_moe::{Llada2MoeModel, Llada2Numerics, ModelArgs};

fn load(dir: &std::path::Path, numerics: Llada2Numerics) -> Llada2MoeModel {
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let args: ModelArgs = serde_json::from_str(&config).unwrap();
    let weights = crate::models::load_text_weights(dir, None).unwrap();
    Llada2MoeModel::from_weights_with_numerics(&weights, &args, numerics).unwrap()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
    (num / den).sqrt()
}

#[test]
fn forward_embeds_matches_the_fused_path_in_f32() {
    let dir = unique_temp_dir("image-numerics");
    build_tiny_checkpoint(&dir);
    let fused = load(&dir, Llada2Numerics::Fused);
    let image = load(&dir, Llada2Numerics::LladaImage);
    let ids = mlxcel_core::from_slice_i32(&[3, 4, 5, 6, 7], &[1, 5]);
    let positions = mlxcel_core::from_slice_i32(&[0, 1, 2, 3, 4], &[1, 5]);

    let Err(err) = fused.forward_embeds(&fused.embed(&ids), None, &positions) else {
        panic!("the fused path must refuse forward_embeds");
    };
    assert!(err.contains("LLaDA-Image numerics"), "{err}");

    let mut caches = fused.make_diffusion_caches();
    let reference = to_host_f32(&fused.forward_append(&ids, &mut caches, 0));
    let got = image
        .forward_embeds(&image.embed(&ids), None, &positions)
        .unwrap();
    assert_eq!(mlxcel_core::array_shape(&got), vec![1, 5, 8]);
    let got = to_host_f32(&got);
    let drift = rel_rms(&got, &reference);
    assert!(drift < 1e-4, "rel RMS {drift}");

    // A block-causal mask: the first two rows see only themselves, so they
    // equal a run over those two tokens alone.
    let allowed = [
        1u8, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    let mask = mlxcel_core::astype(
        &mlxcel_core::from_slice_i32(
            &allowed.iter().map(|&v| i32::from(v)).collect::<Vec<_>>(),
            &[1, 1, 5, 5],
        ),
        mlxcel_core::dtype::BOOL,
    );
    let masked = to_host_f32(
        &image
            .forward_embeds(&image.embed(&ids), Some(&mask), &positions)
            .unwrap(),
    );
    let head_ids = mlxcel_core::from_slice_i32(&[3, 4], &[1, 2]);
    let head_pos = mlxcel_core::from_slice_i32(&[0, 1], &[1, 2]);
    let alone = to_host_f32(
        &image
            .forward_embeds(&image.embed(&head_ids), None, &head_pos)
            .unwrap(),
    );
    assert!(rel_rms(&masked[..16], &alone) < 1e-6);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn topk_sorted_orders_by_descending_value() {
    let x = mlxcel_core::from_slice_f32(&[0.1, 0.9, 0.5, 0.7, 0.3], &[1, 5]);
    let (values, indices) = topk_sorted(&x, 3);
    assert_eq!(to_host_f32(&values), vec![0.9, 0.7, 0.5]);
    assert_eq!(to_host_f32(&indices), vec![1.0, 3.0, 2.0]);
}
