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

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::*;
use crate::vision::processors::molmo::MolmoImageTokens;

#[path = "host_preprocessor_molmo_diagnostics_support.rs"]
mod support;
use support::*;

#[test]
fn pinned_processor_fixture_preserves_patch_grid_and_sparse_mapping() {
    let output = pinned_processor_output();
    assert_eq!(digest_f32(&output.pixel_values), PINNED_PIXEL_VALUES_SHA256);
    assert_eq!(digest_f32(&output.image_masks), PINNED_IMAGE_MASKS_SHA256);
    assert_eq!(
        digest_i32(&output.image_token_ids),
        PINNED_IMAGE_TOKEN_IDS_SHA256
    );
    assert_eq!(
        digest_i32(&output.image_input_idx),
        PINNED_IMAGE_INPUT_IDX_SHA256
    );
    assert_eq!(digest_grid(&output), PINNED_GRID_SHA256);
}

#[test]
fn strict_oracle_rejects_molmo2_token_scan_and_replacement() {
    let canonical =
        mlxcel_xla::MolmoSparseAddPlan::from_image_input_idx(&[-1, 4, 1], 6, 2, 3, 6).unwrap();
    let patch_id = MolmoImageTokens::default().image_patch_id;
    let tokens = [7, patch_id, patch_id, 8, patch_id, 9];
    let token_scanned = tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| **token == patch_id)
        .enumerate()
        .map(
            |(feature_row, (target_position, _))| mlxcel_xla::MolmoSparseAddPair {
                feature_row,
                target_position,
            },
        )
        .collect::<Vec<_>>();
    assert_ne!(
        canonical.pairs(),
        token_scanned,
        "Molmo2 token scanning must not satisfy the Molmo v1 mapping oracle"
    );

    let original = vec![2.0, 3.0, 5.0, 7.0];
    let features = vec![11.0, 13.0];
    let plan = mlxcel_xla::MolmoSparseAddPlan::from_image_input_idx(&[1], 2, 2, 1, 2).unwrap();
    let mut additive = original.clone();
    plan.apply(&mut additive, &features).unwrap();
    let mut replacement = original;
    replacement[2..].copy_from_slice(&features);
    assert_ne!(
        additive, replacement,
        "replacement must not satisfy the Molmo v1 additive oracle"
    );
}

/// Strict actual-checkpoint boundary gate for #870.
///
/// Run from the repository root only with the pinned 5.3 GB checkpoint:
///
/// ```text
/// IREE_DIST=/path/to/iree-dist \
/// MLXCEL_MOLMO_FIXTURE=/path/to/Molmo-7B-D-0924-4bit \
/// cargo test --features xla-reference-diagnostics --lib \
///   pinned_molmo_eager_mlx_matches_iree_boundaries -- --ignored --nocapture
/// ```
///
/// The gate uses one resident IREE decoder for both prepared payloads. It
/// captures K/V and logits propagation but does not claim an independent
/// decoder oracle.
#[test]
#[ignore = "requires pinned Molmo checkpoint and IREE local-task"]
fn pinned_molmo_eager_mlx_matches_iree_boundaries() {
    mlxcel_core::set_default_device(false);
    let model = PathBuf::from(
        std::env::var_os("MLXCEL_MOLMO_FIXTURE")
            .expect("MLXCEL_MOLMO_FIXTURE must name the pinned checkpoint"),
    );
    let image_path = std::env::var_os("MLXCEL_MOLMO_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/test_image.png"));
    let device =
        std::env::var("MLXCEL_MOLMO_IREE_DEVICE").unwrap_or_else(|_| "local-task".to_string());
    assert_eq!(
        device, "local-task",
        "the #870 reference gate qualifies IREE local-task only"
    );
    progress("validate pinned checkpoint and image");
    assert_eq!(pinned_revision(&model), PINNED_REVISION);
    assert_sha256(
        &model.join("config.json"),
        PINNED_CONFIG_SHA256,
        "checkpoint config",
    );
    assert_sha256(
        &model.join("preprocessor_config.json"),
        PINNED_PREPROCESSOR_SHA256,
        "checkpoint preprocessor",
    );
    assert_sha256(
        &model.join("processor_config.json"),
        PINNED_PROCESSOR_SHA256,
        "checkpoint processor",
    );
    assert_sha256(&image_path, PINNED_IMAGE_SHA256, "fixture image");
    let image = image::open(&image_path).expect("open pinned Molmo image");
    let mut first_divergence = None;

    progress("load filtered eager MLX and production IREE preprocessors");
    let host = MolmoHostPreprocessor::load(&model).expect("load Molmo host preprocessor");
    let production = MolmoHostPreprocessor::load_iree(&model, &device)
        .expect("load Molmo production IREE preprocessor");
    let host_processor = host.processor.preprocess_image(&image);
    let iree_processor = production.processor.preprocess_image(&image);
    assert_eq!(
        digest_f32(&host_processor.pixel_values),
        PINNED_PIXEL_VALUES_SHA256
    );
    assert_eq!(
        digest_f32(&host_processor.image_masks),
        PINNED_IMAGE_MASKS_SHA256
    );
    assert_eq!(
        digest_i32(&host_processor.image_input_idx),
        PINNED_IMAGE_INPUT_IDX_SHA256
    );
    compare_stage(
        "processor.patches",
        &iree_processor.pixel_values,
        &host_processor.pixel_values,
        EXACT,
        &mut first_divergence,
    );
    compare_stage(
        "processor.patch_masks",
        &iree_processor.image_masks,
        &host_processor.image_masks,
        EXACT,
        &mut first_divergence,
    );
    compare_exact_i32(
        "processor.image_token_ids",
        &iree_processor.image_token_ids,
        &host_processor.image_token_ids,
        &mut first_divergence,
    );
    compare_exact_i32(
        "processor.image_input_idx",
        &iree_processor.image_input_idx,
        &host_processor.image_input_idx,
        &mut first_divergence,
    );
    compare_exact_i32(
        "processor.grid",
        &[
            iree_processor.pixel_values_shape[0],
            iree_processor.pixel_values_shape[1],
            iree_processor.image_input_idx_len,
        ],
        &[
            host_processor.pixel_values_shape[0],
            host_processor.pixel_values_shape[1],
            host_processor.image_input_idx_len,
        ],
        &mut first_divergence,
    );

    progress("capture eager MLX patch, selected-layer, and projector states");
    let [crops, patches, width] = host_processor.pixel_values_shape;
    let pixels =
        mlxcel_core::from_slice_f32(&host_processor.pixel_values, &[1, crops, patches, width]);
    let masks = mlxcel_core::from_slice_f32(&host_processor.image_masks, &[1, crops, patches]);
    let MolmoVision::Host(vision) = &host.vision else {
        panic!("host preprocessor must retain the eager MLX vision tower");
    };
    let (_, host_diagnostics) = vision.forward_with_diagnostics(&pixels, &masks);
    let host_patch = mlx_f32(host_diagnostics.patch_embeddings.as_ref().unwrap());
    let host_selected = host_diagnostics
        .selected_hidden_states
        .iter()
        .map(|state| mlx_f32(state.as_ref().unwrap()))
        .collect::<Vec<_>>();
    let host_projected = mlx_f32(host_diagnostics.projected_features.as_ref().unwrap());

    progress("capture resident IREE patch, selected-layer, and projector states");
    let mut diagnostic = mlxcel_xla::IreeMolmoVisionDiagnosticProjector::load(&model, &device)
        .expect("load Molmo IREE diagnostic projector");
    let iree_diagnostics = diagnostic
        .project(
            &host_processor.pixel_values,
            &host_processor.image_masks,
            usize::try_from(crops).expect("positive crop count"),
        )
        .expect("run Molmo IREE diagnostic projector");
    compare_stage(
        "vision.patch_embedding",
        &iree_diagnostics.patch_embeddings,
        &host_patch,
        VISION,
        &mut first_divergence,
    );
    assert_eq!(
        iree_diagnostics.selected_hidden_states.len(),
        host_selected.len()
    );
    for ((layer, iree), host) in iree_diagnostics
        .selected_layers
        .iter()
        .zip(&iree_diagnostics.selected_hidden_states)
        .zip(&host_selected)
    {
        compare_stage(
            &format!("vision.selected_layer_{layer}"),
            iree,
            host,
            VISION,
            &mut first_divergence,
        );
    }
    compare_stage(
        "vision.projector",
        &iree_diagnostics.projected_features,
        &host_projected,
        VISION,
        &mut first_divergence,
    );

    progress("compare production prepared sparse-add merge");
    let tokenizer = crate::tokenizer::load_tokenizer(&model).expect("load Molmo tokenizer");
    let token_ids = tokenizer
        .encode("Describe the image.", true)
        .expect("tokenize Molmo prompt")
        .into_iter()
        .map(|token| i32::try_from(token).expect("token id fits i32"))
        .collect::<Vec<_>>();
    let host_prepared = host
        .prepare(&token_ids, std::slice::from_ref(&image))
        .expect("prepare eager MLX Molmo prefill");
    let iree_prepared = production
        .prepare(&token_ids, &[image])
        .expect("prepare production IREE Molmo prefill");
    assert_eq!(iree_prepared.token_ids, host_prepared.token_ids);
    assert_eq!(iree_prepared.positions, host_prepared.positions);
    assert_eq!(iree_prepared.attention_bias, host_prepared.attention_bias);
    assert_eq!(iree_prepared.modalities, host_prepared.modalities);
    let visual_rows = host_processor
        .image_input_idx
        .iter()
        .filter_map(|&position| usize::try_from(position).ok())
        .map(|position| position + 1)
        .collect::<BTreeSet<_>>();
    let hidden = host_prepared.embeddings.shape[2];
    let host_values = prepared_f32(&host_prepared);
    let iree_values = prepared_f32(&iree_prepared);
    let mut host_non_visual = Vec::new();
    let mut iree_non_visual = Vec::new();
    let mut host_visual = Vec::new();
    let mut iree_visual = Vec::new();
    for row in 0..host_prepared.sequence_len {
        let range = row * hidden..(row + 1) * hidden;
        if visual_rows.contains(&row) {
            host_visual.extend_from_slice(&host_values[range.clone()]);
            iree_visual.extend_from_slice(&iree_values[range]);
        } else {
            host_non_visual.extend_from_slice(&host_values[range.clone()]);
            iree_non_visual.extend_from_slice(&iree_values[range]);
        }
    }
    compare_stage(
        "prepared.non_visual_rows",
        &iree_non_visual,
        &host_non_visual,
        EXACT,
        &mut first_divergence,
    );
    compare_stage(
        "prepared.sparse_add_visual_rows",
        &iree_visual,
        &host_visual,
        VISION,
        &mut first_divergence,
    );

    progress("compare one resident decoder's layer-0 KV, all KV, and logits");
    let mut engine = mlxcel_xla::LlavaReferenceDiagnosticEngine::load(
        &model,
        &device,
        host_prepared.sequence_len,
    )
    .expect("load one Molmo IREE decoder diagnostic engine");
    let host_decoder = engine
        .capture(&host_prepared, 64, 1)
        .expect("capture eager-prepared decoder boundary");
    let iree_decoder = engine
        .capture(&iree_prepared, 64, 1)
        .expect("capture IREE-prepared decoder boundary");
    assert_eq!(host_decoder.prefill.kv_width, 64);
    assert_eq!(iree_decoder.prefill.kv_width, 64);
    assert!(host_decoder.prefill.kv.len() >= 128);
    assert!(iree_decoder.prefill.kv.len() >= 128);
    compare_stage(
        "decoder.layer0_kv",
        &iree_decoder.prefill.kv[..128],
        &host_decoder.prefill.kv[..128],
        VISION,
        &mut first_divergence,
    );
    compare_stage(
        "decoder.all_layer_kv",
        &iree_decoder.prefill.kv,
        &host_decoder.prefill.kv,
        VISION,
        &mut first_divergence,
    );
    compare_stage(
        "decoder.prefill_logits",
        &iree_decoder.prefill.logits,
        &host_decoder.prefill.logits,
        VISION,
        &mut first_divergence,
    );
    assert_eq!(
        argmax(&iree_decoder.prefill.logits),
        argmax(&host_decoder.prefill.logits),
        "prepared-path top-1 token diverged"
    );
    if let Some(first_divergence) = first_divergence {
        panic!("Molmo v1 eager MLX/IREE first divergence: {first_divergence}");
    }
}
