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

use super::*;

const VISUAL_MAX_ABS: f32 = 0.25;
const VISUAL_MIN_COSINE: f64 = 0.99;
const VISUAL_MAX_NORMALIZED_RMSE: f64 = 0.05;

fn real_inputs() -> (std::path::PathBuf, Vec<i32>, DynamicImage) {
    let model_path = std::env::var_os("MLXCEL_TEST_MOLMO_MODEL")
        .map(std::path::PathBuf::from)
        .expect("MLXCEL_TEST_MOLMO_MODEL is required");
    let tokenizer = crate::tokenizer::load_tokenizer(&model_path).unwrap();
    let token_ids = tokenizer
        .encode("Describe the image.", true)
        .unwrap()
        .into_iter()
        .map(|token| i32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    let image = image::open("tests/fixtures/test_image.png").unwrap();
    (model_path, token_ids, image)
}

fn assert_real_prepared(prepared: &PreparedPrefill) {
    assert_eq!(prepared.embeddings.dtype, PreparedTensorDType::Float32);
    assert_eq!(prepared.embeddings.shape[0], 1);
    assert_eq!(prepared.embeddings.shape[1], prepared.token_ids.len());
    assert_eq!(prepared.modalities[0].family, "molmo-v1");
    assert_eq!(prepared.modalities[0].item_count, 1);
}

fn real_parity_inputs() -> (std::path::PathBuf, Vec<i32>, DynamicImage, String) {
    let model_path = std::env::var_os("MLXCEL_TEST_MOLMO_MODEL")
        .map(std::path::PathBuf::from)
        .expect("MLXCEL_TEST_MOLMO_MODEL is required");
    let image_path = std::env::var_os("MLXCEL_TEST_MOLMO_IMAGE")
        .map(std::path::PathBuf::from)
        .expect("MLXCEL_TEST_MOLMO_IMAGE is required");
    let device = std::env::var("MLXCEL_TEST_MOLMO_IREE_DEVICE")
        .expect("MLXCEL_TEST_MOLMO_IREE_DEVICE is required");
    let tokenizer = crate::tokenizer::load_tokenizer(&model_path).unwrap();
    let token_ids = tokenizer
        .encode("Describe the image.", true)
        .unwrap()
        .into_iter()
        .map(|token| i32::try_from(token).unwrap())
        .collect::<Vec<_>>();
    let image = image::open(&image_path).unwrap_or_else(|error| {
        panic!(
            "failed to load MLXCEL_TEST_MOLMO_IMAGE {}: {error}",
            image_path.display()
        )
    });
    (model_path, token_ids, image, device)
}

fn prepared_f32(prepared: &PreparedPrefill) -> Vec<f32> {
    assert_eq!(prepared.embeddings.dtype, PreparedTensorDType::Float32);
    prepared
        .embeddings
        .bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| {
            f32::from_ne_bytes(
                bytes
                    .try_into()
                    .expect("chunks_exact yields one native f32"),
            )
        })
        .collect()
}

#[test]
fn prompt_format_matches_molmo_v1_processor_contract() {
    assert_eq!(
        format_prompt("<|image|> Describe the image."),
        " User: Describe the image. Assistant:"
    );
    assert_eq!(
        format_prompt("User: Describe the image. Assistant:"),
        " User: Describe the image. Assistant:"
    );
}

#[test]
#[ignore = "requires MLXCEL_TEST_MOLMO_MODEL and the checked-in image fixture"]
fn real_checkpoint_builds_owned_sparse_add_prefill() {
    mlxcel_core::set_default_device(false);
    let (model_path, token_ids, image) = real_inputs();
    let preprocessor = MolmoHostPreprocessor::load(&model_path).unwrap();
    let prepared = preprocessor.prepare(&token_ids, &[image]).unwrap();
    assert_real_prepared(&prepared);
}

#[cfg(feature = "xla-iree")]
#[test]
#[ignore = "requires MLXCEL_TEST_MOLMO_MODEL and a real IREE runtime"]
fn real_checkpoint_builds_native_iree_sparse_add_prefill() {
    mlxcel_core::set_default_device(false);
    let (model_path, token_ids, image) = real_inputs();
    let device =
        std::env::var("MLXCEL_TEST_MOLMO_IREE_DEVICE").unwrap_or_else(|_| "cuda".to_string());
    let preprocessor = MolmoHostPreprocessor::load_iree(&model_path, &device).unwrap();
    assert_eq!(preprocessor.backend(), super::super::XlaVisionBackend::Iree);
    let prepared = preprocessor.prepare(&token_ids, &[image]).unwrap();
    assert_real_prepared(&prepared);
}

#[cfg(feature = "xla-iree")]
#[test]
#[ignore = "requires MLXCEL_TEST_MOLMO_MODEL, MLXCEL_TEST_MOLMO_IMAGE, and MLXCEL_TEST_MOLMO_IREE_DEVICE"]
fn real_checkpoint_host_and_iree_prepared_prefill_match() {
    mlxcel_core::set_default_device(false);
    let (model_path, token_ids, image, device) = real_parity_inputs();
    let host = MolmoHostPreprocessor::load(&model_path).unwrap();

    // `image_input_idx` is relative to the image-token block. Prepared
    // prefills prepend BOS, so active visual rows are shifted by one.
    let processor_output = host.processor.preprocess_image(&image);
    let active_count = processor_output
        .image_input_idx
        .iter()
        .filter(|&&position| position >= 0)
        .count();
    let visual_rows = processor_output
        .image_input_idx
        .iter()
        .filter_map(|&position| {
            usize::try_from(position)
                .ok()
                .and_then(|position| position.checked_add(1))
        })
        .collect::<BTreeSet<_>>();
    assert!(!visual_rows.is_empty(), "fixture must contain visual rows");
    assert_eq!(
        visual_rows.len(),
        active_count,
        "fixture must retain unique authoritative visual targets"
    );

    let host_prepared = host
        .prepare(&token_ids, std::slice::from_ref(&image))
        .unwrap();
    let iree = MolmoHostPreprocessor::load_iree(&model_path, &device).unwrap();
    let iree_prepared = iree.prepare(&token_ids, &[image]).unwrap();
    assert_real_prepared(&host_prepared);
    assert_real_prepared(&iree_prepared);

    assert_eq!(iree_prepared.token_ids, host_prepared.token_ids);
    assert_eq!(
        iree_prepared.embeddings.shape,
        host_prepared.embeddings.shape
    );
    assert_eq!(
        iree_prepared.embeddings.dtype,
        host_prepared.embeddings.dtype
    );
    assert_eq!(iree_prepared.positions, host_prepared.positions);
    assert_eq!(iree_prepared.attention_bias, host_prepared.attention_bias);
    assert_eq!(iree_prepared.modalities, host_prepared.modalities);
    assert_eq!(iree_prepared.adapter_mode, host_prepared.adapter_mode);

    let host_values = prepared_f32(&host_prepared);
    let iree_values = prepared_f32(&iree_prepared);
    assert_eq!(iree_values.len(), host_values.len());
    let hidden_size = host_prepared.embeddings.shape[2];
    for row in 0..host_prepared.sequence_len {
        assert!(
            row < host_prepared.token_ids.len(),
            "prepared row must have a logical token"
        );
        if !visual_rows.contains(&row) {
            let start = row * hidden_size;
            let end = start + hidden_size;
            assert_eq!(
                &iree_values[start..end],
                &host_values[start..end],
                "non-visual prepared row {row} must be bit-exact"
            );
        }
    }

    let mut max_abs = 0.0f32;
    let mut squared_error = 0.0f64;
    let mut host_square = 0.0f64;
    let mut iree_square = 0.0f64;
    let mut dot = 0.0f64;
    for &row in &visual_rows {
        assert!(
            row < host_prepared.sequence_len,
            "visual target {row} must fit the prepared sequence"
        );
        let start = row * hidden_size;
        let end = start + hidden_size;
        for (&host_value, &iree_value) in
            host_values[start..end].iter().zip(&iree_values[start..end])
        {
            assert!(host_value.is_finite() && iree_value.is_finite());
            let difference = f64::from(iree_value) - f64::from(host_value);
            max_abs = max_abs.max((iree_value - host_value).abs());
            squared_error += difference * difference;
            host_square += f64::from(host_value) * f64::from(host_value);
            iree_square += f64::from(iree_value) * f64::from(iree_value);
            dot += f64::from(host_value) * f64::from(iree_value);
        }
    }
    assert!(host_square > 0.0 && iree_square > 0.0);
    let normalized_rmse = (squared_error / host_square).sqrt();
    let cosine = dot / (host_square.sqrt() * iree_square.sqrt());
    eprintln!(
        "Molmo host/IREE visual prepared parity: rows={} max_abs={max_abs:.8} \
         cosine={cosine:.8} normalized_rmse={normalized_rmse:.8}",
        visual_rows.len()
    );

    // The host reference retains checkpoint BF16/F16 rounding while IREE
    // may use F32 contractions on local-task. The combined envelope remains
    // tight enough to reject projected-row reordering or a materially different
    // vision result without demanding bit equality across the 22-layer ViT,
    // attention pooler, and SwiGLU projector.
    assert!(
        max_abs <= VISUAL_MAX_ABS
            && cosine >= VISUAL_MIN_COSINE
            && normalized_rmse <= VISUAL_MAX_NORMALIZED_RMSE,
        "Molmo visual prepared parity exceeded the documented envelope: \
         max_abs={max_abs}, cosine={cosine}, normalized_rmse={normalized_rmse}"
    );
}
