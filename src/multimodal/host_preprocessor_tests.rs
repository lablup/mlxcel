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

use std::sync::Once;

use image::DynamicImage;
use mlxcel_core::dtype;

use super::{
    FakeHostMultimodalPreprocessor, HostMultimodalPreprocessor, HostPreprocessorError,
    XlaVisionBackend, XlaVisionBackendPolicy, export_llava_prefill, export_mlx_tensor,
    load_xla_image_preprocessor, validate_processor_shape,
};
use crate::multimodal::vlm_prompt::ImageTokenBlockError;
use crate::vision::merge::merge_llava;

fn ensure_cpu_device() {
    static INIT: Once = Once::new();
    INIT.call_once(|| mlxcel_core::set_default_device(false));
}

fn images(count: usize) -> Vec<DynamicImage> {
    (0..count).map(|_| DynamicImage::new_rgb8(2, 2)).collect()
}

fn fake() -> FakeHostMultimodalPreprocessor {
    FakeHostMultimodalPreprocessor {
        image_token_id: -200,
        tokens_per_image: 2,
        hidden_size: 3,
        max_sequence_len: 32,
    }
}

#[test]
fn iree_vision_contract_policy_is_explicit_and_strict() {
    assert_eq!(
        XlaVisionBackendPolicy::from_value(None).unwrap(),
        XlaVisionBackendPolicy::Auto
    );
    assert_eq!(
        XlaVisionBackendPolicy::from_value(Some("auto")).unwrap(),
        XlaVisionBackendPolicy::Auto
    );
    assert_eq!(
        XlaVisionBackendPolicy::from_value(Some("host")).unwrap(),
        XlaVisionBackendPolicy::Host
    );
    assert_eq!(
        XlaVisionBackendPolicy::from_value(Some("iree")).unwrap(),
        XlaVisionBackendPolicy::Iree
    );
    assert!(XlaVisionBackendPolicy::from_value(Some("cuda")).is_err());
    assert_eq!(fake().backend(), XlaVisionBackend::Host);
}

#[test]
fn xla_loader_keeps_text_and_unqualified_vlm_image_capability_false() {
    for model_type in ["llama", "qwen2_vl"] {
        let model_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            model_dir.path().join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .unwrap();
        let preprocessor = load_xla_image_preprocessor(model_dir.path()).unwrap();
        assert!(
            preprocessor.is_none(),
            "{model_type} is not a qualified LLaVA host/runtime pair"
        );
    }
}

#[test]
fn xla_loader_fails_startup_for_llava_missing_required_artifacts() {
    let model_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{
            "model_type": "llava",
            "text_config": {
                "model_type": "llama",
                "hidden_size": 8,
                "max_position_embeddings": 32
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "num_hidden_layers": 1,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 1,
                "patch_size": 2,
                "image_size": 4,
                "num_channels": 3
            },
            "image_token_index": -200
        }"#,
    )
    .unwrap();

    let error = load_xla_image_preprocessor(model_dir.path())
        .err()
        .expect("qualified LLaVA without projector/vision/embedding artifacts must fail");
    assert!(
        !matches!(error, HostPreprocessorError::FamilyMismatch { .. }),
        "a qualified family must fail startup, not silently downgrade capability: {error}"
    );
}

#[test]
fn fake_preprocessor_handles_zero_one_and_multiple_placeholders() {
    let zero = fake().prepare(&[1, 2], &[]).unwrap();
    assert_eq!(zero.token_ids, vec![1, 2]);
    assert!(zero.modalities.is_empty());

    let one = fake().prepare(&[1, -200, 2], &images(1)).unwrap();
    assert_eq!(one.token_ids, vec![1, -200, -200, 2]);
    assert_eq!(one.modalities[0].item_count, 1);
    assert_eq!(one.modalities[0].token_count, 2);

    let multiple = fake().prepare(&[1, -200, 2, -200, 3], &images(2)).unwrap();
    assert_eq!(multiple.token_ids, vec![1, -200, -200, 2, -200, -200, 3]);
    assert_eq!(multiple.modalities[0].item_count, 2);
    assert_eq!(multiple.modalities[0].token_count, 4);
}

#[test]
fn fake_preprocessor_rejects_media_count_mismatch() {
    let error = fake().prepare(&[1, -200, 2], &images(2)).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::Placeholder(ImageTokenBlockError::MediaCardinality {
            placeholder_count: 1,
            image_count: 2,
        })
    ));
}

#[test]
fn fake_preprocessor_rejects_expanded_sequence_over_capacity() {
    let preprocessor = FakeHostMultimodalPreprocessor {
        max_sequence_len: 3,
        ..fake()
    };
    let error = preprocessor.prepare(&[1, -200, 2], &images(1)).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::SequenceCapacity {
            actual: 4,
            maximum: 3,
        }
    ));
}

#[test]
fn processor_shape_validation_rejects_layout_and_size_drift() {
    let error = validate_processor_shape(&[1, 224, 224, 3], 1, 224).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::ProcessorShape { .. }
    ));

    let error = validate_processor_shape(&[1, 3, 336, 336], 1, 224).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::ProcessorShape { .. }
    ));
}

#[test]
fn owned_llava_export_matches_existing_mlx_merge_fixture() {
    ensure_cpu_device();
    let text = mlxcel_core::from_slice_f32(&[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], &[1, 4, 2]);
    let ids = mlxcel_core::from_slice_i32(&[7, 42, 42, 8], &[1, 4]);
    let vision = mlxcel_core::from_slice_f32(&[10.0, 11.0, 12.0, 13.0], &[1, 2, 2]);
    let merged = merge_llava(42, &vision, &text, &ids);
    let expected_bytes = mlxcel_core::try_array_to_raw_bytes(&merged.inputs_embeds).unwrap();

    let prepared = export_llava_prefill(vec![7, 42, 42, 8], merged, 42, 1, 2, 2).unwrap();

    assert_eq!(prepared.embeddings.bytes, expected_bytes);
    assert_eq!(prepared.embeddings.shape, vec![1, 4, 2]);
    assert_eq!(
        prepared.positions,
        mlxcel_core::session::PreparedPositions::Sequential {
            start: 0,
            length: 4,
        }
    );
    assert!(prepared.attention_bias.causal);
    assert_eq!(prepared.attention_bias.tensor.shape, vec![1, 1, 1, 4]);
    assert!(
        prepared
            .attention_bias
            .tensor
            .bytes
            .iter()
            .all(|&byte| byte == 0)
    );
}

#[test]
fn mlx_export_supports_f16_bf16_and_f32_with_exact_byte_counts() {
    ensure_cpu_device();
    let values = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    for mlx_dtype in [dtype::FLOAT16, dtype::BFLOAT16, dtype::FLOAT32] {
        let array = mlxcel_core::astype(&values, mlx_dtype);
        let exported = export_mlx_tensor(&array, "test tensor").unwrap();
        assert_eq!(exported.bytes.len(), 4 * exported.dtype.size_bytes());
        assert_eq!(exported.shape, vec![1, 2, 2]);
    }
}

#[test]
fn llava_export_rejects_hidden_size_mismatch() {
    ensure_cpu_device();
    let text = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let merged = crate::vision::merge::InputEmbeddings {
        inputs_embeds: text,
        attention_mask_4d: None,
    };
    let error = export_llava_prefill(vec![1, 2], merged, 42, 0, 2, 3).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::EmbeddingShape { .. }
    ));
}
