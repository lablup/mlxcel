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
    CONTEXT_CAPACITY_ENV, FakeHostMultimodalPreprocessor, HostMultimodalPreprocessor,
    HostPreprocessorError, XlaVisionBackend, XlaVisionBackendPolicy,
    ensure_xla_image_context_capacity, export_llava_prefill, export_mlx_tensor,
    export_qwen2_vl_prefill, load_xla_image_preprocessor, validate_processor_shape,
    xla_image_context_floor,
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
    // `llama` is text-only and `mllama` is a VLM family whose processor and
    // position contract has never been qualified for XLA. Both are capability
    // downgrades rather than errors. `qwen2_vl` is deliberately not in this set:
    // it is qualified, so a build that cannot run it is an error instead, pinned
    // by `xla_loader_rejects_qwen2_vl_without_the_iree_feature` below.
    for model_type in ["llama", "mllama"] {
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

/// Qwen2-VL is qualified for the XLA vision path but has no MLX fallback, so a
/// build without `xla-iree` has no way to embed its images. The loader reports
/// that as a build-configuration error instead of downgrading capability to
/// `Ok(None)`, which would be indistinguishable from a text-only checkpoint and
/// would start a session that silently ignores images. Gated to the build the
/// contract is about: with `xla-iree` the same config takes the IREE load path.
#[cfg(not(feature = "xla-iree"))]
#[test]
fn xla_loader_rejects_qwen2_vl_without_the_iree_feature() {
    let model_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        model_dir.path().join("config.json"),
        r#"{"model_type":"qwen2_vl"}"#,
    )
    .unwrap();
    let error = load_xla_image_preprocessor(model_dir.path())
        .err()
        .expect("qwen2_vl without xla-iree must fail startup, not downgrade image capability");
    let HostPreprocessorError::InvalidConfig(message) = &error else {
        panic!("expected a build-configuration error, got {error:?}");
    };
    assert!(
        message.contains("requires the xla-iree feature"),
        "the message must name the missing feature: {message}"
    );
    assert!(
        message.contains("--features xla-iree"),
        "the message must carry the actionable rebuild instruction: {message}"
    );
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

#[test]
fn qwen2_vl_export_builds_exact_mrope_positions_and_delta() {
    ensure_cpu_device();
    let merged = crate::vision::merge::InputEmbeddings {
        inputs_embeds: mlxcel_core::from_slice_f32(&[0.0; 12], &[1, 6, 2]),
        attention_mask_4d: None,
    };
    let prepared = export_qwen2_vl_prefill(
        vec![10, 42, 42, 42, 42, 11],
        merged,
        &[(1, 4, 4)],
        42,
        43,
        2,
        2,
    )
    .unwrap();

    let mlxcel_core::session::PreparedPositions::Mrope3D { tensor, rope_delta } =
        prepared.positions
    else {
        panic!("Qwen2-VL must export M-RoPE positions");
    };
    assert_eq!(rope_delta, -2);
    assert_eq!(tensor.shape, vec![3, 6]);
    let positions = tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        positions,
        vec![
            0, 1, 1, 1, 1, 3, // temporal axis
            0, 1, 1, 2, 2, 3, // height axis
            0, 1, 2, 1, 2, 3, // width axis
        ]
    );
    assert_eq!(prepared.modalities[0].family, "qwen2_vl");
    assert_eq!(prepared.modalities[0].item_count, 1);
    assert_eq!(prepared.modalities[0].token_count, 4);
}

#[test]
fn qwen2_vl_export_rejects_video_and_cross_image_run_drift() {
    ensure_cpu_device();
    let input = || crate::vision::merge::InputEmbeddings {
        inputs_embeds: mlxcel_core::from_slice_f32(&[0.0; 12], &[1, 6, 2]),
        attention_mask_4d: None,
    };
    let video = export_qwen2_vl_prefill(
        vec![10, 43, 42, 42, 42, 11],
        input(),
        &[(1, 4, 4)],
        42,
        43,
        2,
        2,
    )
    .unwrap_err();
    assert!(matches!(video, HostPreprocessorError::InvalidConfig(_)));

    let split_runs = export_qwen2_vl_prefill(
        vec![42, 42, 10, 42, 42, 11],
        input(),
        &[(1, 4, 4)],
        42,
        43,
        2,
        2,
    )
    .unwrap_err();
    assert!(matches!(
        split_runs,
        HostPreprocessorError::InvalidConfig(_)
    ));
}

/// The pinned 4B checkpoint's geometry, so the expected numbers below are the
/// ones a real Molmo2 load would produce rather than invented ones.
fn molmo2_checkpoint_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"molmo2","image_patch_id":151938}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"max_crops":8,"patch_size":14,"pooling_size":[2,2],"size":{"height":378,"width":378}}"#,
    )
    .unwrap();
    dir
}

#[test]
fn molmo2_image_floor_is_the_tallest_tiling_not_the_square_one() {
    let dir = molmo2_checkpoint_dir();
    let floor = xla_image_context_floor(dir.path()).expect("Molmo2 geometry must derive a floor");
    // 8x1 tiling: low-res 14*(14+1)=210, high-res 108*(14+1)=1620, framing 4.
    assert_eq!(floor, 1834);
    // A square image reaches only 424 (210 + 210 + 4), which is what the
    // 224x224 fixture produces. Sizing on that would reject tall photographs,
    // so the floor must be strictly larger.
    assert!(
        floor > 424,
        "floor {floor} must exceed the square-image case"
    );
}

#[test]
fn a_checkpoint_without_a_derived_formula_has_no_floor() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen2_vl"}"#,
    )
    .unwrap();
    assert_eq!(xla_image_context_floor(dir.path()), None);
    // No floor means no guard, not a rejection.
    assert!(ensure_xla_image_context_capacity(dir.path(), 256, false).is_ok());
}

#[test]
fn a_missing_preprocessor_config_reports_no_floor_instead_of_guessing() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"molmo2","image_patch_id":151938}"#,
    )
    .unwrap();
    assert_eq!(xla_image_context_floor(dir.path()), None);
}

#[test]
fn the_default_capacity_is_rejected_with_the_derived_requirement() {
    let dir = molmo2_checkpoint_dir();
    let error = ensure_xla_image_context_capacity(dir.path(), 256, false)
        .err()
        .expect("a graph that cannot admit any image must fail at startup");
    let HostPreprocessorError::InvalidConfig(message) = &error else {
        panic!("expected a configuration error, got {error:?}");
    };
    assert!(
        message.contains("1834"),
        "the message must carry the derived requirement: {message}"
    );
    assert!(
        message.contains(CONTEXT_CAPACITY_ENV),
        "the message must name the variable to set: {message}"
    );
    assert!(
        message.contains("256"),
        "the message must show the capacity that was rejected: {message}"
    );
}

#[test]
fn an_operator_pinned_capacity_is_never_second_guessed() {
    let dir = molmo2_checkpoint_dir();
    // Text-only serving from a VLM checkpoint is a real workload, and capacity
    // is what every decode step attends over, so a deliberately small graph
    // must start.
    assert!(ensure_xla_image_context_capacity(dir.path(), 256, true).is_ok());
}

#[test]
fn a_graph_that_fits_the_worst_case_image_starts_without_pinning() {
    let dir = molmo2_checkpoint_dir();
    assert!(ensure_xla_image_context_capacity(dir.path(), 1834, false).is_ok());
    assert!(ensure_xla_image_context_capacity(dir.path(), 1833, false).is_err());
}

#[cfg(feature = "xla-backend")]
#[test]
fn the_capacity_variable_name_matches_the_xla_crate() {
    // The name is duplicated so this guard compiles without the xla crate; this
    // catches a rename on either side.
    assert_eq!(CONTEXT_CAPACITY_ENV, mlxcel_xla::CONTEXT_CAPACITY_ENV);
}
