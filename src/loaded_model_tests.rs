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

use super::{
    VlmRuntimeRef, image_token_block_info_from_runtime, standard_image_token_block_info,
    vision_module_from_runtime,
};
use crate::vision::connectors::MultiModalConnector;
use crate::vision::encoders::{VisionEncoder, VisionEncoderOutput};
use crate::vision::processors::ImageProcessor;
use crate::vision::{MergeStrategy, VisionModule};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

struct DummyEncoder;

impl VisionEncoder for DummyEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        VisionEncoderOutput {
            hidden_states: mlxcel_core::ones(&[1, 1, 1], dtype::FLOAT32),
        }
    }
}

struct DummyConnector;

impl MultiModalConnector for DummyConnector {
    fn forward(&self, _vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::ones(&[1, 1, 1], dtype::FLOAT32)
    }
}

struct DummyProcessor;

impl ImageProcessor for DummyProcessor {
    fn preprocess(&self, _images: &[image::DynamicImage]) -> UniquePtr<MlxArray> {
        mlxcel_core::ones(&[1, 3, 1, 1], dtype::FLOAT32)
    }
}

fn test_vision_module() -> VisionModule {
    VisionModule {
        encoder: Box::new(DummyEncoder),
        connector: Box::new(DummyConnector),
        processor: Box::new(DummyProcessor),
        image_token_id: 99,
        pad_token_id: 0,
        hidden_size: 64,
        boi_token_id: 10,
        eoi_token_id: 11,
        mm_tokens_per_image: 256,
        merge_strategy: MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: Some(12),
        suffix_tokens: vec![13, 14],
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
        pixtral_layout: None,
    }
}

#[test]
fn standard_image_token_block_info_preserves_vision_module_fields() {
    let info = standard_image_token_block_info(&test_vision_module());

    assert!(info.use_boi_eoi);
    assert_eq!(info.image_token_id, 99);
    assert_eq!(info.mm_tokens_per_image, 256);
    assert_eq!(info.boi_token_id, 10);
    assert_eq!(info.eoi_token_id, 11);
    assert!(info.has_bos);
    assert_eq!(info.separator_token_id, Some(12));
    assert_eq!(info.suffix_tokens, vec![13, 14]);
}

#[test]
fn standard_image_token_block_info_disables_boi_eoi_when_tokens_are_absent() {
    let mut module = test_vision_module();
    module.boi_token_id = 0;
    module.eoi_token_id = 0;

    let info = standard_image_token_block_info(&module);

    assert!(!info.use_boi_eoi);
    assert_eq!(info.boi_token_id, 0);
    assert_eq!(info.eoi_token_id, 0);
}

#[test]
fn vision_module_from_runtime_returns_standard_module_reference() {
    let module = test_vision_module();
    let returned = vision_module_from_runtime(VlmRuntimeRef::Standard(&module)).unwrap();

    assert_eq!(returned.image_token_id, module.image_token_id);
    assert_eq!(returned.mm_tokens_per_image, module.mm_tokens_per_image);
}

#[test]
fn image_token_block_info_from_runtime_uses_standard_runtime_path() {
    let module = test_vision_module();
    let info = image_token_block_info_from_runtime(VlmRuntimeRef::Standard(&module)).unwrap();

    assert_eq!(info.image_token_id, module.image_token_id);
    assert_eq!(info.mm_tokens_per_image, module.mm_tokens_per_image);
    assert_eq!(info.separator_token_id, module.separator_token_id);
}
