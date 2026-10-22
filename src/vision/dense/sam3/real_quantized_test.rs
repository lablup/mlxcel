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

use serde_json::{Map, Value};
use std::path::Path;

use super::super::{DenseInput, DensePredictor};
use super::predictor::Sam3Predictor;

const CAT_IMAGE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sam3/cats.png"
));

#[test]
#[ignore = "loads the real affine 4-bit SAM 3 checkpoint and runs inference"]
fn pinned_affine_4bit_checkpoint_runs_inference() {
    let Ok(model_dir) = std::env::var("MLXCEL_SAM3_4BIT_MODEL") else {
        eprintln!("skip: MLXCEL_SAM3_4BIT_MODEL is not set");
        return;
    };
    let _runtime = crate::initialize_runtime_checked().expect("initialize MLX runtime");
    let predictor =
        Sam3Predictor::load(Path::new(&model_dir)).expect("load 4-bit SAM 3 checkpoint");
    assert_eq!(predictor.tasks(), &["detect", "segment", "track"]);

    let image = image::load_from_memory(CAT_IMAGE).expect("decode cat fixture");
    let output = predictor
        .predict(
            &DenseInput::Image(image),
            &Map::from_iter([
                ("task".into(), Value::String("segment".into())),
                (
                    "prompts".into(),
                    Value::Array(vec![Value::String("a cat".into())]),
                ),
                ("threshold".into(), Value::from(0.3)),
            ]),
        )
        .expect("run 4-bit SAM 3 inference");
    assert!(!output.instances.is_empty(), "4-bit model found no cats");
    assert!(output.instances.iter().all(|instance| {
        instance.score.is_finite()
            && instance
                .bbox
                .iter()
                .all(|coordinate| coordinate.is_finite())
            && instance.mask_index.is_some()
    }));
    let masks = output.tensors.get("masks").expect("4-bit masks");
    let shape = mlxcel_core::array_shape(masks);
    assert_eq!(shape[0], output.instances.len() as i32);
    assert_eq!(&shape[1..], &[480, 640]);
    assert!(
        mlxcel_core::utils::array_to_vec_f32(masks)
            .iter()
            .any(|value| *value > 0.5),
        "4-bit model returned only empty masks"
    );
}
