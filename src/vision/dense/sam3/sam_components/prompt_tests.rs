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

use super::*;

#[test]
fn block_float_point_embedding_concatenates_with_bfloat16_sparse_tokens() {
    let source = mlxcel_core::from_slice_f32(&[0.25; 4 * 32], &[4, 32]);
    let quantized = mlxcel_core::quantize_weights_with_mode(&source, 32, 4, "mxfp4");
    let mut weights = WeightMap::new();
    weights.insert(
        "prompt.point_embed.weight".into(),
        mlxcel_core::quantized_weights_w(&quantized),
    );
    weights.insert(
        "prompt.point_embed.scales".into(),
        mlxcel_core::quantized_weights_scales(&quantized),
    );

    let (embedding, output_dtype) =
        load_point_embedding(&weights, "prompt.point_embed", 4, 32, 64, 4)
            .expect("block-float point embedding");
    assert_eq!(output_dtype, mlxcel_core::dtype::BFLOAT16);

    let sparse = mlxcel_core::zeros(&[1, 0, 32], output_dtype);
    let labels = mlxcel_core::from_slice_i32(&[0], &[1, 1]);
    let embedded = embedding.forward(&labels);
    assert_eq!(mlxcel_core::array_dtype(&embedded), output_dtype);
    let sparse = mlxcel_core::concatenate(&sparse, &embedded, 1);
    mlxcel_core::eval(&sparse);
    assert_eq!(mlxcel_core::array_shape(&sparse), vec![1, 1, 32]);
    assert_eq!(mlxcel_core::array_dtype(&sparse), output_dtype);
}
