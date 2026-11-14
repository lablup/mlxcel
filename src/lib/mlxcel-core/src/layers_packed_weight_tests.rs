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

use super::{FusedQKVLinear, Linear, UnifiedLinear};
use crate::{dtype, ffi, weights::WeightMap};

fn packed_plane(weights: &mut WeightMap, prefix: &str) {
    // Logical [4, 4], with four output rows stored in each byte.
    weights.insert(
        format!("{prefix}.weight"),
        ffi::from_bytes(&[0x55; 4], &[1, 4], dtype::UINT8),
    );
    weights.insert(
        format!("{prefix}.weight_scale"),
        ffi::from_slice_f32(&[2.0], &[1]),
    );
}

fn assert_named_refusal<T>(result: Result<T, String>, prefix: &str) {
    let error = result
        .err()
        .expect("packed rows must not become a dense matrix");
    assert!(error.contains(prefix), "{error}");
    assert!(error.contains("uint8"), "{error}");
    assert!(error.contains("weight_scale"), "{error}");
    assert!(error.contains("BitLinear"), "{error}");
}

#[test]
fn packed_bitlinear_is_refused_by_single_and_preconcatenated_loaders() {
    let mut weights = WeightMap::new();
    let prefix = "model.layers.0.self_attn.qkv_proj";
    packed_plane(&mut weights, prefix);
    assert_named_refusal(Linear::from_weights(&weights, prefix), prefix);
    assert_named_refusal(UnifiedLinear::from_weights(&weights, prefix, 64, 4), prefix);
    assert_named_refusal(
        FusedQKVLinear::from_weights_fused(&weights, "model.layers.0.self_attn", 64, 4, 2, 2, 2),
        prefix,
    );
}

#[test]
fn packed_bitlinear_is_refused_in_each_separate_qkv_plane() {
    let prefix = "model.layers.0.self_attn";
    for packed in ["q_proj", "k_proj", "v_proj"] {
        let mut weights = WeightMap::new();
        for plane in ["q_proj", "k_proj", "v_proj"] {
            weights.insert(
                format!("{prefix}.{plane}.weight"),
                ffi::from_slice_f32(&[1.0; 16], &[4, 4]),
            );
        }
        let packed_prefix = format!("{prefix}.{packed}");
        packed_plane(&mut weights, &packed_prefix);
        assert_named_refusal(
            FusedQKVLinear::from_weights_separate(&weights, prefix, 64, 4, 2, 2, 2),
            &packed_prefix,
        );
    }
}

#[test]
fn float_planes_with_a_scale_sidecar_keep_the_dense_path() {
    let prefix = "model.layers.0.self_attn";
    for kind in [dtype::FLOAT32, dtype::FLOAT16, dtype::BFLOAT16] {
        let mut weights = WeightMap::new();
        for plane in ["q_proj", "k_proj", "v_proj"] {
            let name = format!("{prefix}.{plane}");
            let values = ffi::from_slice_f32(&[1.0; 16], &[4, 4]);
            weights.insert(format!("{name}.weight"), ffi::astype(&values, kind));
            weights.insert(
                format!("{name}.weight_scale"),
                ffi::from_slice_f32(&[2.0], &[1]),
            );
        }
        let q = format!("{prefix}.q_proj");
        Linear::from_weights(&weights, &q).expect("float linear still loads");
        UnifiedLinear::from_weights(&weights, &q, 64, 4).expect("float fallback still loads");
        FusedQKVLinear::from_weights_separate(&weights, prefix, 64, 4, 2, 2, 2)
            .expect("float QKV still loads");
    }
}
