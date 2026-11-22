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

//! The identity hyper-connection math, pinned against values computed by hand
//! from the formulas in the module docs (issue #2354).
//!
//! Fixture: `hc_mult = 2`, `hidden = 2`, one position, residual streams
//! `[1, 2]` and `[3, 4]`. `rms(flat) = sqrt(7.5 + 1e-5)`, so the normalized
//! flat vector is `n = [0.36515, 0.73030, 1.09544, 1.46059]`. `hc_fn` is the
//! identity, so the mixes are `n` itself: `n[0..2]` drive `pre`, `n[2..4]`
//! drive `post`. `hc_eps = 0.25` and `hc_magnitude = 3.0` are exaggerated so
//! that dropping the `+ hc_eps` or hard-coding the DeepSeek-V4 magnitude of 2
//! moves every expected value far outside the tolerance.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{IdentityHyperConnection, IdentityHyperHead, hc_expand};

const HC: i32 = 2;
const HIDDEN: i32 = 2;
const HC_EPS: f32 = 0.25;
const HC_MAGNITUDE: f32 = 3.0;
const NORM_EPS: f32 = 1e-5;
const TOLERANCE: f32 = 2e-5;

fn array(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, shape)
}

fn values(array: &MlxArray) -> Vec<f32> {
    let array = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&array);
    mlxcel_core::utils::array_to_vec_f32(&array)
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= TOLERANCE * e.abs().max(1.0),
            "{what}[{i}]: got {a}, expected {e}"
        );
    }
}

fn residual() -> UniquePtr<MlxArray> {
    array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, HC, HIDDEN])
}

fn identity(n: usize) -> Vec<f32> {
    (0..n * n)
        .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
        .collect()
}

fn connection_weights() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert("hc.hc_fn".to_string(), array(&identity(4), &[4, 4]));
    weights.insert(
        "hc.hc_base".to_string(),
        array(&[0.1, -0.2, 0.3, -0.4], &[4]),
    );
    weights.insert("hc.hc_scale".to_string(), array(&[2.0, 0.5], &[2]));
    weights
}

fn connection() -> IdentityHyperConnection {
    IdentityHyperConnection::from_weights(
        &connection_weights(),
        "hc",
        HIDDEN,
        HC,
        HC_EPS,
        HC_MAGNITUDE,
        NORM_EPS,
    )
    .expect("identity hyper connection loads")
}

/// `pre = sigmoid(n[:2] * 2.0 + [0.1, -0.2]) + 0.25 = [0.946418, 1.029128]`,
/// `post = 3.0 * sigmoid(n[2:] * 0.5 + [0.3, -0.4]) = [2.100267, 1.745494]`,
/// `collapsed = pre[0] * [1, 2] + pre[1] * [3, 4] = [4.033802, 6.009347]`.
#[test]
fn pre_adds_hc_eps_and_post_scales_by_hc_magnitude() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let (collapsed, post) = connection().forward(&residual());

    assert_eq!(mlxcel_core::array_shape(&collapsed), [1, 1, HIDDEN]);
    assert_eq!(mlxcel_core::array_shape(&post), [1, 1, HC]);
    assert_close(&values(&collapsed), &[4.033_802, 6.009_347], "collapsed");
    assert_close(&values(&post), &[2.100_267, 1.745_494], "post");
}

/// `h' = h + post[..., None] * out`: with `out = [10, -20]` the first stream
/// becomes `[1, 2] + 2.100267 * [10, -20]` and the second
/// `[3, 4] + 1.745494 * [10, -20]`. No stream reads another.
#[test]
fn expand_adds_the_gated_output_to_each_stream_without_mixing() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let residual = residual();
    let (_, post) = connection().forward(&residual);
    let out = array(&[10.0, -20.0], &[1, 1, HIDDEN]);

    let expanded = hc_expand(&out, &residual, &post);
    assert_eq!(mlxcel_core::array_shape(&expanded), [1, 1, HC, HIDDEN]);
    assert_close(
        &values(&expanded),
        &[22.002_673, -40.005_346, 20.454_944, -30.909_888],
        "expanded",
    );
}

/// A zero sublayer output leaves the residual bit-identical. The DeepSeek-V4
/// expand is `post * x + comb.T @ residual` with a Sinkhorn-normalized `comb`
/// that is never the identity (`softmax + hc_eps`), so substituting it mixes
/// the two streams here and this fails.
#[test]
fn expand_with_a_zero_output_is_the_identity_on_the_residual() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let residual = residual();
    let (_, post) = connection().forward(&residual);
    let zero = array(&[0.0, 0.0], &[1, 1, HIDDEN]);

    let expanded = hc_expand(&zero, &residual, &post);
    assert_eq!(values(&expanded), vec![1.0, 2.0, 3.0, 4.0]);
}

/// The collapsed input and the expanded residual come back in the activation
/// dtype; `post` stays float32.
#[test]
fn outputs_keep_the_activation_dtype() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let f16 = mlxcel_core::dtype::FLOAT16;
    let residual = mlxcel_core::astype(&residual(), f16);
    let (collapsed, post) = connection().forward(&residual);
    assert_eq!(mlxcel_core::array_dtype(&collapsed), f16);
    assert_eq!(mlxcel_core::array_dtype(&post), mlxcel_core::dtype::FLOAT32);
    let expanded = hc_expand(&collapsed, &residual, &post);
    assert_eq!(mlxcel_core::array_dtype(&expanded), f16);
}

/// Head: `pre = sigmoid(n[:2] * 1.5 + [0.2, -0.3]) + 0.25 = [0.928682,
/// 0.938999]`, collapsed `= [3.74568, 5.613361]`.
#[test]
fn head_collapses_with_its_own_gate() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut weights = WeightMap::new();
    let mut head_fn = identity(4);
    head_fn.truncate(8);
    weights.insert("head.hc_head_fn".to_string(), array(&head_fn, &[2, 4]));
    weights.insert("head.hc_head_base".to_string(), array(&[0.2, -0.3], &[2]));
    weights.insert("head.hc_head_scale".to_string(), array(&[1.5], &[1]));
    let head = IdentityHyperHead::from_weights(&weights, "head", HIDDEN, HC, HC_EPS, NORM_EPS)
        .expect("identity hyper head loads");

    let collapsed = head.forward(&residual());
    assert_eq!(mlxcel_core::array_shape(&collapsed), [1, 1, HIDDEN]);
    assert_close(&values(&collapsed), &[3.745_68, 5.613_361], "head");
}

/// A checkpoint whose `hc_*` tensors do not match `hc_mult * hidden_size` is
/// refused at load with both numbers in the message. The DeepSeek-V4 layout,
/// `[(2 + hc) * hc, hc * hidden]`, is one such mismatch.
#[test]
fn load_rejects_hc_shapes_that_disagree_with_the_config() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let weights = connection_weights();

    // Config says hc_mult 4; the tensors were built for 2.
    let err = IdentityHyperConnection::from_weights(
        &weights,
        "hc",
        HIDDEN,
        4,
        HC_EPS,
        HC_MAGNITUDE,
        NORM_EPS,
    )
    .err()
    .expect("hc_mult mismatch must be refused");
    assert!(err.contains("hc.hc_fn"), "{err}");
    assert!(err.contains("hc_mult * hidden_size = 8"), "{err}");
    assert!(err.contains("[4, 4]"), "{err}");

    let mut v4_layout = connection_weights();
    v4_layout.insert(
        "hc.hc_fn".to_string(),
        array(&[0.0; 32], &[(2 + HC) * HC, HC * HIDDEN]),
    );
    let err = IdentityHyperConnection::from_weights(
        &v4_layout,
        "hc",
        HIDDEN,
        HC,
        HC_EPS,
        HC_MAGNITUDE,
        NORM_EPS,
    )
    .err()
    .expect("the DeepSeek-V4 comb layout must be refused");
    assert!(err.contains("expected shape [4, 4]"), "{err}");

    let err = IdentityHyperHead::from_weights(&weights, "hc", HIDDEN, HC, HC_EPS, NORM_EPS)
        .err()
        .expect("a missing head tensor must be refused");
    assert!(err.contains("hc.hc_head_fn"), "{err}");
}
