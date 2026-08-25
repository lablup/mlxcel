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

use std::path::PathBuf;

use mlxcel_core::UniquePtr;
use mlxcel_core::utils::array_to_vec_f32;
use serde_json::json;

use super::pooling::{
    POOLING_ENV, PoolingConfig, PoolingMode, normalize_l2, pool, resolve_pooling_mode,
    truncate_dimensions,
};
use crate::test_support::env_lock::env_lock;

/// `hidden[b, l, :] = (b * 10 + l) * ones(D)` so every row is identifiable.
fn hidden(b: usize, l: usize, d: usize) -> UniquePtr<mlxcel_core::MlxArray> {
    let mut data = Vec::with_capacity(b * l * d);
    for bi in 0..b {
        for li in 0..l {
            data.extend(std::iter::repeat_n((bi * 10 + li) as f32, d));
        }
    }
    mlxcel_core::from_slice_f32(&data, &[b as i32, l as i32, d as i32])
}

fn mask(rows: &[&[i32]]) -> UniquePtr<mlxcel_core::MlxArray> {
    let b = rows.len() as i32;
    let l = rows[0].len() as i32;
    let flat: Vec<i32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    mlxcel_core::from_slice_i32(&flat, &[b, l])
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 1e-5, "index {i}: got {a}, expected {e}");
    }
}

#[test]
fn mean_pooling_ignores_padding() {
    let h = hidden(2, 3, 2);
    let m = mask(&[&[1, 1, 0], &[0, 1, 1]]);
    let out = pool(&h, &m, PoolingMode::Mean);
    assert_eq!(mlxcel_core::array_shape(&out), vec![2, 2]);
    // Row 0: mean of 0 and 1 = 0.5; row 1: mean of 11 and 12 = 11.5.
    assert_close(&array_to_vec_f32(&out), &[0.5, 0.5, 11.5, 11.5]);
}

#[test]
fn mean_pooling_all_padding_row_is_zero_not_nan() {
    let h = hidden(1, 2, 2);
    let m = mask(&[&[0, 0]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::Mean));
    assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    assert_close(&out, &[0.0, 0.0]);
}

#[test]
fn cls_pooling_picks_first_real_token_with_left_padding() {
    let h = hidden(2, 3, 2);
    let m = mask(&[&[1, 1, 1], &[0, 0, 1]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::Cls));
    // Row 0 -> index 0 (value 0); row 1 -> index 2 (value 12).
    assert_close(&out, &[0.0, 0.0, 12.0, 12.0]);
}

#[test]
fn lasttoken_pooling_right_padding() {
    let h = hidden(2, 4, 1);
    let m = mask(&[&[1, 1, 0, 0], &[1, 1, 1, 1]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::LastToken));
    assert_close(&out, &[1.0, 13.0]);
}

#[test]
fn lasttoken_pooling_left_padding() {
    let h = hidden(2, 4, 1);
    let m = mask(&[&[0, 0, 1, 1], &[0, 1, 1, 1]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::LastToken));
    assert_close(&out, &[3.0, 13.0]);
}

#[test]
fn lasttoken_all_padding_row_uses_last_index() {
    let h = hidden(2, 3, 1);
    let m = mask(&[&[0, 0, 0], &[1, 0, 0]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::LastToken));
    // Row 0 has no real token: index L - 1 = 2; row 1: index 0 = 10.
    assert_close(&out, &[2.0, 10.0]);
}

#[test]
fn max_pooling_ignores_padding() {
    // Make the padding row the largest value so a leak would show.
    let data = [1.0, 5.0, 9.0, 3.0, 2.0, 8.0];
    let h = mlxcel_core::from_slice_f32(&data, &[1, 3, 2]);
    let m = mask(&[&[1, 1, 0]]);
    let out = array_to_vec_f32(&pool(&h, &m, PoolingMode::Max));
    assert_close(&out, &[9.0, 5.0]);
}

#[test]
fn legacy_pooling_config_maps_each_flag() {
    let cases = [
        ("pooling_mode_cls_token", PoolingMode::Cls),
        ("pooling_mode_mean_tokens", PoolingMode::Mean),
        ("pooling_mode_max_tokens", PoolingMode::Max),
        ("pooling_mode_lasttoken", PoolingMode::LastToken),
    ];
    for (flag, expected) in cases {
        let mut value = json!({
            "word_embedding_dimension": 384,
            "pooling_mode_cls_token": false,
            "pooling_mode_mean_tokens": false,
            "pooling_mode_max_tokens": false,
            "pooling_mode_mean_sqrt_len_tokens": false,
            "pooling_mode_weightedmean_tokens": false,
            "pooling_mode_lasttoken": false,
            "include_prompt": true
        });
        value[flag] = json!(true);
        assert_eq!(PoolingConfig::parse(&value).unwrap(), expected, "{flag}");
    }
    // No flag set means mean.
    let none = json!({ "word_embedding_dimension": 384 });
    assert_eq!(PoolingConfig::parse(&none).unwrap(), PoolingMode::Mean);
}

#[test]
fn new_style_pooling_config() {
    for (name, expected) in [
        ("cls", PoolingMode::Cls),
        ("mean", PoolingMode::Mean),
        ("max", PoolingMode::Max),
        ("lasttoken", PoolingMode::LastToken),
    ] {
        let value = json!({ "pooling_mode": name });
        assert_eq!(PoolingConfig::parse(&value).unwrap(), expected, "{name}");
    }
}

#[test]
fn unsupported_pooling_mode_is_an_error() {
    let weighted = json!({ "pooling_mode_weightedmean_tokens": true });
    let err = PoolingConfig::parse(&weighted).unwrap_err().to_string();
    assert!(err.contains("weightedmean"), "{err}");

    let sqrt = json!({ "pooling_mode_mean_sqrt_len_tokens": true });
    let err = PoolingConfig::parse(&sqrt).unwrap_err().to_string();
    assert!(err.contains("mean_sqrt_len_tokens"), "{err}");

    let combined = json!({ "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true });
    let err = PoolingConfig::parse(&combined).unwrap_err().to_string();
    assert!(err.contains("combined"), "{err}");

    let no_prompt = json!({ "pooling_mode_mean_tokens": true, "include_prompt": false });
    let err = PoolingConfig::parse(&no_prompt).unwrap_err().to_string();
    assert!(err.contains("include_prompt"), "{err}");

    let unknown = json!({ "pooling_mode": "weightedmean" });
    let err = PoolingConfig::parse(&unknown).unwrap_err().to_string();
    assert!(err.contains("weightedmean"), "{err}");
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mlxcel_pooling_test_{name}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn pooling_config_read_reports_missing_file_as_none() {
    let dir = temp_dir("no_pooling");
    assert_eq!(PoolingConfig::read(&dir).unwrap(), None);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_pooling_mode_prefers_config_then_default_then_env() {
    let _guard = env_lock();
    // SAFETY: the env lock serializes every env-mutating test in this crate.
    unsafe { std::env::remove_var(POOLING_ENV) };

    let dir = temp_dir("resolve");
    // No config: family default wins.
    assert_eq!(
        resolve_pooling_mode(&dir, PoolingMode::LastToken).unwrap(),
        PoolingMode::LastToken
    );

    // Config beats the family default.
    std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
    std::fs::write(
        dir.join(PoolingConfig::RELATIVE_PATH),
        r#"{"pooling_mode_cls_token": true}"#,
    )
    .unwrap();
    assert_eq!(
        resolve_pooling_mode(&dir, PoolingMode::LastToken).unwrap(),
        PoolingMode::Cls
    );

    // The env override beats both.
    // SAFETY: serialized by the env lock held above.
    unsafe { std::env::set_var(POOLING_ENV, "max") };
    assert_eq!(
        resolve_pooling_mode(&dir, PoolingMode::LastToken).unwrap(),
        PoolingMode::Max
    );
    // SAFETY: serialized by the env lock held above.
    unsafe { std::env::set_var(POOLING_ENV, "bogus") };
    assert!(resolve_pooling_mode(&dir, PoolingMode::Mean).is_err());
    // SAFETY: serialized by the env lock held above.
    unsafe { std::env::remove_var(POOLING_ENV) };
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn normalize_unit_norm_and_eps_guard() {
    let x = mlxcel_core::from_slice_f32(&[3.0, 4.0, 0.0, 0.0], &[2, 2]);
    let out = array_to_vec_f32(&normalize_l2(&x));
    assert_close(&out, &[0.6, 0.8, 0.0, 0.0]);
    let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    // The zero row divides by the epsilon, not by zero.
    assert!(out[2].is_finite() && out[3].is_finite());
}

#[test]
fn truncate_dimensions_renormalizes() {
    let x = mlxcel_core::from_slice_f32(&[3.0, 4.0, 12.0], &[1, 3]);
    let unit = normalize_l2(&x);
    let truncated = truncate_dimensions(&unit, 2);
    assert_eq!(mlxcel_core::array_shape(&truncated), vec![1, 2]);
    let before = array_to_vec_f32(&truncated);
    let norm_before = (before[0] * before[0] + before[1] * before[1]).sqrt();
    assert!(
        norm_before < 1.0,
        "truncation shrinks the norm: {norm_before}"
    );
    let after = array_to_vec_f32(&normalize_l2(&truncated));
    assert_close(&after, &[0.6, 0.8]);

    // Truncating to the full width is the identity.
    let full = array_to_vec_f32(&truncate_dimensions(&x, 3));
    assert_close(&full, &[3.0, 4.0, 12.0]);
}

#[test]
fn pooling_mode_parses_and_displays() {
    for mode in [
        PoolingMode::Cls,
        PoolingMode::Mean,
        PoolingMode::Max,
        PoolingMode::LastToken,
    ] {
        let parsed: PoolingMode = mode.as_str().parse().unwrap();
        assert_eq!(parsed, mode);
        assert_eq!(mode.to_string(), mode.as_str());
    }
    assert!("weightedmean".parse::<PoolingMode>().is_err());
}
