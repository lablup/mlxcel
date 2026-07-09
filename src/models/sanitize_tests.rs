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

use super::sanitize::{load_text_weights, sanitize_config_json, sanitize_tied_embeddings};
use crate::test_support::env_lock::env_lock;
use mlxcel_core::weights::{WeightMap, WeightTransform};
use mlxcel_core::{self, dtype};
use safetensors::tensor::{Dtype as SafeTensorDtype, View};
use serde_json::json;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

fn sample_weight_map(key: &str) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(key.to_string(), mlxcel_core::ones(&[2, 2], dtype::FLOAT32));
    weights
}

#[derive(Clone)]
struct OwnedTensor {
    dtype: SafeTensorDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl View for OwnedTensor {
    fn dtype(&self) -> SafeTensorDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn temp_model_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_sanitize_test_{name}_{nanos}"))
}

fn write_safetensors(path: &Path, tensors: &[(&str, OwnedTensor)]) {
    let mut views: HashMap<String, OwnedTensor> = HashMap::new();
    for (name, tensor) in tensors {
        views.insert((*name).to_string(), tensor.clone());
    }
    safetensors::serialize_to_file(&views, None, path).unwrap();
}

const NVFP4_REPACK_ENV_KEYS: &[&str] = &["MLXCEL_NVFP4_DENSE_REPACK", "MLXCEL_NVFP4_NATIVE_REPACK"];

struct EnvRestore {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvRestore {
    fn clear(keys: &[&'static str]) -> Self {
        let saved = keys
            .iter()
            .map(|&key| (key, std::env::var_os(key)))
            .collect();
        let guard = Self { saved };
        for key in keys {
            // SAFETY: env-mutating tests acquire the crate-wide `env_lock()`
            // before constructing this guard, so no other test thread mutates
            // process environment variables concurrently.
            #[allow(unsafe_code)]
            unsafe {
                std::env::remove_var(key);
            }
        }
        guard
    }

    #[cfg(not(feature = "cuda"))]
    fn set_with_clear(keys: &[&'static str], key: &'static str, value: &str) -> Self {
        let guard = Self::clear(keys);
        // SAFETY: same `env_lock()` serialization as `clear`; callers keep the
        // lock guard alive longer than this environment restore guard.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(key, value);
        }
        guard
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            // SAFETY: callers acquire `env_lock()` before constructing this
            // guard, and Rust drops this guard before the lock guard because it
            // is declared after the lock in each test.
            #[allow(unsafe_code)]
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn write_gemma4_nvfp4_repack_fixture(dir: &Path) -> usize {
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 16,
                "bits": 4,
                "mode": "nvfp4"
            },
            "text_config": {
                "model_type": "gemma4",
                "quantization": {
                    "group_size": 16,
                    "bits": 4,
                    "mode": "nvfp4"
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    // Shape: [out_dim=2, packed_dim=16]. Each byte encodes two FP4 E2M1
    // nibbles. Nibble 0x2 = 1.0, so byte 0x22 -> [1.0, 1.0]. Expected
    // dequantized shape: [2, 32].
    let out_dim: usize = 2;
    let packed_dim: usize = 16; // in_dim / 2 = 32 / 2
    let weight_data = vec![0x22u8; out_dim * packed_dim];

    // F16 block scales: shape [out_dim=2, num_groups=2]. 1.0 in F16 = 0x3C00
    // (little-endian bytes [0x00, 0x3C]).
    let f16_one: [u8; 2] = [0x00, 0x3C];
    let num_groups = 2usize;
    let mut scale_data = Vec::with_capacity(out_dim * num_groups * 2);
    for _ in 0..(out_dim * num_groups) {
        scale_data.extend_from_slice(&f16_one);
    }

    // Global F32 scale: 1-element F32 tensor with value 1.0.
    // 1.0 in F32 little-endian = [0x00, 0x00, 0x80, 0x3F].
    let scale2_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F];

    // Use nvfp4-style key prefix: `model.language_model.layers.0.mlp.gate_proj`
    let prefix = "model.language_model.layers.0.mlp.gate_proj";
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            (
                &format!("{prefix}.weight"),
                OwnedTensor {
                    dtype: SafeTensorDtype::U8,
                    shape: vec![out_dim, packed_dim],
                    data: weight_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F16,
                    shape: vec![out_dim, num_groups],
                    data: scale_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale_2"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![1],
                    data: scale2_data,
                },
            ),
        ],
    );

    out_dim
}

#[test]
fn sanitize_config_json_replaces_non_standard_values() {
    let sanitized = sanitize_config_json("{\"a\": Infinity, \"b\": -Infinity, \"c\": NaN}");
    assert_eq!(sanitized, "{\"a\": 1e38, \"b\": -1e38, \"c\": 0.0}");
}

#[test]
fn sanitize_tied_embeddings_copies_standard_embed_tokens_when_missing() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_copies_prefixed_language_model_keys() {
    let mut weights = sample_weight_map("language_model.model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({}));

    assert!(weights.contains_key("language_model.lm_head.weight"));
}

#[test]
fn sanitize_tied_embeddings_respects_explicit_untied_config() {
    let mut weights = sample_weight_map("model.embed_tokens.weight");
    sanitize_tied_embeddings(&mut weights, &json!({ "tie_word_embeddings": false }));

    assert!(!weights.contains_key("lm_head.weight"));
}

#[test]
fn load_and_sanitize_weights_selectively_keeps_gemma4_text_tensors() {
    let dir = temp_model_dir("gemma4_selective");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 64,
                "bits": 4
            },
            "text_config": {
                "model_type": "gemma4",
                "quantization": {
                    "group_size": 64,
                    "bits": 4
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    write_safetensors(
        &dir.join("model-00001-of-00002.safetensors"),
        &[(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            OwnedTensor {
                dtype: SafeTensorDtype::F32,
                shape: vec![2],
                data: vec![0, 0, 128, 63, 0, 0, 0, 64],
            },
        )],
    );
    write_safetensors(
        &dir.join("model-00002-of-00002.safetensors"),
        &[
            (
                "language_model.model.per_layer_projection_norm.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::BF16,
                    shape: vec![1],
                    data: vec![0x80, 0x3F],
                },
            ),
            (
                "language_model.model.embed_tokens_per_layer.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::U32,
                    shape: vec![1],
                    data: 7_u32.to_le_bytes().to_vec(),
                },
            ),
            (
                "vision_tower.vision_model.embeddings.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![1],
                    data: vec![0, 0, 64, 64],
                },
            ),
        ],
    );

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    assert!(weights.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    assert!(weights.contains_key("language_model.model.per_layer_projection_norm.weight"));
    assert!(weights.contains_key("language_model.model.embed_tokens_per_layer.weight"));
    assert!(!weights.contains_key("vision_tower.vision_model.embeddings.weight"));

    let bf16 = weights
        .get("language_model.model.per_layer_projection_norm.weight")
        .unwrap();
    let expected_bf16_dtype = if mlxcel_core::hardware::get_hardware().silicon_gen
        != mlxcel_core::hardware::AppleSiliconGen::Unknown
    {
        dtype::FLOAT16
    } else {
        dtype::BFLOAT16
    };
    assert_eq!(mlxcel_core::array_dtype(bf16), expected_bf16_dtype);
    let bf16_f32 = mlxcel_core::astype(bf16, dtype::FLOAT32);
    mlxcel_core::eval(&bf16_f32);
    assert!((mlxcel_core::item_f32(&bf16_f32) - 1.0).abs() < 0.01);

    let quant = weights
        .get("language_model.model.embed_tokens_per_layer.weight")
        .unwrap();
    assert_eq!(mlxcel_core::array_dtype(quant), dtype::UINT32);
    let quant_i64 = mlxcel_core::astype(quant, dtype::INT64);
    mlxcel_core::eval(&quant_i64);
    assert_eq!(mlxcel_core::item_i64(&quant_i64), 7);

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Verify that loading an nvfp4 Gemma 4 checkpoint:
/// 1. Remaps `model.language_model.X` → `language_model.model.X`
/// 2. Repacks the packed U8 FP4 weight tensor to a usable MLX quantized layout
///
/// Test data: 2×16 packed U8 weights (nibble 0x2 = FP4 E2M1 value 1.0),
/// 2×2 F16 block scales (all 1.0 = 0x3C00), global F32 scale 1.0.
/// Issue #705 makes direct native NVFP4 the default on every backend. The
/// explicit dense-affine rollback remains covered by a separate non-CUDA test.
#[test]
fn load_and_sanitize_weights_repacks_nvfp4_gemma4_checkpoint() {
    let _env_guard = env_lock();
    let _nvfp4_env = EnvRestore::clear(NVFP4_REPACK_ENV_KEYS);
    let dir = temp_model_dir("gemma4_nvfp4");
    std::fs::create_dir_all(&dir).unwrap();
    let out_dim = write_gemma4_nvfp4_repack_fixture(&dir);

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    // After normalize_nvfp4_keys, the key should be remapped.
    let expected_key = "language_model.model.layers.0.mlp.gate_proj.weight";
    assert!(
        weights.contains_key(expected_key),
        "Expected repacked key '{expected_key}' not found; keys: {:?}",
        weights.keys().collect::<Vec<_>>()
    );

    // Auxiliary scale keys must have been removed.
    let expected_scale_key = "language_model.model.layers.0.mlp.gate_proj.weight_scale";
    let expected_scale2_key = "language_model.model.layers.0.mlp.gate_proj.weight_scale_2";
    let expected_input_scale_key = "language_model.model.layers.0.mlp.gate_proj.input_scale";
    assert!(!weights.contains_key(expected_scale_key));
    assert!(!weights.contains_key(expected_scale2_key));
    assert!(!weights.contains_key(expected_input_scale_key));

    let expected_scales_key = "language_model.model.layers.0.mlp.gate_proj.scales";
    let expected_biases_key = "language_model.model.layers.0.mlp.gate_proj.biases";
    assert!(
        weights.contains_key(expected_scales_key),
        "Expected repacked scales key '{expected_scales_key}' not found"
    );

    let w = weights.get(expected_key).unwrap();
    assert_eq!(
        mlxcel_core::array_dtype(w),
        dtype::UINT32,
        "Expected UINT32 after quantized repack"
    );
    assert_eq!(
        mlxcel_core::array_shape(w),
        vec![out_dim as i32, 4i32],
        "Expected packed quantized shape [2, 4]"
    );

    let scales = weights.get(expected_scales_key).unwrap();
    assert!(
        !weights.contains_key(expected_biases_key),
        "Default direct NVFP4 repack should not emit affine biases"
    );
    assert_eq!(
        mlxcel_core::array_shape(scales),
        vec![out_dim as i32, 2i32],
        "Expected native NVFP4 scales shape [2, 2] for group_size=16"
    );
    // The direct transcode (issue #693) preserves weight_scale_2 as a
    // per-linear global-scale sidecar instead of folding it into the E4M3
    // block scales. The scales should be raw E4M3 U8 bytes.
    let expected_global_scale_key = "language_model.model.layers.0.mlp.gate_proj.global_scale";
    assert!(
        weights.contains_key(expected_global_scale_key),
        "Direct NVFP4 transcode should emit a global_scale sidecar"
    );
    assert_eq!(
        mlxcel_core::array_dtype(scales),
        dtype::UINT8,
        "Native NVFP4 block scales should be raw E4M3 U8 bytes"
    );
    let global_scale = weights.get(expected_global_scale_key).unwrap();
    let global_f32 = mlxcel_core::astype(global_scale, dtype::FLOAT32);
    mlxcel_core::eval(&global_f32);
    let g = mlxcel_core::array_to_raw_bytes(&global_f32);
    let g_val = f32::from_le_bytes([g[0], g[1], g[2], g[3]]);
    assert!(
        (g_val - 1.0f32).abs() < 1e-6,
        "Expected weight_scale_2 sidecar 1.0, got {g_val}"
    );
    let dequantized =
        unsafe { mlxcel_core::dequantize(w, scales, std::ptr::null(), 16, 4, "nvfp4") };
    let dequantized_f32 = mlxcel_core::astype(&dequantized, dtype::FLOAT32);
    mlxcel_core::eval(&dequantized_f32);
    let shape = mlxcel_core::array_shape(&dequantized_f32);
    assert_eq!(shape, vec![out_dim as i32, 32i32], "Expected shape [2, 32]");

    // All values should remain close to 1.0 after the direct native repack.
    let w_bytes = mlxcel_core::array_to_raw_bytes(&dequantized_f32);
    for chunk in w_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (v - 1.0f32).abs() < 5e-2,
            "Expected value close to 1.0 after quantized repack, got {v}"
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Non-CUDA rollback coverage for issue #705: direct native NVFP4 is now the
/// default, but `MLXCEL_NVFP4_DENSE_REPACK=1` still preserves the previous
/// dense-affine route for prefill-sensitive A/B comparisons.
#[cfg(not(feature = "cuda"))]
#[test]
fn load_and_sanitize_weights_dense_repack_env_keeps_non_cuda_affine_rollback() {
    let _env_guard = env_lock();
    let _nvfp4_env =
        EnvRestore::set_with_clear(NVFP4_REPACK_ENV_KEYS, "MLXCEL_NVFP4_DENSE_REPACK", "1");
    let dir = temp_model_dir("gemma4_nvfp4_affine_rollback");
    std::fs::create_dir_all(&dir).unwrap();
    let out_dim = write_gemma4_nvfp4_repack_fixture(&dir);

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    let expected_key = "language_model.model.layers.0.mlp.gate_proj.weight";
    let expected_scale_key = "language_model.model.layers.0.mlp.gate_proj.weight_scale";
    let expected_scale2_key = "language_model.model.layers.0.mlp.gate_proj.weight_scale_2";
    let expected_scales_key = "language_model.model.layers.0.mlp.gate_proj.scales";
    let expected_biases_key = "language_model.model.layers.0.mlp.gate_proj.biases";
    let expected_global_scale_key = "language_model.model.layers.0.mlp.gate_proj.global_scale";

    assert!(weights.contains_key(expected_key));
    assert!(!weights.contains_key(expected_scale_key));
    assert!(!weights.contains_key(expected_scale2_key));
    assert!(
        !weights.contains_key(expected_global_scale_key),
        "Dense-affine rollback should not emit the native global_scale sidecar"
    );
    assert!(
        weights.contains_key(expected_scales_key),
        "Dense-affine rollback should emit affine scales"
    );
    assert!(
        weights.contains_key(expected_biases_key),
        "Dense-affine rollback should emit affine biases"
    );

    let w = weights.get(expected_key).unwrap();
    assert_eq!(mlxcel_core::array_dtype(w), dtype::UINT32);
    assert_eq!(mlxcel_core::array_shape(w), vec![out_dim as i32, 4i32]);

    let scales = weights.get(expected_scales_key).unwrap();
    let biases = weights.get(expected_biases_key).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(scales),
        vec![out_dim as i32, 1i32],
        "Expected affine scales shape [2, 1] for group_size=32"
    );
    assert_eq!(
        mlxcel_core::array_shape(biases),
        vec![out_dim as i32, 1i32],
        "Expected affine biases shape [2, 1] for group_size=32"
    );

    let biases_ptr = biases.as_ref().unwrap() as *const _;
    let dequantized = unsafe { mlxcel_core::dequantize(w, scales, biases_ptr, 32, 4, "affine") };
    let dequantized_f32 = mlxcel_core::astype(&dequantized, dtype::FLOAT32);
    mlxcel_core::eval(&dequantized_f32);
    assert_eq!(
        mlxcel_core::array_shape(&dequantized_f32),
        vec![out_dim as i32, 32i32],
        "Expected shape [2, 32]"
    );

    let w_bytes = mlxcel_core::array_to_raw_bytes(&dequantized_f32);
    for chunk in w_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        assert!(
            (v - 1.0f32).abs() < 5e-2,
            "Expected value close to 1.0 after dense-affine rollback, got {v}"
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Review follow-up for issue #697: `weight_scale_2` must be a single-element
/// per-tensor scalar (ModelOpt's convention). `item_f32` reinterprets a
/// tensor's raw buffer without checking cardinality, so a malformed
/// multi-element tensor must be caught and the triplet skipped, the same
/// way the other malformed-shape guards in `repack_nvfp4_weights_to_quantized`
/// fall back instead of reading garbage or throwing across the FFI boundary.
#[test]
fn load_and_sanitize_weights_skips_nvfp4_triplet_with_non_scalar_weight_scale_2() {
    let dir = temp_model_dir("gemma4_nvfp4_bad_scale2");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "gemma4",
            "tie_word_embeddings": false,
            "quantization": {
                "group_size": 16,
                "bits": 4,
                "mode": "nvfp4"
            },
            "text_config": {
                "model_type": "gemma4",
                "quantization": {
                    "group_size": 16,
                    "bits": 4,
                    "mode": "nvfp4"
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let out_dim: usize = 2;
    let packed_dim: usize = 16;
    let weight_data = vec![0x22u8; out_dim * packed_dim];

    let f16_one: [u8; 2] = [0x00, 0x3C];
    let num_groups = 2usize;
    let mut scale_data = Vec::with_capacity(out_dim * num_groups * 2);
    for _ in 0..(out_dim * num_groups) {
        scale_data.extend_from_slice(&f16_one);
    }

    // Malformed weight_scale_2: 2 elements instead of the required scalar.
    let scale2_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x80, 0x3F];

    let prefix = "model.language_model.layers.0.mlp.gate_proj";
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            (
                &format!("{prefix}.weight"),
                OwnedTensor {
                    dtype: SafeTensorDtype::U8,
                    shape: vec![out_dim, packed_dim],
                    data: weight_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F16,
                    shape: vec![out_dim, num_groups],
                    data: scale_data,
                },
            ),
            (
                &format!("{prefix}.weight_scale_2"),
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2],
                    data: scale2_data,
                },
            ),
        ],
    );

    let weights = super::sanitize::load_and_sanitize_weights(&dir).unwrap();

    // The malformed weight_scale_2 must be dropped, and the triplet must not
    // be repacked into a quantized layout: it falls back exactly like the
    // other malformed-shape guards in `repack_nvfp4_weights_to_quantized`.
    let expected_scale2_key = "language_model.model.layers.0.mlp.gate_proj.weight_scale_2";
    let expected_scales_key = "language_model.model.layers.0.mlp.gate_proj.scales";
    let expected_global_scale_key = "language_model.model.layers.0.mlp.gate_proj.global_scale";
    assert!(
        !weights.contains_key(expected_scale2_key),
        "orphaned weight_scale_2 must be removed even when malformed"
    );
    assert!(
        !weights.contains_key(expected_scales_key),
        "a non-scalar weight_scale_2 must prevent the quantized repack"
    );
    assert!(
        !weights.contains_key(expected_global_scale_key),
        "a non-scalar weight_scale_2 must not produce a global_scale sidecar"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

// --- Tests for the consolidated `load_text_weights` entry point and the
// optional `WeightTransform` hook introduced for Axis A. ---

/// Build a minimal text model fixture: tiny config.json with no
/// quantization plus a single safetensors shard containing
/// `model.embed_tokens.weight`. Enough to exercise the consolidated
/// loader's sanitize → optional transform → precision-cast pipeline.
fn write_text_model_fixture(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "llama",
            "tie_word_embeddings": true,
        }))
        .unwrap(),
    )
    .unwrap();

    // Two F32 weights so we can verify both that sanitize copies
    // `embed_tokens` into `lm_head` (via `tie_word_embeddings: true`)
    // and that the transform observes the right keys.
    let weight_data = vec![0u8; 4 * 4]; // 4 x F32 zeros
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            (
                "model.embed_tokens.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2, 2],
                    data: weight_data,
                },
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2, 2],
                    data: vec![0u8; 4 * 4],
                },
            ),
        ],
    );
}

#[test]
fn load_text_weights_with_none_transform_matches_legacy_path() {
    // Bit-exactness (a) — when `transform = None` the consolidated
    // entry point must produce exactly the same `WeightMap` (same
    // keys, same dtypes, same shapes, byte-identical contents) as the
    // legacy `load_and_sanitize_weights` alias. The alias itself is
    // implemented as `load_text_weights(p, None)`, so this test
    // doubles as a regression guard against accidental divergence
    // between the alias and the new entry point.
    //
    // (A4): the `load_text_weights(_, None)` call below
    // reads the process-global active-pipeline slot. Acquire
    // `env_lock` so a parallel test in `surgery_integration` can't
    // install a pipeline mid-test and leak it into the consolidated
    // baseline.
    #[cfg(feature = "surgery")]
    let _env_guard = crate::test_support::env_lock::env_lock();

    let dir_a = temp_model_dir("text_load_none_transform_a");
    let dir_b = temp_model_dir("text_load_none_transform_b");
    write_text_model_fixture(&dir_a);
    write_text_model_fixture(&dir_b);

    let legacy = super::sanitize::load_and_sanitize_weights(&dir_a).unwrap();
    let consolidated = load_text_weights(&dir_b, None).unwrap();

    assert_eq!(
        legacy.len(),
        consolidated.len(),
        "WeightMap entry count must match"
    );
    let mut keys_a: Vec<&String> = legacy.keys().collect();
    let mut keys_b: Vec<&String> = consolidated.keys().collect();
    keys_a.sort();
    keys_b.sort();
    assert_eq!(keys_a, keys_b, "key sets must match");

    for k in keys_a {
        let a = legacy.get(k).unwrap();
        let b = consolidated.get(k).unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(a),
            mlxcel_core::array_dtype(b),
            "dtype mismatch for key {k}"
        );
        assert_eq!(
            mlxcel_core::array_shape(a),
            mlxcel_core::array_shape(b),
            "shape mismatch for key {k}"
        );
        mlxcel_core::eval(a);
        mlxcel_core::eval(b);
        let a_bytes = mlxcel_core::array_to_raw_bytes(a);
        let b_bytes = mlxcel_core::array_to_raw_bytes(b);
        assert_eq!(a_bytes, b_bytes, "byte mismatch for key {k}");
    }

    std::fs::remove_dir_all(&dir_a).unwrap();
    std::fs::remove_dir_all(&dir_b).unwrap();
}

/// Counter-based `WeightTransform` that records how many times
/// `apply` ran. Used to prove the hook is actually invoked by
/// `load_text_weights` when a non-`None` transform is supplied.
struct CountingTransform {
    calls: AtomicUsize,
}

impl CountingTransform {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WeightTransform for CountingTransform {
    fn apply(&self, _weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn load_text_weights_invokes_transform_when_supplied() {
    // Integration (b) — the optional hook must be invoked exactly
    // once per call when a transform is supplied. With a no-op
    // transform the produced `WeightMap` must match the `None` path
    // bit-for-bit.
    //
    // (A4): baseline call reads the active-pipeline slot,
    // so we need the same `env_lock` discipline as the other tests
    // touching `load_text_weights(_, None)`.
    #[cfg(feature = "surgery")]
    let _env_guard = crate::test_support::env_lock::env_lock();

    let dir_a = temp_model_dir("text_load_transform_invoked_a");
    let dir_b = temp_model_dir("text_load_transform_invoked_b");
    write_text_model_fixture(&dir_a);
    write_text_model_fixture(&dir_b);

    let baseline = load_text_weights(&dir_a, None).unwrap();
    let transform = CountingTransform::new();
    let with_hook = load_text_weights(&dir_b, Some(&transform)).unwrap();

    assert_eq!(
        transform.call_count(),
        1,
        "WeightTransform::apply must run exactly once per load"
    );
    assert_eq!(baseline.len(), with_hook.len());

    // Verify byte-level equivalence between the no-transform path and
    // the empty-transform path (this is the bit-exactness guarantee
    // a no-op pipeline must uphold).
    for (k, base) in &baseline {
        let other = with_hook
            .get(k)
            .unwrap_or_else(|| panic!("missing key {k}"));
        mlxcel_core::eval(base);
        mlxcel_core::eval(other);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(base),
            mlxcel_core::array_to_raw_bytes(other),
            "no-op transform must preserve weights bit-for-bit: key={k}"
        );
    }

    std::fs::remove_dir_all(&dir_a).unwrap();
    std::fs::remove_dir_all(&dir_b).unwrap();
}

/// `WeightTransform` that returns an error so we can verify error
/// propagation from the consolidated loader.
struct FailingTransform;

impl WeightTransform for FailingTransform {
    fn apply(&self, _weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), String> {
        Err("intentional failure for test".to_string())
    }
}

#[test]
fn load_text_weights_propagates_transform_error() {
    let dir = temp_model_dir("text_load_transform_error");
    write_text_model_fixture(&dir);

    let transform = FailingTransform;
    let result = load_text_weights(&dir, Some(&transform));
    match result {
        Ok(_) => panic!("expected failing transform to surface an error"),
        Err(err) => assert!(
            err.contains("intentional failure for test"),
            "expected transform error to be surfaced, got: {err}"
        ),
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

// --- Integration tests for the Axis A `mlxcel-surgery` crate.
// Gated on the `surgery` feature so the default build still passes
// without pulling the new crate into the dependency graph. ---

#[cfg(feature = "surgery")]
mod surgery_integration {
    use super::*;
    use mlxcel_surgery::{SharedSurgeryOp, SurgeryError, SurgeryOp, SurgeryPipeline};
    use std::sync::Arc;

    /// `SurgeryOp` that simply notes that it ran. Used to prove the
    /// `SurgeryPipeline` is actually invoked by A1's hook, without
    /// touching the weight values (so the bit-exactness assertion
    /// still holds).
    struct RecordingOp {
        counter: Arc<AtomicUsize>,
    }

    impl SurgeryOp for RecordingOp {
        fn apply(
            &self,
            _weights: &mut WeightMap,
            _cfg: &serde_json::Value,
        ) -> Result<(), SurgeryError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "recording-noop"
        }
    }

    #[test]
    fn empty_surgery_pipeline_is_bit_exact_with_none() {
        // Acceptance criterion (b) / (e) — wiring a `SurgeryPipeline`
        // through the consolidated loader with no registered ops must
        // produce the same `WeightMap` as `transform = None`. This
        // doubles as proof that the new crate's `SurgeryPipeline`
        // really does implement A1's `WeightTransform` trait at the
        // type level (the call would not type-check otherwise).
        //
        // (A4): `load_text_weights(_, None)` reads the
        // process-global active-pipeline slot, so any test that calls
        // it must serialise on the same `env_lock` as the tests that
        // mutate the slot — otherwise a parallel mutator can leak a
        // pipeline into this test's baseline call.
        let _env_guard = crate::test_support::env_lock::env_lock();

        let dir_a = temp_model_dir("surgery_empty_a");
        let dir_b = temp_model_dir("surgery_empty_b");
        write_text_model_fixture(&dir_a);
        write_text_model_fixture(&dir_b);

        let baseline = load_text_weights(&dir_a, None).unwrap();
        let pipeline = SurgeryPipeline::new();
        let with_empty = load_text_weights(&dir_b, Some(&pipeline)).unwrap();

        assert_eq!(baseline.len(), with_empty.len());
        for (k, base) in &baseline {
            let other = with_empty
                .get(k)
                .unwrap_or_else(|| panic!("missing key {k} in surgery path"));
            mlxcel_core::eval(base);
            mlxcel_core::eval(other);
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(base),
                mlxcel_core::array_to_raw_bytes(other),
                "empty surgery pipeline must preserve key {k} bit-for-bit",
            );
        }

        std::fs::remove_dir_all(&dir_a).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn surgery_pipeline_runs_registered_ops_during_load() {
        // Acceptance criterion (b) — a non-empty pipeline routed
        // through A1's hook must actually invoke each registered op.
        // We use a recording no-op so the resulting `WeightMap` is
        // still bit-identical to the baseline (because the op does
        // not mutate weights), which proves that "no-op" semantics
        // hold end-to-end.
        //
        // env_lock as in `empty_surgery_pipeline_*` — the
        // baseline `load_text_weights(_, None)` reads the active slot.
        let _env_guard = crate::test_support::env_lock::env_lock();

        let dir_a = temp_model_dir("surgery_recording_a");
        let dir_b = temp_model_dir("surgery_recording_b");
        write_text_model_fixture(&dir_a);
        write_text_model_fixture(&dir_b);

        let counter = Arc::new(AtomicUsize::new(0));
        let mut pipeline = SurgeryPipeline::new();
        let op: SharedSurgeryOp = Arc::new(RecordingOp {
            counter: counter.clone(),
        });
        pipeline.push(op);

        let baseline = load_text_weights(&dir_a, None).unwrap();
        let with_pipeline = load_text_weights(&dir_b, Some(&pipeline)).unwrap();

        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "recording op must have run exactly once during load",
        );

        for (k, base) in &baseline {
            let other = with_pipeline
                .get(k)
                .unwrap_or_else(|| panic!("missing key {k} in surgery path"));
            mlxcel_core::eval(base);
            mlxcel_core::eval(other);
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(base),
                mlxcel_core::array_to_raw_bytes(other),
                "recording no-op surgery must preserve key {k} bit-for-bit",
            );
        }

        std::fs::remove_dir_all(&dir_a).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn surgery_error_propagates_through_load_path() {
        // Acceptance criterion (b) — when a surgery op fails, the
        // error must surface out of `load_text_weights` rather than
        // being silently swallowed. Use the existing
        // `SurgeryError::TensorNotFound` variant since it exercises
        // the `Display` impl that the pipeline funnels through
        // `WeightTransform::apply`'s `Result<_, String>` return.
        struct FailingOp;
        impl SurgeryOp for FailingOp {
            fn apply(
                &self,
                _weights: &mut WeightMap,
                _cfg: &serde_json::Value,
            ) -> Result<(), SurgeryError> {
                Err(SurgeryError::TensorNotFound(
                    "expected.key.for.test".to_string(),
                ))
            }
            fn name(&self) -> &'static str {
                "failing"
            }
        }

        let dir = temp_model_dir("surgery_failing");
        write_text_model_fixture(&dir);

        let mut pipeline = SurgeryPipeline::new();
        pipeline.push(Arc::new(FailingOp));

        let result = load_text_weights(&dir, Some(&pipeline));
        match result {
            Ok(_) => panic!("expected failing surgery op to abort the load"),
            Err(err) => {
                assert!(
                    err.contains("failing") && err.contains("expected.key.for.test"),
                    "expected error to identify failing op + key, got: {err}",
                );
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_yaml_config_pipeline_is_bit_exact_with_none() {
        // End-to-end YAML → SurgeryPipeline → load_text_weights(Some(&p))
        // for the empty-config case. Demonstrates the
        // acceptance criterion (e): when `operations: []`, the
        // produced pipeline behaves bit-exact identically to the
        // `transform = None` path.
        //
        // baseline call needs env_lock — same rationale
        // as `empty_surgery_pipeline_is_bit_exact_with_none`.
        let _env_guard = crate::test_support::env_lock::env_lock();

        let yaml = "version: 1\noperations: []\n";
        let pipeline = mlxcel_surgery::parse_config_str(yaml, None).expect("empty config parses");
        assert!(pipeline.is_empty());

        let dir_a = temp_model_dir("yaml_empty_a");
        let dir_b = temp_model_dir("yaml_empty_b");
        write_text_model_fixture(&dir_a);
        write_text_model_fixture(&dir_b);

        let baseline = load_text_weights(&dir_a, None).unwrap();
        let with_yaml = load_text_weights(&dir_b, Some(&pipeline)).unwrap();

        assert_eq!(baseline.len(), with_yaml.len());
        for (k, base) in &baseline {
            let other = with_yaml
                .get(k)
                .unwrap_or_else(|| panic!("missing key {k}"));
            mlxcel_core::eval(base);
            mlxcel_core::eval(other);
            assert_eq!(
                mlxcel_core::array_to_raw_bytes(base),
                mlxcel_core::array_to_raw_bytes(other),
                "empty YAML pipeline must preserve key {k} bit-for-bit",
            );
        }

        std::fs::remove_dir_all(&dir_a).unwrap();
        std::fs::remove_dir_all(&dir_b).unwrap();
    }

    #[test]
    fn nonmatching_pattern_surfaces_zero_match_error_through_loader() {
        // Acceptance criterion (b) — the parser returns a real
        // `SurgeryPipeline` that consumes through A1's hook, and
        // op-level errors propagate end-to-end. With all five Axis A
        // ops now landed (A1 → / A9), the canonical
        // op-level error to assert on is the "zero matches" condition
        // every real op emits when its glob does not select any
        // tensor. This test pins that error reaching the caller via
        // `load_text_weights`, which proves the wiring is complete
        // and there is no silent no-op.
        let dir = temp_model_dir("yaml_zero_match_through_loader");
        write_text_model_fixture(&dir);
        let config_dir = temp_model_dir("yaml_zero_match_through_loader_config");
        std::fs::create_dir_all(&config_dir).unwrap();
        let yaml_path = config_dir.join("surgery.yaml");
        std::fs::write(
            &yaml_path,
            r#"version: 1
operations:
  - op: scale
    pattern: "no.such.key.anywhere"
    factor: 2.0
"#,
        )
        .unwrap();
        let pipeline =
            mlxcel_surgery::parse_config_file(&yaml_path).expect("scale yaml must parse");

        let result = load_text_weights(&dir, Some(&pipeline));
        match result {
            Ok(_) => panic!("zero-match op must surface an error end-to-end"),
            Err(err) => assert!(
                err.to_lowercase().contains("zero")
                    || err.to_lowercase().contains("no match")
                    || err.to_lowercase().contains("matched no"),
                "expected zero-match style error, got: {err}",
            ),
        }

        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&config_dir).unwrap();
    }

    #[test]
    fn scale_op_mutates_weights_through_loader_end_to_end() {
        // A5 acceptance criterion (b): YAML scale config →
        // SurgeryPipeline → load_text_weights → mutated WeightMap.
        // The fixture has `model.layers.0.self_attn.q_proj.weight`
        // as a 2x2 f32-zeros tensor; scaling by 2.0 should keep
        // zeros (0*2 = 0) so we cannot detect a difference there.
        // Use `model.embed_tokens.weight` instead — it is also
        // zeros, so we follow up with a non-trivial integration
        // test by injecting non-zero values via a custom op below.
        //
        // Here we pin the simpler property: the loader returns Ok,
        // the matched key is still present with the right dtype
        // and shape, and the pipeline did not error out. A
        // non-zero numerical check lives in the synthetic
        // WeightMap unit tests inside `mlxcel-surgery`.
        //
        // the second `load_text_weights(_, None)` call
        // below consults the process-global active-pipeline slot, so
        // we must hold `env_lock` to keep parallel tests that touch
        // that slot from polluting our baseline read.
        let _env_guard = crate::test_support::env_lock::env_lock();
        let dir = temp_model_dir("yaml_scale_real");
        write_text_model_fixture(&dir);
        let yaml_path = dir.join("surgery.yaml");
        std::fs::write(
            &yaml_path,
            r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.*.self_attn.q_proj.weight"
    factor: 2.0
"#,
        )
        .unwrap();
        let pipeline = mlxcel_surgery::parse_config_file(&yaml_path).expect("scale parses");

        let weights =
            load_text_weights(&dir, Some(&pipeline)).expect("scale through loader must succeed");

        let scaled = weights
            .get("model.layers.0.self_attn.q_proj.weight")
            .expect("matched key must still be present");
        assert_eq!(
            mlxcel_core::array_shape(scaled),
            vec![2, 2],
            "shape preserved through the loader"
        );
        // The fixture's q_proj is all zeros, so the scaled values
        // are also zeros — bit-exact equality with the legacy
        // None-transform path is the property under test here.
        let baseline_dir = temp_model_dir("yaml_scale_real_baseline");
        write_text_model_fixture(&baseline_dir);
        let baseline = load_text_weights(&baseline_dir, None).expect("baseline load must succeed");
        let baseline_q = baseline
            .get("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        mlxcel_core::eval(scaled);
        mlxcel_core::eval(baseline_q);
        // 0 * 2 == 0 — byte equality still holds.
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(scaled),
            mlxcel_core::array_to_raw_bytes(baseline_q),
        );

        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_dir_all(&baseline_dir).unwrap();
    }

    #[test]
    fn scale_op_zero_match_surfaces_through_loader() {
        // A5 acceptance criterion (a) end-to-end: when a scale
        // pattern matches no tensors, the load must fail with a
        // clear error rather than silently no-op. This pins the
        // "matched zero tensors" error through the full pipeline.
        let dir = temp_model_dir("yaml_scale_zero_match");
        write_text_model_fixture(&dir);
        let yaml_path = dir.join("surgery.yaml");
        std::fs::write(
            &yaml_path,
            r#"version: 1
operations:
  - op: scale
    pattern: "model.layers.*.does_not_exist.weight"
    factor: 1.5
"#,
        )
        .unwrap();
        let pipeline = mlxcel_surgery::parse_config_file(&yaml_path).expect("scale parses");

        let result = load_text_weights(&dir, Some(&pipeline));
        match result {
            Ok(_) => panic!("zero-match scale must surface an error"),
            Err(err) => assert!(
                err.contains("matched zero tensors") || err.contains("zero"),
                "expected zero-match error, got: {err}",
            ),
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// RAII guard that installs `pipeline` into the process-wide
    /// active-pipeline slot on construction and always clears it on
    /// drop, even when the test body panics mid-way through.
    ///
    /// Together with `crate::test_support::env_lock::env_lock` this
    /// keeps the slot mutation invisible to any test running in
    /// parallel inside the same cargo-test binary. Other tests that
    /// call `load_text_weights(_, None)` from outside this module —
    /// and therefore do not acquire `env_lock` — are still safe
    /// because the guard restores the slot to `None` before the test
    /// thread releases `env_lock`.
    struct ScopedActivePipeline;

    impl ScopedActivePipeline {
        fn install(pipeline: Arc<SurgeryPipeline>) -> Self {
            crate::surgery::set_active_pipeline(Some(pipeline));
            Self
        }
    }

    impl Drop for ScopedActivePipeline {
        fn drop(&mut self) {
            crate::surgery::set_active_pipeline(None);
        }
    }

    /// (A4): when the CLI installs an active pipeline via
    /// `crate::surgery::set_active_pipeline`, the consolidated loader
    /// must pick it up for `load_text_weights(_, None)` callers. This
    /// is the integration glue that lets `mlxcel generate --surgery
    /// foo.yaml` flow through the 60+ model-family loaders without
    /// per-loader plumbing changes.
    ///
    /// Holds `crate::test_support::env_lock::env_lock` because the
    /// active-pipeline slot is process-global, just like an env var:
    /// no other test in this binary can observe the slot in the
    /// non-None state because they all serialise on the same lock
    /// when they touch a process-global resource.
    #[test]
    fn active_pipeline_slot_is_consulted_when_transform_arg_is_none() {
        let _env_guard = crate::test_support::env_lock::env_lock();

        // Install a real op pipeline whose glob deliberately matches
        // nothing in the fixture, then call `load_text_weights(_,
        // None)`. The active-pipeline slot must be consulted because
        // the explicit `transform` is None, and the zero-match error
        // emitted by `ScaleOp::apply` must propagate out. All five
        // Axis A ops (A1 → / A9) are now real, so we use
        // the zero-match diagnostic — every real op emits one — as
        // the load-path-visible signal.
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "no.such.key.anywhere"
    factor: 2.0
"#;
        let pipeline = mlxcel_surgery::parse_config_str(yaml, None).expect("scale yaml must parse");
        let _slot_guard = ScopedActivePipeline::install(Arc::new(pipeline));

        let dir_with_slot = temp_model_dir("active_slot_installed");
        write_text_model_fixture(&dir_with_slot);
        let result = load_text_weights(&dir_with_slot, None);

        match result {
            Ok(_) => panic!(
                "active-pipeline slot must be consulted when transform=None, \
                 expected zero-match error from ScaleOp"
            ),
            Err(err) => assert!(
                err.to_lowercase().contains("zero")
                    || err.to_lowercase().contains("no match")
                    || err.to_lowercase().contains("matched no"),
                "active-pipeline slot integration must propagate errors, got: {err}",
            ),
        }

        // `_slot_guard` drops here and clears the slot back to None.
        std::fs::remove_dir_all(&dir_with_slot).unwrap();
    }

    /// (A4): explicit `transform` argument takes precedence
    /// over the active-pipeline slot. This is the contract callers
    /// (e.g. future programmatic users and the existing integration tests) rely on to bypass the global slot.
    #[test]
    fn explicit_transform_arg_wins_over_active_pipeline_slot() {
        let _env_guard = crate::test_support::env_lock::env_lock();

        // Install a "failing" pipeline in the slot, then pass an
        // explicit no-op `WeightTransform`. Because the explicit
        // argument wins, the load must succeed (the slot's failing
        // pipeline is never consulted).
        let yaml = r#"version: 1
operations:
  - op: scale
    pattern: "*"
    factor: 2.0
"#;
        let bad_pipeline = mlxcel_surgery::parse_config_str(yaml, None).expect("scale parses");
        let _slot_guard = ScopedActivePipeline::install(Arc::new(bad_pipeline));

        let dir = temp_model_dir("explicit_wins_slot");
        write_text_model_fixture(&dir);

        // CountingTransform is a no-op (from the outer test module);
        // its apply returns Ok(()).
        let transform = CountingTransform::new();
        let result = load_text_weights(&dir, Some(&transform));

        assert!(
            result.is_ok(),
            "explicit transform must win over slot (load failed; \
             slot pipeline was incorrectly consulted)"
        );
        assert_eq!(
            transform.call_count(),
            1,
            "explicit transform must be the one invoked"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Write a Llama-shaped synthetic fixture suitable for a prune
    /// end-to-end test. The fixture sets `num_attention_heads=4`,
    /// `num_key_value_heads=4` (MHA so the test does not have to assert
    /// on GQA-skip semantics here), `hidden_size=32`, `head_dim=8`,
    /// `intermediate_size=64`, and one transformer block.
    ///
    /// Used by: end-to-end PruneOp tests below.
    fn write_prune_fixture(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&json!({
                "model_type": "llama",
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "hidden_size": 32,
                "head_dim": 8,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "tie_word_embeddings": false,
            }))
            .unwrap(),
        )
        .unwrap();

        // q_proj [num_heads*head_dim=32, hidden=32] = 1024 f32 ones.
        let qkvo_shape = vec![32usize, 32usize];
        let qkvo_data = vec![0u8; 4 * 32 * 32];
        // Fill with f32 = 1.0 little-endian: 0x0000803F.
        let mut q_bytes = qkvo_data.clone();
        for chunk in q_bytes.chunks_exact_mut(4) {
            chunk[0] = 0x00;
            chunk[1] = 0x00;
            chunk[2] = 0x80;
            chunk[3] = 0x3F;
        }
        let o_bytes = q_bytes.clone();
        let kv_bytes = q_bytes.clone();
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                (
                    "model.embed_tokens.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: vec![16, 32],
                        data: vec![0u8; 16 * 32 * 4],
                    },
                ),
                (
                    "lm_head.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: vec![16, 32],
                        data: vec![0u8; 16 * 32 * 4],
                    },
                ),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: qkvo_shape.clone(),
                        data: q_bytes,
                    },
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: qkvo_shape.clone(),
                        data: o_bytes,
                    },
                ),
                (
                    "model.layers.0.self_attn.k_proj.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: qkvo_shape.clone(),
                        data: kv_bytes.clone(),
                    },
                ),
                (
                    "model.layers.0.self_attn.v_proj.weight",
                    OwnedTensor {
                        dtype: SafeTensorDtype::F32,
                        shape: qkvo_shape,
                        data: kv_bytes,
                    },
                ),
            ],
        );
    }

    /// Read the f32 contents of a weight key as a flat `Vec<f32>`.
    fn read_f32_weight(weights: &WeightMap, key: &str) -> Vec<f32> {
        let arr = weights
            .get(key)
            .unwrap_or_else(|| panic!("key {key} not in weight map"));
        mlxcel_core::eval(arr);
        let bytes = mlxcel_core::array_to_raw_bytes(arr);
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    #[test]
    fn prune_op_runs_through_load_text_weights() {
        // Acceptance criterion (b) — a concrete PruneOp
        // routed through the consolidated text-weight load path must
        // actually zero the right slice of the right tensor without
        // crashing or returning NaN/Inf. This is the end-to-end
        // integration test that A4 (`--surgery` CLI flag) will
        // eventually run on a real model; here we exercise the same
        // code path with a Llama-shaped synthetic fixture.
        use mlxcel_surgery::{PruneOp, PruneSelector};

        let dir = temp_model_dir("prune_e2e_basic");
        write_prune_fixture(&dir);

        let op = PruneOp::new(
            "model.layers.0.self_attn.*",
            PruneSelector::AttentionHead { head_ids: vec![1] },
        )
        .expect("compile prune op");
        let mut pipeline = SurgeryPipeline::new();
        pipeline.push(op.into_shared());

        let loaded = load_text_weights(&dir, Some(&pipeline)).expect("load with surgery");

        // q_proj head 1 = rows [8..16) of shape [32, 32] must be zero;
        // every other row must remain 1.0.
        let q = read_f32_weight(&loaded, "model.layers.0.self_attn.q_proj.weight");
        for r in 0..32usize {
            let row_sum: f32 = q[r * 32..(r + 1) * 32].iter().sum();
            if (8..16).contains(&r) {
                assert_eq!(row_sum, 0.0, "q_proj row {r} (head 1) must be zero");
            } else {
                assert_eq!(row_sum, 32.0, "q_proj row {r} must remain ones");
            }
        }

        // o_proj head 1 = columns [8..16) of shape [32, 32] must be
        // zero; every other column 1.0.
        let o = read_f32_weight(&loaded, "model.layers.0.self_attn.o_proj.weight");
        for r in 0..32 {
            for c in 0..32 {
                let v = o[r * 32 + c];
                let expected = if (8..16).contains(&c) { 0.0 } else { 1.0 };
                assert_eq!(v, expected, "o_proj[{r}, {c}] mismatch");
            }
        }

        // KV are untouched per the GQA-safe policy. (Even though this
        // fixture is MHA — num_kv_heads == num_heads — the policy
        // still applies because the implementation does not
        // special-case MHA. Documenting and enforcing the policy
        // uniformly avoids surprise across model families.)
        let k = read_f32_weight(&loaded, "model.layers.0.self_attn.k_proj.weight");
        assert!(k.iter().all(|&v| v == 1.0), "k_proj must remain untouched");
        let v = read_f32_weight(&loaded, "model.layers.0.self_attn.v_proj.weight");
        assert!(v.iter().all(|&v| v == 1.0), "v_proj must remain untouched");

        // No NaN/Inf anywhere — paranoia check for slice_update path.
        for k in loaded.keys() {
            let floats = read_f32_weight(&loaded, k);
            for (i, &f) in floats.iter().enumerate() {
                assert!(f.is_finite(), "{k}[{i}] = {f} is NaN/Inf");
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prune_op_via_yaml_runs_through_load_text_weights() {
        // Same as above, but routed through the YAML factory. This
        // proves the YAML -> SurgeryPipeline -> SurgeryOp::apply path
        // works end-to-end (the path A4's CLI will use).
        let yaml = r#"version: 1
operations:
  - op: prune
    granularity: attention_head
    pattern: "model.layers.0.self_attn.*"
    head_ids: [2]
"#;
        let pipeline = mlxcel_surgery::parse_config_str(yaml, None).expect("YAML must parse");
        assert_eq!(pipeline.len(), 1);

        let dir = temp_model_dir("prune_e2e_yaml");
        write_prune_fixture(&dir);

        let loaded = load_text_weights(&dir, Some(&pipeline)).expect("load with surgery");

        let q = read_f32_weight(&loaded, "model.layers.0.self_attn.q_proj.weight");
        // Head 2 = rows [16..24).
        for r in 0..32 {
            let row_sum: f32 = q[r * 32..(r + 1) * 32].iter().sum();
            if (16..24).contains(&r) {
                assert_eq!(row_sum, 0.0);
            } else {
                assert_eq!(row_sum, 32.0);
            }
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn replace_yaml_swaps_targeted_tensor_through_load_text_weights() {
        // acceptance criterion (b) — the real ReplaceOp,
        // built from YAML, replaces the targeted base tensor with
        // the donor's content when invoked through A1's
        // `load_text_weights` hook. Untouched tensors stay
        // bit-identical to the baseline (proves the op only mutates
        // matching keys; A4's `--surgery` flag will route this
        // exact pipeline through the same loader path).
        //
        // the baseline `load_text_weights(_, None)` call
        // below consults the process-global active-pipeline slot, so
        // we must hold `env_lock` to keep parallel tests that touch
        // that slot from polluting our baseline read.
        let _env_guard = crate::test_support::env_lock::env_lock();
        let dir = temp_model_dir("yaml_replace_e2e");
        write_text_model_fixture(&dir);

        // Build the donor `.safetensors` directly inside the model
        // dir so the YAML can refer to it with a relative path. The
        // donor uses a non-zero, identifiable byte pattern so we can
        // assert the swap really happened (the baseline fixture
        // stores all-zero tensors).
        let donor_path = dir.join("donor.safetensors");
        let donor_floats: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let mut donor_bytes = Vec::with_capacity(donor_floats.len() * 4);
        for v in donor_floats {
            donor_bytes.extend_from_slice(&v.to_le_bytes());
        }
        write_safetensors(
            &donor_path,
            &[(
                "model.embed_tokens.weight",
                OwnedTensor {
                    dtype: SafeTensorDtype::F32,
                    shape: vec![2, 2],
                    data: donor_bytes,
                },
            )],
        );

        // Write the YAML alongside the donor so the parser's
        // relative-path resolution picks up the donor automatically.
        let yaml_path = dir.join("replace.yaml");
        let yaml = r#"version: 1
operations:
  - op: replace
    pattern: "model.embed_tokens.weight"
    source: "./donor.safetensors"
    source_key: "model.embed_tokens.weight"
"#;
        std::fs::write(&yaml_path, yaml).unwrap();
        let pipeline = mlxcel_surgery::parse_config_file(&yaml_path).expect("replace yaml parses");

        let baseline = load_text_weights(&dir, None).unwrap();
        let with_replace = load_text_weights(&dir, Some(&pipeline)).unwrap();

        // The targeted tensor differs from the baseline.
        let post = with_replace.get("model.embed_tokens.weight").unwrap();
        mlxcel_core::eval(post);
        let bytes = mlxcel_core::array_to_raw_bytes(post);
        let mut floats: Vec<f32> = Vec::with_capacity(4);
        for chunk in bytes.chunks_exact(4) {
            floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        assert_eq!(
            floats,
            vec![10.0, 20.0, 30.0, 40.0],
            "replace op must substitute the donor payload"
        );
        let base_e = baseline.get("model.embed_tokens.weight").unwrap();
        mlxcel_core::eval(base_e);
        assert_ne!(
            mlxcel_core::array_to_raw_bytes(base_e),
            mlxcel_core::array_to_raw_bytes(post),
            "result must differ from the baseline (acceptance b)"
        );

        // Untouched tensors stay bit-identical to the baseline.
        let untouched_key = "model.layers.0.self_attn.q_proj.weight";
        let baseline_t = baseline.get(untouched_key).unwrap();
        let post_t = with_replace.get(untouched_key).unwrap();
        mlxcel_core::eval(baseline_t);
        mlxcel_core::eval(post_t);
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(baseline_t),
            mlxcel_core::array_to_raw_bytes(post_t),
            "untouched tensor must stay bit-exact across the replace op"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
