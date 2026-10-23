// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Exact released mixed-dtype schema and a single-handle filtered tensor loader.
//! Schema provenance: mlx-community/sam-3d-body-dinov3-bf16 revision
//! 6d8a283b84c5569c088109589bdf938decf8ad48; header SHA256
//! 51bfc5f161f7c4b10450b58c3e309eb9e3657da8c5bea900e7e6b864daabd6e6.
use anyhow::{Context, Result, ensure};
use mlxcel_core::{self as mx, weights::WeightMap};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
#[derive(Deserialize)]
struct Spec {
    dtype: String,
    shape: Vec<i32>,
}
#[derive(Deserialize)]
struct Header {
    dtype: String,
    shape: Vec<i32>,
    data_offsets: [u64; 2],
}
fn retained(k: &str) -> bool {
    if [
        "decoder_hand.",
        "head_pose_hand.",
        "head_camera_hand.",
        "init_to_token_mhr_hand.",
        "prev_to_token_mhr_hand.",
        "keypoint_posemb_linear_hand.",
        "keypoint3d_posemb_linear_hand.",
        "keypoint_feat_linear_hand.",
        "ray_cond_emb_hand.",
        "prompt_encoder.mask_downscaling.",
    ]
    .iter()
    .any(|p| k.starts_with(p))
    {
        return false;
    }
    if [
        "init_pose_hand",
        "init_camera_hand",
        "keypoint_embedding_hand",
        "keypoint3d_embedding_hand",
    ]
    .contains(&k)
    {
        return false;
    }
    if let Some(k) = k.strip_prefix("head_pose.body_model.") {
        return [
            "joint_translation_offsets",
            "joint_prerotations",
            "joint_parents",
            "parameter_transform",
            "base_shape",
            "shape_vectors",
            "face_shape_vectors",
            "inverse_bind_pose",
            "skin_indices",
            "skin_weights",
            "vert_indices",
            "pc_sparse_indices",
            "pc_sparse_weight",
            "pc_linear_weight",
            "minmax_min",
            "minmax_max",
            "minmax_weight",
            "minmax_parameter_index",
        ]
        .contains(&k);
    }
    if let Some(k) = k.strip_prefix("head_pose.") {
        return k.starts_with("proj.")
            || [
                "scale_mean",
                "scale_comps",
                "hand_pose_mean",
                "hand_pose_comps",
                "hand_joint_idxs_left",
                "hand_joint_idxs_right",
                "keypoint_mapping",
                "faces",
            ]
            .contains(&k);
    }
    // The public body-only predictor uses only invalid-point and dense Gaussian PE.
    if k.starts_with("prompt_encoder.") {
        return k == "prompt_encoder.invalid_point_embed.weight"
            || k == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix";
    }
    !k.starts_with("hand_cls_embed.")
        && !k.starts_with("bbox_embed.")
        && !k.starts_with("hand_pe_layer.")
}
fn dtype(name: &str) -> Result<(i32, u64)> {
    Ok(match name {
        "BF16" => (mx::dtype::BFLOAT16, 2),
        "F32" => (mx::dtype::FLOAT32, 4),
        "I32" => (mx::dtype::INT32, 4),
        "U8" => (mx::dtype::UINT8, 1),
        _ => anyhow::bail!("unsupported Body tensor dtype {name}"),
    })
}
// `from_bytes` borrows its host source until evaluation. Materialize inside
// this ownership boundary for every dtype, including integer topology buffers.
fn from_owned_bytes(
    bytes: Vec<u8>,
    shape: &[i32],
    dtype: i32,
    key: &str,
) -> Result<super::layers::Array> {
    let value = mx::from_bytes(&bytes, shape, dtype);
    mx::try_eval(&value).with_context(|| format!("materialize Body tensor {key}"))?;
    Ok(value)
}
/// Read the released single-file bundle through one stable handle. Rejected tensors
/// never enter MLX, and each retained tensor is read exactly once in stored dtype.
pub fn load(path: &Path) -> Result<WeightMap> {
    let expected: BTreeMap<String, Spec> =
        serde_json::from_str(include_str!("released_schema.json"))?;
    let file = path.join("model.safetensors");
    let mut f = std::fs::File::open(&file).with_context(|| format!("open {}", file.display()))?;
    let len = f.metadata()?.len();
    let mut raw = [0u8; 8];
    f.read_exact(&mut raw)?;
    let n = u64::from_le_bytes(raw);
    ensure!(
        n <= 4 * 1024 * 1024 && n.checked_add(8).is_some_and(|v| v <= len),
        "invalid Body safetensors header length"
    );
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let mut values: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
    values.remove("__metadata__");
    ensure!(
        values.contains_key("head_pose.faces"),
        "missing required Body tensor head_pose.faces"
    );
    for k in expected.keys() {
        ensure!(values.contains_key(k), "missing required Body tensor {k}");
    }
    ensure!(
        values.len() == expected.len(),
        "unexpected Body checkpoint tensor keys"
    );
    let mut headers = Vec::new();
    let mut spans = Vec::new();
    for (k, v) in values {
        let h: Header = serde_json::from_value(v)?;
        let e = &expected[&k];
        ensure!(
            h.dtype == e.dtype && h.shape == e.shape,
            "Body tensor {k} must retain {} {:?}",
            e.dtype,
            e.shape
        );
        let (_, size) = dtype(&h.dtype)?;
        let size = h
            .shape
            .iter()
            .try_fold(size, |v, &d| {
                u64::try_from(d).ok().and_then(|d| v.checked_mul(d))
            })
            .context("Body tensor size overflow")?;
        ensure!(
            h.data_offsets[1].checked_sub(h.data_offsets[0]) == Some(size)
                && h.data_offsets[1] <= len - n - 8,
            "invalid Body tensor offsets {k}"
        );
        spans.push(h.data_offsets);
        if retained(&k) {
            headers.push((k, h));
        }
    }
    spans.sort_unstable();
    ensure!(
        spans.windows(2).all(|v| v[0][1] <= v[1][0]),
        "overlapping Body tensor offsets"
    );
    let mut weights = WeightMap::new();
    for (k, h) in headers {
        let size = usize::try_from(h.data_offsets[1] - h.data_offsets[0])?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .context("reserve Body tensor")?;
        bytes.resize(size, 0);
        f.seek(SeekFrom::Start(n + 8 + h.data_offsets[0]))?;
        f.read_exact(&mut bytes)?;
        let value = from_owned_bytes(bytes, &h.shape, dtype(&h.dtype)?.0, &k)?;
        // Retain the released precision. Topology/value checks in MHR constructors
        // cover integer indices; this scalar reduction checks floating payloads.
        if h.dtype == "BF16" || h.dtype == "F32" {
            let finite = mx::all_all(&mx::isfinite(&value));
            ensure!(mx::item_bool(&finite), "non-finite Body weight {k}");
        }
        weights.insert(k, value);
    }
    Ok(weights)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3d_body_released_dtype_inventory_and_filter() {
        let s: BTreeMap<String, Spec> =
            serde_json::from_str(include_str!("released_schema.json")).expect("schema");
        for (dtype, count) in [("BF16", 615), ("F32", 574), ("I32", 11), ("U8", 3)] {
            assert_eq!(s.values().filter(|s| s.dtype == dtype).count(), count);
        }
        assert!(retained("head_pose.faces"));
        assert!(retained("hand_box_embedding"));
        assert!(!retained("init_pose_hand"));
        assert!(!retained("decoder_hand.layers.0.ln1.weight"));
        assert!(!retained("prompt_encoder.mask_downscaling.0.weight"));
        assert!(!s.contains_key("backbone.blocks.0.attention.k_proj.bias"));
    }
    #[test]
    fn sam3d_body_integer_tensor_owns_bytes_after_source_drop() -> Result<()> {
        let integers: Vec<i32> = (0..4096).map(|i| i * 17 - 32768).collect();
        let integer_bytes: Vec<u8> = integers.iter().flat_map(|v| v.to_le_bytes()).collect();
        let masks: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        for (bytes, dtype) in [(integer_bytes, mx::dtype::INT32), (masks, mx::dtype::UINT8)] {
            let expected = bytes.clone();
            let array = from_owned_bytes(bytes, &[4096], dtype, "ownership regression")?;
            // Reuse host allocation classes after the consumed source Vec drops.
            let churn: Vec<Vec<u8>> = (0..32).map(|_| vec![0xa5; expected.len()]).collect();
            assert_eq!(mx::array_dtype(&array), dtype);
            assert_eq!(mx::try_array_to_raw_bytes(&array)?, expected);
            std::hint::black_box(churn);
        }
        Ok(())
    }
}
