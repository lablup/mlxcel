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

//! Weight-map decomposition of `kv_b_proj` into the per-head `embed_q` /
//! `unembed_out` pair that `MultiLinear`-based MLA attention loads.
//!
//! Public `glm4_moe_lite` checkpoints store the MLA up-projection as one
//! `self_attn.kv_b_proj.weight` of shape `[num_heads * (qk_nope + v),
//! kv_lora_rank]` and ship no `embed_q` tensor. Two consumers need the same
//! split of that matrix: the `glm4_moe_lite` weight sanitizer, which runs it
//! over every decoder layer at load, and the `mlxcel split-mtp` surgery op,
//! which runs it once over the next-token-prediction block it extracts from
//! the raw checkpoint (issue #1326). The `mlxcel-surgery` crate sits below the
//! binary crate in the dependency graph, so the one implementation lives here
//! rather than beside the sanitizer.
//!
//! The layout is the mirror of [`super::absorb`]: that module folds
//! `kv_b_proj` into a dense query-side and output-side operand for the
//! absorbed decode graph, while this one produces the stored-tensor pair
//! (`embed_q` as `[H, kv_lora_rank, qk_nope]`, `unembed_out` as `[H, v,
//! kv_lora_rank]`) that the weight loaders expect on disk.

use crate::ffi;
use crate::utils::slice_axis;
use crate::weights::WeightMap;

/// The head dimensions the decomposition needs, read off a family config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBProjGeometry {
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
}

/// Decompose `{attn_prefix}.kv_b_proj.weight` into
/// `{attn_prefix}.embed_q.weight` and `{attn_prefix}.unembed_out.weight`
/// in place.
///
/// Returns `Ok(true)` when the pair was produced and `Ok(false)` when there
/// was nothing to do: a prefix that already carries `embed_q` is left exactly
/// as it is (re-deriving would at best duplicate work and at worst replace a
/// quantized `embed_q` with a dense one built from a stale `kv_b_proj`), and
/// a prefix with no `kv_b_proj` is skipped.
///
/// A quantized `kv_b_proj` is dequantized before the split, because the split
/// runs per head along the row axis and slicing in packed space would leave
/// each half carrying group scales and biases that describe the whole row.
/// The packed pair is solved from the shapes with
/// [`crate::layers::infer_mla_quantization_params`] and bounded before it
/// reaches `dequantize` (issue #958).
///
/// Returns `Err` rather than panicking or aborting when a `kv_b_proj` cannot
/// be decomposed: scales with no biases (a block-float export, which
/// `dequantize("affine")` could not unpack anyway, issue #1026), a solved
/// quantization pair no packing can describe, or a tensor whose shape
/// disagrees with the config (MLX reports a bad reshape by throwing, and the
/// throw crosses the cxx bridge as an abort rather than a `Result`).
///
/// `label` names the tensor's owner in error messages (`layer 3`, `MTP
/// block`).
///
/// Used by: `glm4_moe_lite::sanitize_weights`, `mlxcel_surgery::ops::split_mtp`,
/// the `glm4_moe_lite_mtp` drafter sanitizer.
pub fn decompose_kv_b_proj(
    weights: &mut WeightMap,
    attn_prefix: &str,
    geometry: KvBProjGeometry,
    label: &str,
) -> Result<bool, String> {
    // `as i32` would wrap a hostile or malformed config's field, and the
    // release profile has overflow checks off, so `num_heads * head_dim`
    // below could wrap past the shape check this function exists to perform
    // and reach `reshape` with a bogus extent, which MLX answers by aborting
    // across the cxx bridge. Convert fallibly instead (issue #1326).
    let to_i32 = |value: usize, field: &str| -> Result<i32, String> {
        i32::try_from(value).map_err(|_| {
            format!("{label}: {field} {value} does not fit in i32; check the checkpoint config")
        })
    };
    let num_heads = to_i32(geometry.num_heads, "num_attention_heads")?;
    let head_dim = to_i32(
        geometry
            .qk_nope_head_dim
            .checked_add(geometry.v_head_dim)
            .ok_or_else(|| format!("{label}: qk_nope_head_dim + v_head_dim overflows"))?,
        "qk_nope_head_dim + v_head_dim",
    )?;
    let qk_nope_head_dim = to_i32(geometry.qk_nope_head_dim, "qk_nope_head_dim")?;
    let kv_lora_rank = to_i32(geometry.kv_lora_rank, "kv_lora_rank")?;
    num_heads
        .checked_mul(head_dim)
        .ok_or_else(|| format!("{label}: num_attention_heads * head_dim overflows i32"))?;

    let kv_b_key = format!("{attn_prefix}.kv_b_proj.weight");
    let embed_q_key = format!("{attn_prefix}.embed_q.weight");
    if weights.contains_key(&embed_q_key) || !weights.contains_key(&kv_b_key) {
        return Ok(false);
    }

    let scales_key = format!("{attn_prefix}.kv_b_proj.scales");
    let is_quantized = weights.contains_key(&scales_key);

    let w = weights
        .remove(&kv_b_key)
        .ok_or_else(|| format!("{label}: `{kv_b_key}` vanished from the weight map"))?;

    let w_full = if is_quantized {
        let s = weights
            .remove(&scales_key)
            .ok_or_else(|| format!("{label}: `{scales_key}` vanished from the weight map"))?;
        let b_key = format!("{attn_prefix}.kv_b_proj.biases");
        let b = weights.remove(&b_key).ok_or_else(|| {
            format!(
                "{label}: kv_b_proj has scales but no biases at key `{b_key}`; \
                 the checkpoint may be corrupted or only partially converted"
            )
        })?;

        let (inferred_gs, inferred_bits) = crate::layers::infer_mla_quantization_params(
            &ffi::array_shape(&w),
            &ffi::array_shape(&s),
            kv_lora_rank,
            &format!("{attn_prefix}.kv_b_proj"),
        )?;

        // SAFETY: `w`, `s` and `b` are live UniquePtr-owned arrays taken out
        // of the weight map above and are not dropped until this call
        // returns.
        unsafe {
            ffi::dequantize(
                &w,
                &s,
                &*b as *const _,
                inferred_gs,
                inferred_bits,
                "affine",
            )
        }
    } else {
        ffi::copy(&w)
    };

    let w_shape = ffi::array_shape(&w_full);
    let expected_rows = num_heads * head_dim;
    if w_shape.len() != 2 || w_shape[0] != expected_rows || w_shape[1] != kv_lora_rank {
        return Err(format!(
            "{label}: kv_b_proj shape mismatch: got {w_shape:?}, expected \
             [{expected_rows}, {kv_lora_rank}] (num_heads={num_heads}, head_dim={head_dim}, \
             kv_lora_rank={kv_lora_rank})"
        ));
    }

    // [num_heads * head_dim, kv_lora_rank] -> [num_heads, head_dim, kv_lora_rank]
    let w_3d = ffi::reshape(&w_full, &[num_heads, head_dim, -1]);

    // wk = w[:, :qk_nope_head_dim, :]  (the nope half)
    // wv = w[:, qk_nope_head_dim:, :]  (the v half)
    let wk = slice_axis(&w_3d, 1, 0, qk_nope_head_dim);
    let wv = slice_axis(&w_3d, 1, qk_nope_head_dim, -1);

    // embed_q stores the swapped-axes form: [num_heads, kv_lora_rank, qk_nope_head_dim]
    let wk = ffi::transpose_axes(&wk, &[0, 2, 1]);

    // Copy both halves so `MultiLinear`'s matmul sees well-formed strides
    // regardless of the backend, rather than a view into the parent tensor.
    let wk = ffi::copy(&wk);
    let wv = ffi::copy(&wv);

    weights.insert(embed_q_key, wk);
    weights.insert(format!("{attn_prefix}.unembed_out.weight"), wv);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::MlxArray;

    const HEADS: i32 = 2;
    const NOPE: i32 = 4;
    const V: i32 = 6;
    const RANK: i32 = 16;

    fn geometry() -> KvBProjGeometry {
        KvBProjGeometry {
            num_heads: HEADS as usize,
            qk_nope_head_dim: NOPE as usize,
            v_head_dim: V as usize,
            kv_lora_rank: RANK as usize,
        }
    }

    fn read_f32(arr: &MlxArray) -> Vec<f32> {
        let arr = ffi::astype(arr, crate::dtype::FLOAT32);
        ffi::eval(&arr);
        ffi::array_to_raw_bytes(&arr)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn decomposes_a_dense_kv_b_proj_into_the_stored_pair() {
        let rows = HEADS * (NOPE + V);
        let ramp: Vec<f32> = (0..rows * RANK).map(|i| i as f32).collect();
        let mut weights = WeightMap::new();
        weights.insert(
            "a.kv_b_proj.weight".to_string(),
            ffi::from_slice_f32(&ramp, &[rows, RANK]),
        );

        let produced = decompose_kv_b_proj(&mut weights, "a", geometry(), "layer 0").unwrap();
        assert!(produced);
        assert!(!weights.contains_key("a.kv_b_proj.weight"));

        let embed_q = weights.get("a.embed_q.weight").expect("embed_q");
        assert_eq!(ffi::array_shape(embed_q), vec![HEADS, RANK, NOPE]);
        let unembed = weights.get("a.unembed_out.weight").expect("unembed_out");
        assert_eq!(ffi::array_shape(unembed), vec![HEADS, V, RANK]);

        // Head 1, nope row 2, latent column 5 of the source lands at
        // embed_q[1, 5, 2]; head 1, v row 3, column 7 lands at unembed[1, 3, 7].
        let embed_q = read_f32(embed_q);
        let src = |head: i32, row: i32, col: i32| ((head * (NOPE + V) + row) * RANK + col) as f32;
        let head = 1;
        let idx_q = ((head * RANK + 5) * NOPE + 2) as usize;
        assert_eq!(embed_q[idx_q], src(head, 2, 5));
        let unembed = read_f32(unembed);
        let idx_v = ((head * V + 3) * RANK + 7) as usize;
        assert_eq!(unembed[idx_v], src(head, NOPE + 3, 7));
    }

    #[test]
    fn leaves_a_predecomposed_prefix_and_a_missing_prefix_alone() {
        let mut weights = WeightMap::new();
        weights.insert(
            "a.embed_q.weight".to_string(),
            ffi::zeros(&[HEADS, RANK, NOPE], crate::dtype::FLOAT32),
        );
        weights.insert(
            "a.kv_b_proj.weight".to_string(),
            ffi::zeros(&[HEADS * (NOPE + V), RANK], crate::dtype::FLOAT32),
        );
        assert!(!decompose_kv_b_proj(&mut weights, "a", geometry(), "x").unwrap());
        assert!(weights.contains_key("a.kv_b_proj.weight"));
        assert!(!decompose_kv_b_proj(&mut weights, "b", geometry(), "x").unwrap());
    }

    #[test]
    fn rejects_a_shape_that_disagrees_with_the_geometry() {
        let mut weights = WeightMap::new();
        weights.insert(
            "a.kv_b_proj.weight".to_string(),
            ffi::zeros(&[HEADS * (NOPE + V) + 1, RANK], crate::dtype::FLOAT32),
        );
        let err = decompose_kv_b_proj(&mut weights, "a", geometry(), "MTP block").unwrap_err();
        assert!(err.contains("MTP block"), "{err}");
        assert!(err.contains("shape mismatch"), "{err}");
    }

    #[test]
    fn rejects_scales_without_biases() {
        let mut weights = WeightMap::new();
        weights.insert(
            "a.kv_b_proj.weight".to_string(),
            ffi::zeros(&[HEADS * (NOPE + V), 2], crate::dtype::UINT32),
        );
        weights.insert(
            "a.kv_b_proj.scales".to_string(),
            ffi::zeros(&[HEADS * (NOPE + V), 1], crate::dtype::FLOAT32),
        );
        let err = decompose_kv_b_proj(&mut weights, "a", geometry(), "layer 0").unwrap_err();
        assert!(err.contains("scales but no biases"), "{err}");
        assert!(err.contains("a.kv_b_proj.biases"), "{err}");
    }
}
