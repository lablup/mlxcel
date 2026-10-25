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

//! Shared per-row cache step for latent MLA decode.

use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) type MlaRowSelector<'a> =
    dyn Fn(usize, &MlxArray, &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) + 'a;

/// Validate the common serving decode contract before a family indexes a
/// sequence-owned cache row. Keeping this here gives every MLA failure the
/// model name and row number instead of an opaque bounds panic.
pub(crate) fn validate_mla_decode_batch(
    family: &str,
    input_ids: &MlxArray,
    batch_caches: &[&mut [KVCache]],
    num_layers: usize,
) {
    let shape = mlxcel_core::array_shape(input_ids);
    assert_eq!(
        shape.len(),
        2,
        "{family} batched decode expects rank-2 input ids [B, 1], got {shape:?}"
    );
    assert_eq!(
        shape[0],
        batch_caches.len() as i32,
        "{family} batched decode input/cache batch mismatch: input {}, caches {}",
        shape[0],
        batch_caches.len()
    );
    assert_eq!(
        shape[1], 1,
        "{family} batched decode expects one token per row, got {shape:?}"
    );
    for (row, caches) in batch_caches.iter().enumerate() {
        assert_eq!(
            caches.len(),
            num_layers,
            "{family} batched decode row {row} cache-layer mismatch: expected {num_layers}, got {}",
            caches.len()
        );
    }
}

/// Append one latent MLA token per sequence and attend to each live cache.
///
/// The projections on either side of this helper remain batched. Only the
/// sequence-owned cache update and ragged score/softmax step run per row.
///
/// Used by: `DeepSeekV3Attention` (DeepSeek V3, Youtu), DeepSeek V3.2 MLA,
/// GLM-4.7-Flash MLA.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_latent_decode_rows(
    q_latent: &MlxArray,
    q_pe: &MlxArray,
    kv_latent: &MlxArray,
    k_pe: &MlxArray,
    caches: &mut [&mut KVCache],
    scale: f32,
    select: Option<&MlaRowSelector<'_>>,
) -> UniquePtr<MlxArray> {
    let batch = caches.len();
    assert!(batch > 0, "mla_latent_decode_rows: empty batch");
    for (name, tensor) in [
        ("q_latent", q_latent),
        ("q_pe", q_pe),
        ("kv_latent", kv_latent),
        ("k_pe", k_pe),
    ] {
        let shape = mlxcel_core::array_shape(tensor);
        assert_eq!(
            shape.first().copied(),
            Some(batch as i32),
            "mla_latent_decode_rows: {name} batch {:?} does not match cache batch {batch}",
            shape.first()
        );
        assert_eq!(
            shape.get(2).copied(),
            Some(1),
            "mla_latent_decode_rows: {name} must be a single-token decode tensor, got {shape:?}"
        );
    }

    let mut outputs = Vec::with_capacity(batch);
    for (row, cache) in caches.iter_mut().enumerate() {
        let q_row = row4(q_latent, row);
        let q_pe_row = row4(q_pe, row);
        let kv_row = row4(kv_latent, row);
        let k_pe_row = row4(k_pe, row);
        let (kv_all, k_pe_all) = cache.update_and_fetch(kv_row, k_pe_row);
        let (kv_selected, k_pe_selected) = match select {
            Some(select) => select(row, &kv_all, &k_pe_all),
            None => (kv_all, k_pe_all),
        };

        let scale_value = mlxcel_core::full_f32(&[1], scale, mlxcel_core::array_dtype(&q_pe_row));
        let q_pe_scaled = mlxcel_core::multiply(&q_pe_row, &scale_value);
        let k_pe_t = mlxcel_core::transpose_axes(&k_pe_selected, &[0, 1, 3, 2]);
        let pe_scores = mlxcel_core::matmul(&q_pe_scaled, &k_pe_t);
        let pe_scores_ptr = pe_scores.as_ref().unwrap() as *const MlxArray;
        // SAFETY: `pe_scores_ptr` points to the live local `pe_scores`; the
        // attention call only reads it during this invocation.
        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q_row,
                &kv_selected,
                &kv_selected,
                scale,
                pe_scores_ptr,
                0.0,
                0,
            )
        };
        outputs.push(output);
    }
    let refs: Vec<&MlxArray> = outputs.iter().filter_map(|row| row.as_ref()).collect();
    mlxcel_core::concatenate_many(&refs, 0)
}

fn row4(tensor: &MlxArray, row: usize) -> UniquePtr<MlxArray> {
    mlxcel_core::slice(
        tensor,
        &[row as i32, 0, 0, 0],
        &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "TestMlaModel batched decode input/cache batch mismatch")]
    fn batch_mismatch_names_the_family() {
        let _runtime = crate::initialize_runtime();
        let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
        let mut caches = [vec![KVCache::new()]];
        let rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        validate_mla_decode_batch("TestMlaModel", &input_ids, &rows, 1);
    }

    #[test]
    #[should_panic(expected = "TestMlaModel batched decode row 1 cache-layer mismatch")]
    fn layer_mismatch_names_the_family_and_row() {
        let _runtime = crate::initialize_runtime();
        let input_ids = mlxcel_core::from_slice_i32(&[1, 2], &[2, 1]);
        let mut caches = [vec![KVCache::new()], Vec::new()];
        let rows: Vec<&mut [KVCache]> = caches.iter_mut().map(Vec::as_mut_slice).collect();
        validate_mla_decode_batch("TestMlaModel", &input_ids, &rows, 1);
    }
}
