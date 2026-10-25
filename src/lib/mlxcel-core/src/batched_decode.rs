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

//! Shared primitives for native batched decoder implementations.

use cxx::UniquePtr;

use crate::cache::{BatchedAttentionMetadata, attend_batched};
use crate::ffi::MlxArray;
use crate::layers::KVCache;

/// Static dimensions and scale used by split attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitAttentionShape {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

/// Family hooks around the shared split-attention tensor pipeline.
pub trait SplitAttention {
    fn split_shape(&self) -> SplitAttentionShape;

    fn qk_norm_heads(
        &self,
        q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (q, k)
    }

    fn rope_batched(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        offsets: &[i32],
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>);

    fn custom_row_attend(&self) -> bool {
        false
    }

    fn attend_row(
        &self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        cache.attend(q, k, v, self.split_shape().scale, mask)
    }

    /// Row-aware variant for attention implementations whose auxiliary state
    /// is projected once for the whole batch and then sliced alongside Q/K/V.
    /// The default preserves the existing row-independent contract.
    fn attend_row_indexed(
        &self,
        _row: usize,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.attend_row(q, k, v, cache, mask)
    }
}

/// Apply one decoder attention block to pre-projected `[B, T, H*D]` tensors.
///
/// Used by: Llama3 (and its aliases), Qwen3, Qwen3-MoE.
pub fn forward_split_attention<A: SplitAttention + ?Sized>(
    attn: &A,
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    caches: &mut [&mut KVCache],
    metadata: &BatchedAttentionMetadata,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let batch = caches.len();
    assert!(batch > 0, "forward_split_attention: empty batch");
    debug_assert_eq!(metadata.len(), batch);
    if let Some(mask) = mask {
        let mask_batch = crate::array_shape(mask)[0] as usize;
        assert_eq!(
            mask_batch, batch,
            "forward_split_attention: mask batch {mask_batch} does not match cache batch {batch}"
        );
    }

    let shape = attn.split_shape();
    let seq_len = crate::array_shape(q)[1];
    let q = crate::reshape(q, &[batch as i32, seq_len, shape.num_heads, shape.head_dim]);
    let k = crate::reshape(
        k,
        &[batch as i32, seq_len, shape.num_kv_heads, shape.head_dim],
    );
    let v = crate::reshape(
        v,
        &[batch as i32, seq_len, shape.num_kv_heads, shape.head_dim],
    );
    let (q, k) = attn.qk_norm_heads(q, k);
    let q = crate::transpose_axes(&q, &[0, 2, 1, 3]);
    let k = crate::transpose_axes(&k, &[0, 2, 1, 3]);
    let v = crate::transpose_axes(&v, &[0, 2, 1, 3]);
    let (q, k) = attn.rope_batched(&q, &k, &metadata.rope_offsets);

    let output = if attn.custom_row_attend() {
        let mut outputs = Vec::with_capacity(batch);
        for (row, cache) in caches.iter_mut().enumerate() {
            let q_row = slice_row(&q, row);
            let k_row = slice_row(&k, row);
            let v_row = slice_row(&v, row);
            let mask_row = mask.map(|mask| {
                let row = crate::slice(
                    mask,
                    &[row as i32, 0, 0],
                    &[row as i32 + 1, seq_len, i32::MAX],
                );
                crate::squeeze_axis(&row, 0)
            });
            outputs.push(attn.attend_row_indexed(
                row,
                &q_row,
                k_row,
                v_row,
                cache,
                mask_row.as_deref(),
            ));
        }
        concatenate_rows(outputs)
    } else {
        attend_batched(&q, &k, &v, caches, shape.scale, mask)
    };

    let output = crate::transpose_axes(&output, &[0, 2, 1, 3]);
    crate::reshape(
        &output,
        &[batch as i32, seq_len, shape.num_heads * shape.head_dim],
    )
}

fn slice_row(array: &MlxArray, row: usize) -> UniquePtr<MlxArray> {
    crate::slice(
        array,
        &[row as i32, 0, 0, 0],
        &[row as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

fn concatenate_rows(rows: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let refs: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
    crate::concatenate_many(&refs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Indexed;

    impl SplitAttention for Indexed {
        fn split_shape(&self) -> SplitAttentionShape {
            SplitAttentionShape {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                scale: 1.0,
            }
        }
        fn rope_batched(
            &self,
            q: &MlxArray,
            k: &MlxArray,
            _offsets: &[i32],
        ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
            (crate::copy(q), crate::copy(k))
        }
        fn custom_row_attend(&self) -> bool {
            true
        }
        fn attend_row_indexed(
            &self,
            row: usize,
            q: &MlxArray,
            _k: UniquePtr<MlxArray>,
            _v: UniquePtr<MlxArray>,
            _cache: &mut KVCache,
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::full_f32(
                &crate::array_shape(q),
                row as f32 + 1.0,
                crate::dtype::FLOAT32,
            )
        }
    }

    struct Legacy;

    impl SplitAttention for Legacy {
        fn split_shape(&self) -> SplitAttentionShape {
            Indexed.split_shape()
        }
        fn rope_batched(
            &self,
            q: &MlxArray,
            k: &MlxArray,
            offsets: &[i32],
        ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
            Indexed.rope_batched(q, k, offsets)
        }
        fn custom_row_attend(&self) -> bool {
            true
        }
        fn attend_row(
            &self,
            q: &MlxArray,
            _k: UniquePtr<MlxArray>,
            _v: UniquePtr<MlxArray>,
            _cache: &mut KVCache,
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::copy(q)
        }
    }

    fn run(attn: &impl SplitAttention, q: &MlxArray) -> UniquePtr<MlxArray> {
        let k = crate::copy(q);
        let v = crate::copy(q);
        let mut caches = [KVCache::new(), KVCache::new(), KVCache::new()];
        let mut refs: Vec<&mut KVCache> = caches.iter_mut().collect();
        let metadata = BatchedAttentionMetadata::uniform_kv_caches(&refs, 1, 0).unwrap();
        forward_split_attention(attn, q, &k, &v, &mut refs, &metadata, None)
    }

    #[test]
    fn indexed_row_hook_preserves_row_alignment() {
        let q = crate::zeros(&[3, 1, 1], crate::dtype::FLOAT32);
        assert_eq!(
            crate::utils::array_to_vec_f32(&run(&Indexed, &q)),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn indexed_row_hook_defaults_to_legacy_attend_row() {
        let q = crate::from_slice_f32(&[3.0, 5.0, 7.0], &[3, 1, 1]);
        assert_eq!(
            crate::utils::array_to_vec_f32(&run(&Legacy, &q)),
            vec![3.0, 5.0, 7.0]
        );
    }
}
