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

use crate::layers::{RMSNorm, UnifiedLinear};
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, row_invariant};

use super::conv::InklingShortConv;
use super::{InklingLayerCache, InklingLayerSpec, weight};

/// The most queries MLX's fused SDPA runs through its per-query vector
/// kernel, the one a single-token decode step takes.
const SDPA_VECTOR_ROWS: i32 = 8;

/// Inkling NoPE attention with learned banded relative logits.
pub struct InklingAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    r_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    k_sconv: InklingShortConv,
    v_sconv: InklingShortConv,
    rel_proj: UniquePtr<MlxArray>,
    is_sliding: bool,
    n_heads: i32,
    n_kv: i32,
    head_dim: i32,
    d_rel: i32,
    rel_extent: i32,
    window: i32,
    log_floor: Option<f32>,
    log_alpha: f32,
}

impl InklingAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        spec: &InklingLayerSpec,
    ) -> Result<Self, String> {
        let (n_heads, n_kv, head_dim) = spec.attention_heads();
        let linear = |name: &str| {
            UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                spec.quantization_group_size,
                spec.quantization_bits,
            )
        };
        let rel_extent = spec.relative_extent();
        let rel_proj = weight(weights, &format!("{prefix}.rel_proj"))?;
        let rel_shape = crate::array_shape(&rel_proj);
        if rel_shape != [spec.d_rel as i32, rel_extent as i32] {
            return Err(format!(
                "{prefix}.rel_proj: expected [{}, {}], got {rel_shape:?}",
                spec.d_rel, rel_extent
            ));
        }
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            r_proj: linear("r_proj")?,
            o_proj: linear("o_proj")?,
            q_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.q_norm.weight"))?,
                spec.rms_norm_eps,
            ),
            k_norm: RMSNorm::new(
                weight(weights, &format!("{prefix}.k_norm.weight"))?,
                spec.rms_norm_eps,
            ),
            k_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.k_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
            v_sconv: InklingShortConv::from_weights(
                weights,
                &format!("{prefix}.v_sconv.conv.weight"),
                spec.sconv_kernel_size,
            )?,
            rel_proj,
            is_sliding: spec.is_sliding,
            n_heads: n_heads as i32,
            n_kv: n_kv as i32,
            head_dim: head_dim as i32,
            d_rel: spec.d_rel as i32,
            rel_extent: rel_extent as i32,
            window: spec.sliding_window_size as i32,
            log_floor: (!spec.is_sliding)
                .then_some(spec.log_scaling_n_floor)
                .flatten()
                .map(|value| value as f32),
            log_alpha: spec.log_scaling_alpha,
        })
    }

    pub fn forward(&self, input: &MlxArray, cache: &mut InklingLayerCache) -> UniquePtr<MlxArray> {
        let shape = crate::array_shape(input);
        let (batch, length) = (shape[0], shape[1]);
        // Row-invariant inside the target's scope (issue #3409); the plain
        // projection elsewhere, the MTP drafter's layers included.
        let q = row_invariant::linear(&self.q_proj, input);
        let k = row_invariant::linear(&self.k_proj, input);
        let v = row_invariant::linear(&self.v_proj, input);
        let r = row_invariant::linear(&self.r_proj, input);
        let k = self.k_sconv.forward_in_cache(&k, cache, 0, None);
        let v = self.v_sconv.forward_in_cache(&v, cache, 1, None);

        let q = crate::reshape(&q, &[batch, length, self.n_heads, self.head_dim]);
        let k = crate::reshape(&k, &[batch, length, self.n_kv, self.head_dim]);
        let v = crate::reshape(&v, &[batch, length, self.n_kv, self.head_dim]);
        let r = crate::reshape(&r, &[batch, length, self.n_heads, self.d_rel]);
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);
        let q = crate::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = crate::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = crate::transpose_axes(&v, &[0, 2, 1, 3]);
        let out = self.attend_row(q, k, v, &r, cache, length);
        let out = crate::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = crate::reshape(&out, &[batch, length, self.n_heads * self.head_dim]);
        row_invariant::linear(&self.o_proj, &out)
    }
    fn attend_row(
        &self,
        mut q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        r: &MlxArray,
        cache: &mut InklingLayerCache,
        length: i32,
    ) -> UniquePtr<MlxArray> {
        let (mut keys, mut values) = cache.kv.update_and_fetch(k, v);
        let before = crate::array_shape(&keys)[2];
        if self.is_sliding && before > length + self.window - 1 {
            let excess = before - (length + self.window - 1);
            cache.kv.trim_front(excess);
            keys = crate::utils::slice_axis(&keys, 2, excess, before);
            values = crate::utils::slice_axis(&values, 2, excess, before);
        }
        let source = crate::array_shape(&keys)[2];
        let offset = source - length;
        let mut mask = banded_additive_mask(
            r,
            &self.rel_proj,
            offset,
            source,
            self.is_sliding.then_some(self.window),
            self.rel_extent,
        );
        if let Some(floor) = self.log_floor {
            let tau = log_scaling_tau(length, offset, floor, self.log_alpha);
            let tau4 = crate::reshape(&tau, &[1, 1, length, 1]);
            let tau4 = crate::astype(&tau4, crate::array_dtype(&q));
            q = crate::multiply(&q, &tau4);
            let threshold = crate::full_like(&mask, -1e29);
            let valid = crate::greater(&mask, &threshold);
            let scaled = crate::multiply(&mask, &tau4);
            mask = crate::where_cond(&valid, &scaled, &mask);
        }
        // GQA with a per-head mask: the KV head axis must match the query's
        // outright, so expand it here (see `repeat_kv_heads`).
        if self.n_kv != self.n_heads {
            keys = repeat_kv_heads(&keys, self.n_heads, self.n_kv);
            values = repeat_kv_heads(&values, self.n_heads, self.n_kv);
        }
        let mask = crate::astype(&mask, crate::array_dtype(&q));
        // Inside a verify block, row r of a sliding layer past its window
        // would also see the r oldest keys, masked out, which decode step r
        // has already dropped; the extra leading columns move every visible
        // key to another lane of the SDPA reduction (issues #3280, #3409).
        // Attend row by row from each row's window start; the masked future
        // keys after the visible ones stay, they add exact zeros.
        let first_key = |row: i32| (offset + row - self.window + 1).max(0);
        if self.is_sliding
            && length > 1
            && first_key(length - 1) > 0
            && row_invariant::in_verify_block()
        {
            let outputs: Vec<_> = (0..length)
                .map(|row| {
                    let start = first_key(row);
                    let rows = |value: &MlxArray| crate::utils::slice_axis(value, 2, row, row + 1);
                    let window =
                        |value: &MlxArray| crate::utils::slice_axis(value, 2, start, source);
                    let mask = crate::utils::slice_axis(&rows(&mask), 3, start, source);
                    self.sdpa(&rows(&q), &window(&keys), &window(&values), &mask)
                })
                .collect();
            let refs: Vec<&MlxArray> = outputs.iter().map(|output| &**output).collect();
            return crate::concatenate_many(&refs, 2);
        }
        // MLX's fused SDPA treats each query alone (the decode step's vector
        // kernel) only up to `SDPA_VECTOR_ROWS` queries and switches to its
        // tiled kernel past that, which orders the key reduction differently.
        // A wider verify block (the MTP default is 10 rows on Inkling-Small)
        // therefore attends in groups of at most that many rows (issue #3280).
        if length > SDPA_VECTOR_ROWS && row_invariant::in_verify_block() {
            let outputs: Vec<_> = (0..length)
                .step_by(SDPA_VECTOR_ROWS as usize)
                .map(|start| {
                    let end = (start + SDPA_VECTOR_ROWS).min(length);
                    let rows = |value: &MlxArray| crate::utils::slice_axis(value, 2, start, end);
                    self.sdpa(&rows(&q), &keys, &values, &rows(&mask))
                })
                .collect();
            let refs: Vec<&MlxArray> = outputs.iter().map(|output| &**output).collect();
            return crate::concatenate_many(&refs, 2);
        }
        self.sdpa(&q, &keys, &values, &mask)
    }

    /// SDPA of `q [1, heads, L, d]` over `keys` / `values [1, heads, S, d]`
    /// with the additive per-head `mask [1, heads, L, S]`.
    fn sdpa(
        &self,
        q: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mask_ptr = mask as *const MlxArray;
        // SAFETY: `mask_ptr` points to `mask`, which remains alive for the
        // duration of this call; q/k/v are valid MLX arrays owned above.
        // The MLX fast kernel keeps the query dtype: the composed fallback's
        // softmax promotes to float32 and poisons the whole residual stream,
        // which also breaks the expert-scales dtype match in the fused MoE
        // decode kernel (issue #2479).
        unsafe {
            crate::fast_scaled_dot_product_attention(
                q,
                keys,
                values,
                1.0 / self.head_dim as f32,
                mask_ptr,
            )
        }
    }

    pub(super) fn forward_batched_decode(
        &self,
        input: &MlxArray,
        caches: &mut [&mut InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let b = caches.len() as i32;
        let q = row_invariant::linear(&self.q_proj, input);
        let k = row_invariant::linear(&self.k_proj, input);
        let v = row_invariant::linear(&self.v_proj, input);
        let r = row_invariant::linear(&self.r_proj, input);
        let k = self.k_sconv.forward_batched_decode(&k, caches, 0, None);
        let v = self.v_sconv.forward_batched_decode(&v, caches, 1, None);
        let q = crate::reshape(&q, &[b, 1, self.n_heads, self.head_dim]);
        let k = crate::reshape(&k, &[b, 1, self.n_kv, self.head_dim]);
        let v = crate::reshape(&v, &[b, 1, self.n_kv, self.head_dim]);
        let r = crate::reshape(&r, &[b, 1, self.n_heads, self.d_rel]);
        let q = crate::transpose_axes(&self.q_norm.forward(&q), &[0, 2, 1, 3]);
        let k = crate::transpose_axes(&self.k_norm.forward(&k), &[0, 2, 1, 3]);
        let v = crate::transpose_axes(&v, &[0, 2, 1, 3]);
        // Relative bias and sliding KV length are sequence-local.
        let rows: Vec<_> = caches
            .iter_mut()
            .enumerate()
            .map(|(i, cache)| {
                let row = |x: &MlxArray| crate::utils::slice_axis(x, 0, i as i32, i as i32 + 1);
                self.attend_row(row(&q), row(&k), row(&v), &row(&r), cache, 1)
            })
            .collect();
        let row_refs: Vec<_> = rows.iter().map(|row| &**row).collect();
        let out = crate::concatenate_many(&row_refs, 0);
        let out = crate::transpose_axes(&out, &[0, 2, 1, 3]);
        row_invariant::linear(
            &self.o_proj,
            &crate::reshape(&out, &[b, 1, self.n_heads * self.head_dim]),
        )
    }
}

/// Expand `n_kv` key/value heads to `n_heads` for grouped-query attention.
///
/// MLX's SDPA does GQA on its own, but only when it can broadcast the KV head
/// axis against the query's. Inkling is the one family here that hands it a
/// PER-HEAD mask: `banded_additive_mask` produces `[batch, n_heads, length,
/// source]` because the relative-position logits differ per query head, while
/// every other model in the tree passes the 2-D `[length, source]` mask
/// `create_causal_mask` builds, which broadcasts over heads for free.
///
/// With a per-head mask the KV axis has to match the query axis outright.
/// `n_kv == 1` broadcasts and so worked by accident, which is exactly what
/// every Inkling fixture used and why this reached real checkpoints: the
/// published ones use 4 sliding KV heads against 8 query heads (0.6B) and 8
/// against 32 (Small), and both aborted with
/// `[broadcast_shapes] Shapes (1,8) and (1,4) cannot be broadcast` (issue #1549).
///
/// Repeating here rather than inside the shared SDPA wrapper keeps the cost
/// where the need is. The overwhelming majority of callers pass a 2-D mask and
/// would pay a pointless KV copy, which at long context is the traffic that
/// dominates decode.
///
/// Each KV head serves `n_heads / n_kv` CONSECUTIVE query heads, so the
/// expansion is head-major: insert a repeat axis directly after the KV axis and
/// fold it in, never `tile`, which would interleave the groups and silently
/// pair every query head with the wrong KV head.
///
/// Used by: Inkling (`InklingAttention`), Inkling 2 (`models::inkling2`)
pub fn repeat_kv_heads(x: &MlxArray, n_heads: i32, n_kv: i32) -> UniquePtr<MlxArray> {
    let shape = crate::array_shape(x);
    let (batch, length, head_dim) = (shape[0], shape[2], shape[3]);
    let repeats = n_heads / n_kv;
    let expanded = crate::reshape(x, &[batch, n_kv, 1, length, head_dim]);
    let expanded = crate::broadcast_to(&expanded, &[batch, n_kv, repeats, length, head_dim]);
    let expanded = crate::contiguous(&expanded, false);
    crate::reshape(&expanded, &[batch, n_heads, length, head_dim])
}

pub fn banded_additive_mask(
    r: &MlxArray,
    projection: &MlxArray,
    offset: i32,
    source: i32,
    sliding_window: Option<i32>,
    rel_extent: i32,
) -> UniquePtr<MlxArray> {
    let shape = crate::array_shape(r);
    let (batch, length, heads) = (shape[0], shape[1], shape[2]);
    let relative = crate::matmul(r, projection);
    let relative = crate::transpose_axes(&relative, &[0, 2, 1, 3]);
    let queries = crate::arange_i32(offset, offset + length, 1);
    let queries = crate::reshape(&queries, &[length, 1]);
    let keys = crate::arange_i32(0, source, 1);
    let keys = crate::reshape(&keys, &[1, source]);
    let dist = crate::subtract(&queries, &keys);
    let dist = crate::reshape(&dist, &[1, 1, length, source]);
    let dist = crate::broadcast_to(&dist, &[batch, heads, length, source]);
    let low = crate::from_slice_i32(&[0], &[1]);
    let high = crate::from_slice_i32(&[rel_extent - 1], &[1]);
    let gather = crate::clip(&dist, &low, &high);
    let gathered = crate::take_along_axis(&relative, &gather, -1);
    let extent = crate::from_slice_i32(&[rel_extent], &[1]);
    let past_extent = crate::greater_equal(&dist, &extent);
    let zeros = crate::zeros_like(&gathered);
    let positional = crate::where_cond(&past_extent, &zeros, &gathered);
    let zero = crate::from_slice_i32(&[0], &[1]);
    let future = crate::less(&dist, &zero);
    let invalid = if let Some(window) = sliding_window {
        let window = crate::from_slice_i32(&[window], &[1]);
        crate::logical_or(&future, &crate::greater_equal(&dist, &window))
    } else {
        future
    };
    let blocked = crate::full_like(&positional, -1e30);
    crate::where_cond(&invalid, &blocked, &positional)
}

pub fn log_scaling_tau(length: i32, offset: i32, floor: f32, alpha: f32) -> UniquePtr<MlxArray> {
    let positions = crate::arange_f32((offset + 1) as f32, (offset + length + 1) as f32, 1.0);
    let ratio = crate::divide_scalar(&positions, floor);
    let ratio = crate::maximum(&ratio, &crate::ones_like(&ratio));
    let scaled = crate::multiply_scalar(&crate::log(&ratio), alpha);
    crate::add(&crate::ones_like(&scaled), &scaled)
}
