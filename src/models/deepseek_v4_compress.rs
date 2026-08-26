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

//! DeepSeek-V4 pooled-KV compression: `PoolingCache` (reference
//! `models/cache.py`) and `Compressor` (reference `language.py`).
//!
//! Per compressed layer, every `compress_ratio` consecutive tokens are pooled
//! into one compressed KV row: `wkv` / `wgate` project the hidden state, a
//! learned absolute positional embedding `ape` (`[ratio, out_dim]`) is added
//! to the gate, a softmax over the window axis weights the rows, the weighted
//! sum is RMS-normed and Yarn-roped at `compress_rope_theta` with
//! `freq_scale = ratio`.
//!
//! Two compress functions exist and are NOT interchangeable:
//!
//! * `ratio == 128` uses `_simple_compress_kv`: softmax(gate_f32 + ape) over
//!   the window axis, cast back, weighted sum.
//! * `ratio == 4` runs in *overlap* mode: `out_dim = 2 * head_dim`, and
//!   `_overlap_compress_kv` splits kv/gate in half on the feature axis,
//!   shifts the first half one window BACK (zero / `-inf` prefix window),
//!   concatenates on the window axis (so each pooled row sees `2 * ratio`
//!   candidate rows), then softmaxes. Using the simple path for ratio 4 loads
//!   fine and generates garbage.
//!
//! `PoolingCache` buffers tokens that have not yet formed a full window
//! (`remainder`) and emits pooled rows only for complete windows. Prompt mode
//! (`L > 1`) and decode mode (`L == 1`) take different branches, mirroring
//! the reference exactly; the pooled-visibility mask is
//! `pool_idx < (offset + 1 + j) / ratio` and is `None` for `L == 1`.
//!
//! One consequence worth knowing, faithfully ported: the overlap shift is
//! applied WITHIN each ready batch, so the first window of any batch (every
//! decode-completed window, and the first window a chunked-prefill
//! continuation completes) gets the zero / `-inf` prefix instead of its real
//! predecessor's half. Pooled rows therefore depend slightly on how the
//! token stream was batched; that is reference behavior, not a cache bug.

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::get_weight_copy;
use super::rope::V4Rope;
use super::{ModelArgs, OVERLAP_COMPRESS_RATIO};

/// Cache for pooled (compressed) KV rows plus a remainder buffer of tokens
/// that have not yet formed a full `ratio`-sized window.
pub(crate) struct PoolingCache {
    pub(crate) ratio: i32,
    buf_kv: Option<UniquePtr<MlxArray>>,
    buf_gate: Option<UniquePtr<MlxArray>>,
    pub(crate) remainder: i32,
    pooled: Option<UniquePtr<MlxArray>>,
}

impl PoolingCache {
    pub(crate) fn new(ratio: i32) -> Self {
        Self {
            ratio,
            buf_kv: None,
            buf_gate: None,
            remainder: 0,
            pooled: None,
        }
    }

    /// Accumulate `kv` / `gate` (`[B, L, D]`) into the remainder buffer and
    /// return the window-complete prefix ready for pooling plus the absolute
    /// position of its first token (`pool_base`).
    ///
    /// `offset` is the number of tokens seen BEFORE this call (the local
    /// cache's pre-update offset). Mirrors `PoolingCache.accumulate_windows`.
    pub(crate) fn accumulate_windows(
        &mut self,
        kv: &MlxArray,
        gate: &MlxArray,
        offset: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32) {
        let kv_shape = mlxcel_core::array_shape(kv);
        let (b, l, d1) = (kv_shape[0], kv_shape[1], kv_shape[2]);
        let d2 = mlxcel_core::array_shape(gate)[2];

        if self.buf_kv.is_none() {
            self.buf_kv = Some(mlxcel_core::zeros(
                &[b, self.ratio, d1],
                mlxcel_core::array_dtype(kv),
            ));
            self.buf_gate = Some(mlxcel_core::zeros(
                &[b, self.ratio, d2],
                mlxcel_core::array_dtype(gate),
            ));
        }

        if l > 1 {
            // Prompt mode.
            let total = l + self.remainder;
            let usable = (total / self.ratio) * self.ratio;
            let new_remainder = total % self.ratio;

            let (r_kv, r_gate, r_base, write_start) = if usable > 0 {
                let head_kv = mlxcel_core::utils::slice_axis(kv, 1, 0, usable - self.remainder);
                let head_gate = mlxcel_core::utils::slice_axis(gate, 1, 0, usable - self.remainder);
                let (r_kv, r_gate) = if self.remainder > 0 {
                    let buf_kv = self.buf_kv.as_ref().expect("buf_kv initialized");
                    let buf_gate = self.buf_gate.as_ref().expect("buf_gate initialized");
                    (
                        mlxcel_core::concatenate(
                            &mlxcel_core::utils::slice_axis(buf_kv, 1, 0, self.remainder),
                            &head_kv,
                            1,
                        ),
                        mlxcel_core::concatenate(
                            &mlxcel_core::utils::slice_axis(buf_gate, 1, 0, self.remainder),
                            &head_gate,
                            1,
                        ),
                    )
                } else {
                    (head_kv, head_gate)
                };
                let r_base = offset - self.remainder;
                (r_kv, r_gate, r_base, 0)
            } else {
                let empty_kv = mlxcel_core::utils::slice_axis(kv, 1, 0, 0);
                let empty_gate = mlxcel_core::utils::slice_axis(gate, 1, 0, 0);
                (empty_kv, empty_gate, 0, self.remainder)
            };

            if new_remainder > 0 {
                let tail_len = new_remainder - write_start;
                let tail_kv = mlxcel_core::utils::slice_axis(kv, 1, l - tail_len, l);
                let tail_gate = mlxcel_core::utils::slice_axis(gate, 1, l - tail_len, l);
                let buf_kv = self.buf_kv.take().expect("buf_kv initialized");
                let buf_gate = self.buf_gate.take().expect("buf_gate initialized");
                self.buf_kv = Some(mlxcel_core::slice_update(
                    &buf_kv,
                    &tail_kv,
                    &[0, write_start, 0],
                    &[b, new_remainder, d1],
                ));
                self.buf_gate = Some(mlxcel_core::slice_update(
                    &buf_gate,
                    &tail_gate,
                    &[0, write_start, 0],
                    &[b, new_remainder, d2],
                ));
            }
            self.remainder = new_remainder;

            (r_kv, r_gate, r_base)
        } else {
            // Decode mode: buffer one token; emit the full window when it
            // completes.
            let buf_kv = self.buf_kv.take().expect("buf_kv initialized");
            let buf_gate = self.buf_gate.take().expect("buf_gate initialized");
            let buf_kv = mlxcel_core::slice_update(
                &buf_kv,
                kv,
                &[0, self.remainder, 0],
                &[b, self.remainder + 1, d1],
            );
            let buf_gate = mlxcel_core::slice_update(
                &buf_gate,
                gate,
                &[0, self.remainder, 0],
                &[b, self.remainder + 1, d2],
            );
            self.remainder = (self.remainder + 1) % self.ratio;

            let result = if self.remainder == 0 {
                (
                    mlxcel_core::copy(&buf_kv),
                    mlxcel_core::copy(&buf_gate),
                    offset - self.ratio + 1,
                )
            } else {
                (
                    mlxcel_core::utils::slice_axis(kv, 1, 0, 0),
                    mlxcel_core::utils::slice_axis(gate, 1, 0, 0),
                    0,
                )
            };
            self.buf_kv = Some(buf_kv);
            self.buf_gate = Some(buf_gate);
            result
        }
    }

    /// Append newly pooled rows (`[B, Nw, D]`, possibly `Nw == 0`) and BORROW
    /// back the full pooled buffer (`[B, Np, D]`, `Np` possibly 0).
    ///
    /// The borrow is the point. The reference
    /// (`references/mlx-vlm/mlx_vlm/models/cache.py`,
    /// `PoolingCache.update_and_fetch`) returns `self.pooled` itself, and
    /// `mlxcel_core::copy` is `mlx::core::copy` (`mlx/ops.cpp:340`), which
    /// builds a real `Copy` primitive rather than aliasing a handle. Copying
    /// here materialised the whole `Np * D` buffer on EVERY call, including
    /// the `n_new == 0` calls that dominate decode: 3 of every 4 steps at
    /// compress ratio 4 and 127 of every 128 at ratio 128, with `Np` growing
    /// with context. On the real 43-layer config that was on the order of a
    /// gigabyte of pure memcpy per decoded token at 128k context. Do not
    /// reintroduce the copy: hand callers the borrow and let them clone only
    /// if they genuinely need an owned handle.
    pub(crate) fn update_and_fetch(&mut self, px: UniquePtr<MlxArray>) -> &MlxArray {
        let n_new = mlxcel_core::array_shape(&px)[1];
        if n_new > 0 {
            // An empty placeholder from the branch below is dropped rather
            // than concatenated: the reference never stores one, and feeding
            // a `[B, 0, D]` array built from the hidden state's dtype into
            // `concatenate` would put dtype promotion between the cache and
            // the pooled rows for no gain.
            let prev = self
                .pooled
                .take()
                .filter(|p| mlxcel_core::array_shape(p)[1] > 0);
            self.pooled = Some(match prev {
                Some(prev) => mlxcel_core::concatenate(&prev, &px, 1),
                None => px,
            });
        } else if self.pooled.is_none() {
            // Nothing pooled yet and nothing new: park the empty `[B, 0, D]`
            // array so there is always something to borrow back. The
            // reference returns a freshly built empty array here; holding
            // this one is what lets the return be a borrow.
            self.pooled = Some(px);
        }
        self.pooled.as_deref().expect("pooled buffer set above")
    }

    /// Force MLX to materialise this cache's three buffers (`buf_kv`,
    /// `buf_gate`, `pooled`), detaching the graph that built them from the
    /// hidden states that fed it.
    ///
    /// MLX is lazy, and an unevaluated array keeps its inputs alive through
    /// its primitive until `eval` detaches it. For the ATTENTION compressor's
    /// cache that never needs saying, because `V4Attention::forward` consumes
    /// its `pooled` buffer on every branch it can take (the dense
    /// `concatenate(local_kv, pooled)` path, or `sparse_pooled_attention`), so
    /// `pooled` is in the logits graph and the caller's eval of the logits
    /// already forces it, and forcing it transitively forces the
    /// `copy(&buf_kv)` that fed it and the bounded `slice_update` chains
    /// behind that. Calling this on the attention cache is therefore close to
    /// free: the work is done by the time the barrier runs.
    ///
    /// The INDEXER's cache is what this is for. The only consumer of the
    /// indexer's pooled buffer is the top-k selection `Indexer::forward`
    /// returns, and the `AttnKind::Sparse` arm of `V4Attention::forward`
    /// discards that selection on two of its three branches: `np == 0`, and
    /// `np <= index_topk`, the short-context branch that falls back to the
    /// dense concat. Nothing else reads `idx_pool`. So for as long as the
    /// sparse selection path is not being taken, the indexer cache's `pooled`
    /// concat chain grows by one unevaluated node per completed window and its
    /// `buf_kv` / `buf_gate` `slice_update` chains grow by one per DECODE
    /// STEP, none of them ever forced, each node pinning that step's normed
    /// hidden state plus its `wkv` / `wgate` projections. The sparse branch
    /// only fires past `index_topk * compress_ratio` tokens (512 * 4 = 2048 on
    /// the real config), so every sequence pays this over its first ~2048
    /// tokens and a sequence that never gets that long pays it for its whole
    /// life: on the order of 190 KB per decode step across the real config's
    /// 20 sparse layers, released only when the sparse path finally fires or
    /// the sequence state is dropped, and then forced in one burst down a
    /// chain thousands of nodes deep.
    ///
    /// Modelled on [`mlxcel_core::cache::KVCache::eval_state`], which plays
    /// the same role on the ordinary KV path (upstream mlx-lm evaluates
    /// `[c.state for c in cache]` after each step) and which deliberately
    /// evaluates only the cache state so the LM-head matmul and its peak
    /// allocation are not forced with it.
    ///
    /// The barrier belongs at the end of the model forward, not at the
    /// consumer. A consumer-side eval would fire inside the layer loop, once
    /// per compressed layer, cutting the graph while the rest of the stack is
    /// still being built; and it could not run at all on the branches that
    /// drop the selection, which are exactly the branches that accumulate.
    /// One barrier after the layer loop forces every layer's cache state at a
    /// point where the whole stack is already in the graph.
    pub(crate) fn eval_state(&self) {
        if let Some(buf_kv) = self.buf_kv.as_ref() {
            mlxcel_core::eval(buf_kv);
        }
        if let Some(buf_gate) = self.buf_gate.as_ref() {
            mlxcel_core::eval(buf_gate);
        }
        if let Some(pooled) = self.pooled.as_ref() {
            mlxcel_core::eval(pooled);
        }
    }
}

/// Per-query visible pooled-row counts: query `j` of this call (absolute
/// position `offset + j`) sees pooled rows `< (offset + 1 + j) / ratio`,
/// capped at `np`. Mirrors `PoolingCache.make_mask`.
pub(crate) fn pool_visible_counts(l: i32, offset: i32, ratio: i32, np: i32) -> Vec<i32> {
    (0..l).map(|j| ((offset + 1 + j) / ratio).min(np)).collect()
}

/// Additive f32 `[L, Np]` pooled-visibility mask (`0.0` visible, `-inf`
/// blocked), or `None` for `L == 1` / no pooled rows, matching the reference
/// (`make_mask` returns `None` during decode).
pub(crate) fn pool_mask_additive(
    l: i32,
    offset: i32,
    ratio: i32,
    np: i32,
) -> Option<UniquePtr<MlxArray>> {
    if l == 1 || np == 0 {
        return None;
    }
    let counts = pool_visible_counts(l, offset, ratio, np);
    let counts = mlxcel_core::from_slice_i32(&counts, &[l, 1]);
    let idx = mlxcel_core::reshape(&mlxcel_core::arange_i32(0, np, 1), &[1, np]);
    let visible = mlxcel_core::less(&idx, &counts);
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::dtype::FLOAT32);
    Some(mlxcel_core::where_cond(&visible, &zero, &neg_inf))
}

/// Trim or left-pad a `[L, S]` additive local mask so its key axis matches
/// the live local KV length. Mirrors `_align_local_mask`.
pub(crate) fn align_local_mask(mask: &MlxArray, local_len: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(mask);
    let current = *shape.last().expect("mask must have a key axis");
    if current == local_len {
        return mlxcel_core::copy(mask);
    }
    if current > local_len {
        return mlxcel_core::utils::slice_axis(mask, -1, current - local_len, -1);
    }
    let mut pad_shape = shape.clone();
    *pad_shape.last_mut().expect("mask rank checked") = local_len - current;
    let pad = mlxcel_core::zeros(&pad_shape, mlxcel_core::array_dtype(mask));
    mlxcel_core::concatenate(&pad, mask, -1)
}

/// Concatenate the pooled-visibility mask onto the aligned local mask
/// (additive f32). `None` pooled mask means every pooled row is visible.
/// Mirrors `_extend_mask` for the additive representation.
pub(crate) fn extend_mask(
    local_mask: &MlxArray,
    pool_mask: Option<&MlxArray>,
    np: i32,
) -> UniquePtr<MlxArray> {
    match pool_mask {
        Some(pm) => mlxcel_core::concatenate(local_mask, pm, -1),
        None => {
            let shape = mlxcel_core::array_shape(local_mask);
            let mut pad_shape = shape.clone();
            *pad_shape.last_mut().expect("mask rank checked") = np;
            let visible = mlxcel_core::zeros(&pad_shape, mlxcel_core::array_dtype(local_mask));
            mlxcel_core::concatenate(local_mask, &visible, -1)
        }
    }
}

/// Window compressor: projects, pools, norms, and ropes complete windows.
pub(crate) struct Compressor {
    wkv: UnifiedLinear,
    wgate: UnifiedLinear,
    /// `[ratio, out_dim]` learned absolute positional embedding.
    ape: UniquePtr<MlxArray>,
    norm: RMSNorm,
    rope: V4Rope,
    pub(crate) ratio: i32,
    head_dim: i32,
    overlap: bool,
    out_dim: i32,
}

impl Compressor {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        compress_ratio: i32,
        head_dim: i32,
    ) -> Result<Self, String> {
        let overlap = compress_ratio == OVERLAP_COMPRESS_RATIO;
        let out_dim = head_dim * if overlap { 2 } else { 1 };
        let group_size = args.group_size();
        let bits = args.bits();

        let ape = get_weight_copy(weights, &format!("{prefix}.ape"))?;
        let ape_shape = mlxcel_core::array_shape(&ape);
        if ape_shape != [compress_ratio, out_dim] {
            return Err(format!(
                "{prefix}.ape: expected shape [{compress_ratio}, {out_dim}], checkpoint ships \
                 {ape_shape:?}"
            ));
        }

        Ok(Self {
            wkv: UnifiedLinear::from_weights(weights, &format!("{prefix}.wkv"), group_size, bits)?,
            wgate: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.wgate"),
                group_size,
                bits,
            )?,
            ape,
            norm: RMSNorm::new(
                get_weight_copy(weights, &format!("{prefix}.norm.weight"))?,
                args.rms_norm_eps,
            ),
            rope: V4Rope::new(
                args.qk_rope_head_dim as i32,
                args.compress_rope_theta,
                args.rope_scaling.as_ref(),
                compress_ratio,
                &[(head_dim, false)],
            )?,
            ratio: compress_ratio,
            head_dim,
            overlap,
            out_dim,
        })
    }

    /// Pool the window-complete portion of `x` (`[B, L, hidden]`) and borrow
    /// the full pooled buffer `[B, Np, head_dim]` out of `pool`. `offset` is
    /// the pre-update token offset of this call.
    ///
    /// The return borrows `pool` for exactly the reason
    /// [`PoolingCache::update_and_fetch`] documents: the buffer is the cache's
    /// own, and copying it out per call is `Np * D` of memcpy that grows with
    /// context.
    pub(crate) fn forward<'a>(
        &self,
        x: &MlxArray,
        pool: &'a mut PoolingCache,
        offset: i32,
    ) -> &'a MlxArray {
        let kv = self.wkv.forward(x);
        let gate = self.wgate.forward(x);
        let (ready_kv, ready_gate, pool_base) = pool.accumulate_windows(&kv, &gate, offset);

        let ready_len = mlxcel_core::array_shape(&ready_kv)[1];
        let new_pooled = if ready_len == 0 {
            let b = mlxcel_core::array_shape(x)[0];
            mlxcel_core::zeros(&[b, 0, self.head_dim], mlxcel_core::array_dtype(x))
        } else {
            let b = mlxcel_core::array_shape(&ready_kv)[0];
            let nw = ready_len / self.ratio;
            let kv4 = mlxcel_core::reshape(&ready_kv, &[b, nw, self.ratio, self.out_dim]);
            let gate4 = mlxcel_core::reshape(&ready_gate, &[b, nw, self.ratio, self.out_dim]);
            let pooled = if self.overlap {
                overlap_compress_kv(&kv4, &gate4, &self.ape)
            } else {
                simple_compress_kv(&kv4, &gate4, &self.ape)
            };
            let pooled = self.norm.forward(&pooled);
            let pooled4 = mlxcel_core::expand_dims(&pooled, 1);
            let pooled4 = self.rope.apply(&pooled4, pool_base, false);
            mlxcel_core::squeeze_axis(&pooled4, 1)
        };

        pool.update_and_fetch(new_pooled)
    }
}

/// `_simple_compress_kv`: softmax(gate_f32 + ape) over the window axis in
/// f32, cast back to kv dtype, weighted sum.
pub(crate) fn simple_compress_kv(
    kv: &MlxArray,
    gate: &MlxArray,
    ape: &MlxArray,
) -> UniquePtr<MlxArray> {
    let gate_f32 = mlxcel_core::astype(gate, mlxcel_core::dtype::FLOAT32);
    let gate_f32 = mlxcel_core::add(
        &gate_f32,
        &mlxcel_core::astype(ape, mlxcel_core::dtype::FLOAT32),
    );
    let weights = mlxcel_core::softmax(&gate_f32, -2);
    let weights = mlxcel_core::astype(&weights, mlxcel_core::array_dtype(kv));
    let weighted = mlxcel_core::multiply(kv, &weights);
    mlxcel_core::sum_axis(&weighted, -2, false)
}

/// `_overlap_compress_kv`: split kv/gate in half on the feature axis, shift
/// the first half one window back (zero / `-inf` prefix window), concatenate
/// on the window axis, softmax (precise) over the doubled window, weighted
/// sum. Output feature width is `out_dim / 2 == head_dim`.
pub(crate) fn overlap_compress_kv(
    kv: &MlxArray,
    gate: &MlxArray,
    ape: &MlxArray,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(kv);
    let (b, nw, r, d) = (shape[0], shape[1], shape[2], shape[3]);
    let half = d / 2;
    let kv_dtype = mlxcel_core::array_dtype(kv);

    let gate = mlxcel_core::add(
        gate,
        &mlxcel_core::astype(ape, mlxcel_core::array_dtype(gate)),
    );

    let shift_back = |t: &MlxArray, fill: f32| -> UniquePtr<MlxArray> {
        // t is [B, Nw, R, half]; prepend a fill window and drop the last.
        let prefix = mlxcel_core::full_f32(&[b, 1, r, half], fill, kv_dtype);
        if nw == 1 {
            prefix
        } else {
            let head = mlxcel_core::utils::slice_axis(t, 1, 0, nw - 1);
            mlxcel_core::concatenate(&prefix, &head, 1)
        }
    };

    let kv_a = mlxcel_core::utils::slice_axis(kv, -1, 0, half);
    let kv_b = mlxcel_core::utils::slice_axis(kv, -1, half, -1);
    let kv_a = shift_back(&kv_a, 0.0);
    let kv_cat = mlxcel_core::concatenate(&kv_a, &kv_b, 2);

    let gate_a = mlxcel_core::utils::slice_axis(&gate, -1, 0, half);
    let gate_b = mlxcel_core::utils::slice_axis(&gate, -1, half, -1);
    let gate_a = shift_back(&gate_a, f32::NEG_INFINITY);
    let gate_b = mlxcel_core::astype(&gate_b, kv_dtype);
    let gate_cat = mlxcel_core::concatenate(&gate_a, &gate_b, 2);

    let weights = mlxcel_core::softmax_precise(&gate_cat, -2);
    let weighted = mlxcel_core::multiply(&kv_cat, &weights);
    mlxcel_core::sum_axis(&weighted, -2, false)
}
