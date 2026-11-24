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

//! Inkling 2 attention: l2-clipped q/k/r, gamma-free value norm, banded
//! relative logits, a query-dependent sink and a sigmoid output gate.
//!
//! Derived from mlx-vlm `Attention` (`mlx_vlm/models/inkling2/language.py`,
//! MIT).
//!
//! # KV layout
//!
//! mlx-vlm keeps an `AttentionSinkKVCache` for every visit of a looped model:
//! chronological K/V, a zero row after the newest key, and on a sliding visit
//! the keys `max(0, end - window + 1)..end + length`. Here each visit keeps a
//! chronological `KVCache`; the zero sink row is appended per call, and a
//! sliding visit drops keys from the front once more than
//! `length + window - 1 + SLIDING_KV_MARGIN` are held, then attends over the
//! newest `length + window - 1`. That is the same key set in the same order
//! with the same mask, so the attention output is the same. A tail trim of up
//! to `SLIDING_KV_MARGIN + 1` tokens leaves at least `window - 1` earlier keys,
//! the condition upstream's `trim` limit (`_end - max_size`) enforces.

use mlxcel_core::inkling_layer::{
    InklingLayerCache, banded_additive_mask, log_scaling_tau, repeat_kv_heads,
};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::batched::with_stacked_state;
use super::config::Inkling2TextConfig;
use super::conv::{DynamicShortConv, ShortConv};

/// Keys a sliding visit keeps beyond what the current block attends to.
pub(crate) const SLIDING_KV_MARGIN: i32 = 64;

fn scalar(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(&[value], &[1])
}

/// `n ** -0.5` rounded once from f64, as Python computes the scale.
pub(crate) fn inverse_sqrt(n: i32) -> f32 {
    (1.0 / f64::from(n).sqrt()) as f32
}

fn f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(x, dtype::FLOAT32)
}

/// `x * min(threshold / (||x * scale|| + 1e-6), 1)` in f32, cast back.
pub(crate) fn l2_clip(
    x: &MlxArray,
    threshold: f32,
    scale: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let out_dtype = mlxcel_core::array_dtype(x);
    let mut xf = f32(x);
    if let Some(scale) = scale {
        xf = mlxcel_core::multiply(&xf, scale);
    }
    let norm = mlxcel_core::sqrt(&mlxcel_core::sum_axis(&mlxcel_core::square(&xf), -1, true));
    let factor = mlxcel_core::divide(&scalar(threshold), &mlxcel_core::add(&norm, &scalar(1e-6)));
    let factor = mlxcel_core::minimum(&factor, &scalar(1.0));
    mlxcel_core::astype(&mlxcel_core::multiply(&xf, &factor), out_dtype)
}

enum KvConv {
    Static(ShortConv),
    Dynamic(DynamicShortConv),
}

pub struct Inkling2Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    r_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    gate_proj: UnifiedLinear,
    sink_proj: UnifiedLinear,
    /// f32 `[heads, head_dim]` / `[heads, d_rel]`.
    q_scale: UniquePtr<MlxArray>,
    rel_scale: UniquePtr<MlxArray>,
    rel_proj: UniquePtr<MlxArray>,
    k_conv: KvConv,
    v_conv: KvConv,
    is_sliding: bool,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    window: i32,
    rel_extent: i32,
    qk_clip: f32,
    rel_clip: f32,
    value_norm_eps: Option<f32>,
    log_scaling: Option<(f32, f32)>,
    /// Fused head preparation (issue #3359): a looped model (Mini) with the
    /// gamma-free value norm, kernels on at load.
    head_kernel: bool,
}

fn tensor(weights: &WeightMap, key: &str, shape: &[i32]) -> Result<UniquePtr<MlxArray>, String> {
    let value = weights
        .get(key)
        .ok_or_else(|| format!("Weight not found: {key}"))?;
    if mlxcel_core::array_shape(value) != shape {
        return Err(format!(
            "{key} must be {shape:?}, got {:?}",
            mlxcel_core::array_shape(value)
        ));
    }
    Ok(mlxcel_core::copy(value))
}

impl Inkling2Attention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Inkling2TextConfig,
        layer: usize,
        quant: (i32, i32),
    ) -> Result<Self, String> {
        let is_sliding = config.layer_is_sliding(layer);
        let (heads, kv_heads, head_dim) = config.attention_heads(layer);
        let window = if is_sliding {
            config.sliding_window_size
        } else {
            0
        };
        let rel_extent = if is_sliding {
            window
        } else {
            config.rel_extent
        };
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), quant.0, quant.1)
        };
        let kv_conv = |name: &str| -> Result<KvConv, String> {
            let conv_prefix = format!("{prefix}.{name}");
            if is_sliding {
                ShortConv::from_weights(weights, &conv_prefix, kv_heads * head_dim)
                    .map(KvConv::Static)
            } else {
                DynamicShortConv::from_weights(
                    weights,
                    &conv_prefix,
                    kv_heads,
                    head_dim,
                    config.sconv_kernel_size,
                    quant,
                )
                .map(KvConv::Dynamic)
            }
        };
        let q_scale = tensor(
            weights,
            &format!("{prefix}.q_scale"),
            &[heads as i32, head_dim as i32],
        )?;
        let rel_scale = tensor(
            weights,
            &format!("{prefix}.rel_scale"),
            &[heads as i32, config.d_rel as i32],
        )?;
        let rel_proj = tensor(
            weights,
            &format!("{prefix}.rel_proj"),
            &[config.d_rel as i32, rel_extent as i32],
        )?;
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            r_proj: linear("r_proj")?,
            o_proj: linear("o_proj")?,
            gate_proj: linear("gate_proj")?,
            sink_proj: linear("sink_proj")?,
            q_scale: f32(&q_scale),
            rel_scale: f32(&rel_scale),
            rel_proj,
            k_conv: kv_conv("k_sconv")?,
            v_conv: kv_conv("v_sconv")?,
            is_sliding,
            heads: heads as i32,
            kv_heads: kv_heads as i32,
            head_dim: head_dim as i32,
            d_rel: config.d_rel as i32,
            window: window as i32,
            rel_extent: rel_extent as i32,
            qk_clip: config.qk_l2_clip_threshold,
            rel_clip: config.rel_l2_clip_threshold,
            value_norm_eps: config
                .use_value_norm_no_gamma
                .then_some(config.rms_norm_eps),
            log_scaling: (!is_sliding)
                .then_some(config.log_scaling_n_floor)
                .flatten()
                .map(|floor| (floor as f32, config.log_scaling_alpha)),
            head_kernel: config.use_value_norm_no_gamma
                && super::kernels::loop_kernels_wanted(config, super::kernels::enabled()),
        })
    }

    pub fn is_sliding(&self) -> bool {
        self.is_sliding
    }

    pub fn window(&self) -> i32 {
        self.window
    }

    /// The k or v convolution (`slot` 0 / 1) over every row's history.
    fn conv(
        conv: &KvConv,
        x: &MlxArray,
        dynamic_input: &MlxArray,
        rows: &mut [&mut InklingLayerCache],
        slot: usize,
    ) -> UniquePtr<MlxArray> {
        let kernel = match conv {
            KvConv::Static(conv) => conv.kernel(),
            KvConv::Dynamic(conv) => conv.kernel(),
        };
        with_stacked_state(rows, slot, kernel, x, |state| match conv {
            KvConv::Static(conv) => conv.forward(x, state, None, mlxcel_core::array_dtype(x)),
            KvConv::Dynamic(conv) => conv.forward(x, dynamic_input, state),
        })
    }

    /// The fused kernel's l2-clipped q/k/r and normalized v, or `None` where
    /// its shape gate refuses.
    pub(crate) fn prepare_heads_kernel(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        r: &MlxArray,
    ) -> Option<[UniquePtr<MlxArray>; 4]> {
        let eps = self.value_norm_eps?;
        super::kernels::prepare_attention_heads(
            q,
            k,
            v,
            r,
            &self.q_scale,
            &self.rel_scale,
            self.qk_clip,
            self.rel_clip,
            eps,
        )
    }

    /// The ops path of [`Self::prepare_heads_kernel`].
    pub(crate) fn prepare_heads_ops(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        r: &MlxArray,
    ) -> [UniquePtr<MlxArray>; 4] {
        let q = l2_clip(q, self.qk_clip, Some(&self.q_scale));
        let k = l2_clip(k, self.qk_clip, None);
        let v = match self.value_norm_eps {
            Some(eps) => mlxcel_core::astype(
                &mlxcel_core::fast_rms_norm_no_weight(&f32(v), eps),
                mlxcel_core::array_dtype(v),
            ),
            None => mlxcel_core::copy(v),
        };
        let r = l2_clip(r, self.rel_clip, Some(&self.rel_scale));
        [q, k, v, r]
    }

    /// `x` feeds q, r, the gate and the sink; `key_input` / `value_input`
    /// feed k / v (all three are `x` on a sliding layer without messages).
    /// `rows` holds one cache per batch row of `x`; one row runs the
    /// single-sequence graph unchanged.
    pub fn forward(
        &self,
        x: &MlxArray,
        key_input: &MlxArray,
        value_input: &MlxArray,
        rows: &mut [&mut InklingLayerCache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, length) = (shape[0], shape[1]);
        assert_eq!(
            batch as usize,
            rows.len(),
            "Inkling 2 attention needs one cache per batch row"
        );
        let x_dtype = mlxcel_core::array_dtype(x);
        let k = mlxcel_core::row_invariant::linear(&self.k_proj, key_input);
        let v = mlxcel_core::row_invariant::linear(&self.v_proj, value_input);
        let k = Self::conv(&self.k_conv, &k, key_input, rows, 0);
        let v = Self::conv(&self.v_conv, &v, value_input, rows, 1);

        let q = mlxcel_core::reshape(
            &mlxcel_core::row_invariant::linear(&self.q_proj, x),
            &[batch, length, self.heads, self.head_dim],
        );
        let k = mlxcel_core::reshape(&k, &[batch, length, self.kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[batch, length, self.kv_heads, self.head_dim]);
        let r = mlxcel_core::reshape(
            &mlxcel_core::row_invariant::linear(&self.r_proj, x),
            &[batch, length, self.heads, self.d_rel],
        );
        let heads = match self
            .head_kernel
            .then(|| self.prepare_heads_kernel(&q, &k, &v, &r))
            .flatten()
        {
            Some(heads) => heads,
            None => self.prepare_heads_ops(&q, &k, &v, &r),
        };
        // A query-dependent sink is a zero K/V row with its own logit.
        let sinks = mlxcel_core::row_invariant::linear(&self.sink_proj, x);
        let sinks = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&sinks, &[0, 2, 1]), -1);
        let out = match rows {
            [cache] => {
                let [q, k, v, r] = heads;
                self.attend(q, k, v, r, &sinks, cache, x_dtype)
            }
            rows => {
                // Offsets, sliding windows and relative masks are per row.
                let outputs: Vec<_> = rows
                    .iter_mut()
                    .enumerate()
                    .map(|(i, cache)| {
                        let i = i as i32;
                        let row =
                            |value: &MlxArray| mlxcel_core::utils::slice_axis(value, 0, i, i + 1);
                        let [q, k, v, r] = std::array::from_fn(|part| row(&heads[part]));
                        self.attend(q, k, v, r, &row(&sinks), cache, x_dtype)
                    })
                    .collect();
                let refs: Vec<&MlxArray> = outputs.iter().map(|output| &**output).collect();
                mlxcel_core::concatenate_many(&refs, 0)
            }
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[batch, length, self.heads * self.head_dim]);
        let gate = mlxcel_core::sigmoid(&f32(&mlxcel_core::row_invariant::per_row(x, |x| {
            self.gate_proj.forward(x)
        })));
        let out = mlxcel_core::astype(&mlxcel_core::multiply(&f32(&out), &gate), x_dtype);
        mlxcel_core::row_invariant::linear(&self.o_proj, &out)
    }

    /// The attention core of one row: prepared `[1, L, heads, d]` q/k/v/r and
    /// `[1, heads, L, 1]` sink logits against that row's KV cache; returns
    /// `[1, heads, L, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        mut q: UniquePtr<MlxArray>,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        mut r: UniquePtr<MlxArray>,
        sinks: &MlxArray,
        cache: &mut InklingLayerCache,
        x_dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let length = mlxcel_core::array_shape(&q)[1];
        if let Some((floor, alpha)) = self.log_scaling {
            let tau = log_scaling_tau(length, cache.kv.offset, floor, alpha);
            let tau = mlxcel_core::reshape(&tau, &[length, 1, 1]);
            q = mlxcel_core::astype(&mlxcel_core::multiply(&f32(&q), &tau), x_dtype);
            r = mlxcel_core::astype(&mlxcel_core::multiply(&f32(&r), &tau), x_dtype);
        }
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        let (mut keys, mut values) = cache.kv.update_and_fetch(k, v);
        let held = mlxcel_core::array_shape(&keys)[2];
        if self.is_sliding {
            let keep = length + self.window - 1 + SLIDING_KV_MARGIN;
            if held > keep {
                cache.kv.trim_front(held - keep);
            }
            let visible = length + self.window - 1;
            if held > visible {
                keys = mlxcel_core::utils::slice_axis(&keys, 2, held - visible, held);
                values = mlxcel_core::utils::slice_axis(&values, 2, held - visible, held);
            }
        }
        let source = mlxcel_core::array_shape(&keys)[2];
        let rel_proj = mlxcel_core::astype(&self.rel_proj, x_dtype);
        let mask = banded_additive_mask(
            &r,
            &rel_proj,
            source - length,
            source,
            self.is_sliding.then_some(self.window),
            self.rel_extent,
        );
        // The sink column goes FIRST (issue #3280): ahead of the keys it sits
        // at the same index in a one-token step and in a K-row verify block,
        // so the SDPA reduction assigns every visible key to the same lane
        // in both and the block's masked future keys add exact zeros.
        let mask = mlxcel_core::concatenate(
            &mlxcel_core::astype(sinks, x_dtype),
            &mlxcel_core::astype(&mask, x_dtype),
            -1,
        );
        let zero_row = |value: &MlxArray| {
            let shape = mlxcel_core::array_shape(value);
            let zeros = mlxcel_core::zeros(
                &[shape[0], shape[1], 1, shape[3]],
                mlxcel_core::array_dtype(value),
            );
            mlxcel_core::concatenate(&zeros, value, 2)
        };
        let mut keys = zero_row(&keys);
        let mut values = zero_row(&values);
        // The mask is per head, so the KV head axis must match the query's.
        if self.kv_heads != self.heads {
            keys = repeat_kv_heads(&keys, self.heads, self.kv_heads);
            values = repeat_kv_heads(&values, self.heads, self.kv_heads);
        }
        let mask_ptr = &*mask as *const MlxArray;
        // SAFETY: q, keys, values and mask are live MLX arrays owned above.
        unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q,
                &keys,
                &values,
                inverse_sqrt(self.head_dim),
                mask_ptr,
            )
        }
    }
}
