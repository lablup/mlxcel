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

//! Inkling 2 causal short convolutions.
//!
//! Derived from mlx-vlm `ShortConvolution` and `DynamicShortConvolution`
//! (`mlx_vlm/models/inkling2/language.py`, MIT). Unlike Inkling 1's
//! convolution these include their own input skip and a `1/sqrt` scale, both
//! in f32.
//!
//! Each state is an f32 history `[B, rows, C]` whose last `kernel - 1` rows
//! are the convolution context. It keeps up to [`CONV_HISTORY_MARGIN`] older
//! rows so a tail trim of that many tokens can rewind it exactly (see
//! [`rewind_history`]); the first state is `kernel - 1` zero rows, as upstream.
//!
//! A speculative verify forward ([`mlxcel_core::cache::speculative_append_active`])
//! keeps at least as many older rows as it appends, so its whole block can be
//! rejected and rewound exactly even when it is longer than the margin.
//!
//! [`ShortConv`] runs the fused Metal kernel (mlx-vlm `kernels.py`, issue
//! #3359; gate in `super::kernels`) where it applies and the f32 `conv1d`
//! path otherwise. The kernel writes the same next history, so trims stay
//! exact on either path.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use mlxcel_core::layers::UnifiedLinear;

/// Older history rows kept beyond the convolution context, so a speculative
/// or rejected tail of up to this many tokens can be trimmed exactly.
pub(crate) const CONV_HISTORY_MARGIN: i32 = 16;

thread_local! {
    static WIDE_HISTORY_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// While alive, every forward keeps as many older history rows as it appends,
/// as a speculative verify does. A native batched forward of more than one
/// token (the server's padded prefill cohort) enters it so each row's right
/// padding can be trimmed exactly afterwards, however long it is (#3377).
pub(crate) struct WideHistoryScope(());

impl WideHistoryScope {
    pub(crate) fn enter() -> Self {
        WIDE_HISTORY_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self(())
    }
}

impl Drop for WideHistoryScope {
    fn drop(&mut self) {
        WIDE_HISTORY_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Older rows a forward of `length` tokens keeps: [`CONV_HISTORY_MARGIN`], or
/// `length` when a speculative verify block (or a [`WideHistoryScope`]
/// forward) is longer than that.
pub(crate) fn history_margin(length: i32) -> i32 {
    if mlxcel_core::cache::speculative_append_active()
        || WIDE_HISTORY_DEPTH.with(|depth| depth.get() > 0)
    {
        CONV_HISTORY_MARGIN.max(length)
    } else {
        CONV_HISTORY_MARGIN
    }
}

fn f32(x: &MlxArray) -> UniquePtr<MlxArray> {
    if mlxcel_core::array_dtype(x) == dtype::FLOAT32 {
        mlxcel_core::copy(x)
    } else {
        mlxcel_core::astype(x, dtype::FLOAT32)
    }
}

/// `(context, next_state)` for an f32 input `xf [B, L, C]`.
fn context_and_next(
    state: &Option<UniquePtr<MlxArray>>,
    xf: &MlxArray,
    kernel: i32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(xf);
    let width = kernel - 1;
    let history = match state.as_ref().and_then(|value| value.as_ref()) {
        Some(history) => mlxcel_core::copy(history),
        None => mlxcel_core::zeros(&[shape[0], width, shape[2]], dtype::FLOAT32),
    };
    let rows = mlxcel_core::array_shape(&history)[1];
    let tail = mlxcel_core::utils::slice_axis(&history, 1, rows - width, rows);
    let context = mlxcel_core::concatenate(&tail, xf, 1);
    let extended = mlxcel_core::concatenate(&history, xf, 1);
    let total = rows + shape[1];
    let keep = (width + history_margin(shape[1])).min(total);
    let next = mlxcel_core::contiguous(
        &mlxcel_core::utils::slice_axis(&extended, 1, total - keep, total),
        false,
    );
    (context, next)
}

/// Drop the newest `n` rows of a history. `Err` when fewer than
/// `kernel - 1` rows would remain, i.e. the trim reaches past the margin.
pub(crate) fn rewind_history(
    state: &MlxArray,
    n: i32,
    kernel: i32,
) -> Result<UniquePtr<MlxArray>, String> {
    let rows = mlxcel_core::array_shape(state)[1];
    if rows - n < kernel - 1 {
        return Err(format!(
            "Inkling 2 convolution history holds {rows} rows; trimming {n} tokens needs {}",
            n + kernel - 1
        ));
    }
    Ok(mlxcel_core::utils::slice_axis(state, 1, 0, rows - n))
}

/// Rows a rewind of `n` tokens needs, for checking before any mutation.
pub(crate) fn can_rewind(state: Option<&MlxArray>, n: i32, kernel: i32) -> bool {
    state.is_none_or(|state| mlxcel_core::array_shape(state)[1] - n >= kernel - 1)
}

/// Depthwise causal convolution of an f32 `context [B, K-1+L, C]` with an f32
/// `[C, K, 1]` weight, plus the f32 skip, divided by `sqrt(K + 1)`.
pub(crate) fn short_convolution(
    context: &MlxArray,
    weight: &MlxArray,
    xf: &MlxArray,
    kernel: i32,
) -> UniquePtr<MlxArray> {
    let channels = mlxcel_core::array_shape(xf)[2];
    let out = mlxcel_core::conv1d(context, weight, 1, 0, 1, channels);
    let out = mlxcel_core::add(&out, xf);
    mlxcel_core::divide_scalar(&out, ((kernel + 1) as f32).sqrt())
}

/// Static depthwise convolution (sliding k/v, `attn_sconv`, `mlp_sconv`).
pub struct ShortConv {
    weight: UniquePtr<MlxArray>,
    kernel: i32,
    /// [`super::kernels::KERNELS_ENV`] at load.
    use_kernel: bool,
}

impl ShortConv {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.conv.weight");
        let weight = weights
            .get(&key)
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let shape = mlxcel_core::array_shape(weight);
        if shape.len() != 3 || shape[0] != channels as i32 || shape[2] != 1 || shape[1] < 1 {
            return Err(format!(
                "{key} must be [{channels}, kernel, 1], got {shape:?}"
            ));
        }
        Ok(Self {
            weight: mlxcel_core::contiguous(&f32(weight), false),
            kernel: shape[1],
            use_kernel: super::kernels::enabled(),
        })
    }

    pub fn kernel(&self) -> i32 {
        self.kernel
    }

    /// `((conv + x) / sqrt(k + 1)).astype(x)`, then optionally
    /// `(out.f32 + residual.f32).astype(output_dtype)`.
    pub fn forward(
        &self,
        x: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
        output_dtype: i32,
    ) -> UniquePtr<MlxArray> {
        if self.use_kernel
            && let Some(out) = self.forward_kernel(x, state, residual, output_dtype)
        {
            return out;
        }
        self.forward_ops(x, state, residual, output_dtype)
    }

    /// The fused kernel, or `None` (state untouched) where its gate refuses.
    pub(crate) fn forward_kernel(
        &self,
        x: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
        output_dtype: i32,
    ) -> Option<UniquePtr<MlxArray>> {
        let shape = mlxcel_core::array_shape(x);
        let zeros;
        let history = match state.as_ref().and_then(|value| value.as_ref()) {
            Some(history) => history,
            None => {
                zeros = mlxcel_core::zeros(&[shape[0], self.kernel - 1, shape[2]], dtype::FLOAT32);
                &*zeros
            }
        };
        if !super::kernels::short_convolution_applies(x, history, self.kernel) {
            return None;
        }
        let (out, next) = mlxcel_core::inkling2_kernels::short_convolution(
            history,
            &self.weight,
            x,
            residual,
            output_dtype,
            history_margin(shape[1]),
        )?;
        super::kernels::record_dispatch(super::kernels::Kernel::ShortConvolution);
        *state = Some(next);
        Some(out)
    }

    /// The f32 `conv1d` path.
    pub(crate) fn forward_ops(
        &self,
        x: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
        residual: Option<&MlxArray>,
        output_dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let x_dtype = mlxcel_core::array_dtype(x);
        let xf = f32(x);
        let (context, next) = context_and_next(state, &xf, self.kernel);
        *state = Some(next);
        let out = short_convolution(&context, &self.weight, &xf, self.kernel);
        let out = mlxcel_core::astype(&out, x_dtype);
        let out = match residual {
            Some(residual) => mlxcel_core::add(&f32(&out), &f32(residual)),
            None => out,
        };
        if mlxcel_core::array_dtype(&out) == output_dtype {
            out
        } else {
            mlxcel_core::astype(&out, output_dtype)
        }
    }
}

/// Input-dependent per-head convolution of the global layers' k/v.
pub struct DynamicShortConv {
    /// `static_weight.T` as f32 `[heads, head_dim, kernel]`.
    static_weight: UniquePtr<MlxArray>,
    weight_proj: UnifiedLinear,
    heads: i32,
    kernel: i32,
}

impl DynamicShortConv {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        kernel: usize,
        quant: (i32, i32),
    ) -> Result<Self, String> {
        let key = format!("{prefix}.static_weight");
        let value = weights
            .get(&key)
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let channels = (heads * head_dim) as i32;
        if mlxcel_core::array_shape(value) != [kernel as i32, channels] {
            return Err(format!(
                "{key} must be [{kernel}, {channels}], got {:?}",
                mlxcel_core::array_shape(value)
            ));
        }
        let static_weight = mlxcel_core::reshape(
            &mlxcel_core::transpose(&f32(value)),
            &[heads as i32, head_dim as i32, kernel as i32],
        );
        Ok(Self {
            static_weight: mlxcel_core::contiguous(&static_weight, false),
            weight_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.weight_proj"),
                quant.0,
                quant.1,
            )?,
            heads: heads as i32,
            kernel: kernel as i32,
        })
    }

    pub fn kernel(&self) -> i32 {
        self.kernel
    }

    pub fn forward(
        &self,
        x: &MlxArray,
        dynamic_input: &MlxArray,
        state: &mut Option<UniquePtr<MlxArray>>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, length, channels) = (shape[0], shape[1], shape[2]);
        let k = self.kernel;
        let head_dim = channels / self.heads;
        let xf = f32(x);
        let (context, next) = context_and_next(state, &xf, k);
        *state = Some(next);
        let dynamic = f32(&mlxcel_core::row_invariant::per_row(dynamic_input, |x| {
            self.weight_proj.forward(x)
        }));
        let dynamic = mlxcel_core::reshape(&dynamic, &[batch, length, self.heads, 1, k]);
        let weights = mlxcel_core::add(&self.static_weight, &dynamic);
        let mut out: Option<UniquePtr<MlxArray>> = None;
        for lag in 0..k {
            let start = k - 1 - lag;
            let values = mlxcel_core::utils::slice_axis(&context, 1, start, start + length);
            let values = mlxcel_core::reshape(&values, &[batch, length, self.heads, head_dim]);
            let weight = mlxcel_core::squeeze_axis(
                &mlxcel_core::utils::slice_axis(&weights, 4, lag, lag + 1),
                4,
            );
            let term = mlxcel_core::multiply(&values, &weight);
            out = Some(match out {
                Some(sum) => mlxcel_core::add(&sum, &term),
                None => term,
            });
        }
        let out = mlxcel_core::reshape(
            &out.expect("kernel size is positive"),
            &[batch, length, channels],
        );
        let out =
            mlxcel_core::divide_scalar(&mlxcel_core::add(&out, &xf), ((2 * k + 1) as f32).sqrt());
        mlxcel_core::astype(&out, mlxcel_core::array_dtype(x))
    }
}
