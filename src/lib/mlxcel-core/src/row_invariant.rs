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

//! Row-invariant dense products for speculative verification (issue #3280).
//!
//! A K-row verify block (DSpark, DFlash, prompt lookup) must produce the
//! logits of K single-token decode steps bit for bit, or the model's
//! block-versus-chain exactness probe declines the drafter. `matmul` cannot
//! give that on Metal: MLX runs one row through gemv and several through
//! gemm, and the two sum in different orders. MLX also folds a batch axis
//! into M when the weight is shared, so reshaping rows into a batch does not
//! help.
//!
//! A model opts in by entering a [`Scope`] around every forward (decode steps
//! and verify blocks alike) and building its dense products with [`dense`]
//! and [`linear`]. Outside a scope both are the plain `matmul` and
//! `UnifiedLinear::forward`, so other models are untouched. Inside one:
//!
//! - With [`Mode::kernel`] on Metal, up to 16 rows take [`matvec`], whose
//!   summation order ignores the row count. Decode and verify both run it,
//!   so they agree, and a block reads each weight once.
//! - Otherwise, inside a verify block ([`Mode::verify_block`]), each row is
//!   multiplied alone (the decode step's own gemv) and the rows are
//!   concatenated: exact, but the weight is read once per row. Quantized
//!   projections always take this path, through [`per_row`].
//!
//! [`per_position`] splits any other product whose kernel MLX picks by the
//! block length (a batched matmul over the positions, for example).
//!
//! Attention is not covered: a model must keep its keys, values and masks
//! laid out so a block's row r sees what decode step r sees (for example, a
//! per-query sink column placed before the keys rather than after the
//! newest one).

use crate::layers::UnifiedLinear;
use crate::{MlxArray, UniquePtr, dtype, ffi};

/// The widest product [`matvec`] takes, and the widest verify block this
/// module keeps row-invariant.
pub const MAX_ROWS: i32 = 16;

/// Whether [`matvec`] can run here (Metal default device).
pub fn available() -> bool {
    ffi::row_invariant_matvec_available()
}

/// `x [M, K] @ w [N, K]^T` as `[M, N]` in `output_dtype`, with each output's
/// f32 sum ordered by K and N alone, so row `m` of an `M`-row product equals
/// the one-row product of row `m` bit for bit. `1 <= M <= 16`, `K % 4 == 0`;
/// x and w are bf16, f16 or f32. `None` off Metal or outside the contract.
pub fn matvec(x: &MlxArray, w: &MlxArray, output_dtype: i32) -> Option<UniquePtr<MlxArray>> {
    ffi::row_invariant_matvec(x, w, output_dtype).ok()
}

/// How the products built inside a [`Scope`] are formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mode {
    /// Route dense products of up to [`MAX_ROWS`] rows through [`matvec`]
    /// when it is [`available`]. Must be the same for the decode step and
    /// the verify block it is compared against.
    pub kernel: bool,
    /// This forward is a verify block (see [`is_verify_block`]): products
    /// the kernel does not take are formed one row at a time.
    pub verify_block: bool,
}

thread_local! {
    static MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

fn mode() -> Option<Mode> {
    MODE.with(std::cell::Cell::get)
}

/// Sets the [`Mode`] of the products built while it lives; restores the
/// previous one on drop, so scopes nest.
#[must_use = "the scope ends when the guard is dropped"]
pub struct Scope {
    previous: Option<Mode>,
}

impl Scope {
    pub fn enter(mode: Mode) -> Self {
        Self {
            previous: MODE.with(|cell| cell.replace(Some(mode))),
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        MODE.with(|cell| cell.set(self.previous));
    }
}

/// Whether a forward over `[batch, length, ..]` is a single-sequence verify
/// block: inside the speculative append scope, with 2 to [`MAX_ROWS`]
/// positions. A batched decode step (`length == 1`) is not one.
pub fn is_verify_block(shape: &[i32]) -> bool {
    crate::cache::speculative_append_active()
        && shape.len() >= 2
        && shape[0] == 1
        && (2..=MAX_ROWS).contains(&shape[1])
}

/// Whether a [`Scope`] is open on this thread, for a model whose fused
/// kernels (a compiled MLP, a projection fused with its rotation) would take
/// products the helpers here do not cover: inside a scope it builds the
/// composed form from [`linear_whole_block`] instead, in decode steps and
/// verify blocks alike.
///
/// Used by: `models::llama3::Attention::forward` and `models::llama3::MLP`.
pub fn active() -> bool {
    mode().is_some()
}

/// Whether the products built now belong to a verify block, for callers
/// that split a product the helpers here do not cover.
pub fn in_verify_block() -> bool {
    mode().is_some_and(|m| m.verify_block)
}

/// Inside a verify block, `f` of every leading row of `x [.., D]` alone,
/// concatenated; elsewhere `f(x)`. For products [`dense`] cannot take, such
/// as a quantized projection.
pub fn per_row(x: &MlxArray, f: impl Fn(&MlxArray) -> UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let shape = crate::array_shape(x);
    let rows: i32 = shape[..shape.len() - 1].iter().product();
    if !in_verify_block() || !(2..=MAX_ROWS).contains(&rows) {
        return f(x);
    }
    let width = shape[shape.len() - 1];
    let flat = crate::reshape(x, &[rows, width]);
    let parts: Vec<_> = (0..rows)
        .map(|r| f(&crate::utils::slice_axis(&flat, 0, r, r + 1)))
        .collect();
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    let y = crate::concatenate_many(&refs, 0);
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(*crate::array_shape(&y).last().expect("rank >= 1"));
    crate::reshape(&y, &out_shape)
}

/// Inside a verify block, `f` of every position of axis 1 of `x [1, L, ..]`
/// alone, concatenated on axis 1; elsewhere `f(x)`.
pub fn per_position(
    x: &MlxArray,
    f: impl Fn(&MlxArray) -> UniquePtr<MlxArray>,
) -> UniquePtr<MlxArray> {
    let length = crate::array_shape(x)[1];
    if !in_verify_block() || !(2..=MAX_ROWS).contains(&length) {
        return f(x);
    }
    let parts: Vec<_> = (0..length)
        .map(|i| f(&crate::utils::slice_axis(x, 1, i, i + 1)))
        .collect();
    let refs: Vec<&MlxArray> = parts.iter().map(|p| &**p).collect();
    crate::concatenate_many(&refs, 1)
}

/// `x [.., K] @ w [N, K]^T`, in f32 when either operand is f32 (as `matmul`
/// promotes); row-invariant inside a [`Scope`] (see the module docs).
pub fn dense(x: &MlxArray, w: &MlxArray) -> UniquePtr<MlxArray> {
    let matmul = |x: &MlxArray| crate::matmul(x, &crate::transpose(w));
    let Some(mode) = mode() else {
        return matmul(x);
    };
    let shape = crate::array_shape(x);
    let rows: i32 = shape[..shape.len() - 1].iter().product();
    if mode.kernel && (1..=MAX_ROWS).contains(&rows) && available() {
        let (xd, wd) = (crate::array_dtype(x), crate::array_dtype(w));
        let out_dtype = if xd == dtype::FLOAT32 || wd == dtype::FLOAT32 {
            dtype::FLOAT32
        } else {
            xd
        };
        let flat = crate::reshape(x, &[rows, shape[shape.len() - 1]]);
        if let Some(y) = matvec(&flat, w, out_dtype) {
            let mut out_shape = shape[..shape.len() - 1].to_vec();
            out_shape.push(crate::array_shape(w)[0]);
            return crate::reshape(&y, &out_shape);
        }
    }
    per_row(x, matmul)
}

/// A projection through [`dense`] when it is a plain bias-free dense layer
/// without an active runtime LoRA; otherwise its own forward through
/// [`per_row`].
pub fn linear(layer: &UnifiedLinear, x: &MlxArray) -> UniquePtr<MlxArray> {
    match layer {
        UnifiedLinear::Regular(linear)
            if linear.bias.is_none() && !layer.has_active_runtime_lora() =>
        {
            dense(x, &linear.weight)
        }
        _ => per_row(x, |x| layer.forward(x)),
    }
}

/// [`linear`] for a model that leaves a quantized projection as one product
/// over the whole block: inside a [`Scope`], a plain bias-free dense layer
/// without an active runtime LoRA goes through [`dense`], anything else
/// through its own forward. MLX's quantized product is already row-invariant
/// unless `qmv_wide` is on (`set_qmv_wide`), and splitting it would read the
/// weight once per row, which a verify round without an exactness gate
/// should not pay (issue #3418). Outside a scope this is the layer's own
/// forward, call for call, so a model that enters the scope only on request
/// is otherwise untouched.
///
/// Used by: the DeepSeek V3 decoder (`models::deepseek_v3`),
/// `models::xing4_0`, DeepSeek V3.2 and V2, Arcee, OLMoE, and
/// [`crate::layers::FusedQKVLinear::forward`] (the Llama-class attention).
pub fn linear_whole_block(layer: &UnifiedLinear, x: &MlxArray) -> UniquePtr<MlxArray> {
    if mode().is_none() {
        return layer.forward(x);
    }
    match layer {
        UnifiedLinear::Regular(linear)
            if linear.bias.is_none() && !layer.has_active_runtime_lora() =>
        {
            dense(x, &linear.weight)
        }
        _ => layer.forward(x),
    }
}

#[cfg(test)]
#[path = "row_invariant_tests.rs"]
mod tests;
