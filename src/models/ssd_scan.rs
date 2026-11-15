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

//! Float32-hardened chunked SSD (Mamba2) scan for the hybrid families.
//!
//! The SSD surrogate attention is quadratic in the length it is built over:
//! the segsum decay and the `CB` product are `[batch, heads, L, L]` in float32,
//! with several live copies per layer. Built over a whole prefill piece, one
//! layer of a 28K single-pass prefill needs more than the GPU budget by itself
//! (#3073). This scan builds it over chunks of a fixed length, independent of
//! the prefill piece, and carries the recurrent state between them, as
//! mlx-lm's `ssm_attn` does
//! (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ssm.py).
//!
//! Used by: FalconH1, GraniteMoeHybrid, Plamo2, NemotronH

use mlxcel_core::utils::{schedule_prefill_state, slice_axis};
use mlxcel_core::{MlxArray, UniquePtr, concatenate_many};

/// SSD chunk length for the families that do not configure one, matching the
/// `step` default of mlx-lm's `ssm_attn`.
pub(crate) const SSD_CHUNK_SIZE: usize = 256;

/// Tokens of scan output allowed to accumulate as one lazy graph before the
/// scan starts evaluating them with [`schedule_prefill_state`]. A prefill
/// piece at or below this length (the default 2048-token chunked prefill)
/// stays one lazy graph, as it was before chunking; a longer single pass
/// releases each window's intermediates before the next one is built.
pub(crate) const SSD_SCHEDULE_TOKENS: usize = 2048;

/// When the scan forces evaluation between chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SsdEval {
    /// Synchronously evaluate every chunk's output and state (NemotronH).
    EveryChunk,
    /// Schedule the running output and state with `async_eval` each time this
    /// many tokens have accumulated, leaving the final window lazy.
    ScheduleEvery(usize),
}

/// Concatenates scan outputs along the sequence axis, draining `parts`.
fn concat_seq(parts: &mut Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    if parts.len() == 1 {
        return parts.pop().expect("one SSD output part");
    }
    let refs: Vec<&MlxArray> = parts
        .iter()
        .map(|part| part.as_ref().expect("SSD output part"))
        .collect();
    let output = concatenate_many(&refs, 1);
    parts.clear();
    output
}

/// Repeat an array along `axis` by `repeats` (broadcast then reshape).
fn repeat_axis(x: &MlxArray, repeats: i32, axis: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { ndim + axis } else { axis };

    let mut new_shape: Vec<i32> = shape.iter().take(axis as usize + 1).copied().collect();
    new_shape.push(1);
    new_shape.extend(shape.iter().skip(axis as usize + 1));
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    new_shape[axis as usize + 1] = repeats;
    let x_broad = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let mut final_shape: Vec<i32> = shape.clone();
    final_shape[axis as usize] *= repeats;
    mlxcel_core::reshape(&x_broad, &final_shape)
}

/// Segmented cumulative sum used by the SSD surrogate-attention decay
/// (inclusive cumsum, matching `mx.cumsum`).
fn segsum(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let l = shape[shape.len() - 1];

    let mut new_shape = shape.clone();
    new_shape.push(1);
    let x_exp = mlxcel_core::reshape(x, &new_shape);

    let last_idx = new_shape.len() - 1;
    new_shape[last_idx] = l;
    let x_rep = mlxcel_core::broadcast_to(&x_exp, &new_shape);

    let x_tril = mlxcel_core::tril(&x_rep, -1);
    mlxcel_core::cumsum(&x_tril, -2, false, true)
}

/// Chunked SSD scan over `x` `[batch, seq, heads, head_dim]`, `b`/`c`
/// `[batch, seq, groups, state_dim]` and raw `dt` `[batch, seq, heads]`.
///
/// `dt = clip(softplus(dt + dt_bias), time_step_limit)`, `x`, `B` and `C` are
/// promoted to float32 before the matmuls: the dot products over the state
/// dimension can overflow float16, and mixed float32 x float16 multiplies NaN
/// on M5 Max (Metal GPU Family 4) NAx kernels. The output is cast back to the
/// dtype of `x`; the returned state is float32 `[batch, heads, head_dim,
/// state_dim]`.
///
/// The working set is bounded by `chunk_size`, not by the sequence length.
/// With `chunk_size >= seq` the graph is the unchunked scan op for op.
///
/// Used by: FalconH1, GraniteMoeHybrid, Plamo2 (`SSD_CHUNK_SIZE`,
/// `ScheduleEvery(SSD_SCHEDULE_TOKENS)`), NemotronH (config `chunk_size`,
/// `EveryChunk`)
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_ssd_scan(
    x: &MlxArray,
    a_log: &MlxArray,
    b: &MlxArray,
    c: &MlxArray,
    d: &MlxArray,
    dt: &MlxArray,
    dt_bias: &MlxArray,
    state: Option<&MlxArray>,
    time_step_limit: (f32, f32),
    chunk_size: usize,
    eval_policy: SsdEval,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    assert!(chunk_size > 0, "chunk_size must be greater than zero");

    let input_dtype = mlxcel_core::array_dtype(x);
    let x_shape = mlxcel_core::array_shape(x);
    let batch = x_shape[0];
    let seq_len = x_shape[1] as usize;
    let num_heads = x_shape[2];
    let head_dim = x_shape[3];

    let b_shape = mlxcel_core::array_shape(b);
    let n_groups = b_shape[2];
    let state_dim = b_shape[3];
    let repeats = num_heads / n_groups;

    let f32_dtype = mlxcel_core::dtype::FLOAT32;
    let dt_f32 = mlxcel_core::astype(dt, f32_dtype);
    let dt_bias_f32 = mlxcel_core::astype(dt_bias, f32_dtype);
    let dt_biased = mlxcel_core::add(&dt_f32, &dt_bias_f32);
    let dt_soft = mlxcel_core::softplus(&dt_biased);
    let min_val = mlxcel_core::full_f32(&[1], time_step_limit.0, f32_dtype);
    let max_val = mlxcel_core::full_f32(&[1], time_step_limit.1, f32_dtype);
    let dt = mlxcel_core::clip(&dt_soft, &min_val, &max_val);

    let a = mlxcel_core::negative(&mlxcel_core::exp(a_log));
    let a = mlxcel_core::astype(&a, f32_dtype);
    let a_reshaped = mlxcel_core::reshape(&a, &[1, 1, num_heads]);
    let dt_a = mlxcel_core::multiply(&dt, &a_reshaped);

    let x_f32 = mlxcel_core::astype(x, f32_dtype);
    let dt_exp = mlxcel_core::reshape(&dt, &[batch, seq_len as i32, num_heads, 1]);
    let dtx = mlxcel_core::multiply(&dt_exp, &x_f32);
    let b_f32 = mlxcel_core::astype(b, f32_dtype);
    let c_f32 = mlxcel_core::astype(c, f32_dtype);
    let d_f32 = mlxcel_core::astype(d, f32_dtype);
    let d_reshaped = mlxcel_core::reshape(&d_f32, &[1, 1, num_heads, 1]);

    let output_capacity = seq_len.div_ceil(chunk_size);
    let mut outputs = Vec::with_capacity(output_capacity);
    let mut current_state = state.map(mlxcel_core::copy);
    let mut window = Vec::new();
    let mut window_tokens = 0usize;

    for chunk_start in (0..seq_len).step_by(chunk_size) {
        let chunk_end = chunk_start.saturating_add(chunk_size).min(seq_len);
        let chunk_len = (chunk_end - chunk_start) as i32;
        let (start, end) = (chunk_start as i32, chunk_end as i32);

        let chunk_x = slice_axis(&x_f32, 1, start, end);
        let chunk_dtx = slice_axis(&dtx, 1, start, end);
        let chunk_dt_a = slice_axis(&dt_a, 1, start, end);
        let chunk_b = slice_axis(&b_f32, 1, start, end);
        let chunk_c = slice_axis(&c_f32, 1, start, end);

        // CB = C.swapaxes(1, 2) @ B.transpose(0, 2, 3, 1), repeated per head.
        let b_t = mlxcel_core::transpose_axes(&chunk_b, &[0, 2, 3, 1]);
        let c_t = mlxcel_core::swap_axes(&chunk_c, 1, 2);
        let cb = mlxcel_core::matmul(&c_t, &b_t);
        let cb = repeat_axis(&cb, repeats, 1);

        // decay = exp(segsum(dtA.swapaxes(1, 2)));  surrogate = tril(CB * decay).
        let dt_a_t = mlxcel_core::swap_axes(&chunk_dt_a, 1, 2);
        let decay = mlxcel_core::exp(&segsum(&dt_a_t));
        let attn_matrix = mlxcel_core::tril(&mlxcel_core::multiply(&cb, &decay), 0);
        let dtx_t = mlxcel_core::swap_axes(&chunk_dtx, 1, 2);
        let mut y_chunk = mlxcel_core::matmul(&attn_matrix, &dtx_t);
        y_chunk = mlxcel_core::swap_axes(&y_chunk, 1, 2);

        // This chunk's contribution to the state.
        let decay_shape = mlxcel_core::array_shape(&decay);
        let decay_last = slice_axis(&decay, 2, decay_shape[2] - 1, decay_shape[2]);
        let decay_for_state = mlxcel_core::transpose_axes(&decay_last, &[0, 3, 1, 2]);
        let b_rep = repeat_axis(&b_t, repeats, 1);
        let b_sw = mlxcel_core::swap_axes(&b_rep, 2, 3);
        let dtx_decay = mlxcel_core::multiply(&chunk_dtx, &decay_for_state);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay, 1, 2);
        let dtx_decay_t = mlxcel_core::swap_axes(&dtx_decay_t, 2, 3);
        let mut next_state = mlxcel_core::matmul(&dtx_decay_t, &b_sw);

        // Carry the previous recurrent state forward.
        if let Some(ref previous_state) = current_state {
            let dta_cumsum = mlxcel_core::cumsum(&chunk_dt_a, -2, false, true);
            let exp_dta_cumsum = mlxcel_core::exp(&dta_cumsum);

            let exp_shape = mlxcel_core::array_shape(&exp_dta_cumsum);
            let last_exp = slice_axis(&exp_dta_cumsum, 1, exp_shape[1] - 1, exp_shape[1]);
            let last_exp = mlxcel_core::reshape(&last_exp, &[batch, num_heads, 1, 1]);
            let state_contrib = mlxcel_core::multiply(&last_exp, previous_state);
            next_state = mlxcel_core::add(&next_state, &state_contrib);

            let c_reshaped =
                mlxcel_core::reshape(&chunk_c, &[batch, chunk_len, n_groups, 1, state_dim, 1]);
            let state_reshaped = mlxcel_core::reshape(
                previous_state,
                &[batch, 1, n_groups, repeats, head_dim, state_dim],
            );
            let y_previous = mlxcel_core::matmul(&state_reshaped, &c_reshaped);
            let y_previous = mlxcel_core::squeeze_axis(&y_previous, -1);
            let y_previous =
                mlxcel_core::reshape(&y_previous, &[batch, chunk_len, num_heads, head_dim]);
            let mut exp_shape_y = mlxcel_core::array_shape(&exp_dta_cumsum);
            exp_shape_y.push(1);
            let exp_dta_exp = mlxcel_core::reshape(&exp_dta_cumsum, &exp_shape_y);
            let y_previous = mlxcel_core::multiply(&exp_dta_exp, &y_previous);
            y_chunk = mlxcel_core::add(&y_chunk, &y_previous);
        }

        // y = y + x * D, float32 throughout to avoid mixed-dtype NaN.
        let d_contrib = mlxcel_core::multiply(&chunk_x, &d_reshaped);
        y_chunk = mlxcel_core::add(&y_chunk, &d_contrib);
        y_chunk = mlxcel_core::astype(&y_chunk, input_dtype);

        // Evaluation between chunks is what releases each chunk's quadratic
        // intermediates before the next one is built; evaluating only the
        // final concatenation would hold every chunk's graph at once. A
        // window's outputs are scheduled together: they do not depend on each
        // other, so scheduling only the last would leave the others holding
        // the decay buffers that evaluating the state computed.
        current_state = Some(next_state);
        window.push(y_chunk);
        match eval_policy {
            SsdEval::EveryChunk => {
                let y_chunk = window.pop().expect("one SSD output chunk");
                mlxcel_core::eval(&y_chunk);
                mlxcel_core::eval(current_state.as_ref().expect("SSD state"));
                outputs.push(y_chunk);
            }
            SsdEval::ScheduleEvery(tokens) => {
                window_tokens += chunk_len as usize;
                if window_tokens >= tokens && chunk_end < seq_len {
                    let window_output = concat_seq(&mut window);
                    let rows = window_tokens as i32;
                    schedule_prefill_state(&window_output, rows);
                    schedule_prefill_state(current_state.as_ref().expect("SSD state"), rows);
                    outputs.push(window_output);
                    window_tokens = 0;
                }
            }
        }
    }
    if !window.is_empty() {
        outputs.push(concat_seq(&mut window));
    }
    let output = concat_seq(&mut outputs);

    (
        output,
        current_state.expect("the SSD scan requires a non-empty sequence"),
    )
}

#[cfg(test)]
#[path = "ssd_scan_tests.rs"]
mod tests;
