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

//! Half-precision reductions that match MLX's native `mx.sum`.
//!
//! Shared by the Nemotron VoiceChat codec and EAR-TTS ports, whose residual
//! vector quantizers pick codes by `argmin` over bf16 codebook norms: a
//! reduction that rounds differently from the reference flips codes.
//!
//! Used by: `audio::nemotron_codec`, `models::nemotron_voicechat::tts`

use mlxcel_core::{MlxArray, UniquePtr};

/// `mx.sum(x, axis=axis)` with MLX's native reduction for every dtype.
///
/// [`mlxcel_core::sum_axis`] widens half-precision inputs to `f32`, reduces
/// and rounds once, which differs from the native bf16 `mx.sum` the
/// reference runs (the codec port measured about 20% of bf16 codebook norms
/// differing, enough to flip RVQ `argmin` codes). A pure-reduction `einsum`
/// lowers to the native `sum`. Rank is limited to 8.
pub(crate) fn native_sum_axis(x: &MlxArray, axis: usize) -> UniquePtr<MlxArray> {
    const LETTERS: &str = "abcdefgh";
    let rank = mlxcel_core::array_shape(x).len().min(LETTERS.len());
    let input = &LETTERS[..rank];
    let output: String = input
        .chars()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, c)| c)
        .collect();
    let spec = format!("{input}->{output}");
    let operands = [x as *const MlxArray];
    // SAFETY: the operand pointer borrows `x`, which outlives the call.
    unsafe { mlxcel_core::einsum(&spec, &operands) }
}
