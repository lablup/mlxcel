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

//! Bidirectional LSTM, hand-rolled over the MLX FFI.
//!
//! MLX exposes no recurrent primitive, so the gates are computed step by step
//! (see [`super::ops::lstm_dir`]). Kokoro uses single-layer bi-LSTMs throughout
//! the predictor and text encoders; the PyTorch parameter names are
//! `weight_ih_l0` / `weight_hh_l0` (+ `_reverse`) and matching biases. The
//! forward and reverse hidden sequences are concatenated on the feature axis.

use mlxcel_core::{MlxArray, UniquePtr};

use super::ops;
use super::weights::Weights;

/// A single-layer bidirectional LSTM.
pub(crate) struct BiLstm {
    fwd_ih: UniquePtr<MlxArray>,
    fwd_hh: UniquePtr<MlxArray>,
    fwd_bih: UniquePtr<MlxArray>,
    fwd_bhh: UniquePtr<MlxArray>,
    rev_ih: UniquePtr<MlxArray>,
    rev_hh: UniquePtr<MlxArray>,
    rev_bih: UniquePtr<MlxArray>,
    rev_bhh: UniquePtr<MlxArray>,
    hidden: usize,
}

impl BiLstm {
    /// Load from `prefix`; `hidden` is the per-direction hidden size.
    pub(crate) fn load(w: &Weights, prefix: &str, hidden: i32) -> Result<Self, String> {
        Ok(Self {
            fwd_ih: w.get(&format!("{prefix}.weight_ih_l0"))?,
            fwd_hh: w.get(&format!("{prefix}.weight_hh_l0"))?,
            fwd_bih: w.get(&format!("{prefix}.bias_ih_l0"))?,
            fwd_bhh: w.get(&format!("{prefix}.bias_hh_l0"))?,
            rev_ih: w.get(&format!("{prefix}.weight_ih_l0_reverse"))?,
            rev_hh: w.get(&format!("{prefix}.weight_hh_l0_reverse"))?,
            rev_bih: w.get(&format!("{prefix}.bias_ih_l0_reverse"))?,
            rev_bhh: w.get(&format!("{prefix}.bias_hh_l0_reverse"))?,
            hidden: hidden as usize,
        })
    }

    /// Run over a `(T, in)` sequence, returning `(T, 2*hidden)`.
    pub(crate) fn forward(&self, x: &UniquePtr<MlxArray>, t: usize) -> UniquePtr<MlxArray> {
        let fwd = ops::lstm_dir(
            x,
            t,
            self.hidden,
            &self.fwd_ih,
            &self.fwd_hh,
            &self.fwd_bih,
            &self.fwd_bhh,
            false,
        );
        let rev = ops::lstm_dir(
            x,
            t,
            self.hidden,
            &self.rev_ih,
            &self.rev_hh,
            &self.rev_bih,
            &self.rev_bhh,
            true,
        );
        // Stack each direction's per-step (1, H) rows into (T, H), then concat.
        let fwd_seq = stack_rows(&fwd);
        let rev_seq = stack_rows(&rev);
        ops::concat2(&fwd_seq, &rev_seq, 1)
    }
}

/// Concatenate a list of `(1, H)` step outputs into a `(T, H)` matrix.
fn stack_rows(rows: &[UniquePtr<MlxArray>]) -> UniquePtr<MlxArray> {
    let refs: Vec<&UniquePtr<MlxArray>> = rows.iter().collect();
    ops::concat(&refs, 0)
}
