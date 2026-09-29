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

//! Unidirectional stacked LSTM evaluated one step at a time.
//!
//! Mirrors `mlx.nn.LSTM` as stacked by the `LSTM` wrapper in
//! `mlx_audio/stt/models/nemotron_asr/rnnt.py`, for the batch-1, length-1
//! calls the RNNT prediction network makes. Gate order is `i, f, g, o`. When
//! no previous state exists the step skips the recurrent term entirely and the
//! cell starts at `i * g`, exactly as `nn.LSTM(hidden=None, cell=None)` does,
//! so dtype promotion matches the reference.
//!
//! Weights are accepted either in the PyTorch layout
//! (`weight_ih_l{n}`, `weight_hh_l{n}`, `bias_ih_l{n}`, `bias_hh_l{n}`, the two
//! biases summed in the checkpoint dtype like mlx-vlm's sanitize) or in the
//! already-converted MLX layout (`{n}.Wx`, `{n}.Wh`, `{n}.bias`).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-layer `(hidden, cell)`, each `[1, H]`.
pub type LstmState = Vec<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>;

struct LstmLayer {
    /// `Wx^T`: `[in, 4H]`.
    wx_t: UniquePtr<MlxArray>,
    /// `Wh^T`: `[H, 4H]`.
    wh_t: UniquePtr<MlxArray>,
    /// `[4H]`.
    bias: UniquePtr<MlxArray>,
}

pub struct StackedLstm {
    layers: Vec<LstmLayer>,
    hidden: i32,
}

fn get(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|w| mlxcel_core::copy(w))
}

fn require(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    get(weights, key).ok_or_else(|| format!("RNNT LSTM weight not found: {key}"))
}

impl StackedLstm {
    /// Load `num_layers` layers from `prefix` (e.g.
    /// `stt_model.rnnt_decoder.prediction.dec_rnn.lstm`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        input_size: usize,
        hidden: usize,
        num_layers: usize,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(num_layers);
        for n in 0..num_layers {
            let (wx, wh, bias) = match get(weights, &format!("{prefix}.{n}.Wx")) {
                Some(wx) => (
                    wx,
                    require(weights, &format!("{prefix}.{n}.Wh"))?,
                    require(weights, &format!("{prefix}.{n}.bias"))?,
                ),
                None => {
                    let b_ih = require(weights, &format!("{prefix}.bias_ih_l{n}"))?;
                    let b_hh = require(weights, &format!("{prefix}.bias_hh_l{n}"))?;
                    (
                        require(weights, &format!("{prefix}.weight_ih_l{n}"))?,
                        require(weights, &format!("{prefix}.weight_hh_l{n}"))?,
                        mlxcel_core::add(&b_ih, &b_hh),
                    )
                }
            };
            let in_size = if n == 0 { input_size } else { hidden } as i32;
            let gates = 4 * hidden as i32;
            let check = |name: &str, arr: &MlxArray, want: &[i32]| {
                let shape = mlxcel_core::array_shape(arr);
                if shape != want {
                    return Err(format!(
                        "{prefix} layer {n} {name} has shape {shape:?}, expected {want:?}"
                    ));
                }
                Ok(())
            };
            check("Wx", &wx, &[gates, in_size])?;
            check("Wh", &wh, &[gates, hidden as i32])?;
            check("bias", &bias, &[gates])?;
            layers.push(LstmLayer {
                wx_t: mlxcel_core::transpose_axes(&wx, &[1, 0]),
                wh_t: mlxcel_core::transpose_axes(&wh, &[1, 0]),
                bias,
            });
        }
        Ok(Self {
            layers,
            hidden: hidden as i32,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// One time step. `x: [1, in]`; returns the top layer output `[1, H]` and
    /// the new per-layer state (in whatever dtype the promotion produced).
    pub fn step(
        &self,
        x: &MlxArray,
        state: Option<&LstmState>,
    ) -> (UniquePtr<MlxArray>, LstmState) {
        let h = self.hidden;
        let mut next: LstmState = Vec::with_capacity(self.layers.len());
        let mut input = mlxcel_core::copy(x);
        for (idx, layer) in self.layers.iter().enumerate() {
            let prev = state.and_then(|s| s.get(idx));
            let mut ifgo = mlxcel_core::addmm(&layer.bias, &input, &layer.wx_t, 1.0, 1.0);
            if let Some((h_prev, _)) = prev {
                ifgo = mlxcel_core::addmm(&ifgo, h_prev, &layer.wh_t, 1.0, 1.0);
            }
            let gate = |k: i32| mlxcel_core::slice(&ifgo, &[0, k * h], &[1, (k + 1) * h]);
            let i = mlxcel_core::sigmoid(&gate(0));
            let f = mlxcel_core::sigmoid(&gate(1));
            let g = mlxcel_core::tanh(&gate(2));
            let o = mlxcel_core::sigmoid(&gate(3));
            let ig = mlxcel_core::multiply(&i, &g);
            let cell = match prev {
                Some((_, c_prev)) => mlxcel_core::add(&mlxcel_core::multiply(&f, c_prev), &ig),
                None => ig,
            };
            let hidden = mlxcel_core::multiply(&o, &mlxcel_core::tanh(&cell));
            input = mlxcel_core::copy(&hidden);
            next.push((hidden, cell));
        }
        (input, next)
    }
}
