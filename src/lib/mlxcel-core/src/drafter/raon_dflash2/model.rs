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

//! [`RaonDFlash2DraftModel`]: the assembled DFlash2 drafter.
//!
//! A round first appends the target features of the newly committed
//! positions to every layer's context window ([`Self::add_context`]), then
//! runs the block `[bonus, mask x (n - 1)]` at the positions that follow
//! them ([`Self::propose`]):
//!
//! ```text
//! hn      = hidden_norm(fc(feats))                       context, [T, hidden]
//! x       = embed_tokens(block)                          target's table
//! x, r    = layer_i(x, r) for i in layers
//! h       = norm(x + r)[1:]
//! logits  = lm_head(h)                                   target's head
//! draft   = chain(selector(h, logits, bonus))
//! ```

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::ContextWindow;
use super::config::RaonDFlash2Config;
use super::layer::DFlash2Layer;
use super::selector::{CandidateSelector, Selection, chain_device};

pub struct RaonDFlash2DraftModel {
    pub config: RaonDFlash2Config,
    /// Target embedding table, installed by `bind`.
    pub embed_tokens: Option<UnifiedEmbedding>,
    /// Target output head, installed by `bind`; `None` uses the tied
    /// embedding projection.
    pub lm_head: Option<UnifiedLinear>,
    fc: UnifiedLinear,
    hidden_norm: RMSNorm,
    layers: Vec<DFlash2Layer>,
    norm: RMSNorm,
    selector: CandidateSelector,
    compute_dtype: i32,
}

/// Per-generation drafter state: one context window per layer and the
/// absolute position of the next context row.
pub struct DFlash2State {
    pub windows: Vec<ContextWindow>,
    /// Absolute target position of the next context row, which is also the
    /// position of the next block's bonus row.
    pub ctx_pos: i32,
}

impl RaonDFlash2DraftModel {
    /// Build from a sanitized weight map.
    ///
    /// # Errors
    /// Returns a message naming a missing tensor.
    pub fn from_weights(weights: &WeightMap, config: RaonDFlash2Config) -> Result<Self, String> {
        let norm = |key: &str| -> Result<RMSNorm, String> {
            let w = weights
                .get(key)
                .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))?;
            Ok(RMSNorm::new(ffi::copy(w), config.rms_norm_eps))
        };
        let compute_dtype = weights
            .get("norm.weight")
            .map(|w| ffi::array_dtype(w))
            .ok_or("Raon DFlash2: missing weight norm.weight")?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| DFlash2Layer::from_weights(weights, &format!("layers.{i}"), &config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            fc: UnifiedLinear::from_weights(weights, "fc", 64, 4)?,
            hidden_norm: norm("hidden_norm.weight")?,
            norm: norm("norm.weight")?,
            selector: CandidateSelector::from_weights(weights, config.selector_top_k)?,
            layers,
            embed_tokens: None,
            lm_head: None,
            compute_dtype,
            config,
        })
    }

    /// Fresh state for a new generation.
    pub fn make_state(&self) -> DFlash2State {
        DFlash2State {
            windows: (0..self.layers.len())
                .map(|_| ContextWindow::new(self.config.window as i32))
                .collect(),
            ctx_pos: 0,
        }
    }

    fn cast(&self, x: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        if ffi::array_dtype(&x) == self.compute_dtype {
            x
        } else {
            ffi::astype(&x, self.compute_dtype)
        }
    }

    /// Append target features `feats` (`[1, T, target_feature_size]` or
    /// `[T, target_feature_size]`) for the next `T` positions. Positions
    /// older than the window are skipped before `fc` runs, and a skip
    /// restarts every window, as the vendor `add_context` does.
    ///
    /// # Errors
    /// Returns a message when `feats` is not `target_feature_size` wide (a
    /// target that captured the wrong layers or streams), before any MLX op
    /// that would throw across the bridge.
    pub fn add_context(&self, feats: &MlxArray, state: &mut DFlash2State) -> Result<(), String> {
        let shape = ffi::array_shape(feats);
        let Some(&width) = shape.last() else {
            return Err("Raon DFlash2: target features are a scalar, not [.., T, features]".into());
        };
        if usize::try_from(width).ok() != Some(self.config.target_feature_size) {
            return Err(format!(
                "Raon DFlash2: target features are {width} wide, the drafter reads {}",
                self.config.target_feature_size
            ));
        }
        let t: i32 = shape[..shape.len() - 1].iter().product();
        if t == 0 {
            return Ok(());
        }
        let rows = ffi::reshape(feats, &[t, width]);
        let window = self.config.window as i32;
        let (rows, t) = if t > window {
            let skip = t - window;
            state.ctx_pos += skip;
            state.windows.iter_mut().for_each(ContextWindow::clear);
            (ffi::slice(&rows, &[skip, 0], &[t, width]), window)
        } else {
            (rows, t)
        };
        let rows = self.cast(rows);
        let hn = self.hidden_norm.forward(&self.fc.forward(&rows));
        for (layer, w) in self.layers.iter().zip(state.windows.iter_mut()) {
            let (k, v) = layer.attn.context_kv(&hn, state.ctx_pos);
            w.append(k, v);
        }
        state.ctx_pos += t;
        Ok(())
    }

    /// Advance past `positions` context rows the drafter will never see
    /// (the target captured only the newest window of a long prompt), as a
    /// longer [`Self::add_context`] would have skipped them: the next row
    /// lands at its absolute position and the windows restart.
    pub fn skip_context(&self, positions: i32, state: &mut DFlash2State) {
        if positions <= 0 {
            return;
        }
        state.ctx_pos = state.ctx_pos.saturating_add(positions);
        state.windows.iter_mut().for_each(ContextWindow::clear);
    }

    /// Hidden rows `[n - 1, hidden]` of the block `[bonus, mask x (n - 1)]`
    /// at positions `ctx_pos..ctx_pos + n` (the bonus row dropped).
    pub fn block_hidden(
        &self,
        bonus: &MlxArray,
        n: usize,
        state: &DFlash2State,
    ) -> UniquePtr<MlxArray> {
        let embed = self
            .embed_tokens
            .as_ref()
            .expect("Raon DFlash2 drafter is not bound to a target embedding table");
        let masks = ffi::from_slice_i32(&vec![self.config.mask_token_id; n - 1], &[n as i32 - 1]);
        let ids = crate::ops::concatenate(
            &ffi::astype(&ffi::reshape(bonus, &[1]), dtype::INT32),
            &masks,
            0,
        );
        let x = self.cast(embed.forward(&ffi::reshape(&ids, &[1, n as i32])));
        let mut x = ffi::reshape(&x, &[n as i32, self.config.hidden_size as i32]);
        let mut residual: Option<UniquePtr<MlxArray>> = None;
        for (layer, w) in self.layers.iter().zip(&state.windows) {
            let (h, r) = layer.forward(&x, residual.as_deref(), state.ctx_pos, n as i32, w);
            x = h;
            residual = Some(r);
        }
        let x = match residual {
            Some(r) => ffi::add(&x, &r),
            None => x,
        };
        let h = self.norm.forward(&x);
        ffi::slice(&h, &[1, 0], &[n as i32, self.config.hidden_size as i32])
    }

    /// Float32 logits of hidden rows `h` through the target head.
    pub fn logits(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let logits = match (&self.lm_head, &self.embed_tokens) {
            (Some(head), _) => head.forward(h),
            (None, Some(embed)) => embed.as_linear(h),
            (None, None) => panic!("Raon DFlash2 drafter is not bound to a target head"),
        };
        ffi::astype(&logits, dtype::FLOAT32)
    }

    /// Selector scores and candidates of the block after `bonus`.
    pub fn selection(&self, bonus: &MlxArray, n: usize, state: &DFlash2State) -> Selection {
        let h = self.block_hidden(bonus, n, state);
        let logits = self.logits(&h);
        self.selector.select(&h, &logits, bonus)
    }

    /// The `n - 1` draft tokens after `bonus`, as a lazy `[1, n - 1]` int32
    /// array.
    pub fn propose(&self, bonus: &MlxArray, n: usize, state: &DFlash2State) -> UniquePtr<MlxArray> {
        let chain = chain_device(&self.selection(bonus, n, state));
        ffi::reshape(&chain, &[1, n as i32 - 1])
    }
}
