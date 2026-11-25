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

//! Per-layer block-versus-chain diagnostic (issue #3409).
//!
//! The exactness probe says whether a K-row verify block's logits equal K
//! single-token steps. This says where they first part: a model records the
//! tensors of interest as [`Stage`]s, once for the block and once per chain
//! step, and [`locate`] names the first stage and row that differ.
//!
//! Stages come from either source:
//!
//! - [`capture`] calls placed in a model's forward, collected by
//!   [`with_capture`]. Outside a capture scope a call costs one thread-local
//!   read, so a model can keep them in its serving path. Gemma 4's startup
//!   probe reruns a failed draw this way and puts the location in the decline
//!   reason; the thread-local scope ends before the probe returns, so serving
//!   never retains diagnostic tensors.
//! - [`Stage::of`] over tensors a forward already hands back, such as the
//!   per-layer hidden states of a DFlash target's `forward_speculative`.
//!
//! Generalized from the Gemma 4 probe trace (#1983) and the Inkling 2
//! per-visit comparison (#3280).
use std::cell::RefCell;

use mlxcel_core::{MlxArray, utils::slice_axis};

/// One recorded tensor: the raw bytes of every row of axis 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub layer: usize,
    pub kind: String,
    pub op: &'static str,
    pub rows: Vec<Vec<u8>>,
}

impl Stage {
    /// The stage of `tensor [1, T, ..]`: one entry of `rows` per position.
    pub fn of(layer: usize, kind: &str, op: &'static str, tensor: &MlxArray) -> Self {
        let width = mlxcel_core::array_shape(tensor)[1];
        let rows = (0..width)
            .map(|row| mlxcel_core::array_to_raw_bytes(&slice_axis(tensor, 1, row, row + 1)))
            .collect();
        Self {
            layer,
            kind: kind.to_owned(),
            op,
            rows,
        }
    }
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<Stage>>> = const { RefCell::new(None) };
}

/// Whether a [`with_capture`] scope is open on this thread, for a caller
/// that must build the tensor it would record.
pub fn capturing() -> bool {
    CAPTURE.with(|slot| slot.borrow().is_some())
}

/// Record `tensor [1, T, ..]` when a [`with_capture`] scope is open; a no-op
/// otherwise.
///
/// Used by: the Gemma 4 decoder layers (including the MoE feed-forward
/// stages) and startup probe, the Inkling layer loop and startup probe, and
/// the DeepSeek V3, Xing4.0, Arcee, OLMoE and Llama 3 (K2 Horizon) layer
/// loops (survey diagnostic).
pub fn capture(layer: usize, kind: &str, op: &'static str, tensor: &MlxArray) {
    CAPTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(stages) = slot.as_mut() else { return };
        stages.push(Stage::of(layer, kind, op, tensor));
    });
}

/// Run `run` with capture on and return what it recorded.
pub fn with_capture<T>(run: impl FnOnce() -> T) -> (T, Vec<Stage>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CAPTURE.with(|slot| *slot.borrow_mut() = None);
        }
    }
    CAPTURE.with(|slot| {
        assert!(slot.borrow().is_none(), "probe capture must not nest");
        *slot.borrow_mut() = Some(Vec::new());
    });
    let _reset = Reset;
    let result = run();
    let stages = CAPTURE.with(|slot| slot.borrow_mut().take().unwrap_or_default());
    (result, stages)
}

/// Where a block first parts from the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// Index into the block's stage list.
    pub stage: usize,
    pub layer: usize,
    pub kind: String,
    pub op: &'static str,
    /// Block row, which is also the chain step.
    pub position: usize,
    pub differing_bytes: usize,
    pub total_bytes: usize,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "first divergence: layer {} ({}) {} at position {}",
            self.layer, self.kind, self.op, self.position
        )
    }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// The first stage, then the first row within it, at which `block` differs
/// from `chain` (one stage list per single-token step, each stage holding one
/// row). Scanned in execution order, then token order: a later token at an
/// earlier layer is more useful than an earlier token whose error has
/// propagated. `Ok(None)` when every stage agrees; `Err` when the two traces
/// do not line up, so a malformed trace can never read as equality.
pub fn locate(block: &[Stage], chain: &[Vec<Stage>]) -> Result<Option<Divergence>, &'static str> {
    if block.is_empty() || chain.is_empty() || chain.iter().any(|token| token.len() != block.len())
    {
        return Err("probe trace stage counts differ or are empty");
    }
    for (index, stage) in block.iter().enumerate() {
        if stage.rows.len() != chain.len() || chain.iter().any(|token| token[index].rows.len() != 1)
        {
            return Err("probe trace row counts differ");
        }
        for (position, token) in chain.iter().enumerate() {
            let reference = &token[index];
            if (stage.layer, &stage.kind, stage.op)
                != (reference.layer, &reference.kind, reference.op)
            {
                return Err("probe trace stage order differs");
            }
            let differing_bytes = differing(&stage.rows[position], &reference.rows[0]);
            if differing_bytes > 0 {
                return Ok(Some(Divergence {
                    stage: index,
                    layer: stage.layer,
                    kind: stage.kind.clone(),
                    op: stage.op,
                    position,
                    differing_bytes,
                    total_bytes: stage.rows[position].len(),
                }));
            }
        }
    }
    Ok(None)
}

/// [`locate`] as the one line a decline reason carries; `None` when the
/// traces agree.
pub fn first_divergence(block: &[Stage], chain: &[Vec<Stage>]) -> Option<String> {
    match locate(block, chain) {
        Ok(None) => None,
        Ok(Some(divergence)) => Some(divergence.to_string()),
        Err(why) => Some(why.to_owned()),
    }
}

/// One line per stage with the differing byte count of every row
/// (`r<row> <differing>/<total>`), for a diagnostic that wants the whole
/// picture and not only the first stage. Empty when the traces do not line
/// up; [`locate`] says why.
pub fn report(block: &[Stage], chain: &[Vec<Stage>]) -> Vec<String> {
    if locate(block, chain).is_err() {
        return Vec::new();
    }
    block
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let mut line = format!("layer {:>3} ({}) {}:", stage.layer, stage.kind, stage.op);
            for (position, token) in chain.iter().enumerate() {
                let reference = &token[index].rows[0];
                let n = differing(&stage.rows[position], reference);
                line.push_str(&format!(" r{position} {n}/{}", reference.len()));
            }
            line
        })
        .collect()
}

#[cfg(test)]
#[path = "speculative_exactness_diag_tests.rs"]
mod tests;
