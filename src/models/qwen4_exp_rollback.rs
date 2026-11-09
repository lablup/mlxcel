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

//! Speculative rollback state for `qwen4_exp` (issue #2880).
//!
//! A verify block (`Engine::verify` for prompt lookup, the MTP adapter's
//! `verify_forward`) appends `K` positions to every piece of per-sequence
//! state the model owns. The attention KV cache and the QSA index are trimmed
//! by the rejected count; the GatedDeltaNet recurrence, its convolution
//! window, the PLE convolution window and the PLE n-gram context cannot be
//! trimmed, so the forward captures what a replay over the accepted prefix
//! needs ([`SpeculativeCapture`]) and the rollback recomputes them from the
//! state before the block.
//!
//! The capture types live here rather than inside [`super::Qwen4ExpCache`]:
//! a capture describes one forward, a cache describes a sequence, and the
//! prompt-cache snapshots of #2885 add a third kind of state to the cache
//! without touching this module. The per-sequence [`RollbackLog`] keeps the
//! most recent captures so `LanguageModel::rewind_decode_appends` can unwind
//! across several speculative forwards (the engine's verify widths warm-up
//! issues one forward per width before any round).

use std::collections::VecDeque;

use mlxcel_core::{MlxArray, UniquePtr};

/// Most captured speculative forwards kept per sequence. The engine unwinds
/// the appends of its most recent verify forward (one entry), and the
/// scheduler's lookahead never exceeds
/// `DECODE_LOOKAHEAD_MAX_SPECULATIVE_APPENDS` (two single-token forwards);
/// four leaves room for both without holding block-sized tensors for long.
pub(crate) const ROLLBACK_LOG_DEPTH: usize = 4;

/// What one GatedDeltaNet layer needs to replay a verify block over its
/// accepted prefix: the normalized `q`/`k`, `v`, the raw gate inputs `a`/`b`
/// (one row per block position), the recurrent state the block started from
/// and the convolution input (`kernel - 1` carried rows followed by the
/// block's projected rows). Tensors keep the dtype the forward produced.
pub(crate) struct GdnBlockCapture {
    pub(crate) q: UniquePtr<MlxArray>,
    pub(crate) k: UniquePtr<MlxArray>,
    pub(crate) v: UniquePtr<MlxArray>,
    pub(crate) a: UniquePtr<MlxArray>,
    pub(crate) b: UniquePtr<MlxArray>,
    pub(crate) init_state: Option<UniquePtr<MlxArray>>,
    pub(crate) conv_input: UniquePtr<MlxArray>,
}

/// What a PLE layer needs: the dilated-convolution input (the window carried
/// into the block followed by the block's normalized rows), the n-gram
/// contexts each batch row started the block with, and the block's tokens.
pub(crate) struct PleBlockCapture {
    pub(crate) conv_input: UniquePtr<MlxArray>,
    pub(crate) contexts_before: Vec<Vec<i64>>,
    pub(crate) tokens: Vec<Vec<i64>>,
}

pub(crate) enum LayerCapture {
    Linear {
        delta: GdnBlockCapture,
        ple: Option<PleBlockCapture>,
    },
    /// Attention layers roll back by trimming; nothing to capture.
    Attention,
}

/// One speculative forward's rollback state, in layer order.
pub(crate) struct SpeculativeCapture {
    /// Positions the forward appended to every layer.
    pub(crate) appended: i32,
    pub(crate) layers: Vec<LayerCapture>,
}

impl SpeculativeCapture {
    /// The capture of the same forward restricted to its first `keep`
    /// positions, after a rollback kept exactly those: a later rewind can
    /// then unwind into this block again.
    pub(crate) fn truncated(&self, keep: i32) -> Self {
        let dropped = self.appended - keep;
        let layers = self
            .layers
            .iter()
            .map(|layer| match layer {
                LayerCapture::Attention => LayerCapture::Attention,
                LayerCapture::Linear { delta, ple } => LayerCapture::Linear {
                    delta: GdnBlockCapture {
                        q: slice_rows(&delta.q, keep),
                        k: slice_rows(&delta.k, keep),
                        v: slice_rows(&delta.v, keep),
                        a: slice_rows(&delta.a, keep),
                        b: slice_rows(&delta.b, keep),
                        init_state: delta.init_state.as_ref().map(|s| mlxcel_core::copy(s)),
                        conv_input: drop_last_rows(&delta.conv_input, dropped),
                    },
                    ple: ple.as_ref().map(|ple| PleBlockCapture {
                        conv_input: drop_last_rows(&ple.conv_input, dropped),
                        contexts_before: ple.contexts_before.clone(),
                        tokens: ple
                            .tokens
                            .iter()
                            .map(|row| row[..keep as usize].to_vec())
                            .collect(),
                    }),
                },
            })
            .collect();
        Self {
            appended: keep,
            layers,
        }
    }
}

/// The first `rows` entries along axis 1 (the sequence axis).
fn slice_rows(x: &MlxArray, rows: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let mut stop = shape.clone();
    stop[1] = rows;
    let start = vec![0; shape.len()];
    mlxcel_core::contiguous(&mlxcel_core::slice(x, &start, &stop), false)
}

/// Everything along axis 1 except the last `dropped` entries.
fn drop_last_rows(x: &MlxArray, dropped: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    slice_rows(x, shape[1] - dropped)
}

/// Which pieces of state a rollback restores.
///
/// Production always restores all of them ([`RollbackParts::ALL`]). The
/// rollback tests switch one piece off at a time to show that each test
/// fails without that piece, which is the evidence the issue asks for: a
/// rollback that forgets a component produces a plausible wrong state, not
/// a crash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RollbackParts {
    /// Replay the GatedDeltaNet recurrence and convolution window.
    pub(crate) gdn: bool,
    /// Re-advance the PLE convolution window.
    pub(crate) ple_conv: bool,
    /// Recompute the PLE n-gram context from the accepted tokens.
    pub(crate) ple_context: bool,
    /// Drop the QSA block keys that are no longer complete.
    pub(crate) qsa_blocks: bool,
}

impl RollbackParts {
    pub(crate) const ALL: Self = Self {
        gdn: true,
        ple_conv: true,
        ple_context: true,
        qsa_blocks: true,
    };
}

/// The most recent speculative forwards of one sequence, newest last.
#[derive(Default)]
pub(crate) struct RollbackLog {
    entries: VecDeque<SpeculativeCapture>,
}

impl RollbackLog {
    pub(crate) fn push(&mut self, capture: SpeculativeCapture) {
        if self.entries.len() == ROLLBACK_LOG_DEPTH {
            self.entries.pop_front();
        }
        self.entries.push_back(capture);
    }

    /// A forward that is not speculative commits everything before it: the
    /// entries no longer describe the tail of the sequence.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    pub(crate) fn pop(&mut self) -> Option<SpeculativeCapture> {
        self.entries.pop_back()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Positions the logged forwards appended in total, the most a rewind
    /// can unwind.
    pub(crate) fn appended(&self) -> i32 {
        self.entries.iter().map(|entry| entry.appended).sum()
    }
}
