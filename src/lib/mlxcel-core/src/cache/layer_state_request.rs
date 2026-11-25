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

//! The snapshot vocabulary and the forward request of a scheduler-owned
//! family (issue #3254, ADR 0010). The storage itself is in
//! [`super::layer_state`].

use super::SequenceId;
use crate::ffi::MlxArray;

/// How a family's per-layer state is named inside a
/// [`crate::generate::ModelStateSnapshot`]. The encodings predate scheduler
/// ownership, and stored snapshots must stay readable, so the generic
/// snapshot code writes whichever one the family already used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerStateEncoding {
    /// `layer{i}.{standard}.*` for a full layer and `layer{i}.{rotating}.*`
    /// for a sliding one, each with its own offset and ring scalars.
    Windowed {
        standard: &'static str,
        rotating: &'static str,
    },
    /// `layer{i}.attention.{mode,keys,values}` for a full layer and
    /// `layer{i}.linear.{slot}` for a recurrent one; every offset is the
    /// snapshot's token length.
    Hybrid,
}

/// The snapshot vocabulary of a scheduler-owned family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerStateSnapshotSpec {
    /// The [`crate::generate::ModelStateSnapshot`] family tag, which rejects
    /// a cross-family restore.
    pub family: &'static str,
    /// The family's name in error messages.
    pub label: &'static str,
    pub encoding: LayerStateEncoding,
}

/// One forward over scheduler-owned state.
///
/// `input_ids` is `[B, L]` with one row per entry of the `rows` argument of
/// [`crate::generate::LanguageModel::forward_layer_state`].
#[derive(Clone, Copy)]
pub struct LayerStateForward<'a> {
    pub input_ids: &'a MlxArray,
    /// Input embeddings that replace the token embedding (a multimodal
    /// prefill).
    pub input_embeddings: Option<&'a MlxArray>,
    pub mask: Option<&'a MlxArray>,
    /// Project only this position through the LM head (a prefill piece whose
    /// last real row is the one sampled). `None` returns every position.
    pub last_pos: Option<usize>,
    /// The scheduler's sequence ids of the rows, for per-sequence side state
    /// a model keys by id outside its layers (multimodal position tables).
    /// `None` is the id-less single-sequence run.
    pub seq_ids: Option<&'a [SequenceId]>,
}

impl<'a> LayerStateForward<'a> {
    /// A text forward of `input_ids` returning every position.
    pub const fn tokens(input_ids: &'a MlxArray) -> Self {
        Self {
            input_ids,
            input_embeddings: None,
            mask: None,
            last_pos: None,
            seq_ids: None,
        }
    }

    #[must_use]
    pub const fn with_embeddings(mut self, input_embeddings: Option<&'a MlxArray>) -> Self {
        self.input_embeddings = input_embeddings;
        self
    }

    #[must_use]
    pub const fn with_mask(mut self, mask: Option<&'a MlxArray>) -> Self {
        self.mask = mask;
        self
    }

    #[must_use]
    pub const fn with_last_pos(mut self, last_pos: Option<usize>) -> Self {
        self.last_pos = last_pos;
        self
    }

    #[must_use]
    pub const fn with_seq_ids(mut self, seq_ids: Option<&'a [SequenceId]>) -> Self {
        self.seq_ids = seq_ids;
        self
    }

    /// The id of row `row`, when the scheduler named the rows.
    pub fn seq_id(&self, row: usize) -> Option<SequenceId> {
        self.seq_ids.and_then(|ids| ids.get(row).copied())
    }
}
