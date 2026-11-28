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

//! Which prefill computation wrote each span of a snapshot's rows (issue
//! #3047).
//!
//! A turn-boundary state checkpoint keeps only a hybrid model's recurrent
//! state and takes the attention rows from a longer snapshot of the same
//! conversation. That is exact only when the longer snapshot's leading rows
//! are the very rows the checkpoint's own sequence held: rows written by a
//! different prefill partition of the same tokens can differ by a rounding
//! step, and mixing them with the checkpoint's recurrent state would give a
//! state that no single run produces.
//!
//! A lineage records, for each span of a snapshot's rows, the sequence that
//! computed it. A restore copies rows verbatim, so a sequence that adopted a
//! snapshot at `m` tokens inherits that snapshot's lineage clipped to `m`,
//! and every row it computes itself is attributed to its own id. Two
//! snapshots hold bit-identical rows over a prefix when their lineages
//! clipped to that prefix are equal.

/// Spans of a snapshot's rows and the sequence that computed each.
///
/// Stored as `(end, sequence id)` pairs with strictly increasing ends; span
/// `i` covers `[end_{i-1}, end_i)`. An empty lineage means the producer is
/// unknown, which never matches anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotLineage {
    spans: Vec<(usize, u64)>,
}

impl SnapshotLineage {
    /// Rows the lineage describes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.last().map_or(0, |&(end, _)| end)
    }

    /// Whether no producer is recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// The lineage of the first `len` rows.
    #[must_use]
    pub fn clipped(&self, len: usize) -> Self {
        let mut spans = Vec::with_capacity(self.spans.len());
        for &(end, id) in &self.spans {
            if end >= len {
                if len > spans.last().map_or(0, |&(e, _)| e) {
                    spans.push((len, id));
                }
                break;
            }
            spans.push((end, id));
        }
        Self { spans }
    }

    /// The lineage of `len` rows when the rows past [`Self::len`] were
    /// computed by sequence `computed_by`. A `len` at or below the current
    /// length clips instead.
    #[must_use]
    pub fn extended(&self, len: usize, computed_by: u64) -> Self {
        if len <= self.len() {
            return self.clipped(len);
        }
        let mut spans = self.spans.clone();
        match spans.last_mut() {
            Some((end, id)) if *id == computed_by => *end = len,
            _ => spans.push((len, computed_by)),
        }
        Self { spans }
    }

    /// Whether the first `len` rows of `self` and `other` were computed by the
    /// same sequences in the same spans. False when either side records no
    /// producer for all `len` rows.
    #[must_use]
    pub fn shares_prefix(&self, other: &Self, len: usize) -> bool {
        len > 0
            && self.len() >= len
            && other.len() >= len
            && self.clipped(len) == other.clipped(len)
    }
}

#[cfg(test)]
mod tests {
    use super::SnapshotLineage;

    fn chain(steps: &[(usize, u64)]) -> SnapshotLineage {
        steps
            .iter()
            .fold(SnapshotLineage::default(), |l, &(len, id)| {
                l.extended(len, id)
            })
    }

    #[test]
    fn extension_attributes_new_rows_and_merges_one_producer() {
        let l = chain(&[(100, 1), (250, 1), (400, 2)]);
        assert_eq!(l.len(), 400);
        assert_eq!(l.clipped(250), chain(&[(250, 1)]));
        assert_eq!(l.clipped(300), chain(&[(250, 1), (300, 2)]));
        assert_eq!(l.extended(200, 9), chain(&[(200, 1)]));
        assert!(l.clipped(0).is_empty());
    }

    #[test]
    fn a_shared_prefix_needs_the_same_producers_over_it() {
        // Turn 2 restored turn 1's rows at 250 and computed the rest.
        let turn1 = chain(&[(250, 1)]);
        let turn2 = turn1.extended(400, 2);
        assert!(turn2.shares_prefix(&turn1, 250));
        assert!(turn2.shares_prefix(&turn1, 100));
        // A cold run of the same tokens computed everything itself.
        let cold = chain(&[(400, 3)]);
        assert!(!cold.shares_prefix(&turn1, 250));
        // Past the shorter lineage, or with nothing recorded, nothing matches.
        assert!(!turn2.shares_prefix(&turn1, 300));
        assert!(!SnapshotLineage::default().shares_prefix(&SnapshotLineage::default(), 0));
    }
}
