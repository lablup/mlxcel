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

//! First-divergence comparison of two generated token streams.
//!
//! The parity harness reports, per pair of decode paths, either `identical`
//! or the index of the first token where the two streams differ together
//! with the token id each side produced there. A stream that ended earlier
//! (an end-of-sequence stop on one side only) diverges at its length, with
//! the missing side reported as `<end>`.

use std::fmt;

/// Result of comparing two generated token streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Divergence {
    /// Both streams hold the same ids in the same order and have equal length.
    Identical,
    /// The streams differ first at `index`. `left` / `right` are the ids at
    /// that index, `None` when that stream had already ended.
    At {
        index: usize,
        left: Option<i32>,
        right: Option<i32>,
    },
}

impl Divergence {
    /// Whether the two streams matched exactly.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        matches!(self, Self::Identical)
    }
}

/// Compare `left` and `right` and report the first divergent position.
#[must_use]
pub fn first_divergence(left: &[i32], right: &[i32]) -> Divergence {
    let common = left.iter().zip(right).take_while(|(a, b)| a == b).count();
    if common == left.len() && common == right.len() {
        return Divergence::Identical;
    }
    Divergence::At {
        index: common,
        left: left.get(common).copied(),
        right: right.get(common).copied(),
    }
}

fn token_label(token: Option<i32>) -> String {
    token.map_or_else(|| "<end>".to_string(), |id| id.to_string())
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identical => f.write_str("identical"),
            Self::At { index, left, right } => write!(
                f,
                "diverge@{index}{} left={} right={}",
                if *index == 0 { " (first token)" } else { "" },
                token_label(*left),
                token_label(*right),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Divergence, first_divergence};

    #[test]
    fn identical_streams_report_identical() {
        assert_eq!(
            first_divergence(&[1, 2, 3], &[1, 2, 3]),
            Divergence::Identical
        );
        assert_eq!(first_divergence(&[], &[]), Divergence::Identical);
        assert_eq!(first_divergence(&[1, 2], &[1, 2]).to_string(), "identical");
    }

    #[test]
    fn divergence_reports_index_and_both_ids() {
        let d = first_divergence(&[1, 2, 3, 4], &[1, 2, 9, 4]);
        assert_eq!(
            d,
            Divergence::At {
                index: 2,
                left: Some(3),
                right: Some(9)
            }
        );
        assert_eq!(d.to_string(), "diverge@2 left=3 right=9");
    }

    #[test]
    fn divergence_at_token_zero_is_labelled() {
        let d = first_divergence(&[5, 6], &[7, 6]);
        assert_eq!(d.to_string(), "diverge@0 (first token) left=5 right=7");
        assert!(!d.is_identical());
    }

    #[test]
    fn early_stop_on_one_side_diverges_at_its_length() {
        let d = first_divergence(&[1, 2], &[1, 2, 3]);
        assert_eq!(
            d,
            Divergence::At {
                index: 2,
                left: None,
                right: Some(3)
            }
        );
        assert_eq!(d.to_string(), "diverge@2 left=<end> right=3");
        let d = first_divergence(&[], &[4]);
        assert_eq!(d.to_string(), "diverge@0 (first token) left=<end> right=4");
    }
}
