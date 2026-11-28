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

use super::*;
use crate::models::speculative_exactness::compare_block_against_chain;

/// A stand-in for a sliding-window layer with the defect Inkling 2 had
/// (#3280): a decode step attends the newest `window` tokens, and row `r` of
/// a verify block attends from the window start of the block's first row, so
/// it also reads the `r` oldest keys the step has dropped. Counts forwards,
/// so a test can price a probe run.
struct SlidingToy {
    window: usize,
    /// `false` gives every block row its own window start, the fix.
    block_shares_window_start: bool,
    forwards: std::cell::Cell<usize>,
}

impl SlidingToy {
    fn new(window: usize, block_shares_window_start: bool) -> Self {
        Self {
            window,
            block_shares_window_start,
            forwards: std::cell::Cell::new(0),
        }
    }

    /// The "logits" of the token at `position` reading `context[start..=position]`.
    fn row(context: &[i32], start: usize, position: usize) -> Vec<u8> {
        let sum: i64 = context[start..=position]
            .iter()
            .map(|&t| i64::from(t))
            .sum();
        sum.to_le_bytes().to_vec()
    }

    fn window_start(&self, position: usize) -> usize {
        (position + 1).saturating_sub(self.window)
    }

    fn probe(&self, block_size: usize, draw: usize, prompt_len: usize) -> BlockChainExactness {
        let vocab = 1000;
        let mut context = prompt_tokens(draw, prompt_len, vocab);
        context.extend(block_tokens(draw, block_size, vocab));
        // Two prefills, `block_size` steps and one block, as a real draw runs.
        self.forwards.set(self.forwards.get() + block_size + 3);

        let chain: Vec<Vec<u8>> = (0..block_size)
            .map(|r| Self::row(&context, self.window_start(prompt_len + r), prompt_len + r))
            .collect();
        let block: Vec<Vec<u8>> = (0..block_size)
            .map(|r| {
                let start = if self.block_shares_window_start {
                    self.window_start(prompt_len)
                } else {
                    self.window_start(prompt_len + r)
                };
                Self::row(&context, start, prompt_len + r)
            })
            .collect();
        compare_block_against_chain(&block, &chain)
    }

    fn run(&self, block_size: usize, lengths: &[usize]) -> BlockChainExactness {
        run_draws(lengths, |draw, prompt_len| {
            self.probe(block_size, draw, prompt_len)
        })
    }
}

/// Issue #3409: the short draws end before the window fills, so they pass a
/// model whose block rows read keys the decode steps have dropped. The long
/// draw catches it at the first row that has an extra key, and passes the
/// fixed model.
#[test]
fn the_long_draw_catches_the_sliding_window_case() {
    let (window, block_size) = (512, 8);
    let broken = SlidingToy::new(window, true);

    let short = prompt_lengths_under(8, 3, Some(past_window(window)), LongContext::Off);
    assert_eq!(short, vec![8, 8, 8]);
    assert_eq!(broken.run(block_size, &short), BlockChainExactness::Equal);

    let with_long = prompt_lengths_under(8, 3, Some(past_window(window)), LongContext::Auto);
    assert_eq!(with_long, vec![8, 8, 8, window + LONG_CONTEXT_MARGIN]);
    assert!(
        matches!(
            broken.run(block_size, &with_long),
            BlockChainExactness::Diverges { position: 1, .. }
        ),
        "row 0 shares the step's window; row 1 is the first with an extra key"
    );

    let fixed = SlidingToy::new(window, false);
    assert_eq!(
        fixed.run(block_size, &with_long),
        BlockChainExactness::Equal
    );
}

/// The gate: a family that declares no boundary runs the draws it ran before
/// the long draw existed, forward for forward.
#[test]
fn a_family_without_a_boundary_pays_nothing_for_the_long_draw() {
    assert_eq!(
        prompt_lengths_under(8, 3, None, LongContext::Auto),
        vec![8; 3]
    );
    assert_eq!(
        prompt_lengths_under(48, 2, None, LongContext::Auto),
        vec![48; 2]
    );

    let block_size = 4;
    let toy = SlidingToy::new(512, false);
    toy.run(
        block_size,
        &prompt_lengths_under(8, 3, None, LongContext::Auto),
    );
    let without = toy.forwards.replace(0);
    assert_eq!(without, 3 * (block_size + 3));
    toy.run(
        block_size,
        &prompt_lengths_under(8, 3, Some(past_window(512)), LongContext::Auto),
    );
    assert_eq!(toy.forwards.get(), without + block_size + 3);
}

#[test]
fn the_override_moves_the_gate_either_way() {
    // Off drops a declared long draw; a length adds one where none is declared.
    assert_eq!(
        prompt_lengths_under(8, 3, Some(544), LongContext::Off),
        vec![8; 3]
    );
    assert_eq!(
        prompt_lengths_under(8, 3, None, LongContext::Tokens(2100)),
        vec![8, 8, 8, 2100]
    );
    assert_eq!(
        prompt_lengths_under(8, 3, Some(544), LongContext::Tokens(2100)),
        vec![8, 8, 8, 2100]
    );
    // A "long" draw no longer than the short ones adds nothing.
    assert_eq!(
        prompt_lengths_under(160, 3, None, LongContext::Tokens(64)),
        vec![160; 3]
    );
}

#[test]
fn the_override_parses_off_words_lengths_and_noise() {
    assert_eq!(LongContext::parse(None), LongContext::Auto);
    for off in ["0", "off", " OFF ", "false", "no"] {
        assert_eq!(LongContext::parse(Some(off)), LongContext::Off, "{off:?}");
    }
    assert_eq!(
        LongContext::parse(Some(" 2100 ")),
        LongContext::Tokens(2100)
    );
    for noise in ["1", "", "long", "-5"] {
        assert_eq!(
            LongContext::parse(Some(noise)),
            LongContext::Auto,
            "{noise:?}"
        );
    }
}

#[test]
fn draws_stop_at_the_first_divergence_and_refuse_an_empty_plan() {
    let mut seen = Vec::new();
    let verdict = run_draws(&[8, 8, 544], |draw, prompt_len| {
        seen.push((draw, prompt_len));
        if draw == 1 {
            BlockChainExactness::Diverges {
                position: 0,
                differing_bytes: 1,
                total_bytes: 8,
            }
        } else {
            BlockChainExactness::Equal
        }
    });
    assert!(matches!(verdict, BlockChainExactness::Diverges { .. }));
    assert_eq!(seen, vec![(0, 8), (1, 8)]);
    assert!(matches!(
        run_draws(&[], |_, _| BlockChainExactness::Equal),
        BlockChainExactness::NotRun(_)
    ));
}

#[test]
fn synthetic_tokens_stay_in_the_vocabulary_and_vary_by_draw() {
    let vocab = 97;
    let a = prompt_tokens(0, 600, vocab);
    let b = prompt_tokens(1, 600, vocab);
    assert_eq!(a.len(), 600);
    assert!(a.iter().chain(&b).all(|&t| (0..vocab as i32).contains(&t)));
    assert_ne!(a, b);
    assert_ne!(block_tokens(0, 8, vocab), block_tokens(1, 8, vocab));
}
