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

//! Tests of the classic-decode fallback of the MTP round loop (issue
//! #3425): a chain target that checks every cache row it holds, and a
//! drafter whose proposals are never accepted.

use super::*;

/// What the fallback tests' target and drafter saw, shared with the test
/// body because the generator owns both.
#[derive(Default)]
struct FallbackLog {
    /// `'V'` per verify forward, `'C'` per classic step, in order.
    timeline: String,
    /// Tokens handed to `extend_history`, in order.
    caught_up: Vec<i32>,
}

/// A target whose greedy token after `x` is `x + 1`. A verify block of two
/// or more rows costs `verify_cost`, a classic step `classic_cost`. It keeps
/// the token of every cache row it holds and checks each new row continues
/// the chain, so a step read twice, skipped, or left queued under a drafted
/// round fails the test.
struct ChainTarget {
    exact: bool,
    verify_cost: std::time::Duration,
    classic_cost: std::time::Duration,
    log: std::rc::Rc<RefCell<FallbackLog>>,
    rows: RefCell<Vec<i32>>,
    block_start: std::cell::Cell<usize>,
}

impl ChainTarget {
    /// Append the row of `token`, which must follow the last row.
    fn append(&self, token: i32) {
        let mut rows = self.rows.borrow_mut();
        let want = rows.last().map_or(100, |last| last + 1);
        assert_eq!(token, want, "cache rows out of order after {rows:?}");
        rows.push(token);
    }

    fn output() -> MtpVerifyOutput {
        MtpVerifyOutput {
            next_hidden: MockMtpTarget::dummy_tensor(),
            next_shared_kv: Vec::new(),
            kv_offset: 0,
            bonus_position: 0,
            verify_hidden_full: None,
        }
    }
}

impl MtpTarget for ChainTarget {
    fn prefill_and_seed(
        &self,
        _prompt_tokens: &[i32],
        _sampler: &SamplingConfig,
        _token_history: &[i32],
        _logprobs_config: &LogprobsConfig,
    ) -> (
        i32,
        MtpVerifyOutput,
        Option<crate::sampling::TokenLogprobData>,
    ) {
        (100, Self::output(), None)
    }

    fn embed_token(&self, _token_id: i32) -> UniquePtr<MlxArray> {
        MockMtpTarget::dummy_tensor()
    }

    fn verify_forward(
        &self,
        verify_input: &[i32],
        _sampler: &SamplingConfig,
        _logprobs_config: &LogprobsConfig,
    ) -> VerifyForwardOutput {
        self.log.borrow_mut().timeline.push('V');
        self.append(verify_input[0]);
        self.block_start.set(self.rows.borrow().len() - 1);
        self.rows.borrow_mut().extend_from_slice(&verify_input[1..]);
        if verify_input.len() > 1 {
            std::thread::sleep(self.verify_cost);
        }
        VerifyForwardOutput {
            target_tokens: verify_input.iter().map(|t| t + 1).collect(),
            target_logprobs: None,
            captured: VerifyCaptured {
                tensors: Vec::new(),
                scalars: Vec::new(),
            },
        }
    }

    fn verify_finalize(
        &self,
        accepted: usize,
        _block_size: usize,
        _captured: VerifyCaptured,
    ) -> MtpVerifyOutput {
        self.rows
            .borrow_mut()
            .truncate(self.block_start.get() + accepted + 1);
        Self::output()
    }

    fn exact_at_every_verify_width(&self) -> bool {
        self.exact
    }

    fn classic_step(
        &self,
        input: &MlxArray,
        _sampler: &SamplingConfig,
    ) -> Option<crate::speculative::mtp::target::MtpClassicStep> {
        self.log.borrow_mut().timeline.push('C');
        self.append(ffi::item_i32(&ffi::reshape(input, &[])));
        std::thread::sleep(self.classic_cost);
        let one = ffi::from_slice_i32(&[1], &[1]);
        Some(crate::speculative::mtp::target::MtpClassicStep {
            token: ffi::add(&ffi::reshape(input, &[1]), &one),
            hidden: MockMtpTarget::dummy_tensor(),
        })
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// A drafter whose proposals are never accepted, and which records the
/// classic steps it is handed.
struct MissingDrafter {
    log: std::rc::Rc<RefCell<FallbackLog>>,
}

impl Drafter for MissingDrafter {
    fn bind(&mut self, _target: &dyn crate::generate::LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn set_shared_kv(
        &mut self,
        _shared_kv: SharedKv<'_>,
        _kv_offset: usize,
        _position: usize,
        _left_padding: usize,
    ) -> Result<(), DrafterError> {
        Ok(())
    }

    fn draft_block(
        &mut self,
        _last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        Ok(vec![-7; block_size - 1])
    }

    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    fn extends_history_without_draft(&self) -> bool {
        true
    }

    fn extend_history(
        &mut self,
        hidden: &MlxArray,
        tokens: &[i32],
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        assert_eq!(ffi::array_shape(hidden), [1, tokens.len() as i32, 4]);
        self.log.borrow_mut().caught_up.extend_from_slice(tokens);
        Ok(())
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Mtp
    }
}

/// Run `max_tokens` greedy tokens at `--draft-block-size 4` over a chain
/// target whose drafted rounds cost 3 ms and classic steps 1 ms.
fn run_fallback(exact: bool, max_tokens: usize) -> (Vec<i32>, FallbackLog) {
    let log = std::rc::Rc::new(RefCell::new(FallbackLog::default()));
    let target = ChainTarget {
        exact,
        verify_cost: std::time::Duration::from_millis(3),
        classic_cost: std::time::Duration::from_millis(1),
        log: log.clone(),
        rows: RefCell::new(Vec::new()),
        block_start: std::cell::Cell::new(0),
    };
    let drafter = Box::new(MissingDrafter { log: log.clone() });
    let mut generator = MtpGenerator::new(target, drafter, 4);
    let (tokens, _, _) = generator.generate(
        &[1, 2, 3],
        max_tokens,
        &SamplingConfig::greedy(),
        &[],
        &AtomicBool::new(false),
        &LogprobsConfig::default(),
    );
    // Every emitted token but the last is the input of exactly one kept row:
    // no classic step is left queued at the end of the budget.
    let (target, drafter) = generator.into_parts();
    let rows = target.rows.take();
    drop((target, drafter));
    assert_eq!(rows, tokens[..tokens.len() - 1], "cache rows");
    let log = std::rc::Rc::try_unwrap(log)
        .ok()
        .expect("the generator released the log")
        .into_inner();
    (tokens, log)
}

/// Under acceptance that never pays for a drafted round, the round loop
/// spends most of the run in classic steps, keeps re-trying the drafter,
/// hands the drafter every classic step it ran before each re-try, and
/// emits exactly the target's greedy chain.
#[test]
fn the_fallback_engages_under_low_acceptance_and_reprobes() {
    let (tokens, log) = run_fallback(true, 400);
    let expected: Vec<i32> = (100..500).collect();
    assert_eq!(tokens, expected, "greedy chain");

    let classic = log.timeline.matches('C').count();
    assert!(
        classic * 10 >= tokens.len() * 8,
        "mostly classic steps: {}",
        log.timeline
    );
    let first_classic = log.timeline.find('C').expect("a classic step ran");
    let reprobes = log.timeline[first_classic..].matches('V').count();
    assert!(
        reprobes >= 2,
        "the drafter was re-tried after classic decode took over: {}",
        log.timeline
    );
    // Each re-try is preceded by a catch-up with the classic steps the
    // drafter missed; the run's last classic steps follow the last re-try.
    assert!(!log.caught_up.is_empty() && log.caught_up.len() <= classic);
    let mut previous = 99;
    for &token in &log.caught_up {
        assert!(
            token > previous,
            "caught-up tokens in order: {:?}",
            log.caught_up
        );
        previous = token;
    }
}

/// A target that does not claim exactness at every width keeps the
/// requested width every round, as before issue #3425.
#[test]
fn the_fallback_stays_off_without_exactness_at_every_width() {
    let (tokens, log) = run_fallback(false, 40);
    assert_eq!(tokens, (100..140).collect::<Vec<_>>());
    assert!(!log.timeline.contains('C'), "{}", log.timeline);
    assert!(log.caught_up.is_empty());
}
