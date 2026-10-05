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
use crate::test_support::induction::{
    INDUCTION_MODELS, INDUCTION_VOCAB, InductionModel, lcg_tokens, sequential_reference,
};

fn cfg(ngram_max: usize, ngram_min: usize, max_draft: usize) -> PromptLookupConfig {
    PromptLookupConfig {
        ngram_max,
        ngram_min,
        max_draft,
        adaptive: true,
        policy: DraftPolicy::Graded,
    }
}

fn gated(max_draft: usize) -> PromptLookupConfig {
    PromptLookupConfig {
        policy: DraftPolicy::Gated,
        ..cfg(3, 2, max_draft)
    }
}

#[test]
fn proposes_the_tokens_after_an_earlier_occurrence() {
    // "... 1 2 3 4 5 6 ... 1 2" -> propose "3 4 5".
    let context = [9, 1, 2, 3, 4, 5, 6, 8, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 3)), vec![3, 4, 5]);
}

#[test]
fn draft_is_capped_by_max_draft_and_context_end() {
    let context = [1, 2, 3, 4, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 7)), vec![3, 4, 1, 2]);
    assert_eq!(find_draft(&context, &cfg(2, 1, 1)), vec![3]);
}

#[test]
fn prefers_the_longest_matching_suffix() {
    // Suffix [5, 2]: the 2-gram matches at index 3, the 1-gram [2] matches
    // later at index 6. The longer match must win.
    let context = [0, 0, 0, 5, 2, 7, 2, 9, 5, 2];
    assert_eq!(find_draft(&context, &cfg(2, 1, 1)), vec![7]);
}

#[test]
fn prefers_the_most_recent_occurrence_of_a_given_length() {
    let context = [1, 2, 10, 1, 2, 20, 1, 2];
    assert_eq!(find_draft(&context, &cfg(2, 2, 1)), vec![20]);
}

#[test]
fn never_matches_the_suffix_against_itself() {
    // The only occurrence of [4] is the suffix itself.
    assert!(find_draft(&[1, 2, 3, 4], &cfg(1, 1, 3)).is_empty());
}

#[test]
fn falls_back_to_shorter_ngrams() {
    // [3, 4] never occurred before, [4] did.
    let context = [4, 7, 8, 3, 4];
    assert_eq!(find_draft(&context, &cfg(2, 1, 2)), vec![7, 8]);
    assert!(find_draft(&context, &cfg(2, 2, 2)).is_empty());
}

#[test]
fn short_and_empty_contexts_propose_nothing() {
    let c = cfg(3, 1, 7);
    assert!(find_draft(&[], &c).is_empty());
    assert!(find_draft(&[1], &c).is_empty());
    assert_eq!(find_draft(&[1, 1], &c), vec![1]);
}

#[test]
fn repeated_run_proposes_the_run() {
    // A run of one token: the latest occurrence of [7, 7] is one back, so
    // exactly one token is available after it.
    let context = [7, 7, 7, 7];
    assert_eq!(find_draft(&context, &cfg(2, 1, 5)), vec![7]);
}

#[test]
fn config_validation() {
    assert!(PromptLookupConfig::default().validate().is_ok());
    assert!(cfg(3, 0, 7).validate().is_err());
    assert!(cfg(1, 2, 7).validate().is_err());
    assert!(cfg(3, 1, 0).validate().is_err());
}

#[test]
fn tokens_per_forward_excludes_the_prefill_token() {
    let stats = PromptLookupStats {
        rounds: 4,
        drafted_rounds: 4,
        paused_rounds: 0,
        proposed_draft_tokens: 28,
        accepted_draft_tokens: 20,
        shadow_confirmations: 0,
    };
    // 25 generated: 1 from prefill, 24 over 4 rounds.
    assert!((stats.tokens_per_forward(25) - 6.0).abs() < 1e-9);
    assert!((stats.acceptance_rate() - 20.0 / 28.0).abs() < 1e-9);
    assert_eq!(PromptLookupStats::default().tokens_per_forward(1), 0.0);
}

#[test]
fn governor_starts_with_the_full_block() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    assert_eq!(g.budget(), 7);
}

#[test]
fn governor_keeps_the_full_block_while_blocks_land() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..10 {
        g.record(6);
    }
    assert_eq!(g.budget(), 7);
}

#[test]
fn governor_shortens_the_block_when_few_tokens_land() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..10 {
        g.record(1);
    }
    // EMA converges on 1: twice that plus two.
    assert_eq!(g.budget(), 4);
}

/// Record one miss streak long enough to start a pause.
fn miss_streak(g: &mut DraftGovernor) {
    for _ in 0..MISSES_BEFORE_COOLDOWN {
        g.record(0);
    }
}

/// Count the paused rounds, ending on the first round that proposes again.
fn pause_len(g: &mut DraftGovernor) -> usize {
    let mut paused = 0;
    while g.budget() == 0 {
        paused += 1;
    }
    paused
}

#[test]
fn governor_pauses_after_repeated_misses_and_backs_off() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 1..MISSES_BEFORE_COOLDOWN {
        g.record(0);
    }
    assert!(g.budget() > 0, "a shorter run of misses is not a streak");
    g.record(0);
    assert_eq!(pause_len(&mut g), MIN_COOLDOWN);
    // A second streak pauses twice as long.
    miss_streak(&mut g);
    assert_eq!(pause_len(&mut g), 2 * MIN_COOLDOWN);
}

#[test]
fn governor_backoff_resets_once_a_round_lands() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..3 {
        miss_streak(&mut g);
        pause_len(&mut g);
    }
    g.record(2);
    miss_streak(&mut g);
    assert_eq!(pause_len(&mut g), MIN_COOLDOWN);
}

#[test]
fn governor_backoff_is_capped() {
    let mut g = DraftGovernor::new(&cfg(3, 2, 7));
    let mut last = 0;
    for _ in 0..10 {
        miss_streak(&mut g);
        last = pause_len(&mut g);
    }
    assert_eq!(last, MAX_COOLDOWN);
}

#[test]
fn non_adaptive_governor_always_proposes_max_draft() {
    let mut c = cfg(3, 2, 7);
    c.adaptive = false;
    let mut g = DraftGovernor::new(&c);
    for _ in 0..10 {
        g.record(0);
        assert_eq!(g.budget(), 7);
    }
}

#[test]
fn default_ngram_min_skips_single_token_matches() {
    // Only a 1-gram ([4]) recurs; the default minimum of 2 proposes nothing.
    let context = [4, 7, 8, 3, 4];
    assert!(find_draft(&context, &PromptLookupConfig::default()).is_empty());
}

#[test]
fn index_matches_the_scan_at_every_length() {
    for (seed, alphabet) in [(1, 3), (2, 5), (3, 11), (4, 40)] {
        for config in [cfg(3, 1, 7), cfg(3, 2, 7), cfg(5, 2, 4), cfg(1, 1, 3)] {
            let tokens = lcg_tokens(seed, 300, alphabet);
            let mut index = NgramIndex::new(&config);
            // Grow the context one token at a time, as decoding does.
            for len in 0..=tokens.len() {
                let context = &tokens[..len];
                index.extend(context);
                assert_eq!(
                    index.find(context, &config),
                    find_draft(context, &config),
                    "seed {seed} alphabet {alphabet} {config:?} len {len}"
                );
            }
        }
    }
}

#[test]
fn index_matches_the_scan_when_extended_in_blocks() {
    // A verify round emits several tokens before the next lookup.
    let config = cfg(3, 2, 7);
    let tokens = lcg_tokens(7, 400, 6);
    let mut index = NgramIndex::new(&config);
    let mut len = 50;
    while len <= tokens.len() {
        let context = &tokens[..len];
        index.extend(context);
        assert_eq!(index.find(context, &config), find_draft(context, &config));
        len += 1 + len % 7;
    }
}

/// Target that always predicts token 3 at every position, so a reply is a run
/// of 3s: lookup proposes full blocks and the loop guard has a loop to catch.
struct ConstantModel;

impl LanguageModel for ConstantModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        let seq_len = ffi::array_shape(input_ids)[1] as usize;
        let mut logits = Vec::with_capacity(seq_len * 8);
        for _ in 0..seq_len {
            logits.extend_from_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }
        ffi::from_slice_f32(&logits, &[1, seq_len as i32, 8])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }
}

#[test]
fn loop_guard_stops_mid_block_where_plain_decoding_stops() {
    let sampling = SamplingConfig {
        loop_detection: crate::loop_detection::LoopDetectionConfig::new(1, 1, 8),
        ..SamplingConfig::greedy()
    };
    let prompt = [1, 2, 3, 3];
    let mut plain = crate::generate::CxxGenerator::new(1);
    let expected = plain.generate(&ConstantModel, &prompt, 64, &sampling);
    assert_eq!(
        expected,
        vec![3; 8],
        "plain decoding stops on the 8th repeat"
    );

    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let (tokens, _) = generator.generate(&ConstantModel, &prompt, 64, &sampling);
    assert_eq!(tokens, expected);
    assert!(
        generator.stats().accepted_draft_tokens > 0,
        "the run verified lookup blocks"
    );
}

#[test]
fn without_the_loop_guard_lookup_runs_to_max_tokens() {
    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let (tokens, _) =
        generator.generate(&ConstantModel, &[1, 2, 3, 3], 64, &SamplingConfig::greedy());
    assert_eq!(tokens, vec![3; 64]);
}

#[test]
fn config_validation_bounds_the_index_and_the_verify_block() {
    assert!(cfg(NGRAM_MAX_LIMIT, 2, MAX_DRAFT_LIMIT).validate().is_ok());
    let err = cfg(NGRAM_MAX_LIMIT + 1, 2, 7).validate().unwrap_err();
    assert!(err.contains("ngram-max"), "{err}");
    let err = cfg(3, 2, MAX_DRAFT_LIMIT + 1).validate().unwrap_err();
    assert!(err.contains("max-draft"), "{err}");
}

/// Run prompt lookup on both [`INDUCTION_MODELS`] over several prompts and
/// configurations, assert every reply equals `reference(model, prompt)`, and
/// return the summed acceptance accounting per model, for `policy` only.
fn induction_parity(
    sampling: &SamplingConfig,
    policy: DraftPolicy,
    reference: impl Fn(&InductionModel, &[i32]) -> Vec<i32>,
) -> [PromptLookupStats; 2] {
    let mut totals = [PromptLookupStats::default(); 2];
    for (model, total) in INDUCTION_MODELS.iter().zip(&mut totals) {
        for seed in 1..=6 {
            let prompt = lcg_tokens(seed, 48, INDUCTION_VOCAB as u64);
            let expected = reference(model, &prompt);
            assert_eq!(expected.len(), 96, "seed {seed}: the reference ran short");
            let bases = [
                cfg(3, 2, 7),
                cfg(3, 1, 7),
                cfg(4, 2, 3),
                // Six-token matches are rare in a random prompt and appear
                // once the reply repeats itself, so the loop first pipelines a
                // long run of plain rounds and then switches to verifying with
                // a step still in flight.
                cfg(6, 6, 7),
                PromptLookupConfig {
                    adaptive: false,
                    ..cfg(3, 2, 7)
                },
            ];
            for config in bases
                .iter()
                .map(|&base| PromptLookupConfig { policy, ..base })
            {
                let mut generator = PromptLookupGenerator::new(config);
                let (tokens, _) = generator.generate(model, &prompt, 96, sampling);
                assert_eq!(
                    tokens, expected,
                    "earliest={} seed {seed} {config:?}",
                    model.earliest
                );
                let stats = generator.stats();
                total.rounds += stats.rounds;
                total.drafted_rounds += stats.drafted_rounds;
                total.paused_rounds += stats.paused_rounds;
                total.proposed_draft_tokens += stats.proposed_draft_tokens;
                total.accepted_draft_tokens += stats.accepted_draft_tokens;
                total.shadow_confirmations += stats.shadow_confirmations;
            }
        }
    }
    totals
}

/// Both acceptance regimes showed up under `policy`: blocks that land and
/// blocks that are trimmed, governor pauses long enough for the loop to
/// pipeline, and, for [`DraftPolicy::Gated`] only, resumes on a confirmed
/// proposal.
fn assert_both_regimes(policy: DraftPolicy, totals: &[PromptLookupStats; 2]) {
    for stats in totals {
        assert!(
            stats.accepted_draft_tokens > 0,
            "no proposal landed: {stats:?}"
        );
        assert!(
            stats.proposed_draft_tokens > stats.accepted_draft_tokens,
            "no proposal was rejected, so no trim was exercised: {stats:?}"
        );
    }
    let missing = &totals[1];
    assert!(
        missing.paused_rounds > 0,
        "the missing model never paused the governor: {missing:?}"
    );
    let confirmations: usize = totals.iter().map(|stats| stats.shadow_confirmations).sum();
    match policy {
        DraftPolicy::Gated => assert!(
            confirmations > 0,
            "no paused gated governor ever resumed on a confirmed proposal: {totals:?}"
        ),
        _ => assert_eq!(confirmations, 0, "only a gated governor probes: {totals:?}"),
    }
}

/// The rollback contract: after every verify round the caches must hold
/// exactly the emitted tokens, so a cache-dependent target produces plain
/// decoding's reply token for token, through accepted blocks, rejected tails,
/// and the pipelined plain rounds between them.
#[test]
fn rollback_keeps_a_cache_dependent_target_identical_to_plain_decoding() {
    let sampling = SamplingConfig::greedy();
    for policy in [DraftPolicy::Graded, DraftPolicy::Gated] {
        let totals = induction_parity(&sampling, policy, |model, prompt| {
            let plain =
                crate::generate::CxxGenerator::new(1).generate(model, prompt, 96, &sampling);
            assert_eq!(
                plain,
                sequential_reference(model, prompt, 96, &sampling),
                "greedy plain decoding is the sequential reference"
            );
            plain
        });
        assert_both_regimes(policy, &totals);
    }
}

/// Same contract on the per-position sampler path: a repetition penalty
/// reads the emitted history, so every verify position samples on its own
/// through the incremental sampler state, and each must see every token
/// emitted before it, including the ones accepted earlier in the same block.
#[test]
fn rollback_matches_plain_decoding_under_a_repetition_penalty() {
    let sampling = SamplingConfig {
        repetition_penalty: 1.3,
        penalty_last_n: 3,
        ..SamplingConfig::greedy()
    };
    assert!(sampling.needs_token_history());
    for policy in [DraftPolicy::Graded, DraftPolicy::Gated] {
        let totals = induction_parity(&sampling, policy, |model, prompt| {
            let plain =
                crate::generate::CxxGenerator::new(1).generate(model, prompt, 96, &sampling);
            assert_eq!(
                plain,
                sequential_reference(model, prompt, 96, &sampling),
                "plain decoding under a penalty is the sequential reference (#2090)"
            );
            plain
        });
        assert_both_regimes(policy, &totals);
    }
}

/// A dense-cache target that nonetheless owns its sequence state, the shape
/// Gemma 4 and Llama 4 have: it batches and accepts padded prefill, so only
/// the layout says the external caches are placeholders.
struct ModelOwnedStateModel;

impl LanguageModel for ModelOwnedStateModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        caches: &mut [KVCache],
        mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        ConstantModel.forward(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }

    fn sequence_state_layout(&self) -> crate::cache::SequenceStateLayout {
        crate::cache::SequenceStateLayout::model_owned(1)
    }
}

#[test]
fn model_owned_sequence_state_is_refused_even_when_other_flags_allow_it() {
    let model = ModelOwnedStateModel;
    assert!(model.supports_batching() && model.supports_padded_prefill());
    let reason = prompt_lookup_unsupported_reason(&model).expect("must be refused");
    assert!(reason.contains("internally"), "{reason}");
    assert!(supports_prompt_lookup(&ConstantModel));
}

#[test]
#[should_panic(expected = "trimmable external KV caches")]
fn generate_refuses_a_model_without_external_caches() {
    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let _ = generator.generate(
        &ModelOwnedStateModel,
        &[1, 2, 3],
        8,
        &SamplingConfig::greedy(),
    );
}

#[test]
#[should_panic(expected = "mirostat or adaptive-p")]
fn generate_refuses_a_sampler_with_feedback_state() {
    let sampling = SamplingConfig {
        mirostat: 2,
        ..SamplingConfig::with_temperature(0.8)
    };
    let mut generator = PromptLookupGenerator::new(PromptLookupConfig::default());
    let _ = generator.generate(&ConstantModel, &[1, 2, 3], 8, &sampling);
}

/// Records the width of every forward and keeps the tokens in its cache, so
/// the warmup's trims are observable.
struct WidthRecordingModel {
    widths: std::cell::RefCell<Vec<i32>>,
}

impl LanguageModel for WidthRecordingModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        let seq_len = ffi::array_shape(input_ids)[1];
        self.widths.borrow_mut().push(seq_len);
        let kv = ffi::ones(&[1, 1, seq_len, 1], crate::dtype::FLOAT32);
        caches[0].update(kv, ffi::ones(&[1, 1, seq_len, 1], crate::dtype::FLOAT32));
        ConstantModel.forward(input_ids, &mut [], None)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }
}

#[test]
fn warmup_runs_every_verify_width_and_rolls_each_back() {
    let model = WidthRecordingModel {
        widths: std::cell::RefCell::new(Vec::new()),
    };
    let prompt = [1, 2, 3, 4, 5];
    let mut generator = PromptLookupGenerator::new(cfg(3, 2, 4));
    generator.warm_up_verify_widths(&model, &prompt);
    assert_eq!(*model.widths.borrow(), vec![5, 2, 3, 4, 5]);
    assert_eq!(generator.caches[0].offset, prompt.len() as i32);
}

#[test]
fn gated_governor_starts_narrow_on_probation() {
    let mut g = DraftGovernor::new(&gated(7));
    assert_eq!(g.budget(), GATED_NARROW_DRAFT);
    // The first round of a reply has nothing behind it: one miss pauses.
    g.record(0);
    assert_eq!(g.budget(), 0);
    assert!(g.probes_while_paused());
}

#[test]
fn gated_pause_has_no_length_and_ends_on_a_confirmed_proposal() {
    let mut g = DraftGovernor::new(&gated(7));
    g.record(0);
    for _ in 0..1000 {
        assert_eq!(g.budget(), 0, "a gated pause must not expire on its own");
    }
    g.shadow_confirmed();
    assert!(!g.probes_while_paused());
    assert_eq!(g.budget(), GATED_NARROW_DRAFT, "it resumes narrow");
    // Resumed on probation: one more miss pauses again at once.
    g.record(0);
    assert_eq!(g.budget(), 0);
}

#[test]
fn gated_block_widens_after_a_whole_narrow_block_and_narrows_after_misses() {
    let mut g = DraftGovernor::new(&gated(7));
    g.record(GATED_NARROW_DRAFT);
    assert_eq!(
        g.budget(),
        7,
        "a narrow block that landed whole earns the full one"
    );
    g.record(5);
    assert_eq!(g.budget(), 7);
    // Off probation now, so misses shrink the block before they pause it.
    g.record(0);
    g.record(0);
    assert_eq!(g.budget(), GATED_NARROW_DRAFT);
    g.record(0);
    assert_eq!(g.budget(), 0, "the third miss in a row pauses");
}

#[test]
fn gated_block_never_uses_the_widths_between_narrow_and_full() {
    let mut g = DraftGovernor::new(&gated(7));
    for accepted in [2, 7, 3, 1, 0, 1, 2, 6, 0, 4, 1, 2, 2, 0, 5] {
        let budget = g.budget();
        assert!(
            budget == 0 || budget == GATED_NARROW_DRAFT || budget == 7,
            "budget {budget}"
        );
        if budget > 0 {
            g.record(accepted.min(budget));
        } else {
            g.shadow_confirmed();
        }
    }
}

#[test]
fn gated_respects_a_max_draft_below_the_narrow_block() {
    let mut g = DraftGovernor::new(&gated(1));
    assert_eq!(g.budget(), 1);
    g.record(1);
    assert_eq!(g.budget(), 1);
}

#[test]
fn shadow_confirmation_is_ignored_unless_paused_and_gated() {
    let mut g = DraftGovernor::new(&gated(7));
    g.record(2);
    g.shadow_confirmed();
    assert_eq!(g.budget(), 7, "a drafting governor keeps its full block");

    let mut graded = DraftGovernor::new(&cfg(3, 2, 7));
    for _ in 0..MISSES_BEFORE_COOLDOWN {
        graded.record(0);
    }
    assert!(!graded.probes_while_paused());
    graded.shadow_confirmed();
    assert_eq!(
        pause_len(&mut graded),
        MIN_COOLDOWN,
        "graded pauses are unchanged"
    );

    let mut fixed = DraftGovernor::new(&PromptLookupConfig {
        adaptive: false,
        ..gated(7)
    });
    fixed.record(0);
    assert_eq!(fixed.budget(), 7);
    assert!(!fixed.probes_while_paused());
}

#[test]
fn shadow_probe_settles_on_the_first_two_emitted_tokens() {
    let probe = ShadowProbe {
        start: 3,
        proposal: vec![7, 8, 9],
    };
    assert_eq!(probe.settle(&[1, 2, 3]), None, "nothing emitted yet");
    assert_eq!(probe.settle(&[1, 2, 3, 7]), None, "one token is not enough");
    assert_eq!(probe.settle(&[1, 2, 3, 7, 8]), Some(true));
    assert_eq!(
        probe.settle(&[1, 2, 3, 7, 8, 0]),
        Some(true),
        "only the first two count"
    );
    assert_eq!(
        probe.settle(&[1, 2, 3, 6]),
        Some(false),
        "settles at the first miss"
    );
    assert_eq!(probe.settle(&[1, 2, 3, 7, 6]), Some(false));
}

#[test]
fn default_policy_follows_the_build() {
    let expected = if cfg!(feature = "cuda") {
        DraftPolicy::Gated
    } else {
        DraftPolicy::Graded
    };
    assert_eq!(PromptLookupConfig::default().policy, expected);
}

/// Vocabulary of [`ScriptModel`].
const SCRIPT_VOCAB: usize = 1024;

/// Target that emits a fixed script by absolute position: the logits after
/// position `p` peak at `script[p + 1]`. Every token is unique except where
/// the script copies its own prompt, so a lookup proposal is right exactly
/// when it is aligned with the copy.
struct ScriptModel {
    script: Vec<i32>,
}

impl LanguageModel for ScriptModel {
    fn forward(
        &self,
        input_ids: &ffi::MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&ffi::MlxArray>,
    ) -> UniquePtr<ffi::MlxArray> {
        let seq_len = ffi::array_shape(input_ids)[1];
        let offset = caches[0].offset as usize;
        let kv = || ffi::ones(&[1, 1, seq_len, 1], crate::dtype::FLOAT32);
        caches[0].update(kv(), kv());
        let mut logits = vec![0.0f32; seq_len as usize * SCRIPT_VOCAB];
        for row in 0..seq_len as usize {
            let next = self.script[offset + row + 1] as usize;
            logits[row * SCRIPT_VOCAB + next] = 10.0;
        }
        ffi::from_slice_f32(&logits, &[1, seq_len, SCRIPT_VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        Vec::new()
    }
}

/// A reply that starts copying its prompt, breaks off for one token, and
/// resumes the copy: the first drafted round misses on probation, the
/// governor pauses, and a shadow probe must notice the resumed copy.
///
/// Prompt `100..=119`; reply `300, 100, 101, 999, 102, 103, ..., 119`. After
/// `100 101` the lookup proposes `102 103`, the target emits `999`, and that
/// probation miss pauses. Two rounds later `102 103` matches again and the
/// probe made there proposes `104 105`, which come true, so drafting resumes
/// and copies the rest. A probe recorded one position early or late compares
/// `104` with `103` or `105` and never confirms.
fn copy_break_copy_script() -> (Vec<i32>, usize) {
    let prompt: Vec<i32> = (100..120).collect();
    let mut script = prompt.clone();
    script.extend([300, 100, 101, 999]);
    script.extend(102..120);
    // Room for the model to look one position past the last emitted token.
    script.push(0);
    (script, prompt.len())
}

#[test]
fn gated_shadow_probe_confirms_exactly_the_resumed_copy() {
    let (script, prompt_len) = copy_break_copy_script();
    let model = ScriptModel {
        script: script.clone(),
    };
    let expected = &script[prompt_len..script.len() - 1];
    let penalty = SamplingConfig {
        repetition_penalty: 1.1,
        penalty_last_n: 8,
        ..SamplingConfig::greedy()
    };
    // Pipelined plain rounds, and the synchronous path a history-reading
    // sampler forces.
    for sampling in [SamplingConfig::greedy(), penalty] {
        let mut generator = PromptLookupGenerator::new(gated(7));
        let (tokens, _) =
            generator.generate(&model, &script[..prompt_len], expected.len(), &sampling);
        assert_eq!(tokens, expected, "{sampling:?}");
        let stats = generator.stats();
        assert_eq!(
            stats.shadow_confirmations, 1,
            "the resumed copy is confirmed once: {stats:?}"
        );
        assert!(
            stats.drafted_rounds >= 3 && stats.accepted_draft_tokens >= 10,
            "drafting resumed and copied the rest: {stats:?}"
        );
    }
}

#[test]
fn gated_probation_ends_only_on_a_whole_narrow_block() {
    let mut g = DraftGovernor::new(&gated(7));
    // One stray token is not evidence of a copy: still on probation.
    g.record(1);
    assert_eq!(g.budget(), GATED_NARROW_DRAFT);
    g.record(0);
    assert_eq!(
        g.budget(),
        0,
        "a miss on probation pauses even after a partial round"
    );

    let mut g = DraftGovernor::new(&gated(7));
    g.record(GATED_NARROW_DRAFT);
    g.record(0);
    assert_eq!(
        g.budget(),
        GATED_NARROW_DRAFT,
        "off probation, one miss narrows the block instead of pausing"
    );
}
