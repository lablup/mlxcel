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

//! Decode fault regressions: backend errors stay request-local and speculative
//! state agrees with the committed synchronous prefix.
use super::direct_decode_rows_tests::NoisyModel;
use super::direct_decode_tests::{CountModel, OwnedModel, trace};
use super::*;
use crate::eval_faults::{FaultGuard, FaultPoint};
use crate::from_slice_i32;
use crate::generate::SamplingConfig;
use crate::sampling::{LogprobSource, LogprobsConfig};
use crate::sampling_row_step::RowSampler;

fn configs() -> Vec<SamplingConfig> {
    let base = SamplingConfig {
        temperature: 0.8,
        seed: Some(7),
        ..SamplingConfig::default()
    };
    vec![
        base.clone(),
        SamplingConfig {
            mirostat: 2,
            ..base.clone()
        },
        SamplingConfig {
            adaptive_target: 0.5,
            ..base.clone()
        },
        SamplingConfig {
            repetition_penalty: 1.3,
            ..base
        },
    ]
}

fn check_seeded_submit_stream(mut sampling: SamplingConfig) {
    // A seed fixes the key sequence, but CPU and GPU categorical kernels need
    // not map a key to the same token. Keep the baseline and every faulted run
    // on one default device while sibling tests exercise CPU device guards.
    let _device = crate::streams::lock_default_device();
    sampling.token_bias.insert(15, f32::NEG_INFINITY);
    let model = NoisyModel::default();
    let mut sync = DirectEngine::new(&model, 0).with_force_sync(true);
    let expected = trace(&mut sync, &[1, 4], 32, &sampling, None);
    assert_eq!(expected.tokens.len(), 32);
    for nth in [0, 1, 5] {
        let mut piped = DirectEngine::new(&model, 0).with_force_sync(false);
        let fault = FaultGuard::arm(FaultPoint::Submit, nth);
        let actual = trace(&mut piped, &[1, 4], 32, &sampling, None);
        assert!(fault.fired(), "the request must reach the failed submit");
        assert_eq!(actual, expected, "config {sampling:?}, submit {nth}");
    }
}

#[test]
fn failed_in_loop_submits_preserve_seeded_fused_stream() {
    check_seeded_submit_stream(configs()[0].clone());
}

#[test]
fn failed_in_loop_submits_preserve_seeded_mirostat_stream() {
    check_seeded_submit_stream(configs()[1].clone());
}

#[test]
fn failed_in_loop_submits_preserve_seeded_adaptive_stream() {
    check_seeded_submit_stream(configs()[2].clone());
}

#[test]
fn failed_in_loop_submits_preserve_seeded_penalty_stream() {
    check_seeded_submit_stream(configs()[3].clone());
}

#[test]
fn failed_submit_commits_pending_eos_and_callback_stop() {
    for sampling in [
        SamplingConfig::greedy(),
        SamplingConfig {
            repetition_penalty: 1.3,
            ..SamplingConfig::greedy()
        },
    ] {
        for (prompt, deliveries) in [(&[5][..], None), (&[0][..], Some(2))] {
            let model = CountModel::default();
            let mut sync = DirectEngine::new(&model, 0).with_force_sync(true);
            let expected = trace(&mut sync, prompt, 16, &sampling, deliveries);
            let mut piped = DirectEngine::new(&model, 0).with_force_sync(false);
            let _fault = FaultGuard::arm(FaultPoint::Submit, 1);
            assert_eq!(
                trace(&mut piped, prompt, 16, &sampling, deliveries),
                expected
            );
        }
    }
}

#[test]
fn failed_submit_commits_seeded_pending_eos() {
    let _device = crate::streams::lock_default_device();
    for mut sampling in [configs()[0].clone(), configs()[3].clone()] {
        sampling.token_bias.insert(15, f32::NEG_INFINITY);
        let model = NoisyModel::default();
        let mut sync = DirectEngine::new(&model, 0).with_force_sync(true);
        let baseline = trace(&mut sync, &[1, 4], 16, &sampling, None);
        // Choose an EOS reached in-loop without assuming backend-specific
        // categorical samples. A greedy redraw could conceal a lost pending draw.
        let eos_index = (2..baseline.tokens.len() - 1)
            .find(|&i| !baseline.tokens[..i].contains(&baseline.tokens[i]))
            .expect("the seeded stream must introduce an in-loop stop token");
        sampling.stop_token_ids = vec![baseline.tokens[eos_index]];
        let expected = trace(&mut sync, &[1, 4], 16, &sampling, None);
        assert_eq!(expected.tokens, baseline.tokens[..eos_index]);
        let mut piped = DirectEngine::new(&model, 0).with_force_sync(false);
        // Submit zero primes the pipeline; submit eos_index fails while the
        // already drawn EOS is pending, before its readback and commitment.
        let fault = FaultGuard::arm(FaultPoint::Submit, eos_index);
        let actual = trace(&mut piped, &[1, 4], 16, &sampling, None);
        assert!(
            fault.fired(),
            "the pending EOS must reach the failed submit"
        );
        assert_eq!(actual, expected, "config {sampling:?}");
    }
}

#[test]
fn failed_submit_commits_seeded_pending_callback_stop() {
    let _device = crate::streams::lock_default_device();
    for mut sampling in [configs()[0].clone(), configs()[3].clone()] {
        sampling.token_bias.insert(15, f32::NEG_INFINITY);
        let model = NoisyModel::default();
        let mut sync = DirectEngine::new(&model, 0).with_force_sync(true);
        let expected = trace(&mut sync, &[1, 4], 16, &sampling, Some(2));
        assert_eq!(expected.delivered.len(), 2);
        let mut piped = DirectEngine::new(&model, 0).with_force_sync(false);
        let fault = FaultGuard::arm(FaultPoint::Submit, 1);
        let actual = trace(&mut piped, &[1, 4], 16, &sampling, Some(2));
        assert!(
            fault.fired(),
            "the callback stop must reach the failed submit"
        );
        assert_eq!(actual, expected, "config {sampling:?}");
    }
}

#[test]
fn readback_helpers_return_errors() {
    let tokens = from_slice_i32(&[2], &[1]);
    {
        let _fault = FaultGuard::arm(FaultPoint::Readback, 0);
        assert_eq!(
            try_tokens_to_host(&tokens),
            Err("injected eval failure".into())
        );
    }
    let _fault = FaultGuard::arm(FaultPoint::Readback, 0);
    assert_eq!(
        super::rows::try_read_token(&tokens),
        Err("injected eval failure".into())
    );
}

#[test]
fn engine_token_readback_propagates_a_failed_event() {
    // Use a fresh fixture: Event::check_error consumes its attached error.
    let tokens = crate::array_with_failed_event_for_test();
    let err = try_tokens_to_host(&tokens).expect_err("failed event must reach the engine");
    assert!(
        err.contains("synthetic array read failure"),
        "unexpected engine readback error: {err}"
    );
}

#[test]
fn failed_readbacks_unwind_to_the_committed_prefix() {
    for sampling in [
        SamplingConfig::greedy(),
        SamplingConfig {
            repetition_penalty: 1.3,
            ..SamplingConfig::greedy()
        },
    ] {
        for nth in [0, 2] {
            let model = CountModel::default();
            let mut piped = DirectEngine::new(&model, 0).with_force_sync(false);
            let id = piped.open_sequence().unwrap();
            let mut delivered = Vec::new();
            {
                let _fault = FaultGuard::arm(FaultPoint::Readback, nth);
                let result = piped.run_open_sequence(
                    id,
                    &DirectRequest::text(&[0], 16, &sampling),
                    &sampling,
                    0,
                    |t| {
                        delivered.push(t);
                        true
                    },
                );
                assert!(
                    matches!(result, Err(DirectEngineError::Row(ref msg)) if msg == "injected eval failure")
                );
            }
            let set = piped.engine().pool().get(id).unwrap();
            let actual = (set.current_offset, set.caches[0].seq_len());
            let mut sync = DirectEngine::new(&model, 0).with_force_sync(true);
            let expected = trace(&mut sync, &[0], delivered.len(), &sampling, None);
            assert_eq!(delivered, expected.tokens);
            assert_eq!(actual.1, expected.kv_len.unwrap());
            assert_eq!(actual.0, expected.offset);
            piped.close_sequence(id);
            assert!(piped.engine().pool().get(id).is_none());
        }
    }
}

#[test]
fn discard_failures_release_the_sequence() {
    let _align = crate::generate::pin_tile_alignment(false);
    for sampling in [
        SamplingConfig::greedy(),
        SamplingConfig {
            repetition_penalty: 1.3,
            ..SamplingConfig::greedy()
        },
    ] {
        let mut faults = vec![(FaultPoint::Submit, 1)];
        if sampling.repetition_penalty != 1.0 {
            faults.extend([(FaultPoint::Draw, 0), (FaultPoint::Draw, 1)]);
        }
        for (point, nth) in faults {
            let model = OwnedModel::new(false);
            let mut client = DirectEngine::new(&model, 0).with_force_sync(false);
            let _fault = FaultGuard::arm(point, nth);
            assert!(matches!(
                client.run(&[0], 16, &sampling),
                Err(DirectEngineError::Step(EngineError::Eval(_)))
            ));
            assert_eq!(model.released.borrow().len(), 1);
        }
    }
}

#[test]
fn sampler_reads_report_eval_errors_and_failed_draws_restore_feedback() {
    let configs = [
        SamplingConfig {
            mirostat: 1,
            temperature: 0.8,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            mirostat: 2,
            temperature: 0.8,
            ..SamplingConfig::default()
        },
        SamplingConfig {
            mirostat: 2,
            ..SamplingConfig::greedy()
        },
        SamplingConfig {
            adaptive_target: 0.5,
            temperature: 0.8,
            ..SamplingConfig::default()
        },
    ];
    for sampling in configs {
        // Fail every evaluation/copy boundary, including later reads.
        let reads = if sampling.mirostat == 1 {
            3
        } else if sampling.adaptive_target >= 0.0 {
            2
        } else {
            1
        };
        for point in [FaultPoint::SamplerRead, FaultPoint::Draw] {
            for nth in 0..if point == FaultPoint::Draw { 1 } else { reads } {
                let mut engine = Engine::with_capacity(CountModel::default(), 1);
                let id = engine.open(SequenceSpec::default()).unwrap();
                let input = from_slice_i32(&[3], &[1, 1]);
                let logits = engine
                    .submit_forward(&StepBatch {
                        seq_ids: &[id],
                        input: &input,
                    })
                    .unwrap();
                let mut sampler = RowSampler::new(&sampling);
                // Establish noninitial feedback to ensure restoration retains it.
                let initial = sampler.draw_and_accept(&logits, &sampling, &[3]).unwrap();
                crate::try_eval(&initial.token).unwrap();
                let before = sampler.feedback_snapshot();
                let (mut history, mut generated) = (vec![3], vec![3]);
                let logprobs = LogprobsConfig::default();
                let mut row = StepRow {
                    seq_id: id,
                    sampler: &mut sampler,
                    sampling: &sampling,
                    token_history: &mut history,
                    generated: &mut generated,
                    eos: &[7],
                    max_tokens: 8,
                    logprobs: &logprobs,
                    needs_mask: false,
                    needs_override: false,
                    hooks: BareHooks,
                };
                {
                    let _fault = FaultGuard::arm(point, nth);
                    assert!(
                        matches!(engine.draw_row(&logits, &mut row), Err(EngineError::Eval(ref msg)) if msg == "injected eval failure")
                    );
                }
                assert_eq!(row.sampler.feedback_snapshot(), before);
                if point == FaultPoint::SamplerRead {
                    let distribution = LogprobsConfig {
                        enabled: true,
                        source: LogprobSource::PostSampling,
                        ..LogprobsConfig::default()
                    };
                    for logprobs in [&logprobs, &distribution] {
                        row.logprobs = logprobs;
                        let _fault = FaultGuard::arm(point, nth);
                        let outcome = engine
                            .step(
                                &StepBatch {
                                    seq_ids: &[id],
                                    input: &input,
                                },
                                std::slice::from_mut(&mut row),
                            )
                            .unwrap();
                        assert!(
                            matches!(outcome.rows[0].error, Some(RowError::Eval(ref msg)) if msg == "injected eval failure")
                        );
                        assert_eq!(row.sampler.feedback_snapshot(), before);
                    }
                }
                engine.close(id);
            }
        }
    }
}

#[test]
fn rng_snapshots_restore_only_the_owning_threads_key_sequence() {
    let _device = crate::streams::lock_default_device();
    fn draw() -> Vec<i32> {
        let logits = crate::ffi::zeros(&[32, 16], crate::dtype::FLOAT32);
        let token = crate::ffi::fused_sample(&logits, 0.8, 0, 1.0, 0.0);
        try_tokens_to_host(&token).unwrap()
    }
    crate::ffi::random_seed(7);
    let snapshot = crate::ffi::rng_snapshot().unwrap();
    let first = draw();
    std::thread::spawn(|| {
        crate::ffi::random_seed(99);
        let snapshot = crate::ffi::rng_snapshot().unwrap();
        let first = draw();
        crate::ffi::restore_rng_snapshot(&snapshot);
        assert_eq!(draw(), first);
    })
    .join()
    .unwrap();
    let second = draw();
    assert_ne!(first, second, "successful draws advance the key");
    crate::ffi::restore_rng_snapshot(&snapshot);
    assert_eq!(draw(), first);
    assert_eq!(draw(), second, "another thread must not change this stream");
}

#[test]
fn failed_draws_unwind_before_synchronous_fallback() {
    let _align = crate::generate::pin_tile_alignment(false);
    fn check<M: crate::generate::LanguageModel>(model: &M, sampling: &SamplingConfig) {
        let mut sync = DirectEngine::new(model, 0).with_force_sync(true);
        let expected = trace(&mut sync, &[0], 16, sampling, None);
        for nth in [0, 1] {
            let mut piped = DirectEngine::new(model, 0).with_force_sync(false);
            let fault = FaultGuard::arm(FaultPoint::Draw, nth);
            let actual = trace(&mut piped, &[0], 16, sampling, None);
            assert!(fault.fired());
            assert_eq!(actual, expected);
        }
    }
    for sampling in [
        SamplingConfig {
            repetition_penalty: 1.3,
            ..SamplingConfig::greedy()
        },
        SamplingConfig {
            mirostat: 2,
            ..SamplingConfig::greedy()
        },
    ] {
        check(&CountModel::default(), &sampling);
        let model = OwnedModel::new(true);
        check(&model, &sampling);
        let released = model.released.borrow();
        assert!(released.iter().all(|len| *len == released[0]));
    }
}
