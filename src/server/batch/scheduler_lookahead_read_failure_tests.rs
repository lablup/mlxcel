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

use super::scheduler_model_owned_lookahead_tests::{enqueue, finish, scheduler, tick};
use super::scheduler_model_owned_pad_trim_tests::tiny_gemma3;
use super::*;
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};

use crate::server::config::DecodeStorageBackend;
use crate::server::model_provider::GenerateEvent;

fn model_offsets(sched: &BatchScheduler, seq_id: SequenceId) -> (i32, i32) {
    fn scalar(snapshot: &ModelStateSnapshot, name: &str) -> i32 {
        let value = snapshot
            .tensor(name)
            .unwrap_or_else(|| panic!("snapshot has no {name}"));
        mlxcel_core::utils::array_to_vec_f32(value)[0] as i32
    }

    let snapshot = sched
        .engine
        .model()
        .snapshot_sequence_state(seq_id, 1)
        .expect("Gemma 3 snapshots a live sequence");
    (
        scalar(&snapshot, "layer0.rotating.offset"),
        scalar(&snapshot, "layer1.standard.offset"),
    )
}

fn check_failed_readback(next_prime_issued: bool) {
    let mut sched = scheduler(tiny_gemma3(), DecodeStorageBackend::Dense, false);
    let failed_rx = enqueue(&mut sched, 4, 0, 64);
    while sched.active_batch.is_empty() {
        assert!(tick(&mut sched), "the first request is admitted");
        sched.finalize_completed();
    }
    let failed_id = sched.active_batch.sequence_ids()[0];

    let survivor_rx = enqueue(&mut sched, 4, 1, 6);
    while sched.active_batch.len() < 2 {
        assert!(tick(&mut sched), "the unrelated request is admitted");
        sched.finalize_completed();
    }
    let survivor_id = sched
        .active_batch
        .sequence_ids()
        .into_iter()
        .find(|&id| id != failed_id)
        .expect("the unrelated request is active");

    sched.discard_lookahead();
    let failed_generated = sched
        .active_batch
        .get(failed_id)
        .expect("failed row remains active")
        .generated_tokens
        .clone();
    let survivor_generated = sched
        .active_batch
        .get(survivor_id)
        .expect("survivor remains active")
        .generated_tokens
        .clone();
    let offsets = model_offsets(&sched, failed_id);

    let next_max = failed_generated.len() + if next_prime_issued { 8 } else { 1 };
    sched
        .active_batch
        .get_mut(failed_id)
        .expect("failed row remains active")
        .max_tokens = next_max;
    let params = sched
        .lookahead_params(&[failed_id])
        .expect("the failed row is lookahead eligible");
    sched.maybe_prime_lookahead(&[failed_id]);
    let failed_lookahead = sched
        .decode_lookahead
        .take()
        .expect("the failed cohort has one speculative append");
    let failures = sched.consecutive_decode_eval_failures;
    let forwards = sched.decode_forward_calls;
    let _failure = super::decode_tick::inject_lookahead_read_failure(
        "mlxcel test: synthetic array read failure",
    );

    sched.pipelined_steady_decode(failed_lookahead, &[failed_id], &params);

    assert_eq!(
        sched.decode_forward_calls,
        forwards + usize::from(next_prime_issued),
        "only the submitted-next scenario builds step n+1"
    );
    assert_eq!(
        sched.consecutive_decode_eval_failures,
        failures + 1,
        "one failed command buffer counts once"
    );
    assert!(sched.decode_lookahead.is_none());
    assert!(!sched.shutdown_requested);
    assert_eq!(model_offsets(&sched, failed_id), offsets);
    assert_eq!(
        sched
            .active_batch
            .get(failed_id)
            .expect("failed row remains until finalization")
            .generated_tokens,
        failed_generated,
        "the unread pending token is never emitted"
    );
    assert!(matches!(
        sched.active_batch.get(failed_id).map(|seq| &seq.state),
        Some(SequenceState::Finished(FinishReason::Error(msg)))
            if msg.contains("synthetic array read failure")
    ));
    assert!(matches!(
        sched.active_batch.get(survivor_id).map(|seq| &seq.state),
        Some(SequenceState::Decoding)
    ));
    assert_eq!(
        sched
            .active_batch
            .get(survivor_id)
            .expect("survivor remains active")
            .generated_tokens,
        survivor_generated
    );

    let errors: Vec<_> = failed_rx
        .try_iter()
        .filter_map(|event| match event {
            GenerateEvent::Error(msg) => Some(msg),
            _ => None,
        })
        .collect();
    assert_eq!(errors.len(), 1, "the failed row receives exactly one error");
    assert!(errors[0].contains("synthetic array read failure"));
    assert!(
        survivor_rx
            .try_iter()
            .all(|event| !matches!(event, GenerateEvent::Error(_))),
        "the unrelated active row receives no error"
    );

    sched.finalize_completed();
    for _ in 0..64 {
        if !tick(&mut sched) {
            break;
        }
        sched.finalize_completed();
    }
    let survivor = finish(&survivor_rx);
    assert_eq!(survivor.len(), 6);
    assert!(survivor.len() > survivor_generated.len());
}

#[test]
fn failed_lookahead_read_is_request_local_with_and_without_a_next_prime() {
    check_failed_readback(false);
    check_failed_readback(true);
}
