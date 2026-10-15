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

//! Count-only requests (issue #1788): the scheduler answers the prepared
//! prompt length and releases the sequence before any prefill.

use super::*;

fn enqueue(
    scheduler: &mut BatchScheduler,
    tokens: usize,
    count_prompt_only: bool,
) -> mpsc::Receiver<GenerateEvent> {
    let mut options = super::scheduler_model_owned_cache_tests::options();
    options.count_prompt_only = count_prompt_only;
    options.max_tokens = 1;
    let (tx, rx) = mpsc::channel();
    scheduler.enqueue_request(
        "prompt".into(),
        Some(vec![1; tokens]),
        options,
        vec![],
        vec![],
        vec![],
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

#[test]
fn a_count_only_request_answers_its_length_without_prefill_or_a_live_sequence() {
    let mut scheduler = super::request_memory_tests::scheduler();
    scheduler.request_memory_test_inputs = Some((
        mlxcel_core::memory::MemorySnapshot::default(),
        Some(1 << 30),
    ));

    let counted = enqueue(&mut scheduler, 5, true);
    // The answer is sent before `enqueue_request` returns: a count never waits
    // for a scheduler tick.
    let Ok(GenerateEvent::Done(result)) = counted.try_recv() else {
        panic!("a count-only request answers Done from admission");
    };
    assert_eq!(result.prompt_tokens, 5);
    assert_eq!(result.completion_tokens, 0);
    assert!(result.generated_token_ids.is_empty());
    assert!(
        counted.recv().is_err(),
        "nothing follows the count: the sender is dropped with the request"
    );
    assert!(scheduler.prefill_queue.is_empty(), "nothing was queued");
    assert_eq!(
        scheduler.engine.pool().active_count(),
        0,
        "the sequence the preparation used is released"
    );
    assert!(scheduler.prompt_cache_seq_ctx.is_empty());
    scheduler.execute_prefill(SequenceId::from_raw(0));
    assert!(
        scheduler.prefill_forward_lengths.is_empty(),
        "no forward ran"
    );

    // The scheduler is left as it was: an ordinary request still prefills.
    let generated = enqueue(&mut scheduler, 4, false);
    assert_eq!(scheduler.prefill_queue.len(), 1);
    scheduler.execute_prefill(SequenceId::from_raw(0));
    assert!(matches!(generated.recv(), Ok(GenerateEvent::Admitted)));
    assert_eq!(scheduler.prefill_forward_lengths, vec![4]);
}
