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

//! Server prompt lookup on first-generation Inkling (issue #3425), on the
//! tiny model at real attention geometry. Inkling owns its caches and its
//! short-convolution state, so a token-only verify round in the scheduler
//! goes through the model's own rewind (`rewind_decode_appends`, #3423), not
//! the pool's KV trim the Llama tests in `scheduler_prompt_lookup_tests`
//! exercise. A row with prompt lookup must emit exactly the plain row's
//! tokens, alone and next to a second row in the same batch.

use std::sync::{Arc, mpsc};

use mlxcel_core::speculative::prompt_lookup::PromptLookupConfig;

use super::scheduler_prompt_lookup_tests::{acceptance, enqueue};
use super::*;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const MAX_TOKENS: usize = 40;

/// A prompt whose n-grams recur, so lookup proposes from the first decode
/// step on, and one without a repeat.
fn prompts() -> [Vec<i32>; 2] {
    [
        [1, 2, 3, 4, 5, 6, 7, 8].repeat(3),
        vec![9, 4, 17, 30, 2, 11, 25, 6, 13],
    ]
}

fn scheduler(prompt_lookup: bool) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        crate::LoadedModel::Inkling(
            crate::models::inkling::tiny_tests::tiny_model_with_real_heads(),
        ),
        MlxcelTokenizer::stub(),
        Vec::new(),
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Dense,
    );
    if prompt_lookup {
        sched = sched.with_speculative_dispatch(crate::server::SpeculativeDispatch::PromptLookup {
            config: PromptLookupConfig::default(),
            max_batch: 2,
        });
    }
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

/// Run the scheduler until idle; each finished row's `(prompt, tokens)`.
fn drain(sched: &mut BatchScheduler) -> Vec<(Vec<i32>, Vec<i32>)> {
    let mut done = Vec::new();
    for _ in 0..1000 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => return done,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        for seq in sched.active_batch.iter_sequences() {
            if seq.state.is_finished() {
                done.push((seq.prompt_tokens.clone(), seq.generated_tokens.clone()));
            }
        }
        sched.finalize_completed();
    }
    panic!("the scheduler did not go idle");
}

fn tokens_of(done: &[(Vec<i32>, Vec<i32>)], prompt: &[i32]) -> Vec<i32> {
    done.iter()
        .find(|(p, _)| p == prompt)
        .map(|(_, t)| t.clone())
        .expect("the row finished")
}

#[test]
fn inkling_prompt_lookup_rows_emit_the_plain_tokens() {
    let prompts = prompts();
    let plain: Vec<Vec<i32>> = prompts
        .iter()
        .map(|prompt| {
            let mut sched = scheduler(false);
            let _rx = enqueue(&mut sched, prompt, MAX_TOKENS);
            tokens_of(&drain(&mut sched), prompt)
        })
        .collect();

    // Alone.
    let mut proposed_total = 0;
    for (prompt, want) in prompts.iter().zip(&plain) {
        let mut sched = scheduler(true);
        let rx = enqueue(&mut sched, prompt, MAX_TOKENS);
        assert_eq!(&tokens_of(&drain(&mut sched), prompt), want);
        if let Some((proposed, accepted)) = acceptance(&rx) {
            assert!(accepted <= proposed);
            proposed_total += proposed;
        }
    }
    // The gate admits the family only with the row kernel, which needs the
    // Metal matvec kernel; elsewhere the rows decode plainly.
    if mlxcel_core::row_invariant::available() {
        assert!(
            proposed_total > 0,
            "the repeating prompt drew no proposal, so no verify round ran"
        );
    }

    // Side by side in one batch, within the verify batch limit.
    let mut sched = scheduler(true);
    let _a = enqueue(&mut sched, &prompts[0], MAX_TOKENS);
    let _b = enqueue(&mut sched, &prompts[1], MAX_TOKENS);
    let done = drain(&mut sched);
    for (prompt, want) in prompts.iter().zip(&plain) {
        assert_eq!(&tokens_of(&done, prompt), want, "batched row {prompt:?}");
    }
}
