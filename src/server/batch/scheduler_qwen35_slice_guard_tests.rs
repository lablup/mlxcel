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

//! The Qwen 3.5 MTP slice guard (issue #3254).
//!
//! A Qwen 3.5 slice runs over the layer state the scheduler's pool holds for
//! its sequence. Slice 0 checks that the entry has it; a later slice whose
//! entry lost it must fail the request with a client-visible error instead
//! of letting the target adapter panic.

use std::sync::{Arc, mpsc};

use super::*;
use crate::server::batch::speculative_slice_tests::unfinished_slice_job;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ServerConfig};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

#[test]
fn qwen35_slice_without_pool_layer_state_fails_the_request() {
    let _runtime = crate::initialize_runtime();
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        LoadedModel::Qwen35(crate::models::qwen3_5_tests::two_layer_model()),
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
        ServerConfig::default().max_batch_prefill,
        DecodeStorageBackend::Dense,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());

    // The sequence was never admitted, so the pool holds no entry for it.
    let (job, events) = unfinished_slice_job(901);
    assert!(sched.engine.pool().layer_state(job.seq.seq_id).is_none());
    sched.speculative_slice = Some(Box::new(job));

    sched.execute_speculative_slice_round();

    assert!(sched.speculative_slice.is_none(), "the job must be dropped");
    let mut errors = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let GenerateEvent::Error(message) = event {
            errors.push(message);
        }
    }
    assert_eq!(errors.len(), 1, "exactly one error event: {errors:?}");
    assert!(
        errors[0].contains("holds no scheduler-owned layer state"),
        "{}",
        errors[0]
    );
}
