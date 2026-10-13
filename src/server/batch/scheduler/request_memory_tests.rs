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
use crate::models::nemotron_h::{BlockType, NemotronHConfig, NemotronHModel};
use mlxcel_core::memory::MemorySnapshot;
use mlxcel_core::weights::WeightMap;

fn scheduler() -> BatchScheduler {
    let mut c: NemotronHConfig = serde_json::from_value(serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":8, "hidden_size":8,
        "intermediate_size":16, "num_hidden_layers":1, "num_attention_heads":2,
        "num_key_value_heads":2, "mamba_num_heads":2, "mamba_head_dim":4,
        "ssm_state_size":4, "conv_kernel":4, "n_groups":1,
        "hybrid_override_pattern":"*"
    }))
    .unwrap();
    c.post_init().unwrap();
    let mut weights = WeightMap::new();
    for (name, shape) in [
        ("backbone.embeddings.weight", vec![8, 8]),
        ("backbone.layers.0.norm.weight", vec![8]),
        ("backbone.norm_f.weight", vec![8]),
        ("lm_head.weight", vec![8, 8]),
        ("backbone.layers.0.mixer.q_proj.weight", vec![8, 8]),
        ("backbone.layers.0.mixer.k_proj.weight", vec![8, 8]),
        ("backbone.layers.0.mixer.v_proj.weight", vec![8, 8]),
        ("backbone.layers.0.mixer.o_proj.weight", vec![8, 8]),
    ] {
        weights.insert(
            name.to_string(),
            mlxcel_core::full_f32(&shape, 0.1, mlxcel_core::dtype::FLOAT32),
        );
    }
    let model = NemotronHModel::from_weights(c, weights, vec![BlockType::Attention]).unwrap();
    let (_, rx) = mpsc::channel();
    let mut scheduler = BatchScheduler::with_config(
        LoadedModel::NemotronH(model),
        MlxcelTokenizer::stub(),
        vec![7],
        rx,
        1,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        256,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Dense,
    )
    .with_max_kv_size(Some(128));
    scheduler.request_memory_test_inputs = Some((MemorySnapshot::default(), Some(1)));
    install_thread_local_default_stream(scheduler.generation_stream.as_ref());
    scheduler
}

fn enqueue(scheduler: &mut BatchScheduler, tokens: usize) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    scheduler.enqueue_request(
        "prompt".into(),
        Some(vec![1; tokens]),
        super::scheduler_model_owned_cache_tests::options(),
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
fn request_memory_rejection_has_zero_forward_and_next_request_can_run() {
    let mut scheduler = scheduler();
    let rejected = enqueue(&mut scheduler, 64);
    scheduler.execute_prefill(SequenceId::from_raw(0));
    assert!(matches!(
        rejected.recv().unwrap(),
        GenerateEvent::RequestMemoryRejected(_)
    ));
    assert!(scheduler.prefill_forward_lengths.is_empty());
    assert!(scheduler.prefill_queue.is_empty());
    assert!(scheduler.prompt_cache_seq_ctx.is_empty());
    assert!(scheduler.active_batch.is_empty());

    scheduler.request_memory_test_inputs = Some((MemorySnapshot::default(), Some(1 << 30)));
    let accepted = enqueue(&mut scheduler, 4);
    scheduler.execute_prefill(SequenceId::from_raw(0));
    assert!(matches!(accepted.recv().unwrap(), GenerateEvent::Admitted));
    assert_eq!(scheduler.prefill_forward_lengths, vec![4]);
}

#[test]
fn request_memory_context_precedence_and_cancelled_waiter_cleanup() {
    let mut scheduler = scheduler();
    let rejected = enqueue(&mut scheduler, 128);
    assert!(
        matches!(rejected.recv().unwrap(), GenerateEvent::Error(message) if message.contains("context size"))
    );
    assert!(scheduler.prefill_forward_lengths.is_empty());
    let (tx, rx) = mpsc::channel();
    scheduler.enqueue_request(
        "prompt".into(),
        Some(vec![1; 4]),
        super::scheduler_model_owned_cache_tests::options(),
        vec![],
        vec![],
        vec![],
        tx,
        Arc::new(AtomicBool::new(true)),
        true,
    );
    scheduler.execute_prefill(SequenceId::from_raw(0));
    assert!(rx.recv().is_err());
    assert!(scheduler.prefill_forward_lengths.is_empty());
    assert!(scheduler.prefill_queue.is_empty());
}

#[test]
fn request_memory_other_family_is_unchanged() {
    let store = super::scheduler_model_owned_cache_tests::test_store();
    let scheduler = super::scheduler_model_owned_cache_tests::scheduler(store);
    assert!(!scheduler.request_memory_policy_enabled());
}
