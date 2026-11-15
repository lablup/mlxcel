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

//! `PipelineServerModel` over a fake runtime: lifecycle delegation and the
//! client-safe failure mapping (#2714, #2727).

use std::sync::{Arc, Mutex};

use super::*;
use crate::distributed::pipeline::PipelineModelRuntime;
use mlxcel_core::{dtype, from_slice_f32, from_slice_i32, zeros};

/// The text a remote runtime's errors carry: a stage address and the remote
/// sequence id, neither of which may reach a client.
const STAGE_ADDR: &str = "10.0.0.5:19001";

#[derive(Default)]
struct RuntimeLog {
    prepared: Mutex<Vec<u64>>,
    released: Mutex<Vec<u64>>,
    single_forwards: Mutex<Vec<u64>>,
    /// `(seq, last_pos)` of every `forward_sequence_last_row` call.
    last_row_forwards: Mutex<Vec<(u64, usize)>>,
    batched_forwards: Mutex<Vec<Vec<u64>>>,
}

#[derive(Default)]
struct FakeRuntime {
    log: Arc<RuntimeLog>,
    /// Refuse every prepare, as a remote runtime does when a stage
    /// answers its `PrepareSequence` with an `Error` (#2714 item a).
    refuse_prepare: bool,
    /// Fail every forward, as a remote runtime does when a stage answers
    /// a `RunEntry` with an `Error` or the exchange times out (#2727).
    fail_forward: bool,
}

impl PipelineModelRuntime for FakeRuntime {
    fn prepare_sequence_state(&self, seq_id: SequenceId) -> anyhow::Result<()> {
        if self.refuse_prepare {
            anyhow::bail!(
                "remote stage {STAGE_ADDR} failed request: sequence pp-seq-{}-0123456789abcdef \
                 was prepared by another coordinator identity",
                seq_id.as_u64()
            );
        }
        self.log.prepared.lock().unwrap().push(seq_id.as_u64());
        Ok(())
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.log.released.lock().unwrap().push(seq_id.as_u64());
    }

    fn forward_sequence(
        &self,
        seq_id: SequenceId,
        _input_ids: &MlxArray,
        _mask: Option<&MlxArray>,
    ) -> anyhow::Result<UniquePtr<MlxArray>> {
        if self.fail_forward {
            anyhow::bail!("remote stage {STAGE_ADDR} timed out after 30s");
        }
        self.log
            .single_forwards
            .lock()
            .unwrap()
            .push(seq_id.as_u64());
        Ok(from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]))
    }

    /// A stage that honors the request: only the asked row of the same
    /// `[1, 2, 2]` logits `forward_sequence` returns, as `[1, 1, 2]`.
    fn forward_sequence_last_row(
        &self,
        seq_id: SequenceId,
        _input_ids: &MlxArray,
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> anyhow::Result<UniquePtr<MlxArray>> {
        if self.fail_forward {
            anyhow::bail!("remote stage {STAGE_ADDR} timed out after 30s");
        }
        self.log
            .last_row_forwards
            .lock()
            .unwrap()
            .push((seq_id.as_u64(), last_pos));
        let full = from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        Ok(mlxcel_core::generate::logits_at_position(
            &full,
            last_pos.min(1),
        ))
    }

    fn forward_batched(
        &self,
        seq_ids: &[SequenceId],
        _input_ids: &MlxArray,
    ) -> anyhow::Result<UniquePtr<MlxArray>> {
        if self.fail_forward {
            anyhow::bail!("remote stage {STAGE_ADDR} failed request: stub stage forced error");
        }
        self.log
            .batched_forwards
            .lock()
            .unwrap()
            .push(seq_ids.iter().map(|seq| seq.as_u64()).collect());
        Ok(zeros(&[seq_ids.len() as i32, 1, 2], dtype::FLOAT32))
    }
}

fn model_with(runtime: FakeRuntime) -> PipelineServerModel {
    PipelineServerModel::new(2, vec![2], Box::new(runtime))
}

/// No stage address, remote sequence id or stage-side detail in a text a
/// client may see.
fn assert_client_safe(text: &str) {
    assert!(!text.contains(STAGE_ADDR), "stage address leaked: {text}");
    assert!(
        !text.contains("pp-seq-"),
        "remote sequence id leaked: {text}"
    );
    assert!(!text.contains("stub stage"), "stage detail leaked: {text}");
    assert!(!text.contains("timed out"), "stage detail leaked: {text}");
}

#[test]
fn pipeline_server_model_delegates_sequence_lifecycle() {
    let log = Arc::new(RuntimeLog::default());
    let model = model_with(FakeRuntime {
        log: log.clone(),
        ..FakeRuntime::default()
    });
    let seq_id = SequenceId::from_raw(7);

    model.prepare_sequence_state(seq_id);
    model.release_sequence_state_by_id(seq_id);

    assert_eq!(log.prepared.lock().unwrap().as_slice(), &[7]);
    assert_eq!(log.released.lock().unwrap().as_slice(), &[7]);
}

#[test]
fn pipeline_server_model_uses_runtime_for_forward_paths() {
    let log = Arc::new(RuntimeLog::default());
    let model = PipelineServerModel::new(
        4,
        vec![42],
        Box::new(FakeRuntime {
            log: log.clone(),
            ..FakeRuntime::default()
        }),
    );
    let seq0 = SequenceId::from_raw(1);
    let seq1 = SequenceId::from_raw(2);

    model.prepare_sequence_state(seq0);
    model.release_sequence_state_by_id(seq1);

    let input_ids = from_slice_i32(&[1, 2], &[2, 1]);
    let mut caches = Vec::<KVCache>::new();
    let _ =
        model.forward_with_sequence_id(input_ids.as_ref().unwrap(), Some(seq0), &mut caches, None);
    let mut cache_slices: Vec<&mut [KVCache]> = vec![&mut [], &mut []];
    let _ = model.forward_batched_with_ids(
        input_ids.as_ref().unwrap(),
        Some(&[seq0, seq1]),
        &mut cache_slices,
        None,
    );

    assert_eq!(log.prepared.lock().unwrap().as_slice(), &[1]);
    assert_eq!(log.released.lock().unwrap().as_slice(), &[2]);
    assert_eq!(log.single_forwards.lock().unwrap().as_slice(), &[1]);
    assert_eq!(
        log.batched_forwards.lock().unwrap().as_slice(),
        &[vec![1, 2]]
    );
}

/// The fallible forwards reach the runtime like the infallible ones, and
/// the last-logits variant slices the position the engine asked for.
#[test]
fn pipeline_server_model_try_forwards_use_the_runtime() {
    let log = Arc::new(RuntimeLog::default());
    let model = model_with(FakeRuntime {
        log: log.clone(),
        ..FakeRuntime::default()
    });
    let seq0 = SequenceId::from_raw(1);
    let seq1 = SequenceId::from_raw(2);
    let input_ids = from_slice_i32(&[1, 2], &[1, 2]);
    let mut caches = Vec::<KVCache>::new();

    let logits = model
        .try_forward_with_sequence_id(&input_ids, Some(seq0), &mut caches, None)
        .expect("a served forward is Ok");
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 2, 2]);
    let last = model
        .try_forward_last_logits_with_sequence_id(&input_ids, Some(seq0), &mut caches, None, 1)
        .expect("a served prefill forward is Ok");
    assert_eq!(mlxcel_core::array_shape(&last), vec![1, 1, 2]);
    assert_eq!(
        mlxcel_core::utils::array_to_vec_f32(&last),
        vec![3.0, 4.0],
        "the last position of the runtime's logits"
    );
    let mut cache_slices: Vec<&mut [KVCache]> = vec![&mut [], &mut []];
    let batched = model
        .try_forward_batched_with_ids(&input_ids, Some(&[seq0, seq1]), &mut cache_slices, None)
        .expect("a served batched forward is Ok");
    assert_eq!(mlxcel_core::array_shape(&batched), vec![2, 1, 2]);

    assert_eq!(log.single_forwards.lock().unwrap().as_slice(), &[1]);
    assert_eq!(
        log.last_row_forwards.lock().unwrap().as_slice(),
        &[(1, 1)],
        "the prefill takes the last-row hook, not the all-positions one"
    );
    assert_eq!(
        log.batched_forwards.lock().unwrap().as_slice(),
        &[vec![1, 2]]
    );
}

/// A runtime forward failure (a stage `Error`, a transport error or a
/// timeout) is an `Err` carrying only the fixed client-safe message, on all
/// three fallible forwards (#2727 item 1). The stage's own text, with its
/// address, goes to the log and not to the caller.
#[test]
fn pipeline_server_model_reports_forward_failure_through_try_forward() {
    let log = Arc::new(RuntimeLog::default());
    let model = model_with(FakeRuntime {
        log: log.clone(),
        fail_forward: true,
        ..FakeRuntime::default()
    });
    let seq0 = SequenceId::from_raw(1);
    let seq1 = SequenceId::from_raw(2);
    let input_ids = from_slice_i32(&[1, 2], &[1, 2]);
    let mut caches = Vec::<KVCache>::new();

    let err = model
        .try_forward_with_sequence_id(&input_ids, Some(seq0), &mut caches, None)
        .err()
        .expect("a failed forward must be an error");
    assert_eq!(err, PIPELINE_FORWARD_FAILED);
    assert_client_safe(&err);

    let err = model
        .try_forward_last_logits_with_sequence_id(&input_ids, Some(seq0), &mut caches, None, 1)
        .err()
        .expect("a failed prefill forward must be an error");
    assert_eq!(err, PIPELINE_FORWARD_FAILED);
    assert_client_safe(&err);

    let mut cache_slices: Vec<&mut [KVCache]> = vec![&mut [], &mut []];
    let err = model
        .try_forward_batched_with_ids(&input_ids, Some(&[seq0, seq1]), &mut cache_slices, None)
        .err()
        .expect("a failed batched forward must be an error");
    assert_eq!(err, PIPELINE_FORWARD_FAILED);
    assert_client_safe(&err);

    assert!(log.single_forwards.lock().unwrap().is_empty());
    assert!(log.batched_forwards.lock().unwrap().is_empty());
}

/// The fallible hook carries a runtime's prepare failure to the engine
/// (which fails that open) instead of panicking (#2714 item a), as the
/// fixed client-safe message rather than the stage's text (#2727 item 2).
#[test]
fn pipeline_server_model_reports_prepare_failure_through_try_prepare() {
    let log = Arc::new(RuntimeLog::default());
    let model = model_with(FakeRuntime {
        log: log.clone(),
        refuse_prepare: true,
        ..FakeRuntime::default()
    });
    let err = model
        .try_prepare_sequence_state(SequenceId::from_raw(9))
        .expect_err("a refused prepare must be an error");
    assert_eq!(err, PIPELINE_PREPARE_REFUSED);
    assert!(err.starts_with("model pipeline could not prepare"), "{err}");
    assert!(!err.contains("Server busy"), "{err}");
    assert_client_safe(&err);
    assert!(log.prepared.lock().unwrap().is_empty());

    let model = model_with(FakeRuntime::default());
    model
        .try_prepare_sequence_state(SequenceId::from_raw(9))
        .expect("an accepted prepare is Ok");
}

#[test]
fn pipeline_server_model_reports_its_per_row_schedule() {
    // Both runtimes split the batch into one PipelineWorkerInput per row;
    // each stage therefore runs one forward per row rather than one
    // weight-sharing graph over B rows. The marker must keep the trait's
    // fallback classification until that schedule changes.
    let model = PipelineServerModel::new(1, vec![2], Box::new(FakeRuntime::default()));
    assert_eq!(
        model.batched_decode_kind(),
        mlxcel_core::generate::BatchedDecodeKind::PerRowFallback
    );
}

/// #3074: a prefill reaches the runtime's last-row hook and the single row it
/// answers is the sampled one; the fallible and infallible hooks agree.
#[test]
fn pipeline_server_model_prefill_requests_only_the_sampled_row() {
    let log = Arc::new(RuntimeLog::default());
    let model = model_with(FakeRuntime {
        log: log.clone(),
        ..FakeRuntime::default()
    });
    let ids = from_slice_i32(&[1, 2, 3], &[1, 3]);
    let seq_id = SequenceId::from_raw(5);

    let row = model
        .try_forward_last_logits_with_sequence_id(&ids, Some(seq_id), &mut [], None, 2)
        .expect("prefill forward");
    assert_eq!(mlxcel_core::array_shape(&row), vec![1, 1, 2]);
    let row = model.forward_last_logits_with_sequence_id(&ids, Some(seq_id), &mut [], None, 2);
    assert_eq!(mlxcel_core::array_shape(&row), vec![1, 1, 2]);

    assert_eq!(
        log.last_row_forwards.lock().unwrap().as_slice(),
        &[(5, 2), (5, 2)]
    );
    assert!(
        log.single_forwards.lock().unwrap().is_empty(),
        "a prefill must not take the all-positions path"
    );
}

/// A runtime without the hook (the trait default, as with an older stage)
/// still yields the sampled row: the model slices it out of the full answer.
#[test]
fn pipeline_server_model_slices_the_row_when_the_runtime_answers_every_position() {
    struct FullOnly;
    impl PipelineModelRuntime for FullOnly {
        fn prepare_sequence_state(&self, _seq_id: SequenceId) -> anyhow::Result<()> {
            Ok(())
        }
        fn release_sequence_state_by_id(&self, _seq_id: SequenceId) {}
        fn forward_sequence(
            &self,
            _seq_id: SequenceId,
            _input_ids: &MlxArray,
            _mask: Option<&MlxArray>,
        ) -> anyhow::Result<UniquePtr<MlxArray>> {
            Ok(from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]))
        }
        fn forward_batched(
            &self,
            _seq_ids: &[SequenceId],
            _input_ids: &MlxArray,
        ) -> anyhow::Result<UniquePtr<MlxArray>> {
            unreachable!("no batched decode in this test")
        }
    }
    let model = PipelineServerModel::new(2, vec![2], Box::new(FullOnly));
    let ids = from_slice_i32(&[1, 2, 3], &[1, 3]);
    let row = model
        .try_forward_last_logits_with_sequence_id(
            &ids,
            Some(SequenceId::from_raw(1)),
            &mut [],
            None,
            1,
        )
        .expect("prefill forward");
    assert_eq!(mlxcel_core::array_shape(&row), vec![1, 1, 2]);
    let want = from_slice_f32(&[3.0, 4.0], &[1, 1, 2]);
    assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
        &row, &want, 0.0, 0.0
    )));
}

/// The failure mapping of the full forward applies to the last-row one.
#[test]
fn pipeline_server_model_last_row_failure_is_client_safe() {
    let model = model_with(FakeRuntime {
        fail_forward: true,
        ..FakeRuntime::default()
    });
    let ids = from_slice_i32(&[1, 2, 3], &[1, 3]);
    let Err(err) = model.try_forward_last_logits_with_sequence_id(
        &ids,
        Some(SequenceId::from_raw(3)),
        &mut [],
        None,
        2,
    ) else {
        panic!("a failing stage fails the prefill");
    };
    assert_eq!(err, PIPELINE_FORWARD_FAILED);
    assert_client_safe(&err);
}
