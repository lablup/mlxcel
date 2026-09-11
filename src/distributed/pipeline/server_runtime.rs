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

//! Server-side pipeline model wrapper backed by a pluggable runtime.
//!
//! Used by: server model worker, batch scheduler

use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::runtime::{
    InProcessPipelineRuntime, PipelineModelRuntime, RemotePipelineRuntime,
    RemotePipelineRuntimeConfig,
};

pub struct PipelineServerModel {
    num_layers: usize,
    eos_token_ids: Vec<i32>,
    runtime: Box<dyn PipelineModelRuntime>,
}

impl PipelineServerModel {
    pub fn new(
        num_layers: usize,
        eos_token_ids: Vec<i32>,
        runtime: Box<dyn PipelineModelRuntime>,
    ) -> Self {
        Self {
            num_layers,
            eos_token_ids,
            runtime,
        }
    }

    pub fn load_in_process(
        model_dir: &std::path::Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
    ) -> anyhow::Result<Self> {
        Self::load_in_process_with_adapter(model_dir, pp_layers, pp_micro_batch_size, None)
    }

    /// Load an in-process pipeline server model, optionally composing a LoRA
    /// adapter into each stage.
    ///
    /// This mirrors [`crate::load_model_with_adapter`] for non-PP inference:
    /// the adapter directory format, the `--adapter` CLI flag, and the
    /// rank/scaling semantics are identical. The adapter is loaded once at
    /// stage initialization — runtime hot-swap is out of scope (v1).
    pub fn load_in_process_with_adapter(
        model_dir: &std::path::Path,
        pp_layers: Option<&str>,
        pp_micro_batch_size: usize,
        adapter_path: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        let (num_layers, eos_token_ids, runtime) = InProcessPipelineRuntime::load_with_adapter(
            model_dir,
            pp_layers,
            pp_micro_batch_size,
            adapter_path,
        )?;
        Ok(Self::new(num_layers, eos_token_ids, Box::new(runtime)))
    }

    pub fn load_remote(
        model_dir: &std::path::Path,
        config: RemotePipelineRuntimeConfig,
    ) -> anyhow::Result<Self> {
        let num_layers = super::resolve_in_process_pipeline_num_layers(model_dir)?;
        let eos_token_ids = crate::read_eos_token_ids(model_dir);
        let runtime = RemotePipelineRuntime::new(config)?;
        Ok(Self::new(num_layers, eos_token_ids, Box::new(runtime)))
    }
}

impl LanguageModel for PipelineServerModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        panic!("pipeline server model requires scheduler sequence ids")
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.runtime.release_sequence_state_by_id(seq_id);
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.runtime.prepare_sequence_state(seq_id);
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.num_layers)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let seq_id = seq_id.expect("pipeline server model requires a sequence id");
        self.runtime
            .forward_sequence(seq_id, input_ids, mask)
            .unwrap_or_else(|err| panic!("pipeline forward failed for sequence {seq_id}: {err}"))
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        assert!(
            input_embeddings.is_none(),
            "pipeline server model currently supports text-only requests"
        );
        self.forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
        _context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        let seq_ids = seq_ids.expect("pipeline batched decode requires sequence ids");
        self.runtime
            .forward_batched(seq_ids, input_ids)
            .unwrap_or_else(|err| panic!("pipeline batched decode failed: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::distributed::pipeline::PipelineModelRuntime;
    use mlxcel_core::{dtype, from_slice_f32, from_slice_i32, zeros};

    #[derive(Default)]
    struct RuntimeLog {
        prepared: Mutex<Vec<u64>>,
        released: Mutex<Vec<u64>>,
        single_forwards: Mutex<Vec<u64>>,
        batched_forwards: Mutex<Vec<Vec<u64>>>,
    }

    #[derive(Default)]
    struct FakeRuntime {
        log: Arc<RuntimeLog>,
    }

    impl PipelineModelRuntime for FakeRuntime {
        fn prepare_sequence_state(&self, seq_id: SequenceId) {
            self.log.prepared.lock().unwrap().push(seq_id.as_u64());
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
            self.log
                .single_forwards
                .lock()
                .unwrap()
                .push(seq_id.as_u64());
            Ok(from_slice_f32(&[1.0, 2.0], &[1, 1, 2]))
        }

        fn forward_batched(
            &self,
            seq_ids: &[SequenceId],
            _input_ids: &MlxArray,
        ) -> anyhow::Result<UniquePtr<MlxArray>> {
            self.log
                .batched_forwards
                .lock()
                .unwrap()
                .push(seq_ids.iter().map(|seq| seq.as_u64()).collect());
            Ok(zeros(&[seq_ids.len() as i32, 1, 2], dtype::FLOAT32))
        }
    }

    #[test]
    fn pipeline_server_model_delegates_sequence_lifecycle() {
        let log = Arc::new(RuntimeLog::default());
        let runtime = FakeRuntime { log: log.clone() };
        let model = PipelineServerModel::new(2, vec![2], Box::new(runtime));
        let seq_id = SequenceId::from_raw(7);

        model.prepare_sequence_state(seq_id);
        model.release_sequence_state_by_id(seq_id);

        assert_eq!(log.prepared.lock().unwrap().as_slice(), &[7]);
        assert_eq!(log.released.lock().unwrap().as_slice(), &[7]);
    }

    #[test]
    fn pipeline_server_model_uses_runtime_for_forward_paths() {
        let log = Arc::new(RuntimeLog::default());
        let runtime = FakeRuntime { log: log.clone() };
        let model = PipelineServerModel::new(4, vec![42], Box::new(runtime));
        let seq0 = SequenceId::from_raw(1);
        let seq1 = SequenceId::from_raw(2);

        model.prepare_sequence_state(seq0);
        model.release_sequence_state_by_id(seq1);

        let input_ids = from_slice_i32(&[1, 2], &[2, 1]);
        let mut caches = Vec::<KVCache>::new();
        let _ = model.forward_with_sequence_id(
            input_ids.as_ref().unwrap(),
            Some(seq0),
            &mut caches,
            None,
        );
        let mut cache_slices: Vec<&mut [KVCache]> = vec![&mut [], &mut []];
        let _ = model.forward_batched_with_context_and_ids(
            input_ids.as_ref().unwrap(),
            Some(&[seq0, seq1]),
            &mut cache_slices,
            None,
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
}
