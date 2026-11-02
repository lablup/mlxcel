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
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::runtime::{
    InProcessPipelineRuntime, PipelineModelRuntime, RemotePipelineRuntime,
    RemotePipelineRuntimeConfig,
};
use super::stage_availability::PipelineUnavailable;

/// What a request is told when a stage fails its forward (an `Error` reply,
/// a transport error or a stage timeout; #2727). The stage's own error,
/// which names its address and the remote sequence id, goes to the server
/// log at `warn` and never to a client.
pub(crate) const PIPELINE_FORWARD_FAILED: &str =
    "model pipeline forward failed; see the server log for the stage error";

/// What a request is told when a stage refuses to prepare it (#2714,
/// #2727). It is not a capacity condition, so it carries no `Server busy`
/// label, and like [`PIPELINE_FORWARD_FAILED`] it names no stage.
pub(crate) const PIPELINE_PREPARE_REFUSED: &str =
    "model pipeline could not prepare this request; see the server log for the stage error";

/// Whether `err` is the availability gate's refusal (#2782): a stage is
/// known silent and the request was refused without reaching it. The
/// gate's own open and close lines carry the operator signal, so these
/// refusals are logged at debug instead of one warn per request.
fn is_unavailable(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<PipelineUnavailable>().is_some())
}

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
    /// stage initialization; runtime hot-swap is out of scope (v1).
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
        mut config: RemotePipelineRuntimeConfig,
    ) -> anyhow::Result<Self> {
        let num_layers = super::resolve_in_process_pipeline_num_layers(model_dir)?;
        let eos_token_ids = crate::read_eos_token_ids(model_dir);
        // The coordinator validates every returned logits tensor against the
        // model's vocabulary (#2637); fill it from config.json when the
        // caller did not.
        if config.vocab_size.is_none() {
            config.vocab_size = crate::read_vocab_size(model_dir);
        }
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

    /// The engine opens pipeline sequences through
    /// [`LanguageModel::try_prepare_sequence_state`]; this infallible hook is
    /// kept for callers that have no error path and keeps the fail-fast
    /// behavior of earlier builds for them.
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        if let Err(err) = self.runtime.prepare_sequence_state(seq_id) {
            panic!("pipeline prepare failed for sequence {seq_id}: {err:#}");
        }
    }

    /// A stage that refuses the `PrepareSequence` (#2714) fails this open,
    /// not the model worker: `Engine::open` releases the pool entry and the
    /// scheduler answers the request with [`PIPELINE_PREPARE_REFUSED`]. The
    /// stage's error, with its address, is logged here (#2727).
    fn try_prepare_sequence_state(&self, seq_id: SequenceId) -> Result<(), String> {
        self.runtime.prepare_sequence_state(seq_id).map_err(|err| {
            if is_unavailable(&err) {
                tracing::debug!(
                    seq_id = %seq_id,
                    "pipeline prepare refused while a stage is silent: {err:#}"
                );
            } else {
                tracing::warn!(
                    seq_id = %seq_id,
                    "pipeline stage refused to prepare the sequence: {err:#}"
                );
            }
            PIPELINE_PREPARE_REFUSED.to_string()
        })
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.num_layers)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    /// Opt out of NA tile-aligned padded prefill (issue #1755).
    ///
    /// Each sequence's K/V lives in the pipeline stages, behind the runtime,
    /// and the runtime has no operation to rewind a sequence by a pad width.
    /// The scheduler's own trim cannot reach it (this layout is model-owned),
    /// so a padded prefill would leave every stage's offset ahead of the real
    /// token count.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    /// The engine runs pipeline forwards through the fallible hooks below;
    /// this infallible entry is kept for callers that have no error path and
    /// keeps the fail-fast behavior of earlier builds for them.
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

    /// A stage that fails the forward (an `Error` reply, a transport error or
    /// a timeout) fails only this request (#2727): the runtime has already
    /// released the sequence on every stage, the stage's error is logged
    /// here with the sequence id, and the engine receives
    /// [`PIPELINE_FORWARD_FAILED`] to answer the request with.
    fn try_forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let seq_id = seq_id.expect("pipeline server model requires a sequence id");
        self.runtime
            .forward_sequence(seq_id, input_ids, mask)
            .map_err(|err| {
                if is_unavailable(&err) {
                    tracing::debug!(
                        seq_id = %seq_id,
                        "pipeline forward refused while a stage is silent: {err:#}"
                    );
                } else {
                    tracing::warn!(
                        seq_id = %seq_id,
                        "pipeline forward failed; failing the request: {err:#}"
                    );
                }
                PIPELINE_FORWARD_FAILED.to_string()
            })
    }

    /// [`Self::try_forward_with_sequence_id`] followed by the last-position
    /// slice. A stage may legitimately return the last position only
    /// (`[1, 1, vocab]`, which the runtime's shape check accepts), in which
    /// case that position is the one to sample from.
    fn try_forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let logits = self.try_forward_with_sequence_id(input_ids, seq_id, caches, mask)?;
        let positions = mlxcel_core::array_shape(&logits)
            .get(1)
            .copied()
            .unwrap_or(1);
        let pos = if positions == 1 { 0 } else { last_pos };
        Ok(mlxcel_core::generate::logits_at_position(&logits, pos))
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

    /// Infallible counterpart of [`Self::try_forward_batched_with_ids`]; see
    /// [`Self::forward_with_sequence_id`].
    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let seq_ids = seq_ids.expect("pipeline batched decode requires sequence ids");
        self.runtime
            .forward_batched(seq_ids, input_ids)
            .unwrap_or_else(|err| panic!("pipeline batched decode failed: {err}"))
    }

    /// The batched counterpart of [`Self::try_forward_with_sequence_id`]: a
    /// failed stage forward fails every request of the batch with
    /// [`PIPELINE_FORWARD_FAILED`] (#2727).
    fn try_forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let seq_ids = seq_ids.expect("pipeline batched decode requires sequence ids");
        self.runtime
            .forward_batched(seq_ids, input_ids)
            .map_err(|err| {
                tracing::warn!(
                    seq_ids = ?seq_ids,
                    "pipeline batched decode failed; failing its requests: {err:#}"
                );
                PIPELINE_FORWARD_FAILED.to_string()
            })
    }
}

#[cfg(test)]
#[path = "server_runtime_tests.rs"]
mod tests;
