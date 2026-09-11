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

use anyhow::{Result, bail};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::distributed::pipeline::{
    ChannelConfig, InProcessStageWorkerLoop, LayerFilter, PipelineConfig, PipelineWorkerInput,
    StageAssignment, StageExecutionInput, StageExecutionOutput, StageExecutor,
};
use crate::distributed::request_tracker::RequestId;

struct FakeStageExecutor {
    stage: StageAssignment,
    filter: LayerFilter,
    kind: FakeStageKind,
}

enum FakeStageKind {
    Entry,
    Final,
}

impl FakeStageExecutor {
    fn new(stage_index: usize, kind: FakeStageKind) -> Self {
        let stage = StageAssignment {
            stage_index,
            device_id: format!("stage-{stage_index}"),
            layer_range: stage_index..stage_index + 1,
            has_embedding: stage_index == 0,
            has_lm_head: matches!(kind, FakeStageKind::Final),
            estimated_memory_bytes: 0,
        };
        let filter = LayerFilter::from_stage(&stage);
        Self {
            stage,
            filter,
            kind,
        }
    }
}

impl StageExecutor for FakeStageExecutor {
    fn stage_assignment(&self) -> &StageAssignment {
        &self.stage
    }

    fn layer_filter(&self) -> &LayerFilter {
        &self.filter
    }

    fn make_caches(&self) -> Vec<KVCache> {
        match self.kind {
            FakeStageKind::Entry => Vec::new(),
            FakeStageKind::Final => vec![KVCache::new()],
        }
    }

    fn release_caches(&self, _caches: &[KVCache]) {}

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        match self.kind {
            FakeStageKind::Entry => {
                let input_ids = match input {
                    StageExecutionInput::TokenIds(ids) => ids,
                    StageExecutionInput::HiddenStates(_) => {
                        bail!("entry fake executor expects token ids")
                    }
                };
                let values = i32_values(input_ids)?;
                Ok(StageExecutionOutput::HiddenStates(
                    mlxcel_core::from_slice_f32(
                        &values.iter().map(|v| *v as f32).collect::<Vec<_>>(),
                        &mlxcel_core::array_shape(input_ids),
                    ),
                ))
            }
            FakeStageKind::Final => {
                let hidden = match input {
                    StageExecutionInput::HiddenStates(hidden) => hidden,
                    StageExecutionInput::TokenIds(_) => {
                        bail!("final fake executor expects hidden states")
                    }
                };
                let cache = caches
                    .first_mut()
                    .ok_or_else(|| anyhow::anyhow!("final fake executor requires one cache"))?;
                cache.update_and_fetch(
                    mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1, 1]),
                    mlxcel_core::from_slice_f32(&[1.0], &[1, 1, 1, 1]),
                );
                let offset = cache.seq_len() as f32;
                let values = f32_values(hidden)?
                    .into_iter()
                    .map(|value| value + offset)
                    .collect::<Vec<_>>();
                Ok(StageExecutionOutput::Logits(mlxcel_core::from_slice_f32(
                    &values,
                    &mlxcel_core::array_shape(hidden),
                )))
            }
        }
    }
}

#[test]
fn stage_worker_loop_preserves_input_order_across_micro_batches() {
    let config = PipelineConfig::new(2, 1).unwrap();
    let executors: Vec<Box<dyn StageExecutor>> = vec![
        Box::new(FakeStageExecutor::new(0, FakeStageKind::Entry)),
        Box::new(FakeStageExecutor::new(1, FakeStageKind::Final)),
    ];
    let mut loop_runtime =
        InProcessStageWorkerLoop::new(config, executors, ChannelConfig::default()).unwrap();

    let outputs = loop_runtime
        .run_to_completion(vec![
            PipelineWorkerInput::new(
                RequestId::from_string("req-a".to_string()).unwrap(),
                mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]),
            ),
            PipelineWorkerInput::new(
                RequestId::from_string("req-b".to_string()).unwrap(),
                mlxcel_core::from_slice_i32(&[4, 5], &[1, 2]),
            ),
        ])
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].request_id.as_str(), "req-a");
    assert_eq!(outputs[1].request_id.as_str(), "req-b");
    assert_close(
        outputs[0].logits.as_ref().unwrap(),
        &mlxcel_core::from_slice_f32(&[2.0, 3.0, 4.0], &[1, 3]),
    );
    assert_close(
        outputs[1].logits.as_ref().unwrap(),
        &mlxcel_core::from_slice_f32(&[5.0, 6.0], &[1, 2]),
    );
}

#[test]
fn stage_worker_loop_retains_request_scoped_caches_across_runs() {
    let config = PipelineConfig::new(2, 1).unwrap();
    let executors: Vec<Box<dyn StageExecutor>> = vec![
        Box::new(FakeStageExecutor::new(0, FakeStageKind::Entry)),
        Box::new(FakeStageExecutor::new(1, FakeStageKind::Final)),
    ];
    let mut loop_runtime =
        InProcessStageWorkerLoop::new(config, executors, ChannelConfig::default()).unwrap();

    let first = loop_runtime
        .run_to_completion(vec![PipelineWorkerInput::new(
            RequestId::from_string("shared".to_string()).unwrap(),
            mlxcel_core::from_slice_i32(&[7], &[1, 1]),
        )])
        .unwrap();
    let second = loop_runtime
        .run_to_completion(vec![PipelineWorkerInput::new(
            RequestId::from_string("shared".to_string()).unwrap(),
            mlxcel_core::from_slice_i32(&[7], &[1, 1]),
        )])
        .unwrap();
    let fresh = loop_runtime
        .run_to_completion(vec![PipelineWorkerInput::new(
            RequestId::from_string("fresh".to_string()).unwrap(),
            mlxcel_core::from_slice_i32(&[7], &[1, 1]),
        )])
        .unwrap();

    assert_close(
        first[0].logits.as_ref().unwrap(),
        &mlxcel_core::from_slice_f32(&[8.0], &[1, 1]),
    );
    assert_close(
        second[0].logits.as_ref().unwrap(),
        &mlxcel_core::from_slice_f32(&[9.0], &[1, 1]),
    );
    assert_close(
        fresh[0].logits.as_ref().unwrap(),
        &mlxcel_core::from_slice_f32(&[8.0], &[1, 1]),
    );
}

fn assert_close(actual: &MlxArray, expected: &UniquePtr<MlxArray>) {
    let close = mlxcel_core::allclose(actual, expected.as_ref().unwrap(), 1e-5, 1e-5);
    assert!(mlxcel_core::item_bool(&close));
}

fn i32_values(arr: &MlxArray) -> Result<Vec<i32>> {
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn f32_values(arr: &MlxArray) -> Result<Vec<f32>> {
    let bytes = mlxcel_core::array_to_raw_bytes(arr);
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
