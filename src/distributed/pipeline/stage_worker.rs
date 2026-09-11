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

//! In-process pipeline stage worker loop.
//!
//! Phase 1 keeps execution on a single machine and uses the existing
//! activation channel abstraction to move hidden states forward and logits
//! back upstream during flush. The worker loop is generic over the stage
//! executor so model-specific logic stays inside the stage adapter.
//!
//! Used by: pipeline runtime tests, future CLI pipeline runtime, future server runtime

use std::collections::HashMap;

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::distributed::request_tracker::RequestId;

use super::activation_transfer::{
    ActivationMessage, ActivationReceiver, ActivationSender, ChannelConfig, activation_channel,
};
use super::schedule::{GPipeSchedule, PipelineConfig, PipelineSchedule, ScheduleAction};
use super::stage_executor::{StageExecutionInput, StageExecutionOutput, StageExecutor};
use super::wire_tensor::{deserialize_wire_tensor, sequence_length, serialize_mlx_array};

/// One micro-batch submitted to the in-process stage worker loop.
pub struct PipelineWorkerInput {
    pub request_id: RequestId,
    pub input_ids: UniquePtr<MlxArray>,
    pub attention_mask: Option<UniquePtr<MlxArray>>,
}

impl PipelineWorkerInput {
    pub fn new(request_id: RequestId, input_ids: UniquePtr<MlxArray>) -> Self {
        Self {
            request_id,
            input_ids,
            attention_mask: None,
        }
    }

    pub fn with_attention_mask(mut self, attention_mask: UniquePtr<MlxArray>) -> Self {
        self.attention_mask = Some(attention_mask);
        self
    }
}

/// Final output collected by the entry stage after flush.
pub struct PipelineWorkerOutput {
    pub request_id: RequestId,
    pub logits: UniquePtr<MlxArray>,
}

struct ForwardInboxEntry {
    request_id: RequestId,
    hidden_states: UniquePtr<MlxArray>,
    attention_mask: Option<UniquePtr<MlxArray>>,
}

struct CompletedLastStageOutput {
    message: ActivationMessage,
}

struct CompletedEntryStageOutput {
    request_id: RequestId,
    logits: UniquePtr<MlxArray>,
}

/// One stage-local worker in the in-process pipeline loop.
pub struct PipelineStageWorker {
    stage_index: u32,
    num_stages: u32,
    executor: Box<dyn StageExecutor>,
    caches_by_request: HashMap<String, Vec<KVCache>>,
    forward_inbox: HashMap<u32, ForwardInboxEntry>,
    completed_last_stage: HashMap<u32, CompletedLastStageOutput>,
    completed_entry_stage: HashMap<u32, CompletedEntryStageOutput>,
    incoming_forward: Option<ActivationReceiver>,
    outgoing_forward: Option<ActivationSender>,
    incoming_reverse: Option<ActivationReceiver>,
    outgoing_reverse: Option<ActivationSender>,
}

impl PipelineStageWorker {
    fn new(
        stage_index: u32,
        num_stages: u32,
        executor: Box<dyn StageExecutor>,
        incoming_forward: Option<ActivationReceiver>,
        outgoing_forward: Option<ActivationSender>,
        incoming_reverse: Option<ActivationReceiver>,
        outgoing_reverse: Option<ActivationSender>,
    ) -> Self {
        Self {
            stage_index,
            num_stages,
            executor,
            caches_by_request: HashMap::new(),
            forward_inbox: HashMap::new(),
            completed_last_stage: HashMap::new(),
            completed_entry_stage: HashMap::new(),
            incoming_forward,
            outgoing_forward,
            incoming_reverse,
            outgoing_reverse,
        }
    }

    fn release_request(&mut self, request_id: &RequestId) {
        if let Some(caches) = self.caches_by_request.remove(request_id.as_str()) {
            self.executor.release_caches(&caches);
        }
    }

    fn receive_forward(&mut self, micro_batch_id: u32) -> Result<()> {
        let receiver = self
            .incoming_forward
            .as_mut()
            .ok_or_else(|| anyhow!("stage {} has no incoming forward link", self.stage_index))?;
        let msg = receiver.try_recv().ok_or_else(|| {
            anyhow!(
                "stage {} has no pending forward activation for micro-batch {}",
                self.stage_index,
                micro_batch_id
            )
        })?;
        ensure!(
            !msg.is_reverse_path,
            "stage {} received reverse payload on forward link",
            self.stage_index
        );
        ensure!(
            msg.micro_batch_id == micro_batch_id,
            "stage {} received forward micro-batch {} while waiting for {}",
            self.stage_index,
            msg.micro_batch_id,
            micro_batch_id
        );

        let hidden_states = deserialize_wire_tensor(&msg.tensor_data)?;
        let attention_mask = msg
            .attention_mask
            .as_deref()
            .map(deserialize_wire_tensor)
            .transpose()?;

        self.forward_inbox.insert(
            micro_batch_id,
            ForwardInboxEntry {
                request_id: msg.request_id,
                hidden_states,
                attention_mask,
            },
        );
        Ok(())
    }

    fn forward(
        &mut self,
        micro_batch_id: u32,
        request: Option<&PipelineWorkerInput>,
    ) -> Result<()> {
        let mut owned_hidden = None;
        let (request_id, input_ids, owned_attention_mask) = if self.stage_index == 0 {
            let request = request.ok_or_else(|| {
                anyhow!(
                    "entry stage missing pipeline input for micro-batch {}",
                    micro_batch_id
                )
            })?;
            (
                request.request_id.clone(),
                Some(request.input_ids.as_ref().unwrap()),
                None,
            )
        } else {
            let inbox = self.forward_inbox.remove(&micro_batch_id).ok_or_else(|| {
                anyhow!(
                    "stage {} has no received activation for micro-batch {}",
                    self.stage_index,
                    micro_batch_id
                )
            })?;
            let request_id = inbox.request_id;
            owned_hidden = Some(inbox.hidden_states);
            (request_id, None, inbox.attention_mask)
        };
        let attention_mask = request
            .and_then(|request| {
                request
                    .attention_mask
                    .as_ref()
                    .and_then(|mask| mask.as_ref())
            })
            .or_else(|| owned_attention_mask.as_ref().and_then(|mask| mask.as_ref()));

        let request_key = request_id.as_str().to_string();
        let caches = self
            .caches_by_request
            .entry(request_key)
            .or_insert_with(|| self.executor.make_caches());

        let output = match owned_hidden.as_ref() {
            Some(hidden) => self.executor.execute(
                StageExecutionInput::HiddenStates(hidden.as_ref().unwrap()),
                caches,
                attention_mask,
            )?,
            None => {
                let input_ids = input_ids.ok_or_else(|| anyhow!("missing stage input ids"))?;
                self.executor.execute(
                    StageExecutionInput::TokenIds(input_ids),
                    caches,
                    attention_mask,
                )?
            }
        };

        match output {
            StageExecutionOutput::HiddenStates(hidden_states) => {
                let sender = self.outgoing_forward.as_ref().ok_or_else(|| {
                    anyhow!(
                        "stage {} produced hidden states without a downstream forward link",
                        self.stage_index
                    )
                })?;
                let seq_len = sequence_length(hidden_states.as_ref().unwrap())?;
                let msg = ActivationMessage::forward(
                    request_id,
                    micro_batch_id,
                    self.stage_index,
                    self.num_stages,
                    serialize_mlx_array(hidden_states.as_ref().unwrap())?,
                    attention_mask.map(serialize_mlx_array).transpose()?,
                    None,
                    seq_len,
                );
                sender.try_send(msg)?;
            }
            StageExecutionOutput::Logits(logits) => {
                let seq_len = sequence_length(logits.as_ref().unwrap())?;
                let msg = ActivationMessage::reverse(
                    request_id,
                    micro_batch_id,
                    self.stage_index,
                    self.num_stages,
                    serialize_mlx_array(logits.as_ref().unwrap())?,
                    seq_len,
                );
                self.completed_last_stage
                    .insert(micro_batch_id, CompletedLastStageOutput { message: msg });
            }
        }

        Ok(())
    }

    fn begin_reverse_flush(&mut self, micro_batch_id: u32) -> Result<()> {
        let completed = self
            .completed_last_stage
            .remove(&micro_batch_id)
            .ok_or_else(|| {
                anyhow!(
                    "last stage {} has no completed output for micro-batch {}",
                    self.stage_index,
                    micro_batch_id
                )
            })?;
        let sender = self.outgoing_reverse.as_ref().ok_or_else(|| {
            anyhow!(
                "last stage {} has no upstream reverse link",
                self.stage_index
            )
        })?;
        sender.try_send(completed.message)?;
        Ok(())
    }

    fn relay_reverse(&mut self, micro_batch_id: u32) -> Result<()> {
        let receiver = self
            .incoming_reverse
            .as_mut()
            .ok_or_else(|| anyhow!("stage {} has no incoming reverse link", self.stage_index))?;
        let msg = receiver.try_recv().ok_or_else(|| {
            anyhow!(
                "stage {} has no pending reverse activation for micro-batch {}",
                self.stage_index,
                micro_batch_id
            )
        })?;
        ensure!(
            msg.is_reverse_path,
            "stage {} received forward payload on reverse link",
            self.stage_index
        );
        ensure!(
            msg.micro_batch_id == micro_batch_id,
            "stage {} received reverse micro-batch {} while waiting for {}",
            self.stage_index,
            msg.micro_batch_id,
            micro_batch_id
        );

        if let Some(sender) = self.outgoing_reverse.as_ref() {
            sender.try_send(msg)?;
        } else {
            let logits = deserialize_wire_tensor(&msg.tensor_data)?;
            self.completed_entry_stage.insert(
                micro_batch_id,
                CompletedEntryStageOutput {
                    request_id: msg.request_id,
                    logits,
                },
            );
        }
        Ok(())
    }

    fn take_completed_entry_output(
        &mut self,
        micro_batch_id: u32,
    ) -> Result<CompletedEntryStageOutput> {
        self.completed_entry_stage
            .remove(&micro_batch_id)
            .ok_or_else(|| {
                anyhow!(
                    "entry stage has no completed flushed output for micro-batch {}",
                    micro_batch_id
                )
            })
    }
}

/// In-process pipeline worker loop that executes a schedule against real stage executors.
pub struct InProcessStageWorkerLoop {
    config: PipelineConfig,
    workers: Vec<PipelineStageWorker>,
}

impl InProcessStageWorkerLoop {
    pub fn new(
        config: PipelineConfig,
        executors: Vec<Box<dyn StageExecutor>>,
        channel_config: ChannelConfig,
    ) -> Result<Self> {
        ensure!(
            config.num_stages as usize == executors.len(),
            "pipeline config expects {} stages but {} executors were provided",
            config.num_stages,
            executors.len()
        );

        let link_count = executors.len().saturating_sub(1);
        let mut forward_receivers = Vec::with_capacity(link_count);
        let mut forward_senders = Vec::with_capacity(link_count);
        let mut reverse_receivers = Vec::with_capacity(link_count);
        let mut reverse_senders = Vec::with_capacity(link_count);

        for idx in 0..link_count {
            let (fwd_tx, fwd_rx) = activation_channel(
                format!("stage-{idx}->stage-{}", idx + 1),
                channel_config.clone(),
            );
            let (rev_tx, rev_rx) = activation_channel(
                format!("stage-{}->stage-{idx}", idx + 1),
                channel_config.clone(),
            );
            forward_senders.push(fwd_tx);
            forward_receivers.push(Some(fwd_rx));
            reverse_senders.push(rev_tx);
            reverse_receivers.push(Some(rev_rx));
        }

        let workers = executors
            .into_iter()
            .enumerate()
            .map(|(idx, executor)| {
                PipelineStageWorker::new(
                    idx as u32,
                    config.num_stages,
                    executor,
                    if idx == 0 {
                        None
                    } else {
                        forward_receivers[idx - 1].take()
                    },
                    if idx + 1 == config.num_stages as usize {
                        None
                    } else {
                        Some(forward_senders[idx].clone())
                    },
                    if idx + 1 == config.num_stages as usize {
                        None
                    } else {
                        reverse_receivers[idx].take()
                    },
                    if idx == 0 {
                        None
                    } else {
                        Some(reverse_senders[idx - 1].clone())
                    },
                )
            })
            .collect();

        Ok(Self { config, workers })
    }

    pub fn release_request(&mut self, request_id: &RequestId) {
        for worker in &mut self.workers {
            worker.release_request(request_id);
        }
    }

    /// Execute one GPipe step to completion.
    ///
    /// Cache state is retained inside the workers and keyed by `RequestId`,
    /// so callers may run prompt prefill and subsequent decode steps through
    /// the same loop instance using the same request identifiers.
    pub fn run_to_completion(
        &mut self,
        inputs: Vec<PipelineWorkerInput>,
    ) -> Result<Vec<PipelineWorkerOutput>> {
        ensure!(!inputs.is_empty(), "must submit at least one micro-batch");
        let mut schedule = GPipeSchedule::new(self.config.clone(), inputs.len() as u32)?;
        let input_by_mb: HashMap<u32, PipelineWorkerInput> = inputs
            .into_iter()
            .enumerate()
            .map(|(idx, input)| (idx as u32, input))
            .collect();

        let last_stage = self.workers.len().saturating_sub(1);
        let mut outputs = Vec::new();

        loop {
            match schedule.next_action() {
                ScheduleAction::Forward {
                    stage_index,
                    micro_batch_id,
                } => {
                    let request = if stage_index == 0 {
                        Some(input_by_mb.get(&micro_batch_id).ok_or_else(|| {
                            anyhow!("missing input for micro-batch {}", micro_batch_id)
                        })?)
                    } else {
                        None
                    };
                    self.workers[stage_index as usize].forward(micro_batch_id, request)?;
                    schedule.notify_forward_complete(stage_index, micro_batch_id);
                }
                ScheduleAction::Receive {
                    stage_index,
                    micro_batch_id,
                } => {
                    self.workers[stage_index as usize].receive_forward(micro_batch_id)?;
                }
                ScheduleAction::Flush { micro_batch_id } => {
                    self.workers[last_stage].begin_reverse_flush(micro_batch_id)?;
                    for stage_index in (0..last_stage).rev() {
                        self.workers[stage_index].relay_reverse(micro_batch_id)?;
                    }
                    let completed = self.workers[0].take_completed_entry_output(micro_batch_id)?;
                    outputs.push(PipelineWorkerOutput {
                        request_id: completed.request_id,
                        logits: completed.logits,
                    });
                    schedule.notify_flush_complete(micro_batch_id);
                }
                ScheduleAction::Idle => {
                    if schedule.is_complete() {
                        continue;
                    }
                }
                ScheduleAction::Done => break,
            }
        }

        ensure!(
            outputs.len() == input_by_mb.len(),
            "worker loop produced {} outputs for {} inputs",
            outputs.len(),
            input_by_mb.len()
        );

        Ok(outputs)
    }
}

#[cfg(test)]
#[path = "stage_worker_tests.rs"]
mod tests;
