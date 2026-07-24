// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Bounded host audio preprocessing outside the XLA scheduling loop.
//!
//! This stage owns encoded request buffers, waveform decode/resampling, and a
//! future family feature producer. It intentionally has no production
//! Phi4MM/Gemma3n producer yet; therefore XLA admission remains audio-false.

use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use mlxcel_core::session::{OwnedTensor, PreparedPositions, PreparedPrefill};
use thiserror::Error;

use super::observability::BatchObservability;
use crate::audio::{
    AudioEncodedClip, AudioFamilyPolicy, AudioPreprocessCheckpoint, AudioPreprocessError,
    AudioWaveformBatch, preprocess_wav_batch,
};

pub(crate) struct AudioPreprocessJob {
    pub job_id: u64,
    pub token_ids: Vec<i32>,
    pub max_prefill_tokens: usize,
    pub clips: Vec<AudioEncodedClip>,
    pub policy: AudioFamilyPolicy,
    pub cancelled: Arc<AtomicBool>,
}

pub(crate) enum AudioPreprocessOutcome {
    Prepared(PreparedPrefill),
    Cancelled(AudioPreprocessCheckpoint),
    Failed(AudioStageError),
}

pub(crate) struct AudioPreprocessResult {
    pub job_id: u64,
    pub outcome: AudioPreprocessOutcome,
    _reservation: HostMemoryReservation,
}

#[derive(Debug, Error)]
pub(crate) enum AudioStageError {
    #[error(transparent)]
    Waveform(#[from] AudioPreprocessError),
    #[error("audio feature preprocessing failed for {family}: {reason}")]
    Feature {
        family: &'static str,
        reason: String,
    },
    #[error("audio feature preprocessing panicked for {family}")]
    FeaturePanic { family: &'static str },
    #[error(
        "audio prepared-prefill context limit exceeded for {family}: actual {actual}, maximum {maximum}"
    )]
    ContextLimit {
        family: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error(
        "audio prepared-prefill host-memory limit exceeded for {family}: actual {actual} bytes, maximum {maximum} bytes"
    )]
    ResultMemoryLimit {
        family: &'static str,
        actual: usize,
        maximum: usize,
    },
}

#[derive(Debug, Error)]
pub(crate) enum AudioQueueError {
    #[error("audio preprocessing was cancelled before queue admission")]
    Cancelled {
        checkpoint: AudioPreprocessCheckpoint,
    },
    #[error("audio preprocessing queue is full")]
    Full,
    #[error("audio preprocessing worker is unavailable")]
    Disconnected,
    #[error(
        "audio queued-memory limit exceeded: request {request_bytes} bytes, queued {queued_bytes}, maximum {maximum_bytes}"
    )]
    MemoryLimit {
        request_bytes: usize,
        queued_bytes: usize,
        maximum_bytes: usize,
    },
    #[error(
        "audio in-flight host-memory limit exceeded: request reservation {request_bytes} bytes, in flight {in_flight_bytes}, maximum {maximum_bytes}"
    )]
    HostMemoryLimit {
        request_bytes: usize,
        in_flight_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("audio queued-memory size calculation overflowed")]
    Overflow,
}

pub(crate) trait AudioFeatureProducer: Send + 'static {
    fn prepare(
        &mut self,
        waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        cancelled: &AtomicBool,
    ) -> Result<PreparedPrefill, String>;
}

pub(crate) struct AudioPreprocessLimits {
    pub queue_depth: usize,
    pub result_queue_depth: usize,
    pub max_queued_encoded_bytes: usize,
    pub max_in_flight_host_bytes: usize,
}

impl Default for AudioPreprocessLimits {
    fn default() -> Self {
        Self {
            queue_depth: 8,
            result_queue_depth: 2,
            max_queued_encoded_bytes: 512 * 1024 * 1024,
            // One maximum-size Phi4MM request reserves 1 GiB across encoded,
            // waveform-working, and prepared-result policy budgets. Leave a
            // bounded 256 MiB envelope margin for Vec capacities and DTO
            // metadata so that exact family maxima remain admissible.
            max_in_flight_host_bytes: 1280 * 1024 * 1024,
        }
    }
}

pub(crate) struct AudioPreprocessMetrics {
    accepted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    rejected: AtomicU64,
    source_duration_micros: AtomicU64,
    source_samples: AtomicU64,
    normalized_samples: AtomicU64,
    feature_frames: AtomicU64,
    effective_audio_tokens: AtomicU64,
    effective_prefill_tokens: AtomicU64,
    preprocessing_latency_micros: AtomicU64,
    queued_encoded_bytes: AtomicUsize,
    in_flight_host_bytes: AtomicUsize,
    reject_queue_full: AtomicU64,
    reject_memory_limit: AtomicU64,
    reject_worker_unavailable: AtomicU64,
    reject_overflow: AtomicU64,
    reject_waveform: AtomicU64,
    reject_feature: AtomicU64,
    reject_feature_panic: AtomicU64,
    reject_context_limit: AtomicU64,
    observability: Arc<BatchObservability>,
}

impl AudioPreprocessMetrics {
    fn new(observability: Arc<BatchObservability>) -> Self {
        Self {
            accepted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            source_duration_micros: AtomicU64::new(0),
            source_samples: AtomicU64::new(0),
            normalized_samples: AtomicU64::new(0),
            feature_frames: AtomicU64::new(0),
            effective_audio_tokens: AtomicU64::new(0),
            effective_prefill_tokens: AtomicU64::new(0),
            preprocessing_latency_micros: AtomicU64::new(0),
            queued_encoded_bytes: AtomicUsize::new(0),
            in_flight_host_bytes: AtomicUsize::new(0),
            reject_queue_full: AtomicU64::new(0),
            reject_memory_limit: AtomicU64::new(0),
            reject_worker_unavailable: AtomicU64::new(0),
            reject_overflow: AtomicU64::new(0),
            reject_waveform: AtomicU64::new(0),
            reject_feature: AtomicU64::new(0),
            reject_feature_panic: AtomicU64::new(0),
            reject_context_limit: AtomicU64::new(0),
            observability,
        }
    }

    pub fn snapshot(&self) -> AudioPreprocessMetricsSnapshot {
        AudioPreprocessMetricsSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            source_duration_micros: self.source_duration_micros.load(Ordering::Relaxed),
            source_samples: self.source_samples.load(Ordering::Relaxed),
            normalized_samples: self.normalized_samples.load(Ordering::Relaxed),
            feature_frames: self.feature_frames.load(Ordering::Relaxed),
            effective_audio_tokens: self.effective_audio_tokens.load(Ordering::Relaxed),
            effective_prefill_tokens: self.effective_prefill_tokens.load(Ordering::Relaxed),
            preprocessing_latency_micros: self.preprocessing_latency_micros.load(Ordering::Relaxed),
            queued_encoded_bytes: self.queued_encoded_bytes.load(Ordering::Acquire),
            in_flight_host_bytes: self.in_flight_host_bytes.load(Ordering::Acquire),
            reject_queue_full: self.reject_queue_full.load(Ordering::Relaxed),
            reject_memory_limit: self.reject_memory_limit.load(Ordering::Relaxed),
            reject_worker_unavailable: self.reject_worker_unavailable.load(Ordering::Relaxed),
            reject_overflow: self.reject_overflow.load(Ordering::Relaxed),
            reject_waveform: self.reject_waveform.load(Ordering::Relaxed),
            reject_feature: self.reject_feature.load(Ordering::Relaxed),
            reject_feature_panic: self.reject_feature_panic.load(Ordering::Relaxed),
            reject_context_limit: self.reject_context_limit.load(Ordering::Relaxed),
        }
    }

    fn record_waveforms(&self, waveforms: &AudioWaveformBatch) {
        self.source_duration_micros
            .fetch_add(waveforms.total_source_duration_micros, Ordering::Relaxed);
        self.observability
            .audio_source_duration_micros
            .fetch_add(waveforms.total_source_duration_micros, Ordering::Relaxed);
        self.source_samples
            .fetch_add(waveforms.total_source_samples as u64, Ordering::Relaxed);
        self.observability
            .audio_source_samples
            .fetch_add(waveforms.total_source_samples as u64, Ordering::Relaxed);
        self.normalized_samples
            .fetch_add(waveforms.total_samples as u64, Ordering::Relaxed);
        self.observability
            .audio_normalized_samples
            .fetch_add(waveforms.total_samples as u64, Ordering::Relaxed);
        self.feature_frames
            .fetch_add(waveforms.estimated_frames as u64, Ordering::Relaxed);
        self.observability
            .audio_feature_frames
            .fetch_add(waveforms.estimated_frames as u64, Ordering::Relaxed);
        self.effective_audio_tokens
            .fetch_add(waveforms.effective_audio_tokens as u64, Ordering::Relaxed);
        self.observability
            .audio_effective_tokens
            .fetch_add(waveforms.effective_audio_tokens as u64, Ordering::Relaxed);
    }

    fn record_effective_prefill(&self, sequence_len: usize) {
        self.effective_prefill_tokens
            .fetch_add(sequence_len as u64, Ordering::Relaxed);
        self.observability
            .audio_effective_prefill_tokens
            .fetch_add(sequence_len as u64, Ordering::Relaxed);
    }

    fn record_rejection(&self, reason: AudioRejectionReason) {
        self.rejected.fetch_add(1, Ordering::Relaxed);
        self.observability
            .audio_preprocess_rejections
            .fetch_add(1, Ordering::Relaxed);
        let (local, shared) = match reason {
            AudioRejectionReason::QueueFull => (
                &self.reject_queue_full,
                &self.observability.audio_reject_queue_full,
            ),
            AudioRejectionReason::MemoryLimit => (
                &self.reject_memory_limit,
                &self.observability.audio_reject_memory_limit,
            ),
            AudioRejectionReason::WorkerUnavailable => (
                &self.reject_worker_unavailable,
                &self.observability.audio_reject_worker_unavailable,
            ),
            AudioRejectionReason::Overflow => (
                &self.reject_overflow,
                &self.observability.audio_reject_overflow,
            ),
            AudioRejectionReason::Waveform => (
                &self.reject_waveform,
                &self.observability.audio_reject_waveform,
            ),
            AudioRejectionReason::Feature => (
                &self.reject_feature,
                &self.observability.audio_reject_feature,
            ),
            AudioRejectionReason::FeaturePanic => (
                &self.reject_feature_panic,
                &self.observability.audio_reject_feature_panic,
            ),
            AudioRejectionReason::ContextLimit => (
                &self.reject_context_limit,
                &self.observability.audio_reject_context_limit,
            ),
        };
        local.fetch_add(1, Ordering::Relaxed);
        shared.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioRejectionReason {
    QueueFull,
    MemoryLimit,
    WorkerUnavailable,
    Overflow,
    Waveform,
    Feature,
    FeaturePanic,
    ContextLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AudioPreprocessMetricsSnapshot {
    pub accepted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub rejected: u64,
    pub source_duration_micros: u64,
    pub source_samples: u64,
    pub normalized_samples: u64,
    pub feature_frames: u64,
    pub effective_audio_tokens: u64,
    pub effective_prefill_tokens: u64,
    pub preprocessing_latency_micros: u64,
    pub queued_encoded_bytes: usize,
    pub in_flight_host_bytes: usize,
    pub reject_queue_full: u64,
    pub reject_memory_limit: u64,
    pub reject_worker_unavailable: u64,
    pub reject_overflow: u64,
    pub reject_waveform: u64,
    pub reject_feature: u64,
    pub reject_feature_panic: u64,
    pub reject_context_limit: u64,
}

struct HostMemoryReservation {
    encoded_bytes: usize,
    host_bytes: usize,
    metrics: Arc<AudioPreprocessMetrics>,
}

impl Drop for HostMemoryReservation {
    fn drop(&mut self) {
        self.metrics
            .queued_encoded_bytes
            .fetch_sub(self.encoded_bytes, Ordering::AcqRel);
        self.metrics
            .in_flight_host_bytes
            .fetch_sub(self.host_bytes, Ordering::AcqRel);
        self.metrics
            .observability
            .audio_preprocess_queued_bytes
            .store(
                self.metrics.queued_encoded_bytes.load(Ordering::Acquire),
                Ordering::Release,
            );
        self.metrics
            .observability
            .audio_preprocess_inflight_host_bytes
            .store(
                self.metrics.in_flight_host_bytes.load(Ordering::Acquire),
                Ordering::Release,
            );
    }
}

struct QueuedJob {
    job: Option<AudioPreprocessJob>,
    reservation: Option<HostMemoryReservation>,
}

impl QueuedJob {
    fn dequeue(mut self) -> Option<(AudioPreprocessJob, HostMemoryReservation)> {
        self.job.take().zip(self.reservation.take())
    }
}

pub(crate) struct AudioPreprocessStage {
    sender: Option<mpsc::SyncSender<QueuedJob>>,
    result_rx: Option<mpsc::Receiver<AudioPreprocessResult>>,
    metrics: Arc<AudioPreprocessMetrics>,
    healthy: Arc<AtomicBool>,
    max_queued_encoded_bytes: usize,
    max_in_flight_host_bytes: usize,
    worker: Option<thread::JoinHandle<()>>,
}

impl AudioPreprocessStage {
    pub fn spawn<P: AudioFeatureProducer>(
        mut producer: P,
        limits: AudioPreprocessLimits,
        observability: Arc<BatchObservability>,
    ) -> Result<Self, String> {
        let (sender, receiver) = mpsc::sync_channel::<QueuedJob>(limits.queue_depth.max(1));
        let (result_tx, result_rx) = mpsc::sync_channel(limits.result_queue_depth.max(1));
        let metrics = Arc::new(AudioPreprocessMetrics::new(observability));
        let worker_metrics = metrics.clone();
        let healthy = Arc::new(AtomicBool::new(true));
        let worker_healthy = healthy.clone();
        let worker = thread::Builder::new()
            .name("mlxcel-xla-audio-preprocess".to_string())
            .spawn(move || {
                while let Ok(queued) = receiver.recv() {
                    let Some((job, reservation)) = queued.dequeue() else {
                        continue;
                    };
                    let result = process_job(&mut producer, job, reservation, &worker_metrics);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
                worker_healthy.store(false, Ordering::Release);
            })
            .map_err(|error| format!("failed to spawn audio preprocessing worker: {error}"))?;
        Ok(Self {
            sender: Some(sender),
            result_rx: Some(result_rx),
            metrics,
            healthy,
            max_queued_encoded_bytes: limits.max_queued_encoded_bytes,
            max_in_flight_host_bytes: limits.max_in_flight_host_bytes,
            worker: Some(worker),
        })
    }

    pub fn try_submit(&self, job: AudioPreprocessJob) -> Result<(), AudioQueueError> {
        if job.cancelled.load(std::sync::atomic::Ordering::Acquire) {
            self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .observability
                .audio_preprocess_cancelled
                .fetch_add(1, Ordering::Relaxed);
            return Err(AudioQueueError::Cancelled {
                checkpoint: AudioPreprocessCheckpoint::Queue,
            });
        }
        let request_bytes = job.clips.iter().try_fold(0usize, |sum, clip| {
            sum.checked_add(clip.bytes.len())
                .ok_or(AudioQueueError::Overflow)
        });
        let request_bytes = match request_bytes {
            Ok(bytes) => bytes,
            Err(error) => {
                self.metrics
                    .record_rejection(AudioRejectionReason::Overflow);
                return Err(error);
            }
        };
        if request_bytes > job.policy.max_encoded_bytes_per_request {
            self.metrics
                .record_rejection(AudioRejectionReason::MemoryLimit);
            return Err(AudioQueueError::MemoryLimit {
                request_bytes,
                queued_bytes: self.metrics.queued_encoded_bytes.load(Ordering::Acquire),
                maximum_bytes: job.policy.max_encoded_bytes_per_request,
            });
        }
        let host_reservation_bytes = job_envelope_host_bytes(&job)
            .and_then(|bytes| bytes.checked_add(job.policy.max_waveform_working_bytes_per_request))
            .and_then(|bytes| bytes.checked_add(job.policy.max_prepared_result_bytes_per_request));
        let host_reservation_bytes = match host_reservation_bytes {
            Some(bytes) => bytes,
            None => {
                self.metrics
                    .record_rejection(AudioRejectionReason::Overflow);
                return Err(AudioQueueError::Overflow);
            }
        };
        if let Err(error) = reserve_queued_bytes(
            &self.metrics.queued_encoded_bytes,
            request_bytes,
            self.max_queued_encoded_bytes,
        ) {
            self.metrics.record_rejection(match &error {
                AudioQueueError::MemoryLimit { .. } => AudioRejectionReason::MemoryLimit,
                AudioQueueError::Overflow => AudioRejectionReason::Overflow,
                _ => AudioRejectionReason::WorkerUnavailable,
            });
            return Err(error);
        }
        if let Err(error) = reserve_host_bytes(
            &self.metrics.in_flight_host_bytes,
            host_reservation_bytes,
            self.max_in_flight_host_bytes,
        ) {
            self.metrics
                .queued_encoded_bytes
                .fetch_sub(request_bytes, Ordering::AcqRel);
            self.metrics.record_rejection(match &error {
                AudioQueueError::HostMemoryLimit { .. } => AudioRejectionReason::MemoryLimit,
                AudioQueueError::Overflow => AudioRejectionReason::Overflow,
                _ => AudioRejectionReason::WorkerUnavailable,
            });
            return Err(error);
        }
        self.metrics
            .observability
            .audio_preprocess_queued_bytes
            .store(
                self.metrics.queued_encoded_bytes.load(Ordering::Acquire),
                Ordering::Release,
            );
        self.metrics
            .observability
            .audio_preprocess_inflight_host_bytes
            .store(
                self.metrics.in_flight_host_bytes.load(Ordering::Acquire),
                Ordering::Release,
            );
        let queued = QueuedJob {
            job: Some(job),
            reservation: Some(HostMemoryReservation {
                encoded_bytes: request_bytes,
                host_bytes: host_reservation_bytes,
                metrics: self.metrics.clone(),
            }),
        };
        let Some(sender) = self.sender.as_ref() else {
            self.metrics
                .record_rejection(AudioRejectionReason::WorkerUnavailable);
            return Err(AudioQueueError::Disconnected);
        };
        match sender.try_send(queued) {
            Ok(()) => {
                self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(mpsc::TrySendError::Full(_)) => {
                self.metrics
                    .record_rejection(AudioRejectionReason::QueueFull);
                Err(AudioQueueError::Full)
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.metrics
                    .record_rejection(AudioRejectionReason::WorkerUnavailable);
                Err(AudioQueueError::Disconnected)
            }
        }
    }

    pub fn try_recv(&self) -> Result<AudioPreprocessResult, mpsc::TryRecvError> {
        self.result_rx
            .as_ref()
            .ok_or(mpsc::TryRecvError::Disconnected)?
            .try_recv()
    }

    pub fn recv(&self) -> Result<AudioPreprocessResult, mpsc::RecvError> {
        self.result_rx.as_ref().ok_or(mpsc::RecvError)?.recv()
    }

    pub fn metrics(&self) -> &Arc<AudioPreprocessMetrics> {
        &self.metrics
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

impl Drop for AudioPreprocessStage {
    fn drop(&mut self) {
        self.sender.take();
        self.result_rx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn process_job<P: AudioFeatureProducer>(
    producer: &mut P,
    job: AudioPreprocessJob,
    reservation: HostMemoryReservation,
    metrics: &AudioPreprocessMetrics,
) -> AudioPreprocessResult {
    let started = Instant::now();
    let job_id = job.job_id;
    let family = job.policy.family;
    let max_prefill_tokens = job.max_prefill_tokens;
    let max_result_bytes = job.policy.max_prepared_result_bytes_per_request;
    let outcome = match preprocess_wav_batch(&job.clips, job.policy, job.cancelled.as_ref()) {
        Err(AudioPreprocessError::Cancelled { checkpoint, .. }) => {
            AudioPreprocessOutcome::Cancelled(checkpoint)
        }
        Err(error) => AudioPreprocessOutcome::Failed(AudioStageError::Waveform(error)),
        Ok(waveforms) => {
            metrics.record_waveforms(&waveforms);
            if job.cancelled.load(Ordering::Acquire) {
                AudioPreprocessOutcome::Cancelled(AudioPreprocessCheckpoint::Feature)
            } else {
                let prepared = catch_unwind(AssertUnwindSafe(|| {
                    producer.prepare(waveforms, job.token_ids, job.cancelled.as_ref())
                }));
                match prepared {
                    Ok(Ok(_prepared)) if job.cancelled.load(Ordering::Acquire) => {
                        AudioPreprocessOutcome::Cancelled(AudioPreprocessCheckpoint::Feature)
                    }
                    Ok(Ok(prepared)) if prepared.sequence_len > max_prefill_tokens => {
                        AudioPreprocessOutcome::Failed(AudioStageError::ContextLimit {
                            family,
                            actual: prepared.sequence_len,
                            maximum: max_prefill_tokens,
                        })
                    }
                    Ok(Ok(prepared)) => {
                        let result_bytes =
                            prepared_prefill_host_bytes(&prepared).unwrap_or(usize::MAX);
                        if result_bytes > max_result_bytes {
                            AudioPreprocessOutcome::Failed(AudioStageError::ResultMemoryLimit {
                                family,
                                actual: result_bytes,
                                maximum: max_result_bytes,
                            })
                        } else {
                            metrics.record_effective_prefill(prepared.sequence_len);
                            AudioPreprocessOutcome::Prepared(prepared)
                        }
                    }
                    Ok(Err(reason)) => {
                        AudioPreprocessOutcome::Failed(AudioStageError::Feature { family, reason })
                    }
                    Err(_) => {
                        AudioPreprocessOutcome::Failed(AudioStageError::FeaturePanic { family })
                    }
                }
            }
        }
    };
    let elapsed_micros = (started.elapsed().as_micros().min(u64::MAX as u128) as u64).max(1);
    metrics
        .preprocessing_latency_micros
        .fetch_add(elapsed_micros, Ordering::Relaxed);
    metrics
        .observability
        .audio_preprocess_latency_micros
        .fetch_add(elapsed_micros, Ordering::Relaxed);
    match &outcome {
        AudioPreprocessOutcome::Prepared(_) => {
            metrics.completed.fetch_add(1, Ordering::Relaxed);
        }
        AudioPreprocessOutcome::Cancelled(_) => {
            metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            metrics
                .observability
                .audio_preprocess_cancelled
                .fetch_add(1, Ordering::Relaxed);
        }
        AudioPreprocessOutcome::Failed(error) => {
            metrics.record_rejection(match error {
                AudioStageError::Waveform(_) => AudioRejectionReason::Waveform,
                AudioStageError::Feature { .. } => AudioRejectionReason::Feature,
                AudioStageError::FeaturePanic { .. } => AudioRejectionReason::FeaturePanic,
                AudioStageError::ContextLimit { .. } => AudioRejectionReason::ContextLimit,
                AudioStageError::ResultMemoryLimit { .. } => AudioRejectionReason::MemoryLimit,
            });
        }
    }
    AudioPreprocessResult {
        job_id,
        outcome,
        _reservation: reservation,
    }
}

fn prepared_prefill_host_bytes(prepared: &PreparedPrefill) -> Option<usize> {
    fn tensor_bytes(tensor: &OwnedTensor) -> Option<usize> {
        tensor
            .bytes
            .capacity()
            .checked_add(tensor.shape.capacity().checked_mul(size_of::<usize>())?)
    }

    let mut total = size_of::<PreparedPrefill>().checked_add(
        prepared
            .token_ids
            .capacity()
            .checked_mul(size_of::<i32>())?,
    )?;
    total = total.checked_add(tensor_bytes(&prepared.embeddings)?)?;
    if let PreparedPositions::Explicit(tensor) | PreparedPositions::Mrope3D { tensor, .. } =
        &prepared.positions
    {
        total = total.checked_add(tensor_bytes(tensor)?)?;
    }
    total = total.checked_add(tensor_bytes(&prepared.attention_bias.tensor)?)?;
    total = total.checked_add(
        prepared
            .modalities
            .capacity()
            .checked_mul(size_of::<mlxcel_core::session::PreparedModality>())?,
    )?;
    for modality in &prepared.modalities {
        total = total.checked_add(modality.family.capacity())?;
    }
    Some(total)
}

fn job_envelope_host_bytes(job: &AudioPreprocessJob) -> Option<usize> {
    let mut total = size_of::<AudioPreprocessJob>();
    total = total.checked_add(job.token_ids.capacity().checked_mul(size_of::<i32>())?)?;
    total = total.checked_add(
        job.clips
            .capacity()
            .checked_mul(size_of::<AudioEncodedClip>())?,
    )?;
    for clip in &job.clips {
        total = total.checked_add(clip.bytes.capacity())?;
    }
    Some(total)
}

fn reserve_queued_bytes(
    queued: &AtomicUsize,
    request: usize,
    maximum: usize,
) -> Result<(), AudioQueueError> {
    let mut current = queued.load(Ordering::Acquire);
    loop {
        let next = current
            .checked_add(request)
            .ok_or(AudioQueueError::Overflow)?;
        if next > maximum {
            return Err(AudioQueueError::MemoryLimit {
                request_bytes: request,
                queued_bytes: current,
                maximum_bytes: maximum,
            });
        }
        match queued.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(actual) => current = actual,
        }
    }
}

fn reserve_host_bytes(
    in_flight: &AtomicUsize,
    request: usize,
    maximum: usize,
) -> Result<(), AudioQueueError> {
    let mut current = in_flight.load(Ordering::Acquire);
    loop {
        let next = current
            .checked_add(request)
            .ok_or(AudioQueueError::Overflow)?;
        if next > maximum {
            return Err(AudioQueueError::HostMemoryLimit {
                request_bytes: request,
                in_flight_bytes: current,
                maximum_bytes: maximum,
            });
        }
        match in_flight.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
#[path = "xla_audio_preprocess_tests.rs"]
mod tests;
