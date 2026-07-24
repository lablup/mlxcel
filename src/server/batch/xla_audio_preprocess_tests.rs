use std::sync::{Arc, Mutex, mpsc};

use mlxcel_core::session::{
    OwnedTensor, PreparedAttentionBias, PreparedPositions, PreparedPrefill, PreparedTensorDType,
};

use super::*;
use crate::audio::{AudioSourceKind, AudioWaveformBatch};

fn wav_pcm16_at(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut wav = Vec::with_capacity(44 + data_len);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

fn wav_pcm16(samples: &[i16]) -> Vec<u8> {
    wav_pcm16_at(16_000, samples)
}

fn job(id: u64, samples: usize) -> AudioPreprocessJob {
    AudioPreprocessJob {
        job_id: id,
        token_ids: vec![1, id as i32, 2],
        max_prefill_tokens: 4_096,
        clips: vec![AudioEncodedClip {
            bytes: wav_pcm16(&vec![0; samples]),
            source: AudioSourceKind::ServerInline,
            placeholder_ordinal: 1,
        }],
        policy: AudioFamilyPolicy::gemma3n(),
        cancelled: Arc::new(AtomicBool::new(false)),
    }
}

fn prepared(token_ids: Vec<i32>) -> PreparedPrefill {
    let sequence = token_ids.len();
    let embeddings = OwnedTensor::new(
        vec![0; sequence * 2 * 4],
        PreparedTensorDType::Float32,
        vec![1, sequence, 2],
    )
    .unwrap();
    let bias = OwnedTensor::new(
        vec![0; sequence * 4],
        PreparedTensorDType::Float32,
        vec![1, 1, 1, sequence],
    )
    .unwrap();
    PreparedPrefill::new(
        token_ids,
        embeddings,
        PreparedPositions::Sequential {
            start: 0,
            length: sequence,
        },
        PreparedAttentionBias {
            tensor: bias,
            causal: true,
        },
        Vec::new(),
    )
    .unwrap()
}

struct RecordingProducer {
    order: Arc<Mutex<Vec<u64>>>,
}

impl AudioFeatureProducer for RecordingProducer {
    fn prepare(
        &mut self,
        _waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        _cancelled: &AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        self.order.lock().unwrap().push(token_ids[1] as u64);
        Ok(prepared(token_ids))
    }
}

#[test]
fn single_worker_preserves_fifo_and_records_separate_audio_metrics() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let stage = AudioPreprocessStage::spawn(
        RecordingProducer {
            order: order.clone(),
        },
        AudioPreprocessLimits {
            queue_depth: 32,
            max_queued_encoded_bytes: 1_000_000,
        },
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    for id in 1..=32 {
        stage.try_submit(job(id, 800)).unwrap();
    }
    for expected in 1..=32 {
        let result = stage.recv().unwrap();
        assert_eq!(result.job_id, expected);
        let AudioPreprocessOutcome::Prepared(prefill) = result.outcome else {
            panic!("job must prepare");
        };
        // Logical/public prompt ids remain the producer input. Effective audio
        // tokens are reported only through the separate metric below.
        assert_eq!(prefill.token_ids, [1, expected as i32, 2]);
    }
    assert_eq!(*order.lock().unwrap(), (1..=32).collect::<Vec<_>>());
    let snapshot = stage.metrics().snapshot();
    assert_eq!(snapshot.accepted, 32);
    assert_eq!(snapshot.completed, 32);
    assert_eq!(snapshot.source_samples, 32 * 800);
    assert_eq!(snapshot.normalized_samples, 32 * 800);
    assert_eq!(snapshot.effective_audio_tokens, 32 * 188);
    assert_eq!(snapshot.effective_prefill_tokens, 32 * 3);
    assert_eq!(snapshot.queued_encoded_bytes, 0);
    assert!(stage.is_healthy());
}

struct BlockingProducer {
    started: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl AudioFeatureProducer for BlockingProducer {
    fn prepare(
        &mut self,
        _waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        _cancelled: &AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        let _ = self.started.send(());
        let _ = self.release.recv();
        Ok(prepared(token_ids))
    }
}

#[test]
fn queue_depth_and_queued_memory_are_bounded_and_released_on_full() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let stage = AudioPreprocessStage::spawn(
        BlockingProducer {
            started: started_tx,
            release: release_rx,
        },
        AudioPreprocessLimits {
            queue_depth: 1,
            max_queued_encoded_bytes: 1_000_000,
        },
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    stage.try_submit(job(1, 32)).unwrap();
    started_rx.recv().unwrap();
    let second = job(2, 32);
    let second_bytes = second.clips[0].bytes.len();
    stage.try_submit(second).unwrap();
    assert_eq!(
        stage.metrics().snapshot().queued_encoded_bytes,
        second_bytes
    );
    assert!(matches!(
        stage.try_submit(job(3, 32)),
        Err(AudioQueueError::Full)
    ));
    assert_eq!(stage.metrics().snapshot().reject_queue_full, 1);
    assert_eq!(
        stage.metrics().snapshot().queued_encoded_bytes,
        second_bytes,
        "the rejected envelope must release its reservation"
    );
    release_tx.send(()).unwrap();
    stage.recv().unwrap();
    // The producer blocks once for every call.
    release_tx.send(()).unwrap();
    stage.recv().unwrap();
    assert_eq!(stage.metrics().snapshot().queued_encoded_bytes, 0);
}

struct CancelAfterFeature;

impl AudioFeatureProducer for CancelAfterFeature {
    fn prepare(
        &mut self,
        _waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        cancelled: &AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        cancelled.store(true, Ordering::Release);
        Ok(prepared(token_ids))
    }
}

#[test]
fn queue_and_feature_completion_are_cancellable_without_leaks() {
    let stage = AudioPreprocessStage::spawn(
        CancelAfterFeature,
        AudioPreprocessLimits {
            queue_depth: 1,
            max_queued_encoded_bytes: 1_000_000,
        },
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    let cancelled = job(1, 32);
    cancelled.cancelled.store(true, Ordering::Release);
    assert!(matches!(
        stage.try_submit(cancelled),
        Err(AudioQueueError::Cancelled {
            checkpoint: AudioPreprocessCheckpoint::Queue
        })
    ));

    stage.try_submit(job(2, 32)).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Cancelled(AudioPreprocessCheckpoint::Feature)
    ));
    let snapshot = stage.metrics().snapshot();
    assert_eq!(snapshot.cancelled, 2);
    assert_eq!(snapshot.queued_encoded_bytes, 0);
}

struct PanicOnce(bool);

impl AudioFeatureProducer for PanicOnce {
    fn prepare(
        &mut self,
        _waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        _cancelled: &AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        if !self.0 {
            self.0 = true;
            panic!("synthetic feature panic");
        }
        Ok(prepared(token_ids))
    }
}

#[test]
fn panic_and_oversize_reject_only_the_affected_request() {
    let stage = AudioPreprocessStage::spawn(
        PanicOnce(false),
        AudioPreprocessLimits {
            queue_depth: 2,
            max_queued_encoded_bytes: 256,
        },
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    assert!(matches!(
        stage.try_submit(job(0, 200)),
        Err(AudioQueueError::MemoryLimit { .. })
    ));
    assert_eq!(stage.metrics().snapshot().reject_memory_limit, 1);
    stage.try_submit(job(1, 8)).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Failed(AudioStageError::FeaturePanic { .. })
    ));
    assert_eq!(stage.metrics().snapshot().reject_feature_panic, 1);
    assert!(stage.is_healthy(), "per-request panic must not kill worker");
    stage.try_submit(job(2, 8)).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Prepared(_)
    ));
    let mut context_oversize = job(3, 8);
    context_oversize.max_prefill_tokens = 2;
    stage.try_submit(context_oversize).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Failed(AudioStageError::ContextLimit {
            actual: 3,
            maximum: 2,
            ..
        })
    ));
    assert_eq!(stage.metrics().snapshot().reject_context_limit, 1);
    assert!(stage.is_healthy(), "context rejection must not kill worker");
    stage.try_submit(job(4, 8)).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Prepared(_)
    ));
    assert_eq!(stage.metrics().snapshot().queued_encoded_bytes, 0);
}

#[test]
fn source_and_normalized_samples_and_effective_prefill_are_distinct() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let stage = AudioPreprocessStage::spawn(
        RecordingProducer { order },
        AudioPreprocessLimits {
            queue_depth: 1,
            max_queued_encoded_bytes: 1_000_000,
        },
        Arc::new(BatchObservability::new()),
    )
    .unwrap();
    let mut resampled = job(7, 1);
    resampled.clips[0].bytes = wav_pcm16_at(8_000, &[0; 800]);
    stage.try_submit(resampled).unwrap();
    assert!(matches!(
        stage.recv().unwrap().outcome,
        AudioPreprocessOutcome::Prepared(_)
    ));
    let snapshot = stage.metrics().snapshot();
    assert_eq!(snapshot.source_samples, 800);
    assert_eq!(snapshot.normalized_samples, 1_600);
    assert_eq!(snapshot.effective_audio_tokens, 188);
    assert_eq!(snapshot.effective_prefill_tokens, 3);
}
