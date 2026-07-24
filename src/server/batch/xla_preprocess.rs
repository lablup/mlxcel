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

//! Bounded host-image preprocessing for the OpenXLA serve worker.
//!
//! The MLX-backed vision tower remains on this dedicated thread. The IREE
//! scheduler only enqueues bounded jobs and polls owned `PreparedPrefill`
//! results, so image decoding/vision execution cannot stall active decode rows.

use std::any::Any;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use mlxcel_core::session::PreparedPrefill;

use crate::{
    HostMultimodalPreprocessor, XlaVisionBackend, load_xla_image_preprocessor,
    server::model_provider::model_worker::decode_request_images,
};

pub(super) struct ImagePreprocessJob {
    pub job_id: u64,
    pub token_ids: Vec<i32>,
    /// Declaration count captured at the HTTP boundary.
    ///
    /// The worker rejects any decode result that does not preserve this count,
    /// preventing malformed or oversized images from silently disappearing.
    pub expected_image_count: usize,
    pub images: Vec<Vec<u8>>,
    pub cancelled: Arc<AtomicBool>,
}

pub(super) enum ImagePreprocessOutcome {
    Prepared(PreparedPrefill),
    Cancelled,
    Failed(String),
}

pub(super) struct ImagePreprocessResult {
    pub job_id: u64,
    pub outcome: ImagePreprocessOutcome,
}

/// A single-worker, bounded preprocessing queue.
///
/// The host preprocessor is constructed inside its owning thread, so its MLX
/// handles never cross threads and the trait does not need an artificial
/// `Send` bound.
pub(super) struct ImagePreprocessStage {
    job_tx: mpsc::SyncSender<ImagePreprocessJob>,
    result_rx: mpsc::Receiver<ImagePreprocessResult>,
    vision_backend: XlaVisionBackend,
}

impl ImagePreprocessStage {
    pub(super) fn spawn_for_model(
        model_path: PathBuf,
        queue_depth: usize,
    ) -> Result<Option<Self>, String> {
        Self::spawn_with_loader(queue_depth, move || {
            load_xla_image_preprocessor(&model_path).map_err(|error| error.to_string())
        })
    }

    pub(super) fn spawn_with_loader<F>(
        queue_depth: usize,
        loader: F,
    ) -> Result<Option<Self>, String>
    where
        F: FnOnce() -> Result<Option<Box<dyn HostMultimodalPreprocessor>>, String> + Send + 'static,
    {
        let (job_tx, job_rx) = mpsc::sync_channel(queue_depth.max(1));
        let (result_tx, result_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("model-worker-xla-images".to_string())
            .spawn(move || {
                let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(loader));
                let Some(preprocessor) = (match loaded {
                    Ok(Ok(preprocessor)) => {
                        let backend = preprocessor.as_ref().map(|loaded| loaded.backend());
                        let _ = ready_tx.send(Ok(backend));
                        preprocessor
                    }
                    Ok(Err(error)) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                    Err(payload) => {
                        let _ = ready_tx.send(Err(format!(
                            "OpenXLA image preprocessor panicked during startup: {}",
                            panic_message(payload.as_ref())
                        )));
                        return;
                    }
                }) else {
                    return;
                };

                while let Ok(job) = job_rx.recv() {
                    let result = process_job(preprocessor.as_ref(), job);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("failed to spawn OpenXLA image preprocessor: {error}"))?;

        match ready_rx.recv() {
            Ok(Ok(Some(vision_backend))) => {
                let stage = Self {
                    job_tx,
                    result_rx,
                    vision_backend,
                };
                tracing::info!(
                    vision_backend = %stage.vision_backend(),
                    queue_depth = queue_depth.max(1),
                    "OpenXLA image preprocessing stage ready"
                );
                Ok(Some(stage))
            }
            Ok(Ok(None)) => Ok(None),
            Ok(Err(error)) => Err(error),
            Err(_) => {
                Err("OpenXLA image preprocessor exited before reporting startup status".to_string())
            }
        }
    }

    #[must_use]
    pub(super) fn vision_backend(&self) -> XlaVisionBackend {
        self.vision_backend
    }

    pub(super) fn try_submit(
        &self,
        job: ImagePreprocessJob,
    ) -> Result<(), mpsc::TrySendError<ImagePreprocessJob>> {
        self.job_tx.try_send(job)
    }

    pub(super) fn try_recv(&self) -> Result<ImagePreprocessResult, mpsc::TryRecvError> {
        self.result_rx.try_recv()
    }
}

fn process_job(
    preprocessor: &dyn HostMultimodalPreprocessor,
    job: ImagePreprocessJob,
) -> ImagePreprocessResult {
    let ImagePreprocessJob {
        job_id,
        token_ids,
        expected_image_count,
        images,
        cancelled,
    } = job;
    if cancelled.load(Ordering::Acquire) {
        return ImagePreprocessResult {
            job_id,
            outcome: ImagePreprocessOutcome::Cancelled,
        };
    }

    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let decoded = decode_request_images(&images).map_err(|error| error.to_string())?;
        if decoded.len() != expected_image_count {
            return Err(format!(
                "decoded image cardinality mismatch: expected {expected_image_count}, decoded {}; \
                 refusing partial multimodal execution",
                decoded.len()
            ));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err("request cancelled during image decoding".to_string());
        }
        preprocessor
            .prepare(&token_ids, &decoded)
            .map_err(|error| error.to_string())
    }));

    let outcome = if cancelled.load(Ordering::Acquire) {
        ImagePreprocessOutcome::Cancelled
    } else {
        match prepared {
            Ok(Ok(prepared)) => ImagePreprocessOutcome::Prepared(prepared),
            Ok(Err(error)) => ImagePreprocessOutcome::Failed(error),
            Err(payload) => ImagePreprocessOutcome::Failed(format!(
                "image preprocessing panicked: {}",
                panic_message(payload.as_ref())
            )),
        }
    };
    ImagePreprocessResult { job_id, outcome }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    use image::{DynamicImage, ImageFormat};
    use mlxcel_core::session::PreparedPrefill;

    use super::*;
    use crate::{FakeHostMultimodalPreprocessor, HostPreprocessorError};

    struct BlockingPreprocessor {
        started_tx: mpsc::Sender<i32>,
        release_rx: mpsc::Receiver<()>,
        finished_tx: Option<mpsc::Sender<i32>>,
        inner: FakeHostMultimodalPreprocessor,
    }

    struct TaggedPreprocessor {
        backend: XlaVisionBackend,
        inner: FakeHostMultimodalPreprocessor,
    }

    impl HostMultimodalPreprocessor for TaggedPreprocessor {
        fn backend(&self) -> XlaVisionBackend {
            self.backend
        }

        fn prepare(
            &self,
            token_ids: &[i32],
            images: &[DynamicImage],
        ) -> Result<PreparedPrefill, HostPreprocessorError> {
            self.inner.prepare(token_ids, images)
        }
    }

    impl HostMultimodalPreprocessor for BlockingPreprocessor {
        fn prepare(
            &self,
            token_ids: &[i32],
            images: &[DynamicImage],
        ) -> Result<PreparedPrefill, HostPreprocessorError> {
            let marker = token_ids[0];
            self.started_tx.send(marker).unwrap();
            self.release_rx.recv().unwrap();
            let prepared = self.inner.prepare(token_ids, images);
            if let Some(finished_tx) = &self.finished_tx {
                let _ = finished_tx.send(marker);
            }
            prepared
        }
    }

    fn png_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        DynamicImage::new_rgb8(2, 2)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        bytes
    }

    fn stage(max_sequence_len: usize) -> ImagePreprocessStage {
        ImagePreprocessStage::spawn_with_loader(1, move || {
            Ok(Some(Box::new(FakeHostMultimodalPreprocessor {
                image_token_id: -200,
                tokens_per_image: 2,
                hidden_size: 3,
                max_sequence_len,
            })))
        })
        .unwrap()
        .unwrap()
    }

    fn receive(stage: &ImagePreprocessStage) -> ImagePreprocessResult {
        for _ in 0..200 {
            match stage.try_recv() {
                Ok(result) => return result,
                Err(mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(error) => panic!("preprocessor disconnected: {error}"),
            }
        }
        panic!("timed out waiting for image preprocessing")
    }

    fn job(job_id: u64, marker: i32, cancelled: Arc<AtomicBool>) -> ImagePreprocessJob {
        ImagePreprocessJob {
            job_id,
            token_ids: vec![marker, -200, marker + 1],
            expected_image_count: 1,
            images: vec![png_bytes()],
            cancelled,
        }
    }

    #[test]
    fn bounded_stage_prepares_owned_payload() {
        let stage = stage(16);
        assert_eq!(stage.vision_backend(), XlaVisionBackend::Host);
        stage
            .try_submit(ImagePreprocessJob {
                job_id: 7,
                token_ids: vec![1, -200, 2],
                expected_image_count: 1,
                images: vec![png_bytes()],
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        let result = receive(&stage);
        assert_eq!(result.job_id, 7);
        let ImagePreprocessOutcome::Prepared(prepared) = result.outcome else {
            panic!("expected prepared payload");
        };
        assert_eq!(prepared.token_ids, vec![1, -200, -200, 2]);
        assert_eq!(prepared.sequence_len, 4);
    }

    #[test]
    fn readiness_preserves_selected_vision_backend() {
        let stage = ImagePreprocessStage::spawn_with_loader(1, || {
            Ok(Some(Box::new(TaggedPreprocessor {
                backend: XlaVisionBackend::Iree,
                inner: FakeHostMultimodalPreprocessor {
                    image_token_id: -200,
                    tokens_per_image: 2,
                    hidden_size: 3,
                    max_sequence_len: 16,
                },
            })))
        })
        .unwrap()
        .unwrap();
        assert_eq!(stage.vision_backend(), XlaVisionBackend::Iree);
    }

    #[test]
    fn malformed_media_and_placeholder_mismatch_are_per_job_failures() {
        let stage = stage(16);
        stage
            .try_submit(ImagePreprocessJob {
                job_id: 1,
                token_ids: vec![1, -200, 2],
                expected_image_count: 1,
                images: vec![b"not-an-image".to_vec()],
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        assert!(matches!(
            receive(&stage).outcome,
            ImagePreprocessOutcome::Failed(_)
        ));

        stage
            .try_submit(ImagePreprocessJob {
                job_id: 5,
                token_ids: vec![1, -200, 2, -200, 3],
                expected_image_count: 2,
                images: vec![png_bytes(), b"not-an-image".to_vec()],
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        let ImagePreprocessOutcome::Failed(error) = receive(&stage).outcome else {
            panic!("partial decode must fail the whole request");
        };
        assert!(error.contains("decoded image cardinality mismatch"));

        stage
            .try_submit(ImagePreprocessJob {
                job_id: 2,
                token_ids: vec![1, -200, -200, 2],
                expected_image_count: 1,
                images: vec![png_bytes()],
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        assert!(matches!(
            receive(&stage).outcome,
            ImagePreprocessOutcome::Failed(_)
        ));
    }

    #[test]
    fn expanded_capacity_and_cancellation_are_isolated() {
        let stage = stage(3);
        stage
            .try_submit(ImagePreprocessJob {
                job_id: 3,
                token_ids: vec![1, -200, 2],
                expected_image_count: 1,
                images: vec![png_bytes()],
                cancelled: Arc::new(AtomicBool::new(false)),
            })
            .unwrap();
        let ImagePreprocessOutcome::Failed(error) = receive(&stage).outcome else {
            panic!("expected capacity failure");
        };
        assert!(error.contains("exceeds model capacity"));

        let cancelled = Arc::new(AtomicBool::new(true));
        stage
            .try_submit(ImagePreprocessJob {
                job_id: 4,
                token_ids: vec![1, -200, 2],
                expected_image_count: 1,
                images: vec![png_bytes()],
                cancelled,
            })
            .unwrap();
        assert!(matches!(
            receive(&stage).outcome,
            ImagePreprocessOutcome::Cancelled
        ));
    }

    #[test]
    fn bounded_stage_preserves_fifo_and_progress_after_inflight_cancel() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let stage = ImagePreprocessStage::spawn_with_loader(1, move || {
            Ok(Some(Box::new(BlockingPreprocessor {
                started_tx,
                release_rx,
                finished_tx: None,
                inner: FakeHostMultimodalPreprocessor {
                    image_token_id: -200,
                    tokens_per_image: 2,
                    hidden_size: 3,
                    max_sequence_len: 16,
                },
            })))
        })
        .unwrap()
        .unwrap();

        let cancelled_a = Arc::new(AtomicBool::new(false));
        stage.try_submit(job(10, 10, cancelled_a.clone())).unwrap();
        assert_eq!(
            started_rx.recv_timeout(std::time::Duration::from_secs(1)),
            Ok(10)
        );

        stage
            .try_submit(job(20, 20, Arc::new(AtomicBool::new(false))))
            .unwrap();
        let full = stage
            .try_submit(job(30, 30, Arc::new(AtomicBool::new(false))))
            .unwrap_err();
        assert!(matches!(full, mpsc::TrySendError::Full(_)));

        cancelled_a.store(true, Ordering::Release);
        release_tx.send(()).unwrap();
        assert!(matches!(
            receive(&stage).outcome,
            ImagePreprocessOutcome::Cancelled
        ));
        assert_eq!(
            started_rx.recv_timeout(std::time::Duration::from_secs(1)),
            Ok(20),
            "the queued request must start next in FIFO order"
        );

        release_tx.send(()).unwrap();
        let result = receive(&stage);
        assert_eq!(result.job_id, 20);
        assert!(matches!(
            result.outcome,
            ImagePreprocessOutcome::Prepared(_)
        ));
    }

    #[test]
    fn result_receiver_disconnect_does_not_strand_inflight_preprocessor() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let stage = ImagePreprocessStage::spawn_with_loader(1, move || {
            Ok(Some(Box::new(BlockingPreprocessor {
                started_tx,
                release_rx,
                finished_tx: Some(finished_tx),
                inner: FakeHostMultimodalPreprocessor {
                    image_token_id: -200,
                    tokens_per_image: 2,
                    hidden_size: 3,
                    max_sequence_len: 16,
                },
            })))
        })
        .unwrap()
        .unwrap();

        stage
            .try_submit(job(40, 40, Arc::new(AtomicBool::new(false))))
            .unwrap();
        assert_eq!(
            started_rx.recv_timeout(std::time::Duration::from_secs(1)),
            Ok(40)
        );
        drop(stage);
        release_tx.send(()).unwrap();
        assert_eq!(
            finished_rx.recv_timeout(std::time::Duration::from_secs(1)),
            Ok(40)
        );
    }

    #[test]
    fn unsupported_loader_returns_no_stage() {
        let stage = ImagePreprocessStage::spawn_with_loader(1, || Ok(None)).unwrap();
        assert!(stage.is_none());
    }

    #[test]
    fn supported_loader_error_fails_startup() {
        let error = ImagePreprocessStage::spawn_with_loader(1, || {
            Err(HostPreprocessorError::WeightLoad("missing projector".to_string()).to_string())
        })
        .err()
        .expect("startup should fail");
        assert!(error.contains("missing projector"));
    }
}
