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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use super::ImageWorkerProvider;
use crate::imagegen::testing::{STUB_FAMILIES, StubModel, write_model_index};
use crate::imagegen::{
    GeneratedImage, ImageGenerationModel, ImageRequest, ImageSamplingDefaults, ImageTask,
};
use crate::server::image_model::{ImageError, ImageModelProvider};
use tempfile::TempDir;

fn stub_provider() -> ImageWorkerProvider {
    ImageWorkerProvider::from_loader("stub-image".to_string(), 4, Duration::from_secs(30), || {
        Ok(Box::new(StubModel {
            family: "flux2",
            channels: 3,
            edit: true,
        }))
    })
    .expect("stub worker spawns")
}

fn request(n: u32, seed: u64) -> ImageRequest {
    let mut req = ImageRequest::generate("a cat", seed);
    req.width = 32;
    req.height = 16;
    req.n = n;
    req
}

#[test]
fn worker_reports_the_models_static_facts() {
    let p = stub_provider();
    assert_eq!(p.model_id(), "stub-image");
    assert_eq!(p.family(), "flux2");
    assert!(p.supports(ImageTask::Generate));
    assert!(p.supports(ImageTask::Edit));
    assert_eq!(p.size_multiple(), 16);
    let d = p.defaults(ImageTask::Generate);
    assert_eq!((d.steps, d.guidance), (4, 1.0));
    assert!(p.created_at() > 0);
}

#[test]
fn worker_round_trips_a_request_on_its_own_thread() {
    let p = stub_provider();
    let images = p.generate(request(2, 10)).expect("generates");
    assert_eq!(images.len(), 2);
    assert_eq!(images[0].seed, 10);
    assert_eq!(images[1].seed, 11);
    assert_eq!((images[0].width, images[0].height), (32, 16));
}

#[test]
fn worker_serves_requests_one_after_another() {
    let p = stub_provider();
    for seed in 0..5 {
        assert_eq!(p.generate(request(1, seed)).unwrap()[0].seed, seed);
    }
}

#[test]
fn worker_surfaces_loader_failure() {
    let err =
        ImageWorkerProvider::from_loader("broken".to_string(), 4, Duration::from_secs(30), || {
            Err(anyhow::anyhow!("synthetic load failure"))
        })
        .expect_err("loader failure propagates");
    assert!(err.to_string().contains("synthetic load failure"), "{err}");
}

#[test]
fn load_reports_an_unrecognized_checkpoint() {
    let dir = TempDir::new().unwrap();
    let err = ImageWorkerProvider::load(dir.path(), "x".to_string(), 4, Duration::from_secs(30))
        .expect_err("empty directory");
    assert!(
        err.to_string()
            .contains("no image generation family recognized"),
        "{err}"
    );
}

#[test]
fn load_reports_a_recognized_but_unported_family() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2Pipeline");
    // A table without a flux2 row, as a build that has not ported it.
    let err = ImageWorkerProvider::load_with_families(
        dir.path(),
        "x".to_string(),
        4,
        Duration::from_secs(30),
        &[],
    )
    .expect_err("unported");
    assert!(
        err.to_string()
            .contains("image generation family flux2 is not supported by this build"),
        "{err}"
    );
}

#[test]
fn load_with_families_builds_the_registered_family() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2Pipeline");
    let p = ImageWorkerProvider::load_with_families(
        dir.path(),
        "served".to_string(),
        4,
        Duration::from_secs(30),
        &STUB_FAMILIES,
    )
    .expect("loads");
    assert_eq!(p.family(), "flux2");
    assert_eq!(p.model_id(), "served");
}

/// Panics on the first generation, then behaves.
struct PanicsOnce {
    armed: AtomicBool,
}

impl ImageGenerationModel for PanicsOnce {
    fn family(&self) -> &'static str {
        "flux2"
    }
    fn supports(&self, _: ImageTask) -> bool {
        true
    }
    fn defaults(&self, _: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(1, 0.0).unwrap()
    }
    fn generate(&self, req: &ImageRequest) -> anyhow::Result<Vec<GeneratedImage>> {
        if self.armed.swap(false, Ordering::SeqCst) {
            panic!("synthetic kernel failure");
        }
        StubModel {
            family: "flux2",
            channels: 3,
            edit: false,
        }
        .generate(req)
    }
}

#[test]
fn a_panic_is_an_error_and_the_worker_survives() {
    let p =
        ImageWorkerProvider::from_loader("panicky".to_string(), 4, Duration::from_secs(30), || {
            Ok(Box::new(PanicsOnce {
                armed: AtomicBool::new(true),
            }))
        })
        .unwrap();
    let err = p.generate(request(1, 1)).expect_err("first request panics");
    match err {
        ImageError::Internal(message) => {
            assert!(message.contains("synthetic kernel failure"), "{message}")
        }
        other => panic!("expected Internal, got {other:?}"),
    }
    assert_eq!(p.generate(request(1, 2)).unwrap()[0].seed, 2);
}

/// Blocks every generation until the test sends a token.
struct Gated {
    gate: Mutex<mpsc::Receiver<()>>,
}

impl ImageGenerationModel for Gated {
    fn family(&self) -> &'static str {
        "flux2"
    }
    fn supports(&self, _: ImageTask) -> bool {
        true
    }
    fn defaults(&self, _: ImageTask) -> ImageSamplingDefaults {
        ImageSamplingDefaults::new(1, 0.0).unwrap()
    }
    fn generate(&self, req: &ImageRequest) -> anyhow::Result<Vec<GeneratedImage>> {
        let _ = self
            .gate
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(30));
        StubModel {
            family: "flux2",
            channels: 3,
            edit: false,
        }
        .generate(req)
    }
}

fn gated_provider(
    queue_depth: usize,
    timeout: Duration,
) -> (Arc<ImageWorkerProvider>, mpsc::Sender<()>) {
    let (release, gate) = mpsc::channel();
    let provider =
        ImageWorkerProvider::from_loader("gated".to_string(), queue_depth, timeout, move || {
            Ok(Box::new(Gated {
                gate: Mutex::new(gate),
            }))
        })
        .unwrap();
    (Arc::new(provider), release)
}

#[test]
fn a_full_queue_sheds_load() {
    let (provider, release) = gated_provider(1, Duration::from_secs(30));
    let (results_tx, results) = mpsc::channel();
    let mut handles = Vec::new();
    for seed in 0..4 {
        let provider = Arc::clone(&provider);
        let tx = results_tx.clone();
        handles.push(std::thread::spawn(move || {
            let _ = tx.send(provider.generate(request(1, seed)));
        }));
    }
    drop(results_tx);

    // One request runs and one waits in the queue; the other two are rejected
    // immediately, so their results arrive while the gate is still closed.
    let mut rejected = 0;
    while rejected < 2 {
        match results.recv_timeout(Duration::from_secs(10)) {
            Ok(Err(ImageError::QueueFull)) => rejected += 1,
            other => panic!("expected a QueueFull result first, got {other:?}"),
        }
    }
    release.send(()).unwrap();
    release.send(()).unwrap();
    for handle in handles {
        handle.join().unwrap();
    }
    let accepted: Vec<_> = results.iter().collect();
    assert_eq!(accepted.len(), 2);
    assert!(accepted.iter().all(Result::is_ok));
}

#[test]
fn a_slow_request_times_out_without_killing_the_worker() {
    let (provider, release) = gated_provider(4, Duration::from_millis(100));
    let err = provider.generate(request(1, 1)).expect_err("times out");
    assert!(matches!(err, ImageError::Timeout), "{err:?}");
    // Let the abandoned generation finish, then the worker serves again.
    release.send(()).unwrap();
    release.send(()).unwrap();
    // Two tokens are queued, one for the abandoned request and one for this.
    let again = provider
        .generate(request(1, 2))
        .expect("worker still serves");
    assert_eq!(again[0].seed, 2);
}

#[test]
fn expired_queued_request_never_reaches_the_model() {
    // PanicsOnce makes observing the first model invocation deterministic: the
    // live request must receive the panic, not an already-expired queued one.
    let worker =
        super::ImageWorker::spawn("expired-image-test", 4, Duration::from_secs(30), || {
            Ok(Box::new(PanicsOnce {
                armed: AtomicBool::new(true),
            }))
        })
        .unwrap();
    let (respond, reply) = mpsc::channel();
    worker
        .dispatch(super::ImageCommand::Generate {
            request: Box::new(request(1, 1)),
            deadline: std::time::Instant::now() - Duration::from_secs(1),
            respond,
        })
        .unwrap();
    assert!(matches!(
        reply.recv_timeout(Duration::from_secs(10)).unwrap(),
        Err(ImageError::Timeout)
    ));
    assert!(matches!(
        worker.generate(request(1, 2)),
        Err(ImageError::Internal(_))
    ));
    assert_eq!(worker.generate(request(1, 3)).unwrap()[0].seed, 3);
}

#[test]
fn load_reports_a_flux2_checkpoint_missing_a_component() {
    let dir = TempDir::new().unwrap();
    write_model_index(dir.path(), "Flux2KleinPipeline");
    let err = ImageWorkerProvider::load(dir.path(), "x".to_string(), 4, Duration::from_secs(30))
        .expect_err("incomplete checkpoint");
    assert!(
        err.to_string().contains("FLUX.2 checkpoint is missing"),
        "{err}"
    );
}
