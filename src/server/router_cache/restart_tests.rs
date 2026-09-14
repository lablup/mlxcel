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

//! Owned test-process restarts, not production CLI or checkpoint acceptance.
//! Abrupt termination deliberately bypasses Drop: abandoned stages are retained,
//! never promoted, resumed, or cleaned by a different process at startup.

use super::{AnchoredStage, CacheSource, RouterDownloader, STAGING_DIR};
use crate::downloader::{DownloadHooks, DownloadPlan};
use crate::server::ServerStartupConfig;
use crate::server::router_lifecycle::OperationState;
use crate::server::router_models::{RouterPool, RouterSources};
use std::ffi::CString;
use std::fs::{self, File};
use std::io::Write;
use std::os::fd::FromRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

const CHILD_TEST: &str = "server::router_cache::restart_tests::restart_child";
const REPO: &str = "test-owner/restart-model";
const REVISION: &str = "0123456789012345678901234567890123456789";
const LIMIT: Duration = Duration::from_secs(20);

struct LocalDownloader {
    interrupt: bool,
    evidence: PathBuf,
}

impl RouterDownloader for LocalDownloader {
    fn validate(&self, _: &str, _: Option<&str>) -> anyhow::Result<()> {
        Ok(())
    }

    fn download(
        &self,
        repo: &str,
        _: Option<&str>,
        root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        let mut stage = AnchoredStage::create(root, repo)?;
        if let Some(plan) = &hooks.plan {
            plan(DownloadPlan {
                repo_id: repo.into(),
                requested_revision: REVISION.into(),
                resolved_revision: REVISION.into(),
                destination: root.join(repo),
                selected_files: 2,
                total_bytes: Some(6),
            });
        }
        write_at(&stage, "config.json", b"{}")?;
        if self.interrupt {
            write_at(&stage, "weights.partial", b"partial")?;
            fs::write(self.evidence.join("writer-ready"), b"ready")?;
            // A bounded escape path makes an orphaned test process harmless.
            std::thread::sleep(LIMIT);
            anyhow::bail!("parent did not interrupt its owned test process");
        }
        write_at(&stage, "model.safetensors", b"fake")?;
        if let Some(begin) = &hooks.begin_publish {
            anyhow::ensure!(begin(), "unexpected cancellation");
        }
        stage.publish()
    }
}

fn write_at(stage: &AnchoredStage, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
    let name = CString::new(name)?;
    // SAFETY: the stage owns its live directory descriptor and name is a C string.
    let fd = unsafe {
        libc::openat(
            stage.stage_fd(),
            name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
            0o600,
        )
    };
    anyhow::ensure!(fd >= 0, "openat: {}", std::io::Error::last_os_error());
    // SAFETY: successful openat returned a new owned descriptor.
    unsafe { File::from_raw_fd(fd) }.write_all(bytes)?;
    Ok(())
}

fn pool(root: &Path, evidence: &Path, interrupt: bool) -> Arc<RouterPool> {
    Arc::new(
        RouterPool::new(
            RouterSources {
                cache: Some(CacheSource::new(
                    root.to_path_buf(),
                    Arc::new(LocalDownloader {
                        interrupt,
                        evidence: evidence.into(),
                    }),
                )),
                ..Default::default()
            },
            ServerStartupConfig::default(),
            Default::default(),
            Default::default(),
            2,
            false,
        )
        .expect("CPU-only pool without any model load"),
    )
}

async fn wait_for(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + LIMIT;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "bounded child observation timed out"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[test]
fn restart_child() {
    let Ok(phase) = std::env::var("MLXCEL_RESTART_TEST_PHASE") else {
        return;
    };
    let evidence = PathBuf::from(std::env::var_os("MLXCEL_RESTART_TEST_DIR").unwrap());
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(child_phase(&evidence, &phase));
}

async fn child_phase(evidence: &Path, phase: &str) {
    let root = evidence.join("store");
    let pool = pool(&root, evidence, phase == "interrupt");
    let coordinator = pool.lifecycle_coordinator();
    let instance = coordinator.server_instance_id();
    if phase == "interrupt" {
        assert!(
            !root.exists(),
            "pool construction must not create the store"
        );
        let first = pool
            .submit_download(REPO, Some(REVISION), Some("restart-first"))
            .unwrap();
        wait_for(|| evidence.join("writer-ready").exists()).await;
        let queued = pool
            .submit_download("test-owner/queued", Some(REVISION), Some("restart-queued"))
            .unwrap();
        assert_eq!(
            coordinator
                .get_operation(&first.operation_id)
                .unwrap()
                .state,
            OperationState::Running
        );
        assert_eq!(
            coordinator
                .get_operation(&queued.operation_id)
                .unwrap()
                .state,
            OperationState::Queued
        );
        fs::write(
            evidence.join("interrupted.json"),
            serde_json::to_vec(&serde_json::json!({
                "instance": instance, "running": first.operation_id, "queued": queued.operation_id
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(evidence.join("parent-ready"), b"ready").unwrap();
        tokio::time::sleep(LIMIT).await;
        panic!("parent must terminate the owned child before its deadline");
    }
    let interrupted: serde_json::Value =
        serde_json::from_slice(&fs::read(evidence.join("interrupted.json")).unwrap()).unwrap();
    assert_ne!(instance, interrupted["instance"].as_str().unwrap());
    for key in ["running", "queued"] {
        #[cfg(feature = "webui")]
        assert_missing_operation(pool.clone(), interrupted[key].as_str().unwrap()).await;
        assert!(
            coordinator
                .get_operation(interrupted[key].as_str().unwrap())
                .is_none(),
            "old session history must not report terminal success"
        );
    }
    let stages = fs::read_dir(root.join(STAGING_DIR)).unwrap().count();
    assert_eq!(stages, 1, "startup must retain the abandoned private stage");
    if phase == "retry" {
        assert!(
            pool.catalog_snapshot().is_empty(),
            "config-first partial must never enter catalog"
        );
        let accepted = pool
            .submit_download(REPO, Some(REVISION), Some("restart-first"))
            .unwrap();
        assert_ne!(
            (instance, accepted.operation_id.as_str()),
            (
                interrupted["instance"].as_str().unwrap(),
                interrupted["running"].as_str().unwrap()
            ),
            "operation identity is scoped by server instance"
        );
        assert!(
            !accepted.idempotent_replay,
            "idempotency belongs to the new session"
        );
        wait_for(|| {
            coordinator
                .get_operation(&accepted.operation_id)
                .is_some_and(|op| op.state == OperationState::Succeeded)
        })
        .await;
        fs::write(
            evidence.join("published.json"),
            serde_json::to_vec(&serde_json::json!({
                "instance": instance, "terminal": accepted.operation_id,
            "model_id": pool.catalog_snapshot()[0].ui_model_id
            }))
            .unwrap(),
        )
        .unwrap();
    } else {
        assert_eq!(phase, "discover");
        let published: serde_json::Value =
            serde_json::from_slice(&fs::read(evidence.join("published.json")).unwrap()).unwrap();
        assert_ne!(instance, published["instance"].as_str().unwrap());
        assert_eq!(
            pool.catalog_snapshot()[0].ui_model_id,
            published["model_id"].as_str().unwrap()
        );
        assert!(
            coordinator
                .get_operation(published["terminal"].as_str().unwrap())
                .is_none()
        );
        #[cfg(feature = "webui")]
        assert_missing_operation(pool.clone(), published["terminal"].as_str().unwrap()).await;
    }
    let catalog = pool.catalog_snapshot();
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog[0].name, REPO);
    assert_eq!(
        fs::read(root.join(REPO).join("model.safetensors")).unwrap(),
        b"fake"
    );
    assert_eq!(
        fs::read_dir(root.join(STAGING_DIR)).unwrap().count(),
        stages
    );
}

struct OwnedChild(Child);

impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn(evidence: &Path, phase: &str) -> OwnedChild {
    OwnedChild(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CHILD_TEST, "--nocapture"])
            .env("MLXCEL_RESTART_TEST_PHASE", phase)
            .env("MLXCEL_RESTART_TEST_DIR", evidence)
            .spawn()
            .expect("owned CPU-only child"),
    )
}

#[test]
fn killed_writer_and_terminal_history_reconcile_across_processes() {
    let evidence = tempfile::tempdir().unwrap();
    let mut child = spawn(evidence.path(), "interrupt");
    let deadline = Instant::now() + LIMIT;
    while !evidence.path().join("parent-ready").exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "child exited before interruption barrier"
        );
        assert!(
            Instant::now() < deadline,
            "parent interruption barrier timed out"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    let abandoned = fs::read_dir(evidence.path().join("store").join(STAGING_DIR))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert_eq!(
        fs::read(abandoned.join("weights.partial")).unwrap(),
        b"partial"
    );
    for phase in ["retry", "discover"] {
        let mut child = spawn(evidence.path(), phase);
        let deadline = Instant::now() + LIMIT;
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "restart phase {phase}: {status}");
                assert_eq!(
                    fs::read(abandoned.join("weights.partial")).unwrap(),
                    b"partial"
                );
                break;
            }
            assert!(Instant::now() < deadline, "restart phase {phase} timed out");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(feature = "webui")]
async fn assert_missing_operation(pool: Arc<RouterPool>, id: &str) {
    use crate::server::config::ServerConfig;
    use crate::server::router_server::{
        RouterServerState, create_router_app_with_authenticated_ui,
    };
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let config = ServerConfig {
        api_keys: crate::server::resolve_api_keys(&["restart-test-key".into()], &[]).unwrap(),
        ..Default::default()
    };
    let app = create_router_app_with_authenticated_ui(RouterServerState {
        pool,
        config: Arc::new(config),
        startup: Arc::new(ServerStartupConfig::default()),
        catalog_cache: Arc::new(crate::server::webui::catalog::CatalogProjectionCache::new()),
    });
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/ui-api/v1/operations/{id}"))
                .header("authorization", "Bearer restart-test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let mut actual: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(actual["request_id"].as_str().unwrap().starts_with("req_"));
    actual["request_id"] = serde_json::json!("<request>");
    assert_eq!(
        actual,
        serde_json::json!({
            "request_id": "<request>",
            "error": {"code": "not_found", "message": "operation not found", "retryable": true}
        })
    );
}
