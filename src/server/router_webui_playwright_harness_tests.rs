// Copyright 2026 Lablup Inc.
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

//! Opt-in real-router WebUI browser harness for #1848.
//!
//! This test deliberately starts the real secured Axum router and serves the
//! checked-in embedded WebUI bundle over TCP. Only the model/downloader leaves
//! are fake, through existing test-only traits, so authentication, CSP, static
//! assets, router lifecycle, operations and dispatch remain production code.

use std::fs::File;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::HeaderValue;

use super::{RouterServerState, create_router_app_with_secured_ui};
use crate::downloader::DownloadHooks;
use crate::server::config::ServerConfig;
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::router_cache::{CacheSource, RouterDownloader};
use crate::server::router_lifecycle::{EVENT_RING_LIMIT, ResetEventKind};
use crate::server::router_models::{RouterPool, RouterSources};
use crate::server::router_presets::PresetCliOverrides;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerStartupConfig};

const PLAYWRIGHT_TIMEOUT: Duration = Duration::from_secs(240);
const PLAYWRIGHT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const SERVER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const PROCESS_GROUP_EMPTY_TIMEOUT: Duration = Duration::from_secs(5);
const FAILING_DOWNLOAD_REPO: &str = "mlx-community/Router-Harness-Fail-4bit";
const CANCELLABLE_DOWNLOAD_REPO: &str = "mlx-community/Router-Harness-Slow-4bit";

struct HarnessDownloader;

impl RouterDownloader for HarnessDownloader {
    fn validate(&self, repo_id: &str, _revision: Option<&str>) -> anyhow::Result<()> {
        if repo_id.trim().is_empty() || repo_id.contains("..") {
            anyhow::bail!("invalid test repo id");
        }
        Ok(())
    }

    fn download(
        &self,
        repo_id: &str,
        revision: Option<&str>,
        dest_root: &Path,
        hooks: DownloadHooks,
    ) -> anyhow::Result<()> {
        if let Some(plan) = &hooks.plan {
            plan(crate::downloader::DownloadPlan {
                repo_id: repo_id.to_string(),
                requested_revision: revision.unwrap_or("main").to_string(),
                resolved_revision: revision.unwrap_or("main").to_string(),
                destination: dest_root.join(repo_id),
                selected_files: 2,
                total_bytes: Some(2),
            });
        }
        let config_url = format!("https://example.invalid/{repo_id}/config.json");
        if let Some(progress) = &hooks.progress {
            progress(&config_url, 1, 2);
        }
        if repo_id == FAILING_DOWNLOAD_REPO {
            anyhow::bail!("deterministic router harness download failure for {repo_id}");
        }
        if repo_id == CANCELLABLE_DOWNLOAD_REPO {
            let cancel = hooks
                .cancel
                .clone()
                .ok_or_else(|| anyhow::anyhow!("missing download cancel flag"))?;
            let started = std::time::Instant::now();
            while !cancel.load(Ordering::Relaxed) {
                if started.elapsed() > Duration::from_secs(20) {
                    anyhow::bail!("deterministic router harness download was never cancelled");
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            return Err(anyhow::Error::new(crate::downloader::DownloadCancelled));
        }
        let dest = dest_root.join(repo_id);
        std::fs::create_dir_all(&dest)?;
        std::fs::write(
            dest.join("config.json"),
            br#"{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"],"quantization_config":{"bits":4}}"#,
        )?;
        std::fs::write(
            dest.join("model.safetensors"),
            b"fake weights for router harness",
        )?;
        std::fs::write(
            dest.join("mlxcel-router-harness.json"),
            serde_json::json!({"repo_id": repo_id, "revision": revision.unwrap_or("main")})
                .to_string(),
        )?;
        if let Some(progress) = &hooks.progress {
            progress(
                &format!("https://example.invalid/{repo_id}/model.safetensors"),
                2,
                2,
            );
        }
        Ok(())
    }
}

#[cfg(unix)]
fn write_private_key(path: &Path, key: &str) -> anyhow::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(key.as_bytes())?;
    let mode = std::fs::metadata(path)?.permissions().mode() & 0o777;
    anyhow::ensure!(
        mode == 0o600,
        "private key file mode was {mode:o}, not 0600"
    );
    Ok(())
}

#[cfg(not(unix))]
fn write_private_key(path: &Path, key: &str) -> anyhow::Result<()> {
    std::fs::write(path, key)?;
    Ok(())
}

#[cfg(unix)]
fn create_private_output(path: &Path) -> anyhow::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?)
}

#[cfg(not(unix))]
fn create_private_output(path: &Path) -> anyhow::Result<File> {
    Ok(std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?)
}

fn default_artifacts_parent(repo_root: &Path) -> PathBuf {
    repo_root.join("target").join("webui-router-artifacts")
}

fn artifact_run_dir(parent: PathBuf) -> PathBuf {
    parent.join(format!("run-{}", uuid::Uuid::new_v4()))
}

fn create_canonical_artifact_run_dir(parent: PathBuf) -> anyhow::Result<PathBuf> {
    let run_dir = artifact_run_dir(parent);
    std::fs::create_dir_all(&run_dir)?;
    run_dir.canonicalize().map_err(|err| {
        anyhow::anyhow!(
            "canonicalize artifact run dir {} before exporting child paths: {err}",
            run_dir.display()
        )
    })
}

#[test]
fn artifact_run_dirs_export_absolute_child_paths_without_chdir() -> anyhow::Result<()> {
    let absolute_parent = tempfile::tempdir()?;
    let absolute_run = create_canonical_artifact_run_dir(absolute_parent.path().to_path_buf())?;
    anyhow::ensure!(
        absolute_run.is_absolute(),
        "absolute parent produced relative run dir"
    );
    anyhow::ensure!(absolute_run.exists(), "absolute run dir was not created");
    anyhow::ensure!(
        absolute_run.join("control").is_absolute(),
        "absolute run control path would be relative"
    );

    let relative_parent = PathBuf::from("target")
        .join("webui-router-artifact-path-regression")
        .join(uuid::Uuid::new_v4().to_string());
    let relative_run = create_canonical_artifact_run_dir(relative_parent.clone())?;
    anyhow::ensure!(
        relative_run.is_absolute(),
        "relative parent was not canonicalized"
    );
    anyhow::ensure!(relative_run.exists(), "relative run dir was not created");
    anyhow::ensure!(
        relative_run.join("control").is_absolute(),
        "relative run control path would resolve under the child cwd"
    );
    let _ = std::fs::remove_dir_all(relative_parent);
    Ok(())
}

#[cfg(unix)]
fn signal_process_group(child_id: Option<u32>, signal: libc::c_int) {
    if let Some(child_id) = child_id {
        // The Playwright launcher starts in its own process group below, so a
        // timeout can terminate browser children as well as the pnpm wrapper.
        unsafe {
            let _ = libc::kill(-(child_id as libc::pid_t), signal);
        }
    }
}

#[cfg(unix)]
fn process_group_is_empty(child_id: Option<u32>) -> bool {
    let Some(child_id) = child_id else {
        return true;
    };
    let result = unsafe { libc::kill(-(child_id as libc::pid_t), 0) };
    result != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(unix)]
async fn wait_process_group_empty(child_id: Option<u32>, timeout: Duration) -> bool {
    let start = tokio::time::Instant::now();
    while start.elapsed() < timeout {
        if process_group_is_empty(child_id) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    process_group_is_empty(child_id)
}

#[cfg(not(unix))]
fn signal_process_group(_child_id: Option<u32>, _signal: i32) {}

#[cfg(not(unix))]
async fn wait_process_group_empty(_child_id: Option<u32>, _timeout: Duration) -> bool {
    true
}

async fn run_playwright(
    repo_root: &Path,
    url: &str,
    key_file: &Path,
    artifacts: &Path,
    models_dir: &Path,
    control_dir: &Path,
) -> Result<ExitStatus, String> {
    let stdout_path = artifacts.join("playwright.stdout.log");
    let stderr_path = artifacts.join("playwright.stderr.log");
    let stdout = create_private_output(&stdout_path)
        .map_err(|err| format!("create {}: {err}", stdout_path.display()))?;
    let stderr = create_private_output(&stderr_path)
        .map_err(|err| format!("create {}: {err}", stderr_path.display()))?;
    let mut command = tokio::process::Command::new("pnpm");
    command
        .current_dir(repo_root)
        .args([
            "--dir",
            "webui",
            "exec",
            "playwright",
            "test",
            "--config",
            "playwright.rust-router.config.ts",
        ])
        .env("MLXCEL_WEBUI_ROUTER_URL", url)
        .env("MLXCEL_WEBUI_ROUTER_KEY_FILE", key_file)
        .env("MLXCEL_WEBUI_ROUTER_ARTIFACTS", artifacts)
        .env("MLXCEL_WEBUI_ROUTER_MODELS_DIR", models_dir)
        .env("MLXCEL_WEBUI_ROUTER_CONTROL_DIR", control_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|err| {
        format!(
            "spawn playwright: {err}; install pnpm dependencies before running this opt-in target"
        )
    })?;
    let child_id = child.id();

    match tokio::time::timeout(PLAYWRIGHT_TIMEOUT, child.wait()).await {
        Ok(Ok(status)) => Ok(status),
        Ok(Err(err)) => Err(format!("wait for playwright: {err}")),
        Err(_) => {
            #[cfg(unix)]
            signal_process_group(child_id, libc::SIGTERM);
            #[cfg(not(unix))]
            signal_process_group(child_id, 15);
            let reaped = tokio::time::timeout(PLAYWRIGHT_SHUTDOWN_TIMEOUT, child.wait()).await;
            let empty_after_term =
                wait_process_group_empty(child_id, PROCESS_GROUP_EMPTY_TIMEOUT).await;
            if reaped.is_err() || !empty_after_term {
                #[cfg(unix)]
                signal_process_group(child_id, libc::SIGKILL);
                #[cfg(not(unix))]
                signal_process_group(child_id, 9);
                let _ = child.start_kill();
                let _ = tokio::time::timeout(PLAYWRIGHT_SHUTDOWN_TIMEOUT, child.wait()).await;
            }
            let empty_after_kill =
                wait_process_group_empty(child_id, PROCESS_GROUP_EMPTY_TIMEOUT).await;
            Err(format!(
                "playwright timed out after {:?}; terminate/reap result: {reaped:?}; group_empty_after_term={empty_after_term}; group_empty_after_kill={empty_after_kill}",
                PLAYWRIGHT_TIMEOUT
            ))
        }
    }
}

fn read_artifact_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|err| format!("<{} unavailable: {err}>", path.display()))
}

fn install_model_app_factory(pool: &RouterPool, handles: Arc<Mutex<Vec<ScriptedStreamHandle>>>) {
    pool.set_model_app_factory_for_tests(Some(Arc::new(move |path, config| {
        let (options_tx, _options_rx) = std::sync::mpsc::channel();
        let (provider, handle) = ModelProvider::scripted_streaming_for_route_tests(options_tx);
        let provider = Arc::new(provider);
        handles.lock().expect("scripted handle lock").push(handle);
        let state = AppState::new(
            provider.clone(),
            config,
            ChatTemplateProcessor::with_template("{{ messages[-1].content }}".to_string()),
            crate::tokenizer::MlxcelTokenizer::stub(),
            path.to_path_buf(),
            provider.batch_metrics().clone(),
        );
        let router = crate::server::app::create_app_without_cors(state.clone());
        Ok((state, router))
    })));
}

fn start_scripted_stream_feeder(
    handles: Arc<Mutex<Vec<ScriptedStreamHandle>>>,
) -> (Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = stop.clone();
    let thread = std::thread::spawn(move || {
        let mut completed_by_handle: Vec<[bool; 2]> = Vec::new();
        while !stop_for_thread.load(Ordering::Relaxed) {
            if let Ok(guard) = handles.lock() {
                if completed_by_handle.len() < guard.len() {
                    completed_by_handle.resize(guard.len(), [false, false]);
                }
                for (index, handle) in guard.iter().enumerate() {
                    if handle.cancellation_flag(0).is_some() && !completed_by_handle[index][0] {
                        handle.token("router ");
                        handle.token("harness");
                        handle.finish();
                        completed_by_handle[index][0] = true;
                    }
                    if let Some(cancelled) = handle.cancellation_flag(1)
                        && !completed_by_handle[index][1]
                    {
                        if cancelled.load(Ordering::Relaxed) {
                            handle.finish();
                            completed_by_handle[index][1] = true;
                        } else {
                            handle.token("streaming ");
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    });
    (stop, thread)
}

struct HarnessServer {
    addr: SocketAddr,
    server_instance_id: String,
    pool: Arc<RouterPool>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl HarnessServer {
    async fn stop_with_timeout(mut self, timeout: Duration) -> anyhow::Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        let mut task = self.task;
        match tokio::time::timeout(timeout, &mut task).await {
            Ok(joined) => {
                joined??;
                Ok(())
            }
            Err(err) => {
                task.abort();
                let _ = tokio::time::timeout(SERVER_SHUTDOWN_TIMEOUT, task).await;
                Err(err.into())
            }
        }
    }

    async fn stop(self) -> anyhow::Result<()> {
        self.stop_with_timeout(SERVER_SHUTDOWN_TIMEOUT).await
    }
}

fn build_router_state(
    models_dir: &Path,
    cache_root: &Path,
    router_key: &str,
    stream_handles: Arc<Mutex<Vec<ScriptedStreamHandle>>>,
) -> RouterServerState {
    let router_keys = vec![router_key.to_string()];
    let api_keys = crate::server::resolve_api_keys(&router_keys, &[]).expect("keys");
    let mut startup = ServerStartupConfig {
        webui_enabled: true,
        model_store_root: Some(cache_root.to_path_buf()),
        router_models_dir: Some(models_dir.to_path_buf()),
        models_max: 1,
        models_autoload: false,
        ..Default::default()
    };
    startup.api_prefix = "/lab".to_string();
    let config = ServerConfig {
        api_prefix: "/lab".to_string(),
        enable_settings_endpoint: true,
        enable_props_endpoint: true,
        enable_metrics_endpoint: true,
        api_keys: api_keys.clone(),
        ..Default::default()
    };
    let sources = RouterSources {
        models_dir: Some(models_dir.to_path_buf()),
        cache: Some(CacheSource::new(
            cache_root.to_path_buf(),
            Arc::new(HarnessDownloader),
        )),
        presets: Default::default(),
    };
    let pool = Arc::new(
        RouterPool::new(
            sources,
            startup.clone(),
            api_keys,
            PresetCliOverrides::default(),
            1,
            false,
        )
        .expect("router pool"),
    );
    install_model_app_factory(&pool, stream_handles);
    RouterServerState {
        pool,
        config: Arc::new(config),
        startup: Arc::new(startup),
        catalog_cache: Arc::new(crate::server::webui::catalog::CatalogProjectionCache::new()),
    }
}

async fn start_harness_server(
    bind_addr: SocketAddr,
    models_dir: &Path,
    cache_root: &Path,
    router_key: &str,
    stream_handles: Arc<Mutex<Vec<ScriptedStreamHandle>>>,
) -> anyhow::Result<HarnessServer> {
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    let addr = listener.local_addr()?;
    let origin = format!("http://127.0.0.1:{}", addr.port());
    let state = build_router_state(models_dir, cache_root, router_key, stream_handles);
    let pool = state.pool.clone();
    let server_instance_id = pool
        .lifecycle_coordinator()
        .server_instance_id()
        .to_string();
    let policy =
        crate::server::webui::security::WebUiSecurityPolicy::with_prefixes_limits_and_rate(
            vec![format!("127.0.0.1:{}", addr.port())],
            vec![HeaderValue::from_str(&origin).expect("origin header")],
            "/lab/webui",
            "/lab",
            32,
            32,
            120,
        )
        .expect("security policy");
    let app = create_router_app_with_secured_ui(state, policy);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Ok(HarnessServer {
        addr,
        server_instance_id,
        pool,
        shutdown_tx: Some(shutdown_tx),
        task,
    })
}

#[cfg(unix)]
fn write_private_json(path: &Path, value: serde_json::Value) -> anyhow::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(serde_json::to_string_pretty(&value)?.as_bytes())?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private_json(path: &Path, value: serde_json::Value) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}

fn take_marker(path: &Path) -> bool {
    if path.exists() {
        let _ = std::fs::remove_file(path);
        return true;
    }
    false
}

async fn run_control_loop(
    current: Arc<tokio::sync::Mutex<Option<HarnessServer>>>,
    control_dir: PathBuf,
    models_dir: PathBuf,
    cache_root: PathBuf,
    router_key: String,
    stream_handles: Arc<Mutex<Vec<ScriptedStreamHandle>>>,
    done: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let restart_request = control_dir.join("restart.request");
    let restart_done = control_dir.join("restart.done.json");
    let gap_request = control_dir.join("gap.request");
    let gap_done = control_dir.join("gap.done.json");
    while !done.load(Ordering::Relaxed) {
        if take_marker(&gap_request) {
            let guard = current.lock().await;
            let Some(server) = guard.as_ref() else {
                anyhow::bail!("server not running for gap request");
            };
            let coordinator = server.pool.lifecycle_coordinator();
            for _ in 0..(EVENT_RING_LIMIT + 2) {
                coordinator.publish_reset("harness_gap_fill", ResetEventKind::Reset);
            }
            write_private_json(
                &gap_done,
                serde_json::json!({
                    "server_instance_id": server.server_instance_id,
                    "after_sequence": 0,
                    "snapshot_sequence": coordinator.snapshot_sequence()
                }),
            )?;
        }
        if take_marker(&restart_request) {
            let old = {
                let mut guard = current.lock().await;
                guard.take().expect("server present for restart")
            };
            let old_addr = old.addr;
            let old_server_instance_id = old.server_instance_id.clone();
            let _ = old.stop_with_timeout(Duration::from_secs(1)).await;
            let mut new = None;
            let mut last_error = None;
            for _ in 0..40 {
                match start_harness_server(
                    old_addr,
                    &models_dir,
                    &cache_root,
                    &router_key,
                    stream_handles.clone(),
                )
                .await
                {
                    Ok(server) => {
                        new = Some(server);
                        break;
                    }
                    Err(err) => {
                        last_error = Some(err);
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                }
            }
            let new = new.ok_or_else(|| {
                anyhow::anyhow!("restart rebind failed after stop: {:?}", last_error)
            })?;
            let new_server_instance_id = new.server_instance_id.clone();
            {
                let mut guard = current.lock().await;
                *guard = Some(new);
            }
            write_private_json(
                &restart_done,
                serde_json::json!({
                    "old_server_instance_id": old_server_instance_id,
                    "new_server_instance_id": new_server_instance_id
                }),
            )?;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    Ok(())
}

#[tokio::test]
#[ignore = "opt-in real-router WebUI browser harness for #1848; use make verify-webui-integration-fake"]
async fn real_router_browser_harness() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = repo_root.join("webui/playwright.rust-router.config.ts");
    assert!(
        config.is_file(),
        "{} is required; missing setup is not a skipped pass",
        config.display()
    );

    let temp = tempfile::tempdir().expect("harness tempdir");
    let models_dir = temp.path().join("models-dir");
    let cache_root = temp.path().join("model-store");
    let artifact_parent = std::env::var_os("MLXCEL_WEBUI_ROUTER_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_artifacts_parent(&repo_root));
    let artifacts =
        create_canonical_artifact_run_dir(artifact_parent).expect("canonical artifact dir");
    let control_dir = artifacts.join("control");
    std::fs::create_dir_all(&models_dir).expect("models dir");
    std::fs::create_dir_all(&cache_root).expect("cache root");
    std::fs::create_dir_all(&control_dir).expect("control dir");

    let router_key = format!("router-{}", uuid::Uuid::new_v4());
    let stream_handles = Arc::new(Mutex::new(Vec::new()));
    let (feeder_stop, feeder_thread) = start_scripted_stream_feeder(stream_handles.clone());
    let server = start_harness_server(
        "127.0.0.1:0".parse().expect("loopback addr"),
        &models_dir,
        &cache_root,
        &router_key,
        stream_handles.clone(),
    )
    .await
    .expect("start harness server");
    let origin = format!("http://127.0.0.1:{}", server.addr.port());
    let current_server = Arc::new(tokio::sync::Mutex::new(Some(server)));
    let control_done = Arc::new(AtomicBool::new(false));
    let control_task = tokio::spawn(run_control_loop(
        current_server.clone(),
        control_dir.clone(),
        models_dir.clone(),
        cache_root.clone(),
        router_key.clone(),
        stream_handles,
        control_done.clone(),
    ));

    let key_file = temp.path().join("router-key.txt");
    write_private_key(&key_file, &router_key).expect("private key file");
    let url = format!("{origin}/lab/webui/");
    let playwright_result = run_playwright(
        &repo_root,
        &url,
        &key_file,
        &artifacts,
        &models_dir,
        &control_dir,
    )
    .await;

    let _ = std::fs::remove_file(&key_file);
    control_done.store(true, Ordering::Relaxed);
    let control_result = tokio::time::timeout(SERVER_SHUTDOWN_TIMEOUT, control_task)
        .await
        .map_err(|err| anyhow::anyhow!("control task stop: {err}"))
        .and_then(|joined| joined.map_err(|err| anyhow::anyhow!("control task join: {err}")))
        .and_then(|inner| inner);
    let server_stop_result = match current_server.lock().await.take() {
        Some(server) => server.stop().await,
        None => Ok(()),
    };
    feeder_stop.store(true, Ordering::Relaxed);
    let _ = feeder_thread.join();
    assert!(
        control_result.is_ok(),
        "control task failed: {control_result:?}"
    );
    assert!(
        server_stop_result.is_ok(),
        "server stop failed: {server_stop_result:?}"
    );

    match playwright_result {
        Ok(status) if status.success() => {}
        Ok(status) => {
            panic!(
                "Playwright real-router harness failed with status {:?}; artifacts: {}; stdout:\n{}\nstderr:\n{}",
                status.code(),
                artifacts.display(),
                read_artifact_text(&artifacts.join("playwright.stdout.log")),
                read_artifact_text(&artifacts.join("playwright.stderr.log"))
            );
        }
        Err(err) => {
            panic!(
                "Playwright real-router harness failed before completion: {err}; artifacts: {}; stdout:\n{}\nstderr:\n{}",
                artifacts.display(),
                read_artifact_text(&artifacts.join("playwright.stdout.log")),
                read_artifact_text(&artifacts.join("playwright.stderr.log"))
            );
        }
    }
}
