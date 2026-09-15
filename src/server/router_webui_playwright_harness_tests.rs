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
use crate::server::router_models::{RouterPool, RouterSources};
use crate::server::router_presets::PresetCliOverrides;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider, ServerStartupConfig};

const PLAYWRIGHT_TIMEOUT: Duration = Duration::from_secs(240);
const PLAYWRIGHT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);
const SERVER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

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
                &format!("https://example.invalid/{repo_id}/config.json"),
                1,
                2,
            );
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

fn default_artifacts_dir(repo_root: &Path) -> PathBuf {
    repo_root
        .join("target")
        .join("webui-router-artifacts")
        .join(format!("run-{}", uuid::Uuid::new_v4()))
}

#[cfg(unix)]
fn terminate_process_group(child_id: Option<u32>) {
    if let Some(child_id) = child_id {
        // The Playwright launcher starts in its own process group below, so a
        // timeout can terminate browser children as well as the pnpm wrapper.
        unsafe {
            let _ = libc::kill(-(child_id as libc::pid_t), libc::SIGTERM);
        }
    }
}

#[cfg(not(unix))]
fn terminate_process_group(_child_id: Option<u32>) {}

async fn run_playwright(
    repo_root: &Path,
    url: &str,
    key_file: &Path,
    artifacts: &Path,
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

    match tokio::time::timeout(PLAYWRIGHT_TIMEOUT, child.wait()).await {
        Ok(Ok(status)) => Ok(status),
        Ok(Err(err)) => Err(format!("wait for playwright: {err}")),
        Err(_) => {
            terminate_process_group(child.id());
            let reaped = tokio::time::timeout(PLAYWRIGHT_SHUTDOWN_TIMEOUT, child.wait()).await;
            if reaped.is_err() {
                let _ = child.start_kill();
                let _ = tokio::time::timeout(PLAYWRIGHT_SHUTDOWN_TIMEOUT, child.wait()).await;
            }
            Err(format!(
                "playwright timed out after {:?}; terminate/reap result: {reaped:?}",
                PLAYWRIGHT_TIMEOUT
            ))
        }
    }
}

fn read_artifact_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|err| format!("<{} unavailable: {err}>", path.display()))
}

fn add_models_dir_seed(root: &Path) -> anyhow::Result<()> {
    let seed = root.join("seed-local");
    std::fs::create_dir_all(&seed)?;
    std::fs::write(
        seed.join("config.json"),
        br#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"quantization_config":{"bits":4}}"#,
    )?;
    std::fs::write(seed.join("model.safetensors"), b"seed weights")?;
    Ok(())
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
    let artifacts = std::env::var_os("MLXCEL_WEBUI_ROUTER_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_artifacts_dir(&repo_root));
    std::fs::create_dir_all(&models_dir).expect("models dir");
    std::fs::create_dir_all(&cache_root).expect("cache root");
    std::fs::create_dir_all(&artifacts).expect("artifact dir");
    add_models_dir_seed(&models_dir).expect("seed model");

    let router_key = format!("router-{}", uuid::Uuid::new_v4());
    let api_keys =
        crate::server::resolve_api_keys(std::slice::from_ref(&router_key), &[]).expect("keys");
    let mut startup = ServerStartupConfig {
        webui_enabled: true,
        model_store_root: Some(cache_root.clone()),
        router_models_dir: Some(models_dir.clone()),
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
        models_dir: Some(models_dir),
        cache: Some(CacheSource::new(
            cache_root.clone(),
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
    let stream_handles = Arc::new(Mutex::new(Vec::new()));
    install_model_app_factory(&pool, stream_handles.clone());
    let (feeder_stop, feeder_thread) = start_scripted_stream_feeder(stream_handles);

    let state = RouterServerState {
        pool,
        config: Arc::new(config),
        startup: Arc::new(startup),
        catalog_cache: Arc::new(crate::server::webui::catalog::CatalogProjectionCache::new()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    let origin = format!("http://127.0.0.1:{}", addr.port());
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
    let server = tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });

    let key_file = temp.path().join("router-key.txt");
    write_private_key(&key_file, &router_key).expect("private key file");
    let url = format!("{origin}/lab/webui/");
    let playwright_result = run_playwright(&repo_root, &url, &key_file, &artifacts).await;

    let _ = std::fs::remove_file(&key_file);
    let _ = shutdown_tx.send(());
    feeder_stop.store(true, Ordering::Relaxed);
    let _ = feeder_thread.join();
    let served = tokio::time::timeout(SERVER_SHUTDOWN_TIMEOUT, server).await;
    let served = match served {
        Ok(joined) => joined.expect("server task"),
        Err(_) => panic!("server did not stop within {:?}", SERVER_SHUTDOWN_TIMEOUT),
    };
    assert!(served.is_ok(), "server failed: {served:?}");

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
