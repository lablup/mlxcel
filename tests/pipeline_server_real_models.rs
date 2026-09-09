mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    port
}

async fn wait_for_health(client: &reqwest::Client, base_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("server did not become healthy at {base_url}");
}

fn spawn_server(args: &[&str]) -> Child {
    Command::new(repo_binary_path("mlxcel-server"))
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mlxcel-server")
}

fn stop_server(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

async fn post_completion(
    client: reqwest::Client,
    base_url: String,
    model: &'static str,
    prompt: &'static str,
    max_tokens: u32,
) -> serde_json::Value {
    client
        .post(format!("{base_url}/v1/completions"))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0
        }))
        .send()
        .await
        .expect("send completion request")
        .json::<serde_json::Value>()
        .await
        .expect("parse completion response")
}

async fn fetch_health(client: &reqwest::Client, base_url: &str) -> serde_json::Value {
    client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .expect("health request")
        .json::<serde_json::Value>()
        .await
        .expect("parse health")
}

fn completion_text_is_non_empty(response: &serde_json::Value) -> bool {
    matches!(
        response["choices"][0]["text"].as_str(),
        Some(text) if !text.is_empty()
    )
}

#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn pipeline_server_llama_multi_request_smoke() {
    let model_dir = repo_model_dir("llama-3.2-1b-instruct-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let model_arg = model_dir.to_string_lossy().to_string();
    let port_arg = port.to_string();
    let mut child = spawn_server(&[
        "-m",
        &model_arg,
        "--alias",
        "llama-pp-test",
        "--host",
        "127.0.0.1",
        "--port",
        &port_arg,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--max-queue-depth",
        "8",
        "--no-warmup",
        "--metrics",
        "--pp-layers",
        "0-7,8-15",
        "--pp-micro-batch-size",
        "2",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let base_url_a = base_url.clone();
    let client_a = client.clone();
    let handle_a = tokio::spawn(post_completion(
        client_a,
        base_url_a,
        "llama-pp-test",
        "Write a detailed numbered list about oranges with one short sentence per item.",
        96,
    ));

    let base_url_b = base_url.clone();
    let client_b = client.clone();
    let handle_b = tokio::spawn(post_completion(
        client_b,
        base_url_b,
        "llama-pp-test",
        "Write a detailed numbered list about apples with one short sentence per item.",
        96,
    ));

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut saw_concurrency = false;
    while Instant::now() < deadline {
        let health = fetch_health(&client, &base_url).await;
        let active = health["batch"]["active_sequences"].as_u64().unwrap_or(0);
        let current_batch = health["observability"]["current_batch_size"]
            .as_u64()
            .unwrap_or(0);
        if active >= 2 || current_batch >= 2 {
            saw_concurrency = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let response_a = handle_a.await.expect("join request A");
    let response_b = handle_b.await.expect("join request B");
    let final_health = fetch_health(&client, &base_url).await;
    stop_server(&mut child);

    assert!(completion_text_is_non_empty(&response_a));
    assert!(completion_text_is_non_empty(&response_b));
    assert!(
        final_health["observability"]["sequences_completed"]
            .as_u64()
            .unwrap_or(0)
            >= 2,
        "server did not complete both requests: {final_health}"
    );
    if !saw_concurrency {
        eprintln!(
            "pipeline_server_llama_multi_request_smoke did not observe concurrent gauges; final health: {final_health}"
        );
    }
}

#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn pipeline_server_llama_dense_baseline_smoke() {
    let model_dir = repo_model_dir("llama-3.2-1b-instruct-4bit");
    if !model_dir.exists() {
        eprintln!(
            "Skipping test: model directory not found at {}",
            model_dir.display()
        );
        return;
    }

    let port = reserve_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let model_arg = model_dir.to_string_lossy().to_string();
    let port_arg = port.to_string();
    let mut child = spawn_server(&[
        "-m",
        &model_arg,
        "--alias",
        "llama-dense-test",
        "--host",
        "127.0.0.1",
        "--port",
        &port_arg,
        "--parallel",
        "1",
        "--max-batch-size",
        "1",
        "--no-warmup",
    ]);

    let client = reqwest::Client::new();
    wait_for_health(&client, &base_url).await;

    let response = post_completion(
        client,
        base_url,
        "llama-dense-test",
        "Say hello in one short sentence.",
        24,
    )
    .await;
    stop_server(&mut child);

    assert!(completion_text_is_non_empty(&response));
}
