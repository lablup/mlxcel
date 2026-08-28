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

//! Real-checkpoint smoke test for the Vertex AI predict adapter (#1456):
//! boots `mlxcel-server` with `AIP_MODE=PREDICTION` against a dense text
//! model and exercises one `chatCompletions` batch through
//! `POST /predict`, asserting the port override and the prediction shapes.

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
    let deadline = Instant::now() + Duration::from_secs(120);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(format!("{base_url}/health")).send().await
            && response.status().is_success()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    panic!("server did not become healthy at {base_url}");
}

fn stop_server(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[tokio::test]
#[ignore = "requires local model weights and a built mlxcel-server binary"]
async fn predict_serves_a_chat_completions_batch_against_a_real_model() {
    let model_dir = repo_model_dir("mlx/llama-3.2-1b-4bit");
    if !model_dir.exists() {
        eprintln!(
            "skipping: model checkpoint not found at {}",
            model_dir.display()
        );
        return;
    }

    let aip_port = reserve_port();
    let flag_port = reserve_port();
    let model_arg = model_dir.to_string_lossy().to_string();
    let flag_port_arg = flag_port.to_string();

    // AIP_HTTP_PORT must override --port: the server has to listen on the
    // Vertex-assigned port, not the flag's.
    let mut child = Command::new(repo_binary_path("mlxcel-server"))
        .args([
            "-m",
            &model_arg,
            "--port",
            &flag_port_arg,
            "--host",
            "127.0.0.1",
        ])
        .env("AIP_MODE", "PREDICTION")
        .env("AIP_HTTP_PORT", aip_port.to_string())
        .env("AIP_HEALTH_ROUTE", "hp")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mlxcel-server");

    let client = reqwest::Client::new();
    let base_url = format!("http://127.0.0.1:{aip_port}");
    wait_for_health(&client, &base_url).await;

    // The AIP health alias answers on the overridden port.
    let health_alias = client
        .get(format!("{base_url}/hp"))
        .send()
        .await
        .expect("health alias request");
    assert!(health_alias.status().is_success());

    // One chatCompletions batch of two instances, dispatched through the
    // real model; predictions come back in order with generated content.
    let response = client
        .post(format!("{base_url}/predict"))
        .json(&serde_json::json!({
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "model": "llama-3.2-1b-4bit",
                    "messages": [{"role": "user", "content": "Reply with one word: hello"}],
                    "max_tokens": 8,
                    "temperature": 0
                },
                {
                    "@requestFormat": "chatCompletions",
                    "model": "llama-3.2-1b-4bit",
                    "messages": [{"role": "user", "content": "Reply with one word: goodbye"}],
                    "max_tokens": 8,
                    "temperature": 0
                }
            ]
        }))
        .send()
        .await
        .expect("predict request");
    assert!(response.status().is_success(), "predict answered an error");
    let parsed: serde_json::Value = response.json().await.expect("predict body parses");
    let predictions = parsed["predictions"].as_array().expect("predictions array");
    assert_eq!(predictions.len(), 2);
    for prediction in predictions {
        let content = prediction["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_else(|| panic!("prediction is not a chat completion: {prediction}"));
        assert!(
            !content.trim().is_empty(),
            "the model generated no content: {prediction}"
        );
    }

    // The flag port must NOT be serving: the override replaced it.
    let flag_result = client
        .get(format!("http://127.0.0.1:{flag_port}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await;
    assert!(
        flag_result.is_err(),
        "--port must be overridden by AIP_HTTP_PORT"
    );

    stop_server(&mut child);
}
