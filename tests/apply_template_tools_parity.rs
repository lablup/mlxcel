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

//! Real-checkpoint parity for the rendered prompt of a tools-less chat request
//! (issue #1597).
//!
//! Both checkpoints here guard their tool-calling preamble on whether `tools`
//! is defined, in the two shapes that a defined-but-empty tool list used to
//! satisfy: Youtu-LLM uses the DeepSeek V3 form
//! `{% if tools is defined and tools is not none %}`, and Llama 3.2 defaults
//! `tools` to `none` only when it is undefined and then branches on
//! `{%- if tools is not none %}`. Rendering either one with `tools = []`
//! produced a phantom preamble with an empty function list on every plain chat
//! request, inflating the prompt from 8 to 76 tokens on Youtu and from 40 to 97
//! on Llama 3.2 and steering the model toward a function-call format nobody
//! asked for.
//!
//! The oracle strings below are what transformers, mlx-lm and a Python jinja2
//! render with `tools` undefined or `tools=None` produce from the very same
//! template files.
//!
//! The two servers are started one after the other inside a single test rather
//! than in two parallel tests: the suite shares one Metal device, and two
//! concurrently resident checkpoints is the shape that aborts runs.
//!
//! To run the gated test:
//! ```text
//! cargo test --test apply_template_tools_parity --release -- --ignored
//! ```

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::{repo_binary_path, repo_model_dir};

/// DeepSeek V3-style template, shipped as a standalone `chat_template.jinja`.
const YOUTU_MODEL: &str = "mlx/youtu-llm-2b-4bit";

/// Llama 3.2 template, carried in `tokenizer_config.json`.
const LLAMA_MODEL: &str = "mlx/llama-3.2-1b-4bit";

/// The one-line user turn every request below sends.
const PROMPT: &str = "The Fibonacci sequence begins with";

/// What transformers / mlx-lm render from the Youtu template with no tools.
const YOUTU_ORACLE: &str =
    "<|begin_of_text|><|User|>The Fibonacci sequence begins with<|Assistant|>";

/// What Python jinja2 renders from the Llama 3.2 template with `tools`
/// undefined, with the `Today Date:` line replaced by [`DATE_PLACEHOLDER`]
/// because the template resolves it through `strftime_now` at render time.
const LLAMA_ORACLE: &str = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: <DATE>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe Fibonacci sequence begins with<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

const DATE_PLACEHOLDER: &str = "<DATE>";

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
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    panic!("mlxcel-server did not become healthy at {base_url}");
}

fn spawn_server(model_dir: &str, port: &str) -> Child {
    Command::new(repo_binary_path("mlxcel-server"))
        .args([
            "--model",
            model_dir,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--no-warmup",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mlxcel-server")
}

fn stop_server(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Replace the template-resolved `Today Date:` value with a fixed placeholder
/// so the assertion does not depend on the day the suite runs.
fn normalize_date(prompt: &str) -> String {
    let mut out = String::with_capacity(prompt.len());
    for (index, line) in prompt.split('\n').enumerate() {
        if index > 0 {
            out.push('\n');
        }
        match line.strip_prefix("Today Date: ") {
            Some(_) => {
                out.push_str("Today Date: ");
                out.push_str(DATE_PLACEHOLDER);
            }
            None => out.push_str(line),
        }
    }
    out
}

/// Body for the one-line user turn, with `tools` set only when `tools` is
/// `Some`. `Some(json!([]))` is the explicit-empty-list case, which OpenAI and
/// llama-server both read as "no tools".
fn body(tools: Option<serde_json::Value>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "messages": [{"role": "user", "content": PROMPT}]
    });
    if let Some(tools) = tools {
        body["tools"] = tools;
    }
    body
}

fn one_tool() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}}
        }
    }])
}

async fn rendered_prompt(
    client: &reqwest::Client,
    base_url: &str,
    tools: Option<serde_json::Value>,
) -> String {
    let response = client
        .post(format!("{base_url}/apply-template"))
        .json(&body(tools))
        .send()
        .await
        .expect("send /apply-template request");
    assert!(
        response.status().is_success(),
        "/apply-template must succeed; got {}",
        response.status()
    );
    let json: serde_json::Value = response.json().await.expect("parse /apply-template body");
    json["prompt"]
        .as_str()
        .expect("/apply-template answers a string prompt")
        .to_string()
}

async fn input_tokens(
    client: &reqwest::Client,
    base_url: &str,
    tools: Option<serde_json::Value>,
) -> u64 {
    let response = client
        .post(format!("{base_url}/v1/chat/completions/input_tokens"))
        .json(&body(tools))
        .send()
        .await
        .expect("send input_tokens request");
    assert!(
        response.status().is_success(),
        "input_tokens must succeed; got {}",
        response.status()
    );
    let json: serde_json::Value = response.json().await.expect("parse input_tokens body");
    json["input_tokens"]
        .as_u64()
        .expect("input_tokens answers a number")
}

#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn apply_template_omits_the_tool_preamble_when_the_request_carries_no_tools() {
    let youtu_dir = repo_model_dir(YOUTU_MODEL);
    let llama_dir = repo_model_dir(LLAMA_MODEL);
    if !youtu_dir.exists() || !llama_dir.exists() {
        eprintln!(
            "Skipping: need both {} and {}",
            youtu_dir.display(),
            llama_dir.display()
        );
        return;
    }

    let client = reqwest::Client::new();

    // --- Youtu-LLM: `tools is defined and tools is not none` ---------------
    let port = reserve_port().to_string();
    let base_url = format!("http://127.0.0.1:{port}");
    let mut child = spawn_server(&youtu_dir.to_string_lossy(), &port);
    wait_for_health(&client, &base_url).await;

    let no_tools = rendered_prompt(&client, &base_url, None).await;
    let empty_tools = rendered_prompt(&client, &base_url, Some(serde_json::json!([]))).await;
    let no_tools_count = input_tokens(&client, &base_url, None).await;
    let with_tools = rendered_prompt(&client, &base_url, Some(one_tool())).await;
    stop_server(&mut child);

    assert_eq!(
        no_tools, YOUTU_ORACLE,
        "a tools-less request must render the transformers prompt"
    );
    assert_eq!(
        empty_tools, YOUTU_ORACLE,
        "an explicit empty tool list must render as no tools"
    );
    assert_eq!(
        no_tools_count, 8,
        "the tools-less Youtu prompt is 8 tokens, not the 76 the phantom preamble cost"
    );
    assert!(
        with_tools.contains("<|begin_of_tool_description|>") && with_tools.contains("get_weather"),
        "a request that does carry a tool must still render the tool block; got {with_tools:?}"
    );

    // --- Llama 3.2: `set tools = none` default, then `tools is not none` ---
    let port = reserve_port().to_string();
    let base_url = format!("http://127.0.0.1:{port}");
    let mut child = spawn_server(&llama_dir.to_string_lossy(), &port);
    wait_for_health(&client, &base_url).await;

    let no_tools = rendered_prompt(&client, &base_url, None).await;
    let no_tools_count = input_tokens(&client, &base_url, None).await;
    let with_tools = rendered_prompt(&client, &base_url, Some(one_tool())).await;
    stop_server(&mut child);

    assert_eq!(
        normalize_date(&no_tools),
        LLAMA_ORACLE,
        "a tools-less request must render the jinja2 tools-undefined prompt"
    );
    assert!(
        !no_tools.contains("Environment: ipython")
            && !no_tools.contains("Given the following functions"),
        "neither tool branch may fire without tools; got {no_tools:?}"
    );
    assert_eq!(
        no_tools_count, 40,
        "the tools-less Llama 3.2 prompt is 40 tokens, not the 97 the phantom preamble cost"
    );
    assert!(
        with_tools.contains("Environment: ipython") && with_tools.contains("get_weather"),
        "a request that does carry a tool must still render the tool branch; got {with_tools:?}"
    );
}
