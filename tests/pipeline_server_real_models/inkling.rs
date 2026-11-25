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

//! Server prompt lookup on first-generation Inkling (issue #3425).
//!
//! `--draft-kind prompt-lookup` runs the token-only verify rounds inside the
//! batch scheduler, through the same `Engine::verify_round` the CLI's
//! `--prompt-lookup` uses (#3423). These tests start one server with it and
//! one without on `inkling-0.6b-a0.6b`, send each the same greedy requests,
//! and require the same text: completion and chat, streamed and not, a
//! prompt the reply copies and one it does not, and a concurrent pair, which
//! the scheduler verifies side by side up to `--prompt-lookup-max-batch`.

use std::process::Child;

use super::{reserve_port, spawn_server, stop_server, wait_for_health};
use crate::common::repo_model_dir;

const MODEL: &str = "inkling-0.6b-a0.6b";
const ALIAS: &str = "inkling-pl";
const MAX_TOKENS: u32 = 64;

/// This checkpoint answers most prompts with one memorized passage and then
/// stops. The prompt holds that passage and then its opening again, so the
/// reply repeats n-grams of the prompt and prompt lookup drafts them (on the
/// chat route the server reports 21 proposals, 17 accepted, in 3 rounds).
const COPY_PROMPT: &str = "Darth Plagueis was a Dark Lord of the Sith, so powerful and so wise he could use the Force to influence the midichlorians to create life. It's not a story the Jedi would tell you. It's a Sith legend. Darth Plagueis was a Dark Lord of the Sith";
/// A prompt with no repeat; the raw completion of an instruction ends at once
/// on this checkpoint, so it is the opening of a passage.
const PROSE_PROMPT: &str = "The rain had not stopped for three days, and the old lighthouse keeper";

struct Server {
    child: Child,
    base_url: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        stop_server(&mut self.child);
    }
}

async fn start(client: &reqwest::Client, model_dir: &str, prompt_lookup: bool) -> Server {
    let port = reserve_port().to_string();
    let mut args = vec![
        "-m",
        model_dir,
        "--alias",
        ALIAS,
        "--host",
        "127.0.0.1",
        "--port",
        &port,
        "--parallel",
        "2",
        "--max-batch-size",
        "2",
        "--no-warmup",
    ];
    if prompt_lookup {
        args.extend([
            "--draft-kind",
            "prompt-lookup",
            "--prompt-lookup-max-batch",
            "2",
        ]);
    }
    let server = Server {
        child: spawn_server(&args),
        base_url: format!("http://127.0.0.1:{port}"),
    };
    wait_for_health(client, &server.base_url).await;
    server
}

/// One greedy request: `(endpoint, body)` for `prompt`.
fn request(chat: bool, stream: bool, prompt: &str) -> (&'static str, serde_json::Value) {
    let mut body = if chat {
        serde_json::json!({"messages": [{"role": "user", "content": prompt}]})
    } else {
        serde_json::json!({"prompt": prompt})
    };
    body["model"] = ALIAS.into();
    body["max_tokens"] = MAX_TOKENS.into();
    body["temperature"] = 0.into();
    body["stream"] = stream.into();
    let path = if chat {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    (path, body)
}

/// The generated text of a response (both channels of a chat reply), and
/// the `timings.draft_n` the server reported, when it did.
fn text_of(chat: bool, stream: bool, raw: &str) -> (String, Option<u64>) {
    let pieces: Vec<serde_json::Value> = if stream {
        raw.lines()
            .filter_map(|line| line.trim().strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    } else {
        vec![serde_json::from_str(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}"))]
    };
    let mut text = String::new();
    let mut draft_n = None;
    for piece in &pieces {
        let choice = &piece["choices"][0];
        let part = match (chat, stream) {
            (false, _) => &choice["text"],
            (true, false) => &choice["message"],
            (true, true) => &choice["delta"],
        };
        if chat {
            for field in ["reasoning_content", "content"] {
                text.push_str(part[field].as_str().unwrap_or(""));
            }
        } else {
            text.push_str(part.as_str().unwrap_or(""));
        }
        draft_n = draft_n.or(piece["timings"]["draft_n"].as_u64());
    }
    (text, draft_n)
}

async fn send(
    client: &reqwest::Client,
    server: &Server,
    chat: bool,
    stream: bool,
    prompt: &str,
) -> (String, Option<u64>) {
    let (path, body) = request(chat, stream, prompt);
    let response = client
        .post(format!("{}{path}", server.base_url))
        .json(&body)
        .send()
        .await
        .expect("send request");
    assert!(
        response.status().is_success(),
        "{path} returned {}",
        response.status()
    );
    let raw = response.text().await.expect("read body");
    text_of(chat, stream, &raw)
}

#[tokio::test]
#[ignore = "requires local model weights and the mlxcel-server binary"]
async fn inkling_server_prompt_lookup_matches_plain_decode() {
    let model_dir = repo_model_dir(MODEL);
    if !model_dir.exists() {
        eprintln!("Skipping test: {} not found", model_dir.display());
        return;
    }
    let model_arg = model_dir.to_string_lossy().to_string();
    let client = mlxcel::http_tls::client();
    let plain = start(&client, &model_arg, false).await;
    let lookup = start(&client, &model_arg, true).await;

    let mut drafted_copy_replies = 0;
    for chat in [false, true] {
        for stream in [false, true] {
            for (name, prompt) in [("copy", COPY_PROMPT), ("prose", PROSE_PROMPT)] {
                let (want, _) = send(&client, &plain, chat, stream, prompt).await;
                let (got, draft_n) = send(&client, &lookup, chat, stream, prompt).await;
                assert!(
                    !want.is_empty(),
                    "{name} chat={chat} stream={stream}: empty"
                );
                assert_eq!(
                    got, want,
                    "{name} chat={chat} stream={stream}: prompt lookup changed the text"
                );
                if name == "copy" && draft_n.is_some_and(|n| n > 0) {
                    drafted_copy_replies += 1;
                }
            }
        }
    }
    // The non-streamed chat reply carries `timings`; a copy reply that
    // reports no proposal would mean the verify rounds never ran.
    assert!(
        drafted_copy_replies > 0,
        "no copy reply reported prompt-lookup proposals"
    );

    // A concurrent pair: both rows decode in one batch and each runs its own
    // verify rounds (the batch is within --prompt-lookup-max-batch).
    for chat in [false, true] {
        let (want_copy, _) = send(&client, &plain, chat, false, COPY_PROMPT).await;
        let (want_prose, _) = send(&client, &plain, chat, false, PROSE_PROMPT).await;
        let (copy, prose) = tokio::join!(
            send(&client, &lookup, chat, false, COPY_PROMPT),
            send(&client, &lookup, chat, true, PROSE_PROMPT),
        );
        assert_eq!(copy.0, want_copy, "concurrent copy, chat={chat}");
        assert_eq!(prose.0, want_prose, "concurrent prose, chat={chat}");
    }
}
