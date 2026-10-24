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

//! Completion input contracts exercised through the in-process worker path.

use std::path::PathBuf;
use std::sync::mpsc;

use serde_json::{Value, json};

use super::*;
use crate::server::batch::BatchObservability;
use crate::server::model_provider::ScriptedStreamHandle;
use crate::server::types::CompletionPrompt;
use crate::server::{AppState, ChatTemplateProcessor, ModelProvider};
use crate::tokenizer::MlxcelTokenizer;

struct Harness {
    server: InProcessServer,
    handle: ScriptedStreamHandle,
    options: mpsc::Receiver<ServerGenerateOptions>,
    prompts: mpsc::Receiver<String>,
}

fn harness() -> Harness {
    let (options_tx, options) = mpsc::channel();
    let (prompt_tx, prompts) = mpsc::channel();
    let (provider, handle) =
        ModelProvider::scripted_streaming_for_route_tests_with_prompts(options_tx, prompt_tx);
    let metrics = provider.batch_metrics().clone();
    let model_path = PathBuf::from("completion-contract-test-model");
    let state = AppState::new(
        Arc::new(provider),
        ServerConfig::default(),
        ChatTemplateProcessor::with_template("unused".to_string()),
        MlxcelTokenizer::stub_all_byte_fallback(),
        model_path.clone(),
        metrics,
    );
    Harness {
        server: InProcessServer {
            state,
            observability: Arc::new(BatchObservability::new()),
            model_path,
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        },
        handle,
        options,
        prompts,
    }
}

fn request(prompt: Value) -> CompletionRequest {
    completion_request_from_json(json!({
        "model": "completion-contract-test-model",
        "prompt": prompt,
        "max_tokens": 4,
    }))
    .expect("the representation must parse before execution validation")
}

#[test]
fn completion_prompt_string_and_singleton_array_dispatch_identical_prompt_bytes() {
    for text in ["Hello", "  안녕하세요\n\"quoted\" ", ""] {
        for mut request in [request(json!(text)), request(json!([text]))] {
            for stream in [false, true] {
                request.stream = stream;
                let h = harness();
                h.handle.token("ok");
                h.handle.finish();
                let mut output = String::new();
                let turn = h
                    .server
                    .complete(request.clone(), Arc::new(AtomicBool::new(false)), |delta| {
                        if let ChatDelta::Content(piece) = delta {
                            output.push_str(piece);
                        }
                    })
                    .expect("valid completion");
                assert_eq!(h.prompts.try_iter().collect::<Vec<_>>(), vec![text]);
                assert_eq!(h.options.try_iter().count(), 1);
                assert_eq!(output, "ok");
                assert_eq!(turn.content, "ok");
            }
        }
    }
}

#[test]
fn invalid_completion_prompts_do_not_reach_the_in_process_worker() {
    let prompts = [
        CompletionPrompt::Texts(vec![]),
        CompletionPrompt::Texts(vec!["first".into(), "second".into()]),
        CompletionPrompt::Text(" \n\t".into()),
        CompletionPrompt::Texts(vec![" \n\t".into()]),
    ];
    for prompt in prompts {
        for stream in [false, true] {
            let h = harness();
            // Let an accidental dispatch complete, so a regression fails promptly.
            h.handle.finish();
            let mut request = request(json!("valid"));
            // Direct Rust construction bypasses JSON parsing and still must
            // apply the shared execution policy inside complete().
            request.prompt = prompt.clone();
            request.stream = stream;
            let result = h
                .server
                .complete(request, Arc::new(AtomicBool::new(false)), |_| {
                    panic!("invalid input must not emit content")
                });
            assert_eq!(
                result.unwrap_err().to_string(),
                prompt.single_prompt().unwrap_err()
            );
            assert_eq!(
                h.prompts.try_iter().count(),
                0,
                "invalid prompt reached worker"
            );
            assert_eq!(h.options.try_iter().count(), 0);
        }
    }
}
