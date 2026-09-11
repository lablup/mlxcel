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

//! LLM-jp-VL generation-prompt completion.
//!
//! The shipped `chat_template.jinja` ends an `add_generation_prompt` render at
//! `<|start|>assistant`; the checkpoint's own `LLMjpVLProcessor.
//! apply_chat_template` then appends `<|channel|>final<|message|>`. A bare
//! template render is therefore an incomplete prompt.
//!
//! Both front ends build their prompt through
//! [`ChatTemplateProcessor::from_model_path`] (the CLI in
//! `commands::generate::load_cli_prompt`, the server in
//! `server::startup::resolve_chat_template`), so completing the render inside
//! the processor is what makes the two agree. These gates check the completion
//! on the pinned template text and then, when a checkpoint is downloaded, on
//! the file the checkpoint actually ships.

use std::path::PathBuf;

use super::chat_template::{ChatMessage, ChatTemplateProcessor, GenerationPromptSuffix};
use crate::multimodal::llmjp_vl_prompt::{LLMJP_ASSISTANT_OPENER, LLMJP_GENERATION_PROMPT_SUFFIX};
use crate::server::types::ChatCompletionRequest;

/// The Harmony template both released checkpoints ship, verbatim.
const LLMJP_TEMPLATE: &str = "{%- set ns = namespace(system_message='You are LLM-jp-VL, a Multimodal LLM trained by LLM-jp.', last_assistant_idx=-1) -%}{%- for message in messages -%}{%- if message['role'] == 'system' -%}{%- set ns.system_message = message['content'] -%}{%- elif message['role'] == 'assistant' -%}{%- set ns.last_assistant_idx = loop.index0 -%}{%- endif -%}{%- endfor -%}<|start|>system<|message|>{{ ns.system_message }}<|end|>{%- for message in messages -%}{%- if message['role'] == 'user' -%}<|start|>user<|message|>{{ message['content'] }}<|end|>{%- elif message['role'] == 'assistant' -%}{%- if not add_generation_prompt and loop.index0 == ns.last_assistant_idx -%}<|start|>assistant<|channel|>final<|message|>{{ message['content'] }}<|return|>{%- else -%}<|start|>assistant<|channel|>final<|message|>{{ message['content'] }}<|end|>{%- endif -%}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}";

fn llmjp_rule() -> GenerationPromptSuffix {
    GenerationPromptSuffix {
        opener: LLMJP_ASSISTANT_OPENER,
        suffix: LLMJP_GENERATION_PROMPT_SUFFIX,
    }
}

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: text.to_string(),
    }]
}

#[test]
fn a_generation_render_is_completed_with_the_final_channel() {
    let processor = ChatTemplateProcessor::with_template(LLMJP_TEMPLATE.to_string())
        .with_generation_prompt_suffix(llmjp_rule());
    let rendered = processor
        .apply(&user("この画像について説明してください。"), None)
        .expect("template renders");

    assert!(
        rendered.ends_with("<|start|>assistant<|channel|>final<|message|>"),
        "generation prompt tail: {rendered:?}"
    );
    // The suffix is appended once, not doubled onto the user turn.
    assert_eq!(rendered.matches("<|channel|>").count(), 1);
    assert!(rendered.contains("<|start|>user<|message|>この画像について説明してください。<|end|>"));
}

#[test]
fn a_history_render_is_left_alone() {
    // The prompt cache keys its boundary snapshot on the history render, whose
    // last assistant turn is already terminated; appending a channel opener
    // there would both corrupt the prefix and break prefix-stability.
    let processor = ChatTemplateProcessor::with_template(LLMJP_TEMPLATE.to_string())
        .with_generation_prompt_suffix(llmjp_rule());
    let history = processor
        .apply_history_with_kwargs(&user("hello"), None, &Default::default())
        .expect("template renders");
    assert!(
        !history.ends_with(LLMJP_GENERATION_PROMPT_SUFFIX),
        "{history:?}"
    );
    assert!(history.ends_with("<|end|>"), "{history:?}");
}

#[test]
fn a_template_without_the_rule_is_untouched() {
    // Other Harmony templates (gpt-oss) also end a generation render at
    // `<|start|>assistant` and must not be forced onto the final channel, which
    // is why the rule is keyed on the model family rather than on the tail text.
    let processor = ChatTemplateProcessor::with_template(LLMJP_TEMPLATE.to_string());
    assert!(processor.generation_prompt_suffix().is_none());
    let rendered = processor.apply(&user("hello"), None).expect("renders");
    assert!(rendered.ends_with(LLMJP_ASSISTANT_OPENER), "{rendered:?}");
    assert!(!rendered.ends_with(LLMJP_GENERATION_PROMPT_SUFFIX));
}

/// The server render must land on the same prompt as the CLI's.
///
/// The two front ends reach the template differently: the CLI hands it a
/// string `content`, while an OpenAI request carries a content *list* with an
/// `image_url` part. A template that never inspects typed media items (this one
/// never says `image` at all) expects the string, and handing it the list makes
/// minijinja print the list into the prompt. `prepare_chat_request` flattens
/// for exactly that case, so the assertion here is byte equality with the CLI
/// render rather than a token count, which two different prompts can share.
#[tokio::test]
async fn the_server_image_request_renders_the_same_prompt_as_the_cli() {
    let processor = ChatTemplateProcessor::with_template(LLMJP_TEMPLATE.to_string())
        .with_generation_prompt_suffix(llmjp_rule());

    let cli = processor
        .apply(&user("この画像の色は何色ですか。"), None)
        .expect("CLI render");

    let request = image_request("この画像の色は何色ですか。");
    let prepared = crate::server::chat_request::prepare_chat_request(&processor, &request, None)
        .await
        .expect("server render");

    assert_eq!(prepared.prompt, cli);
    assert!(
        prepared.prompt.ends_with(LLMJP_GENERATION_PROMPT_SUFFIX),
        "server prompt tail: {:?}",
        prepared.prompt
    );
    assert!(
        !prepared.prompt.contains("image_url") && !prepared.prompt.contains("\"type\""),
        "the content list must not be printed into the prompt: {:?}",
        prepared.prompt
    );
    assert_eq!(prepared.image_data.len(), 1, "the image is still extracted");
}

/// One user turn carrying an image part and a question, in OpenAI transport
/// shape.
fn image_request(question: &str) -> ChatCompletionRequest {
    use crate::server::types::{ContentPart, ImageUrl, Message, MessageContent, Role};

    ChatCompletionRequest {
        model: "llmjpvl".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::ImageUrl {
                    image_url: ImageUrl::new("data:image/png;base64,aGVsbG8=".to_string()),
                },
                ContentPart::Text {
                    text: question.to_string(),
                },
            ]),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        }],
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: None,
        cache_prompt: None,
        user: None,
        reasoning_effort: None,
        extra_body_fields: serde_json::Map::new(),
        response_format: None,
        params: crate::server::types::SamplingParams::default(),
    }
}

/// Locate a released checkpoint, soft-skipping when it is not downloaded.
fn gate_checkpoint(env_key: &str, store_repo: &str, local_dirs: &[&str]) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var(env_key) {
        let dir = PathBuf::from(dir);
        if dir.join("config.json").is_file() {
            return Some(dir);
        }
        return None;
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates: Vec<PathBuf> = crate::downloader::model_dir(store_repo)
        .into_iter()
        .collect();
    candidates.extend(local_dirs.iter().map(|dir| manifest.join(dir)));
    candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file())
}

/// The real gate: build the processor the way both front ends build it, from
/// the checkpoint directory, and check the tail of what it renders. A
/// checkpoint whose template or `model_type` changed would fail here rather
/// than quietly serving a prompt with no channel.
#[test]
fn both_released_checkpoints_render_a_completed_generation_prompt() {
    let cases: [(&str, &str, &[&str]); 2] = [
        (
            "MLXCEL_TEST_JAGLE_VL_DIR",
            "llm-jp/Jagle-VL-2.2B-Jagle-FineVision",
            &[
                "models/mlx/jagle-vl-2.2b-jagle-finevision",
                "models/Jagle-VL-2.2B-Jagle-FineVision",
            ],
        ),
        (
            "MLXCEL_TEST_LLMJP_4VL_9B_DIR",
            "llm-jp/llm-jp-4-vl-9B-beta",
            &[
                "models/mlx/llm-jp-4-vl-9b-beta",
                "models/llm-jp-4-vl-9B-beta",
            ],
        ),
    ];

    let mut checked = 0usize;
    for (env_key, repo, dirs) in cases {
        let Some(dir) = gate_checkpoint(env_key, repo, dirs) else {
            eprintln!("skipping real-checkpoint gate: {repo} not present");
            continue;
        };
        checked += 1;
        let processor = ChatTemplateProcessor::from_model_path(&dir)
            .expect("template loads")
            .expect("checkpoint ships a chat template");
        assert!(
            processor.generation_prompt_suffix().is_some(),
            "{repo}: the completion rule must be installed from the model type"
        );
        let rendered = processor
            .apply(&user("この画像について説明してください。"), None)
            .expect("template renders");
        assert!(
            rendered.ends_with("<|start|>assistant<|channel|>final<|message|>"),
            "{repo}: generation prompt tail: {rendered:?}"
        );
    }

    if checked == 0 {
        eprintln!("skipping: neither LLM-jp-VL checkpoint is downloaded");
    }
}
