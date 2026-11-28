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

//! DeepSeek-VL2's processor selects the conversation by `sft_format` and
//! tokenizes one message at a time in `format_messages_v2`. Joining turns
//! before BPE can merge the assistant colon with the first answer token.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::{path::Path, sync::Arc};

use super::chat_template::TemplateRejection;
use crate::tokenizer::MlxcelTokenizer;

const EOS: &str = "<｜end▁of▁sentence｜>";

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Format {
    #[default]
    Deepseek,
    Deepseekv2,
    Plain,
}

#[derive(Clone)]
pub(crate) struct Renderer {
    format: Format,
    pub tokenizer: Option<Arc<MlxcelTokenizer>>,
}

impl Renderer {
    pub fn from_dir(path: &Path) -> Result<Self> {
        #[derive(Deserialize, Default)]
        struct Processor {
            #[serde(default)]
            sft_format: Format,
        }
        let config: Processor = match std::fs::read(path.join("processor_config.json")) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).context("DeepSeek-VL2 processor_config.json")?
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Processor::default(),
            Err(err) => return Err(err.into()),
        };
        Ok(Self {
            format: config.sft_format,
            tokenizer: None,
        })
    }

    pub fn name(&self) -> &'static str {
        match self.format {
            Format::Deepseek => "deepseek-vl2/deepseek",
            Format::Deepseekv2 => "deepseek-vl2/deepseekv2",
            Format::Plain => "deepseek-vl2/plain",
        }
    }

    pub fn turns(&self, messages: &Value, generation: bool) -> Result<Vec<String>> {
        let messages = messages
            .as_array()
            .context("expected DeepSeek-VL2 messages")?;
        let mut turns = Vec::with_capacity(messages.len());
        for (i, message) in messages.iter().enumerate() {
            let content = message_content(&message["content"])?;
            if message["role"] == "user"
                && message["content"].is_string()
                && content.contains("<image>")
            {
                return Err(TemplateRejection::new(
                    "DeepSeek-VL2 user text has an image marker without an attached image",
                )
                .into());
            }
            let text = content.trim();
            let last = i + 1 == messages.len();
            let turn = match message["role"].as_str() {
                Some("system") if i == 0 => text.to_owned(),
                Some("assistant") => {
                    ensure!(
                        !text.contains("<image>"),
                        "DeepSeek-VL2 assistant text contains an image"
                    );
                    if last && text.is_empty() {
                        String::new()
                    } else {
                        format!("{text}{EOS}")
                    }
                }
                Some("user") => match self.format {
                    Format::Deepseek => {
                        let mut question = String::from("<|User|>:");
                        if !text.is_empty() {
                            question.push(' ');
                            question.push_str(text);
                            question.push_str("\n\n");
                        }
                        if generation || !last {
                            question.push_str("<|Assistant|>:");
                        }
                        question
                    }
                    // The registered roles are "|<User>|" and "|<Assistant>|".
                    // Neither matches conversation.py's literal "User" branch.
                    Format::Deepseekv2 if !text.is_empty() => format!("{text}{EOS}"),
                    Format::Deepseekv2 | Format::Plain => text.to_owned(),
                },
                role => bail!("invalid DeepSeek-VL2 role {role:?} at turn {i}"),
            };
            turns.push(turn);
        }
        Ok(turns)
    }
}

/// Image parts retain their position. An explicit set of markers in the text
/// already locates the supplied images; adding the typed markers would double it.
fn message_content(content: &Value) -> Result<String> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    let parts = content
        .as_array()
        .context("expected DeepSeek-VL2 text or content parts")?;
    let images = parts.iter().filter(|part| part["type"] == "image").count();
    let explicit: usize = parts
        .iter()
        .filter_map(|part| part["text"].as_str())
        .map(|text| text.matches("<image>").count())
        .sum();
    if explicit != 0 && explicit != images {
        return Err(TemplateRejection::new(&format!(
            "DeepSeek-VL2 image markers ({explicit}) do not match attached images ({images})"
        ))
        .into());
    }
    let mut text = String::new();
    for part in parts {
        if part["type"] == "image" && explicit == 0 {
            text.push_str("<image>\n");
        } else if let Some(value) = part["text"].as_str() {
            text.push_str(value);
        }
    }
    Ok(text)
}

pub(crate) fn encode_turns(tokenizer: &MlxcelTokenizer, turns: &[String]) -> Result<Vec<i32>> {
    let bos = tokenizer
        .token_to_id("<｜begin▁of▁sentence｜>")
        .context("missing DeepSeek-VL2 BOS")?;
    let mut ids = vec![i32::try_from(bos)?];
    for turn in turns {
        ids.extend(
            tokenizer
                .encode(turn, false)?
                .into_iter()
                .map(|id| id as i32),
        );
    }
    Ok(ids)
}
