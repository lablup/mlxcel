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

//! ERNIE-Image prompt enhancer: a generative pass over `pe/` that rewrites the
//! prompt into a detailed caption, returned as `revised_prompt`.
//!
//! A port of `ErnieImagePromptEnhancer` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/prompting.py>
//! on the in-tree Ministral 3 decoder (`pe/` is a `Ministral3ForCausalLM`):
//! the user turn is the JSON `{"prompt", "width", "height"}` rendered with
//! `pe_tokenizer`'s chat template (no generation prompt), then tokens are
//! sampled at temperature 0.6, top-p 0.95 from `seed` until the EOS token.
//! The sampled text is not expected to match upstream token for token; the
//! sampler shares MLX's RNG but not upstream's exact call sequence.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::generate::SamplingConfig;
use serde_json::{Value, json};

use super::config::MAX_TEXT_TOKENS;
use super::text_encoder::load_ministral3;
use crate::imagegen::pipeline::InvalidImageRequest;
use crate::models::ministral3::Ministral3Model;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// Upstream's sampling temperature and nucleus.
pub const TEMPERATURE: f32 = 0.6;
pub const TOP_P: f32 = 0.95;
/// Upstream caps new tokens at `min(model_max_length, 2048)`.
pub const MAX_NEW_TOKENS: usize = 2048;

/// Longest rendered enhancer input, in tokens.
///
/// Upstream prefills whatever it is given, and `pe/` declares a 262144-token
/// context, so on a server one long prompt (the JSON body limit admits
/// megabytes) would be a full-context prefill on every image of the request.
/// Twice [`MAX_TEXT_TOKENS`] admits every prompt the conditioner reads in full
/// plus the chat template and the JSON wrapper around it.
pub const MAX_ENHANCER_INPUT_TOKENS: usize = 2 * MAX_TEXT_TOKENS;

/// Reject a rendered enhancer input longer than [`MAX_ENHANCER_INPUT_TOKENS`]
/// as the caller's error (`400`). Without the enhancer the conditioner
/// truncates instead, so the message names that way out.
pub fn check_enhancer_input(tokens: usize) -> Result<()> {
    if tokens > MAX_ENHANCER_INPUT_TOKENS {
        return Err(InvalidImageRequest(format!(
            "the prompt is too long for the ERNIE-Image prompt enhancer: {tokens} tokens \
             (at most {MAX_ENHANCER_INPUT_TOKENS}); shorten it or pass \
             \"use_prompt_enhancer\": false"
        ))
        .into());
    }
    Ok(())
}

/// Upstream's error when `use_prompt_enhancer` is requested without `pe/`.
pub const MISSING_ENHANCER: &str = "Prompt enhancement was requested but pe/ and pe_tokenizer/ are not present in the model snapshot";

/// Whether `root` ships an enhancer: a `pe/*.safetensors` and
/// `pe_tokenizer/tokenizer.json` (upstream `_should_enhance_prompt`).
#[must_use]
pub fn enhancer_available(root: &Path) -> bool {
    let has_weights = std::fs::read_dir(root.join("pe")).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") && !name.starts_with("._")
        })
    });
    has_weights && root.join("pe_tokenizer").join("tokenizer.json").is_file()
}

/// Upstream's rule: an explicit choice wins (and `true` without `pe/` is an
/// error); with none, enhance text-to-image when `pe/` is present and never
/// enhance an edit, whose prompt depends on a source image the enhancer cannot
/// see.
pub fn should_enhance(requested: Option<bool>, available: bool, for_edit: bool) -> Result<bool> {
    match requested {
        Some(true) if !available => bail!(MISSING_ENHANCER),
        Some(choice) => Ok(choice),
        None => Ok(available && !for_edit),
    }
}

/// `json.dumps({"prompt": p, "width": w, "height": h}, ensure_ascii=False)`:
/// Python's `", "` and `": "` separators.
pub fn enhancer_content(prompt: &str, width: u32, height: u32) -> Result<String> {
    let prompt = serde_json::to_string(prompt)?;
    Ok(format!(
        "{{\"prompt\": {prompt}, \"width\": {width}, \"height\": {height}}}"
    ))
}

/// The loaded `pe/` decoder and `pe_tokenizer/`.
pub struct ErniePromptEnhancer {
    tokenizer: MlxcelTokenizer,
    template: ChatTemplateProcessor,
    model: Ministral3Model,
    eos_ids: Vec<u32>,
}

impl ErniePromptEnhancer {
    /// Load `<root>/pe/` and `<root>/pe_tokenizer/`.
    pub fn load(root: &Path) -> Result<Self> {
        let tokenizer_dir = root.join("pe_tokenizer");
        let tokenizer = load_tokenizer(&tokenizer_dir)
            .with_context(|| format!("cannot load tokenizer from {}", tokenizer_dir.display()))?;
        let template = ChatTemplateProcessor::from_model_path(&tokenizer_dir)?
            .with_context(|| format!("{} ships no chat template", tokenizer_dir.display()))?;
        let eos = eos_token(&tokenizer_dir)?;
        let eos_id = tokenizer
            .token_to_id(&eos)
            .with_context(|| format!("eos token {eos:?} is not in the vocabulary"))?;
        let (model, _) =
            load_ministral3(&root.join("pe")).context("ERNIE-Image prompt enhancer")?;
        Ok(Self {
            tokenizer,
            template,
            model,
            eos_ids: vec![eos_id],
        })
    }

    /// The rendered enhancer prompt for `prompt` at `width x height`.
    pub fn format(&self, prompt: &str, width: u32, height: u32) -> Result<String> {
        let messages =
            json!([{ "role": "user", "content": enhancer_content(prompt, width, height)? }]);
        self.template
            .apply_raw_history_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
    }

    /// Rewrite `prompt`; sampling is seeded with `seed`. `max_new_tokens`
    /// defaults to [`MAX_NEW_TOKENS`].
    pub fn enhance(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        seed: u64,
        max_new_tokens: Option<usize>,
    ) -> Result<String> {
        let formatted = self.format(prompt, width, height)?;
        let ids = self.tokenizer.encode(&formatted, false)?;
        if ids.is_empty() {
            bail!("the ERNIE-Image enhancer prompt encodes to no tokens");
        }
        check_enhancer_input(ids.len())?;
        let mut input: Vec<i32> = ids
            .iter()
            .map(|&id| i32::try_from(id).context("token id exceeds i32"))
            .collect::<Result<_>>()?;
        let config = SamplingConfig {
            temperature: TEMPERATURE,
            top_p: TOP_P,
            ..SamplingConfig::default()
        };
        mlxcel_core::random_seed(seed);
        let mut caches = self.model.make_caches();
        let mut generated: Vec<u32> = Vec::new();
        let mut history: Vec<i32> = input.clone();
        for _ in 0..max_new_tokens.unwrap_or(MAX_NEW_TOKENS) {
            let x = mlxcel_core::from_slice_i32(&input, &[1, input.len() as i32]);
            let hidden = self
                .model
                .forward_hidden_states(&x, &mut caches, None, true)
                .map_err(anyhow::Error::msg)?;
            let width = mlxcel_core::array_shape(&hidden)[2];
            let last_pos = mlxcel_core::array_shape(&hidden)[1] - 1;
            let last = mlxcel_core::slice(&hidden, &[0, last_pos, 0], &[1, last_pos + 1, width]);
            let logits = match &self.model.lm_head {
                Some(head) => head.forward(&last),
                None => self.model.embed_tokens.as_linear(&last),
            };
            let logits = mlxcel_core::reshape(&logits, &[1, -1]);
            let (token, _) =
                mlxcel_core::sampling::sample_token_optimized(&logits, &config, &history);
            mlxcel_core::try_eval(&token).context("ERNIE-Image prompt enhancement failed")?;
            let id = mlxcel_core::item_i32(&token);
            let id = u32::try_from(id).context("sampled a negative token id")?;
            if self.eos_ids.contains(&id) {
                break;
            }
            generated.push(id);
            history.push(id as i32);
            input = vec![id as i32];
        }
        Ok(self.tokenizer.decode(&generated, true)?.trim().to_string())
    }
}

/// `eos_token` of `tokenizer_config.json`, as a string or `{content: ...}`.
fn eos_token(dir: &Path) -> Result<String> {
    let path = dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let config: Value =
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))?;
    match config.get("eos_token") {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Object(o)) => o
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .context("eos_token has no content"),
        _ => bail!("{} declares no eos_token", path.display()),
    }
}

#[cfg(test)]
#[path = "prompt_enhancer_tests.rs"]
mod tests;
