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

//! Chat template processing for OpenAI-compatible messages
//!
//! Applies Jinja2 chat templates from tokenizer_config.json to format
//! conversation messages into model-specific prompts.

use std::path::Path;

use anyhow::{Context, Result};
use minijinja::{Environment, ErrorKind, Value};
use serde::{Deserialize, Serialize};

use super::chat_template_kwargs::ChatTemplateKwargs;
use super::types::request::Tool;

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat template processor
#[derive(Clone)]
pub struct ChatTemplateProcessor {
    template: String,
    bos_token: String,
    eos_token: String,
    add_generation_prompt: bool,
    /// Cached result of `supports_tools()` introspection.
    /// `None` means not yet computed.
    supports_tools_cached: Option<bool>,
    /// Default value for the `enable_thinking` Jinja kwarg when the request
    /// (or its server-side merge with CLI/env defaults) does not provide one.
    ///
    /// Mirrors upstream `TokenizerWrapper.apply_chat_template`'s
    /// `enable_thinking=self.has_thinking` default added in mlx-lm PR #1114
    /// Plumbed in by `server::startup::resolve_chat_template`
    /// from [`crate::tokenizer::ThinkingMarkers::has_thinking`] right after
    /// the tokenizer loads — so a thinking model defaults to `enable_thinking=true`
    /// while a non-thinking model defaults to `false`.
    ///
    /// `false` is the conservative earlier default and is kept for any
    /// path that constructs a processor without a corresponding tokenizer
    /// (template-string overrides, `Default`, tests).
    default_enable_thinking: bool,
}

impl ChatTemplateProcessor {
    /// Create a new processor by loading template from tokenizer_config.json or chat_template.jinja
    pub fn from_model_path(model_path: &Path) -> Result<Option<Self>> {
        let config_path = model_path.join("tokenizer_config.json");
        let config: Option<serde_json::Value> = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read {:?}", config_path))?;
            Some(
                serde_json::from_str(&content)
                    .with_context(|| "Failed to parse tokenizer_config.json")?,
            )
        } else {
            None
        };

        // Try tokenizer_config.json "chat_template" field first, then chat_template.jinja file.
        // Some models (e.g. Gemma 4 MLX Community quantizations) have an empty string
        // for chat_template in tokenizer_config.json but ship a separate .jinja file.
        let template = config
            .as_ref()
            .and_then(|c| c.get("chat_template"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .or_else(|| {
                let jinja_path = model_path.join("chat_template.jinja");
                std::fs::read_to_string(jinja_path).ok()
            });

        let Some(template) = template else {
            return Ok(None);
        };

        // Extract special tokens
        let bos_token = config
            .as_ref()
            .and_then(|c| extract_token(c, "bos_token"))
            .unwrap_or_default();
        let eos_token = config
            .as_ref()
            .and_then(|c| extract_token(c, "eos_token"))
            .unwrap_or_default();

        Ok(Some(Self {
            template: preprocess_template(template),
            bos_token,
            eos_token,
            add_generation_prompt: true,
            supports_tools_cached: None,
            default_enable_thinking: false,
        }))
    }

    /// Create a processor with a custom template string
    pub fn with_template(template: String) -> Self {
        Self {
            template: preprocess_template(template),
            bos_token: String::new(),
            eos_token: String::new(),
            add_generation_prompt: true,
            supports_tools_cached: None,
            default_enable_thinking: false,
        }
    }

    /// Set whether to add a generation prompt at the end
    pub fn set_add_generation_prompt(&mut self, add: bool) {
        self.add_generation_prompt = add;
    }

    /// Set the default value of the `enable_thinking` Jinja kwarg.
    ///
    /// Mirrors upstream `TokenizerWrapper.apply_chat_template`'s
    /// `enable_thinking=self.has_thinking` defaulting (mlx-lm PR #1114).
    /// Callers should pass `true` when the underlying tokenizer recognizes
    /// any think marker pair (single or multi token) — see
    /// [`crate::tokenizer::ThinkingMarkers::has_thinking`].
    ///
    /// The default is `false` for backward compatibility with earlier
    /// behavior so callers that do not explicitly enable this still see the
    /// historic default; `server::startup` opts into the upstream-aligned
    /// default for thinking models.
    pub fn set_default_enable_thinking(&mut self, value: bool) {
        self.default_enable_thinking = value;
    }

    /// Read the default value of the `enable_thinking` Jinja kwarg.
    ///
    /// Used by:
    /// `server::chat_template::build_template_context` (chat-template
    /// rendering path) and tests that verify the upstream-aligned default.
    pub fn default_enable_thinking(&self) -> bool {
        self.default_enable_thinking
    }

    /// Raw Jinja template source post-preprocessing.
    ///
    /// Used by the prompt-cache key composer to hash
    /// a stable identifier of the rendering pipeline into `template_sig`. The
    /// returned slice is the exact string used as the minijinja template, so
    /// any non-determinism that depends on template text will be captured.
    ///
    /// Used by: server/prompt_cache/key::template_sig
    pub fn template_source(&self) -> &str {
        &self.template
    }

    /// Detect a Gemma-4-style chat template whose `enable_thinking=true`
    /// branch produces a generation prompt ending at `<|turn>model\n`
    /// (no `<|channel>thought` priming).
    ///
    /// Without a priming marker, quantized Gemma 4 produces degenerate
    /// first-token logits when the system turn also carries non-trivial
    /// tool declarations ("own- own-", "fear- own-" style loops). The
    /// companion [`Self::patch_gemma4_generation_prompt`] fixes this by
    /// appending an OPEN `<|channel>thought\n` marker so the model reliably
    /// enters its reasoning channel and can emit `<|tool_call>` blocks
    /// afterwards.
    ///
    /// The detection is intentionally conservative: it requires BOTH the
    /// `<|channel>thought` marker AND the `not enable_thinking` guard that
    /// define this branch, so Qwen3's `<think>` path and other non-Gemma
    /// templates are unaffected.
    fn enable_thinking_drops_priming(&self) -> bool {
        self.template.contains("<|channel>thought") && self.template.contains("not enable_thinking")
    }

    /// Append an open `<|channel>thought\n` priming to a Gemma-4-rendered
    /// prompt when the template produced the un-primed `<|turn>model\n`
    /// ending. Returns the input unchanged when the template is not
    /// Gemma-4 style, `enable_thinking` is not `true`, or the rendered
    /// prompt already ends with a priming marker.
    fn patch_gemma4_generation_prompt(
        &self,
        rendered: String,
        kwargs: &ChatTemplateKwargs,
    ) -> String {
        if !self.add_generation_prompt || !self.enable_thinking_drops_priming() {
            return rendered;
        }
        let thinking_on = match kwargs
            .get("enable_thinking")
            .and_then(serde_json::Value::as_bool)
        {
            Some(v) => v,
            None => self.default_enable_thinking,
        };
        if !thinking_on {
            return rendered;
        }
        // Only patch the exact ending the un-primed branch produces. Any
        // trailing whitespace, extra marker, or pre-existing priming means
        // the prompt is already in a shape we don't want to touch.
        const BAD_END: &str = "<|turn>model\n";
        if !rendered.ends_with(BAD_END) {
            return rendered;
        }
        tracing::debug!(
            "Gemma 4: appending open <|channel>thought\\n priming for enable_thinking=true \
             (raw template leaves the prompt at <|turn>model\\n, which destabilizes first \
             token logits and suppresses tool-call emission)"
        );
        let mut out = rendered;
        out.push_str("<|channel>thought\n");
        out
    }

    /// Return whether the template uses the `tools` variable, without caching.
    ///
    /// Uses a conservative string-based heuristic so it can be called from
    /// contexts where only a shared reference is available (e.g. the `/health`
    /// route handler, which holds `Arc<ChatTemplateProcessor>`).
    ///
    /// Returns `true` when the template source contains well-known tool-related
    /// markers.  May produce both false-negatives and (rarely) false-positives —
    /// e.g., when both `tools` and `function` appear in unrelated parts of the
    /// template such as comments.  Use `supports_tools()` for ground-truth
    /// introspection when `&mut self` is available.
    ///
    /// Used by: routes/health
    pub fn supports_tools_hint(&self) -> bool {
        self.template_mentions_tools()
    }

    /// Conservative string-based heuristic for whether the Jinja template
    /// references the `tools` variable.  Shared between [`supports_tools_hint`]
    /// (cheap, `&self`) and [`compute_supports_tools`] (used as a fallback
    /// when the render-comparison probe errors out).
    fn template_mentions_tools(&self) -> bool {
        self.template.contains("for tool in tools")
            || self.template.contains("tools | tojson")
            || self.template.contains("tools|tojson")
            || (self.template.contains("tools") && self.template.contains("function"))
    }

    /// Check if the template uses the `tools` variable.
    ///
    /// Returns true when the template produces different output when `tools`
    /// is set vs. `None`.  The result is computed once by rendering the
    /// template with a sentinel tool and comparing output, then cached.
    ///
    /// Falls back to the string-heuristic if template rendering fails.
    pub fn supports_tools(&mut self) -> bool {
        if let Some(cached) = self.supports_tools_cached {
            return cached;
        }
        let result = self.compute_supports_tools();
        self.supports_tools_cached = Some(result);
        result
    }

    /// Inner computation for `supports_tools` — tries template rendering
    /// introspection and falls back to string heuristics on failure.
    fn compute_supports_tools(&self) -> bool {
        // Sentinel tool used for the probe render
        let sentinel_tool = Tool {
            tool_type: "function".to_string(),
            function: super::types::request::FunctionDefinition {
                name: "__test__".to_string(),
                description: Some("test".to_string()),
                parameters: None,
            },
        };

        let probe_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        let with_tools = self.apply(&probe_messages, Some(&[sentinel_tool]));
        let without_tools = self.apply(&probe_messages, None);

        match (with_tools, without_tools) {
            (Ok(with), Ok(without)) => with != without,
            _ => {
                // Rendering failed — fall back to string heuristic
                self.template_mentions_tools()
            }
        }
    }

    /// Check if the template handles multimodal content with image items.
    ///
    /// Returns true when the Jinja2 template iterates over content items and
    /// checks for `type == 'image'`, as Gemma3 VLM templates do.  Templates
    /// without this pattern expect `content` to be a plain string.
    pub fn supports_image_content(&self) -> bool {
        self.template.contains("'image'") || self.template.contains("\"image\"")
    }

    /// Check if the template handles multimodal content with video items.
    ///
    /// Returns true when the Jinja2 template iterates over content items and
    /// checks for `type == 'video'`, as the Gemma 4 templates do (they emit a
    /// `<|video|>` marker per video item). Templates without this pattern
    /// expect `content` to be a plain string.
    pub fn supports_video_content(&self) -> bool {
        self.template.contains("'video'") || self.template.contains("\"video\"")
    }

    /// Check if the template handles multimodal content with audio items.
    ///
    /// Returns true when the Jinja2 template iterates over content items and
    /// checks for `type == 'audio'`, as the Gemma 4 audio templates do (they
    /// emit an `<|audio|>` marker per audio item). Templates without this
    /// pattern expect `content` to be a plain string, so the CLI keeps the
    /// audio block out of the rendered prompt and lets the per-family token
    /// expansion place it instead.
    pub fn supports_audio_content(&self) -> bool {
        self.template.contains("'audio'") || self.template.contains("\"audio\"")
    }

    /// Apply the chat template with raw JSON messages (for multimodal content).
    ///
    /// This allows passing messages with list-type content entries (e.g.,
    /// `[{"type": "image"}, {"type": "text", "text": "..."}]`) that Jinja2
    /// templates like Gemma3 VLM can iterate over.
    ///
    /// When `tools` is `Some`, the tool definitions are passed to the Jinja2
    /// template context, enabling tool-calling prompt formatting.
    ///
    /// Thin wrapper that delegates to [`Self::apply_raw_with_kwargs`] with
    /// empty kwargs. Preserved for legacy callers and tests.
    // Used by: chat_request, routes/chat
    pub fn apply_raw(
        &self,
        messages: &serde_json::Value,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        self.apply_raw_with_kwargs(messages, tools, &ChatTemplateKwargs::new())
    }

    /// Apply the chat template with raw JSON messages and additional Jinja
    /// kwargs.
    ///
    /// Every key in `kwargs` is forwarded to the minijinja template context
    /// under its original name (with a handful of canonical keys such as
    /// `enable_thinking` overriding our default context entries so templates
    /// that rely on those names see the operator-provided value). This is the
    /// plumbing path used's `preserve_thinking` feature and
    /// generalizes to any future kwarg a HuggingFace chat template expects.
    ///
    /// When a kwarg key duplicates an already-provided context entry
    /// (`messages`, `bos_token`, `eos_token`, `add_generation_prompt`,
    /// `tools`), the kwarg value wins — this lets operators override defaults
    /// if a template ships with unusual expectations.
    // Used by: chat_request, routes/chat
    pub fn apply_raw_with_kwargs(
        &self,
        messages: &serde_json::Value,
        tools: Option<&[Tool]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String> {
        let mut env = Environment::new();
        configure_environment(&mut env);

        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        let tmpl = env.get_template("chat")?;

        // Convert serde_json::Value to minijinja::Value
        let messages_val = minijinja::Value::from_serialize(messages);

        // Always pass `tools` as an iterable (possibly empty) rather than
        // `None` so that `{% if tools is iterable and tools | length > 0 %}`
        // works under minijinja — its short-circuit evaluation still tries
        // to compute `| length` of `none`, which raises an error.
        let tools_val = match tools {
            Some(t) => minijinja::Value::from_serialize(t),
            None => minijinja::Value::from_serialize(Vec::<Tool>::new()),
        };

        let context = build_template_context(
            messages_val,
            &self.bos_token,
            &self.eos_token,
            self.add_generation_prompt,
            tools_val,
            kwargs,
            self.default_enable_thinking,
        );

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(self.patch_gemma4_generation_prompt(result, kwargs))
    }

    /// Apply the chat template to messages.
    ///
    /// When `tools` is `Some`, the tool definitions are passed to the Jinja2
    /// template context, enabling tool-calling prompt formatting.
    ///
    /// Thin wrapper that delegates to [`Self::apply_with_kwargs`] with empty
    /// kwargs. Preserved for legacy callers and tests.
    // Used by: chat_request, routes/chat
    pub fn apply(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> Result<String> {
        self.apply_with_kwargs(messages, tools, &ChatTemplateKwargs::new())
    }

    /// Apply the chat template with additional Jinja kwargs.
    ///
    /// See [`Self::apply_raw_with_kwargs`] for kwarg semantics. Non-thinking
    /// models silently ignore unknown kwargs (the template simply does not
    /// reference them), so it is safe to pass `preserve_thinking` or similar
    /// keys universally.
    // Used by: chat_request, routes/chat
    pub fn apply_with_kwargs(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String> {
        let mut env = Environment::new();
        configure_environment(&mut env);

        env.add_template("chat", &self.template)
            .with_context(|| "Failed to parse chat template")?;

        let tmpl = env.get_template("chat")?;

        // Many templates conditionally check variables like `tools`,
        // `enable_thinking`, etc. We must provide them with the right
        // concrete type to avoid undefined/None errors — in particular,
        // `tools` must be an iterable (even when empty) so that the common
        // `{% if tools is iterable and tools | length > 0 %}` guard used by
        // Qwen/Nemotron templates works correctly under minijinja.
        let tools_val = match tools {
            Some(t) => minijinja::Value::from_serialize(t),
            None => minijinja::Value::from_serialize(Vec::<Tool>::new()),
        };

        let messages_val = minijinja::Value::from_serialize(messages);
        let context = build_template_context(
            messages_val,
            &self.bos_token,
            &self.eos_token,
            self.add_generation_prompt,
            tools_val,
            kwargs,
            self.default_enable_thinking,
        );

        let result = tmpl
            .render(context)
            .with_context(|| "Failed to render chat template")?;

        Ok(self.patch_gemma4_generation_prompt(result, kwargs))
    }
}

/// Build the minijinja template context merging the standard chat fields with
/// caller-provided kwargs.
///
/// Standard fields (`messages`, `bos_token`, `eos_token`,
/// `add_generation_prompt`, `tools`) are always present and are **reserved**
/// from kwargs overlay — a request cannot overwrite the canonical conversation
/// or tool list by smuggling those keys through `chat_template_kwargs`.
///
/// `enable_thinking` defaults to `default_enable_thinking` (upstream PR #1114) — `true` when the underlying tokenizer recognizes a
/// think marker pair, `false` otherwise.  The default is overridable through
/// `kwargs` (the same mechanism that lets's `preserve_thinking` reach the template).  Any other kwarg key — including future
/// template-specific hints — is passed through unchanged.
fn build_template_context(
    messages: minijinja::Value,
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
    tools: minijinja::Value,
    kwargs: &ChatTemplateKwargs,
    default_enable_thinking: bool,
) -> minijinja::Value {
    // Start with the default context fields.
    let mut ctx: std::collections::BTreeMap<&str, minijinja::Value> =
        std::collections::BTreeMap::new();
    ctx.insert("messages", messages);
    ctx.insert("bos_token", minijinja::Value::from(bos_token));
    ctx.insert("eos_token", minijinja::Value::from(eos_token));
    ctx.insert(
        "add_generation_prompt",
        minijinja::Value::from(add_generation_prompt),
    );
    ctx.insert("tools", tools);
    // Tokenizer-derived default: `has_thinking` for thinking models, `false`
    // otherwise (upstream PR #1114). Overridden below by any
    // matching kwarg from the request / server-default merge.
    ctx.insert(
        "enable_thinking",
        minijinja::Value::from(default_enable_thinking),
    );

    // Overlay kwargs. Canonical prompt-construction keys are reserved so a
    // client cannot smuggle a replacement `messages` array, swap `tools`, or
    // flip `add_generation_prompt` through the request-side kwargs pass-through.
    // Only `enable_thinking` (an intentional, tested override point) and any
    // non-reserved key (e.g. `preserve_thinking`, future template hints)
    // actually reach the Jinja context.
    //
    // We rebuild the map with owned String keys after merging so the final
    // minijinja::Value owns all its entries.
    const RESERVED_KEYS: &[&str] = &[
        "messages",
        "bos_token",
        "eos_token",
        "add_generation_prompt",
        "tools",
    ];
    let mut owned: std::collections::BTreeMap<String, minijinja::Value> =
        ctx.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    // Security (M-1): collect all reserved-key override attempts into a
    // single bounded log line to prevent log-amplification DoS via
    // attacker-controlled very-long kwargs keys. Keys are truncated to
    // 64 chars before logging.
    let mut dropped_keys: Vec<String> = Vec::new();
    for (k, v) in kwargs.as_map() {
        if RESERVED_KEYS.contains(&k.as_str()) {
            dropped_keys.push(truncate_key_for_log(k));
            continue;
        }
        owned.insert(k.clone(), minijinja::Value::from_serialize(v));
    }
    if !dropped_keys.is_empty() {
        tracing::warn!(
            "chat_template_kwargs keys [{}] are reserved for the server-managed \
             template context and will be ignored; remove them from the request \
             body or server default",
            dropped_keys.join(", ")
        );
    }
    minijinja::Value::from_serialize(&owned)
}

/// Security (M-1): truncate an attacker-controlled kwargs key to at most 64
/// Unicode scalar values before including it in a log message.
///
/// An adversary can supply kwargs keys that are hundreds of kilobytes long.
/// Logging them verbatim (e.g. via `{:?}`) multiplies that size by the number
/// of reserved keys checked per request.  Bounding the logged representation
/// keeps each log record small regardless of input.
fn truncate_key_for_log(key: &str) -> String {
    const MAX_CHARS: usize = 64;
    let mut chars = key.chars();
    let truncated: String = chars.by_ref().take(MAX_CHARS).collect();
    if chars.next().is_some() {
        // Key had more than MAX_CHARS chars — append ellipsis.
        format!("{truncated}\u{2026}")
    } else {
        truncated
    }
}

/// Strip HuggingFace `transformers` Jinja2 extensions that minijinja does not
/// implement.
///
/// `{% generation %}...{% endgeneration %}` is a transformers extension used
/// to mark assistant-generation regions for token-level masking. It has no
/// effect on the rendered prompt, so we simply drop both delimiters so the
/// template parses cleanly under minijinja. Affected models: SmolLM3, and
/// any other HF template that adopts this marker.
fn preprocess_template(template: String) -> String {
    // Handle all whitespace-control variants of the two delimiters. Using
    // explicit replaces keeps this allocation-light and regex-free.
    let variants: &[&str] = &[
        "{% generation %}",
        "{%- generation %}",
        "{% generation -%}",
        "{%- generation -%}",
        "{% endgeneration %}",
        "{%- endgeneration %}",
        "{% endgeneration -%}",
        "{%- endgeneration -%}",
    ];
    let mut out = template;
    for v in variants {
        if out.contains(v) {
            out = out.replace(v, "");
        }
    }
    out
}

/// Configure a minijinja environment with common settings and Python-compat methods.
fn configure_environment(env: &mut Environment<'_>) {
    env.set_keep_trailing_newline(true);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    // Bound rendering cost as defense-in-depth against denial-of-service from a
    // pathological chat template (e.g. deeply nested or effectively unbounded
    // `{% for %}` loops). `set_fuel` caps the number of VM instructions a single
    // render may execute; once the budget is exhausted minijinja returns an
    // `ErrorKind::OutOfFuel` error rather than spinning on CPU/memory. Both
    // render call sites propagate that error via `Result` (`apply*`), and the
    // load-time `supports_tools` probe already degrades to a string heuristic on
    // render failure, so exhaustion never panics on any path.
    //
    // Template source is operator-controlled today (the model the operator loads
    // at startup, or the `--chat-template` flag), so this is a low-severity
    // preventive control; it becomes materially important in a multi-tenant
    // deployment where untrusted parties could cause arbitrary models — and thus
    // arbitrary templates — to be loaded.
    //
    // The budget is deliberately generous. Real templates — including tool-rich
    // Qwen/Nemotron-style templates rendered over long multi-turn histories —
    // execute well under ~1M instructions (verified against every locally
    // available model template via `test_all_local_model_templates_render`),
    // while an unbounded loop blows past 50M almost immediately, bounding a
    // malicious render to a fraction of a second of CPU instead of forever. Note
    // that fuel bounds instruction *count*, not output size, so it does not by
    // itself cap memory from a single very large emit; that is out of scope.
    env.set_fuel(Some(50_000_000));

    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    env.add_function("strftime_now", |_format: String| -> String {
        chrono::Utc::now().format("%d %b %Y").to_string()
    });

    // Handle Python methods not natively supported by minijinja.
    //
    // Many HuggingFace chat templates use Python-style string and dict
    // methods (`.split()`, `.strip()`, `.startswith()`, `.get()`, `.items()`
    // …) that are standard in Python/Jinja2 but not built into minijinja's
    // value types. Without these shims, rendering silently falls back to
    // the `to_prompt()` "User: ... Assistant:" format, and the model echoes
    // those labels instead of producing a real response.
    //
    // Scope: cover every method encountered in the chat templates we ship
    // against (Gemma 4, Qwen 3, GLM 4/5, Jamba, Exaone 4, olmo 3, …).
    env.set_unknown_method_callback(|_state, value, method, args| {
        use minijinja::value::ValueKind;

        // --- String methods ---------------------------------------------
        if value.kind() == ValueKind::String {
            let s = value.as_str().unwrap_or_default();
            let arg_str = || args.first().and_then(|a| a.as_str()).unwrap_or_default();

            // `chars` argument for strip/lstrip/rstrip. `None` → whitespace.
            let strip_chars = args.first().and_then(|a| a.as_str());
            let strip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim().to_string(),
                    Some(chars) => s.trim_matches(|c: char| chars.contains(c)).to_string(),
                }
            };
            let lstrip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim_start().to_string(),
                    Some(chars) => s
                        .trim_start_matches(|c: char| chars.contains(c))
                        .to_string(),
                }
            };
            let rstrip_matches = |s: &str| -> String {
                match strip_chars {
                    None => s.trim_end().to_string(),
                    Some(chars) => s.trim_end_matches(|c: char| chars.contains(c)).to_string(),
                }
            };

            match method {
                "split" => {
                    let sep = arg_str();
                    // Python `s.split()` with no arg splits on any whitespace
                    // and drops empty strings; `s.split(sep)` splits on sep.
                    let parts: Vec<Value> = if sep.is_empty() {
                        s.split_whitespace()
                            .map(|p| Value::from(p.to_string()))
                            .collect()
                    } else {
                        s.split(sep).map(|p| Value::from(p.to_string())).collect()
                    };
                    return Ok(Value::from(parts));
                }
                "rsplit" => {
                    let sep = arg_str();
                    let parts: Vec<Value> = if sep.is_empty() {
                        s.split_whitespace()
                            .map(|p| Value::from(p.to_string()))
                            .collect()
                    } else {
                        s.rsplit(sep).map(|p| Value::from(p.to_string())).collect()
                    };
                    return Ok(Value::from(parts));
                }
                "strip" => return Ok(Value::from(strip_matches(s))),
                "lstrip" => return Ok(Value::from(lstrip_matches(s))),
                "rstrip" => return Ok(Value::from(rstrip_matches(s))),
                "startswith" => {
                    let prefix = arg_str();
                    return Ok(Value::from(s.starts_with(prefix)));
                }
                "endswith" => {
                    let suffix = arg_str();
                    return Ok(Value::from(s.ends_with(suffix)));
                }
                "replace" => {
                    let old = args.first().and_then(|a| a.as_str()).unwrap_or_default();
                    let new = args.get(1).and_then(|a| a.as_str()).unwrap_or_default();
                    return Ok(Value::from(s.replace(old, new)));
                }
                "upper" => return Ok(Value::from(s.to_uppercase())),
                "lower" => return Ok(Value::from(s.to_lowercase())),
                "title" => {
                    // Simple title-case: uppercase first letter of each word.
                    let mut result = String::with_capacity(s.len());
                    let mut prev_alphabetic = false;
                    for ch in s.chars() {
                        if ch.is_alphabetic() {
                            if prev_alphabetic {
                                result.extend(ch.to_lowercase());
                            } else {
                                result.extend(ch.to_uppercase());
                            }
                            prev_alphabetic = true;
                        } else {
                            result.push(ch);
                            prev_alphabetic = false;
                        }
                    }
                    return Ok(Value::from(result));
                }
                "capitalize" => {
                    // Python `s.capitalize()` — uppercase first char, lowercase the rest.
                    let mut chars = s.chars();
                    let result = match chars.next() {
                        Some(first) => {
                            let mut out = String::with_capacity(s.len());
                            out.extend(first.to_uppercase());
                            out.extend(chars.flat_map(|c| c.to_lowercase()));
                            out
                        }
                        None => String::new(),
                    };
                    return Ok(Value::from(result));
                }
                "casefold" => {
                    // Python `s.casefold()` — approximation via lowercase.
                    return Ok(Value::from(s.to_lowercase()));
                }
                "swapcase" => {
                    let result: String = s
                        .chars()
                        .flat_map(|c| {
                            if c.is_uppercase() {
                                c.to_lowercase().collect::<Vec<_>>()
                            } else if c.is_lowercase() {
                                c.to_uppercase().collect::<Vec<_>>()
                            } else {
                                vec![c]
                            }
                        })
                        .collect();
                    return Ok(Value::from(result));
                }
                "join" => {
                    // Python `sep.join(iterable)` — called as method on the
                    // separator string, with the iterable as the argument.
                    let Some(iter_arg) = args.first() else {
                        return Err(minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "str.join() requires an iterable argument",
                        ));
                    };
                    let iter = iter_arg.try_iter().map_err(|_| {
                        minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "str.join() argument is not iterable",
                        )
                    })?;
                    let parts: Vec<String> = iter
                        .map(|v| {
                            v.as_str()
                                .map(String::from)
                                .unwrap_or_else(|| v.to_string())
                        })
                        .collect();
                    return Ok(Value::from(parts.join(s)));
                }
                "isdigit" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()),
                    ));
                }
                "isalpha" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_alphabetic()),
                    ));
                }
                "isalnum" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()),
                    ));
                }
                "isspace" => {
                    return Ok(Value::from(
                        !s.is_empty() && s.chars().all(|c| c.is_whitespace()),
                    ));
                }
                "isupper" => {
                    return Ok(Value::from(
                        s.chars().any(|c| c.is_uppercase()) && !s.chars().any(|c| c.is_lowercase()),
                    ));
                }
                "islower" => {
                    return Ok(Value::from(
                        s.chars().any(|c| c.is_lowercase()) && !s.chars().any(|c| c.is_uppercase()),
                    ));
                }
                "find" => {
                    let needle = arg_str();
                    let idx = s.find(needle).map(|i| i as i64).unwrap_or(-1);
                    return Ok(Value::from(idx));
                }
                "count" => {
                    let needle = arg_str();
                    if needle.is_empty() {
                        return Ok(Value::from(0_i64));
                    }
                    return Ok(Value::from(s.matches(needle).count() as i64));
                }
                _ => {}
            }
        }

        // --- Map (dict) methods -----------------------------------------
        if value.kind() == ValueKind::Map {
            match method {
                // Python-style `dict.get(key[, default])`: returns the
                // value at `key` or `default` (or Undefined) when absent.
                // Jinja2 treats Undefined as falsy, matching Python's
                // `None` semantics in `m.get('a') or m.get('b')` chains.
                "get" => {
                    let Some(key) = args.first() else {
                        return Err(minijinja::Error::new(
                            ErrorKind::InvalidOperation,
                            "dict.get() requires at least one argument",
                        ));
                    };
                    let default = args.get(1).cloned().unwrap_or(Value::UNDEFINED);
                    return match value.get_item(key) {
                        Ok(v) if v.is_undefined() => Ok(default),
                        Ok(v) => Ok(v),
                        Err(_) => Ok(default),
                    };
                }
                // Python-style `dict.items()` / `keys()` / `values()` —
                // return flat sequences usable directly in `{% for %}`.
                "items" => {
                    let mut pairs: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            let v = value.get_item(&k).unwrap_or(Value::UNDEFINED);
                            pairs.push(Value::from(vec![k, v]));
                        }
                    }
                    return Ok(Value::from(pairs));
                }
                "keys" => {
                    let mut keys: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            keys.push(k);
                        }
                    }
                    return Ok(Value::from(keys));
                }
                "values" => {
                    let mut vals: Vec<Value> = Vec::new();
                    if let Ok(iter) = value.try_iter() {
                        for k in iter {
                            if let Ok(v) = value.get_item(&k) {
                                vals.push(v);
                            }
                        }
                    }
                    return Ok(Value::from(vals));
                }
                _ => {}
            }
        }

        Err(minijinja::Error::from(ErrorKind::UnknownMethod))
    });
}

/// Extract a token from config, handling both string and object formats
fn extract_token(config: &serde_json::Value, key: &str) -> Option<String> {
    config.get(key).and_then(|v| {
        if v.is_string() {
            v.as_str().map(String::from)
        } else if v.is_object() {
            // Some configs use {"content": "<token>"}
            v.get("content").and_then(|c| c.as_str()).map(String::from)
        } else {
            None
        }
    })
}

/// Default fallback template for models without chat_template
pub fn default_chat_template() -> &'static str {
    r#"{% for message in messages %}{% if message.role == 'system' %}System: {{ message.content }}

{% elif message.role == 'user' %}User: {{ message.content }}

{% elif message.role == 'assistant' %}Assistant: {{ message.content }}

{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#
}

impl Default for ChatTemplateProcessor {
    fn default() -> Self {
        Self::with_template(default_chat_template().to_string())
    }
}

impl std::fmt::Debug for ChatTemplateProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplateProcessor")
            .field("bos_token", &self.bos_token)
            .field("eos_token", &self.eos_token)
            .field("add_generation_prompt", &self.add_generation_prompt)
            .field("supports_tools_cached", &self.supports_tools_cached)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_audio_content_detects_audio_branch() {
        // A Gemma-4-style template that handles `type == 'audio'` is detected,
        // so the CLI renders the `<|audio|>` marker into the user turn instead
        // of relying on the per-family expansion fallback (issue #436).
        let with_audio = ChatTemplateProcessor::with_template(
            "{% if item['type'] == 'audio' %}<|audio|>{% endif %}".to_string(),
        );
        assert!(with_audio.supports_audio_content());

        // A plain string-content template is not flagged as audio-capable.
        let without_audio = ChatTemplateProcessor::with_template(
            "{{ messages[0].role }}: {{ messages[0].content }}".to_string(),
        );
        assert!(!without_audio.supports_audio_content());
    }

    #[test]
    fn test_default_template() {
        let processor = ChatTemplateProcessor::default();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert!(result.contains("User: Hello"));
        assert!(result.ends_with("Assistant: "));
    }

    #[test]
    fn test_chatml_template() {
        let template = r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_with_tools_none_still_works() {
        let processor = ChatTemplateProcessor::default();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert!(result.contains("User: Hi"));
    }

    #[test]
    fn test_apply_with_tools_passes_to_template() {
        // Template that explicitly uses tools
        let template = r#"{% if tools %}Tools: {{ tools | length }}{% endif %}
{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Call a tool".to_string(),
        }];

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::server::types::request::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: None,
            },
        }];

        let result = processor.apply(&messages, Some(&tools)).unwrap();
        assert!(result.contains("Tools: 1"));
    }

    #[test]
    fn test_tojson_filter_serializes_tools_into_prompt() {
        // Regression: Qwen3-Coder's official chat template serializes the tools
        // array into the prompt with `{{ tools | tojson }}`. The `tojson` filter
        // is gated behind minijinja's `json` feature; without it the template
        // fails to render and the processor falls back to a default template
        // that drops the tools section entirely, so the model never learns
        // which tools exist and stops emitting tool calls.
        let template = r#"{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}
{% for m in messages %}{{ m.content }}{% endfor %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::server::types::request::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: None,
            },
        }];

        let result = processor.apply(&messages, Some(&tools)).unwrap();
        // The custom `<tools>[...]` markup only survives if the real template
        // rendered; a render-failure fallback would drop it. `[` confirms
        // `tojson` emitted a JSON array rather than erroring on an unknown filter.
        assert!(
            result.contains("<tools>["),
            "tojson must serialize the tools array into the prompt: {result}"
        );
        assert!(result.contains("get_weather"));
    }

    #[test]
    fn test_tool_parameter_key_order_preserved_through_render() {
        // Regression (Issue 7): serde_json's default `Map` is a `BTreeMap`,
        // which alphabetizes object keys. Tool parameter schemas therefore
        // rendered into the prompt in a DIFFERENT order than the client sent
        // (e.g. grep's `pattern, include` became `include, pattern`), shifting
        // Qwen3-Coder's tool-selection logits at temperature 0 and producing
        // worse tool choices than mlx-serve (which preserves wire order). The
        // `preserve_order` serde_json feature keeps insertion order.
        //
        // Deserialize from a wire JSON string with deliberately
        // non-alphabetical keys so the test fails if BTreeMap sorting returns.
        let tools: Vec<Tool> = serde_json::from_str(
            r#"[{"type":"function","function":{"name":"grep","parameters":{"properties":{"pattern":{"type":"string"},"include":{"type":"string"}}}}}]"#,
        )
        .unwrap();

        let template = r#"{{ tools | tojson }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "x".to_string(),
        }];

        let result = processor.apply(&messages, Some(&tools)).unwrap();
        let pattern_idx = result.find("pattern").expect("pattern key present");
        let include_idx = result.find("include").expect("include key present");
        assert!(
            pattern_idx < include_idx,
            "tool parameter keys must render in wire order (pattern before include), got: {result}"
        );
    }

    #[test]
    fn test_supports_tools_detection() {
        // Template that uses tools — rendering differs with/without tools
        let with_tools =
            r#"{% if tools %}[TOOLS]{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(with_tools.to_string());
        assert!(processor.supports_tools());
        // Second call must return cached result
        assert!(processor.supports_tools());

        // Template without tools — rendering is identical with/without tools
        let without = r#"{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(without.to_string());
        assert!(!processor.supports_tools());
    }

    #[test]
    fn test_supports_tools_caching() {
        let template =
            r#"{% if tools %}[TOOLS]{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        // Not cached yet
        assert!(processor.supports_tools_cached.is_none());
        // First call computes
        let _ = processor.supports_tools();
        // Now cached
        assert!(processor.supports_tools_cached.is_some());
    }

    #[test]
    fn test_dict_get_method_present_key() {
        // Python-style dict.get('key') should return the value when present.
        let template = r#"{{ messages[0].get('content') }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello world".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "hello world");
    }

    #[test]
    fn test_dict_get_method_missing_key_is_falsy() {
        // Missing key should be falsy so `get('a') or get('b')` works.
        let template = r#"{% if messages[0].get('missing_field') %}HAS{% else %}NONE{% endif %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "NONE");
    }

    #[test]
    fn test_dict_get_method_or_chain() {
        // `m.get('a') or m.get('b')` — the Gemma 4 idiom.
        let template = r#"{{ messages[0].get('reasoning') or messages[0].get('content') }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "fallback value".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "fallback value");
    }

    #[test]
    fn test_dict_get_method_with_default() {
        // `m.get('missing', 'default')` — two-arg form.
        let template = r#"{{ messages[0].get('missing', 'fallback') }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "fallback");
    }

    #[test]
    fn test_string_strip_default_whitespace() {
        let template = r#"{{ messages[0].content.strip() }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "  hello  ".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn test_string_strip_with_chars() {
        // Python: `"\n\nhello\n".strip('\n')` → `"hello"`
        let template = r#"{{ messages[0].content.strip('\n') }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "\n\nhello\n".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn test_string_lstrip_rstrip() {
        // Pad the output so we can assert on exact leading/trailing content
        // without being confused by minijinja's whitespace-control modes.
        let template_l = r#"[{{ messages[0].content.lstrip('\n') }}]"#;
        let template_r = r#"[{{ messages[0].content.rstrip('\n') }}]"#;
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "\n\nhello\n\n".to_string(),
        }];
        let pl = ChatTemplateProcessor::with_template(template_l.to_string());
        assert_eq!(pl.apply(&messages, None).unwrap().trim(), "[hello\n\n]");
        let pr = ChatTemplateProcessor::with_template(template_r.to_string());
        assert_eq!(pr.apply(&messages, None).unwrap().trim(), "[\n\nhello]");
    }

    #[test]
    fn test_string_startswith_endswith() {
        let template = r#"{% if messages[0].content.startswith('<tool>') %}T{% else %}F{% endif %}{% if messages[0].content.endswith('</tool>') %}T{% else %}F{% endif %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "<tool>call</tool>".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "TT");
    }

    #[test]
    fn test_string_replace() {
        let template = r#"{{ messages[0].content.replace('foo', 'bar') }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "foo and foo".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        assert_eq!(result.trim(), "bar and bar");
    }

    #[test]
    fn test_map_items_iteration() {
        // `{% for k, v in d.items() %}` — destructuring iteration.
        let template = r#"{% for k, v in messages[0].items() %}{{ k }}={{ v }};{% endfor %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let result = processor.apply(&messages, None).unwrap();
        // Order is stable for struct-derived maps (serde preserves field order).
        assert!(result.contains("role=user"));
        assert!(result.contains("content=hi"));
    }

    #[test]
    fn test_qwen3_template_thinking_chain_renders() {
        // Representative fragment from Qwen 3 / Jamba templates — uses
        // `.strip('\n')`, `.split()`, `.lstrip()`, `.startswith()`,
        // `.endswith()` all in the same expression.
        let template = r#"{%- set content = messages[0].content.split('</think>')[-1].lstrip('\n') -%}
{%- if messages[0].content.startswith('<tool_response>') and messages[0].content.endswith('</tool_response>') -%}
TOOL
{%- else -%}
{{ content.strip('\n') }}
{%- endif -%}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "<think>\nplanning\n</think>\n\nthe answer\n".to_string(),
        }];
        let result = processor
            .apply(&messages, None)
            .expect("Qwen 3 style template must render");
        assert_eq!(result.trim(), "the answer");
    }

    #[test]
    fn test_gemma4_template_renders() {
        // Regression guard for the Gemma 4 chat template, which uses
        // `message.get('reasoning')` idioms that require unknown_method_callback
        // support for dict.get().
        let template = r#"{{- bos_token -}}
{%- for message in messages -%}
    {%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}
    {%- if thinking_text -%}
        <|channel>thought
{{ thinking_text }}
<channel|>
    {%- endif -%}
    <|turn>{{ message['role'] }}
{{ message['content'] }}<turn|>
{%- endfor -%}
{%- if add_generation_prompt -%}<|turn>model
{%- endif -%}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "containerization question".to_string(),
        }];
        let result = processor
            .apply(&messages, None)
            .expect("Gemma 4 template must render successfully");
        assert!(result.contains("<|turn>user"));
        assert!(result.contains("containerization question"));
        assert!(result.contains("<|turn>model"));
    }

    #[test]
    fn test_pathological_template_is_bounded_by_fuel() {
        // Defense-in-depth: a maliciously expensive template (here, a nested
        // loop whose iteration count dwarfs the fuel budget) must be terminated
        // by minijinja's fuel ceiling rather than running until CPU/memory
        // exhaustion. The control case uses the identical template shape with
        // tiny bounds, proving the failure below is the fuel ceiling and not a
        // parse/structural error. Nested *bounded* ranges (rather than one
        // enormous `range(...)`) keep per-iteration allocation small while still
        // exceeding the budget near-instantly — each loop pass costs fuel via
        // the `Iterate` instruction even with an empty body.
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        let bounded = ChatTemplateProcessor::with_template(
            "{% for a in range(3) %}{% for b in range(3) %}{% endfor %}{% endfor %}".to_string(),
        );
        assert!(
            bounded.apply(&messages, None).is_ok(),
            "a small bounded nested loop must render within the fuel budget"
        );

        let pathological = ChatTemplateProcessor::with_template(
            "{% for a in range(100000) %}{% for b in range(100000) %}{% endfor %}{% endfor %}"
                .to_string(),
        );
        let err = pathological
            .apply(&messages, None)
            .expect_err("an unbounded loop must be terminated by the fuel budget");
        // Confirm the failure is specifically fuel exhaustion (not some other
        // render error) by inspecting the underlying minijinja error kind
        // through the anyhow context chain — asserting the kind is more precise
        // and durable than substring-matching the rendered message.
        let kind = err.downcast_ref::<minijinja::Error>().map(|e| e.kind());
        assert_eq!(
            kind,
            Some(ErrorKind::OutOfFuel),
            "expected ErrorKind::OutOfFuel, got: {err:#}"
        );
    }

    /// Renders every locally-available model's chat template with a few
    /// representative message patterns and reports all failures at the end.
    ///
    /// Marked `#[ignore]` because it requires the developer's `models/`
    /// directory, which is not present in CI. Run with:
    ///
    /// ```text
    /// cargo test --release --lib -p mlxcel test_all_local_model_templates_render -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires local models/ directory; run with --ignored"]
    fn test_all_local_model_templates_render() {
        let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        if !models_dir.exists() {
            eprintln!("skip: {} not found", models_dir.display());
            return;
        }

        // Three canonical scenarios exercising different template branches:
        //   simple          — single user turn (default path)
        //   system          — system + user (system-message branch)
        //   multi_turn_think — multi-turn with `<think>...</think>`,
        //                       exercising `.split('</think>')` + friends
        let scenarios: Vec<(&str, Vec<ChatMessage>)> = vec![
            (
                "simple",
                vec![ChatMessage {
                    role: "user".into(),
                    content: "Hello, what is 2+2?".into(),
                }],
            ),
            (
                "system",
                vec![
                    ChatMessage {
                        role: "system".into(),
                        content: "You are a helpful assistant.".into(),
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: "Hi".into(),
                    },
                ],
            ),
            (
                "multi_turn_think",
                vec![
                    ChatMessage {
                        role: "user".into(),
                        content: "What is 2+2?".into(),
                    },
                    ChatMessage {
                        role: "assistant".into(),
                        content: "<think>\nLet me calculate.\n</think>\n\n2+2 equals 4.".into(),
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: "And 3+3?".into(),
                    },
                ],
            ),
        ];

        // Errors raised by the template itself via `raise_exception(...)`
        // are intentional rejections (e.g. "system role not supported",
        // "roles must alternate"). They indicate the template is WORKING,
        // just for a scenario it explicitly refuses. We want the audit to
        // flag only genuine engine/unknown-method failures.
        let is_intentional_reject = |err: &str| -> bool {
            const INTENTIONAL_PHRASES: &[&str] = &[
                "System role not supported",
                "System messages not supported",
                "must alternate",
                "Only user and assistant roles",
                "does not support tool",
            ];
            INTENTIONAL_PHRASES.iter().any(|p| err.contains(p))
        };

        let mut failures: Vec<(String, String, String)> = Vec::new();
        let mut intentional: Vec<(String, String, String)> = Vec::new();
        let mut passed = 0usize;
        let mut skipped = 0usize;
        let mut models_checked = 0usize;

        let mut entries: Vec<_> = std::fs::read_dir(&models_dir)
            .expect("read models/")
            .flatten()
            .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let model_name = path.file_name().unwrap().to_string_lossy().to_string();

            let processor = match ChatTemplateProcessor::from_model_path(&path) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    skipped += 1;
                    continue; // no chat template shipped
                }
                Err(e) => {
                    failures.push((model_name.clone(), "load".into(), format!("{e:#}")));
                    continue;
                }
            };

            models_checked += 1;

            for (scenario_name, messages) in &scenarios {
                match processor.apply(messages, None) {
                    Ok(_) => passed += 1,
                    Err(e) => {
                        let msg = format!("{e:#}");
                        if is_intentional_reject(&msg) {
                            intentional.push((
                                model_name.clone(),
                                (*scenario_name).to_string(),
                                msg,
                            ));
                        } else {
                            failures.push((model_name.clone(), (*scenario_name).to_string(), msg));
                        }
                    }
                }
            }
        }

        eprintln!();
        eprintln!("=== chat template render audit ===");
        eprintln!("models checked:          {models_checked}");
        eprintln!("scenarios passed:        {passed}");
        eprintln!("intentional rejections:  {}", intentional.len());
        eprintln!("skipped (no template):   {skipped}");
        eprintln!("failures:                {}", failures.len());

        if !intentional.is_empty() {
            eprintln!();
            eprintln!("--- intentional template rejections (expected) ---");
            for (model, scenario, err) in &intentional {
                let first_line = err.lines().next().unwrap_or("");
                eprintln!("  {model}::{scenario} — {first_line}");
            }
        }

        if !failures.is_empty() {
            eprintln!();
            eprintln!("--- failures ---");
            for (model, scenario, err) in &failures {
                let first_line = err.lines().next().unwrap_or("");
                eprintln!("  {model}::{scenario} — {first_line}");
            }
            panic!(
                "{} model/scenario combinations failed to render",
                failures.len()
            );
        }
    }

    #[test]
    fn test_apply_raw_with_tools() {
        let template = r#"{% if tools %}[TOOLS]{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}"#;

        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = serde_json::json!([
            {"role": "user", "content": "hi"}
        ]);

        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::server::types::request::FunctionDefinition {
                name: "test_fn".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let result = processor.apply_raw(&messages, Some(&tools)).unwrap();
        assert!(result.contains("[TOOLS]"));

        // Without tools
        let result_no_tools = processor.apply_raw(&messages, None).unwrap();
        assert!(!result_no_tools.contains("[TOOLS]"));
    }

    #[test]
    fn test_apply_with_kwargs_exposes_preserve_thinking() {
        // verify that kwargs reach the Jinja template under the
        // provided names. The template branches on preserve_thinking to emit
        // a marker we can assert on.
        let template = r#"{% if preserve_thinking %}[KEEP]{% else %}[STRIP]{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        // preserve_thinking=true
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(out.contains("[KEEP]"));
        assert!(!out.contains("[STRIP]"));

        // preserve_thinking=false explicitly
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": false}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(out.contains("[STRIP]"));

        // No kwarg: defaults to the template's `preserve_thinking` check
        // (undefined/false → [STRIP]).
        let out = processor
            .apply_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
            .unwrap();
        assert!(out.contains("[STRIP]"));
    }

    #[test]
    fn test_apply_with_kwargs_passes_through_arbitrary_keys() {
        // A future-proof sanity check: a template that references a
        // hypothetical `custom_flag` kwarg must see the value we pass.
        let template = r#"flag={{ custom_flag }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"custom_flag": 42}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(out.contains("flag=42"));
    }

    // -- enable_thinking default plumbing -----------------------

    #[test]
    fn default_enable_thinking_defaults_to_false_for_back_compat() {
        // Pre- behavior: a freshly constructed processor leaves
        // `enable_thinking` at `false` so callers that never opt in see the
        // historic default.
        let template = r#"{% if enable_thinking %}[THINK]{% else %}[NOTHINK]{% endif %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        assert!(!processor.default_enable_thinking());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let out = processor
            .apply_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
            .unwrap();
        assert!(
            out.contains("[NOTHINK]"),
            "back-compat default must keep enable_thinking=false; got: {out:?}"
        );
    }

    #[test]
    fn set_default_enable_thinking_flips_default_for_thinking_models() {
        // upstream PR #1114: when the tokenizer recognizes a
        // think marker pair, the chat-template default flips to `true`. The
        // request must then render the THINK branch without setting any
        // kwarg.
        let template = r#"{% if enable_thinking %}[THINK]{% else %}[NOTHINK]{% endif %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        processor.set_default_enable_thinking(true);
        assert!(processor.default_enable_thinking());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let out = processor
            .apply_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
            .unwrap();
        assert!(
            out.contains("[THINK]"),
            "default_enable_thinking=true must flip the template default; got: {out:?}"
        );
    }

    #[test]
    fn kwargs_enable_thinking_false_overrides_default_true() {
        // Even with the upstream-aligned default of `true`, an explicit
        // kwarg `enable_thinking=false` must win. This is the reverse of
        // `test_apply_with_kwargs_allows_overriding_enable_thinking` and
        // guards the per-request override symmetry.
        let template = r#"{% if enable_thinking %}[THINK]{% else %}[NOTHINK]{% endif %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        processor.set_default_enable_thinking(true);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"enable_thinking": false}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(
            out.contains("[NOTHINK]"),
            "explicit kwarg must override the upstream-aligned default; got: {out:?}"
        );
    }

    #[test]
    fn apply_raw_with_kwargs_honours_default_enable_thinking() {
        // The multimodal raw-JSON path (`apply_raw_with_kwargs`) shares the
        // build_template_context plumbing, so the upstream-aligned default
        // must reach it too.
        let template = r#"{% if enable_thinking %}[THINK]{% else %}[NOTHINK]{% endif %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        processor.set_default_enable_thinking(true);
        let raw = serde_json::json!([{"role": "user", "content": "hi"}]);
        let out = processor
            .apply_raw_with_kwargs(&raw, None, &ChatTemplateKwargs::new())
            .unwrap();
        assert!(
            out.contains("[THINK]"),
            "default must reach raw-JSON rendering path too; got: {out:?}"
        );
    }

    #[test]
    fn test_apply_with_kwargs_allows_overriding_enable_thinking() {
        // enable_thinking defaults to false for backward compat. Confirm
        // kwargs can override that default without a dedicated plumbing
        // path — critical for future `enable_thinking` support.
        let template = r#"{% if enable_thinking %}[THINK]{% else %}[NOTHINK]{% endif %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"enable_thinking": true}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(
            out.contains("[THINK]"),
            "enable_thinking kwarg must override default, got: {out:?}"
        );
    }

    #[test]
    fn test_apply_raw_with_kwargs_exposes_preserve_thinking() {
        // Multimodal-path parity: apply_raw_with_kwargs must plumb kwargs too.
        let template = r#"{% if preserve_thinking %}[KEEP]{% else %}[STRIP]{% endif %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"preserve_thinking": true}"#).unwrap();
        let out = processor
            .apply_raw_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(out.contains("[KEEP]"));
    }

    #[test]
    fn test_kwargs_cannot_override_reserved_messages_key() {
        // Regression guard: a request-side kwarg `messages` must NOT replace
        // the real conversation. Template echoes content of the first message
        // so we can detect replacement attempts.
        let template = r#"[{{ messages[0].content }}]"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let real_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "real-user-content".to_string(),
        }];
        let hostile_kwargs = ChatTemplateKwargs::from_json_str(
            r#"{"messages": [{"role": "user", "content": "INJECTED"}]}"#,
        )
        .unwrap();
        let out = processor
            .apply_with_kwargs(&real_messages, None, &hostile_kwargs)
            .expect("render must succeed");
        assert!(
            out.contains("real-user-content"),
            "real messages must survive kwargs overlay; got: {out:?}"
        );
        assert!(
            !out.contains("INJECTED"),
            "kwarg `messages` must NOT replace the real conversation; got: {out:?}"
        );
    }

    #[test]
    fn test_kwargs_cannot_override_reserved_tools_key() {
        // Regression guard: a kwarg `tools` must NOT change the tool set the
        // template iterates over. The template writes `T` per tool so an
        // override from an empty real-tools slice would be detectable.
        let template = r#"{% for t in tools %}T{% endfor %}{% for m in messages %}{{ m.content }}{% endfor %}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        // Request no tools in the real context.
        let hostile_kwargs = ChatTemplateKwargs::from_json_str(
            r#"{"tools": [{"type":"function","function":{"name":"injected","description":null,"parameters":null}}]}"#,
        )
        .unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &hostile_kwargs)
            .expect("render must succeed");
        assert!(
            !out.contains("T"),
            "kwarg `tools` must NOT populate the template tool list; got: {out:?}"
        );
        assert!(out.contains("hi"));
    }

    #[test]
    fn test_kwargs_cannot_flip_reserved_add_generation_prompt() {
        // Regression guard: a kwarg `add_generation_prompt` must NOT override
        // the server-managed value. Template renders a marker only when
        // add_generation_prompt is true.
        let template = r#"{% if add_generation_prompt %}GEN{% else %}NOGEN{% endif %}"#;
        let mut processor = ChatTemplateProcessor::with_template(template.to_string());
        processor.set_add_generation_prompt(false);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let hostile_kwargs =
            ChatTemplateKwargs::from_json_str(r#"{"add_generation_prompt": true}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &hostile_kwargs)
            .expect("render must succeed");
        assert!(
            out.contains("NOGEN"),
            "kwarg must not flip add_generation_prompt; got: {out:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Security M-1: reserved-key filtering — truncation and aggregation
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_key_for_log_short_key_unchanged() {
        // A key within the 64-char limit must come back unchanged.
        let key = "messages";
        assert_eq!(truncate_key_for_log(key), "messages");
    }

    #[test]
    fn truncate_key_for_log_exactly_64_chars_unchanged() {
        let key: String = "x".repeat(64);
        let out = truncate_key_for_log(&key);
        assert_eq!(out.len(), 64, "exactly-64-char key must not be truncated");
        assert!(
            !out.contains('\u{2026}'),
            "no ellipsis for exactly-64-char key"
        );
    }

    #[test]
    fn truncate_key_for_log_long_key_gets_ellipsis() {
        // A key longer than 64 chars must be truncated and have the ellipsis appended.
        let key: String = "a".repeat(400_000);
        let out = truncate_key_for_log(&key);
        // The result is 64 ASCII chars + 3-byte UTF-8 ellipsis (U+2026).
        assert!(
            out.ends_with('\u{2026}'),
            "truncated key must end with ellipsis"
        );
        // The visible character count is 64 'a's + 1 ellipsis = 65 chars.
        assert_eq!(out.chars().count(), 65);
    }

    #[test]
    fn truncate_key_for_log_multibyte_chars_counted_by_char() {
        // Use CJK characters (3 bytes each in UTF-8) to confirm we count
        // Unicode scalar values, not bytes.
        let key: String = "\u{4e2d}".repeat(65); // 65 Chinese chars
        let out = truncate_key_for_log(&key);
        assert!(out.ends_with('\u{2026}'));
        assert_eq!(out.chars().count(), 65); // 64 CJK + ellipsis
    }

    #[test]
    fn m1_multiple_reserved_key_overrides_all_filtered_correctly() {
        // Supplying kwargs for three reserved keys — `messages`, `tools`, and
        // `add_generation_prompt` — must:
        //   1. Not panic.
        //   2. Still render the real messages unchanged (no injection).
        //   3. The normal kwarg key (`preserve_thinking`) still reaches the template.
        let template =
            r#"{% if preserve_thinking %}KEEP{% else %}STRIP{% endif %}|{{ messages[0].content }}"#;
        let processor = ChatTemplateProcessor::with_template(template.to_string());
        let real_messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "real".to_string(),
        }];
        let kwargs = ChatTemplateKwargs::from_json_str(
            r#"{
                "messages": [{"role":"user","content":"INJECTED"}],
                "tools": [{"type":"function","function":{"name":"bad","description":null,"parameters":null}}],
                "add_generation_prompt": false,
                "preserve_thinking": true
            }"#,
        )
        .unwrap();
        let out = processor
            .apply_with_kwargs(&real_messages, None, &kwargs)
            .expect("render must succeed despite reserved-key kwargs");

        // Reserved keys must be silently dropped.
        assert!(out.contains("real"), "real message content must survive");
        assert!(
            !out.contains("INJECTED"),
            "injected messages must be blocked"
        );
        // Non-reserved kwarg must still reach the template.
        assert!(
            out.contains("KEEP"),
            "preserve_thinking=true must reach template"
        );
    }

    // ----- Gemma 4 enable_thinking priming fix -----
    //
    // These tests cover the regression where Gemma 4's chat template
    // dropped the generation-prompt priming when `enable_thinking=true`,
    // leaving the prompt at `<|turn>model\n` and destabilizing first-token
    // logits. The fix appends an open `<|channel>thought\n` priming so the
    // model reliably enters its reasoning channel.

    /// Minimal Gemma-4-shaped template that reproduces the pattern of
    /// interest: opens a `<|turn>model\n` generation prompt and guards the
    /// `<|channel>thought\n<channel|>` priming behind `not enable_thinking`.
    /// Small enough to keep tests fast; keeps the exact marker strings the
    /// detection helpers look for and mirrors the upstream template's use
    /// of explicit `\n` literals inside whitespace-stripped blocks so the
    /// rendered prompt really ends with `<|turn>model\n`.
    const GEMMA4_LIKE_TEMPLATE: &str = "{{- bos_token -}}<|turn>user\n\
        {{- messages[0].content -}}<turn|>\n\
        {{- '<|turn>model\\n' -}}\
        {%- if not enable_thinking | default(false) -%}\
        {{- '<|channel>thought\\n<channel|>' -}}\
        {%- endif -%}";

    #[test]
    fn enable_thinking_drops_priming_detects_gemma4_shape() {
        let gemma4 = ChatTemplateProcessor::with_template(GEMMA4_LIKE_TEMPLATE.to_string());
        assert!(gemma4.enable_thinking_drops_priming());

        // Plain default (no channel marker) must NOT trigger — guards against
        // false positives on non-Gemma templates such as Qwen3's <think> path.
        let plain = ChatTemplateProcessor::default();
        assert!(!plain.enable_thinking_drops_priming());

        // A template with `<|channel>` but no `not enable_thinking` guard
        // (hypothetical unconditional priming) must also not trigger, since
        // we only patch the specific un-primed branch.
        let unconditional = r#"<|turn>model
<|channel>thought
<channel|>"#;
        let p = ChatTemplateProcessor::with_template(unconditional.to_string());
        assert!(!p.enable_thinking_drops_priming());
    }

    #[test]
    fn patch_gemma4_generation_prompt_appends_open_thinking_when_primed() {
        let gemma4 = ChatTemplateProcessor::with_template(GEMMA4_LIKE_TEMPLATE.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        // enable_thinking=false (default branch): template already primes
        // with `<|channel>thought\n<channel|>`, so the patch is a no-op.
        let closed = ChatTemplateKwargs::from_json_str(r#"{"enable_thinking": false}"#).unwrap();
        let rendered_closed = gemma4.apply_with_kwargs(&messages, None, &closed).unwrap();
        assert!(
            rendered_closed.contains("<|channel>thought\n<channel|>"),
            "default branch must still prime a closed thinking block"
        );
        assert!(
            !rendered_closed.ends_with("<|turn>model\n"),
            "closed priming must remain the final content"
        );

        // enable_thinking=true: without the patch the prompt would end at
        // `<|turn>model\n`. With the patch, an OPEN `<|channel>thought\n`
        // marker is appended so the model enters the reasoning channel.
        let open = ChatTemplateKwargs::from_json_str(r#"{"enable_thinking": true}"#).unwrap();
        let rendered_open = gemma4.apply_with_kwargs(&messages, None, &open).unwrap();
        assert!(
            rendered_open.ends_with("<|channel>thought\n"),
            "enable_thinking=true must append an open thinking priming, got: {rendered_open:?}"
        );
        assert!(
            !rendered_open.contains("<channel|>"),
            "the appended priming must be OPEN (no <channel|> close)"
        );
    }

    #[test]
    fn patch_gemma4_generation_prompt_uses_default_enable_thinking_when_kwarg_absent() {
        // Regression guard for HIGH-1 (review): when the server startup
        // hook sets `default_enable_thinking=true` for a Gemma 4 thinking model,
        // a request that arrives with empty kwargs must still trigger the
        // `<|channel>thought\n` priming. Before the fix, `kwargs.get("enable_thinking")`
        // returned `None` which was treated as `false`, so the patch was skipped and
        // the model saw an unprimed `<|turn>model\n` — causing degenerate first-token
        // output / broken tool-call emission.
        let mut gemma4 = ChatTemplateProcessor::with_template(GEMMA4_LIKE_TEMPLATE.to_string());
        gemma4.set_default_enable_thinking(true);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];

        // Empty kwargs — no "enable_thinking" key at all. The processor must
        // fall back to `default_enable_thinking = true` and apply the patch.
        let out = gemma4
            .apply_with_kwargs(&messages, None, &ChatTemplateKwargs::new())
            .unwrap();
        assert!(
            out.ends_with("<|channel>thought\n"),
            "default_enable_thinking=true with empty kwargs must append open priming; got: {out:?}"
        );
        assert!(
            !out.contains("<channel|>"),
            "priming must be open (no <channel|> close); got: {out:?}"
        );
    }

    #[test]
    fn patch_gemma4_generation_prompt_is_noop_on_non_gemma_templates() {
        // Qwen3-shaped template uses <think> / </think>, not <|channel>, so
        // `enable_thinking=true` must flow through untouched even though the
        // kwarg value matches — we don't want to accidentally inject Gemma 4
        // markers into other model families.
        let qwen = r#"{% if enable_thinking %}<think>
{% endif %}{{ messages[0].content }}"#;
        let processor = ChatTemplateProcessor::with_template(qwen.to_string());
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let kwargs = ChatTemplateKwargs::from_json_str(r#"{"enable_thinking": true}"#).unwrap();
        let out = processor
            .apply_with_kwargs(&messages, None, &kwargs)
            .unwrap();
        assert!(out.starts_with("<think>"));
        assert!(!out.contains("<|channel>thought"));
    }
}
