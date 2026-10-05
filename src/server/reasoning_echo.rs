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

//! Re-inject the reasoning the server generated for an assistant turn when the
//! client echoes only `content` (issue #2110).
//!
//! Some chat templates render an earlier turn differently depending on whether
//! the following assistant message carries `reasoning_content`. AI21
//! Jamba-Reasoning keeps its thinking instruction on an earlier user turn only
//! in that case, so a client that echoes `content` alone (the OpenAI SDK
//! default) re-renders that turn shorter than it was generated, the
//! history-boundary snapshot (#1143) stops prefixing the follow-up, and every
//! turn misses the prompt cache.
//!
//! This store remembers, per finished chat completion, the exact reasoning text
//! the response carried. On a later request, an assistant message that has no
//! reasoning of its own gets that text back, exactly as if the client had
//! echoed it, so the turn re-renders the way it was generated.
//!
//! An entry is found only by a request that could itself have produced it. The
//! key covers:
//!
//! * the model id, the chat-template signature (template source, kwargs, tool
//!   choice and tools, see [`super::prompt_cache::key::template_sig`]) and the
//!   resolved prompt-cache session (`prompt_cache_key`, then `user`, then the
//!   anonymous bucket);
//! * a digest of every message that preceded the assistant turn, exactly as the
//!   client sent them; and
//! * the assistant `content` text the client received, with leading and
//!   trailing whitespace trimmed: a stream delivers the blank line after the
//!   close marker as content while the non-streaming reply strips it, and
//!   clients commonly trim before echoing.
//!
//! A request that edits the assistant content, edits or reorders any earlier
//! message, switches template kwargs or tools, or uses another session key
//! therefore misses, and a miss changes nothing. Echoed reasoning always wins:
//! a message that carries its own non-empty reasoning is never looked up.
//!
//! The store keeps only the 32-byte key and the reasoning text, never the
//! conversation, and is bounded by a byte budget and an entry cap with
//! least-recently-used eviction (a hit refreshes an entry, so a conversation in
//! progress keeps its turns).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use super::types::{ChatCompletionRequest, Message, Role};

/// Default byte budget for stored reasoning text (16 MiB).
pub(crate) const DEFAULT_REASONING_ECHO_MAX_BYTES: usize = 16 * 1024 * 1024;

/// Upper bound on the number of entries regardless of the byte budget, so a
/// stream of tiny traces cannot grow the index without limit.
pub(crate) const REASONING_ECHO_MAX_ENTRIES: usize = 4096;

/// Environment variable overriding [`DEFAULT_REASONING_ECHO_MAX_BYTES`]. `0`
/// disables the store.
pub(crate) const REASONING_ECHO_MAX_BYTES_ENV: &str = "MLXCEL_REASONING_ECHO_MAX_BYTES";

/// Fixed per-entry overhead charged against the byte budget: the key in both
/// indexes, the recency tick, and the map bookkeeping.
const ENTRY_OVERHEAD_BYTES: usize = 96;

/// Domain separator for the entry key, so the digest cannot collide with any
/// other BLAKE3 use in the server.
const KEY_DOMAIN: &[u8] = b"mlxcel:reasoning-echo:v1";

/// Identity of the request stream an entry belongs to.
///
/// Built from the same values the prompt-cache key uses for the request, so a
/// stored trace is only visible to requests that share the cache bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReasoningEchoScope {
    pub(crate) model_id: String,
    pub(crate) template_sig: String,
    pub(crate) session_key: String,
}

type EntryKey = [u8; 32];

struct Entry {
    reasoning: Arc<str>,
    tick: u64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<EntryKey, Entry>,
    /// Recency index: tick -> key. The smallest tick is the eviction victim.
    recency: BTreeMap<u64, EntryKey>,
    next_tick: u64,
    bytes: usize,
}

impl Inner {
    fn touch(&mut self, key: &EntryKey) -> Option<Arc<str>> {
        let tick = self.next_tick;
        let entry = self.entries.get_mut(key)?;
        self.recency.remove(&entry.tick);
        entry.tick = tick;
        self.recency.insert(tick, *key);
        self.next_tick += 1;
        Some(entry.reasoning.clone())
    }

    fn remove(&mut self, key: &EntryKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.recency.remove(&entry.tick);
            self.bytes = self.bytes.saturating_sub(entry_cost(&entry.reasoning));
        }
    }

    fn evict_until_within(&mut self, max_bytes: usize, max_entries: usize) {
        while self.bytes > max_bytes || self.entries.len() > max_entries {
            let Some((_, victim)) = self.recency.pop_first() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(entry_cost(&entry.reasoning));
            }
        }
    }
}

fn entry_cost(reasoning: &str) -> usize {
    reasoning.len().saturating_add(ENTRY_OVERHEAD_BYTES)
}

/// Bounded, evicting store of generated reasoning keyed by the conversation
/// that produced it. See the module documentation.
pub(crate) struct ReasoningEchoStore {
    inner: Mutex<Inner>,
    max_bytes: usize,
    max_entries: usize,
}

impl std::fmt::Debug for ReasoningEchoStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReasoningEchoStore")
            .field("max_bytes", &self.max_bytes)
            .field("max_entries", &self.max_entries)
            .field("len", &self.len())
            .finish()
    }
}

impl ReasoningEchoStore {
    /// A store bounded by `max_bytes` of reasoning text and `max_entries`
    /// entries. `max_bytes == 0` or `max_entries == 0` disables it.
    pub(crate) fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            max_bytes,
            max_entries,
        }
    }

    /// The store sized from [`REASONING_ECHO_MAX_BYTES_ENV`], falling back to
    /// [`DEFAULT_REASONING_ECHO_MAX_BYTES`] when the variable is unset or not a
    /// non-negative integer.
    pub(crate) fn from_env() -> Self {
        let max_bytes = match std::env::var(REASONING_ECHO_MAX_BYTES_ENV) {
            Ok(raw) => raw.trim().parse::<usize>().unwrap_or_else(|_| {
                tracing::warn!(
                    value = %raw,
                    "{REASONING_ECHO_MAX_BYTES_ENV} is not a non-negative integer; \
                     using the default of {DEFAULT_REASONING_ECHO_MAX_BYTES} bytes"
                );
                DEFAULT_REASONING_ECHO_MAX_BYTES
            }),
            Err(_) => DEFAULT_REASONING_ECHO_MAX_BYTES,
        };
        Self::new(max_bytes, REASONING_ECHO_MAX_ENTRIES)
    }

    /// Whether the store can hold anything at all.
    pub(crate) fn enabled(&self) -> bool {
        self.max_bytes > 0 && self.max_entries > 0
    }

    /// Largest single trace the store accepts: an eighth of the budget, so one
    /// pathological reply cannot flush every conversation in progress.
    fn max_entry_bytes(&self) -> usize {
        self.max_bytes / 8
    }

    /// Number of live entries.
    pub(crate) fn len(&self) -> usize {
        self.inner.lock().map_or(0, |inner| inner.entries.len())
    }

    /// Bytes charged against the budget by the live entries.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.inner.lock().map_or(0, |inner| inner.bytes)
    }

    /// Remember the reasoning a finished completion returned.
    ///
    /// `prompt_messages` is the request's message list exactly as the client
    /// sent it (before any re-injection), `content` is the assistant text the
    /// client received, and `reasoning` is the `reasoning_content` it received.
    /// Nothing is stored when the request ends with an assistant message (an
    /// assistant-prefill continuation, whose reply is not a turn of its own),
    /// when the reasoning is blank, or when the trace exceeds the per-entry
    /// cap. A blank `content` is recorded: a reply that `max_tokens` cut off
    /// inside its thinking block is all reasoning, and the client echoes it as
    /// an empty assistant turn. The preceding-message digest in the key keeps
    /// such entries apart, so an empty reply only ever matches the exact
    /// conversation that produced it.
    pub(crate) fn record(
        &self,
        scope: &ReasoningEchoScope,
        prompt_messages: &[Message],
        content: &str,
        reasoning: &str,
    ) {
        if !self.enabled()
            || reasoning.trim().is_empty()
            || prompt_messages
                .last()
                .is_some_and(|m| m.role == Role::Assistant)
        {
            return;
        }
        if entry_cost(reasoning) > self.max_entry_bytes() {
            tracing::debug!(
                bytes = reasoning.len(),
                "reasoning echo: trace exceeds the per-entry cap; not stored"
            );
            return;
        }
        let mut prefix = PrefixDigest::new();
        for message in prompt_messages {
            prefix.push(message);
        }
        let key = entry_key(scope, &prefix.digest(), content);
        let reasoning: Arc<str> = Arc::from(reasoning);
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner.remove(&key);
        let tick = inner.next_tick;
        inner.next_tick += 1;
        inner.bytes = inner.bytes.saturating_add(entry_cost(&reasoning));
        inner.recency.insert(tick, key);
        inner.entries.insert(key, Entry { reasoning, tick });
        inner.evict_until_within(self.max_bytes, self.max_entries);
    }

    /// Fill `reasoning` on every assistant message that lacks it and whose
    /// conversation prefix and content match a stored completion.
    ///
    /// Returns the request unchanged (borrowed) when nothing matched. Messages
    /// that carry their own non-empty reasoning, tool calls, or an inline
    /// `<think>` block are never touched. The digest of each
    /// earlier message is taken over the request as received, so a re-injected
    /// trace does not change the key a later turn is looked up under.
    pub(crate) fn fill<'r>(
        &self,
        scope: &ReasoningEchoScope,
        request: &'r ChatCompletionRequest,
    ) -> Cow<'r, ChatCompletionRequest> {
        if !self.enabled()
            || self.len() == 0
            || !request.messages.iter().any(is_injection_candidate)
        {
            return Cow::Borrowed(request);
        }
        let mut filled: Option<ChatCompletionRequest> = None;
        let mut prefix = PrefixDigest::new();
        for (idx, message) in request.messages.iter().enumerate() {
            if is_injection_candidate(message) {
                let key = entry_key(scope, &prefix.digest(), &message.content.text());
                let hit = self
                    .inner
                    .lock()
                    .ok()
                    .and_then(|mut inner| inner.touch(&key));
                if let Some(reasoning) = hit {
                    let target = filled.get_or_insert_with(|| request.clone());
                    target.messages[idx].reasoning = Some(reasoning.to_string());
                    tracing::debug!(
                        message_index = idx,
                        bytes = reasoning.len(),
                        "reasoning echo: re-injected the stored trace into a content-only \
                         assistant turn"
                    );
                }
            }
            prefix.push(message);
        }
        filled.map_or(Cow::Borrowed(request), Cow::Owned)
    }
}

/// The re-echo scope for a chat request, or `None` when re-injection must not
/// run for it.
///
/// `None` when the prompt cache is not installed (the feature exists only to
/// keep the cache's prefix stable), when the request opted out with
/// `cache_prompt: false`, when the store is disabled, and when the server's
/// reasoning placement does not return the trace as a separate
/// `reasoning_content` field (`--reasoning-format none` / `deepseek-legacy`
/// keep it inline in `content`, which the client echoes back as is;
/// `--skip-chat-parsing` produces no reasoning field at all).
pub(crate) fn chat_scope(
    state: &super::AppState,
    live: &super::LiveSettings,
    request: &ChatCompletionRequest,
) -> Option<ReasoningEchoScope> {
    let format = state.config.reasoning_format;
    if state.prompt_cache.is_none()
        || !state.reasoning_echo.enabled()
        || request.resolve_cache_prompt() == Some(false)
        || state.config.skip_chat_parsing
        || !format.emits_reasoning_content()
        || format.keeps_thoughts_in_content()
    {
        return None;
    }
    Some(ReasoningEchoScope {
        model_id: state.display_model_id().to_string(),
        template_sig: super::routes::chat::chat_template_signature(state, live, request),
        session_key: super::prompt_cache::key::resolve_session_key(
            request.resolve_prompt_cache_key(),
            request.resolve_user(),
        )
        .to_string(),
    })
}

/// Whether `message` is an assistant turn the store may fill.
fn is_injection_candidate(message: &Message) -> bool {
    message.role == Role::Assistant
        && message.reasoning.as_deref().is_none_or(str::is_empty)
        && message.tool_calls.as_ref().is_none_or(Vec::is_empty)
        && { !message.content.text().contains("<think>") }
}

/// Running BLAKE3 digest over a message list, one length-prefixed JSON
/// encoding per message, so the digest at position `i` identifies exactly the
/// messages before `i`.
struct PrefixDigest {
    hasher: blake3::Hasher,
}

impl PrefixDigest {
    fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }

    fn push(&mut self, message: &Message) {
        // `Message` serializes deterministically (struct field order, no maps),
        // and failing to serialize is not possible for these plain types. If it
        // ever did, an empty encoding still separates positions by length.
        let encoded = serde_json::to_vec(message).unwrap_or_default();
        self.hasher.update(&(encoded.len() as u64).to_le_bytes());
        self.hasher.update(&encoded);
    }

    fn digest(&self) -> [u8; 32] {
        *self.hasher.clone().finalize().as_bytes()
    }
}

fn entry_key(scope: &ReasoningEchoScope, prefix: &[u8; 32], content: &str) -> EntryKey {
    let mut hasher = blake3::Hasher::new();
    for field in [
        KEY_DOMAIN,
        scope.model_id.as_bytes(),
        scope.template_sig.as_bytes(),
        scope.session_key.as_bytes(),
        prefix.as_slice(),
        content.trim().as_bytes(),
    ] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
#[path = "reasoning_echo_tests.rs"]
mod tests;
