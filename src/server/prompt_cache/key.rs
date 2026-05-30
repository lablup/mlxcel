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

//! Cache-key hashing and composition for the prompt prefix cache.
//!
//! A [`PromptCacheKey`] composes the identity dimensions that define a
//! reusable KV-cache bucket:
//!
//! * `model_id`          — which loaded model produced these tensors.
//! * `lora_id`           — LoRA adapter identifier (`None` means base model).
//! * `template_sig`      — chat-template digest so two requests with different
//!   `<|im_start|>` / tool prompts never collide. Composed via
//!   [`template_sig`] from `(chat_template_source, chat_template_kwargs,
//!   tool_choice, tools_digest)`.
//! * `session_key`       — optional caller-supplied tenancy / conversation scope.
//!   Composed via [`resolve_session_key`] from `(prompt_cache_key, user,
//!   anonymous bucket sentinel)`.
//! * `mm_digest`         — stable, order-preserving digest of all multimodal
//!   (image + audio) content resolved from the request messages. Computed via
//!   [`multimodal_digest`] on the post-resolved byte slices, so two requests
//!   that share text but differ on image/audio bytes always produce different
//!   bucket digests. See [`MultimodalDigest`].
//! * `tokens[..prefix_len]` — the actual prefix of token ids that the cache
//!   bucket represents. Only the first `prefix_len` entries participate in
//!   the bucket digest, so partially matching prefixes still map to
//!   distinct buckets.
//!
//! The bucket digest uses BLAKE3 because it is fast, stable across processes
//! and architectures, and has no per-call allocation overhead for typical
//! prompt prefix sizes. The digest is **not** treated as a security primitive;
//! it exists purely to support `HashMap`-style lookup across sequences.

use std::fmt;

use serde_json::{Map, Value};

use crate::server::chat_template_kwargs::ChatTemplateKwargs;
use crate::server::types::request::{Tool, ToolChoice};

/// A 32-byte BLAKE3 digest identifying a prompt-prefix cache bucket.
///
/// Used by [`super::store::PromptCacheStore`] as the primary map key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PromptCacheKeyDigest(pub(crate) [u8; 32]);

impl PromptCacheKeyDigest {
    /// Raw byte view, useful for logging.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex representation of the 32-byte digest.
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Short hex prefix (16 hex chars, 8 bytes) for log lines.
    pub fn short_hex(self) -> String {
        let mut out = String::with_capacity(16);
        for byte in &self.0[..8] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

impl fmt::Debug for PromptCacheKeyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PromptCacheKeyDigest({})", self.short_hex())
    }
}

impl fmt::Display for PromptCacheKeyDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ---------------------------------------------------------------------------
// Multimodal digest
// ---------------------------------------------------------------------------

/// A 32-byte BLAKE3 digest of the multimodal (image + audio) content present
/// in a request's messages, computed in message-arrival order.
///
/// ## Relationship to `ModelVisionCaches`
///
/// `ModelVisionCaches` (see `src/vision/feature_cache.rs`) caches
/// **post-projection vision embeddings** per image, keyed by image content
/// identity. The prompt cache, by contrast, holds the **KV cache** over the
/// full token stream, which includes image placeholder tokens interleaved with
/// the text tokens. These are orthogonal caches:
///
/// * A `ModelVisionCaches` hit skips the vision tower forward pass.
/// * A prompt cache hit (discriminated by `MultimodalDigest`) skips the
///   language-model prefill for the cached token prefix.
///
/// Changing the multimodal payload invalidates the prompt cache key (because
/// the image placeholder tokens change) but does **not** affect the
/// `ModelVisionCaches` in any way — it uses its own SHA-256 image-content
/// keying scheme and manages its own LRU eviction independently.
///
/// ## Digest construction
///
/// Built by [`multimodal_digest`] over the post-resolved byte slices (i.e.
/// after base64 decoding, file reads, and URL fetches). The digest is
/// **not** derived from image URLs or file paths — two requests that reference
/// different URLs pointing to the same bytes produce the same digest and
/// correctly share a cache bucket.
///
/// The `EMPTY` constant represents "no multimodal content", and is distinct
/// from a hash that happens to have the same bit pattern by domain-separation
/// in the construction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MultimodalDigest(pub(crate) [u8; 32]);

impl MultimodalDigest {
    /// Digest that represents "no multimodal content".
    ///
    /// This is the BLAKE3 hash of the domain-separated empty sequence, *not* a
    /// zero array, so it is distinct from any hash that an empty-payload
    /// collision could accidentally produce.
    pub fn empty() -> Self {
        multimodal_digest(&[], &[])
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex string (64 chars).
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

/// Compute a stable, **order-preserving** BLAKE3 digest of all resolved
/// multimodal (image + audio) bytes in a request.
///
/// ## Arguments
///
/// * `image_bytes` — resolved image payloads in the order they appear across
///   all messages. Each `&[u8]` is the raw post-decoded bytes of one image
///   (PNG, JPEG, etc.), obtained after base64 decoding or URL fetch. Pass the
///   same slice that `extract_chat_image_data` returns.
/// * `audio_bytes` — resolved audio payloads in message-arrival order, same
///   semantics as `image_bytes`.
///
/// ## Design rationale
///
/// Digests are computed on **post-resolution bytes** (the content the model
/// actually sees) rather than raw URLs or base64 strings. This ensures that
/// two requests referencing different URLs that point to the same image bytes
/// map to the same cache bucket.
///
/// The construction is domain-separated and length-prefixed, which prevents
/// boundary-collision attacks where adjacent fields with overlapping byte
/// sequences could otherwise hash identically.
///
/// ## Relationship to `ModelVisionCaches`
///
/// `ModelVisionCaches` (see `src/vision/feature_cache.rs`) is **not touched**
/// by this function. It caches post-projection vision tower outputs per image
/// using its own SHA-256 keying and LRU eviction. The prompt cache (keyed
/// partially by this digest) caches the KV state over the token stream. The
/// two caches are orthogonal: a `ModelVisionCaches` hit only bypasses the
/// vision encoder; a prompt cache hit additionally bypasses the LLM prefill.
/// Changing the multimodal content invalidates the prompt cache key but has
/// no effect on `ModelVisionCaches` state.
///
/// ## Order sensitivity
///
/// The digest **is** order-sensitive: swapping two image messages produces a
/// different digest. This is intentional — the order of image placeholder
/// tokens in the LLM token stream is determined by the order images appear
/// in the request, so two requests with the same images in different orders
/// receive different KV activations and must not share a cache bucket.
pub fn multimodal_digest(image_bytes: &[&[u8]], audio_bytes: &[&[u8]]) -> MultimodalDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mlxcel:prompt-cache:multimodal:v1");

    // Image section: count then each payload length-prefixed.
    hasher.update(&(image_bytes.len() as u64).to_le_bytes());
    for img in image_bytes {
        write_field(&mut hasher, img);
    }

    // Audio section: count then each payload length-prefixed.
    hasher.update(&(audio_bytes.len() as u64).to_le_bytes());
    for aud in audio_bytes {
        write_field(&mut hasher, aud);
    }

    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    MultimodalDigest(out)
}

/// Build a [`MultimodalDigest`] from owned `Vec<Vec<u8>>` slices returned by
/// `extract_chat_image_data` / `extract_chat_audio_data`.
///
/// This is a thin convenience wrapper over [`multimodal_digest`] that avoids
/// callers having to collect references manually.
pub fn multimodal_digest_from_vecs(
    image_bytes: &[Vec<u8>],
    audio_bytes: &[Vec<u8>],
) -> MultimodalDigest {
    let img_refs: Vec<&[u8]> = image_bytes.iter().map(|v| v.as_slice()).collect();
    let aud_refs: Vec<&[u8]> = audio_bytes.iter().map(|v| v.as_slice()).collect();
    multimodal_digest(&img_refs, &aud_refs)
}

/// Composite identity for a prompt-prefix cache bucket.
///
/// Construct one per logical lookup or insert; the struct intentionally
/// borrows its inputs so callers don't have to clone strings and token
/// vectors on the hot path.
#[derive(Clone, Copy, Debug)]
pub struct PromptCacheKey<'a> {
    /// Loaded model identifier (typically the display `model_id` — whatever
    /// `AppState::display_model_id` returns).
    pub model_id: &'a str,
    /// LoRA adapter identifier; `None` for the base model.
    pub lora_id: Option<&'a str>,
    /// Chat-template signature. Full wiring is; for now any stable
    /// digest of the template + tool-schema inputs is acceptable and will
    /// simply slot into this field.
    pub template_sig: &'a str,
    /// Caller-supplied tenancy / conversation scope. `None` means global.
    pub session_key: Option<&'a str>,
    /// Stable, order-preserving digest of all multimodal (image + audio)
    /// content resolved from the request messages.
    ///
    /// Ensures two requests with the same text but different images produce
    /// different cache keys. Use [`MultimodalDigest::empty()`] for text-only
    /// requests. Build from resolved bytes via [`multimodal_digest`] or
    /// [`multimodal_digest_from_vecs`].
    pub mm_digest: MultimodalDigest,
    /// Prefix slice of token ids. Only the first `prefix_len` elements are
    /// hashed.
    pub tokens: &'a [i32],
    /// Number of tokens from the start of `tokens` to include in the bucket
    /// digest. Must be `<= tokens.len()`; longer values saturate.
    pub prefix_len: usize,
}

impl<'a> PromptCacheKey<'a> {
    /// Build a cache key where the full `tokens` slice participates in the
    /// bucket digest.
    ///
    /// For text-only requests pass [`MultimodalDigest::empty()`] as
    /// `mm_digest`. For multimodal requests pass the digest produced by
    /// [`multimodal_digest`] or [`multimodal_digest_from_vecs`].
    pub fn new_full(
        model_id: &'a str,
        lora_id: Option<&'a str>,
        template_sig: &'a str,
        session_key: Option<&'a str>,
        mm_digest: MultimodalDigest,
        tokens: &'a [i32],
    ) -> Self {
        Self {
            model_id,
            lora_id,
            template_sig,
            session_key,
            mm_digest,
            tokens,
            prefix_len: tokens.len(),
        }
    }

    /// Build a cache key where only the first `prefix_len` tokens are hashed.
    ///
    /// For text-only requests pass [`MultimodalDigest::empty()`] as
    /// `mm_digest`. For multimodal requests pass the digest produced by
    /// [`multimodal_digest`] or [`multimodal_digest_from_vecs`].
    pub fn new_prefix(
        model_id: &'a str,
        lora_id: Option<&'a str>,
        template_sig: &'a str,
        session_key: Option<&'a str>,
        mm_digest: MultimodalDigest,
        tokens: &'a [i32],
        prefix_len: usize,
    ) -> Self {
        Self {
            model_id,
            lora_id,
            template_sig,
            session_key,
            mm_digest,
            tokens,
            prefix_len: prefix_len.min(tokens.len()),
        }
    }

    /// Effective number of tokens that participate in the bucket digest.
    pub fn effective_prefix_len(&self) -> usize {
        self.prefix_len.min(self.tokens.len())
    }

    /// Compute the 32-byte BLAKE3 digest that identifies this bucket.
    ///
    /// The input is a length-prefixed, domain-separated concatenation of:
    ///
    /// ```text
    ///     "mlxcel:prompt-cache:v2"
    ///     model_id              (len-prefixed utf-8 bytes)
    ///     lora_id               (len-prefixed utf-8 bytes, empty if None)
    ///     template_sig          (len-prefixed utf-8 bytes)
    ///     session_key           (len-prefixed utf-8 bytes, empty if None)
    ///     mm_digest             (32 raw bytes from MultimodalDigest)
    ///     prefix_len            (u64 little-endian)
    ///     tokens[..prefix_len]  (each token as i32 little-endian)
    /// ```
    ///
    /// Length prefixes and the fixed `v2` tag keep the digest resistant to
    /// accidental collisions between fields with overlapping bytes, and make
    /// it safe to extend the schema later (new fields bump the version tag).
    /// The version changed from `v1` to `v2` when added `mm_digest`
    /// to prevent old v1 buckets from accidentally matching new multimodal keys.
    pub fn digest(&self) -> PromptCacheKeyDigest {
        let mut hasher = blake3::Hasher::new();

        // Domain separator so this digest cannot be reused from a different
        // hashing context accidentally.
        // NOTE: bumped to v2 when mm_digest was added so that
        // stale v1 cache entries do not collide with new multimodal-aware keys.
        hasher.update(b"mlxcel:prompt-cache:v2");

        write_field(&mut hasher, self.model_id.as_bytes());
        write_field(
            &mut hasher,
            self.lora_id.map(str::as_bytes).unwrap_or_default(),
        );
        write_field(&mut hasher, self.template_sig.as_bytes());
        write_field(
            &mut hasher,
            self.session_key.map(str::as_bytes).unwrap_or_default(),
        );

        // Multimodal digest: 32 raw bytes, already a BLAKE3 hash
        // so no length-prefix needed — the fixed size makes it unambiguous.
        hasher.update(self.mm_digest.as_bytes());

        let prefix_len = self.effective_prefix_len();
        hasher.update(&(prefix_len as u64).to_le_bytes());
        for tok in &self.tokens[..prefix_len] {
            hasher.update(&tok.to_le_bytes());
        }

        let mut out = [0u8; 32];
        hasher.finalize_xof().fill(&mut out);
        PromptCacheKeyDigest(out)
    }
}

fn write_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

// ---------------------------------------------------------------------------
// Session-key resolution
// ---------------------------------------------------------------------------

/// Sentinel bucket name used when neither a `prompt_cache_key` nor a `user`
/// is supplied.
///
/// Using a non-empty string means the "anonymous" bucket has a well-defined
/// identity distinct from `None` in the cache key's length-prefixed format,
/// so two anonymous callers collide on the same bucket and benefit from
/// cross-request reuse, while an explicit `user=""` is still normalized to
/// this same sentinel upstream (see [`resolve_session_key`]).
pub const ANONYMOUS_SESSION_SENTINEL: &str = "__mlxcel_anon__";

/// Resolve the cache-key `session_key` from the OpenAI-compatible request
/// hints.
///
/// Precedence (first non-empty wins):
///   1. `prompt_cache_key` — explicit client hint (addition).
///   2. `user` — OpenAI-standard stable end-user identifier.
///   3. [`ANONYMOUS_SESSION_SENTINEL`] — shared fallback bucket.
///
/// Empty strings in either input are treated as "not supplied" so a caller
/// cannot accidentally collide everyone into an empty-string bucket and the
/// anonymous sentinel is used instead. The returned string is borrowed from
/// the inputs when possible; only the sentinel path yields a `&'static str`.
pub fn resolve_session_key<'a>(
    prompt_cache_key: Option<&'a str>,
    user: Option<&'a str>,
) -> &'a str {
    if let Some(k) = prompt_cache_key
        && !k.is_empty()
    {
        return k;
    }
    if let Some(u) = user
        && !u.is_empty()
    {
        return u;
    }
    ANONYMOUS_SESSION_SENTINEL
}

// ---------------------------------------------------------------------------
// Template signature
// ---------------------------------------------------------------------------

/// BLAKE3-based stable hash of the chat-template rendering pipeline inputs.
///
/// The returned 64-char lowercase hex string is suitable for use as the
/// `template_sig` dimension of [`PromptCacheKey`]. The hash covers everything
/// that would cause the same conversation tokens to render differently on the
/// wire:
///
/// * `chat_template_source` — the raw Jinja template string (post our own
///   preprocessing). Any template edit, special-token change, or even a
///   whitespace tweak invalidates the cache cleanly.
/// * `chat_template_kwargs` — the merged per-request + server-default kwargs
///   (e.g. `preserve_thinking`, `enable_thinking`, future model-specific
///   hints). Canonicalized via [`canonicalize_value`] so a reordered but
///   semantically identical map still produces the same digest.
/// * `tool_choice_mode` — "auto" / "none" / "required" / "specific". The
///   effective-tools slice selected inside [`prepare_chat_request`] depends
///   on this mode, so requests that differ only here must produce different
///   template signatures.
/// * `tools_digest` — see [`tools_digest`]. Order-preserving: reordering the
///   `tools` array is a semantic change because the template iterates them
///   in order and some models (notably Qwen and Nemotron) key their tool
///   prompts off the iteration order.
///
/// This is a BLAKE3 digest, not a cryptographic commitment. Callers must not
/// treat a `template_sig` match as proof of template identity; its sole
/// purpose is to fingerprint the input so a mismatch forces cache invalidation.
pub fn template_sig(
    chat_template_source: &str,
    chat_template_kwargs: &ChatTemplateKwargs,
    tool_choice: Option<&ToolChoice>,
    tools: Option<&[Tool]>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mlxcel:prompt-cache:template-sig:v1");

    write_field(&mut hasher, chat_template_source.as_bytes());

    // Canonicalize the kwargs map so {"a":1,"b":2} and {"b":2,"a":1} hash
    // identically. serde_json::Map uses BTreeMap by default when the
    // `preserve_order` feature is off, but we explicitly canonicalize anyway
    // to be robust against future feature-flag drift.
    let canonical_kwargs = canonicalize_map(chat_template_kwargs.as_map());
    write_field(&mut hasher, canonical_kwargs.as_bytes());

    // tool_choice influences which tools the template sees. Hash the normalized
    // string form.
    let tc_tag = match tool_choice {
        None => String::from("__absent__"),
        Some(ToolChoice::Mode(s)) => format!("mode:{s}"),
        Some(ToolChoice::Specific(f)) => format!("specific:{}", f.function.name),
    };
    write_field(&mut hasher, tc_tag.as_bytes());

    // Tools participate in the signature with their order preserved.
    let tools_hex = tools_digest(tools);
    write_field(&mut hasher, tools_hex.as_bytes());

    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    let mut hex = String::with_capacity(64);
    for byte in out {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Compute a stable digest of the `tools` array.
///
/// The digest is **order-preserving**: reordering the tools list changes the
/// output. This is intentional — HuggingFace chat templates iterate tools in
/// the order the client supplied them, and some models embed the index in
/// their tool-call protocol. The digest captures:
///
/// * each tool's `type`,
/// * function `name`,
/// * function `description` (if present),
/// * canonicalized function `parameters` JSON Schema.
///
/// Returns a 64-char lowercase hex string. When `tools` is `None` or empty,
/// returns the digest of a stable "empty" marker so that both cases collapse
/// to a single canonical no-tools signature distinct from any real tool list.
pub fn tools_digest(tools: Option<&[Tool]>) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mlxcel:prompt-cache:tools-digest:v1");

    let tools = tools.unwrap_or(&[]);
    hasher.update(&(tools.len() as u64).to_le_bytes());

    for tool in tools {
        write_field(&mut hasher, tool.tool_type.as_bytes());
        write_field(&mut hasher, tool.function.name.as_bytes());
        write_field(
            &mut hasher,
            tool.function
                .description
                .as_deref()
                .unwrap_or("")
                .as_bytes(),
        );
        let canonical_params = match &tool.function.parameters {
            Some(v) => canonicalize_value(v),
            None => String::from("null"),
        };
        write_field(&mut hasher, canonical_params.as_bytes());
    }

    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    let mut hex = String::with_capacity(64);
    for byte in out {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Canonical JSON rendering with sorted object keys.
///
/// `serde_json::to_string` by default preserves insertion order (or BTreeMap
/// order, depending on feature flags), which is not enough for a stable
/// digest under the full range of client shapes we accept. This helper walks
/// the tree and re-emits objects with keys in sorted order.
fn canonicalize_value(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}")),
        Value::Array(arr) => {
            let mut out = String::from("[");
            for (idx, v) in arr.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                out.push_str(&canonicalize_value(v));
            }
            out.push(']');
            out
        }
        Value::Object(map) => canonicalize_map(map),
    }
}

fn canonicalize_map(map: &Map<String, Value>) -> String {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    let mut out = String::from("{");
    for (idx, k) in keys.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| format!("{k:?}")));
        out.push(':');
        if let Some(v) = map.get(*k) {
            out.push_str(&canonicalize_value(v));
        } else {
            out.push_str("null");
        }
    }
    out.push('}');
    out
}

#[cfg(test)]
#[path = "key_tests.rs"]
mod tests;
