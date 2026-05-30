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

//! OpenAI-compatible `response_format: {"type": "json_schema",...}`
//! support via constrained decoding.
//!
//! Mirrors the upstream mlx-vlm PR #1047 design but in pure Rust:
//!
//! 1. The HTTP layer extracts a JSON-Schema document from the request body.
//! 2. [`build_json_schema_constraint`] compiles the schema into an
//!    `llguidance` grammar and instantiates a per-request [`StructuredOutputConstraint`].
//! 3. The scheduler attaches the constraint to a [`crate::server::batch::SequenceInfo`]
//!    and, for every step, calls [`StructuredOutputConstraint::compute_mask`]
//!    before sampling and [`StructuredOutputConstraint::consume_token`] after
//!    sampling — guaranteeing every emitted token keeps the partial output
//!    grammatically conforming.
//!
//! `llguidance` is the same library upstream uses (PR #1047 commit 5e1102a),
//! so behavior should match the Python implementation closely. The Rust crate
//! exposes `Matcher` / `ParserFactory` directly, eliminating the need for any
//! Python interop on the hot path.
//!
//! # Library choice
//!
//! `llguidance` was selected over `outlines-core` because it:
//!
//! 1. **Matches upstream**: mlx-vlm PR #1047 uses `llguidance`. Keeping the
//!    same backend reduces drift when mirroring upstream test cases.
//! 2. **Has first-class JSON-Schema support**: `TopLevelGrammar::from_json_schema`
//!    handles `$ref`, `enum`, `additionalProperties: false`, and the other
//!    constructs OpenAI's structured-output spec relies on.
//! 3. **Is cheap per token**: roughly ~50 microseconds to recompute the mask
//!    over a 150k-vocabulary tokenizer (per maintainer benchmark), small
//!    relative to a typical decode step.
//! 4. **Is permissively licensed (MIT)** and pure Rust with optional features,
//!    so adding it to mlxcel does not pull in C build dependencies.

use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};
use llguidance::{
    Matcher, ParserFactory,
    api::TopLevelGrammar,
    toktrie::{InferenceCapabilities, TokEnv},
};
use sha2::{Digest, Sha256};
use thiserror::Error;
// `toktrie_hf_tokenizers` is pinned to `tokenizers = 0.21` upstream while
// mlxcel itself uses `tokenizers = 0.22`. The two crate versions ship
// incompatible `Tokenizer` types, so we cannot pass mlxcel's tokenizer
// directly. The serialized JSON form is stable across these versions, so
// we round-trip through `ByteTokenizer::from_json_bytes` to bridge the
// gap. This is a one-time per-tokenizer cost paid behind the
// `TOK_ENV_CACHE`.
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use crate::tokenizer::MlxcelTokenizer;

// ---------------------------------------------------------------------------
// Schema size / depth limits — applied BEFORE compiling the grammar so an
// adversarial schema cannot exhaust CPU / memory inside llguidance.
// ---------------------------------------------------------------------------

/// Maximum serialized size (UTF-8 bytes) for a user-supplied JSON schema.
///
/// 64 KiB is generous for hand-written schemas (the OpenAI examples that motivated are all under 4 KiB) but small enough that an
/// attacker cannot use the schema as a payload-size amplification vector
/// against the grammar compiler.
pub(crate) const MAX_SCHEMA_BYTES: usize = 64 * 1024;

/// Maximum nesting depth (objects / arrays) inside a user-supplied schema.
/// llguidance compiles every layer into Earley productions, so deep
/// schemas blow up grammar size super-linearly.
pub(crate) const MAX_SCHEMA_DEPTH: usize = 32;

/// Maximum number of `$ref` entries allowed inside a single schema.
/// Each `$ref` expands into a separate sub-grammar; capping the count
/// keeps an attacker from defining a tiny schema that references itself
/// 10k times to explode compilation cost.
pub(crate) const MAX_SCHEMA_REFS: usize = 64;

/// Tightened llguidance parser limits. Defaults are 500k grammar symbols
/// and 250k lexer states — generous for trusted offline use, too generous
/// for an HTTP endpoint exposed to arbitrary clients.
pub(crate) const MAX_GRAMMAR_SIZE: usize = 100_000;
pub(crate) const MAX_LEXER_STATES: usize = 50_000;

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

/// Errors raised while building or driving a structured-output constraint.
///
/// The HTTP layer translates these into 4xx/5xx responses; the scheduler
/// translates them into a `Finished(Error(...))` event so the SSE stream
/// terminates cleanly rather than silently emitting non-conforming output.
///
/// **Sanitization invariant**: every public message produced by this enum
/// is a short, fixed string or a length-bounded sanitized echo of caller
/// intent. Verbose llguidance internals (parser state, partial token
/// streams, expanded grammar rules) are never surfaced — they are routed
/// to `tracing::error!` server-side instead. This prevents an attacker
/// from probing schema-compilation behaviour via crafted inputs.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StructuredOutputError {
    /// The request omitted the schema or supplied an empty / unrecognized
    /// `response_format` shape.
    #[error("invalid response_format: {0}")]
    InvalidRequest(String),

    /// The supplied JSON Schema could not be compiled into a grammar.
    /// Public message is intentionally generic ("Invalid JSON schema");
    /// detailed llguidance error is logged via `tracing::error!`.
    #[error("invalid JSON schema for response_format: {0}")]
    InvalidSchema(String),

    /// The supplied JSON Schema exceeds one of the hard limits enforced
    /// before compilation (size, nesting depth, `$ref` count). Sized
    /// distinctly so operators can tell DoS-class rejection from genuine
    /// "schema is malformed" rejection in logs / metrics.
    #[error("response_format schema too large: {0}")]
    SchemaTooLarge(String),

    /// The active tokenizer is incompatible with `llguidance`. mlxcel's
    /// SentencePiece and Tiktoken backends do not yet expose a byte-level
    /// vocabulary that `llguidance` can drive (MVP scope).
    #[error("tokenizer backend not supported for structured outputs: {0}")]
    UnsupportedTokenizer(String),

    /// `llguidance` raised an error while computing the next-token mask or
    /// advancing the matcher state. Public message is generic; verbose
    /// llguidance details go to server logs only.
    #[error("constrained-decoding error: {0}")]
    Matcher(String),
}

// ---------------------------------------------------------------------------
// Tokenizer environment cache
// ---------------------------------------------------------------------------

/// Build a `TokEnv` from a `tokenizers::Tokenizer` is expensive (walks the
/// entire vocabulary, ~1-2s for a 150k-vocab model). We cache one per
/// process keyed by a SHA-256 digest of the tokenizer's serialised JSON
/// bytes so back-to-back requests with the same model share the work.
///
/// `OnceLock<Mutex<...>>` is used because (a) the cache is touched only at
/// request-start, never on the per-token decode path, and (b) the underlying
/// `TokEnv` is `Arc`-cloned out of the lock so concurrent readers do not
/// contend during compute_mask.
///
/// Cache key uses SHA-256 (32 bytes) — collision-resistant against worst-case
/// adversarial inputs, unlike the previous `DefaultHasher` (SipHash) which is
/// only resistant against accidental collisions. Since the consequence of a
/// collision would be serving a tokenizer environment built from a different
/// tokenizer.json, a strong hash is the right discipline even though the
/// cache key never crosses a trust boundary directly.
static TOK_ENV_CACHE: OnceLock<Mutex<TokEnvCache>> = OnceLock::new();

/// 32-byte SHA-256 digest of the serialised tokenizer.json.
type TokenizerFingerprint = [u8; 32];

struct TokEnvCache {
    /// Last-resolved `TokEnv` keyed by tokenizer fingerprint. We keep a
    /// single slot because mlxcel-server runs one model per process; a
    /// `HashMap` would only matter if multi-tokenizer workloads showed up.
    entries: Vec<(TokenizerFingerprint, TokEnv)>,
}

impl TokEnvCache {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Cheap lookup that does NOT serialise the tokenizer. Returns the
    /// cached `TokEnv` if a previous request already inserted one under
    /// `fingerprint`. The caller should compute `fingerprint` from raw
    /// bytes once per request to avoid double-serialising on cache hits.
    fn lookup(&self, fingerprint: &TokenizerFingerprint) -> Option<TokEnv> {
        self.entries
            .iter()
            .find(|(fp, _)| fp == fingerprint)
            .map(|(_, env)| env.clone())
    }

    /// Insert a freshly-built `TokEnv`. Caps the cache at 4 entries to
    /// bound memory if a long-running server ever swaps tokenizers
    /// (mlxcel-server holds one model per process, so the common case
    /// stays at one entry).
    fn insert(&mut self, fingerprint: TokenizerFingerprint, env: TokEnv) {
        // Avoid duplicates when two cold-start requests race past the
        // pre-lock check; whichever one arrives second is a no-op insert.
        if self.entries.iter().any(|(fp, _)| fp == &fingerprint) {
            return;
        }
        if self.entries.len() >= 4 {
            self.entries.remove(0);
        }
        self.entries.push((fingerprint, env));
    }
}

fn cache() -> &'static Mutex<TokEnvCache> {
    TOK_ENV_CACHE.get_or_init(|| Mutex::new(TokEnvCache::new()))
}

/// Build a `TokEnv` from already-serialised tokenizer JSON bytes. Used by
/// both the cold-cache path and tests.
fn build_tok_env_from_bytes(bytes: &[u8]) -> Result<TokEnv> {
    let byte_tokenizer = ByteTokenizer::from_json_bytes(bytes)
        .map_err(|e| anyhow!("failed to wrap HuggingFace tokenizer for llguidance: {e}"))?;
    let env = ByteTokenizerEnv::new(byte_tokenizer, None)
        .map_err(|e| anyhow!("failed to build byte-level token environment: {e}"))?;
    Ok(env.to_env())
}

/// SHA-256 digest of raw bytes.
fn sha256_bytes(bytes: &[u8]) -> TokenizerFingerprint {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let out = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&out);
    digest
}

/// Compute a content-addressed fingerprint of `tokenizer` and resolve a
/// cached `TokEnv` if one exists, building it from the supplied bytes
/// only on a cold cache.
///
/// The cache lock is taken twice (double-checked-lock pattern):
///
/// 1. Briefly, to check whether an entry already exists.
/// 2. After the (potentially expensive) `build_tok_env_from_bytes` call,
///    to insert the result.
///
/// This intentionally allows two cold-start requests racing the same
/// fingerprint to do the build twice — the second insertion is a no-op
/// per [`TokEnvCache::insert`]. The trade-off is doing rare double work
/// vs holding a process-wide mutex for several seconds while one request
/// builds, blocking unrelated requests.
fn resolve_tok_env(serialized_bytes: &[u8]) -> Result<(TokenizerFingerprint, TokEnv)> {
    let fingerprint = sha256_bytes(serialized_bytes);

    // Step 1: cheap check.
    if let Ok(guard) = cache().lock()
        && let Some(env) = guard.lookup(&fingerprint)
    {
        return Ok((fingerprint, env));
    }

    // Step 2: build outside the lock.
    let env = build_tok_env_from_bytes(serialized_bytes)?;

    // Step 3: insert (no-op if a racing request already inserted).
    if let Ok(mut guard) = cache().lock() {
        guard.insert(fingerprint, env.clone());
    }
    Ok((fingerprint, env))
}

// ---------------------------------------------------------------------------
// Per-request constraint
// ---------------------------------------------------------------------------

/// Per-request constrained-decoding state.
///
/// One [`StructuredOutputConstraint`] is built per HTTP request that supplies
/// a `response_format: {"type": "json_schema", ...}`. The scheduler keeps it
/// in [`crate::server::batch::SequenceInfo::structured`] and consults it
/// before/after sampling on every step.
///
/// `mask_buf` and `bias_buf` are reusable scratch buffers — pre-allocated at
/// construction and reset (not reallocated) on each per-token call. With a
/// 150k-vocab tokenizer that saves roughly 750 KiB of allocator churn per
/// emitted token per sequence.
pub struct StructuredOutputConstraint {
    matcher: Matcher,
    vocab_size: usize,
    /// Scratch buffer for [`Self::compute_mask`] — reused across calls so
    /// the per-token decode path does not allocate a fresh `Vec<bool>` of
    /// length `vocab_size` (~150 KB for a 150k-vocab tokenizer).
    mask_buf: Vec<bool>,
    /// Scratch buffer for [`apply_structured_mask_to_logits`] — reused so
    /// the per-token decode path does not allocate a fresh `Vec<f32>` of
    /// length `vocab_size_hint` (~600 KB for a 150k-vocab model). Filled
    /// in-place on every call.
    bias_buf: Vec<f32>,
}

impl std::fmt::Debug for StructuredOutputConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `llguidance::Matcher` does not implement `Debug`, so we expose only
        // the fields that are useful in logs without leaking schema internals.
        f.debug_struct("StructuredOutputConstraint")
            .field("vocab_size", &self.vocab_size)
            .field("is_stopped", &self.matcher.is_stopped())
            .finish()
    }
}

impl StructuredOutputConstraint {
    /// Vocabulary size the matcher exposes. The mask returned by
    /// [`Self::compute_mask`] has exactly this many bits.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Compute the set of token ids that keep the partial output conforming.
    ///
    /// Returns `Ok(allowed)` where `allowed[i] == true` iff sampling token `i`
    /// next is grammatically valid. Returns `Err(_)` when `llguidance` reports
    /// a parser error — the scheduler propagates this as a clean
    /// `FinishReason::Error` rather than letting non-conforming output leak.
    ///
    /// Uses `compute_mask_or_eos` so that once the matcher reaches a terminal
    /// accepting state (the JSON object is complete), the EOS bit in the
    /// returned mask flips on. This pins the sampler to an EOS-equivalent
    /// token after the schema is satisfied, preventing the model from
    /// emitting "and here's another object…" continuations that some chat
    /// templates and weakly-instruction-tuned models will otherwise produce.
    ///
    /// **Returns a borrowed slice** so callers can read the mask without
    /// copying, which keeps the per-token decode path free of large
    /// `Vec<bool>` allocations. The slice is valid until the next call to
    /// `compute_mask` or any other `&mut self` method.
    ///
    /// Returns an empty slice if the matcher is already stopped — a
    /// stopped matcher accepts no further tokens, so the scheduler should
    /// not attempt to bias logits in that state. The caller is expected
    /// to short-circuit on `is_stopped()` first; this is defence in depth.
    pub fn compute_mask(&mut self) -> Result<&[bool], StructuredOutputError> {
        // Short-circuit when the matcher is already stopped: producing a
        // mask in that state is wasted work — the sequence is finished.
        // Returning an empty slice rather than the previous mask makes the
        // "already stopped" condition observable to the caller.
        if self.matcher.is_stopped() {
            self.mask_buf.clear();
            return Ok(&self.mask_buf);
        }

        let vob = self.matcher.compute_mask_or_eos().map_err(|e| {
            // Verbose llguidance details go to server logs only.
            tracing::error!("structured-output compute_mask_or_eos failed: {e}");
            StructuredOutputError::Matcher("compute_mask failed".to_string())
        })?;

        if let Some(err) = self.matcher.get_error() {
            tracing::error!("structured-output matcher error after compute_mask: {err}");
            return Err(StructuredOutputError::Matcher(
                "matcher entered error state".to_string(),
            ));
        }

        // Reuse the pre-allocated buffer. `resize` keeps the capacity and
        // only does a fill on the existing memory, so the per-token path
        // never reallocates after the first call.
        self.mask_buf.clear();
        self.mask_buf.resize(self.vocab_size, false);
        let mask = self.mask_buf.as_mut_slice();
        let vocab_size = self.vocab_size;
        vob.iter_set_entries(|idx| {
            if idx < vocab_size {
                mask[idx] = true;
            }
        });
        Ok(&self.mask_buf)
    }

    /// Advance the matcher state by the just-sampled token.
    ///
    /// Must be called once per emitted token, right after the sampler picks
    /// from the masked logits. Failing to call this would desync the matcher
    /// and the next mask would no longer reflect the partial output.
    ///
    /// Returns `Ok(())` when the matcher already reached a terminal accepting
    /// state — the caller will typically have just sampled an EOS-equivalent
    /// token (the model's eos_token_id, or a chat-template stop sequence)
    /// and the matcher already recognised the JSON object as complete on
    /// the previous step. Treating that as a no-op keeps the scheduler from
    /// abort-on-stop and matches upstream mlx-vlm's behavior, which simply
    /// drops the consume call when `matcher.is_stopped()`.
    pub fn consume_token(&mut self, token: i32) -> Result<(), StructuredOutputError> {
        if self.matcher.is_stopped() {
            return Ok(());
        }
        if token < 0 {
            return Err(StructuredOutputError::Matcher(format!(
                "invalid token id {token}: must be non-negative"
            )));
        }
        // Bounds-check against the matcher vocabulary so an out-of-range
        // token id never reaches `Matcher::consume_token` (where the
        // failure would leak into a verbose llguidance error).
        let token_u32 = token as u32;
        if token_u32 as usize >= self.vocab_size {
            return Err(StructuredOutputError::Matcher(format!(
                "token id {token_u32} is out of range for matcher vocab ({})",
                self.vocab_size
            )));
        }
        self.matcher.consume_token(token_u32).map_err(|e| {
            tracing::error!("structured-output consume_token failed: {e}");
            StructuredOutputError::Matcher("consume_token failed".to_string())
        })?;
        if let Some(err) = self.matcher.get_error() {
            tracing::error!("structured-output matcher error after consume_token: {err}");
            return Err(StructuredOutputError::Matcher(
                "matcher entered error state".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns `true` when the matcher has reached a terminal accepting state.
    /// Once true, subsequent tokens would either be EOS or cause an error.
    pub fn is_stopped(&self) -> bool {
        self.matcher.is_stopped()
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Pre-compilation guard: validates that a JSON-Schema document is small
/// enough to compile without exhausting CPU / memory.
///
/// Run BEFORE [`TopLevelGrammar::from_json_schema`] so an adversarial
/// schema cannot trigger expensive grammar-construction work. Each
/// rejection emits a short, sanitized public message — the Schema itself
/// is never echoed back so an attacker cannot probe limits via crafted
/// inputs.
fn validate_schema_bounds(schema: &serde_json::Value) -> Result<(), StructuredOutputError> {
    // Serialise once to measure size.
    let serialized = serde_json::to_string(schema).map_err(|e| {
        tracing::error!("schema serialisation for size-check failed: {e}");
        StructuredOutputError::InvalidSchema("schema is not serialisable".to_string())
    })?;
    if serialized.len() > MAX_SCHEMA_BYTES {
        return Err(StructuredOutputError::SchemaTooLarge(format!(
            "schema serialised size {} bytes exceeds limit {} bytes",
            serialized.len(),
            MAX_SCHEMA_BYTES
        )));
    }

    // Walk depth and $ref count in a single pass.
    let (depth, refs) = measure_schema_complexity(schema);
    if depth > MAX_SCHEMA_DEPTH {
        return Err(StructuredOutputError::SchemaTooLarge(format!(
            "schema nesting depth {} exceeds limit {}",
            depth, MAX_SCHEMA_DEPTH
        )));
    }
    if refs > MAX_SCHEMA_REFS {
        return Err(StructuredOutputError::SchemaTooLarge(format!(
            "schema $ref count {} exceeds limit {}",
            refs, MAX_SCHEMA_REFS
        )));
    }
    Ok(())
}

/// Measure (max nesting depth, $ref count) of a JSON Schema document.
/// Linear in the schema size so it is cheap relative to the grammar
/// compilation that follows.
fn measure_schema_complexity(value: &serde_json::Value) -> (usize, usize) {
    fn walk(value: &serde_json::Value, depth: usize, max_depth: &mut usize, refs: &mut usize) {
        if depth > *max_depth {
            *max_depth = depth;
        }
        match value {
            serde_json::Value::Object(map) => {
                if map.contains_key("$ref") {
                    *refs += 1;
                }
                for (_, v) in map {
                    walk(v, depth + 1, max_depth, refs);
                }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    walk(v, depth + 1, max_depth, refs);
                }
            }
            _ => {}
        }
    }
    let mut max_depth = 0usize;
    let mut refs = 0usize;
    walk(value, 0, &mut max_depth, &mut refs);
    (max_depth, refs)
}

/// Build a constraint from a raw JSON-Schema [`serde_json::Value`].
///
/// The schema is wrapped exactly like upstream mlx-vlm wraps it — via
/// `TopLevelGrammar::from_json_schema`. mlxcel keeps this entry point public
/// so unit tests can build constraints without going through the HTTP layer.
///
/// **Security**: this function applies pre-compilation size / depth / `$ref`
/// limits and configures llguidance with tightened `ParserLimits`
/// (`max_grammar_size`, `max_lexer_states`) and `verbose_errors: false`
/// so an adversarial schema cannot exhaust the compiler nor leak parser
/// state via the public error message. Verbose llguidance error details
/// are routed to `tracing::error!` instead of the public surface.
pub fn build_json_schema_constraint(
    tokenizer: &MlxcelTokenizer,
    schema: serde_json::Value,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    let hf_tokenizer = tokenizer.hf_tokenizer().ok_or_else(|| {
        StructuredOutputError::UnsupportedTokenizer(
            "structured outputs require a HuggingFace tokenizer.json; the loaded \
             tokenizer is SentencePiece or Tiktoken"
                .to_string(),
        )
    })?;

    // Pre-compilation guard: reject oversized / deeply-nested schemas
    // BEFORE any grammar work. This is the first line of defence against
    // CPU/memory-exhaustion DoS via crafted schemas.
    validate_schema_bounds(&schema)?;

    // Serialise the tokenizer ONCE per request. The bytes feed both the
    // SHA-256 fingerprint and (on cache miss) the `ByteTokenizer::from_json_bytes`
    // call. Previously we serialised twice on every request (once for the
    // hash, once for the build) even when the cache was hot.
    let serialized = hf_tokenizer.to_string(false).map_err(|e| {
        tracing::error!("tokenizer serialisation failed: {e}");
        StructuredOutputError::UnsupportedTokenizer(
            "tokenizer could not be serialised for structured-output adapter".to_string(),
        )
    })?;
    let serialized_bytes = serialized.as_bytes();

    let (_fingerprint, tok_env) = resolve_tok_env(serialized_bytes).map_err(|e| {
        tracing::error!("tokenizer-env resolution failed: {e}");
        StructuredOutputError::UnsupportedTokenizer(
            "failed to build byte-level token environment".to_string(),
        )
    })?;

    let vocab_size = tok_env.tok_trie().vocab_size();

    let grammar = TopLevelGrammar::from_json_schema(schema);

    let mut factory = ParserFactory::new(
        &tok_env,
        InferenceCapabilities {
            // mlxcel does not currently expose ff-token streams from the
            // sampler, so we keep the safe defaults: per-step mask only.
            ff_tokens: false,
            backtrack: false,
            conditional_ff_tokens: false,
            fork: false,
        },
        &[],
    )
    .map_err(|e| {
        tracing::error!("ParserFactory build failed: {e}");
        StructuredOutputError::InvalidSchema("schema compilation failed".to_string())
    })?;

    // Tighten the grammar / lexer caps and disable verbose-error output
    // so llguidance does not leak parser state via its `e.to_string()`.
    {
        let limits = factory.limits_mut();
        limits.max_grammar_size = MAX_GRAMMAR_SIZE;
        limits.max_lexer_states = MAX_LEXER_STATES;
        limits.verbose_errors = false;
    }

    let parser = factory.create_parser(grammar);
    let matcher = Matcher::new(parser);
    if let Some(err) = matcher.get_error() {
        // `verbose_errors: false` already strips schema/grammar internals
        // from `err`, but we still avoid echoing the raw error back to
        // the client — log it server-side and surface a generic message.
        tracing::error!("matcher build error: {err}");
        return Err(StructuredOutputError::InvalidSchema(
            "schema compilation failed".to_string(),
        ));
    }

    Ok(Arc::new(Mutex::new(StructuredOutputConstraint {
        matcher,
        vocab_size,
        mask_buf: Vec::with_capacity(vocab_size),
        // The bias buffer is sized lazily on first call (vocab_size_hint
        // depends on the model logits axis, which the constraint builder
        // does not know about). Pre-reserving avoids reallocs on the
        // first per-token call as well.
        bias_buf: Vec::with_capacity(vocab_size),
    })))
}

// ---------------------------------------------------------------------------
// HTTP request shape parsing
// ---------------------------------------------------------------------------

/// Extract the JSON Schema (if any) from an OpenAI-compatible
/// `response_format` field.
///
/// Accepts:
///
/// * `{"type": "json_schema", "json_schema": {"schema": { ... }}}` — Chat
///   Completions API shape (also supports `"name"` / `"strict"` siblings,
///   matching upstream).
/// * `{"type": "text"}` or `null` — returns `Ok(None)` so the caller skips
///   constrained decoding.
///
/// The legacy `{"type": "json_object"}` (no schema) is **not** supported in
/// this MVP and surfaces a clean error, matching upstream's PR #1047 scope
/// note ("`json_object` mode tracked separately").
pub fn extract_json_schema_from_response_format(
    response_format: Option<&serde_json::Value>,
) -> Result<Option<serde_json::Value>, StructuredOutputError> {
    let Some(value) = response_format else {
        return Ok(None);
    };
    let Some(obj) = value.as_object() else {
        return Err(StructuredOutputError::InvalidRequest(
            "response_format must be an object".to_string(),
        ));
    };

    let format_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("text");

    match format_type {
        "text" => Ok(None),
        "json_schema" => {
            let json_schema = obj.get("json_schema").and_then(|v| v.as_object());
            // Spec compliance: a json_schema-typed response_format MUST carry
            // a `json_schema` wrapper containing a `schema` field.
            let Some(json_schema) = json_schema else {
                return Err(StructuredOutputError::InvalidRequest(
                    "response_format.type == \"json_schema\" requires a json_schema object \
                     (try {\"json_schema\": {\"schema\": {...}}})"
                        .to_string(),
                ));
            };
            let Some(schema) = json_schema.get("schema") else {
                return Err(StructuredOutputError::InvalidRequest(
                    "response_format.json_schema must include a schema field".to_string(),
                ));
            };
            Ok(Some(schema.clone()))
        }
        "json_object" => Err(StructuredOutputError::InvalidRequest(
            "response_format type \"json_object\" is not supported; supply \
             type=\"json_schema\" with a schema"
                .to_string(),
        )),
        other => Err(StructuredOutputError::InvalidRequest(format!(
            "unsupported response_format type: {other:?}"
        ))),
    }
}

/// Top-level helper: build a constraint directly from the raw HTTP
/// `response_format` JSON value, returning `Ok(None)` when the request did not
/// ask for structured output.
pub fn build_constraint_from_response_format(
    tokenizer: &MlxcelTokenizer,
    response_format: Option<&serde_json::Value>,
) -> Result<Option<Arc<Mutex<StructuredOutputConstraint>>>, StructuredOutputError> {
    let Some(schema) = extract_json_schema_from_response_format(response_format)? else {
        return Ok(None);
    };
    Ok(Some(build_json_schema_constraint(tokenizer, schema)?))
}

// ---------------------------------------------------------------------------
// Logits-mask application
// ---------------------------------------------------------------------------

/// Apply the structured-output mask to a 2-D `[1, vocab]` logits array.
///
/// Returns a fresh array with `f32::NEG_INFINITY` written at every position
/// that the matcher disallows. Allowed positions pass through unchanged.
/// `f32::NEG_INFINITY` composes correctly with the downstream
/// `sample_token_optimized` pipeline (`-inf + x == -inf`,
/// `softmax([..., -inf, ...]) -> 0`), so the sampler can never select a
/// disallowed token.
///
/// Returns `Err(Matcher(...))` when the matcher reports a parser error so
/// the scheduler can transition the sequence to `Finished(Error)` rather
/// than emit non-conforming output.
///
/// # Vocab-size handling
///
/// `vocab_size_hint` is the vocabulary size the model's logits axis
/// exposes. The returned bias array has exactly `vocab_size_hint`
/// entries so it broadcasts cleanly onto the model's logits — any other
/// shape would trigger a hard FFI error inside `mlxcel_core::add`.
///
/// Two directions are possible:
///
/// 1. `matcher_vocab >= vocab_size_hint`: the matcher carries entries
///    that the model cannot emit. Trailing matcher-only positions are
///    silently dropped when building the bias; the sampler never sees
///    them, so they cannot violate the schema.
/// 2. `matcher_vocab < vocab_size_hint`: rare, happens when the model
///    has padded its embedding table beyond the tokenizer's natural
///    vocabulary. Positions in `[matcher_vocab, vocab_size_hint)` are
///    conservatively masked out — an unknown token id can never satisfy
///    the grammar.
pub fn apply_structured_mask_to_logits(
    constraint: &mut StructuredOutputConstraint,
    logits: &mlxcel_core::MlxArray,
    vocab_size_hint: usize,
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, StructuredOutputError> {
    // Bias must broadcast onto the model's logits, so the bias length is
    // pinned to `vocab_size_hint`. Anything the matcher allows beyond
    // that is unreachable by the sampler (the model cannot emit a token
    // id past its own logits axis); anything the matcher does NOT cover
    // up to `vocab_size_hint` defaults to disallowed (conservative — an
    // unknown id cannot satisfy the grammar).
    let vocab_size = vocab_size_hint;

    // Compute the mask (borrows `constraint.mask_buf` mutably). Read out
    // the count of in-range allowed bits, then drop the borrow before
    // touching `constraint.bias_buf`.
    {
        let allowed = constraint.compute_mask()?;
        let usable_allowed_count = allowed.iter().take(vocab_size).filter(|x| **x).count();
        if usable_allowed_count == 0 {
            // No legal continuation reachable by the sampler — surface
            // as a clean error so the scheduler can stop the sequence
            // with a 5xx-equivalent FinishReason::Error rather than
            // silently emitting an arbitrary token.
            return Err(StructuredOutputError::Matcher(
                "structured-output matcher returned an empty mask: \
                 no token can extend the partial output without violating \
                 the schema. The model is stuck — this is usually a sign that \
                 the schema is too restrictive for the supplied prompt."
                    .to_string(),
            ));
        }
    }

    // Build the bias from the now-populated `constraint.mask_buf`. We use
    // the bias_buf directly (not via `bias_scratch`, which would conflict
    // with the read of `mask_buf` since both alias `constraint`); resize
    // in place and fill from the mask.
    constraint.bias_buf.clear();
    constraint.bias_buf.resize(vocab_size, 0.0);
    let mask_len = constraint.mask_buf.len();
    for (i, slot) in constraint.bias_buf.iter_mut().enumerate() {
        // SAFETY of indexing: `mask_buf` was just (re-)sized to
        // `constraint.vocab_size` by `compute_mask`, so the bounds check
        // here matches what the previous `Vec<bool>`-returning version did.
        let allowed = i < mask_len && constraint.mask_buf[i];
        if !allowed {
            *slot = f32::NEG_INFINITY;
        }
    }

    let bias_arr = mlxcel_core::from_slice_f32(&constraint.bias_buf, &[1, vocab_size as i32]);
    Ok(mlxcel_core::add(logits, &bias_arr))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;
