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

use crate::server::gbnf::{GbnfError, GbnfVocab, compile_gbnf};
use crate::server::grammar::{GrammarSpec, LazyGate, LazyOutcome};
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

    /// A GBNF grammar or lazy-trigger surface mlxcel refused. The message is
    /// b10621's own diagnostic wherever upstream has one, so a client that
    /// already handles llama-server's grammar errors keeps working; it is
    /// surfaced verbatim rather than prefixed.
    #[error("{0}")]
    InvalidGrammar(String),
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
/// `mask_buf` and `packed_buf` are reusable scratch buffers, pre-allocated at
/// construction and reset (not reallocated) on each per-token call. With a
/// 150k-vocab tokenizer that saves roughly 750 KiB of allocator churn per
/// emitted token per sequence.
pub struct StructuredOutputConstraint {
    matcher: Matcher,
    vocab_size: usize,
    /// Token environment, kept so the lazy gate can read a token's raw piece
    /// and re-tokenize the buffered text it activates on.
    tok_env: TokEnv,
    /// Lazy-grammar gate (b10621 `grammar_lazy`). `None` for an eager
    /// constraint, which is every `response_format` request and every
    /// non-lazy grammar.
    lazy: Option<LazyGate>,
    /// Scratch buffer for [`Self::compute_mask`] — reused across calls so
    /// the per-token decode path does not allocate a fresh `Vec<bool>` of
    /// length `vocab_size` (~150 KB for a 150k-vocab tokenizer).
    mask_buf: Vec<bool>,
    /// Scratch buffer for [`Self::compute_packed_mask`], holding the matcher
    /// mask in its packed form: 32 token bits per `u32`, bit `i % 32` of word
    /// `i / 32` for token `i`. Reused across calls so the per-token decode
    /// path neither allocates nor walks the vocabulary. For a 248320-row
    /// logits axis this is 31 KB against the 993 KB `Vec<f32>` bias the
    /// previous implementation rebuilt and uploaded on every step.
    packed_buf: Vec<u32>,
}

impl std::fmt::Debug for StructuredOutputConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `llguidance::Matcher` does not implement `Debug`, so we expose only
        // the fields that are useful in logs without leaking schema internals.
        f.debug_struct("StructuredOutputConstraint")
            .field("vocab_size", &self.vocab_size)
            .field("is_stopped", &self.matcher.is_stopped())
            .field("lazy", &self.lazy)
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

    /// Compute the same allow-set as [`Self::compute_mask`], but leave it in
    /// the matcher's packed form: bit `i % 32` of word `i / 32` is token `i`.
    ///
    /// This is the form the per-token decode path wants. [`Self::compute_mask`]
    /// exists to let callers inspect individual entries and is what the
    /// integration tests drive; it costs one host write per vocabulary entry,
    /// which for a 248k-row head is 248320 writes that
    /// [`apply_structured_mask_to_logits`] then reads back to build an f32
    /// bias of the same length. Keeping the mask packed removes both walks:
    /// the matcher already stores its answer as a bitset, so this copies
    /// `ceil(vocab_size_hint / 32)` words straight out of it.
    ///
    /// `vocab_size_hint` is the model's logits width, the same value
    /// [`apply_structured_mask_to_logits`] takes. The returned slice is
    /// always exactly `ceil(vocab_size_hint / 32)` words long, so it expands
    /// to a mask that broadcasts onto the model's logits: bits past the
    /// matcher's vocabulary read zero, which keeps a padded head row masked.
    ///
    /// Returns an empty slice if the matcher is already stopped, matching
    /// [`Self::compute_mask`]. Callers are expected to short-circuit on
    /// `is_stopped()` first; this is defence in depth.
    pub fn compute_packed_mask(
        &mut self,
        vocab_size_hint: usize,
    ) -> Result<&[u32], StructuredOutputError> {
        if self.matcher.is_stopped() {
            self.packed_buf.clear();
            return Ok(&self.packed_buf);
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

        // `compute_mask` drops every entry at or past `self.vocab_size`, and a
        // bitset shorter than that simply has no bit to read. Taking the
        // smaller of the two reproduces that rule exactly.
        let matcher_vocab = self.vocab_size.min(vob.len());
        pack_mask_words(
            vob.as_slice(),
            matcher_vocab,
            vocab_size_hint,
            &mut self.packed_buf,
        );
        Ok(&self.packed_buf)
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
    /// `true` while a lazy grammar is still waiting for its trigger.
    ///
    /// b10621 applies no mask at all in this state
    /// (`llama_sampler_grammar_apply` returns early on `awaiting_trigger`), so
    /// the caller must leave the logits untouched rather than treat an empty
    /// mask as "nothing is allowed".
    pub fn is_gated(&self) -> bool {
        self.lazy.as_ref().is_some_and(LazyGate::awaiting)
    }

    /// Feed the buffered text a lazy trigger activated on into the matcher.
    ///
    /// b10621's grammar is code-point structured, so it replays a partial token
    /// piece directly. `llguidance` advances by whole tokens, so the activated
    /// byte range is re-tokenized instead. The matcher consumes bytes
    /// underneath, so a re-tokenization of the same bytes reaches the same
    /// state; the divergence is confined to a trigger that starts inside a
    /// token, where the re-tokenized split may differ from upstream's.
    fn feed_activation(&mut self, outcome: LazyOutcome) -> Result<(), StructuredOutputError> {
        let tokens = match outcome {
            LazyOutcome::StillWaiting => return Ok(()),
            LazyOutcome::ActivateTokens(tokens) => tokens,
            LazyOutcome::ActivateBytes(bytes) => self.tok_env.tokenize_bytes(&bytes),
        };
        for token in tokens {
            if token as usize >= self.vocab_size {
                continue;
            }
            self.matcher.consume_token(token).map_err(|e| {
                tracing::error!("lazy-grammar activation replay failed: {e}");
                StructuredOutputError::Matcher(
                    "lazy grammar activated on text the grammar does not accept; the trigger \
                     must match where the grammar can start"
                        .to_string(),
                )
            })?;
        }
        if let Some(err) = self.matcher.get_error() {
            tracing::error!("matcher error after lazy-grammar activation: {err}");
            return Err(StructuredOutputError::Matcher(
                "matcher entered error state".to_string(),
            ));
        }
        Ok(())
    }

    pub fn consume_token(&mut self, token: i32) -> Result<(), StructuredOutputError> {
        if self.matcher.is_stopped() {
            return Ok(());
        }
        if self.is_gated() {
            if token < 0 || token as usize >= self.vocab_size {
                // Out-of-range ids cannot be trigger tokens and have no piece;
                // an inert grammar simply keeps waiting.
                return Ok(());
            }
            let token_u32 = token as u32;
            let piece = self.tok_env.tok_trie().token(token_u32).to_vec();
            let outcome = self
                .lazy
                .as_mut()
                .map_or(LazyOutcome::StillWaiting, |gate| {
                    gate.observe(token_u32, &piece)
                });
            return self.feed_activation(outcome);
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

    /// Consume one sampled token and report whether it completes the grammar.
    ///
    /// Scheduler call sites use this combined operation so a terminal matcher
    /// becomes a normal stop immediately after its schema-closing token. This
    /// avoids attempting another decode step, where a stopped matcher has no
    /// mask to apply.
    pub fn consume_token_and_check_stopped(
        &mut self,
        token: i32,
    ) -> Result<bool, StructuredOutputError> {
        self.consume_token(token)?;
        Ok(self.is_stopped())
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

/// Resolve the byte-level token environment for `tokenizer`.
///
/// Shared by every constraint builder: the schema path, the GBNF path and the
/// tests. Serializes the tokenizer once and reuses the process-wide
/// fingerprinted cache.
fn tok_env_for(tokenizer: &MlxcelTokenizer) -> Result<TokEnv, StructuredOutputError> {
    let hf_tokenizer = tokenizer.hf_tokenizer().ok_or_else(|| {
        StructuredOutputError::UnsupportedTokenizer(
            "structured outputs require a HuggingFace tokenizer.json; the loaded \
             tokenizer is SentencePiece or Tiktoken"
                .to_string(),
        )
    })?;
    let serialized = hf_tokenizer.to_string(false).map_err(|e| {
        tracing::error!("tokenizer serialisation failed: {e}");
        StructuredOutputError::UnsupportedTokenizer(
            "tokenizer could not be serialised for structured-output adapter".to_string(),
        )
    })?;
    let (_fingerprint, tok_env) = resolve_tok_env(serialized.as_bytes()).map_err(|e| {
        tracing::error!("tokenizer-env resolution failed: {e}");
        StructuredOutputError::UnsupportedTokenizer(
            "failed to build byte-level token environment".to_string(),
        )
    })?;
    Ok(tok_env)
}

/// Compile a `TopLevelGrammar` into a per-request constraint.
///
/// **Security**: configures `llguidance` with tightened `ParserLimits`
/// (`max_grammar_size`, `max_lexer_states`) and `verbose_errors: false` so an
/// adversarial grammar cannot exhaust the compiler nor leak parser state via
/// the public error message. Verbose details go to `tracing::error!`.
fn build_constraint(
    tok_env: TokEnv,
    grammar: TopLevelGrammar,
    lazy: Option<LazyGate>,
    on_error: impl Fn() -> StructuredOutputError,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    let vocab_size = tok_env.tok_trie().vocab_size();

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
        on_error()
    })?;

    {
        let limits = factory.limits_mut();
        limits.max_grammar_size = MAX_GRAMMAR_SIZE;
        limits.max_lexer_states = MAX_LEXER_STATES;
        limits.verbose_errors = false;
    }

    let parser = factory.create_parser(grammar);
    let matcher = Matcher::new(parser);
    if let Some(err) = matcher.get_error() {
        tracing::error!("matcher build error: {err}");
        return Err(on_error());
    }

    Ok(Arc::new(Mutex::new(StructuredOutputConstraint {
        matcher,
        vocab_size,
        tok_env,
        lazy,
        mask_buf: Vec::with_capacity(vocab_size),
        // The packed buffer is sized lazily on first call: its width comes
        // from `vocab_size_hint`, the model logits axis, which the constraint
        // builder does not know about. Reserving the matcher's own word count
        // covers the common case without a realloc on the first per-token
        // call, and costs 1/32 of what the old f32 bias buffer reserved.
        packed_buf: Vec::with_capacity(vocab_size.div_ceil(32)),
    })))
}

/// Build a constraint from a raw JSON-Schema [`serde_json::Value`].
///
/// The schema is wrapped exactly like upstream mlx-vlm wraps it — via
/// `TopLevelGrammar::from_json_schema`. mlxcel keeps this entry point public
/// so unit tests can build constraints without going through the HTTP layer.
pub fn build_json_schema_constraint(
    tokenizer: &MlxcelTokenizer,
    schema: serde_json::Value,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    // Pre-compilation guard: reject oversized / deeply-nested schemas BEFORE
    // any grammar work. This is the first line of defence against
    // CPU/memory-exhaustion DoS via crafted schemas.
    validate_schema_bounds(&schema)?;
    let tok_env = tok_env_for(tokenizer)?;
    build_constraint(
        tok_env,
        TopLevelGrammar::from_json_schema(schema),
        None,
        || StructuredOutputError::InvalidSchema("schema compilation failed".to_string()),
    )
}

/// Build a constraint from llguidance Lark grammar text.
///
/// Used by the forced `tool_choice` path (#1319), which spells a tool format's
/// fixed wrapper around `%json { ... }` tool schemas. Shares the tokenizer
/// environment cache, the tightened `ParserLimits` and the size cap of the
/// JSON-schema path: the Lark text embeds the serialized tool schemas, so
/// [`MAX_SCHEMA_BYTES`] bounds it exactly as it bounds a `response_format`
/// schema before any grammar work starts. Grammar failures surface as the
/// same generic `InvalidGrammar` the GBNF path uses; verbose details stay in
/// the server log.
pub fn build_lark_constraint(
    tokenizer: &MlxcelTokenizer,
    lark: &str,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    if lark.len() > MAX_SCHEMA_BYTES {
        return Err(StructuredOutputError::SchemaTooLarge(format!(
            "grammar serialised size {} bytes exceeds limit {} bytes",
            lark.len(),
            MAX_SCHEMA_BYTES
        )));
    }
    let tok_env = tok_env_for(tokenizer)?;
    build_constraint(
        tok_env,
        TopLevelGrammar::from_lark(lark.to_string()),
        None,
        || StructuredOutputError::InvalidGrammar("failed to compile grammar".to_string()),
    )
}

/// `<token-text>` resolution for the GBNF front end, wired to mlxcel's
/// tokenizer with b10621's own flags.
struct TokenizerVocab<'a>(&'a MlxcelTokenizer);

impl GbnfVocab for TokenizerVocab<'_> {
    fn tokenize_exact_one(&self, text: &str) -> Option<u32> {
        let ids = self.0.encode_with_special(text, false, true).ok()?;
        match ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        }
    }
}

/// Compile GBNF text into a constraint.
///
/// Grammar parse failures carry b10621's own diagnostic; only the `llguidance`
/// compilation step is genericised, because its messages describe the lowered
/// Lark rather than anything the caller wrote.
pub fn build_gbnf_constraint(
    tokenizer: &MlxcelTokenizer,
    gbnf: &str,
    lazy: Option<LazyGate>,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    let vocab = TokenizerVocab(tokenizer);
    let lark = compile_gbnf(gbnf, Some(&vocab)).map_err(|e| match e {
        GbnfError::Parse(m) | GbnfError::Token(m) | GbnfError::Unsupported(m) => {
            StructuredOutputError::InvalidGrammar(m)
        }
    })?;
    let tok_env = tok_env_for(tokenizer)?;
    build_constraint(tok_env, TopLevelGrammar::from_lark(lark), lazy, || {
        StructuredOutputError::InvalidGrammar("failed to compile grammar".to_string())
    })
}

/// Build a constraint from a resolved [`GrammarSpec`].
///
/// The two sources land on different `llguidance` front ends: a JSON schema
/// goes through `from_json_schema` (mlxcel's existing path, and a closer match
/// to the schema semantics than round-tripping through GBNF would be), while a
/// GBNF grammar goes through the [`crate::server::gbnf`] lowering.
pub fn build_constraint_from_grammar_spec(
    tokenizer: &MlxcelTokenizer,
    spec: &GrammarSpec,
) -> Result<Arc<Mutex<StructuredOutputConstraint>>, StructuredOutputError> {
    let lazy = if spec.lazy {
        Some(
            LazyGate::new(&spec.triggers)
                .map_err(|e| StructuredOutputError::InvalidGrammar(e.to_string()))?,
        )
    } else {
        None
    };
    if let Some(gbnf) = spec.gbnf.as_deref() {
        return build_gbnf_constraint(tokenizer, gbnf, lazy);
    }
    let schema = spec
        .schema
        .clone()
        .unwrap_or(serde_json::Value::Object(Default::default()));
    validate_schema_bounds(&schema)?;
    let tok_env = tok_env_for(tokenizer)?;
    build_constraint(
        tok_env,
        TopLevelGrammar::from_json_schema(schema),
        lazy,
        || StructuredOutputError::InvalidSchema("schema compilation failed".to_string()),
    )
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

/// Bit position of each token inside its packed `u32` word, `0..32`.
///
/// Uploaded as a `u32[1, 32]` row and broadcast against the packed mask column
/// so one shift expands every word into its 32 token bits at once. This is the
/// same expansion MLX itself uses to unpack non-power-of-two quantized weights
/// (see `bitwise_and(right_shift(w, arange(32, uint32)), 1)` in `dequantize`).
const PACKED_MASK_BIT_POSITIONS: [u32; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

/// Copy a matcher bitset into `out`, sized and trimmed for a logits axis of
/// `vocab_size_hint` tokens.
///
/// `src` is the matcher's own packed mask: bit `i % 32` of word `i / 32` is
/// token `i`. `matcher_vocab` is the number of leading bits in `src` that name
/// a real token; a bit at or past it is dropped, which is what keeps a padded
/// head row masked. `out` ends up exactly `ceil(vocab_size_hint / 32)` words
/// long, zero past the matcher's vocabulary, with the final partial word
/// trimmed to its valid bit range so the matcher's own excess bits cannot leak
/// in as spuriously allowed tokens.
///
/// This reproduces, in packed form, what `compute_mask` plus the old f32 bias
/// fill computed one entry at a time: token `i` is allowed exactly when
/// `i < min(matcher_vocab, vocab_size_hint)` and `src` has its bit set.
fn pack_mask_words(src: &[u32], matcher_vocab: usize, vocab_size_hint: usize, out: &mut Vec<u32>) {
    let n_words = vocab_size_hint.div_ceil(32);
    out.clear();
    out.resize(n_words, 0);

    // A bit is only meaningful when it is inside the matcher's vocabulary,
    // inside the model's logits axis, and actually backed by a source word.
    let valid_bits = matcher_vocab.min(vocab_size_hint).min(src.len() * 32);
    let full_words = valid_bits / 32;
    let rem = valid_bits % 32;

    // `full_words <= n_words` because `valid_bits <= vocab_size_hint`, and
    // `full_words <= src.len()` because `valid_bits <= src.len() * 32`.
    out[..full_words].copy_from_slice(&src[..full_words]);
    if rem != 0 {
        // `valid_bits` is not a word multiple here, so `full_words` is a
        // strict index into both slices. Keep only the bits below it.
        out[full_words] = src[full_words] & ((1u32 << rem) - 1);
    }
}

/// Expand a packed mask into a `bool[1, vocab_size]` array on the device.
///
/// The packed words go up as a `u32[n_words, 1]` column and the bit positions
/// as a `u32[1, 32]` row, so a single broadcast shift produces `[n_words, 32]`
/// where element `(w, b)` carries bit `b` of word `w`. Row-major that is
/// exactly token id `w * 32 + b`, so a reshape to one row and a trim to
/// `vocab_size` recovers the mask in token order with no host-side unpacking
/// and no per-token index table.
///
/// Only `ceil(vocab_size / 32) * 4` bytes cross the bus, 31 KB for a
/// 248320-row head against the 993 KB the f32 bias needed.
fn expand_packed_mask(
    words: &[u32],
    vocab_size: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    debug_assert_eq!(
        words.len(),
        vocab_size.div_ceil(32),
        "the packed mask must cover exactly the logits width"
    );
    let n_words = words.len() as i32;
    let packed = mlxcel_core::from_slice_u32(words, &[n_words, 1]);
    let bit_pos = mlxcel_core::from_slice_u32(&PACKED_MASK_BIT_POSITIONS, &[1, 32]);
    let one = mlxcel_core::from_slice_u32(&[1u32], &[1]);

    // [n_words, 1] >> [1, 32] broadcasts to [n_words, 32]; masking with 1
    // leaves the single bit that names token `w * 32 + b`.
    let shifted = mlxcel_core::right_shift(&packed, &bit_pos);
    let bits = mlxcel_core::bitwise_and(&shifted, &one);

    let flat_len = n_words * 32;
    let flat = mlxcel_core::reshape(&bits, &[1, flat_len]);
    // `n_words * 32` overshoots `vocab_size` whenever the vocabulary is not a
    // word multiple. Those trailing lanes are zero (see `pack_mask_words`), but
    // the mask still has to match the logits width to broadcast.
    let trimmed = if flat_len as usize == vocab_size {
        flat
    } else {
        mlxcel_core::slice(&flat, &[0, 0], &[1, vocab_size as i32])
    };

    // The values here are already 0 or 1, so the cast is exact.
    mlxcel_core::astype(&trimmed, mlxcel_core::dtype::BOOL)
}

/// Select `logits` where the packed mask allows a token and `-inf` where it
/// does not.
///
/// `mlxcel_core::where_cond` promotes its two value operands to their common
/// dtype, so pairing f16 or bf16 logits with an f32 `-inf` yields the same f32
/// output the previous `add(logits, f32_bias)` produced, and an allowed logit
/// passes through with the identical value it had.
fn apply_packed_mask_to_logits(
    words: &[u32],
    vocab_size: usize,
    logits: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let allowed = expand_packed_mask(words, vocab_size);
    let neg_inf = mlxcel_core::from_slice_f32(&[f32::NEG_INFINITY], &[1]);
    mlxcel_core::where_cond(&allowed, logits, &neg_inf)
}

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
/// # How the mask reaches the device
///
/// The matcher already answers as a bitset, so the mask stays packed all the
/// way to the GPU: [`StructuredOutputConstraint::compute_packed_mask`] copies
/// `ceil(vocab_size_hint / 32)` words out of it and [`expand_packed_mask`]
/// turns those back into one bit per token with a broadcast shift, a mask and
/// a cast. Nothing on the scheduler thread walks the vocabulary, and the
/// per-step host-to-device copy is 1/32 of the f32 bias it replaces.
///
/// The result is the same array the previous additive bias produced.
/// `where_cond` passes an allowed logit through untouched instead of adding
/// `0.0` to it, which for IEEE floats is the same value, and writes the same
/// `-inf` at every disallowed position.
///
/// # Vocab-size handling
///
/// `vocab_size_hint` is the vocabulary size the model's logits axis exposes.
/// The mask is built with exactly `vocab_size_hint` entries so it broadcasts
/// cleanly onto the model's logits; any other shape would trigger a hard FFI
/// error inside `mlxcel_core::where_cond`.
///
/// Two directions are possible:
///
/// 1. `matcher_vocab >= vocab_size_hint`: the matcher carries entries
///    that the model cannot emit. Trailing matcher-only positions are
///    silently dropped when packing; the sampler never sees them, so they
///    cannot violate the schema.
/// 2. `matcher_vocab < vocab_size_hint`: rare, happens when the model
///    has padded its embedding table beyond the tokenizer's natural
///    vocabulary. Positions in `[matcher_vocab, vocab_size_hint)` read a zero
///    bit and are therefore masked out, conservatively: an unknown token id
///    can never satisfy the grammar.
pub fn apply_structured_mask_to_logits(
    constraint: &mut StructuredOutputConstraint,
    logits: &mlxcel_core::MlxArray,
    vocab_size_hint: usize,
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, StructuredOutputError> {
    // A lazy grammar that has not triggered constrains nothing at all, so the
    // logits pass through untouched. This is not the same as an empty mask,
    // which below is (correctly) an error.
    if constraint.is_gated() {
        return Ok(mlxcel_core::copy(logits));
    }

    // Packed to the model's logits width, so every word this reads back is
    // reachable by the sampler and the emptiness test below needs no separate
    // bound. A stopped matcher yields an empty slice, which is all-zero and
    // therefore takes the same error path.
    let words = constraint.compute_packed_mask(vocab_size_hint)?;
    if words.iter().all(|word| *word == 0) {
        // No legal continuation reachable by the sampler. Surface it as a
        // clean error so the scheduler can stop the sequence with a
        // 5xx-equivalent FinishReason::Error rather than silently emitting an
        // arbitrary token.
        return Err(StructuredOutputError::Matcher(
            "structured-output matcher returned an empty mask: \
             no matcher-allowed token is reachable in the model's logits \
             vocabulary for the current constrained-decoding state."
                .to_string(),
        ));
    }

    Ok(apply_packed_mask_to_logits(words, vocab_size_hint, logits))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "structured_tests.rs"]
mod tests;
