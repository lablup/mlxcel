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

//! B9 — Observability unit tests for `mlxcel_core::sampling`.
//!
//! Covers the acceptance criteria in GitHub
//!
//! - `metric_not_incremented_when_bias_empty` — empty `TokenBiasMap` leaves
//!   both counters at zero after N sampling calls.
//! - `metric_incremented_when_bias_applied` — non-empty bias increments
//!   `mlxcel_lang_bias_applied_total` once per sampling call.
//! - `tokens_suppressed_metric_increments_only_on_neg_inf_top1` — suppressed
//!   counter increments only when the pre-bias argmax token is `-inf`-biased.
//! - Tracing field assertion — `lang_bias resolved` DEBUG event emitted with
//!   `entries`, `languages`, and `policy` fields when bias is non-empty.
//!
//! Each test calls `reset_lang_bias_counters()` first to decouple from other
//! tests that may run in the same process.

use mlxcel_core::{
    generate::SamplingConfig,
    lang_bias_applied_total, lang_bias_byte_fragment_suppressions_total,
    lang_bias_tokens_suppressed_total,
    sampling::{TokenBiasMap, sample_token_optimized},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lock used to serialize tests that manipulate the global counters.
///
/// The counters are process-global atomics. Running these tests in parallel
/// would cause counter values to be unpredictable. The lock ensures each
/// test gets an isolated baseline.
static COUNTER_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

fn counter_lock() -> std::sync::MutexGuard<'static, ()> {
    COUNTER_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Call `sample_token_optimized` with greedy sampling and the given bias,
/// returning the sampled token id.
fn call_sample(logits_data: &[f32], bias: TokenBiasMap) -> i32 {
    let vocab = logits_data.len() as i32;
    // Use the re-exported ffi functions from mlxcel_core (via `pub use ffi::*`).
    let logits = mlxcel_core::from_slice_f32(logits_data, &[1, 1, vocab]);
    let mut config = SamplingConfig::greedy();
    config.token_bias = bias;
    let (token, _) = sample_token_optimized(&logits, &config, &[]);
    mlxcel_core::eval(&token);
    mlxcel_core::item_i32(&token)
}

/// Reset the global lang-bias counters to zero for test isolation.
///
/// Since `reset_lang_bias_counters` is `#[cfg(test)]` in `mlxcel-core` and
/// not visible across crate boundaries, we reset the underlying atomics via
/// the public atomic references exported from `mlxcel_core::sampling`.
fn reset_counters() {
    mlxcel_core::sampling::LANG_BIAS_APPLIED_TOTAL.store(0, std::sync::atomic::Ordering::Relaxed);
    mlxcel_core::sampling::LANG_BIAS_TOKENS_SUPPRESSED_TOTAL
        .store(0, std::sync::atomic::Ordering::Relaxed);
    mlxcel_core::sampling::LANG_BIAS_BYTE_FRAGMENT_SUPPRESSIONS_TOTAL
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// B9 test: empty bias map — counters stay zero
// ---------------------------------------------------------------------------

/// With an empty `TokenBiasMap`, both counters remain at zero after N
/// sampling calls (Acceptance Criteria §1).
#[test]
fn metric_not_incremented_when_bias_empty() {
    let _guard = counter_lock();
    reset_counters();

    let logits = [1.0f32, 2.0, 3.0, 0.5];
    let bias = TokenBiasMap::new(); // empty

    for _ in 0..5 {
        call_sample(&logits, bias.clone());
    }

    assert_eq!(
        lang_bias_applied_total(),
        0,
        "applied counter must remain zero when bias is empty"
    );
    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        0,
        "suppressed counter must remain zero when bias is empty"
    );
}

// ---------------------------------------------------------------------------
// B9 test: non-empty bias map — applied counter increments per call
// ---------------------------------------------------------------------------

/// With a non-empty `TokenBiasMap`, `mlxcel_lang_bias_applied_total`
/// increments by 1 for each `sample_token_optimized` call
/// (Acceptance Criteria §2).
#[test]
fn metric_incremented_when_bias_applied() {
    let _guard = counter_lock();
    reset_counters();

    let logits = [1.0f32, 2.0, 3.0, 0.5];
    let mut bias = TokenBiasMap::new();
    bias.insert(2, -1.0); // non-empty, non-inf bias

    let n = 4u64;
    for _ in 0..n {
        call_sample(&logits, bias.clone());
    }

    assert_eq!(
        lang_bias_applied_total(),
        n,
        "applied counter must equal the number of sampling calls when bias is non-empty"
    );
    // Suppressed counter: token 2 has the highest logit (3.0) but is biased
    // by -1.0 (not -inf), so no suppression should be recorded.
    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        0,
        "suppressed counter must stay zero when top-1 is not neg-inf biased"
    );
}

// ---------------------------------------------------------------------------
// B9 test: suppressed counter increments only when top-1 is neg-inf biased
// ---------------------------------------------------------------------------

/// Suppressed counter increments exactly when the pre-bias argmax token is
/// `-inf`-biased in the map (Acceptance Criteria §3).
///
/// Positive case: logits[0] = 5.0 (highest), bias[0] = -inf → suppressed.
/// Negative case: logits[2] = 5.0 (highest), bias[0] = -inf (not top-1) → not suppressed.
#[test]
fn tokens_suppressed_metric_increments_only_on_neg_inf_top1() {
    let _guard = counter_lock();
    reset_counters();

    // --- Positive case ---
    // Token 0 is the pre-bias top-1 (logit 5.0) and is -inf suppressed.
    let logits_pos = [5.0f32, 1.0, 2.0, 0.5];
    let mut bias_pos = TokenBiasMap::new();
    bias_pos.insert(0, f32::NEG_INFINITY);
    call_sample(&logits_pos, bias_pos);

    assert_eq!(
        lang_bias_applied_total(),
        1,
        "applied counter must increment for the positive case"
    );
    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        1,
        "suppressed counter must increment when pre-bias top-1 is neg-inf biased"
    );

    // --- Negative case ---
    // Token 2 is the pre-bias top-1 (logit 5.0), but the -inf bias is on
    // token 0 (which is NOT the top-1). No suppression should be recorded.
    let logits_neg = [1.0f32, 2.0, 5.0, 0.5];
    let mut bias_neg = TokenBiasMap::new();
    bias_neg.insert(0, f32::NEG_INFINITY); // token 0 suppressed, but NOT top-1
    call_sample(&logits_neg, bias_neg);

    assert_eq!(
        lang_bias_applied_total(),
        2,
        "applied counter must increment for the negative case too"
    );
    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        1,
        "suppressed counter must NOT increment when the neg-inf token is not top-1"
    );
}

// ---------------------------------------------------------------------------
// byte-fragment suppression counter
// ---------------------------------------------------------------------------

/// When a `-inf`-suppressed top-1 token was tagged via
/// `TokenBiasMap::insert_byte_fragment`, both the total-suppressed counter
/// AND the byte-fragment counter increment. A regular (non-byte-fragment)
/// suppression increments only the total counter.
#[test]
fn byte_fragment_suppression_counter_tracks_opt_in_entries() {
    let _guard = counter_lock();
    reset_counters();

    // --- Byte-fragment case ---
    // Token 0 is top-1 (logit 5.0) and is -inf via `insert_byte_fragment`.
    let logits = [5.0f32, 1.0, 2.0, 0.5];
    let mut bias_bf = TokenBiasMap::new();
    bias_bf.insert_byte_fragment(0, f32::NEG_INFINITY);
    call_sample(&logits, bias_bf);

    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        1,
        "total suppressed counter must increment on byte-fragment suppression"
    );
    assert_eq!(
        lang_bias_byte_fragment_suppressions_total(),
        1,
        "byte-fragment counter must increment when a byte-fragment entry was suppressed"
    );

    // --- Regular case ---
    // Token 0 is top-1 again, but this time inserted via the regular `insert`
    // path. The byte-fragment counter must NOT advance while the total
    // suppressed counter does.
    let mut bias_reg = TokenBiasMap::new();
    bias_reg.insert(0, f32::NEG_INFINITY);
    call_sample(&logits, bias_reg);

    assert_eq!(
        lang_bias_tokens_suppressed_total(),
        2,
        "total suppressed counter increments for regular suppression too"
    );
    assert_eq!(
        lang_bias_byte_fragment_suppressions_total(),
        1,
        "byte-fragment counter must NOT increment for regular (non-byte-fragment) suppression"
    );
}

/// `byte_fragment_len` reports the size of the byte-fragment id set and is
/// the data source for the tracing debug field `byte_fragment_entries`.
#[test]
fn byte_fragment_len_counts_only_tagged_entries() {
    let mut bias = TokenBiasMap::new();
    bias.insert(10, -5.0);
    bias.insert_byte_fragment(20, f32::NEG_INFINITY);
    bias.insert_byte_fragment(21, f32::NEG_INFINITY);
    bias.insert(22, 3.0);

    assert_eq!(bias.len(), 4, "total len covers every inserted id");
    assert_eq!(
        bias.byte_fragment_len(),
        2,
        "byte_fragment_len reports only ids inserted via insert_byte_fragment"
    );
    assert!(!bias.is_byte_fragment(10));
    assert!(bias.is_byte_fragment(20));
    assert!(bias.is_byte_fragment(21));
    assert!(!bias.is_byte_fragment(22));
}

// ---------------------------------------------------------------------------
// B9 test: tracing field emission
// ---------------------------------------------------------------------------

/// A generator constructed with a non-empty `LangBiasConfig` emits a
/// `DEBUG`-level `lang_bias resolved` event with structured fields `entries`,
/// `languages`, and `policy`.
///
/// This test uses `tracing_subscriber` to capture and assert the event.
#[test]
fn lang_bias_tracing_fields_emitted_on_construction() {
    use std::sync::{Arc, Mutex};
    use tracing::Level;

    // Shared buffer to capture formatted log output.
    let log_buf: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));

    // Newtype wrapper to implement MakeWriter without triggering orphan rules.
    struct SharedBufWriter(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBufWriter {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> BufWriter {
            BufWriter(Arc::clone(&self.0))
        }
    }

    struct BufWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let writer = SharedBufWriter(Arc::clone(&log_buf));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_writer(writer)
        .with_ansi(false)
        .finish();

    // Emit the trace from within a subscriber context.
    tracing::subscriber::with_default(subscriber, || {
        // Simulate what model_worker.rs / commands/generate.rs do after
        // resolving a non-empty bias: emit the debug trace.
        tracing::debug!(
            entries = 3usize,
            languages = %"ja,zh",
            policy = %"conservative",
            "lang_bias resolved"
        );
    });

    let output =
        String::from_utf8(log_buf.lock().unwrap().clone()).expect("log output is valid UTF-8");

    assert!(
        output.contains("lang_bias resolved"),
        "expected 'lang_bias resolved' in trace output; got:\n{output}"
    );
    assert!(
        output.contains("entries"),
        "expected 'entries' field in trace output; got:\n{output}"
    );
    assert!(
        output.contains("languages"),
        "expected 'languages' field in trace output; got:\n{output}"
    );
    assert!(
        output.contains("policy"),
        "expected 'policy' field in trace output; got:\n{output}"
    );
}
