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

//! Prompt-cache snapshot reuse under a non-FP16 KV cache mode (issue #3250).
//!
//! A snapshot-reuse family used to store nothing at all once the server ran
//! with a quantized KV cache: the first layer in that mode refused the
//! snapshot, the donation was recorded as `empty_set`, and every turn
//! prefilled its whole prompt. The tiny Gemma 3 here has one full-attention
//! layer, so under `Int8`, `Turbo4Asym` or `Turbo4` that layer really holds
//! quantized state, and the scheduler builds it the way the server does:
//! prompt cache first, then the KV mode, then the batch KV config whose
//! builder logs the applied mode.

use super::scheduler_completion_snapshot_tests::run_chat_request;
use super::scheduler_model_owned_cache_tests::{cache_ctx, prompt, scheduler, test_store};
use super::*;
use mlxcel_core::cache::{BatchKvQuantConfig, KVCacheMode};
use mlxcel_core::generate::LanguageModel;
use std::sync::Arc;

use crate::server::prompt_cache::PromptCacheStore;

/// The text of the startup warning for a KV mode no snapshot can carry.
const REUSE_DISABLED: &str = "prompt-cache prefix reuse is disabled for this model";

/// The tiny Gemma 3 scheduler under `mode`, in the server's builder order.
fn scheduler_in(mode: KVCacheMode, store: Arc<PromptCacheStore>) -> BatchScheduler {
    scheduler(store)
        .with_kv_cache_mode(mode)
        .with_batch_kv_quant(BatchKvQuantConfig::default())
}

/// Run `log` under a subscriber that writes into a buffer, and return what
/// it wrote.
fn capture_log_output(log: impl FnOnce()) -> String {
    use std::sync::Mutex;

    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    struct BufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log buffer").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBuf {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> BufWriter {
            BufWriter(Arc::clone(&self.0))
        }
    }

    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(SharedBuf(Arc::clone(&buf)))
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, log);
    let bytes = buf.lock().expect("log buffer").clone();
    String::from_utf8(bytes).expect("UTF-8 log")
}

/// Whether the snapshot stored for `seen` holds the tensor `name`.
fn stored_snapshot_holds(
    sched: &BatchScheduler,
    store: &PromptCacheStore,
    seen: &[i32],
    name: &str,
) -> bool {
    let ctx = cache_ctx();
    let key =
        BatchScheduler::compose_prompt_cache_key(&ctx, seen, sched.rope_regime_for(seen.len()));
    let (entry, _) = store
        .lookup_snapshot_prefix(&key, seen)
        .expect("the turn donated a snapshot that prefixes what it saw");
    entry.with_snapshot(|snapshot| snapshot.tensor(name).is_some())
}

/// A turn under a quantized KV mode donates its history-boundary snapshot,
/// and the next turn, which extends it, resumes from that boundary instead of
/// prefilling cold. What the resumed turn decodes is what a cold run of the
/// same request under the same mode decodes.
#[test]
fn a_quantized_turn_donates_a_snapshot_and_the_next_turn_resumes_from_it() {
    let _tail = crate::server::prompt_cache::decoded_tail::override_decoded_tail_for_test(false);
    for (mode, sidecar) in [
        (KVCacheMode::Int8, "layer0.standard.key_scales"),
        (KVCacheMode::Turbo4Asym, "layer0.standard.v_packed"),
        (KVCacheMode::Turbo4, "layer0.standard.k_packed"),
    ] {
        let store = test_store();
        let mut sched = scheduler_in(mode, store.clone());
        assert_eq!(
            sched.engine.model().kv_cache_storage_modes(),
            Some(vec![Some(mode)]),
            "{mode:?}: the tiny model's one layer really stores this mode"
        );

        let history = prompt(36);
        let mut turn_1 = history.clone();
        turn_1.extend([3, 4, 5]);
        let (seen, adopted) = run_chat_request(&mut sched, &turn_1, Some(&history), true, 4);
        assert_eq!(adopted, 0, "{mode:?}: a cold store restores nothing");
        let stats = store.stats();
        assert!(
            stats.snapshot_inserts >= 1 && stats.snapshot_entries >= 1,
            "{mode:?}: the turn donated a snapshot (inserts {}, entries {})",
            stats.snapshot_inserts,
            stats.snapshot_entries
        );
        assert!(
            stored_snapshot_holds(&sched, &store, &seen, sidecar),
            "{mode:?}: the stored snapshot carries {sidecar}"
        );

        // The next turn re-renders the reply into its history.
        let mut history_2 = seen.clone();
        history_2.extend([1, 2]);
        let mut turn_2 = history_2.clone();
        turn_2.extend([3, 4, 5]);
        let (served, adopted) = run_chat_request(&mut sched, &turn_2, Some(&history_2), true, 4);
        assert_eq!(
            adopted,
            history.len(),
            "{mode:?}: the next turn resumes at the stored boundary"
        );
        assert!(store.stats().snapshot_hits >= 1, "{mode:?}: a snapshot hit");

        let mut cold = scheduler_in(mode, test_store());
        let (cold_seen, cold_adopted) =
            run_chat_request(&mut cold, &turn_2, Some(&history_2), true, 4);
        assert_eq!(cold_adopted, 0);
        assert_eq!(
            served, cold_seen,
            "{mode:?}: the hit decodes what the cold run decodes"
        );

        let obs = sched.batch_observability.snapshot();
        assert_eq!(
            obs.prompt_cache_reject_kv_mode_not_snapshotable, 0,
            "{mode:?}"
        );
        assert_eq!(obs.prompt_cache_reject_empty_set, 0, "{mode:?}");
        assert_eq!(obs.prompt_cache_reject_layout_constraints, 0, "{mode:?}");
    }
}

/// `Turbo4Delegated` and `Turbo3Asym` on a full-attention layer cannot be
/// stored. The server says so once when it starts, naming the mode and the
/// flags that turn reuse back on.
#[test]
fn an_uncarried_kv_mode_is_warned_about_once_at_startup() {
    for mode in [KVCacheMode::Turbo4Delegated, KVCacheMode::Turbo3Asym] {
        let log = capture_log_output(|| {
            let _ = scheduler_in(mode, test_store());
        });
        assert_eq!(
            log.matches(REUSE_DISABLED).count(),
            1,
            "{mode:?}: exactly one warning, got: {log}"
        );
        assert!(log.contains("WARN"), "{mode:?}: at warn level: {log}");
        assert!(
            log.contains(&mode.to_string()),
            "{mode:?}: the warning names the mode: {log}"
        );
        assert!(
            log.contains("--cache-type-k fp16 --cache-type-v fp16"),
            "{mode:?}: the warning says how to turn reuse back on: {log}"
        );
    }
}

/// Each donation under such a mode is counted under its own reason instead of
/// `empty_set`, and nothing is stored, so the next request finds no prefix.
///
/// Run under `Turbo4Delegated` only: the 3-bit packing of `Turbo3Asym` needs
/// a head dim that is a multiple of 8 and cannot run on the tiny model's head
/// dim of 2. Both modes take the same branch, which reads the mode table.
#[test]
fn an_uncarried_kv_mode_is_counted_under_its_own_reason() {
    let mode = KVCacheMode::Turbo4Delegated;
    let store = test_store();
    let mut sched = scheduler_in(mode, store.clone());
    assert_eq!(
        sched.engine.model().kv_cache_storage_modes(),
        Some(vec![Some(mode)])
    );

    // A request that finishes inside its prefill, so the test is about the
    // donation and not about decoding in this mode.
    let first = prompt(40);
    let (_, adopted) = run_chat_request(&mut sched, &first, None, true, 1);
    assert_eq!(adopted, 0);
    assert_eq!(store.stats().snapshot_entries, 0, "nothing stored");
    let obs = sched.batch_observability.snapshot();
    assert!(
        obs.prompt_cache_reject_kv_mode_not_snapshotable >= 1,
        "the decline is counted under its own reason"
    );
    assert_eq!(
        obs.prompt_cache_reject_empty_set, 0,
        "and not as an empty capture"
    );
    assert_eq!(
        obs.prompt_cache_last_reject.map(|r| r.reason),
        Some("kv_mode_not_snapshotable"),
        "last_reject names the reason"
    );

    // The same request again finds nothing to reuse.
    let (_, adopted) = run_chat_request(&mut sched, &first, None, true, 1);
    assert_eq!(adopted, 0, "no prefix was ever stored");
}

/// The warning is for a mode no snapshot can carry and for nothing else: not
/// for FP16, not for the quantized modes that are stored, and not when there
/// is no prompt cache to reuse a prefix from.
#[test]
fn a_carried_kv_mode_and_a_server_without_a_prompt_cache_are_not_warned_about() {
    for mode in [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
    ] {
        let log = capture_log_output(|| {
            let _ = scheduler_in(mode, test_store());
        });
        assert!(!log.contains(REUSE_DISABLED), "{mode:?}: {log}");
    }
    let log = capture_log_output(|| {
        let _ = scheduler(test_store())
            .with_prompt_cache(None)
            .with_kv_cache_mode(KVCacheMode::Turbo4Delegated)
            .with_batch_kv_quant(BatchKvQuantConfig::default());
    });
    assert!(!log.contains(REUSE_DISABLED), "no prompt cache: {log}");
}
