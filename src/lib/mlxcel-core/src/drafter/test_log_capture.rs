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

//! Test-only capture of the `tracing` events emitted by this module.
//!
//! ## Why this exists instead of `tracing-test`
//!
//! `resolve_drafter_kind` emits operator-facing diagnostics, and three tests
//! assert on them. Asserting on a log means depending on `tracing`'s
//! process-global state, which is shared with every other test running
//! concurrently in the same binary. Getting that dependency wrong does not
//! fail loudly; it fails rarely, under load, in somebody else's test run.
//!
//! The trap is in how `tracing` caches per-callsite `Interest`:
//!
//! 1. Building a `Dispatch` (`Dispatch::new`, run by `set_global_default`'s
//!    `Into<Dispatch>` conversion) registers the subscriber and publishes its
//!    max-level hint globally, which is what un-gates the `info!` macros.
//! 2. Installing it as the current default is a *separate*, later step, and
//!    it does not rebuild the interest cache.
//!
//! Between those two steps the process has a permissive max level but
//! `get_default()` still resolves to `NoSubscriber`. A callsite first reached
//! inside that window asks `NoSubscriber` whether anyone is interested, is
//! told `Interest::never()`, and caches that verdict permanently: the
//! registration is one-shot, and nothing registers another dispatcher later
//! to trigger a rebuild. That callsite is then dead for the rest of the
//! process, so the event never fires and the assertion fails no matter how
//! the capture is implemented.
//!
//! The window is only a few hundred nanoseconds wide, but the tests that
//! reach these callsites are neighbours in the same module: they are
//! dispatched to the test harness's worker threads at the same moment, and
//! any one of them can land inside it. See issue #1023 for the measured
//! dose-response.
//!
//! ## The invariant
//!
//! **No thread may reach a drafter callsite before this subscriber is the
//! current default.** [`install`] is the barrier: it is `Once`-gated, so a
//! caller either installs the subscriber or blocks until the installing
//! thread has finished, and only then proceeds to the callsite. Every test in
//! this module reaches `resolve_drafter_kind` / `load_drafter` through the
//! `resolve` / `load` helpers in `mod.rs`, which call [`install`] first, so
//! the window cannot be observed at these callsites at all. That is
//! prevention, not mitigation: there is no interleaving left to lose.
//!
//! [`drafter_tests_reach_the_resolver_only_through_the_install_guard`] holds
//! the invariant in place for tests added later. It checks the drafter test
//! module, which is where the realistic mistake is (copying an existing test).
//! `resolve_drafter_kind` and `load_drafter` currently have no in-crate caller
//! outside that module; if one is added and some other test exercises it, that
//! test needs [`install`] too, for the same reason.
//!
//! ## What is captured
//!
//! Events are recorded into a thread-local sink that is armed only for the
//! duration of [`capture`], so concurrent tests cannot contaminate each
//! other's assertions and no per-test scoping by span name is needed.

use std::cell::RefCell;
use std::fmt;
use std::sync::Once;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};
use tracing::{Level, level_filters::LevelFilter};

/// Only events from this module tree are captured. This must be a decision
/// the subscriber can make from `Metadata` alone: `Subscriber::enabled` is
/// consulted once per callsite and the answer is cached globally, so it can
/// never depend on which thread is asking or on whether a capture is armed.
const TARGET_PREFIX: &str = "mlxcel_core::drafter";

/// The most verbose level any captured event uses. Reported as the max-level
/// hint, which keeps `debug!` / `trace!` callsites elsewhere in the crate
/// switched off in test builds. Widen this together with [`TARGET_PREFIX`] if
/// a future test needs to assert on a more verbose event.
const MAX_CAPTURED_LEVEL: Level = Level::INFO;

thread_local! {
    /// `Some` only while this thread is inside [`capture`].
    static SINK: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

static INSTALLED: Once = Once::new();

/// Install the capture subscriber as the process-wide default, once.
///
/// Returns only when the subscriber is the current default, on every thread.
/// Call this before reaching any callsite whose emission a test asserts on;
/// see the module docs for why the ordering is load-bearing.
pub(super) fn install() {
    INSTALLED.call_once(|| {
        tracing::subscriber::set_global_default(CaptureSubscriber)
            .expect("no other subscriber may be installed in the mlxcel-core test binary");
    });
}

/// Run `f` with the drafter events it emits on this thread recorded.
pub(super) fn capture<T>(f: impl FnOnce() -> T) -> (T, Captured) {
    install();
    let armed = Armed::new();
    let out = f();
    (out, Captured(armed.take()))
}

/// Arms the thread-local sink and disarms it on drop, so a panicking `f`
/// cannot leave a stale sink behind on a reused thread.
struct Armed;

impl Armed {
    fn new() -> Self {
        SINK.with(|sink| {
            let previous = sink.borrow_mut().replace(Vec::new());
            assert!(
                previous.is_none(),
                "captures cannot be nested: the outer sink would be discarded"
            );
        });
        Self
    }

    fn take(&self) -> Vec<String> {
        SINK.with(|sink| sink.borrow_mut().take())
            .unwrap_or_default()
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        SINK.with(|sink| {
            sink.borrow_mut().take();
        });
    }
}

/// Events captured by [`capture`], one formatted line each.
pub(super) struct Captured(Vec<String>);

impl Captured {
    /// Whether any captured event's rendered line contains `needle`.
    pub(super) fn contains(&self, needle: &str) -> bool {
        self.0.iter().any(|line| line.contains(needle))
    }
}

impl fmt::Debug for Captured {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

struct CaptureSubscriber;

impl Subscriber for CaptureSubscriber {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target().starts_with(TARGET_PREFIX) && *metadata.level() <= MAX_CAPTURED_LEVEL
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::from_level(MAX_CAPTURED_LEVEL))
    }

    fn event(&self, event: &Event<'_>) {
        SINK.with(|sink| {
            let mut sink = sink.borrow_mut();
            let Some(lines) = sink.as_mut() else {
                // This thread is not capturing; drop the event.
                return;
            };
            let meta = event.metadata();
            let mut line = format!("{} {}:", meta.level(), meta.target());
            event.record(&mut LineVisitor(&mut line));
            lines.push(line);
        });
    }

    // Spans are not used by any assertion here, so they are accepted and
    // discarded rather than stored.
    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }
    fn record(&self, _span: &Id, _values: &Record<'_>) {}
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    fn enter(&self, _span: &Id) {}
    fn exit(&self, _span: &Id) {}
}

/// Renders an event's fields onto one line: the `message` field as bare
/// text, every other field as `name=value`.
struct LineVisitor<'a>(&'a mut String);

impl Visit for LineVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        use fmt::Write as _;
        if field.name() == "message" {
            let _ = write!(self.0, " {value:?}");
        } else {
            let _ = write!(self.0, " {}={value:?}", field.name());
        }
    }
}

#[cfg(test)]
mod tests {
    /// Holds the invariant documented at the top of this file: every call
    /// into the resolver from the drafter test module must go through the
    /// `resolve` / `load` helpers, which run [`super::install`] first.
    ///
    /// A test that calls the resolver directly would reintroduce the failure
    /// this file exists to remove, and it would do so silently: the log
    /// assertions would still pass almost every time. So the rule is checked
    /// mechanically rather than left as a comment. The helpers are the only
    /// place allowed to name the real functions, which they do through
    /// `super::`.
    #[test]
    fn drafter_tests_reach_the_resolver_only_through_the_install_guard() {
        let module = include_str!("mod.rs");
        let test_module = module
            .split_once("mod tests {")
            .expect("drafter/mod.rs must contain a `mod tests {` block")
            .1;

        for symbol in ["resolve_drafter_kind(", "load_drafter("] {
            let calls = test_module.matches(symbol).count();
            let through_helper = test_module.matches(&format!("super::{symbol}")).count();
            assert_eq!(
                calls, through_helper,
                "{calls} call(s) to `{symbol}` in drafter's test module but only \
                 {through_helper} go through the `super::` helper. Tests must call \
                 the `resolve` / `load` helpers so the capture subscriber is \
                 installed before the callsite is first reached; see \
                 drafter/test_log_capture.rs for why."
            );
        }
    }
}
