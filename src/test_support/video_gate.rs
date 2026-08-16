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

//! Shared gate for tests that need a working `ffmpeg` / `ffprobe` on PATH.
//!
//! Video frame extraction shells out to the system ffmpeg, so the tests that
//! cover it cannot run on a host without it. The pre-#1172 handling was to
//! print `SKIP: ffmpeg not available` and `return`, which libtest counts as a
//! **pass**. A host with no ffmpeg reported `34 passed; 0 failed` and a host
//! with ffmpeg reported `31 passed; 3 failed`, and the first of those looks
//! exactly like a healthy run. The three failures were real: `-vsync` had been
//! removed by ffmpeg 8, so every video input path in the runtime was broken,
//! and the suite had been hiding it for as long as CI lacked ffmpeg.
//!
//! The rule this module enforces is that a capability the suite could not
//! exercise must never be indistinguishable from one it verified. Two pieces
//! together deliver that, because neither is sufficient alone:
//!
//! 1. The tests that need the subprocess carry `#[ignore]`. Stable libtest
//!    has no way to report a skip decided at run time, so a test that
//!    inspects PATH and returns early is counted as passing no matter what it
//!    prints. `#[ignore]` is resolved at compile time and is the only form
//!    that reaches the summary line, which then reads
//!    `29 passed; 0 failed; 5 ignored` instead of `34 passed`. A reader can
//!    tell the two states apart without reading the log.
//! 2. This gate turns the skip into a hard failure when
//!    `MLXCEL_TEST_VIDEO=1`. `#[ignore]` alone would let the opt-in run
//!    silently degrade back into a no-op on a host that lost its ffmpeg, so
//!    the run that is *supposed* to exercise video asserts that it can.
//!
//! `make verify-test-video` is the combination: it provisions nothing but
//! passes both `--include-ignored` and `MLXCEL_TEST_VIDEO=1`, so the video
//! path is either genuinely exercised or loudly red. `nightly-verify.yml`
//! installs ffmpeg and runs that target, which is the hole #1172 exposed.
//!
//! Like `pinned_checkpoint`, this module only ever turns a skip into a
//! failure. It must never be used to turn a failure into a skip: a genuine
//! decode or assertion error has to fail on its own assertion rather than
//! route through here.

/// Name of the environment variable that makes the video capability mandatory.
pub(crate) const REQUIRE_VIDEO_ENV: &str = "MLXCEL_TEST_VIDEO";

/// True when the caller has declared that this host must be able to run the
/// video tests.
///
/// The crate-wide env lock is held only for the read. Rust 2024 made
/// `set_var` unsafe precisely because an unsynchronized concurrent read of the
/// env block is undefined behavior, and other tests in this binary do mutate
/// the environment. The guard is dropped before any assertion so a failure
/// here cannot poison the mutex for later tests.
fn video_required() -> bool {
    let _env_guard = crate::test_support::env_lock::env_lock();
    std::env::var(REQUIRE_VIDEO_ENV).is_ok_and(|value| value == "1")
}

/// Return whether `test_name` may proceed to exercise the ffmpeg subprocess
/// path.
///
/// Returns `true` when `ffmpeg` and `ffprobe` are both invokable. Otherwise
/// panics when `MLXCEL_TEST_VIDEO=1`, and returns `false` after announcing the
/// skip when it is not set.
///
/// `test_name` is the caller's own `#[test]` function name, so the skip or
/// failure message still identifies which test is affected even though the
/// gate is shared across call sites.
#[must_use]
pub(crate) fn video_capability_available(test_name: &str) -> bool {
    if crate::multimodal::video::ffmpeg_available() {
        return true;
    }
    assert!(
        !video_required(),
        "{REQUIRE_VIDEO_ENV}=1 but {test_name} cannot run: ffmpeg and ffprobe must both be \
         invokable on PATH (ffmpeg 5.0 or newer). Install ffmpeg, or unset {REQUIRE_VIDEO_ENV} to \
         skip the video tests on this host."
    );
    eprintln!(
        "SKIP {test_name}: ffmpeg/ffprobe not on PATH, video decoding was NOT exercised. Set \
         {REQUIRE_VIDEO_ENV}=1 to make this a failure."
    );
    false
}

/// Report that a synthetic test clip could not be produced, and either fail or
/// skip depending on `MLXCEL_TEST_VIDEO`.
///
/// Reached only when ffmpeg is present but the fixture-building invocation
/// still failed, which usually means the local build lacks the encoder the
/// fixture asks for. That is a real gap in coverage on a host that was
/// supposed to have one, so a host that set `MLXCEL_TEST_VIDEO=1` gets a
/// failure rather than a skip.
pub(crate) fn skip_or_fail_video_fixture(test_name: &str, reason: &str) {
    assert!(
        !video_required(),
        "{REQUIRE_VIDEO_ENV}=1 but the synthetic clip {test_name} needs could not be produced \
         even though ffmpeg is on PATH: {reason}"
    );
    eprintln!("SKIP {test_name}: {reason}, video decoding was NOT exercised.");
}
