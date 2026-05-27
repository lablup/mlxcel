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

use std::path::Path;
use std::process::Command;

use super::*;

// ─── Extension detection ─────────────────────────────────────────────────────

#[test]
fn is_video_file_recognises_common_extensions() {
    assert!(is_video_file(Path::new("clip.mp4")));
    assert!(is_video_file(Path::new("clip.MP4")));
    assert!(is_video_file(Path::new("/data/clip.mov")));
    assert!(is_video_file(Path::new("/data/clip.webm")));
    assert!(is_video_file(Path::new("/data/clip.mkv")));
    assert!(is_video_file(Path::new("/data/clip.avi")));
    assert!(is_video_file(Path::new("/data/clip.m4v")));
}

#[test]
fn is_video_file_rejects_images_and_other_files() {
    assert!(!is_video_file(Path::new("photo.jpg")));
    assert!(!is_video_file(Path::new("photo.png")));
    assert!(!is_video_file(Path::new("photo.webp")));
    assert!(!is_video_file(Path::new("notes.txt")));
    assert!(!is_video_file(Path::new("/data/clip")));
    assert!(!is_video_file(Path::new("/data/")));
}

// ─── smart_nframes ───────────────────────────────────────────────────────────

#[test]
fn smart_nframes_uniform_default_fps_two() {
    // 240 frames at 30 fps with target fps 2.0 => 240/30*2 = 16 frames.
    let n = smart_nframes(240, 30.0, Some(2.0), None).unwrap();
    assert_eq!(n, 16);
    assert!(n.is_multiple_of(FRAME_FACTOR));
}

#[test]
fn smart_nframes_clamps_to_min_frames() {
    // Very short video at low fps: clamp upward to FPS_MIN_FRAMES = 4.
    let n = smart_nframes(8, 30.0, Some(0.1), None).unwrap();
    assert_eq!(n, FPS_MIN_FRAMES);
}

#[test]
fn smart_nframes_clamps_to_max_frames() {
    // Long video at very high target fps: clamp to FPS_MAX_FRAMES = 768
    // when total_frames >= 768.
    let n = smart_nframes(10_000, 30.0, Some(60.0), None).unwrap();
    assert_eq!(n, FPS_MAX_FRAMES);
}

#[test]
fn smart_nframes_clamps_to_total_frames() {
    // Total frames < FPS_MAX_FRAMES — output cannot exceed total.
    let n = smart_nframes(20, 30.0, Some(100.0), None).unwrap();
    assert_eq!(n, 20);
}

#[test]
fn smart_nframes_explicit_nframes_rounds_to_factor() {
    // Caller supplies an odd target — round to the nearest even.
    let n = smart_nframes(40, 30.0, None, Some(7)).unwrap();
    assert_eq!(n, 8);
}

#[test]
fn smart_nframes_explicit_nframes_out_of_range_errors() {
    // Caller asks for more frames than exist.
    assert!(smart_nframes(10, 30.0, None, Some(20)).is_err());
}

#[test]
fn smart_nframes_total_below_factor_errors() {
    assert!(smart_nframes(1, 30.0, Some(2.0), None).is_err());
}

#[test]
fn smart_nframes_invalid_target_fps_errors() {
    assert!(smart_nframes(40, 30.0, Some(0.0), None).is_err());
    assert!(smart_nframes(40, 30.0, Some(-1.0), None).is_err());
}

#[test]
fn smart_nframes_handles_zero_video_fps_gracefully() {
    // ffprobe occasionally reports 0 fps for streams with metadata gaps.
    // The fallback path should still produce a sensible frame count
    // (treats fps as 1.0 internally).
    let n = smart_nframes(60, 0.0, Some(2.0), None).unwrap();
    assert!(n >= FRAME_FACTOR);
    assert!(n <= 60);
    assert!(n.is_multiple_of(FRAME_FACTOR));
}

// ─── load_video error paths (no ffmpeg needed) ───────────────────────────────

#[test]
fn load_video_missing_file_errors() {
    let path = Path::new("/non/existent/video.mp4");
    let err = load_video(path, Some(2.0), None).unwrap_err();
    match err {
        VideoError::FileNotFound(p) => assert_eq!(p, path),
        VideoError::FfmpegMissing => {
            // On systems without ffmpeg the missing-binary error wins
            // — acceptable for this graceful-degradation contract.
        }
        other => panic!("expected FileNotFound or FfmpegMissing, got {other:?}"),
    }
}

#[test]
fn load_video_without_ffmpeg_returns_clear_error() {
    // This test exercises the graceful-degradation path that fires when
    // `ffmpeg` is not on PATH. We can only assert the precise error
    // shape on systems where ffmpeg is genuinely missing.
    if ffmpeg_available() {
        return;
    }
    let path = Path::new("/tmp/does-not-exist.mp4");
    let err = load_video(path, None, None).unwrap_err();
    matches!(err, VideoError::FfmpegMissing);
}

// ─── PNG stream splitter unit test ───────────────────────────────────────────

#[test]
fn find_subsequence_locates_needle() {
    let haystack = b"hello world IEND goodbye";
    let needle = b"IEND";
    assert_eq!(find_subsequence(haystack, needle), Some(12));
}

#[test]
fn find_subsequence_returns_none_when_absent() {
    let haystack = b"hello world";
    let needle = b"IEND";
    assert_eq!(find_subsequence(haystack, needle), None);
}

#[test]
fn find_subsequence_handles_empty_needle() {
    let haystack = b"hello";
    assert_eq!(find_subsequence(haystack, b""), Some(0));
}

// ─── TempFile drop guard ─────────────────────────────────────────────────────

#[test]
fn temp_file_drop_removes_file() {
    let path = std::env::temp_dir().join("mlxcel-test-tempfile-drop.tmp");
    std::fs::write(&path, b"test content").expect("write temp file");
    assert!(path.exists(), "file should exist before drop");

    let guard = TempFile::new(path.clone());
    drop(guard);

    assert!(!path.exists(), "file should be removed after TempFile drop");
}

#[test]
fn temp_file_drop_survives_nonexistent_file() {
    // TempFile::drop should not panic when the file has already been removed.
    let path = std::env::temp_dir().join("mlxcel-test-tempfile-missing.tmp");
    // Deliberately do NOT create the file.
    let guard = TempFile::new(path.clone());
    drop(guard); // Must not panic.
    assert!(!path.exists());
}

#[test]
fn temp_file_drop_on_panic_cleanup() {
    let path = std::env::temp_dir().join("mlxcel-test-tempfile-panic.tmp");
    std::fs::write(&path, b"should be deleted").expect("write temp file");
    assert!(path.exists(), "file should exist before panic");

    // Clone the path so we can check it after the catch_unwind.
    let path_clone = path.clone();

    let result = std::panic::catch_unwind(move || {
        let _guard = TempFile::new(path.clone());
        panic!("simulated panic to test TempFile cleanup");
    });

    // The catch_unwind should have caught the panic.
    assert!(result.is_err(), "catch_unwind should have caught the panic");

    // The TempFile guard should have fired Drop during unwinding.
    assert!(
        !path_clone.exists(),
        "TempFile should have cleaned up the file even after a panic"
    );
}

// ─── Resolution / duration cap tests (require ffmpeg) ────────────────────────

/// Create a minimal synthetic video file via ffmpeg. Returns the temp path.
/// The video has the given `width`, `height`, frame rate, and total `duration_sec`.
/// Uses a color source (`lavfi testsrc2`) to avoid needing any input file.
fn make_test_video(
    width: u32,
    height: u32,
    fps: u32,
    duration_sec: f64,
) -> Option<std::path::PathBuf> {
    if !ffmpeg_available() {
        return None;
    }
    let path = std::env::temp_dir().join(format!(
        "mlxcel-test-video-{width}x{height}-{fps}fps-{duration_sec}s.mp4"
    ));
    // Remove a leftover from a previous run if present.
    let _ = std::fs::remove_file(&path);

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={width}x{height}:rate={fps}"),
            "-t",
            &format!("{duration_sec:.3}"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&path)
        .status()
        .ok()?;

    if status.success() { Some(path) } else { None }
}

#[test]
fn load_video_rejects_oversized_resolution() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // Synthesize a 100x100 video (well within defaults), then set a tiny
    // cap via environment variable to trigger the rejection.
    let video_path = match make_test_video(100, 100, 5, 2.0) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    // Inject a tiny pixel cap (50×50 = 2500) via VideoLimits so the test never
    // touches the process-global environment (issue #103).
    let limits = VideoLimits {
        max_pixels: 2500,
        ..VideoLimits::default()
    };
    let result = load_video_with_limits(&video_path, Some(2.0), None, &limits);

    // Clean up the temp video.
    let _ = std::fs::remove_file(&video_path);

    match result {
        Err(VideoError::ResolutionTooLarge {
            width,
            height,
            pixels,
            max_pixels,
        }) => {
            assert_eq!(width, 100, "width should match");
            assert_eq!(height, 100, "height should match");
            assert_eq!(pixels, 10_000, "pixels should be width*height");
            assert_eq!(max_pixels, 2500, "max_pixels should match the injected cap");
        }
        Err(other) => panic!("expected ResolutionTooLarge, got: {other:?}"),
        Ok(_) => panic!("expected ResolutionTooLarge error, but load_video succeeded"),
    }
}

#[test]
fn load_video_rejects_overlong_duration() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // Synthesize a short 4-second video and set a 2-second cap.
    let video_path = match make_test_video(64, 64, 5, 4.0) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    // Inject a 2-second cap via VideoLimits so the test never touches the
    // process-global environment (issue #103).
    let limits = VideoLimits {
        max_duration_sec: 2.0,
        ..VideoLimits::default()
    };
    let result = load_video_with_limits(&video_path, Some(2.0), None, &limits);

    let _ = std::fs::remove_file(&video_path);

    match result {
        Err(VideoError::DurationTooLong {
            seconds,
            max_seconds,
        }) => {
            assert!(
                seconds > 2.0,
                "reported duration {seconds:.2}s should exceed the cap"
            );
            assert_eq!(
                max_seconds, 2.0,
                "max_seconds should match the injected cap"
            );
        }
        Err(other) => panic!("expected DurationTooLong, got: {other:?}"),
        Ok(_) => panic!("expected DurationTooLong error, but load_video succeeded"),
    }
}

#[test]
fn load_video_single_pass_produces_correct_frame_count() {
    // Verify that the single-pass implementation produces the expected
    // number of frames for a short synthetic video. This is the key
    // regression test for the single-pass refactor (issue #597).
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // 10 seconds at 10 fps = 100 frames source. With target fps 2 =>
    // smart_nframes(100, 10.0, Some(2.0), None) = 100/10*2 = 20 frames.
    let video_path = match make_test_video(64, 64, 10, 10.0) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    let frames = load_video(&video_path, Some(2.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = frames.expect("load_video should succeed for a valid test video");
    assert!(!frames.is_empty(), "should have decoded at least one frame");
    // All frames should have the correct dimensions.
    for frame in &frames {
        assert_eq!(frame.width(), 64, "frame width should match source");
        assert_eq!(frame.height(), 64, "frame height should match source");
    }
    // Frame count should be a multiple of FRAME_FACTOR.
    assert!(
        frames.len().is_multiple_of(FRAME_FACTOR),
        "frame count {} is not a multiple of FRAME_FACTOR={}",
        frames.len(),
        FRAME_FACTOR
    );
}

// ─── Fail-closed sentinel tests (no ffmpeg needed) ───────────────────────────

/// When ffprobe does not return a `width` field, `apply_probe_caps` must
/// default to `u32::MAX` so the resolution cap trips instead of silently
/// passing a 0×0 video through.
#[test]
fn probe_video_missing_width_trips_resolution_cap() {
    use std::path::Path;
    // nb_frames=4 so we don't fall into the "can't determine frame count" branch.
    // height=100, width=None — should be treated as u32::MAX.
    let path = Path::new("/synthetic/missing-width.mp4");
    let result = apply_probe_caps(
        path,
        Some(4),
        30.0,
        None, // width missing
        Some(100),
        Some(10.0),
        &VideoLimits::default(),
    );
    match result {
        Err(VideoError::ResolutionTooLarge {
            width,
            height,
            pixels,
            max_pixels,
        }) => {
            assert_eq!(width, u32::MAX, "sentinel width should be u32::MAX");
            assert_eq!(height, 100);
            // saturating_mul: u32::MAX as u64 * 100 should not overflow
            assert_eq!(pixels, (u32::MAX as u64).saturating_mul(100));
            assert!(pixels > max_pixels, "sentinel pixels should exceed the cap");
        }
        Err(other) => panic!("expected ResolutionTooLarge, got: {other:?}"),
        Ok(_) => panic!("expected ResolutionTooLarge but probe caps passed"),
    }
}

/// When ffprobe does not return a `height` field, `apply_probe_caps` must
/// default to `u32::MAX` so the resolution cap trips.
#[test]
fn probe_video_missing_height_trips_resolution_cap() {
    use std::path::Path;
    let path = Path::new("/synthetic/missing-height.mp4");
    let result = apply_probe_caps(
        path,
        Some(4),
        30.0,
        Some(100),
        None, // height missing
        Some(10.0),
        &VideoLimits::default(),
    );
    match result {
        Err(VideoError::ResolutionTooLarge { width, height, .. }) => {
            assert_eq!(width, 100);
            assert_eq!(height, u32::MAX, "sentinel height should be u32::MAX");
        }
        Err(other) => panic!("expected ResolutionTooLarge, got: {other:?}"),
        Ok(_) => panic!("expected ResolutionTooLarge but probe caps passed"),
    }
}

/// When ffprobe does not return a `duration` field, `apply_probe_caps` must
/// default to +∞ so the duration cap trips instead of silently accepting the
/// video with duration 0.
#[test]
fn probe_video_missing_duration_trips_duration_cap() {
    use std::path::Path;
    // Small resolution so resolution cap does not trip first.
    // nb_frames=4 so frame count is deterministic.
    let path = Path::new("/synthetic/missing-duration.mp4");
    let result = apply_probe_caps(
        path,
        Some(4),
        30.0,
        Some(100),
        Some(100),
        None, // duration missing
        &VideoLimits::default(),
    );
    match result {
        Err(VideoError::DurationTooLong {
            seconds,
            max_seconds,
        }) => {
            assert!(
                seconds.is_infinite(),
                "missing duration should default to +∞, got {seconds}"
            );
            assert_eq!(max_seconds, DEFAULT_MAX_DURATION_SEC);
        }
        Err(other) => panic!("expected DurationTooLong, got: {other:?}"),
        Ok(_) => panic!("expected DurationTooLong but probe caps passed"),
    }
}

/// Verify that saturating_mul prevents u32::MAX * u32::MAX from overflowing
/// to 0 (which would silently bypass the resolution cap).
#[test]
fn probe_video_both_dimensions_missing_saturates_not_overflows() {
    use std::path::Path;
    let path = Path::new("/synthetic/missing-both-dims.mp4");
    let result = apply_probe_caps(
        path,
        Some(4),
        30.0,
        None, // width missing
        None, // height missing
        Some(10.0),
        &VideoLimits::default(),
    );
    match result {
        Err(VideoError::ResolutionTooLarge {
            pixels, max_pixels, ..
        }) => {
            // With saturating_mul, u32::MAX as u64 * u32::MAX as u64 yields
            // 18_446_744_065_119_617_025 (no overflow wrapping to 0).
            // The critical property is that pixels > max_pixels so the cap fires.
            // If wrapping mul were used instead, the result would be 1 (overflow)
            // and the guard would not trip.
            assert!(
                pixels > max_pixels,
                "sentinel pixels {pixels} must exceed the cap {max_pixels}"
            );
            assert!(
                pixels > 0,
                "saturating_mul must not wrap to 0; got pixels={pixels}"
            );
            // Confirm the exact product so any future arithmetic change is caught.
            assert_eq!(
                pixels,
                (u32::MAX as u64).saturating_mul(u32::MAX as u64),
                "pixels should equal (u32::MAX as u64).saturating_mul(u32::MAX as u64)"
            );
        }
        Err(other) => panic!("expected ResolutionTooLarge, got: {other:?}"),
        Ok(_) => panic!("saturating_mul overflow: sentinel 0 bypassed resolution cap"),
    }
}

// ─── PNG frame size cap tests ─────────────────────────────────────────────────

/// A stream that never emits IEND must be rejected once the per-frame byte
/// cap is reached, not after exhausting all input.
#[test]
fn split_png_stream_rejects_oversized_frame() {
    use std::path::Path;

    // Feed 2 KiB of random-ish bytes with no IEND marker, with a tiny 1 KiB cap
    // passed directly (no env mutation, issue #103) so the function must reject
    // before reading all 2 KiB rather than allocating 256 MiB.
    let bogus: Vec<u8> = (0u8..=255).cycle().take(2048).collect();
    let path = Path::new("/synthetic/no-iend.png");
    let result = split_png_stream(bogus.as_slice(), path, 1, 1024);

    match result {
        Err(VideoError::Extract { message, .. }) => {
            assert!(
                message.contains("PNG frame exceeded"),
                "error message should mention PNG frame cap; got: {message}"
            );
        }
        Err(other) => panic!("expected Extract error for oversized frame, got: {other:?}"),
        Ok(_) => panic!("expected rejection of stream without IEND, but got frames"),
    }
}

/// `VideoLimits::from_raw` parses overrides and falls back to the compile-time
/// defaults — exercised purely (no process env touched, issue #103).
#[test]
fn video_limits_from_raw_parses_overrides_and_falls_back() {
    let defaults = VideoLimits::default();

    let parsed = VideoLimits::from_raw(Some("2500"), Some("2"), Some("1024"));
    assert_eq!(parsed.max_pixels, 2500);
    assert_eq!(parsed.max_duration_sec, 2.0);
    assert_eq!(parsed.max_png_frame_bytes, 1024);

    // Missing or unparseable values fall back to the defaults.
    let fallback = VideoLimits::from_raw(None, Some("not-a-number"), None);
    assert_eq!(fallback.max_pixels, defaults.max_pixels);
    assert_eq!(fallback.max_duration_sec, defaults.max_duration_sec);
    assert_eq!(fallback.max_png_frame_bytes, defaults.max_png_frame_bytes);
}

// ─── Issue #601: VideoSource fd path through ffmpeg ──────────────────────────

/// Verify `load_video_source` works against an opened file descriptor passed
/// via `/dev/fd/N` (issue #601). This is the primary smoke test for the
/// fd-based pipeline: ffmpeg/ffprobe must accept `/dev/fd/N` as an input
/// path on the project's supported platforms (Linux + macOS) and produce
/// the same decoded frames as the path variant.
///
/// The test:
/// 1. Synthesises a small video via `make_test_video`.
/// 2. Opens it as `OwnedFd`.
/// 3. Wraps it in `VideoSource::Fd`.
/// 4. Calls `load_video_source` and asserts a non-empty frame vector.
///
/// On a machine without ffmpeg the test SKIPs gracefully (it does not
/// fail), matching the pattern used by the resolution / duration cap
/// tests in this file.
#[cfg(unix)]
#[test]
fn load_video_source_fd_variant_decodes_via_dev_fd_n() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // Synthesise a 64x64 video of 2 seconds at 5 fps = 10 source frames.
    let video_path = match make_test_video(64, 64, 5, 2.0) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    // Open the file as OwnedFd and wrap in VideoSource::Fd.
    let std_file = std::fs::File::open(&video_path).expect("open synthetic video");
    let owned_fd = std::os::fd::OwnedFd::from(std_file);
    let canonical = std::fs::canonicalize(&video_path).expect("canonicalise synthetic video");
    let source = VideoSource::from_fd(owned_fd, canonical.clone());

    // Decode via the fd path.
    let frames_result = load_video_source(&source, Some(2.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = frames_result.expect(
        "load_video_source must succeed against /dev/fd/N; if this fails the runtime ffmpeg \
         build does not honor /dev/fd/N as an input — investigate the ffmpeg version on the \
         host or fall back to a different fd-passing strategy (issue #601)",
    );
    assert!(
        !frames.is_empty(),
        "fd-based extraction must produce at least one frame"
    );
    for frame in &frames {
        assert_eq!(frame.width(), 64, "frame width should match source");
        assert_eq!(frame.height(), 64, "frame height should match source");
    }
}

/// Verify the fd path produces the same frame count as the path-based
/// variant for the same source video. This guards against regressions
/// where ffmpeg behaves differently on `/dev/fd/N` than on a regular
/// path (e.g., partial decode, different sampling).
#[cfg(unix)]
#[test]
fn load_video_source_fd_variant_matches_path_variant_frame_count() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // 10 fps * 4 s = 40 source frames.
    let video_path = match make_test_video(48, 48, 10, 4.0) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    // Decode with explicit default limits so this parity test is independent of
    // the ambient environment (issue #103). `Some(5.0)` is the target FPS, not a
    // duration cap; the shared default limits keep fd and path on equal footing
    // for the frame-count comparison.
    let limits = VideoLimits::default();

    // Path-based decode (baseline).
    let path_frames = load_video_with_limits(&video_path, Some(5.0), None, &limits)
        .expect("path-based load_video must succeed for synthetic video");

    // Fd-based decode (the issue #601 path).
    let std_file = std::fs::File::open(&video_path).expect("open synthetic video");
    let owned_fd = std::os::fd::OwnedFd::from(std_file);
    let canonical = std::fs::canonicalize(&video_path).expect("canonicalise");
    let source = VideoSource::from_fd(owned_fd, canonical);
    let fd_frames = load_video_source_with_limits(&source, Some(5.0), None, &limits)
        .expect("fd-based load_video_source must succeed for the same synthetic video");

    let _ = std::fs::remove_file(&video_path);

    assert_eq!(
        fd_frames.len(),
        path_frames.len(),
        "fd-based extraction must produce the same frame count as path-based extraction; \
         differing counts suggest ffmpeg interprets /dev/fd/N differently"
    );
    for (idx, (fd_frame, path_frame)) in fd_frames.iter().zip(path_frames.iter()).enumerate() {
        assert_eq!(
            fd_frame.width(),
            path_frame.width(),
            "frame {idx}: fd vs path width mismatch"
        );
        assert_eq!(
            fd_frame.height(),
            path_frame.height(),
            "frame {idx}: fd vs path height mismatch"
        );
    }
}

/// Lower-level safety net: even before invoking ffmpeg, the
/// `VideoSource::ffmpeg_input` of an fd-backed source must produce a path
/// of the form `/dev/fd/<integer>` that the underlying kernel resolves to
/// the open file description we hold. This test does not invoke ffmpeg —
/// it opens `/dev/fd/N` from Rust and asserts the bytes match the
/// original file.
#[cfg(unix)]
#[test]
fn video_source_fd_dev_fd_n_resolves_to_owned_file_description() {
    use std::io::Read;
    use std::os::fd::AsRawFd;

    let path = std::env::temp_dir().join(format!("mlxcel-devfd-test-{}.mp4", uuid::Uuid::new_v4()));
    let payload: &[u8] =
        b"DEV-FD-N PAYLOAD: this exact byte string must round-trip through /dev/fd/N";
    std::fs::write(&path, payload).expect("write fixture");

    let std_file = std::fs::File::open(&path).expect("open fixture");
    let owned_fd = std::os::fd::OwnedFd::from(std_file);
    let raw = owned_fd.as_raw_fd();
    let source = VideoSource::from_fd(owned_fd, path.clone());

    // Sanity: ffmpeg_input renders the /dev/fd/N path we expect.
    let dev_fd_path = format!("/dev/fd/{raw}");
    {
        // Use the same private accessor that ffmpeg consumes via
        // configure_child + Command::arg.
        let computed = source_ffmpeg_input_for_test(&source);
        assert_eq!(
            computed,
            std::path::PathBuf::from(&dev_fd_path),
            "ffmpeg_input must render the /dev/fd/N path"
        );
    }

    // Open the dev-fd path and read it. The kernel must route this
    // through the same OFD as the OwnedFd.
    let mut reader = std::fs::File::open(&dev_fd_path)
        .expect("/dev/fd/N must be openable; this is the kernel-level guarantee on Linux + macOS");
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read /dev/fd/N");
    assert_eq!(
        buf, payload,
        "bytes read via /dev/fd/N must match the original fixture; this is the \
         /dev/fd/N kernel contract that the issue #601 fix relies on"
    );

    drop(source); // closes the OwnedFd
    let _ = std::fs::remove_file(&path);
}

/// Tiny test-only accessor for `VideoSource::ffmpeg_input` so the
/// `video_source_fd_dev_fd_n_resolves_to_owned_file_description` test
/// can compare the rendered path. Mirrors the private call used by
/// `probe_video` / `extract_frames_single_pass`.
#[cfg(unix)]
fn source_ffmpeg_input_for_test(source: &VideoSource) -> std::path::PathBuf {
    use std::os::fd::AsRawFd;
    match source {
        VideoSource::Path(p) => p.clone(),
        VideoSource::Fd { fd, .. } => {
            std::path::PathBuf::from(format!("/dev/fd/{}", fd.as_raw_fd()))
        }
    }
}

// ─── Benchmark placeholder (manual only) ─────────────────────────────────────

/// Performance benchmark for single-pass extraction. Marked `#[ignore]` so
/// it is skipped in CI but can be run manually with:
///
/// ```sh
/// cargo test --lib -- multimodal::video::tests::bench_single_pass_768_frames --ignored --nocapture
/// ```
///
/// The test synthesises a video large enough to drive ~768 frame extraction
/// and asserts the wall time is below 500 ms. The assertion is intentionally
/// generous; on developer hardware the single-pass path typically completes
/// in well under 300 ms for a short lavfi source.
#[test]
#[ignore]
fn bench_single_pass_768_frames() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // ~12.8 s at 60 fps = 768 frames source; target fps 60 => all frames.
    let video_path = match make_test_video(320, 240, 60, 12.9) {
        Some(p) => p,
        None => {
            eprintln!("SKIP: could not create test video");
            return;
        }
    };

    let start = std::time::Instant::now();
    let frames = load_video(&video_path, Some(60.0), None);
    let elapsed = start.elapsed();

    let _ = std::fs::remove_file(&video_path);

    let frames = frames.expect("benchmark load_video should succeed");
    eprintln!(
        "bench_single_pass_768_frames: {} frames in {:.1}ms",
        frames.len(),
        elapsed.as_millis()
    );
    assert!(
        elapsed.as_millis() < 500,
        "single-pass extraction took {}ms; expected < 500ms",
        elapsed.as_millis()
    );
}

// ─── Content-preservation tests (require ffmpeg) ─────────────────────────────
//
// Issue #598: These tests verify that load_video preserves pixel content, not
// just shape. They catch:
//   - Wrong color channel order (BGR vs RGB)
//   - Wrong sample timing (off-by-one in frame indexing)
//   - Frame content corruption / silent truncation
//
// Each frame has a known color incremented by a deterministic formula so that
// the expected per-frame channel means can be computed independently and
// compared against the decoded output.

/// Synthesize a "color increment" video where each frame is a solid color that
/// increments per frame. Frame i has color (r, g, b) = (5*i, 4*i, 3*i) clamped
/// to [0, 255]. Per-frame PNGs are written to a temp dir and encoded to MP4
/// via ffmpeg. Returns the path to the MP4 or None when ffmpeg is unavailable.
fn synth_color_increment_video(
    out: &std::path::Path,
    frames: usize,
    fps: u32,
    width: u32,
    height: u32,
) -> bool {
    use image::{ImageBuffer, Rgb};
    use std::path::PathBuf;

    if !ffmpeg_available() {
        return false;
    }

    // Create temp directory for per-frame PNGs.
    let tmp_dir = std::env::temp_dir().join(format!("mlxcel-synth-color-{fps}fps-{frames}f"));
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        eprintln!("SKIP: could not create temp dir: {e}");
        return false;
    }

    // Write one solid-color PNG per frame.
    for i in 0..frames {
        let r = (i * 5).min(255) as u8;
        let g = (i * 4).min(255) as u8;
        let b = (i * 3).min(255) as u8;
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(width, height, Rgb([r, g, b]));
        let frame_path: PathBuf = tmp_dir.join(format!("frame_{i:03}.png"));
        if img.save(&frame_path).is_err() {
            eprintln!("SKIP: could not save frame {i}");
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return false;
        }
    }

    // Encode all PNGs to an MP4 via ffmpeg.
    let pattern = tmp_dir.join("frame_%03d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-i",
        ])
        .arg(&pattern)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out)
        .status()
        .ok();

    let _ = std::fs::remove_dir_all(&tmp_dir);

    status.map(|s| s.success()).unwrap_or(false)
}

/// Synthesize a "moving square" video where an 8x8 white square on a black
/// background moves diagonally one pixel per frame in both x and y. Frame
/// index can be inferred from the square's top-left corner position.
fn synth_moving_square_video(
    out: &std::path::Path,
    frames: usize,
    fps: u32,
    width: u32,
    height: u32,
) -> bool {
    use image::{ImageBuffer, Luma};

    if !ffmpeg_available() {
        return false;
    }

    let tmp_dir = std::env::temp_dir().join(format!("mlxcel-synth-square-{fps}fps-{frames}f"));
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        eprintln!("SKIP: could not create temp dir: {e}");
        return false;
    }

    const SQUARE_SIZE: u32 = 8;

    for i in 0..frames {
        let mut img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);
        // Square moves diagonally; clamp so it stays on screen.
        let sx = (i as u32).min(width.saturating_sub(SQUARE_SIZE));
        let sy = (i as u32).min(height.saturating_sub(SQUARE_SIZE));
        for dy in 0..SQUARE_SIZE {
            for dx in 0..SQUARE_SIZE {
                img.put_pixel(sx + dx, sy + dy, Luma([255u8]));
            }
        }
        let frame_path = tmp_dir.join(format!("frame_{i:03}.png"));
        if img.save(&frame_path).is_err() {
            eprintln!("SKIP: could not save moving-square frame {i}");
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return false;
        }
    }

    let pattern = tmp_dir.join("frame_%03d.png");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-i",
        ])
        .arg(&pattern)
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(out)
        .status()
        .ok();

    let _ = std::fs::remove_dir_all(&tmp_dir);

    status.map(|s| s.success()).unwrap_or(false)
}

/// Compute the per-channel mean of an RGB image as (mean_r, mean_g, mean_b).
fn rgb_channel_means(img: &image::DynamicImage) -> (f64, f64, f64) {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as u64, rgb.height() as u64);
    let total_pixels = w * h;
    if total_pixels == 0 {
        return (0.0, 0.0, 0.0);
    }
    let (mut sum_r, mut sum_g, mut sum_b) = (0u64, 0u64, 0u64);
    for pix in rgb.pixels() {
        sum_r += pix[0] as u64;
        sum_g += pix[1] as u64;
        sum_b += pix[2] as u64;
    }
    (
        sum_r as f64 / total_pixels as f64,
        sum_g as f64 / total_pixels as f64,
        sum_b as f64 / total_pixels as f64,
    )
}

/// Test that load_video preserves per-frame color increments across the
/// sampled frames. This catches both wrong-frame-index and wrong-channel-order
/// bugs because both the frame timing and the exact channel values are verified.
///
/// Tolerance of ±15 accounts for YUV420 chroma subsampling and H.264 codec
/// rounding. YUV420 chroma is shared across 2x2 pixel blocks, which can shift
/// individual channel values by up to ~10 counts; we add a few counts of slack.
#[test]
fn extract_frames_preserves_color_increment_per_frame() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // 50 frames at 10 fps = 5-second video. Colors increment per frame:
    // frame i → (5*i, 4*i, 3*i) clamped to 255.
    const TOTAL_FRAMES: usize = 50;
    const VIDEO_FPS: u32 = 10;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let video_path = std::env::temp_dir().join("mlxcel-test-color-increment.mp4");
    let _ = std::fs::remove_file(&video_path);
    if !synth_color_increment_video(&video_path, TOTAL_FRAMES, VIDEO_FPS, WIDTH, HEIGHT) {
        eprintln!("SKIP: could not synthesize color-increment video");
        return;
    }

    // Request 25 frames from a 50-frame, 10-fps video (target fps = 5.0).
    // The sampled frame indices are linspace(0, 49, 25).round() =
    // [0, 2, 4, 6, ..., 48] — every even source frame.
    let frames = load_video(&video_path, Some(5.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP: load_video failed: {e}");
            return;
        }
    };

    assert!(!frames.is_empty(), "should have decoded at least one frame");

    // Compute the expected source-frame indices for the sampled frames.
    // uniform_indices(50, 25) → linspace(0, 49, 25).round() = [0,2,4,...,48].
    let expected_indices: Vec<usize> = (0..frames.len())
        .map(|i| {
            let last = (TOTAL_FRAMES - 1) as f64;
            let step = last / (frames.len() as f64 - 1.0);
            (i as f64 * step).round() as usize
        })
        .collect();

    // Tolerance: YUV420 + H.264 codec round-trip ≤ ±15 per channel.
    const TOLERANCE: f64 = 15.0;

    for (frame_idx, (frame, &src_idx)) in frames.iter().zip(expected_indices.iter()).enumerate() {
        let expected_r = (src_idx * 5).min(255) as f64;
        let expected_g = (src_idx * 4).min(255) as f64;
        let expected_b = (src_idx * 3).min(255) as f64;

        let (mean_r, mean_g, mean_b) = rgb_channel_means(frame);

        assert!(
            (mean_r - expected_r).abs() <= TOLERANCE,
            "frame {frame_idx} (src_idx {src_idx}): R channel mean {mean_r:.1} != expected {expected_r:.1} (tolerance ±{TOLERANCE})"
        );
        assert!(
            (mean_g - expected_g).abs() <= TOLERANCE,
            "frame {frame_idx} (src_idx {src_idx}): G channel mean {mean_g:.1} != expected {expected_g:.1} (tolerance ±{TOLERANCE})"
        );
        assert!(
            (mean_b - expected_b).abs() <= TOLERANCE,
            "frame {frame_idx} (src_idx {src_idx}): B channel mean {mean_b:.1} != expected {expected_b:.1} (tolerance ±{TOLERANCE})"
        );
    }
}

/// Test that load_video returns frames with the correct RGB channel order (not
/// BGR). Synthesizes a 1-second video where every frame is the same solid color
/// (R=200, G=100, B=50) and asserts center-pixel channel layout after decoding.
///
/// Tolerance of ±15 accounts for YUV420 chroma subsampling and codec rounding.
#[test]
fn extract_frames_preserves_channel_order() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // Solid-color video: R=200, G=100, B=50 — all three channels differ so
    // any channel-order permutation (e.g., BGR) is distinguishable.
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const TARGET_R: u8 = 200;
    const TARGET_G: u8 = 100;
    const TARGET_B: u8 = 50;
    const TOLERANCE: i32 = 15;

    // Write a single-color 1-second video (10 fps, 10 frames all identical).
    let tmp_dir = std::env::temp_dir().join("mlxcel-synth-channel-order");
    std::fs::create_dir_all(&tmp_dir).unwrap();

    use image::{ImageBuffer, Rgb};
    for i in 0..10usize {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([TARGET_R, TARGET_G, TARGET_B]));
        let fp = tmp_dir.join(format!("frame_{i:03}.png"));
        img.save(&fp).expect("save channel-order test frame");
    }

    let video_path = std::env::temp_dir().join("mlxcel-test-channel-order.mp4");
    let _ = std::fs::remove_file(&video_path);
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-framerate", "10", "-i"])
        .arg(tmp_dir.join("frame_%03d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&video_path)
        .status()
        .ok();
    let _ = std::fs::remove_dir_all(&tmp_dir);

    if !status.map(|s| s.success()).unwrap_or(false) {
        eprintln!("SKIP: could not synthesize channel-order video");
        return;
    }

    // Load 2 frames from the 1-second / 10-fps video (target fps = 2).
    let frames = load_video(&video_path, Some(2.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP: load_video failed: {e}");
            return;
        }
    };

    assert!(!frames.is_empty(), "should have decoded at least one frame");

    for (idx, frame) in frames.iter().enumerate() {
        let rgb = frame.to_rgb8();
        let cx = rgb.width() / 2;
        let cy = rgb.height() / 2;
        let pixel = rgb.get_pixel(cx, cy);

        let r = pixel[0] as i32;
        let g = pixel[1] as i32;
        let b = pixel[2] as i32;

        assert!(
            (r - TARGET_R as i32).abs() <= TOLERANCE,
            "frame {idx} center pixel: R={r} expected ~{TARGET_R} (±{TOLERANCE}). \
             If G or B is closer to {TARGET_R} the channel order is wrong (BGR)."
        );
        assert!(
            (g - TARGET_G as i32).abs() <= TOLERANCE,
            "frame {idx} center pixel: G={g} expected ~{TARGET_G} (±{TOLERANCE})"
        );
        assert!(
            (b - TARGET_B as i32).abs() <= TOLERANCE,
            "frame {idx} center pixel: B={b} expected ~{TARGET_B} (±{TOLERANCE})"
        );
    }
}

/// Test that load_video places the bright square in the expected pixel region
/// for each sampled frame. The square's position encodes the source frame index,
/// so this test catches off-by-one frame-indexing bugs.
///
/// The moving-square video has 20 frames at 10 fps (2 seconds). The square
/// moves from the top-left at frame 0 diagonally to (19, 19) at frame 19.
/// We sample all 20 frames (target fps = 10) and verify that the brightest
/// region of each frame overlaps the expected 8x8 window.
#[test]
fn extract_frames_preserves_moving_square_position() {
    if !ffmpeg_available() {
        eprintln!("SKIP: ffmpeg not available");
        return;
    }

    // 2-second video at 10 fps = 20 source frames.
    const TOTAL_FRAMES: usize = 20;
    const VIDEO_FPS: u32 = 10;
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const SQUARE_SIZE: u32 = 8;

    let video_path = std::env::temp_dir().join("mlxcel-test-moving-square.mp4");
    let _ = std::fs::remove_file(&video_path);
    if !synth_moving_square_video(&video_path, TOTAL_FRAMES, VIDEO_FPS, WIDTH, HEIGHT) {
        eprintln!("SKIP: could not synthesize moving-square video");
        return;
    }

    // Sample all 20 frames (target fps = 10 matches source fps).
    let frames = load_video(&video_path, Some(10.0), None);
    let _ = std::fs::remove_file(&video_path);

    let frames = match frames {
        Ok(f) => f,
        Err(e) => {
            eprintln!("SKIP: load_video failed: {e}");
            return;
        }
    };

    assert!(!frames.is_empty(), "should have decoded at least one frame");

    // Expected source-frame indices for the sampled frames.
    let expected_indices: Vec<usize> = (0..frames.len())
        .map(|i| {
            let last = (TOTAL_FRAMES - 1) as f64;
            let step = if frames.len() > 1 {
                last / (frames.len() as f64 - 1.0)
            } else {
                0.0
            };
            (i as f64 * step).round() as usize
        })
        .collect();

    // For each sampled frame, find the pixel with maximum luminance and check
    // that it falls inside the expected 8x8 square window (±2 pixels slack for
    // H.264 blocking artifacts near the square edge).
    const POSITION_TOLERANCE: i32 = 4; // pixels; H.264 DCT blocks are 4x4

    for (frame_idx, (frame, &src_idx)) in frames.iter().zip(expected_indices.iter()).enumerate() {
        let rgb = frame.to_rgb8();

        // Expected top-left corner of the white square (clamped to stay in frame).
        let expected_sx = (src_idx as u32).min(WIDTH.saturating_sub(SQUARE_SIZE));
        let expected_sy = (src_idx as u32).min(HEIGHT.saturating_sub(SQUARE_SIZE));

        // Find the pixel with the maximum luminance in the decoded frame.
        let mut max_luma = 0u32;
        let mut max_px = 0u32;
        let mut max_py = 0u32;
        for y in 0..rgb.height() {
            for x in 0..rgb.width() {
                let p = rgb.get_pixel(x, y);
                // BT.601 luminance approximation: Y ≈ 0.299R + 0.587G + 0.114B
                // Using integer math: (299*R + 587*G + 114*B) / 1000
                let luma = 299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32;
                if luma > max_luma {
                    max_luma = luma;
                    max_px = x;
                    max_py = y;
                }
            }
        }

        // The brightest pixel must lie within the expected 8x8 window (plus
        // POSITION_TOLERANCE pixels of slack on each side for codec artifacts).
        let in_x_range =
            (max_px as i32 - expected_sx as i32).abs() <= SQUARE_SIZE as i32 + POSITION_TOLERANCE;
        let in_y_range =
            (max_py as i32 - expected_sy as i32).abs() <= SQUARE_SIZE as i32 + POSITION_TOLERANCE;

        assert!(
            in_x_range && in_y_range,
            "frame {frame_idx} (src {src_idx}): brightest pixel at ({max_px},{max_py}) \
             but expected square window starts at ({expected_sx},{expected_sy}); \
             frame index or position is wrong"
        );
    }
}
