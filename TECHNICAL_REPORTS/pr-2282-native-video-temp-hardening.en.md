# Technical Report: PR #2282 - Native-Video Temporary File Hardening

**Date**: 2026-10-13
**Status**: Ready for review
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

Native-video data URI and HTTP payloads are now stored as create-new 0600 files inside per-request 0700 directories. The directory guard is created inside the blocking closure before any byte is written, so request cancellation and partial-write failures remove the complete private directory after the writer finishes.

Native-video request preparation now receives the same immutable `MLXCEL_VIDEO_DIR_ALLOWLIST` snapshot as the frame fallback. Single-server, router-loaded model, and in-process startup paths all install this snapshot, and operator documentation states that environment or directory changes require a restart.

## 1. Problem Statement

The native-video resolver previously wrote directly under the system temporary directory with default permissions and created its cleanup guard only after a successful asynchronous write. A dropped request could therefore leave a readable clip behind, while a failed write could leave partial content. It also re-read and canonicalized the directory allowlist per request, producing behavior different from the frame fallback.

## 2. Technical Review

The implementation reuses `PrivateTempDir`, which creates its directory with mode 0700 and files with mode 0600 and `create_new`. The blocking closure owns the guard before invoking the writer; if its awaiting future is aborted, the closure completes and its unclaimed guarded result is dropped. Owned payload buffers move into the closure without an additional payload-sized copy.

The startup allowlist is passed explicitly into request preparation. A review found that the public in-process server installed media capability flags without the allowlist; a shared startup helper now installs both together, and a constructor regression test covers that integration.

## 3. Technical Decisions

### Keep synchronous file creation on the blocking pool

`PrivateTempDir` uses synchronous filesystem operations to guarantee exclusive creation and fixed Unix modes. Running the entire guard-create/write/result sequence in `spawn_blocking` avoids blocking Tokio workers while preserving one ownership boundary for cleanup.

### Treat the allowlist as startup configuration

All native and frame-fallback requests use the canonical directory list captured during model startup. A directory added later remains unavailable until restart, so request behavior no longer depends on when a path happens to appear.

## 7. Change Summary

| Category | Count | Summary |
|---|---:|---|
| Security | 2 | Private temporary storage and cancellation-safe cleanup |
| Integration | 1 | Startup allowlist propagated through every production server constructor |
| Tests | 18 | Temp lifecycle, startup snapshot, in-process constructor, and existing resolver coverage |
| Documentation | 2 | Restart requirements for native and fallback video paths |

## Appendix: Validation Evidence

- Implementation source commit: `790d981647901fdb690e223749a0162ed8f632f6`.
- Candidate `mlxcel-server` SHA-256: `fc7c3ebc3f1948fcd2dd9d535428123535a9cc12fd88b2139f576f87f635f9b4`.
- Focused CUDA-feature library tests passed: 3 temp lifecycle tests, 1 request allowlist snapshot test, 13 existing resolver tests, and 1 in-process startup test.
- A Qwen2.5-VL 3B 4-bit native data-video request answered `The main color in the video is red.` with 83 prompt tokens and 9 completion tokens. The private temp root was empty before and after the request, matching the baseline answer and counts.
- Contract gates passed with 123 Python checks and 49 WebUI checks. Formatting and diff checks passed.
- Independent implementation, security, and finalization reviews reported no remaining findings after the in-process constructor fix.

The end-to-end model run used Linux CUDA. macOS Metal inference was not run locally.
