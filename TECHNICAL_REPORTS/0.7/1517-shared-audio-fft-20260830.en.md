# Technical Report: PR #1517 - shared-audio-fft

**Date**: 2026-08-30

**Status**: Completed with benchmark and real-hardware validation limits

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

PR #1517 fixes issue #1224 by replacing duplicated audio-spectrum transforms with one shared bounded real FFT implementation. The previous code had a Gemma4/Whisper host-side DFT helper, a separate Nemotron DFT copy, and a Gemma3n-specific radix-2 FFT copy. That duplicated the most expensive part of log-mel feature extraction and made cardinality, malformed input handling, and future fixes inconsistent across audio frontends.

The new helper lives in `src/audio/fft.rs`. It computes exact-grid real magnitude spectra for all currently used frame sizes: Whisper's non-power-of-two 400-sample grid uses Bluestein convolution, while 512- and 1024-sample power-of-two grids use radix-2 FFT. Callers request the output bin count they need, and the helper preserves that cardinality for normal and empty input while failing closed for oversized transform sizes.

Validation covered shared FFT unit tests, Whisper, Gemma4, Gemma3n, Nemotron-Omni, formatting, clippy, whitespace checks, stale-marker scans, benchmark target compilation, and an optimized standalone microprobe. No real checkpoint, ffmpeg, or hardware-backed audio inference was run in this implementation pass.

## 1. Problem Statement

Audio feature extraction repeatedly computes one-sided magnitude spectra before applying mel filters. Before this PR, the same responsibility existed in three forms:

- Gemma4 and Whisper used a naïve DFT loop that was O(N*K) per frame.
- Nemotron-Omni carried a second copy of the same DFT pattern.
- Gemma3n carried a specialized 1024-point radix-2 FFT helper that only served that frontend.

This was a correctness and maintenance risk as well as a performance issue. Each copy had to independently preserve bin cardinality, empty-frame behavior, normalization assumptions, and malformed configuration bounds. The issue asked for one shared implementation that safely supports actual frame sizes, including non-power-of-two input.

## 2. Change Summary

| Area | Change |
|------|--------|
| Shared FFT | Added `src/audio/fft.rs` with a bounded host-side real FFT helper, radix-2 support, Bluestein support for non-power-of-two exact grids, cardinality tests, and oversized-input fail-closed behavior. |
| Whisper | Replaced the shared DFT import with the new FFT helper while preserving 400-point DFT-grid semantics and existing mel output fixtures. |
| Gemma4 | Replaced the old local DFT helper with the shared helper and preserved existing golden log-mel tests. |
| Gemma3n | Removed the private 1024-point FFT copy and routed feature extraction through the shared helper while reusing a per-clip frame buffer. |
| Nemotron-Omni | Removed the copied DFT path, routed through the shared helper, and added spectrum-config guards for invalid or oversized metadata. |
| Benchmarks | Added `benches/audio_fft.rs`, a Criterion target comparing the old DFT baseline against the shared FFT for 400, 512, and 1024 sample frames. |
| Review fix | Added checked arithmetic around Nemotron padded-waveform and output-buffer sizing so extreme inputs fall back to the existing single zero-mel frame shape instead of relying on unchecked allocation math. |

### Statistics

| Item | Value |
|------|-------|
| Implementation files changed | 9 |
| Report files changed | 2 |
| Primary implementation commit | `7da16bee` `fix(audio): share bounded real FFT frontend` |
| Review-fix commit | `890dbf75` `fix(audio): bound Nemotron FFT allocation math` |
| PR | #1517 |
| Issue | #1224 |

## 3. Technical Decisions

### 3.1 Preserve exact DFT grids instead of silently remapping non-power-of-two frames

Whisper uses a 400-sample FFT length. Zero-padding that input to 512 and returning 512-grid bins would move spectral frequencies and risk golden fixture drift. The shared helper therefore uses Bluestein convolution for non-power-of-two lengths. This computes the same DFT grid as the direct transform while still using a radix-2 convolution internally.

### 3.2 Make bin cardinality a caller contract

The helper accepts the requested number of output bins. For valid input it computes as many real FFT bins as exist and zero-fills any extra positive-frequency bins. Empty input returns exactly the requested number of zeros. Requests above the bounded maximum return an empty vector instead of allocating unbounded memory. This keeps caller-visible shapes deterministic.

### 3.3 Bound host-side allocation before transform setup

`MAX_REAL_FFT_LEN` is set to 131072 samples, matching the largest expected configured audio frontend budget in this code path. Inputs above that bound return a fail-closed result before transform scratch allocation. Nemotron config loading also clamps invalid `n_fft`, `hop_length`, mel-bin count, and non-finite preemphasis values to safe defaults.

### 3.4 Keep normalization in the mel frontends

The shared helper returns magnitudes only. Each frontend retains its existing downstream power, mel-filter, log, clamp, or normalization logic. This limits semantic drift and lets existing golden tests detect unintended changes at the feature level.

## 4. Correctness Review

- The shared helper covers power-of-two and non-power-of-two frame sizes used by current audio frontends.
- Unit tests compare the helper against a direct DFT baseline for 400, 512, and 1024 sample frames.
- Output-bin cardinality is tested directly for normal and empty input.
- Oversized input and oversized bin-count requests fail closed without large transform allocation.
- Whisper, Gemma4, Gemma3n, and Nemotron-Omni route through the same helper.
- Static stale-marker scans found no remaining targeted DFT helper names, Gemma3n-specific FFT helper, or obsolete TODO markers in the touched audio paths.
- A final review pass found unchecked Nemotron arithmetic around padded waveform and feature output sizing; the follow-up commit added checked addition and multiplication before allocation.

## 5. Security Review

The PR does not add file deletion, shell execution, network access, credential handling, SQL, deserialization of untrusted executable content, or web rendering paths. The main security-relevant concern is resource exhaustion through malformed audio metadata or extreme inputs.

The implementation addresses that risk by bounding FFT length, bounding requested FFT bins, sanitizing Nemotron spectrum config, and checking Nemotron allocation arithmetic before allocating padded waveform or feature buffers. Unsupported or extreme shapes fail closed to deterministic zero-output behavior rather than attempting unbounded allocation.

## 6. Performance Review

The old DFT paths were O(N*K) per frame. The shared helper changes 512- and 1024-point paths to O(N log N) radix-2 FFT and changes the 400-point path to Bluestein convolution over a bounded radix-2 convolution size. Gemma3n also avoids recreating its frame scratch buffer for every frame by reusing a per-clip buffer.

The Criterion benchmark target compiles, but the full optimized Criterion run did not complete within the bounded execution window because the cold bench-profile root-crate build was still running. A standalone optimized microprobe against the exact shared helper showed the expected speedup and numerical agreement:

| Frame length | DFT time | FFT time | Speedup | Max abs error |
|--------------|----------|----------|---------|---------------|
| 400 | 824565 ns | 34815 ns | 23.7x | 7.034e-13 |
| 512 | 1062967 ns | 4806 ns | 221.2x | 5.776e-13 |
| 1024 | 4153601 ns | 10649 ns | 390.0x | 2.863e-12 |

These numbers are microprobe evidence, not a substitute for retained Criterion artifacts or real checkpoint throughput measurement.

## 7. Validation Record

| Check | Result | Notes |
|-------|--------|-------|
| `cargo fmt --all --check` | Pass | Formatting check after implementation and review fix. |
| `cargo test --lib audio::fft:: -- --nocapture` | Pass | 4 passed, 7326 filtered; covers DFT agreement, cardinality, empty input, and oversized fail-closed behavior. |
| `cargo test --lib audio::feature_extractor::tests:: -- --nocapture` | Pass | 8 passed; covers Gemma4 audio frontend golden behavior including log-mel fixtures. |
| `cargo test --lib audio::gemma3n::feature_extractor::tests:: -- --nocapture` | Pass | 8 passed; covers Gemma3n deterministic and fixture behavior after removing the private FFT copy. |
| `cargo test --lib whisper_mel -- --nocapture` | Pass | 13 passed; covers Whisper mel output compatibility after switching the 400-point path to Bluestein-backed exact-grid FFT. |
| `cargo test --lib nemotron_h_nano_omni::feature_extractor -- --nocapture` | Pass | 5 passed; includes malformed spectrum config fallback. |
| `cargo test --lib deterministic_waveform_matches_pinned_speechlib_features -- --nocapture` | Pass | The selector exists under Phi4MM on current main, not under Nemotron; the named available selector passed. |
| `cargo test --bench audio_fft --no-run` | Pass | Confirms the Criterion benchmark target compiles. |
| `cargo clippy --lib --bench audio_fft -- -D warnings` | Pass | Focused lint gate after implementation and review fix. |
| `git diff --check` | Pass | No whitespace errors. |
| Static stale-marker scan | Pass | No targeted old DFT helper or obsolete TODO markers remained in `src/audio` and `benches/audio_fft.rs`. |
| `cargo bench --bench audio_fft` | Bounded stop | Attempted twice; stopped during cold bench-profile root-crate compilation after bounded polling. |
| Standalone optimized microprobe | Pass | Shows 23.7x, 221.2x, and 390.0x speedups with max absolute error below 3e-12 for 400, 512, and 1024 sample frames. |

## 8. Validation Limits

- No real Whisper, Gemma4, Gemma3n, or Nemotron-Omni checkpoint inference was run.
- No ffmpeg-backed media ingestion path was run.
- No hardware-specific MLX or GPU qualification was run.
- No broad workspace test, broad `cargo test --lib`, broad workspace clippy, serial all-tests, or release build was run, matching the unit constraints.
- The issue referenced a Nemotron test named `deterministic_waveform_matches_pinned_speechlib_features`, but current main has that exact selector under Phi4MM.

## 9. Follow-up Actions

- Run the retained Criterion benchmark on a warmed bench-profile build and archive the timing artifacts.
- Run real checkpoint audio feature extraction for Whisper, Gemma4, Gemma3n, and Nemotron-Omni on available target hardware.
- Add a frontend-level oversized-input integration test if a future media ingestion API exposes user-controlled frame sizing directly.

## Appendix

- Issue: #1224
- PR: #1517
- Branch: `fix/issue-1224-shared-audio-fft`
