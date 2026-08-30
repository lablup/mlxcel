# 기술 보고서: PR #1517 - shared-audio-fft

**작성일**: 2026-08-30

**상태**: Benchmark 및 real-hardware validation 한계가 있는 구현 완료

**언어**: Rust

**위험도**: Medium

## 요약

PR #1517은 issue #1224를 해결하기 위해 중복된 audio spectrum transform을 하나의 shared bounded real FFT 구현으로 교체한다. 기존 코드는 Gemma4/Whisper용 host-side DFT helper, 별도 Nemotron DFT copy, Gemma3n 전용 radix-2 FFT copy를 가지고 있었다. 이 중복은 log-mel feature extraction에서 가장 비용이 큰 부분을 반복 구현하게 만들고, cardinality, malformed input 처리, 향후 수정의 일관성을 약하게 만들었다.

새 helper는 `src/audio/fft.rs`에 있다. 현재 사용되는 모든 frame size에 대해 exact-grid real magnitude spectrum을 계산한다. Whisper의 non-power-of-two 400-sample grid는 Bluestein convolution을 사용하고, 512 및 1024 sample power-of-two grid는 radix-2 FFT를 사용한다. Caller는 필요한 output bin 수를 요청하며, helper는 normal/empty input에서 그 cardinality를 보존하고 oversized transform size는 fail closed한다.

검증은 shared FFT unit test, Whisper, Gemma4, Gemma3n, Nemotron-Omni, formatting, clippy, whitespace check, stale-marker scan, benchmark target compilation, optimized standalone microprobe를 포함했다. 이번 구현 pass에서는 real checkpoint, ffmpeg, hardware-backed audio inference를 실행하지 않았다.

## 1. 문제 정의

Audio feature extraction은 mel filter 적용 전에 one-sided magnitude spectrum을 반복적으로 계산한다. 이번 PR 전에는 같은 책임이 세 가지 형태로 존재했다.

- Gemma4와 Whisper는 frame마다 O(N*K)인 naïve DFT loop를 사용했다.
- Nemotron-Omni는 같은 DFT pattern의 두 번째 copy를 가지고 있었다.
- Gemma3n은 해당 frontend 전용 1024-point radix-2 FFT helper를 가지고 있었다.

이는 performance issue일 뿐 아니라 correctness 및 maintenance risk였다. 각 copy가 bin cardinality, empty-frame behavior, normalization assumption, malformed configuration bound를 따로 보존해야 했기 때문이다. Issue는 실제 frame size를 안전하게 지원하는 하나의 shared implementation을 요구했으며, non-power-of-two input도 포함한다.

## 2. 변경 요약

| 영역 | 변경 |
|------|------|
| Shared FFT | Bounded host-side real FFT helper, radix-2 support, non-power-of-two exact grid용 Bluestein support, cardinality test, oversized-input fail-closed behavior를 `src/audio/fft.rs`에 추가했다. |
| Whisper | Shared DFT import를 새 FFT helper로 교체하면서 400-point DFT-grid semantics와 기존 mel output fixture를 보존했다. |
| Gemma4 | 기존 local DFT helper를 shared helper로 교체하고 golden log-mel test를 보존했다. |
| Gemma3n | Private 1024-point FFT copy를 제거하고 shared helper로 라우팅했으며, per-clip frame buffer를 재사용하게 했다. |
| Nemotron-Omni | Copied DFT path를 제거하고 shared helper를 사용하게 했으며, invalid 또는 oversized metadata에 대한 spectrum-config guard를 추가했다. |
| Benchmarks | 400, 512, 1024 sample frame에서 기존 DFT baseline과 shared FFT를 비교하는 Criterion target `benches/audio_fft.rs`를 추가했다. |
| Review fix | Extreme input이 unchecked allocation math에 의존하지 않고 기존 single zero-mel frame shape로 fallback하도록 Nemotron padded-waveform 및 output-buffer sizing에 checked arithmetic을 추가했다. |

### 통계

| 항목 | 값 |
|------|----|
| 구현 변경 파일 수 | 9 |
| 보고서 변경 파일 수 | 2 |
| 주요 구현 커밋 | `7da16bee` `fix(audio): share bounded real FFT frontend` |
| Review-fix 커밋 | `890dbf75` `fix(audio): bound Nemotron FFT allocation math` |
| PR | #1517 |
| Issue | #1224 |

## 3. 기술적 선택

### 3.1 Non-power-of-two frame을 silent remap하지 않고 exact DFT grid 보존

Whisper는 400-sample FFT length를 사용한다. 이 input을 512로 zero-padding한 뒤 512-grid bin을 반환하면 spectral frequency가 이동하고 golden fixture drift가 생길 수 있다. 따라서 shared helper는 non-power-of-two length에 Bluestein convolution을 사용한다. 내부적으로는 radix-2 convolution을 사용하지만, caller에게는 direct transform과 같은 DFT grid를 제공한다.

### 3.2 Bin cardinality를 caller contract로 명시

Helper는 요청된 output bin 수를 입력받는다. Valid input에서는 존재하는 real FFT bin을 계산하고, 추가 positive-frequency bin은 zero-fill한다. Empty input은 요청된 수만큼의 zero vector를 반환한다. Bounded maximum을 넘는 요청은 unbounded memory allocation 대신 empty vector를 반환한다. 이로써 caller-visible shape가 결정적으로 유지된다.

### 3.3 Transform setup 전에 host-side allocation bound 적용

`MAX_REAL_FFT_LEN`은 131072 sample로 설정했다. 이는 이 code path에서 예상되는 가장 큰 configured audio frontend budget과 맞는다. 이 bound를 넘는 input은 transform scratch allocation 전에 fail-closed result를 반환한다. Nemotron config loading도 invalid `n_fft`, `hop_length`, mel-bin count, non-finite preemphasis 값을 safe default로 clamp한다.

### 3.4 Normalization은 각 mel frontend에 유지

Shared helper는 magnitude만 반환한다. 각 frontend의 downstream power, mel-filter, log, clamp, normalization logic은 기존 위치에 남겨 두었다. 이렇게 semantic drift를 제한하고, 기존 golden test가 feature level에서 의도하지 않은 변경을 잡을 수 있게 했다.

## 4. Correctness review

- Shared helper는 현재 audio frontend가 사용하는 power-of-two 및 non-power-of-two frame size를 모두 다룬다.
- Unit test는 400, 512, 1024 sample frame에서 direct DFT baseline과 shared helper를 비교한다.
- Output-bin cardinality는 normal input과 empty input에 대해 직접 테스트한다.
- Oversized input과 oversized bin-count request는 대형 transform allocation 없이 fail closed한다.
- Whisper, Gemma4, Gemma3n, Nemotron-Omni가 같은 helper를 사용한다.
- Static stale-marker scan에서 touched audio path에 기존 target DFT helper name, Gemma3n-specific FFT helper, obsolete TODO marker가 남아 있지 않음을 확인했다.
- 최종 review pass에서 Nemotron padded waveform 및 feature output sizing 주변 unchecked arithmetic을 발견했고, 후속 commit에서 allocation 전 checked addition/multiplication을 추가했다.

## 5. Security review

이 PR은 file deletion, shell execution, network access, credential handling, SQL, untrusted executable content deserialization, web rendering path를 추가하지 않는다. 주요 security-relevant concern은 malformed audio metadata 또는 extreme input을 통한 resource exhaustion이다.

구현은 FFT length bound, requested FFT bin bound, Nemotron spectrum config sanitization, Nemotron allocation arithmetic check로 이 risk를 줄인다. Unsupported 또는 extreme shape는 unbounded allocation을 시도하지 않고 deterministic zero-output behavior로 fail closed한다.

## 6. Performance review

기존 DFT path는 frame마다 O(N*K)였다. Shared helper는 512 및 1024 point path를 O(N log N) radix-2 FFT로 바꾸고, 400 point path를 bounded radix-2 convolution size 위의 Bluestein convolution으로 바꾼다. Gemma3n은 frame마다 scratch buffer를 새로 만들지 않고 per-clip buffer를 재사용한다.

Criterion benchmark target은 compile된다. 다만 full optimized Criterion run은 cold bench-profile root-crate build가 bounded execution window 안에 끝나지 않아 완료하지 못했다. Exact shared helper를 대상으로 한 standalone optimized microprobe는 기대한 speedup과 numerical agreement를 보여주었다.

| Frame length | DFT time | FFT time | Speedup | Max abs error |
|--------------|----------|----------|---------|---------------|
| 400 | 824565 ns | 34815 ns | 23.7x | 7.034e-13 |
| 512 | 1062967 ns | 4806 ns | 221.2x | 5.776e-13 |
| 1024 | 4153601 ns | 10649 ns | 390.0x | 2.863e-12 |

이 숫자는 microprobe evidence이며, Criterion artifact 또는 real checkpoint throughput measurement를 대체하지 않는다.

## 7. Validation record

| Check | 결과 | Notes |
|-------|------|-------|
| `cargo fmt --all --check` | Pass | 구현 및 review fix 이후 formatting check. |
| `cargo test --lib audio::fft:: -- --nocapture` | Pass | 4 passed, 7326 filtered. DFT agreement, cardinality, empty input, oversized fail-closed behavior 검증. |
| `cargo test --lib audio::feature_extractor::tests:: -- --nocapture` | Pass | 8 passed. Gemma4 audio frontend golden behavior와 log-mel fixture 검증. |
| `cargo test --lib audio::gemma3n::feature_extractor::tests:: -- --nocapture` | Pass | 8 passed. Private FFT copy 제거 이후 Gemma3n deterministic 및 fixture behavior 검증. |
| `cargo test --lib whisper_mel -- --nocapture` | Pass | 13 passed. 400-point path를 Bluestein-backed exact-grid FFT로 바꾼 뒤 Whisper mel output compatibility 검증. |
| `cargo test --lib nemotron_h_nano_omni::feature_extractor -- --nocapture` | Pass | 5 passed. Malformed spectrum config fallback 포함. |
| `cargo test --lib deterministic_waveform_matches_pinned_speechlib_features -- --nocapture` | Pass | Current main에서 이 selector는 Nemotron이 아니라 Phi4MM 아래에 있으며, 사용 가능한 named selector가 통과했다. |
| `cargo test --bench audio_fft --no-run` | Pass | Criterion benchmark target compile 확인. |
| `cargo clippy --lib --bench audio_fft -- -D warnings` | Pass | 구현 및 review fix 이후 focused lint gate. |
| `git diff --check` | Pass | Whitespace error 없음. |
| Static stale-marker scan | Pass | `src/audio` 및 `benches/audio_fft.rs`에 target old DFT helper나 obsolete TODO marker가 남아 있지 않음. |
| `cargo bench --bench audio_fft` | Bounded stop | 두 번 시도했으나 cold bench-profile root-crate compilation 중 bounded polling 후 중단. |
| Standalone optimized microprobe | Pass | 400, 512, 1024 sample frame에서 max absolute error 3e-12 미만과 23.7x, 221.2x, 390.0x speedup 확인. |

## 8. Validation limits

- 실제 Whisper, Gemma4, Gemma3n, Nemotron-Omni checkpoint inference는 실행하지 않았다.
- ffmpeg-backed media ingestion path는 실행하지 않았다.
- Hardware-specific MLX 또는 GPU qualification은 실행하지 않았다.
- Unit constraint에 따라 broad workspace test, broad `cargo test --lib`, broad workspace clippy, serial all-tests, release build는 실행하지 않았다.
- Issue가 언급한 `deterministic_waveform_matches_pinned_speechlib_features`라는 Nemotron test는 current main에서 정확히 같은 selector가 Phi4MM 아래에 있다.

## 9. 후속 작업

- Warmed bench-profile build에서 retained Criterion benchmark를 실행하고 timing artifact를 보관한다.
- Target hardware에서 Whisper, Gemma4, Gemma3n, Nemotron-Omni real checkpoint audio feature extraction을 실행한다.
- 향후 media ingestion API가 user-controlled frame sizing을 직접 노출한다면 frontend-level oversized-input integration test를 추가한다.

## Appendix

- Issue: #1224
- PR: #1517
- Branch: `fix/issue-1224-shared-audio-fft`
