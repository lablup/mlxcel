# 기술 보고서: PR #1548 - Inkling dMel 오디오 입력

**작성일**: 2026-08-31
**작성자**: mlxcel maintainers
**상태**: 리뷰 중. 결정론적 frontend 및 통합 검증 완료
**언어**: Rust, Markdown
**위험 수준**: 높음

---

## 요약

PR #1548은 Inkling dMel 오디오 frontend를 추가하고 PR #1535의 HMLP multimodal shell에 연결한다. CLI `--audio`와 OpenAI 호환 서버 `input_audio`는 이제 제한된 WAV decode, mono downmix, 16 kHz resampling, dMel feature 추출, 엄격한 16-bin 양자화, 유효 row compact 처리, channel-sum embedding tower, prompt 확장, prepared-embedding merge를 공유한다. 오디오는 단독으로 사용하거나 still image와 결합할 수 있다.

Embedding 순서는 명시적이다. Token embedding에 Inkling의 선택적 input RMS normalization을 한 번 적용하고 HMLP image feature를 먼저 scatter한 뒤 audio feature를 두 번째로 scatter한다. 완성된 prepared tensor는 wrapper의 prepared-embedding method로 decoder에 들어가므로 새 token embedding으로 교체되거나 두 번 정규화되지 않는다. 오디오가 있는 요청은 classic multimodal 경로를 유지하고, PR #1540의 text-only native MTP 경로는 raw audio와 이미 준비된 audio tensor를 모두 거부한다.

구현은 결정론적 f64/CPU 레퍼런스 계산과 합성 MLX graph로 검증했다. 약 153.5 GB 공개 affine checkpoint와 약 170.7 GB native NVFP4 checkpoint는 검증 호스트에 없었으므로 이 보고서는 실제 checkpoint transcription, 품질, 메모리, throughput 결과를 주장하지 않는다.

## 1. 문제 정의

Inkling은 mlxcel에 이미 있는 Whisper 계열 frontend를 사용하지 않는다. 초당 20 frame을 소비하며 각 frame은 80개 log-mel channel을 16개 discrete bin으로 양자화해 표현한다. 각 channel은 하나의 1,280-row embedding table에서 자신에게 배정된 16-row 구간의 row 하나를 선택한다. 선택한 80개 vector를 합하고 RMS-normalize해 text width의 feature row 하나를 만든다.

여러 수치 세부사항이 결과에 직접 영향을 준다. STFT는 800 sample마다 center 없이 1,600-sample periodic Hann window를 사용한다. Slaney filterbank는 f64로 만들고 f32로 cast하며 mel 누적은 row별 f32 dot product를 사용한다. Quantizer boundary는 f64 midpoint를 f32 lattice에서 아래 방향으로 반올림하고 strict greater-than 비교를 사용하므로 boundary와 같은 값은 낮은 bin에 남는다. 이 요소를 조금만 바꿔도 tensor shape는 정상인 채 인접 bin 선택이 달라질 수 있다.

Control plane도 정확한 cardinality와 순서를 지켜야 한다. Prompt placeholder 하나는 유효 audio frame마다 token 하나로 확장해야 한다. Right-padding row는 tower에 들어가면 안 된다. Image/audio 혼합 요청에서는 공개 모델과 같은 순서로 feature를 scatter해야 한다. 서버의 prompt 합성은 system message나 history의 end marker를 선택하지 않고 현재 사용자 turn 안에 머물러야 한다. Prepared multimodal input은 text-only speculative 경로에 들어가면 안 된다.

## 2. 변경 요약

| 영역 | 결과 |
| --- | --- |
| Host 전처리 | WAV decode, stereo 평균, linear 16 kHz resampling, cancellation 검사, 최대 16 clip, 합계 5분 제한에 공통 bounded boundary 재사용 |
| dMel 추출 | Periodic Hann, uncentered RFFT, librosa 호환 Slaney filter, f32 mel 누적, `log10`, batching, mask, 정확한 frame 수 구현 |
| 양자화 | f64 center에서 아래 반올림한 f32 boundary를 만들고 strict lower-bin tie 규칙 적용 |
| Compact bridge | MLX 할당 전 모든 padding row를 제거하고 요청 순서대로 유효 frame만 연결 |
| Audio tower | Channel마다 16-row 구간 offset 적용, dense 또는 affine-quantized embedding gather, 80 channel 합산, RMS normalization, 최대 256 frame chunk 처리 |
| Weight loading | `model.audio.encoder.*`와 `model.audio.final_norm.weight`를 재사용 가능한 audio tower로 rename하고 dense/quantized layout 검증 |
| VLM 통합 | `InklingVlModel`에 processor metadata를 저장하고 normalized text -> image scatter -> audio scatter 구현 |
| Prompt 통합 | Template placeholder는 제자리 확장하고, 없으면 검증된 현재 사용자 end boundary에 완전한 wrapper 합성. Plain CLI mode는 명시적 raw prompt 뒤에 추가 |
| CLI 및 서버 | CLI `--audio`와 서버 `input_audio`에 Inkling dispatch 추가. Optional still image와 공통 preparation 통계 지원 |
| Capability 및 detection | Loaded-runtime audio capability를 노출하면서 InklingVLM은 계속 `vision_config`와 `model.visual.*` weight가 모두 있어야 한다는 visual-shell detection 규칙 유지 |
| Speculative 안전성 | Raw audio 및 raw payload 소비 후 prepared embedding 상태에서 MTP burst 거부 |
| 문서 | Frontend 공식, 제한, 지원 surface, scatter 순서, 실제 모델 검증 한계 기록 |

## 3. 구조와 데이터 흐름

1. CLI는 WAV path 하나를 제공하고 서버는 하나 이상의 `input_audio` byte payload를 제공한다. 두 경로 모두 공통 `AudioFamilyPolicy::inkling()` boundary에 들어간다.
2. Boundary는 요청 제한을 검사하고 WAV를 decode하며 channel을 평균하고 16 kHz mono로 linear-resample한 뒤 source 및 normalized work metric을 기록한다.
3. Inkling feature extractor는 batch를 가장 긴 clip에 맞춰 padding하고 STFT left context를 더한 뒤 50 ms frame마다 80개 log-mel 값을 계산한다. Padding row를 mask하고 clip별 valid frame 수를 보고한다.
4. Quantizer는 값을 `[-7, 2]`로 clamp하고 아래 반올림한 15개 boundary에 대한 strict 비교 횟수를 세어 `[0, 15]` 범위의 int32 ID를 만든다.
5. Host bridge는 유효한 `[frame, 80]` row를 clip 순서로 compact한다. Prompt preparation은 clip마다 audio placeholder 하나를 해당 valid-frame 수만큼 정확히 확장한다.
6. Optional image는 기존 Inkling tiler와 HMLP tower로 처리한다. Image placeholder는 실제 tile 수로 확장한다.
7. 모델은 text embedding을 한 번 정규화하고 HMLP row를 먼저 scatter한다. Compact dMel ID를 channel-sum audio tower에 통과시키고 audio row를 두 번째로 scatter한다.
8. Wrapper의 prepared-embedding prefill method가 merge tensor를 Inkling decoder에 직접 전달한다. Server scheduling, chunked-prefill, Neural Accelerator alignment, MTP burst gate는 prepared tensor를 확인하고 요청을 classic 경로에 유지한다.

## 4. 기술적 선택과 이유

### 4.1 근삿값 대신 레퍼런스의 bin 결정을 보존하기

Frontend는 filterbank와 bin center를 안정적인 f64로 계산한 뒤 공개 구현의 f32 실행 경계를 따른다. 모든 quantizer midpoint는 f64 midpoint보다 크지 않은 가장 큰 표현 가능 f32로 조정한다. 비교는 `>=`가 아니라 `value > boundary`이므로 정확한 tie는 결정론적으로 낮은 bin에 남는다.

Mel multiplication은 output row마다 f32 dot product 하나를 누적한다. 이는 bin boundary 근처의 값을 바꿀 수 있는 더 넓거나 결합 순서가 다른 matrix multiplication을 의도적으로 피한다. 테스트는 random-noise output을 독립 f64 reference와 비교하고 모든 boundary를 직접 검사한다.

### 4.2 Tower 전에 padding을 compact하기

Batch extraction은 효율적인 host 작업을 위해 right padding을 사용하지만 모델 계약에는 padding mask가 없다. 따라서 bridge는 feature row와 boolean mask를 함께 순회해 valid row만 유지하고 정확한 allocation cardinality를 검증한 뒤 `[total_valid_frames, 80]` MLX array를 만든다. Scatter 전에 placeholder 수와 tower output row 수가 정확히 같아야 한다.

### 4.3 이중 정규화 없이 HMLP와 결합하기

Text backbone은 normalized input embedding과 prepared-embedding forward method를 제공한다. `InklingVlModel`은 decoder 동작을 다시 구현하지 않고 이 API를 사용한다. Image preparation은 이미 정규화하고 image-scatter한 tensor를 반환하며 audio merge는 해당 tensor의 audio placeholder row만 바꾼다. Audio-only 요청도 같은 normalized text tensor에서 시작한다. 합성 회귀 테스트는 combined helper를 명시적인 image-first/audio-second merge와 비교하고 special-token suppression도 검사한다.

### 4.4 합성한 서버 audio를 현재 사용자 turn에 유지하기

서버의 normalized text view는 `input_audio` part를 생략할 수 있다. Template placeholder가 남아 있으면 제자리에서 확장하고 정확한 media cardinality를 요구한다. 없으면 runtime이 공개 Inkling structural token을 resolve하고 마지막 `[message_user, content_text]` 경계를 찾으며 이어지는 end marker를 요구한다. 뒤에 user content가 더 있으면 거부하고 end marker 바로 앞에 audio wrapper를 삽입한다. 회귀 테스트는 system/history content, companion image part, 현재 질문, model-generation tail을 포함한다.

### 4.5 Multimodal prefill을 text-only MTP에서 제외하기

InklingVLM은 text-only 요청에 계속 MTP target으로 동작한다. Raw media는 speculative dispatch 전에 소비될 수 있으므로 raw image/audio payload와 `vlm_embeddings`를 독립적으로 검사한다. 테스트는 raw audio 상태와 준비 후 merge tensor만 남은 상태를 모두 다룬다.

## 5. 리뷰와 보강

Inline correctness, security, performance review에서 미해결 CRITICAL 또는 HIGH 문제를 찾지 못했다. 최종 구현은 다음을 보강했다.

- Checked arithmetic으로 host 또는 MLX 구성 전 feature, token, allocation 수를 제한한다.
- WAV 및 aggregate-duration admission을 model-specific feature 작업 전에 적용한다.
- Non-finite input, 잘못된 sample rate, zero frame, malformed shape, 범위를 벗어난 bin, placeholder mismatch는 unchecked MLX operation에 도달하지 않고 오류로 끝난다.
- Audio special-token ID는 `processor_config.json`에서 resolve하며 placeholder ID가 `config.json`과 일치해야 한다.
- Visual-shell detection은 visual config와 visual weight에 계속 의존하므로 audio weight만으로 VLM runtime을 만들 수 없다.
- 서버 audio admission은 preprocessing 전에 loaded model capability를 확인한다.
- Prompt 합성은 전역 end-token search 대신 구조화된 현재 사용자 boundary를 사용한다.
- Prepared audio tensor는 이를 버리거나 misalign할 수 있는 chunked/speculative 경로에서 제외한다.

Merge 전에는 독립 PR 리뷰와 CI가 필요하다. 이 보고서는 local focused test 통과를 merge 승인으로 간주하지 않는다.

## 6. 검증

| 게이트 | 결과 |
| --- | --- |
| `cargo test --lib inkling -- --nocapture` | 통과, 77/77 |
| `cargo test --lib inkling_audio -- --nocapture` | 통과, 6/6 |
| `cargo test --lib mtp_burst_declines -- --nocapture` | 통과, 3/3 |
| `cargo test --lib detect_model_media_support_recognises_inkling_video -- --nocapture` | 통과, 1/1 |
| `cargo check --lib` | 통과 |
| `cargo check --bin mlxcel` | 통과 |
| `cargo clippy --lib -- -D warnings` | 통과 |
| `cargo clippy --bin mlxcel -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `git diff --check` 및 `git diff --cached --check` | 통과 |

Focused Inkling filter는 f64 reference frontend, frame mask, strict quantizer boundary, compact row 순서, channel-sum tower, processor config, weight normalization, visual detection, 현재 사용자 prompt 삽입, image/audio 혼합 scatter, prepared prefill, MTP state 동작, raw/prepared speculative gate를 포함한다. 더 넓은 workspace, feature, platform matrix는 PR CI가 담당한다.

## 7. 검증 한계와 후속 작업

요청한 `models/Inkling-Small-mlx-4bit` checkpoint는 약 153.5 GB이고 native NVFP4 checkpoint는 약 170.7 GB다. 두 artifact 모두 검증 호스트에 없었다. Apple GPU generation, 실제 CLI/server transcription, word error rate, peak unified memory, throughput은 측정하지 않았다. 결정론적 frontend나 tiny synthetic graph에서 실제 모델 결과를 추론하지 않는다.

Checkpoint 기반 후속 검증에서는 정확한 host dMel ID를 공개 mlx-vlm과 비교하고 issue의 relative tolerance 안에서 tower row를 비교해야 한다. 같은 clean speech fixture를 CLI와 `/v1/chat/completions`에 전달해 placeholder 수가 `ceil(samples / 800)`인지 확인하고, image/audio 혼합 prefill memory를 이후 text decode와 분리해 측정해야 한다.

## 참고

- 에픽 #1313 및 이슈 #1311
- PR #1548 및 선행 PR #1532, PR #1535, PR #1540, PR #1546
- 공개 mlx-vlm Inkling audio, processor, feature-extractor, model 구현
- `docs/supported-models.md`
