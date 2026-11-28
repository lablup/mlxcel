# 기술 보고서: PR #2048 - VoiceChat 마이크 점검, 프런트엔드 양자화, 어텐션 중복 제거

**날짜**: 2026-09-30

**상태**: 4비트 체크포인트에서 구현과 검증을 마쳤습니다. 실시간 마이크 대화는 검증하지 못했습니다. 머지 대기 중입니다.

**언어**: Rust

**위험도**: 낮음. 로더 시그니처에 인자가 하나 추가되고, 어텐션 리팩터링은 기존 일치성 테스트를 그대로 통과합니다.

## 요약

에픽 #1372에서 남은 세 가지를 이슈 #2043으로 묶어 PR 하나로 처리했습니다. 마이크 예제는 입력 장치 오류에 장치 이름과 macOS 권한 안내를 붙이고, 기본 설정을 읽지 못하면 지원되는 설정으로 대체합니다. 음성 프런트엔드(perception, FastConformer, RNNT joint)는 `(64, 4)` 대신 체크포인트의 `(group_size, bits)`로 로드합니다. `RelPositionMultiHeadAttention::forward`와 `stream`은 비공개 `attend` 하나를 공유합니다.

## 1. 문제 정의

- 마이크 예제는 CoreAudio 오류를 그대로 보여 주었고, 마이크 권한이 없는 에이전트 세션에서는 멈추거나 "Unknown property"로 실패했습니다.
- 프런트엔드의 모든 `UnifiedLinear`가 그룹 크기 64를 가정했습니다. 다른 그룹 크기로 양자화된 체크포인트는 잘못된 파라미터로 역양자화되어 오류 없이 쓰레기 값을 냅니다. LM과 TTS 로더는 이미 체크포인트 값을 읽습니다.
- `forward`와 `stream`이 헤드 분할, 위치 항, SDPA 호출, 출력 투영을 중복해서 가지고 있었습니다.

## 2. 변경 요약

- `examples/voicechat_microphone.rs`: `input_error`가 `input device {name}: {err}.`와 macOS 전용 안내를 만들고, `input_config`는 `supported_input_configs()`의 최대 샘플레이트 설정으로 대체하며, `build_input`과 `play`가 이 래퍼를 사용합니다.
- 로더: `VoiceChatPerception`, `FastConformerEncoder`, `CausalDwStridingSubsampling`, `ConformerBlock`(과 feed-forward), `RelPositionMultiHeadAttention`, `RnntDecoder`에 `quantization: (i32, i32)`를 추가했습니다. `model::front_end_quantization`이 `load`에 값을 공급하고, LM 호출은 `config.default_quantization()`을 그대로 씁니다.
- `attention.rs`: `attend(q_in, kv_in, pos_emb, mask)` 하나로 통합했습니다. `stream`은 기존 shape와 `pos_emb` 행 수 검사, 오류 문구를 유지하며 어텐션 전에 `pos_emb`를 검증합니다.
- `rnnt/mod.rs`: `step_frame`에서 `joint_logits`를 분리했습니다.
- 테스트: `attention_loads_non_default_quantization`, `rnnt_joint_loads_non_default_quantization`, `front_end_quantization_follows_checkpoint_config`를 추가했고, 기존 호출부는 `(64, 4)`를 넘깁니다. 이슈에 없던 `tests/nemotron_voicechat_stream_real.rs`도 포함합니다.

## 3. 기술적 선택과 그 이유

### 역양자화한 dense 가중치와 비교

`(32, 8)` 맵을 `(32, 8)`로 로드하고, 같은 양자화 결과를 `mlxcel_core::dequantize`로 풀어 만든 모듈의 출력과 1e-4 이내로 비교합니다. 이슈는 `(64, 4)`로 로드하면 일치하지 않음도 확인하라고 했지만, 그 경로는 `quantized_matmul` 안에서 MLX C++ `std::invalid_argument`를 던져 프로세스가 abort되고 잡을 수 없습니다. 그래서 저장된 scales 레이아웃을 검사합니다.

### LM은 Option 유지

이슈는 LM에도 `(64, 4)` 기본값을 재사용하라고 제안했습니다. 하지만 LM 로더는 `Option`을 받아 값이 있을 때만 텍스트 설정의 양자화를 채웁니다. dense 체크포인트에 `Some((64, 4))`를 넘기면 동작이 달라지므로 `config.default_quantization()`을 그대로 넘깁니다.

## 4. 검증

- `cargo test --release --lib -- models::nemotron_voicechat audio::fastconformer audio::rnnt`: 48개 통과.
- 환경 변수로 켜는 `nemotron_voicechat_front_real`, `nemotron_voicechat_streaming_real`이 4비트 체크포인트에서 통과했습니다.
- `mlxcel generate --audio question.wav --stream --seed 0`이 여전히 "[user] What is the capital of France"와 "The capital of France is Paris."를 출력합니다.
- clippy(lib, tests, examples, `voicechat-mic` 예제), `cargo fmt --check`, 계약 테스트 `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest`가 통과했습니다.
- `voicechat-mic` 예제가 빌드되고 `--list-devices`가 입출력 장치를 나열합니다.

## 5. 알려진 한계

- 실시간 마이크 대화, 권한을 끈 상태에서의 래핑된 오류 확인, #1376 체크박스는 마이크 접근이 되는 터미널에서 직접 실행해야 합니다(절차는 PR 본문).
- `quantization` 맵의 모듈별 재정의는 이전과 같이 지원하지 않습니다.
