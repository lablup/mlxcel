# 기술 보고서: PR #2037 - Nemotron VoiceChat 오프라인 전이중 추론

**날짜**: 2026-09-29

**상태**: 4비트 체크포인트에서 구현과 검증을 마쳤습니다. 머지 대기 중입니다.

**언어**: Rust

**위험도**: 낮음. 새 패밀리는 전용 감지 분기 뒤에 있습니다. 공유 코드 변경은 다음 세 가지뿐입니다.
- `NemotronHModel`에 추가된 메서드 두 개
- 접두사를 인자로 받게 바꾼 Gemma 3 블록 로더
- 새 `--extra-decoding-seconds` 플래그

## 요약

이 PR은 전이중 음성 모델인 NemotronLabs VoiceChat(`nemotron_voicechat`)을 추가하고 오프라인으로 실행합니다. 이 모델은 80ms 단위의 단일 타임라인 위에서 듣고, 받아쓰고, 텍스트로 답하고, 그 답을 음성으로 말합니다. 체크포인트 하나에 네트워크 네 개가 들어 있습니다.
- RNNT 받아쓰기 분기가 달린 캐시 인식 FastConformer 음성 인코더
- 텍스트 헤드와 함수 헤드를 가진 56층 Nemotron-H LLM
- EAR-TTS 음성 디코더
- 31 코드북 신경 코덱

`mlxcel generate --audio q.wav --output-audio a.wav -p "<시스템 프롬프트>"`는 받아쓴 내용과 답을 출력하고, 내장 `Aria` 음성의 22.05kHz 음성을 WAV로 씁니다.

기준 구현은 mlx-vlm 0.7.4 / mlx-audio입니다. 이 포팅은 4비트 체크포인트에서 기준 구현을 정확히 재현합니다. 테스트 발화의 받아쓰기, 모든 텍스트 id와 함수 id가 같고, `--seed 0`이면 샘플링된 음성 코드 1922개도 모두 같습니다. 이 PR은 에픽 #1372의 첫 번째 하위 이슈입니다. 캐시 인식 온라인 세션(#1378)과 `/v1/realtime` 엔드포인트(#1376)가 이 위에 올라갑니다.

## 1. 문제 정의

mlxcel은 음성 인식(Whisper)과 음성 합성(Kokoro)을 각각 별도의 요청/응답 엔드포인트로 제공해 왔습니다. 전이중 모델은 지원하지 않았습니다. 전이중 모델에서는 침묵과 겹치는 발화도 입력의 일부이고, 출력 길이는 토큰 예산이 아니라 입력 타임라인이 정합니다.

네 네트워크 중 셋은 mlxcel에 대응하는 구현이 없었습니다. chunked-limited 어텐션을 쓰는 인과적 FastConformer, mixture-of-Gaussians 헤드를 쓰는 RVQ 음성 디코더, ConvNeXt/iSTFT 신경 코덱입니다. 나머지 하나인 Nemotron-H는 이미 있었지만, 주입된 임베딩을 받아 최종 norm까지 통과시키는 경로와 두 번째 헤드가 없었습니다.

## 2. 변경 요약

- **감지와 로딩**: 체크포인트의 text config 안에 `nemotron_h`가 들어 있으므로, `nemotron_voicechat`을 `nemotron_h` 분기보다 먼저 매칭합니다. 매칭 결과는 `ModelType::NemotronVoiceChat`과 `LoadedModel::NemotronVoiceChat`이며, 레지스트리에는 generate 런타임, 오디오 입출력, `speech_to_speech` 카테고리로 등록됩니다. `LanguageModel` 구현은 텍스트 전용 호출을 Nemotron-H 백본에 넘기며, 트레이트를 채우기 위한 용도일 뿐입니다.
- **음성 프런트엔드**(`audio::nemotron_mel`, `audio::fastconformer`, `audio::rnnt`):
  - 대칭 Hann 윈도를 쓰는 preemphasis log-mel
  - 인과적 depthwise-striding 서브샘플링
  - `[70, 0]` chunked-limited 마스크를 쓰는 상대 위치 어텐션
  - LayerNorm을 쓰는 인과적 depthwise 합성곱
  - 2층 LSTM 예측망 위의 탐욕적 RNNT
- **코덱**(`audio::nemotron_codec`): ConvNeXt 인코더와 디코더, PRVQ, n_fft 16 STFT/iSTFT입니다. 온라인 세션에서 쓸 `decode_step`과 `CausalConv1dCache`도 함께 들어 있습니다.
- **TTS**(`models::gemma3_backbone`, `models::nemotron_voicechat::tts`):
  - 주입된 임베딩으로 동작하는 재사용 가능한 Gemma 3 스택
  - 문자 인식 서브워드 인코더
  - gated fusion
  - MoG 헤드
  - 마스크 기반 RVQ 정제
  - Aria 워밍업과 스텝별 생성
- **LLM 연결부**: `NemotronHModel::forward_embeds_to_hidden`과 `apply_lm_head`를 추가했습니다. `VoiceChatLanguageModel`은 `stt_model.*` 키 이름을 기존 로더의 이름으로 바꾸고 함수 헤드를 붙입니다.
- **오프라인 세션과 CLI**: `NemotronVoiceChatModel::generate_offline`이 타임라인을 실행합니다. `mlxcel generate`는 `-m` 해석 직후 VoiceChat 경로로 분기하고 `--extra-decoding-seconds`를 추가합니다. `--seed`를 주면 음성이 재현됩니다.

## 3. 기술적 선택과 그 이유

### 수치는 이슈 본문이 아니라 Python 기준 구현을 따름

이슈는 변환된 체크포인트를 실제로 돌려 보기 전에 작성됐습니다. 이슈와 기준 구현이 다르면 기준 구현을 따랐고, 그런 경우를 모두 PR에 기록했습니다.
- 체크포인트는 torch 레이아웃입니다. 로더가 멱등적인 shape 검사로 변환합니다.
- 서브샘플링 뒤 주파수 축 길이는 16이 아니라 17입니다.
- 코덱은 인코더와 디코더가 각각 13층입니다.
- perception, TTS 백본, MoG 헤드는 bf16 가중치에 대해 f32로 계산됩니다.

이슈의 "코덱 톤 SNR 20dB 초과" 기준은 달성할 수 없습니다. 손실 신경 코덱이라 기준 구현도 0.7dB에 그치기 때문입니다. 그래서 코덱은 기준 구현의 재구성 결과와 비교해 검증했습니다.

### 코드를 정확히 일치시키려면 네이티브 반정밀도 합산이 필요함

샘플링된 코드는 잔차 VQ 거리에 대한 `argmin`으로 결정되고, 그 거리 계산에는 bf16 코드북 norm이 들어갑니다. `mlxcel_core::sum_axis`는 bf16을 f32로 넓혀 합산한 뒤 한 번 반올림합니다. 이 결과가 MLX 네이티브 bf16 `mx.sum`과 norm의 약 5분의 1에서 달랐고, 코드가 뒤집히기에 충분한 차이였습니다. 두 호출 지점 모두 네이티브 합산으로 낮춰지는 einsum 합산(`audio::native_reduce`)을 쓰도록 바꿨습니다.

크레이트의 fused GeGLU도 bf16 원소의 약 절반에서 `mlx.nn.gelu_approx`와 결과가 달랐습니다. 그래서 TTS 스택은 연산 순서까지 똑같이 맞춘 `gelu_approx`를 씁니다. `gemma3::MLP`는 건드리지 않았습니다.

이 두 가지를 바꾸고 기준 구현의 전역 RNG 호출 순서를 따르면, 시드를 고정한 실행에서 모든 코드가 재현됩니다. 이 두 발견은 앞으로 mlx-vlm과 비트 단위로 일치해야 하는 모든 bf16 포팅에 그대로 적용됩니다.

### 오프라인 경로에서도 언어 모델 캐시 유지

기준 구현의 오프라인 경로는 기본적으로 타임라인 위치마다 LLM 전체 이력을 다시 계산합니다. 기준 구현의 캐시 모드도 같은 토큰과 코드를 냅니다. 이 포팅은 Nemotron-H 캐시를 유지해 위치당 작업량을 일정하게 제한합니다. #1378의 온라인 세션도 같은 방식으로 동작합니다.

### CLI 조기 분기

CLI는 VoiceChat을 모델 로드 이후가 아니라, 모든 텍스트 모델이 거치는 채팅 템플릿, 토크나이저, 메모리 사전 점검 설정보다 먼저 분기시킵니다. Florence-2가 모델 로드 이후에 분기하는 것과 다릅니다. 전이중 모델에는 그 설정이 전혀 해당되지 않기 때문입니다. `-p`는 시스템 프롬프트이고 `-n`은 의미가 없습니다. `-p` 없이 실행한 VoiceChat은 대화형 채팅이 아니라 시스템 프롬프트 없는 1회 실행으로 처리됩니다.

## 4. 검증

테스트 조건은 다음과 같습니다.
- 체크포인트: `mlx-community/NemotronLabs-VoiceChat-11B-4bit`
- 입력: 합성 음성 "What is the capital of France?"(1.8초)
- 시스템 프롬프트: "Be concise and answer in one sentence."
- 추가 디코딩 3초, 시드 0
- mlx-vlm 0.7.4 기준 덤프와 비교

| 항목 | 결과 |
|---|---|
| 받아쓰기 / 답 | "What is the capital of France?" / "The capital of France is Paris."(동일) |
| 텍스트 id, 함수 id(62개 위치) | 동일 |
| EAR-TTS 코드(62 x 31) | 1922개 중 1922개 동일 |
| 답 오디오와 기준 구현 비교 | 디코드 차이 1e-7 이내 |
| FastConformer 인코더 | 최대 절대 오차 5.4e-7 |
| 코덱 톤 코드 / 프롬프트 코드 | 372개 중 0개, 1085개 중 0개 불일치 |
| 시드 고정 CLI 실행 | WAV가 바이트 단위로 동일 |
| 답 WAV를 모델에 다시 입력 | "The capital of France is Paris" |

로컬 검증 항목은 다음과 같습니다.
- fmt, clippy(`-D warnings`)
- 변경된 모듈의 단위 테스트
- `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest`
- 환경 변수로 켜는 실체크포인트 테스트 `tests/nemotron_voicechat_{llm,front,codec,tts,offline}_real.rs`

`detection_tests` 실패 두 건은 main에서 온 것입니다. #2031이 `has_vision_config`를 바꾸면서 해당 테스트를 고치지 않았습니다.

## 5. 학습 포인트

- MLX Python 기준 구현과 정확히 일치하는 포팅을 하려면 수식만이 아니라 합산과 활성화 함수 공식까지 연산 순서대로 맞춰야 합니다. fused 커널이나 정밀도를 넓히는 합산 하나만으로도 샘플링되는 이산 출력이 뒤집힙니다.
- MLX의 dtype 흐름은 타입 승격 규칙을 따릅니다. bf16 가중치가 f32 입력을 만나면 f32 활성값이 나옵니다. 그래서 양자화 체크포인트를 "bf16으로 실행한다"고 이해하면 틀립니다.
- 양쪽이 같은 호출 순서로 MLX 전역 RNG를 쓰면 언어가 달라도 시드 샘플링이 재현됩니다. 이 모델에서는 정제 패스마다 uniform 한 번, normal 한 번을 호출합니다.

## 6. 알려진 한계

- 이번 실행에서는 GPU를 다른 작업과 공유했으므로 실시간 계수를 측정하지 않았습니다. `docs/nemotron-voicechat.md`에 오케스트레이터가 채울 자리 표시 표가 표시되어 있습니다. 프레임별 프로파일러는 #1378에서 들어옵니다.
- 체크포인트에 포함된 `Aria` 음성만, 배치 크기 1로만 지원합니다.
- 변환된 MLX safetensors 레이아웃만 로드할 수 있습니다.
