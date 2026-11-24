# PR #3382: Inkling 2 오디오 입력 거부

**날짜**: 2026-11-24
**상태**: 초안 PR 열림 (#3376 종료 예정). 좁은 범위 테스트 통과, `models/inkling-2-mini`로 CLI와 서버 확인

## 요약

Inkling 2에는 오디오 인코더가 없으므로, 모든 오디오 경로가 클립을 가져오거나 가중치를 로드하기 전에 하나의 메시지로 입력을 거부한다. 변경 전에는 서버가 `input_audio`를 받아들인 뒤 늦게 500으로 실패했고, `/v1/messages`의 오디오 블록은 조용히 버려졌으며, CLI는 템플릿 예외를 그대로 노출했다.

## 1. 문제

Mini의 가중치 맵에는 `model.llm.*`와 `model.visual.*`만 있고 `config.json`에 `audio_config`가 없다. 공식 채팅 템플릿은 `audio`, `input_audio`, `audio_url` 파트에서 `Inkling 2 does not accept audio input`을 발생시킨다. Small의 `config.json`에도 오디오 키가 없으며, PR 시점에는 가중치 인덱스가 아직 내려받아지지 않았다.

| 경로 | 변경 전 | 변경 후 |
|---|---|---|
| CLI `generate`/`run --audio` | 템플릿 렌더링 후 오류 | 공통 메시지, 종료 코드 1, 가중치 로드 전 |
| `/v1/chat/completions` `input_audio` | 클립 디코딩 후 워커에서 500 | 501 `not_supported_error` |
| `/v1/chat/completions` `audio_url` | 400 본문 역직렬화 오류 | 501 `not_supported_error` |
| `/v1/responses` `input_audio` | 워커에서 500 | 501 |
| `/v1/messages` 오디오 블록 | 200, 오디오 조용히 무시 | Anthropic 형식의 501 |
| `/v1/audio/transcriptions` | 워커에서 500 | 501 |
| `/apply-template` | 200, 오디오 센티널이 든 프롬프트 | 501 |

## 2. 설계

`ModelMediaSupport`에 `audio_refusal: Option<&'static str>`을 추가했다. `detect_model_media_support`는 `Inkling2VLM`에 대해 `audio: false`와 함께 `INKLING2_AUDIO_REFUSAL`을 설정하므로 어떤 라우트도 모델 id 문자열을 비교하지 않는다. `media_capability_rejection`은 오디오 종류에 이 문구를 사용하고, `--no-mmproj`는 `ModelMediaSupport::none()`에 거부 문구가 없어 기존 힌트를 유지한다. 전사 라우트도 일반 문구 대신 같은 필드를 읽는다.

거부 지점에 도달하도록 파싱 단계도 두 곳 바꿨다. `ContentPart::AudioUrl`로 `audio_url` 파트가 역직렬화되며, 오디오를 지원하는 체크포인트에서는 어떤 모델도 이 파트를 읽지 않으므로 여전히 요청 오류(400)다. `AnthropicContentBlock::Audio`(`input_audio`, `audio_url` 별칭 포함)는 번역기에서 `Unknown`과 똑같이 버려지고, `audio_refusal`이 설정된 경우에만 라우트가 거부하므로 다른 모델 계열의 동작은 그대로다.

CLI 검사 `refuse_audio_without_encoder`(`commands/generate.rs`)는 `-m` 해석 직후, 다른 로딩 이전에 실행된다. 워커의 일반 거부는 방어 계층으로 남긴다.

## 3. 검증

| 항목 | 결과 |
|---|---|
| `--lib server::media`, `server::startup`, `server::routes::prompt_inspection::inkling2`, anthropic 요청/번역기/라우트, `inkling`, `coverage_tests` | 통과 |
| `--bin mlxcel commands::generate` (설정 파일만 있는 Inkling 2 디렉터리 포함) | 통과, 가중치가 없어도 되므로 로드되지 않음 |
| `models/inkling-2-mini` CLI 및 서버 실행 | 표와 같은 메시지와 상태 코드, 텍스트 전용 채팅 요청은 여전히 200 |

## 4. 남은 항목

Small의 다운로드가 끝나면 `model.safetensors.index.json`에서 오디오 텐서가 없는지 확인해야 한다.
