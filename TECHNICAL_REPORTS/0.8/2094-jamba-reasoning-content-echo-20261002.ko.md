# 기술 보고서: PR #2094 - 되돌려 보낸 reasoning을 `reasoning_content`로도 전달

**날짜**: 2026-10-02

**상태**: GB10에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust (chat request 렌더링, 테스트), Markdown

**위험도**: 낮음 (클라이언트가 reasoning을 되돌려 보낸 경우에만, chat template에 전달되는 raw JSON 메시지에 기존 `reasoning` 키와 같은 값을 가진 `reasoning_content` 키가 추가됩니다)

## 요약

이슈 #2089는 `AI21-Jamba-Reasoning-3B-4bit`(`jamba-v0.1-4bit`로 저장됨)가 빈 `content`를 반환하고 매 turn prompt cache를 놓친다고 보고했습니다. 조사 결과 두 가지로 나뉩니다. 빈 `content`는 `finish_reason=length`입니다. template이 항상 `<think>`를 미리 열어 두므로 64 토큰 예산은 reasoning 블록 안에서 끝납니다. 2048 토큰에서는 모델이 `</think>`를 닫고 `content`가 채워지므로 marker 인식은 정상이며 예산 처리는 바꾸지 않았습니다. cache miss는 서버 쪽 원인이 있었습니다. 되돌려 보낸 reasoning이 chat template에 `reasoning`으로만 전달되었는데, 이 template은 `reasoning_content`를 읽습니다.

## 1. 측정 (수정 전)

| max_tokens | turn 1 | turn 2 | turn 3 |
|---|---|---|---|
| 64 | length, content 비어 있음 | length, content 비어 있음 | length, content 비어 있음 |
| 2048 | stop, `Paris.` | length (인구 질문에서 반복) | stop, `The Louvre.` |

turn 1이 비어 있지 않은 content로 끝나도 turn 2와 3은 `cached_tokens=0`이었습니다. `reasoning_content`를 되돌려 보내도 prompt 길이가 같았고(두 경우 모두 66 토큰), 이는 서버가 렌더링 전에 그 값을 버렸다는 뜻입니다.

## 2. 근본 원인

Jamba template은 마지막 user turn에 thinking 지시문을 붙입니다. 이전 user turn에는 다음 assistant 메시지가 `<think>`로 시작하거나 `reasoning_content`를 정의할 때만 지시문을 유지합니다. `build_raw_json_messages_with_thinking`은 두 wire 표기를 하나의 필드로 합친 뒤 `reasoning`(Gemma 4 표기)으로만 내보냈습니다. 그래서 template은 reasoning을 보지 못했고, turn 2에서 turn 1의 user 메시지를 지시문 없이 렌더링했으며, 지시문을 포함한 turn 1의 history-boundary snapshot은 더 이상 turn 2 prompt의 prefix가 아니었습니다.

## 3. 변경 내용

전달되는 reasoning을 이제 `reasoning`과 `reasoning_content` 두 키에 모두 씁니다. gating은 이전과 같습니다(strip 대상 turn이거나 content에 이미 inline `<think>` 블록이 있으면 생략). 두 키를 모두 받는 기존 template(Gemma 4, Laguna)은 둘을 대안으로 읽으므로 reasoning이 두 번 렌더링되지 않으며, 테스트로 고정했습니다.

`content`만 되돌려 보내는 클라이언트는 여전히 지시문이 빠진 이전 user turn을 받습니다. 이는 template 자체의 규칙입니다. 테스트로 고정했고, `chat_request.rs`의 prefix 안정성 설명과 `docs/supported-models.md`에 `max_tokens` 안내와 함께 기록했습니다.

## 4. 검증

- `server::chat_request` 테스트 126개 통과, `chat_request_reasoning_content_tests.rs`의 새 테스트 5개 포함.
- `-D warnings` clippy와 rustfmt 통과.
- GB10 실서버, 새 프로세스, max_tokens 2048에서 `reasoning_content`를 되돌려 보내는 3 turn: finish stop/stop/stop, content `Paris.` / `4` / `Louvre`, prompt 46/93/135, cached_tokens 0/41/88.
- 같은 서버에서 content만 되돌려 보낸 경우: content는 비어 있지 않고 cached_tokens 0/0/0, 문서화된 동작과 일치.

## 5. 다루지 않은 부분

content만 보내는 클라이언트의 cache 재사용을 되살리려면 template이 history를 다시 쓰기 시작하는 지점(마지막 user turn 앞)의 snapshot과 그 지점에서 시작할 수 있는 warm-up이 필요합니다. 이는 boundary snapshot 설계 전반의 변경이므로 별도 이슈로 남깁니다.
