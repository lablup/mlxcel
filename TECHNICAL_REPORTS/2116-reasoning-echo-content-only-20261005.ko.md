# 기술 보고서: PR #2116 - content만 되돌려 보내는 history에 생성된 reasoning 재주입

**작성일**: 2026-10-05

**상태**: GB10에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (chat route, 새 bounded store, 테스트), Markdown

**위험도**: 낮음에서 중간 (content만 있는 assistant 턴이 서버가 그 턴에 생성한 reasoning과 함께 렌더링될 수 있음. 클라이언트가 `reasoning_content`를 되돌려 보낸 경우와 정확히 같으며, prompt cache가 켜져 있을 때만 동작)

## 요약

Issue #2110은 PR #2094의 후속 작업이다. #2094 이후 AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`)는 클라이언트가 `reasoning_content`를 되돌려 보낼 때만 prompt cache에 적중했다. OpenAI SDK 기본 동작처럼 `content`만 되돌려 보내는 클라이언트는 매 턴 `cached_tokens=0`이었다. 이제 chat route는 각 응답이 반환한 reasoning을 기억해 두었다가, 렌더링 전에 이후 요청의 content만 있는 assistant 턴에 다시 채워 넣는다. 키는 그 응답을 만든 대화만 일치시킬 수 있다.

## 1. 근본 원인

Jamba 템플릿은 다음 assistant 메시지에 `reasoning_content`가 정의되어 있을 때(또는 `<think>`로 시작할 때)만 이전 user 턴에 thinking 지시문을 유지한다. trace가 되돌아오지 않으면 turn 2에서 turn 1의 user 메시지가 지시문 없이 다시 렌더링되고, 지시문을 포함한 turn-1 history-boundary snapshot (#1143)이 turn-2 prompt의 prefix가 되지 못한다. 이는 템플릿 자체의 규칙이며, trace 없이 서버가 렌더링 방식을 바꿔서 해결할 수는 없다.

## 2. 설계

Issue가 제안한 키는 `(template_sig, assistant content 해시)`에 session을 더한 것이었다. 대부분의 SDK 요청이 공유하는 anonymous session에서는 이 키만으로는 안전하지 않다. 답이 둘 다 "Yes."인 서로 무관한 대화가 reasoning을 교차 주입받을 수 있기 때문이다. 구현된 키는 해당 assistant 턴 앞의 모든 메시지를 클라이언트가 보낸 그대로 BLAKE3로 요약한 digest를 추가한다. 전체 키는 domain separator, model id, `template_sig`, 확정된 session key, 이 prefix digest, 앞뒤 공백을 제거한 응답 content로 구성된다(스트리밍은 `</think>` 뒤의 빈 줄을 content로 전달하지만 non-streaming 응답은 이를 제거한다). 따라서 trace는 그 응답을 스스로 만들어 낼 수 있었던 요청에서만 발견된다.

- 주입은 chat route에서 `prepare_chat_request_with_cache` 이전에 요청의 복사본에 대해 수행된다. 렌더러는 되돌아온 trace와 동일한 raw-JSON 경로를 타므로, 주입된 형태는 구조적으로 되돌아온 형태와 바이트 단위로 같게 렌더링된다. cache context, tool parsing, 기록 단계는 받은 그대로의 요청을 읽으므로 재주입된 턴이 이후 턴의 키를 바꾸지 않는다.
- 되돌아온 reasoning이 우선한다. 비어 있지 않은 reasoning을 가진 메시지는 조회하지 않는다. tool-call 턴과 inline `<think>` 블록이 있는 content는 채우지 않는다.
- content 수정, 이전 메시지의 수정이나 순서 변경, kwargs나 tools 변경, 다른 session이나 다른 model은 모두 miss이며, miss는 아무것도 바꾸지 않는다.
- thinking 블록 안에서 잘린 응답은 content가 비어 있으며 이것도 기록한다. prefix digest가 이 항목을 해당 대화에만 묶는다. 같은 대화를 다시 실행하면 최신 trace가 이전 것을 대체하며, 이것 역시 그 대화가 만들어 낼 수 있었던 trace다. 아래 Jamba 스트리밍 실행(turn 2가 `max_tokens`에 도달)에서 이 처리가 필요했다.
- 상한: store는 32바이트 키와 텍스트만 보관하며 16 MiB, 4096개 항목으로 제한되고 LRU로 축출한다. 적중 시 항목이 갱신되며, trace 하나는 예산의 1/8까지만 쓸 수 있다. `MLXCEL_REASONING_ECHO_MAX_BYTES`로 예산을 정하며 `0`이면 비활성화된다.
- prompt cache가 없을 때, `cache_prompt: false`, `--reasoning-format none` 또는 `deepseek-legacy`(trace가 이미 content에 inline으로 들어 있음), `--skip-chat-parsing`에서는 꺼진다.

Issue가 언급한 대안(마지막 user 턴 이전의 snapshot)은 필요하지 않았다.

## 3. 검증

`reasoning_echo_tests.rs`의 단위 테스트는 채워진 형태와 되돌아온 형태의 동일 렌더링, 저장되지 않은 턴, 되돌아온 reasoning의 우선, content와 history 수정, 다른 session/template/model, tool-call 및 inline-think 턴, prefill continuation, reasoning-only 응답, turn-3 조회, LRU 및 바이트 상한을 다룬다. `reasoning_echo_route_tests.rs`의 route 테스트는 scripted model로 실제 handler(스트리밍과 non-streaming)를 구동하며, store를 끄면 실패한다. prompt를 캡처하는 새 test-support 생성자를 쓰므로 main에서는 컴파일되지 않는다. `server::chat_request`, `server::prompt_cache`, `server::routes`, `reasoning` 필터가 통과하고 clippy `-D warnings`와 rustfmt도 깨끗하다.

GB10 실제 서버, content만 있는 history, 3턴, `max_tokens` 2048, temperature 0. "이전"은 같은 바이너리에 `MLXCEL_REASONING_ECHO_MAX_BYTES=0`을 준 것으로 기존 동작과 같다. Issue는 33c45053에서 0을 기록했다.

| 모델 | 모드 | cached_tokens 이전 | cached_tokens 이후 |
|---|---|---|---|
| jamba-v0.1-4bit | non-streaming | 0 / 0 / 0 | 0 / 41 / 89 |
| jamba-v0.1-4bit | streaming | 미실행 | 0 / 41 / 91 |
| qwen3-0.6b-4bit | non-streaming | 0 / 0 / 32 | 0 / 0 / 32 |
| qwen3.5-0.8b-4bit | non-streaming | 0 / 0 / 36 | 0 / 0 / 36 |

Qwen3은 양쪽에서 출력과 prompt 길이가 동일했다. 이 템플릿은 history의 assistant 턴을 content로만 렌더링하므로 주입된 trace는 영향이 없다. turn 2의 miss는 기존부터 있던 것으로 이번 범위 밖이다. Qwen3.5-0.8b는 temperature 0에서 매 턴 thinking 블록에서 반복하다 `max_tokens`에 도달했으므로 해당 행은 약한 근거다.

## 4. 한계

- 재주입은 `/v1/chat/completions`에서만 동작한다. `/v1/responses`, `/v1/messages`, ASR 호환 스트림, prompt-inspection route, router front는 해당하지 않는다.
- store는 프로세스 단위이며 재시작하면 비어 있다.
- 다음 턴 warm-up (#1144)은 확장하지 않았다. Jamba에서는 이전과 같이 probe 검사에서 중단된다.
- 모든 anonymous 클라이언트는 하나의 session bucket을 공유하므로, 서로 다른 anonymous 클라이언트의 동일한 대화는 서로의 항목에 일치할 수 있다. 그런 요청은 같은 응답을 스스로 만들어 낼 수 있었고, trace는 prompt에만 들어가며 반환되지 않는다.
