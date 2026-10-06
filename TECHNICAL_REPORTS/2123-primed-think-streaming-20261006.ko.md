# 기술 보고서: Issue #2123 - 스트리밍 /v1/responses와 /v1/messages에서 primed think reasoning 분리

**작성일**: 2026-10-06

**상태**: GB10에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (Responses 및 Anthropic route, route 테스트), Markdown

**위험도**: 낮음에서 중간 (클라이언트에 보이는 변경: 열린 thinking 블록을 prime하는 템플릿에서 스트리밍 reasoning이 답변 텍스트에서 reasoning 이벤트나 thinking 블록으로 옮겨간다. prime되지 않은 프롬프트는 그대로다)

## 요약

AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`)는 모든 생성 프롬프트를 `<|im_start|>assistant\n<think>\n`로 끝내므로, 모델은 여는 마커 없이 reasoning을 쓰고 `</think>`로만 닫는다. `/v1/chat/completions`는 이미 `is_prompt_primed_open_thinking`으로 이를 감지해 stream filter를 thinking 상태로 시작하고 `thinking_enter_block_on_start`를 설정했다. 스트리밍 `/v1/responses`와 `/v1/messages`는 항상 `StreamFilter::new()`를 만들었고, Responses와 Anthropic의 네 경로 어디에서도 budget 플래그를 설정하지 않았다. 그래서 스트리밍 reasoning이 답변 텍스트로 클라이언트에 전달되었고, #2121의 reasoning 기록은 스트리밍 턴에서 아무것도 저장하지 못했으며, reasoning budget은 prime된 토큰을 세지 않았다.

## 1. 변경

- `stream_create_response`와 `stream_messages`는 `prepared`가 blocking task로 이동하기 전에 `state.thinking_markers`와 `prepared.prompt`로 `primed_open_thinking`을 계산하고, 참이면 `StreamFilter::new_primed_open_thinking()`을 사용한다.
- 네 경로 모두 (두 엔드포인트, 스트리밍과 비스트리밍) `options.thinking_enter_block_on_start = primed_open_thinking`을 설정한다. 비스트리밍 경로는 이미 `split_reasoning` / `split_visible_reasoning`으로 reasoning을 분리했고, budget 플래그만 빠져 있었다.
- chat route의 `ThinkingDelimiterEcho`(#1470 마커 재출력)는 적용하지 않는다. 두 엔드포인트는 reasoning을 별도 채널로 전달하므로 텍스트에 되돌려 넣을 구분자가 없다.

## 2. 스트림 형태

변경 전 스트리밍 `/v1/responses`: `output_item.added` (message), `content_part.added`, 이어서 reasoning과 답변의 모든 토큰이 `response.output_text.delta`로 나갔고(`</think>` 자체는 filter가 소비), `output_text.done`과 `completed`에는 "We need to answer: ..."로 시작하는 `message` 항목 하나만 있었다. `reasoning` 항목은 없었다.

변경 후: `output_item.added` (reasoning), reasoning에 대한 `response.reasoning_text.delta`, `reasoning_text.done`, `output_item.done` (reasoning), 그 다음 `\n\nParis.`만 담은 `output_text.delta`의 message 항목. `completed`는 `reasoning`, `message` 순으로 나열한다.

변경 전 extended thinking을 켠 스트리밍 `/v1/messages`: `text` 타입의 `content_block_start` 하나에 `text_delta` 이벤트로 reasoning과 답변이 함께 나갔다. 변경 후: `thinking` 블록(`thinking_delta` 이벤트) 다음에 답변만 담은 `text` 블록이 온다. extended thinking이 꺼져 있으면 비스트리밍 경로처럼 reasoning은 스트림에서 제외된다.

## 3. 검증

`reasoning_echo_endpoint_tests.rs`에서 마운트하는 새 `reasoning_echo_primed_stream_tests.rs`는 생성 프롬프트를 prime하는 축약 Jamba 템플릿과 닫는 마커만 있는 scripted 토큰을 사용한다. 두 스트림의 reasoning과 텍스트 분리, `completed` 출력 순서, 네 경로의 budget 플래그, prime되지 않은 프롬프트에서 플래그가 꺼져 있는지, 그리고 두 엔드포인트의 3턴 content-only 스트리밍 대화에서 3번째 턴이 앞선 두 user 턴을 thinking 지시와 함께 렌더링하는지(두 스트리밍 턴 모두 비어 있지 않은 reasoning을 기록했을 때만 가능)를 확인한다. 이전 route 코드에서는 6개 중 5개가 실패한다(prime되지 않은 대조 테스트만 통과). `reasoning_echo`, `server::routes`, `responses_`, `anthropic_`, `reasoning_stream`, `stream_filter`, `stream` 필터가 통과하고, clippy `--lib --tests -D warnings`와 rustfmt도 깨끗하다.

GB10 실서버, `jamba-v0.1-4bit`, `max_tokens` 2048, temperature 0, 클라이언트가 받은 텍스트를 되돌려 보내는 3턴 content-only 스트리밍 대화. cached 토큰은 scheduler의 `prompt-cache: request completed: cached=N/M` 로그 값이다(`/v1/responses` usage와 일치).

| 엔드포인트 | 변경 전 reasoning 노출 | 변경 전 cached | 변경 후 reasoning 노출 | 변경 후 cached |
|---|---|---|---|---|
| `/v1/responses` stream | 없음 (전부 `output_text`) | 0/46, 0/217, 0/358 | `reasoning` 항목, 텍스트 `Paris.` | 0/46, 41/90, 85/142 |
| `/v1/messages` stream, thinking | 없음 (`text` 블록 하나) | 41/46, 212/217, 353/358 | `thinking` + `text` 블록 | 0/46, 41/90, 137/142 |

변경 전 `/v1/messages` 수치가 높은 것은 클라이언트가 새어 나온 reasoning을 assistant content로 되돌려 보내 생성 텍스트가 그대로 재현되었기 때문이다. 그 history는 2~3배 길고 답변 채널에 reasoning을 담고 있다.

## 4. 한계

- 스트리밍 텍스트는 `/v1/chat/completions` 스트리밍과 마찬가지로 `</think>` 뒤의 `\n\n`을 유지한다. 비스트리밍 분리는 이를 잘라낸다.
- 닫는 마커를 끝내 쓰지 않는 prime된 생성은 전부 reasoning으로 스트리밍되고 텍스트는 없다. chat 스트림과 비스트리밍 `max_tokens` 동작과 같다.
