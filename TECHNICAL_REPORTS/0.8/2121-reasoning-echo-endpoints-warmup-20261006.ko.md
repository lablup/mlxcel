# 기술 보고서: PR #2121 - reasoning 재주입을 /v1/responses, /v1/messages, warm-up으로 확장

**작성일**: 2026-10-06

**상태**: GB10에서 구현 및 검증 완료, 머지 대기.

**언어**: Rust (Responses 및 Anthropic route, chat route warm-up, 다음 턴 history 렌더링, 테스트), Markdown

**위험도**: 낮음에서 중간 (두 엔드포인트에서도 content만 있는 assistant 턴이 서버가 반환한 reasoning과 함께 렌더링될 수 있음. #2116 이후의 `/v1/chat/completions`와 같다. chat warm-up은 이제 응답을 reasoning과 함께, history에 reasoning이나 tool 필드가 있으면 raw-JSON 경로로 렌더링한다)

## 요약

Issue #2118은 PR #2116의 후속 작업이다. #2116의 `ReasoningEchoStore`는 `/v1/chat/completions`에서만 서버가 생성한 reasoning을 content만 있는 assistant 턴에 재주입했다. 이제 `/v1/responses`와 `/v1/messages`도 같은 방식으로 채우고 기록하며, #1144 다음 턴 warm-up은 content만 보내는 후속 요청이 자체 fill 후 렌더링하는 형태를 그대로 렌더링한다. AI21 Jamba-Reasoning-3B (`jamba-v0.1-4bit`)에서 두 엔드포인트는 매 턴 cached token 0에서 turn 2와 3 적중으로 바뀌었고, 이 모델에서 한 번도 실행되지 않던 chat warm-up이 이제 turn 2의 적중을 제공한다.

## 1. /v1/responses 및 /v1/messages

두 handler는 번역된 요청을 그대로 `prepare_chat_request_with_cache`에 넘겼다. 이제 `reasoning_echo.rs`의 두 helper로 chat route를 따른다. `render_request`(scope와 `fill`, 렌더링 전용)와 `record_reply`이다. cache context, tool parsing, 기록은 클라이언트가 보낸 그대로의 번역된 요청을 읽으므로, 이후 턴의 조회 키는 주입된 trace에 의존하지 않는다.

- Responses는 `split_reasoning`의 visible text와 `reasoning` item(non-streaming), stream emitter의 message 및 reasoning 누적값(streaming)을 기록한다. `previous_response_id` 체인은 저장된 message item은 재생하지만 reasoning item은 재생하지 않으므로 체인된 턴도 content만 있는 history이며, 되돌아온 턴과 같이 채워진다.
- Anthropic은 클라이언트가 받은 text(stop-sequence 처리 후)와 `thinking` 블록을 non-streaming과 streaming 모두에서 기록한다. 요청이 extended thinking을 켰을 때만 기록한다. 그렇지 않으면 thinking 블록이 전송되지 않아 어떤 클라이언트도 되돌려 보낼 수 없기 때문이다. 숨겨진 reasoning을 기록하는 것은 store의 "클라이언트가 되돌려 보낸 것과 같다"는 계약을 바꾸므로 별도 결정으로 남긴다.
- 우선순위는 그대로다. Responses `reasoning` input item과 Anthropic `thinking` 블록은 `Message.reasoning`으로 매핑되어 우선하며, tool-call 턴은 기록하지도 채우지도 않는다.

## 2. 다음 턴 warm-up (#1144)

warm-up은 채워지지 않은 요청을 렌더링하고 응답을 `reasoning: None`으로 덧붙였지만, 후속 요청은 둘 다 채운다. 그래서 Jamba에서는 probe가 이전 user 턴을 지시문 없이 렌더링해 "이번 턴 history의 확장" 검사에 실패했고 warm-up이 한 번도 실행되지 않았다. 변경은 세 가지다.

- route가 먼저 기록하며, `ReasoningEchoStore::record`는 이제 trace를 저장했는지 반환한다. warm-up은 저장된 경우에만 응답 reasoning을 받는다. 그 경우에만 후속 요청의 fill이 이를 재현하기 때문이다.
- `submit_next_turn_warmup`은 채워진 렌더링 요청을 받는다(non-streaming은 `render_request`, streaming은 그 복사본).
- `render_next_turn_history`는 `reply_reasoning`을 받고, `prepare_chat_request_with_cache`와 같은 조건으로 raw-JSON 또는 typed 렌더링 경로를 고른다. raw 경로만 `reasoning` / `reasoning_content`를 전달한다. 이로써 tool call이 있는 history의 warm-up도 정렬된다. 이전에는 실제 요청은 raw로, warm-up은 typed로 렌더링했다.

streaming 경로는 stream filter가 flush한 마지막 부분도 warm-up 응답에 덧붙인다. 기록에는 이미 덧붙이고 있었다.

## 3. 검증

새 테스트: `reasoning_echo_endpoint_tests.rs`는 scripted model과 축약한 Jamba 템플릿으로 실제 `/v1/responses`(input 배열과 `previous_response_id` 체인)와 `/v1/messages` handler를 streaming과 non-streaming으로 구동한다. 재주입 테스트 세 개는 모두 `MLXCEL_REASONING_ECHO_MAX_BYTES=0`에서 실패하며, 네 번째 테스트는 extended thinking이 없는 `/v1/messages`가 아무것도 기록하지 않는지 확인한다. `reasoning_echo_warmup_tests.rs`는 turn 1에서 2, turn 2에서 3으로 갈 때 warm-up 대상이 후속 요청의 채워진 렌더링의 prefix인지, reasoning 없는 응답이 중단되는지(이전 동작), 채워지지 않은 요청의 대상은 prefix가 아닌지, `record`의 반환값을 확인한다. `reasoning_echo`, `server::routes`, `server::chat_request`, `server::batch::scheduler::prompt_cache`, `server::prompt_cache`, `server::responses`, `responses_translator`, `anthropic`, warm-up 필터가 통과하고 clippy `--lib --tests -D warnings`와 rustfmt도 깨끗하다.

GB10 실제 서버, `jamba-v0.1-4bit`, content만 있는 3턴, `max_tokens` 2048, temperature 0, prompt cache 기본값. "이전"은 같은 바이너리에 `MLXCEL_REASONING_ECHO_MAX_BYTES=0`을 준 것으로, 세 경로 모두에서 기존 동작을 재현한다. `/v1/messages`는 cached token 필드를 보고하지 않으므로 scheduler의 `prompt-cache snapshot hit: restored N/M` debug 로그에서 값을 얻었다. Responses 값은 `usage.input_tokens_details.cached_tokens`이며 로그와 일치한다.

| 엔드포인트 | 모드 | cached 이전 | cached 이후 |
|---|---|---|---|
| `/v1/responses` | non-streaming | 0 / 0 / 0 | 0 / 52 / 102 |
| `/v1/messages` (thinking 켬) | non-streaming | 0 / 0 / 0 | 0 / 51 / 101 |
| `/v1/chat/completions` | non-streaming | 0 / 0 / 0 | 0 / 91 / 101 |
| `/v1/chat/completions` | streaming | 미실행 | 0 / 93 / 154 |

warm-up 귀속: 이전에는 `snapshot_warmups_run`이 0에 머물렀다. 이후 non-streaming turn 2는 `warm-up completed restored=51 delta=40 warmed=91` 다음에 `snapshot hit: restored 91/106 ... stored=91`을 기록했고 `snapshot_warmups_run`이 0에서 1이 되었다. boundary snapshot만으로는 51이었을 것이다. streaming turn 2와 3도 같은 방식으로 93, 154 토큰의 warm-up 항목에 적중했다. non-streaming turn 3은 boundary(101)에 적중했는데, turn 2가 thinking 블록 안에서 끝나 content가 비었고 warm-up은 빈 응답을 건너뛰기 때문이다.

## 4. 한계

- streaming `/v1/responses`와 `/v1/messages`는 priming되지 않은 `StreamFilter`를 만들지만 Jamba 템플릿은 `<think>`를 priming하므로, reasoning이 text로 스트리밍되어 기록할 것이 없다(둘 다 0 / 0 / 0). 이 routing 문제는 이번 변경 이전부터 있었으며 후속 작업으로 남긴다. 기록 경로 자체는 streaming route 테스트가 다룬다.
- `/v1/responses`는 `output_text` input part를 거부하므로, `response.output` item을 그대로 되돌려 보내는 클라이언트는 HTTP 400을 받는다. 위 실행은 message text를 문자열로 되돌려 보냈다.
- extended thinking이 없는 `/v1/messages`는 아무것도 기록하지 않는다(1절 참고).
- 범위 밖: ASR 호환 스트림, prompt-inspection route, router front, `/v1/responses`나 `/v1/messages`의 warm-up(둘 다 없음). store는 여전히 프로세스 단위다.
