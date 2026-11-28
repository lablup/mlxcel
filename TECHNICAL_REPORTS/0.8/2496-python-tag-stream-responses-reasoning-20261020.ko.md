# PR #2496: 스트림에서 Llama 3.x python_tag 호출 숨김, Responses reasoning 추가

**Date**: 2026-10-20
**Status**: Completed (PR open, 미검증 항목은 5절에 정리)

## 요약

PR #2496(#2470 해결)은 GB10의 0.8.0-beta smoke test에서 발견된 서버 출력 문제 두 가지를 고친다. 스트리밍에서 Llama 3.x `<|python_tag|>` tool call이 `delta.content`로 먼저 새어 나가고 `delta.tool_calls`로 다시 오던 문제를 막는다. `/v1/responses`는 gpt-oss Harmony analysis를 `reasoning` 출력 항목으로 반환한다. `reasoning_tokens`도 고정값 0 대신 실제 개수를 보고하며, 이 값은 스트리밍과 비스트리밍에서 같다. 리뷰 과정에서 세 가지를 더 고쳤다. 닫히지 않은 primed `<think>`에서 텍스트가 중복되는 문제, 비스트리밍 reasoning 재생의 2차 시간 비용, async executor에서 실행되던 토큰 계산이다. 변경 규모는 11개 파일, +439/-33이다.

## 1. 문제 정의

- **python_tag 노출.** `try_llama3`(`formats.rs`)는 생성이 끝난 뒤 `<|python_tag|>{...}`를 파싱했다. 하지만 `stream_filter.rs`의 `CHAT_DELIMITERS`에는 이 태그에 해당하는 행이 없었다. 그래서 스트림 필터는 호출 텍스트를 content로 흘려보냈고, 마지막 파싱이 같은 호출을 `tool_calls`로 다시 보냈다. 비스트리밍 출력은 정상이었으므로 #2458의 parity 목표가 충족되지 않은 상태였다.
- **Responses reasoning.** `routes/responses.rs`의 `split_reasoning`과 `extract_reasoning_from_raw`는 `<think>` 형식만 처리했다. 그래서 Harmony analysis가 버려졌다. `reasoning_tokens`는 두 경로 모두 리터럴 `0`이었다. chat 경로는 같은 analysis를 이미 `reasoning_content`로 제공하고 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `tool_calls/stream_filter.rs` | `("<|python_tag|>", EnterToolCall)`, `("<|eom_id|>", ExitToolCall)` 행 추가 |
| `tool_calls/parser.rs` | `clean_content_markers`가 호출하는 `strip_python_tag_blocks` 추가 |
| `tool_calls/formats.rs` | `try_llama3`가 보이는 텍스트 뒤의 태그도 받아들이고 `<|eom_id|>`에서 멈춤 |
| `routes/responses.rs` | 출처 세 개를 순서대로 읽는 `split_reasoning`, `count_reasoning_tokens`와 `count_reasoning_tokens_off_executor` |
| `routes/chat.rs` | `strip_unclosed_primed_thinking`을 `pub(crate)`로 공개, `REASONING_REPLAY_CHUNK_BYTES = 128`를 쓰는 `collect_reasoning_chunked` |
| 테스트 | `stream_parity_tests.rs`에 parity 테스트 4개, `responses.rs`에 테스트 7개, `chat.rs`에 청크 재생 테스트 |
| 문서 | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

커밋: `7026cfdb`(구현), `27928305`(닫히지 않은 primed thinking), `82846066`(청크 단위 재생), `9af71a31`(executor 밖에서 계산), `b7327309`(문서).

## 3. 기술적 선택과 그 이유

### 3.1 python_tag는 두 경로 모두에서 tool markup으로 취급한다

필터는 `<|python_tag|>`에서 tool-call 상태로 들어가고, `<|eom_id|>`나 스트림 끝에서 나온다. `<|p`로 시작하는 다른 delimiter는 없다. `<|eom_id|>`는 Muse의 `<|eom|>`와 5번째 바이트에서 갈라진다. 따라서 `chunkings()`의 모든 분할 지점에서 부분 일치 보류가 정상적으로 해소된다. 비스트리밍 경로에서는 `clean_content_markers`가 같은 블록을 지운다. 그래서 도구 없는 요청이나 알 수 없는 도구 호출도 두 경로에서 같은 content를 만든다. 이는 tool markup은 content가 아니라는 #1743 규칙을 따른 것이다.

그 결과 JSON이 아닌 python_tag 블록(Llama 내장 code interpreter)은 이제 content에서 빠진다. 스트리밍 중에는 그 블록이 JSON일지 필터가 알 수 없다. 두 경로가 함께 따를 수 있는 동작은 숨기는 것뿐이다. mlxcel은 Llama 내장 도구를 실행하지 않는다.

**태그 없는 JSON은 보류하지 않는다.** 태그 없는 Llama 3.x 형식에는 표지가 없다. 스트림에서 이를 구분하려면 끝날 때까지 모든 `{`를 붙잡아 두어야 하고, 그러면 JSON 모양의 답변이 전부 멈춘다. 이 형식은 지금처럼 content로 스트리밍된 뒤 `tool_calls`로 파싱된다.

### 3.2 두 번째 Harmony 파서 대신 chat 경로의 reasoning 출처를 재사용한다

`split_reasoning(raw, parsed, primed_open_thinking)`은 다음 순서로 reasoning을 읽는다.

1. `parse_tool_calls`의 `parsed.reasoning_content`(Harmony와 Muse가 설정)
2. 기존 `<think>` 추출
3. `chat::extract_reasoning_content`, 즉 Harmony 채널을 이미 분리하는 `StreamFilter` 분할

뒤의 출처는 앞의 출처가 아무것도 반환하지 않을 때만 실행되므로 reasoning을 두 번 추출하지 않는다. 보고된 시나리오에는 도구가 없었으므로 실제로 문제를 고치는 것은 3번이다. 이슈는 두 번째 Harmony 파서를 만들지 말라고 명시했다.

**리뷰 지적(HIGH, `27928305`).** primed `<think>`가 끝내 닫히지 않으면(`max_tokens` 도달 등) 3번 출처가 출력 전체를 reasoning으로 반환했다. 그런데 보이는 텍스트에도 같은 내용이 남아 있어 비스트리밍 응답이 같은 텍스트를 두 번 보냈다. 이제 `split_reasoning`은 보이는 텍스트에 chat 경로의 기존 규칙인 `strip_unclosed_primed_thinking`을 적용한다.

### 3.3 reasoning 토큰은 다시 인코딩해서 센다

기존에는 reasoning 토큰을 세는 기능이 없었다. `count_reasoning_tokens`는 추출한 reasoning을 trim한 뒤 `tokenizer.encode(text, false)`를 호출한다. 인코딩 오류가 나면 경고를 남기고 0으로 센다. 두 경로가 같은 텍스트를 세므로 값이 같다. 테스트는 스트리밍 경로처럼 텍스트를 한 글자씩 `StreamFilter`에 넣은 결과와 비교한다.

| 방안 | 장점 | 단점 |
|---|---|---|
| 스트리밍 중 필터에서 토큰 위치를 셈 | 재인코딩 없음 | 두 경로 모두 토큰별 채널 추적 필요, 비스트리밍 경로에는 관찰할 토큰 스트림이 없음 |
| **추출한 텍스트를 다시 인코딩** | 두 경로가 함수 하나를 공유, 구조상 값이 같음 | 재인코딩 비용, 세그먼트 경계에서 생성 id와 토큰 하나 차이가 날 수 있음 |

**보안 지적(MEDIUM, `9af71a31`).** 이 코드베이스의 tiktoken BPE는 piece마다 2차 시간이 걸린다. 25k자 구두점 연속 입력에 약 26초가 걸린다. 스트리밍 경로는 이미 `spawn_blocking` 안에서 셌지만, 비스트리밍 경로는 async executor에서 셌다. 이제 `count_reasoning_tokens_off_executor`가 이 계산을 blocking pool로 옮긴다.

### 3.4 비스트리밍 reasoning 재생은 제한된 크기의 청크로 넣는다

**보안 지적(HIGH, `82846066`).** `extract_reasoning_content`는 응답 전체를 `StreamFilter::feed` 한 번에 넣었다. 필터는 delimiter를 만날 때마다 버퍼 전체를 다시 훑으므로 비용이 "일치 횟수 × 길이"로 늘었다. `<tool_call>` 3.2만 회 반복에 80초, `<|python_tag|>` 3.2만 회에 42초가 걸렸다. 이 문제는 이 PR 이전부터 chat 경로에 있었다. 다만 이 PR이 이를 유발하는 marker 두 개를 추가했고, 비스트리밍 `/v1/responses`도 같은 함수를 거치게 했다. 이제 `collect_reasoning_chunked`가 UTF-8 문자 경계에서 자른 128바이트 청크를 넣어 feed 한 번의 버퍼 크기를 제한한다. 같은 입력이 약 0.29초에 끝난다. 필터 내부는 바꾸지 않았다. 테스트는 반복 marker, 한글 텍스트, Harmony 응답에서 청크 단위 결과가 한 글자씩 넣은 결과와 같은지 확인한다.

## 4. 검증

| 항목 | 결과 |
|---|---|
| `server::tool_calls::` | 457개 통과 |
| `server::routes::responses` | 25개 통과 |
| `server::responses` | 56개 통과 |
| `server::muse_atem`, `server::reasoning_echo` | 11개, 30개 통과 |
| `server::routes::chat`, `server::chat_request`(보안 리뷰) | 72개 통과, 128개 통과 및 1개 ignored |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. parity 테스트 4개는 `stream_filter.rs`, `parser.rs`, `formats.rs` 변경을 stash하면 모두 실패한다. 처음 추가한 `responses.rs` 테스트 5개는 새 코드를 호출하므로 main에서는 컴파일되지 않는다. 그래서 main의 로직을 그 자리에 되돌려 놓고 실패하는 것을 확인했다.

## 5. 미검증 항목과 남은 항목

- **GB10 시나리오 미실행:** Llama 3.1 8B 스트리밍 tool call, gpt-oss-20b `/v1/responses` 스트리밍과 비스트리밍.
- **#2464로 넘김:** 여러 세그먼트로 된 Harmony analysis를 파서는 `\n`으로, `StreamFilter`는 구분자 없이 잇는다. 그래서 두 경로의 텍스트가 다를 수 있고 토큰 수도 하나 차이 날 수 있다. tool-call marker 바로 앞의 공백은 스트리밍에서는 content로 나가지만 비스트리밍 응답에서는 잘린다.
- **기존 문제:** `/v1/responses`는 `--reasoning-format`을 무시한다. `extract_reasoning_from_raw`는 텍스트 어디서든 처음 나오는 `</think>` 리터럴에 매칭한다. 따라서 Harmony final 채널이 `</think>`를 언급하면 비스트리밍 응답에서 텍스트 일부가 reasoning으로 넘어간다.
- **LOW:** 버린 non-JSON python_tag 블록은 로그를 남기지 않는다. `try_llama3`는 첫 번째 python_tag 호출만 읽는다.

## 6. 학습 포인트

- 스트리밍 필터와 전체 텍스트 파서는 무엇이 markup인지 같은 기준을 써야 한다. 한쪽에만 delimiter가 빠지면 crash가 아니라 parity 버그로 나타난다. 다양한 청크 분할에 대한 parity 테스트가 가장 중요한 방어 수단이다.
- 이벤트마다 대기 버퍼를 다시 훑는 증분 필터에 문서 전체를 한 번에 넣으면, 청크로 나눠 넣을 때보다 훨씬 느릴 수 있다. 호출 한 번의 입력 크기를 제한해야 한다.
- 토큰 수를 세려고 텍스트를 다시 인코딩하는 방식은 단순하고 두 코드 경로를 같게 유지한다. 비용은 tokenizer가 결정하므로 blocking이 허용되는 곳에서 실행해야 한다.
