# PR #2500: 스트리밍과 비스트리밍 출력의 남은 차이 해소

**Date**: 2026-10-20
**Status**: Completed (PR open, 남은 항목은 5절에 정리)

## 요약

PR #2500(#2464 해결)은 PR #2458(#2394) 이후 남아 있던 스트리밍과 비스트리밍 chat 출력의 차이 8가지를 없앤다. 6가지는 이슈 본문에 있던 항목이다. 나머지 2가지(Harmony `analysis` 연결, tool-call marker 앞 공백)는 PR #2496에서 넘어와 이슈 코멘트로 추가했다. 각 항목에서 클라이언트는 스트리밍 여부에 따라 서로 다른 `content`나 `reasoning`을 받았다. 기준은 비스트리밍 cleaner다. 항목 2, 5, 6은 두 경로를 함께 바꾼다. 리뷰에서 `</think>` 뒤 공백과 공백이 특이한 role header를 추가로 고쳤다. 변경 규모는 9개 파일, +799/-278이다.

## 1. 문제 정의

| # | main의 차이 | 기준 동작 |
|---|---|---|
| 1 | `--reasoning-format none`에서 gpt-oss가 analysis와 답을 함께 content로 스트리밍 | 답만 반환 |
| 2 | `<|start|>assistant to=functions.X<|channel|>...`를 비스트리밍은 commentary로, 스트림은 호출로 처리 | 두 경로 모두 호출 |
| 3 | `<tool_call>..</tool_call>\nB`가 `\nB`로 스트리밍 | `B` |
| 4 | 여러 Harmony content 메시지가 구분자 없이 이어져 스트리밍 | `\n`으로 연결 |
| 5 | 도구 없는 MiniMax M3 출력에 `]<]minimax[>[` 잔여물이 남음(두 경로 모두) | 잔여물 없음 |
| 6 | Kimi-VL `◁think▷` / `◁/think▷`가 content에 들어감(두 경로 모두) | 안쪽 텍스트를 reasoning으로 보냄 |
| 7 | 여러 Harmony `analysis` 메시지가 구분자 없이 이어져 스트리밍(Chat `reasoning_content`, Responses `reasoning`) | `\n`으로 연결 |
| 8 | `Let me check.\n\n<tool_call>`의 끝 공백이 스트리밍됨 | 잘라냄 |

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `tool_calls/stream_filter.rs` (+312/-70) | 끝 공백 보류(`held_ws`)와 버림 규칙, Harmony content와 analysis의 메시지별 trim 및 `\n` 연결, `post_close_ws` 구간, `pending_harmony_role_header` 보류, `CHAT_DELIMITERS`에 Kimi-VL marker 추가 |
| `tool_calls/formats.rs` | 공유 함수 `harmony_role_suffix`, `harmony_route(role, header)`, `HARMONY_MAX_ROLE_SUFFIX = 256` |
| `tool_calls/parser.rs` | Hermes 처리 전에 MiniMax namespace 블록 제거, `THINKING_MARKER_PAIRS`와 `canonical_thinking_pair`에 Kimi-VL 쌍 추가 |
| `routes/chat.rs`, `router_front.rs` | chat 스트림 callback, chat flush, router_front 두 경로가 `ThinkingDelimiterEcho::thoughts_in_content` 사용 |
| 테스트 | `stream_parity_tests.rs`에 parity 테스트 9개(+223), `chat.rs`에 route 테스트, 필터 단위 테스트 10개 수정 |
| 문서 | `CHANGELOG.md`, `docs/supported-models.md` |

커밋: `4d0a6787`(구현), `1edd9916`(close 뒤 공백), `8adab292`(role header 보류), `2dfca808`(문서).

## 3. 기술적 선택과 그 이유

### 3.1 항목 1: `none`에서는 답만 반환한다

Harmony에는 표준 thinking marker 쌍이 없다. 그래서 `none`에서 스트림은 marker 기반 모델처럼 analysis를 `<think>...</think>`로 그려 줄 수 없다. 이슈는 비스트리밍 응답과 맞추기로 결정했다. `thoughts_in_content`는 marker가 echo되어 열린 블록 안에서만 thought를 content에 넣는다. Muse `to=self` reasoning도 marker 쌍이 없으므로 같은 규칙을 따른다. router_front의 비스트리밍 `none` 응답도 이제 marker와 close 뒤 간격을 포함한다. 이는 chat 경로의 `content_with_thinking_block`과 같다. 예전에는 marker 없이 reasoning만 덧붙였다.

### 3.2 항목 3과 8: 공백은 보류하고, 이미 보낸 텍스트는 되돌리지 않는다

이번 작업의 제약은 스트림이 증분 방식을 유지하고, 한 번 보낸 텍스트를 나중에 되돌리지 않는 것이었다. 따라서 marker 앞 공백을 나중에 잘라낼 수 없다. 필터는 끝 공백을 붙잡아 두었다가, 다음 조각이 들어오면 그 조각을 보고 처리 방법을 정한다.

| 다음에 오는 것 | 보류한 공백 |
|---|---|
| 보이는 텍스트 | 그 앞에 붙여 내보냄(코드 들여쓰기 유지) |
| 앞에 보이는 텍스트가 없는 tool call, 스트림 끝, Harmony 메시지 끝 | 버림(비스트리밍 cleaner가 잘라내는 위치) |
| 또 다른 공백 조각 | 계속 보류 |

비용도 있다. 공백만 있는 토큰은 다음 보이는 텍스트와 함께 도착한다. 들여쓰기가 이어지면 몇 토큰 늦어질 수 있다. 생성 맨 앞의 공백은 여전히 스트리밍된다. 이것을 버리면 `--reasoning-format none`에서 바이트 단위 일치가 깨지기 때문이다.

**리뷰 지적(HIGH, `1edd9916`).** 보류 로직이 thinking close 뒤의 공백까지 삼켰다. 답이 없는 `<think>r</think>  \n`, 그리고 `</think> <tool_call>..</tool_call>B`의 공백이 그 예다. 비스트리밍 경로는 둘 다 `post_close_whitespace`로 유지한다. 이제 필터는 close marker부터 다음 delimiter까지를 `post_close_ws` 구간으로 기록한다. 보류한 공백 중 이 구간이 버려지면 `post_thinking_gap`으로 보고한다. content 출력은 바뀌지 않는다.

### 3.3 항목 2와 그 보완: 두 경로에 같은 role header 규칙을 적용한다

`try_harmony`와 필터가 `harmony_role_suffix`를 공유한다. 그래서 role header의 `to=functions.X` 수신자는 두 경로 모두에서 호출로 처리된다. Muse header matcher는 `to=`…`<|message|>` 구간에서 다른 `<|`보다 `<|channel|>`이 먼저 나올 때만 물러난다. Muse header에는 `<|channel|>`이 들어가지 않는다.

**보안 지적(MEDIUM, `8adab292`).** 스트리밍 중에는 정확히 `<|start|>assistant to=` 형태만 보류했다. `to=` 앞에 공백 두 칸, 탭, 줄바꿈이 오면 `<|channel|>`이 도착하기 전에 role 부분이 content로 흘러나갔고, 호출 인자도 노출됐다. 이제 `pending_harmony_role_header`는 `<|start|>assistant` 뒤에 공백이 오고 `<|`가 없는 경우를 최대 256바이트까지 보류한다. `HARMONY_MAX_ROLE_SUFFIX`는 서로 달랐던 두 한계(스트림의 274바이트 lookback과 `try_harmony`의 무제한 탐색)를 하나로 대체한다. 일반 문장 속의 `<|start|>assistant `는 한계에 도달하면 그대로 내보낸다.

### 3.4 항목 4와 7: Harmony 본문을 메시지별로 trim하고 `\n`으로 잇는다

`try_harmony`는 메시지 본문마다 trim한 뒤 `\n`으로 잇는다. 이제 필터도 content와 `analysis` reasoning에 같은 방식을 쓴다. 그 결과 Responses의 `reasoning_tokens`도 두 경로에서 같게 유지된다. PR #2496에서 도입한 `collect_reasoning_chunked`(128바이트 청크)는 바꾸지 않았으므로 비스트리밍 재생은 계속 선형이다.

## 4. 검증

| 항목 | 결과 |
|---|---|
| `server::tool_calls::` | 468개 통과 |
| `server::routes::` | 461개 통과, 1개 ignored |
| `server::responses`, `server::router_front`, `server::muse_atem`, `server::reasoning_echo` | 각각 56, 29, 11, 30개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |
| 공백 보류, 1 MiB 공백을 바이트 단위로 입력 | 3.6초, 1 MiB 문자와 같음(선형) |
| Harmony 메시지 5만 개 스트리밍 | 0.47초 |
| Fuzzing, 무작위 입력 2만 개(Kimi marker, 한글, 이모지, U+3000, U+00A0) | panic 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. `router_front`와 `anthropic_count_tokens_tests`는 loopback 포트를 열어야 해서 sandbox 밖에서 실행했다. 항목별 테스트 8개는 모두 main에서 실패했다. 소스 변경을 stash하고 테스트 파일만 남겨 확인했으며, 항목 1은 main과 같은 동작을 하는 임시 `thoughts_in_content`를 넣어 확인했다. 리뷰에서 추가한 테스트 2개는 각 수정 직전 커밋에서 실패했다.

## 5. 남은 항목

- **기존 성능 문제:** `try_harmony`는 메시지 수에 대해 2차 시간이 걸린다. 본문마다 존재하지 않는 종결자를 텍스트 끝까지 찾기 때문이다. 메시지 5만 개에 18초, 32k 토큰 기준 약 40 ms다.
- **맨 앞 공백:** 생성 맨 앞의 공백은 스트리밍되지만 비스트리밍 응답에서는 잘린다.
- **처리하지 않은 형태:** 프롬프트가 이미 `<|start|>assistant`로 끝나면 ` to=functions.X<|channel|>`만 남는데, 두 경로 모두 이를 읽지 못한다. 첫 Harmony header 앞의 텍스트는 스트리밍되지만 `try_harmony`는 버린다. 프롬프트에서 미리 연 Kimi-VL `◁think▷`는 `primed_open_thinking_close_marker`가 인식하지 못한다.
- **상한 없음:** `held_ws`에는 상한이 없다. 다만 출력 길이를 넘지 않으며, route는 이미 전체 원문을 저장하고 있다.
- **미실행:** GB10 서버 시나리오.

## 6. 학습 포인트

- "되돌리지 않는다"는 제약이 있으면 모든 trim 규칙이 보류 규칙이 된다. 보류한 텍스트를 무엇이 결정하는지 정하고, 그 결정을 모든 청크 경계에서 테스트해야 한다.
- 같은 구조에 대해 두 경로가 각자 다른 한계를 두면(274바이트 lookback과 무제한 탐색) 긴 입력에서 결과가 갈린다. 상수 하나를 공유하면 이런 종류의 불일치가 사라진다.
- `chunkings()` 전체에 대한 parity 테스트는 문자열 전체만 넣는 테스트가 놓치는 버그를 찾는다. role header 노출이 그 예다. 이번 차이 대부분은 한 글자씩 넣을 때만 드러났다.
