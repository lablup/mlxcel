# PR #2504: Chat과 Responses에서 tool-call 인자를 증분 스트리밍

**Date**: 2026-10-21
**Status**: Completed (PR open, 미검증 항목은 5절에 정리)

## 요약

PR #2504(#2396 해결)는 모델이 생성하는 대로 tool-call 인자 조각을 스트리밍한다. 예전에는 생성이 끝날 때까지 호출에 대해 아무것도 보내지 않았고, 끝나면 인자 문자열 전체를 한 번에 보냈다. 이제 Chat Completions는 `delta.tool_calls[i].function.arguments` 조각을, Responses는 `response.function_call_arguments.delta` 이벤트를 보낸다. 대상 형식은 Hermes, Mistral `[ARGS]`와 목록 형식, Llama 3 JSON(태그 없는 형태와 `<|python_tag|>` 뒤), Qwen3-Coder XML이다. 생성 끝의 `parse_tool_calls`가 계속 최종 기준이다. 스트리밍 밖에서도 바뀌는 동작이 하나 있다. Hermes, Mistral, Llama 3, Granite는 이제 객체 `arguments`를 모델이 쓴 JSON 텍스트 그대로 반환한다. 리뷰 과정에서 문자 경계 panic 세 건을 고쳤다. 또 panic 격리를 추가했고, 스트리밍한 `response.completed`가 실제로 보낸 항목과 일치하게 했다. 변경 규모는 20개 파일, +3756/-150이다.

## 1. 문제 정의

인자가 도착하는 대로 화면에 그리거나 미리 검증하는 agent 클라이언트는 오래 아무것도 받지 못하다가 한꺼번에 받았다. Chat에서는 생성 중 `StreamFilter`가 tool markup을 숨겼다. 생성이 끝난 뒤에야 route가 호출마다 `tool_call_start` 청크 하나와 `tool_call_arguments` 청크 하나를 보냈다. Responses도 같은 구조여서 `FunctionCallArgumentsDelta` 하나에 전체 문자열이 담겼다. upstream mlx-vlm(Blaizzy/mlx-vlm#2502)은 이미 조각 단위로 스트리밍한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `tool_calls/json_scan.rs` (신규, 481) | 깊이 카운터를 쓰는 반복형 JSON 값 경계 스캐너. 최종 파서와 스트리머가 함께 사용 |
| `tool_calls/incremental.rs` + `incremental/` (신규, 약 1690) | `IncrementalToolCallStreamer`, 형식별 reader `json_call.rs`, `mistral.rs`, `qwen.rs`, `reconcile.rs`(`plan_final`) |
| `routes/tool_call_stream.rs` (신규, 420) | 토큰별 callback이 호출하는 `ChatToolCallStream`, `ResponsesToolCallStream` |
| `routes/chat.rs`, `routes/responses.rs`, `streaming_responses.rs` | route 연결, `ResponseStreamSender`가 최종 객체용으로 `output_item.done`을 기록 |
| `tool_calls/formats.rs`, `parser.rs` | `json_scan`으로 raw `arguments`를 잘라냄, 공유용으로 `resolve_tool_name`, `MISTRAL_HEADER_PATTERN`, `xml_param_keeps_raw_string` 분리 |
| 테스트 | `incremental_tests.rs`(543), `tool_call_stream_route_tests.rs`(342), `json_scan` 단위 테스트 |
| 문서 | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

커밋: `fc323904`(스트리머), `b4c4f5cb`(route), `6d1e6191`, `44959fc7`(문서), `eeb9ee0d`(테스트), `f29a87dc`, `b06f234e`(panic 수정과 격리), `5897a60f`(Responses 출력 일치).

## 3. 기술적 선택과 그 이유

### 3.1 인자는 원문 텍스트로, 스캐너 하나를 공유한다

이슈는 Chat에서 한 호출의 delta를 이어 붙이면 최종 `arguments`와 정확히 같아야 한다고 요구한다. 스트리머는 도착한 모델 텍스트만 보낼 수 있다. 그래서 최종 문자열도 그 텍스트여야 한다. `serde_json`으로 다시 직렬화하면 공백과 escape가 바뀌기 때문이다. `json_scan`이 `arguments` 값의 경계를 찾는다. 최종 파서는 `serde_json`이 호출 객체 전체를 검증한 뒤에만 원문을 잘라낸다. `object_members`가 뒤에 붙은 잡음을 거부하므로 잘라낸 결과는 정확히 `{`..`}`이고 JSON으로 읽힌다. 스트리머도 같은 스캐너를 바이트 단위로 돌리므로 두 경로의 인자 텍스트가 달라질 수 없다.

대가는 비스트리밍 응답에서 보이는 변화다. `{"city": "Seoul"}`는 이제 모델이 넣은 공백을 그대로 유지한다. Granite도 `try_hermes`에 위임하므로 함께 바뀐다. 문자열 값 `arguments`는 그대로다.

### 3.2 Qwen3-Coder: 스트리밍하면서 compact JSON을 만든다

Qwen3-Coder XML에는 잘라낼 JSON이 없다. 그래서 스트리머는 `extract_qwen_parameters`와 같이 생성 순서의 키로 compact JSON을 만든다. schema 타입이 일반 문자열인 파라미터는 JSON escape된 조각으로 스트리밍한다. 어떤 파라미터를 스트리밍할지는 `xml_param_keeps_raw_string`이 정하고, 테스트가 이 규칙을 `coerce_xml_param`에 묶어 둔다. 최종 파서가 값을 trim하므로 스트리머는 앞쪽 공백을 버리고, 끝 공백은 다음 텍스트가 올 때까지 보류한다. 두 쪽 모두 Unicode `is_whitespace`를 쓰므로 보류 결과가 정확히 같다. 다른 타입의 값은 `</parameter>`까지 버퍼링한 뒤 변환한다.

### 3.3 스트리머는 최종 파서의 규칙을 그대로 쓴다

한 이름으로 알린 호출이 최종적으로 다른 이름이 되면 프로토콜 오류다. 그래서 스트리머는 최종 파서의 판정 규칙을 재사용한다.

- thinking 블록은 건너뛴다(primed thinking 포함).
- Mistral과 태그 없는 JSON은 보이는 텍스트의 맨 앞에서만 인식한다.
- 이름은 `resolve_tool_name`을 거친다.
- `match_qwen_name`은 결과가 더 이상 바뀌지 않을 때만 실행한다.
- Mistral 헤더는 공유 정규식을 쓰고, 호출 수 한도도 같다.
- 스트림이 한 형식으로 정해지면 다른 형식의 호출은 스트리밍하지 않는다.
- 태그 없는 JSON은 `arguments` 멤버가 나온 뒤에야 `Start`를 보낸다. 그래서 `name` 키가 있는 JSON 답변을 호출로 알리지 않는다.
- 함수를 지정한 `tool_choice`에서는 다른 함수의 호출이 index 자리만 차지하고 스트리밍되지 않는다.

### 3.4 Tool-call id와 결과 맞추기

Chat은 호출의 첫 청크에 id가 있어야 한다. 그래서 `Start`가 id를 담는다. Mistral 목록 형식의 `id`는 `name`보다 먼저 올 때만 쓴다. 이때 `is_valid_model_call_id`로 검증하고, `try_mistral_nemo`와 같은 방식으로 중복을 제거한다. 그 밖의 경우에는 `Start`에서 id를 만들고, 나중에 오는 모델 id는 그 호출에 대해 무시한다. `plan_final`이 스트리밍한 id를 파싱된 호출에 다시 써 넣으므로 id가 두 번 생성되지 않는다.

`plan_final`은 스트리밍한 호출과 파싱된 호출을 index와 이름으로 짝짓는다.

| 경우 | 처리 |
|---|---|
| 일치하고, 보낸 텍스트가 최종 인자의 접두사 | 아직 보내지 않은 나머지만 전송 |
| 일치하지만 접두사가 아님 | 요청 id와 함께 `tracing::warn!`, 이후 delta 없음 |
| 이름 불일치 | 경고, 파싱된 호출을 새 index로 한 번에 전송 |
| 스트리밍한 호출이 파싱 결과에 없음 | 경고, Responses에서는 해당 항목을 `incomplete`로 닫음 |

### 3.5 리뷰: 신뢰할 수 없는 텍스트에서 나는 panic과 그 격리

**CRITICAL(`f29a87dc`).** `step_outer`는 한 바이트씩 진행하면서 `&s[i..]`로 잘랐다. 그래서 보이는 텍스트에 비ASCII 문자(한글, 악센트, 이모지)가 있으면 도구가 있는 모든 스트리밍 요청의 토큰별 callback에서 panic이 났다. Chat에서는 이 panic이 `cb_state` lock 안에서 일어나 mutex를 오염시킨다. 이제 `char::len_utf8`만큼 진행한다. 모든 marker는 ASCII 바이트로 시작하므로 문자 중간에서 marker가 시작될 수 없다.

**HIGH(`b06f234e`).** fuzzing으로 같은 종류의 panic 두 건을 더 찾았다. `ObjectStep::Invalid`를 처리하는 `step_json`, 그리고 첫 문자가 멀티바이트인 Mistral `ValueLead`다. 두 곳 모두 이제 문자 단위로 소비한다. 같은 커밋에서 `feed`를 `catch_unwind`로 감쌌다. panic이 잡히면 해당 조각의 이벤트를 버리고, 가장 최근 호출의 `sent`를 클라이언트가 실제로 받은 지점으로 되돌린다. 그다음 그 요청의 증분 스트리밍을 멈추고 error 수준으로 로그를 남긴다. 나머지는 최종 계획이 보낸다. 이 되돌림은 "인자 텍스트를 받는 것은 가장 최근 호출뿐"이라는 불변식에 기댄다. `send`의 `debug_assert`가 이를 확인한다.

### 3.6 리뷰: Responses 최종 객체가 스트림과 일치한다

호출이 문장 중간에서 시작하면 `Start`가 열린 message 항목을 닫고, 호출 뒤의 텍스트가 새 항목을 연다. 그런데 `build_response_object`는 호출을 맨 앞에 두고 message 텍스트를 합쳤으며 `fc_` id를 새로 만들었다. 그래서 `output_index`나 항목 id로 누적하는 SDK는 `response.completed`와 어긋났다. `5897a60f` 이후에는 `ResponseStreamSender`가 모든 `output_item.done`을 index와 함께 기록한다. tool call을 파싱한 스트리밍 요청에서는 `response.output`과 `previous_response_id`로 읽는 저장본을 이 기록으로 만든다. tool call이 없는 스트리밍 요청은 계속 `build_response_object`를 쓴다.

## 4. 검증

| 항목 | 결과 |
|---|---|
| `server::tool_calls::` | 496개 통과 |
| `server::routes::` | 483개 통과, 1개 ignored |
| `server::responses`, `server::muse_atem`, `server::router_front` | 56, 11, 29개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |
| Fuzzing(보안 리뷰): 입력 40만 개를 한 글자씩, 통째로, 무작위 청크로 입력 | 수정 후 120만 회 실행에서 panic 없음 |
| NUL, 제어 문자, U+2028, U+FEFF, `"`, `\`, `<`를 포함한 Qwen 문자열 값 | 15만 회. delta를 이어 붙이면 항상 유효한 JSON이고 최종 파싱 결과와 같음 |
| 처리량, 4글자 토큰 | 토큰당 28-130 ns(1 MB 문장, 1 MB 값, 짧은 호출 2만 개) |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. `server::routes::`와 `server::router_front`는 loopback 포트를 열어야 해서 sandbox 밖에서 실행했다. 단위 테스트는 각 사례를 한 글자씩, 그리고 다른 청크 분할로 넣어 `parse_tool_calls` 결과와 비교한다. route 테스트는 `scripted_streaming_for_route_tests`로 실제 handler를 구동한다.

## 5. 미검증 항목과 남은 항목

- **미실행:** `cargo test --workspace --features metal,accelerate`, 그리고 OpenAI Python SDK로 `Qwen3-Coder-30B-A3B-Instruct-4bit`를 확인하는 통합 검사. 이 호스트에는 Metal과 CUDA가 없다.
- **스트리밍하지 않는 형식:** `<tool_call>`로 감싸지 않은 Qwen `<function=` 호출, `<tool_call>` 하나에 들어간 여러 호출, Functionary v3.1과 GLM-4.7 본문. 이들은 최종 파싱이 계속 한 번에 보낸다.
- **알려진 동작:** 태그 없는 Llama 3 객체 뒤에 문장이 오면 스트리밍은 되었는데 `try_llama3`가 거부할 수 있다. 그러면 Chat에 `finish_reason: stop`인 고아 호출이 남는다. 이슈의 불일치 처리 규칙이 허용하는 경우다. 앞서 스트리밍한 호출이 최종 파싱에서 실패하고 다음 호출의 이름이 같으면 index 매칭이 잘못 짝지을 수 있다. 이때는 접두사 검사가 delta를 멈춘다.
- **바뀌지 않은 부분:** Hermes 블록 수에는 상한이 없다(최종 파서와 같다). Responses 스트리밍은 예전처럼 `max_tool_calls`를 무시한다. `Start` 이후 생성 오류가 나면 Responses 항목이 `in_progress`로 남는다.
- **규모:** +3756줄로 저장소의 1,000줄 분할 기준을 넘는다. `incremental.rs`는 550줄이다.

## 6. 학습 포인트

- 스트리밍 조각이 최종 값과 같아야 한다면 최종 값이 원문 텍스트여야 한다. 재직렬화와 증분 출력을 동시에 정확하게 맞출 수는 없다.
- 스트리밍 경로와 최종 경로가 스캐너 하나를 공유하면 둘이 어긋나는 문제가 통째로 사라진다. #2500에서 상수 하나를 공유한 것과 같은 이유다.
- 신뢰할 수 없는 텍스트를 바이트 단위로 훑는 코드는 자르기 전에 문자 단위로 진행해야 한다. lock을 잡은 callback 안의 panic은 틀린 답보다 피해가 크다. panic을 격리하고 최종 기준 경로로 넘겨야 한다.
