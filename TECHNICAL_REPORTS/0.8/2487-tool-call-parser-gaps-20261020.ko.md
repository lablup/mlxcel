# PR #2487: qwen3_coder, Mistral, pythonic tool-call 파서 결함 수정

**Date**: 2026-10-20
**Status**: Completed (PR open, 미검증 항목은 5절에 정리)

## 요약

PR #2487(#2392 해결)은 `src/server/tool_calls/formats.rs`의 tool-call 파서 세 개를 고친다. 이 파서들은 upstream mlx-lm과 mlx-vlm이 이미 처리하는 호출을 버리거나 망가뜨렸다. 해당 사례는 모델이 `>`를 빠뜨린 Qwen3-Coder 이름, 병렬 호출과 하이픈 이름을 쓴 Mistral `[ARGS]` 호출, 호출 여러 개나 중첩 값을 담은 pythonic 목록(LFM2 / LFM2-VL)이다. `ParsedToolCall.id` 필드도 추가했다. 이제 유효한 Mistral Nemo id는 새로 만든 id로 바뀌지 않고 Chat과 Responses 클라이언트에 그대로 전달된다. 변경 규모는 12개 파일, +563/-62이며 `tool_calls::` 테스트 453개가 통과한다.

## 1. 문제 정의

세 파서는 모두 생성이 끝난 텍스트를 한 번 읽어 구조체로 바꾸는 순수 함수다.

- **Qwen3-Coder.** `try_qwen3_coder`는 다음 `>`까지를 함수 이름으로 잘랐다. `<function=get_time\n<parameter=tz>UTC</parameter>`에서는 그 `>`가 파라미터 태그의 것이다. 그래서 이름이 `get_time\n<parameter=tz`가 되었고, 클라이언트는 알 수 없는 도구로 거부했다. 파라미터 이름도 같은 방식으로 잘렸다.
- **Mistral bracket.** `try_mistral_bracket`은 `^\s*(\w+)\[ARGS\]\s*(\{.*\})`를 한 번만 매칭했다. 호출 두 개가 하나로 합쳐져 두 번째 호출이 인자 안으로 들어갔다. `get-weather`처럼 하이픈이 있는 이름은 `\w+`에 매칭되지 않았다.
- **Mistral Nemo id.** JSON 목록 형식에는 `id`가 들어 있다. 하지만 `ParsedToolCall`에 받을 필드가 없었고, 응답 경로마다 id를 새로 만들었다. 모델 id를 되돌려 보내는 클라이언트는 결과와 호출을 짝지을 수 없었다.
- **Pythonic.** `\[(\w+)\((.*?)\)\]`는 첫 번째 캡처만 남겼다. lazy `.*?`가 처음 나온 `)]`에서 멈추므로 `cfg={"k": [1, 2]}`나 `p=g(1)` 같은 중첩 값이 잘렸다. `split_pythonic_commas`는 `[`/`]` 깊이만 추적했다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `tool_calls/formats.rs` (+456/-56) | `match_qwen_name` 추가, Mistral 헤더 반복 스캔과 `MISTRAL_MAX_CALLS`, `try_mistral_nemo`의 id 검증, 정규식을 대체하는 `scan_pythonic_calls` / `scan_pythonic_list` / `find_pythonic_call_end`, `()[]{}`를 모두 추적하는 `split_pythonic_commas`, `coerce_pythonic_value`의 JSON dict 리터럴 처리 |
| `tool_calls/types.rs` | `ParsedToolCall.id: Option<String>`, `Default` derive |
| `tool_calls/mod.rs`, `routes/chat.rs` | Chat id가 `call.id`를 우선 사용(비스트리밍, 스트리밍) |
| `responses_translator.rs`, `routes/responses.rs` | Responses `call_id`가 `call.id`를 우선 사용, `fc_` item id는 계속 새로 생성 |
| `anthropic_translator.rs`, `atem.rs`, `in_process/chat_tests.rs` | 생성 지점에 `id: None`만 추가 |
| 문서 | `docs/server-features.md`, `docs/responses-api.md`, `CHANGELOG.md` |

커밋: `587a3a99`(구현), `07175912`(끝이 잘린 이름), `76f314c2`(리뷰 수정), `5656c5ea`(보안 리뷰 보강), `eefb6099`(문서).

## 3. 기술적 선택과 그 이유

### 3.1 Qwen3-Coder 이름 매칭은 upstream을 따르되 예외 하나를 둔다

`match_qwen_name`은 upstream `_name_regex = r"\s*([^\s<>]+)>?"`를 옮긴 것이다. 이름은 처음 나오는 공백, `<`, `>`에서 끝나고, 뒤따르는 `>` 하나를 소비한다. 이름을 매칭할 수 없는 파라미터는 호출 전체를 실패시키지 않고 건너뛴다. 따옴표 형식도 유지하되 닫는 `>` 탐색은 다음 `<`에서 멈춘다.

예외는 종결자 없이 입력 끝까지 이어지는 이름이다. 이런 이름은 거부한다. upstream은 출력 끝의 `<function=read_file`을 호출로 받아들이지만 main은 거부했고, 기존 테스트 `qwen3_coder_malformed_missing_closing_tags_keeps_prior_calls`가 이 동작을 고정한다. 생성이 중간에 끊겼다면 인자를 알 수 없는 호출을 만들면 안 된다.

**리뷰 지적(HIGH, `76f314c2`에서 수정).** 파라미터 이름 실패 처리를 `break`에서 "건너뛰고 진행"으로 바꾸자 문제가 생겼다. 따옴표 분기가 `<parameter="a` 여는 태그마다 나머지 전체에서 `>`를 다시 찾았다. 0.5 MB 입력에서 294 ms가 걸렸고, 따옴표 없는 경우는 0.3 ms였다. 탐색 범위를 다음 `<`까지로 제한해 선형 시간을 되찾았고, 시간 측정 테스트로 고정했다.

### 3.2 Mistral: greedy 정규식 하나 대신 헤더마다 JSON 값 하나

| 방안 | 장점 | 단점 |
|---|---|---|
| greedy `\{.*\}` 정규식(기존) | 단순함 | 뒤 호출을 삼킴, 병렬 호출을 나눌 수 없음 |
| 중괄호 깊이 스캐너 | 의존성 없음 | JSON 문자열 escape 처리를 다시 구현해야 함 |
| **헤더 정규식 + `serde_json::Deserializer::into_iter::<Value>()`** | Python `raw_decode`와 같음, JSON 경계가 정확함 | 호출마다 헤더 정규식 한 번 |

`([\w-]+)\s*\[ARGS\]\s*` 헤더마다 JSON 값을 정확히 하나 디코딩하고, 그 값이 끝난 위치부터 스캔을 이어간다. 디코딩에 실패하면 파싱을 끝낸다. 그때까지 모은 호출이 있으면 반환하고, 없으면 `None`을 반환한다. 보안 리뷰에서 이 스캔이 선형임을 확인했다(호출 10만 개에 17 ms). 다른 파서와 맞추기 위해 `MISTRAL_MAX_CALLS = 1024`를 추가했다.

### 3.3 모델 id는 그대로 돌려줘도 안전할 때만 전달한다

보안 리뷰에서 빈 id, 중복 id, 임의 문자열 id가 그대로 통과한다는 점을 찾았다. id는 serde JSON이나 SSE로 직렬화되므로 주입은 불가능하다. 서버 코드도 `call_id`를 키로 쓰지 않는다. 위험은 클라이언트 쪽에 있다. id가 중복되면 결과와 호출을 짝짓지 못한다. 그래서 `try_mistral_nemo`는 `^[A-Za-z0-9_-]{1,64}$`에 맞고 결과 안에서 아직 쓰이지 않은 id만 남긴다. 나머지는 `None`으로 두고, 응답 경로가 기존처럼 id를 만든다.

### 3.4 Pythonic: 정규식 대신 스캐너

정규식으로는 짝이 맞는 괄호를 매칭할 수 없어서 작은 스캐너를 썼다. 스캐너는 바깥 `[`를 찾고 `name(`을 읽는다. 따옴표 상태와 `()[]{}` 깊이를 추적하며 짝이 되는 `)`까지 이동해 호출 하나를 만든다. 그다음 `,`와 공백을 건너뛰고, 마지막 호출 바로 뒤에 닫는 `]`가 있어야 한다. 리뷰에서 두 가지를 바로잡았다.

- **닫히지 않은 목록(MEDIUM, `76f314c2`).** `]` 확인이 없을 때는 `see [note(appendix) for details`나 `[sq(x) for x in range(10)]` 같은 일반 문장도 호출로 해석됐다. 앞의 예는 main에서 `None`이었다.
- **재시도 한도(MEDIUM, `5656c5ea`).** 모든 `[`가 후보 재시도 64회 중 하나를 썼다. 그래서 `[1] ` 접두사 65개 뒤의 유효한 호출을 놓쳤는데, main 정규식은 이 호출을 찾았다. 이제 `name(`까지 도달한 후보만 횟수에 넣는다.

본문 전체가 유효한 JSON이면 처리하지 않는 기존 가드는 그대로다. 따라서 문자열 인자에 `[calc(x=1)]`이 들어 있는 JSON 호출은 여전히 JSON 파서가 처리한다.

## 4. 검증

| 항목 | 결과 |
|---|---|
| `MLXCEL_BUILD_METAL=0 cargo test --profile test-fast -p mlxcel tool_calls::` | 453개 통과, 0개 실패 |
| `MLXCEL_BUILD_METAL=0 cargo clippy -p mlxcel --lib --tests -- -D warnings` | 경고 없음 |
| `cargo fmt --check` | 통과 |
| Fuzzing(보안 리뷰, 임시 모듈) | 멀티바이트 UTF-8을 포함한 무작위 입력 약 30만 개, panic 없음 |
| 시간 측정 | 시도한 모든 악성 입력 형태에서 선형(약 300 KB pythonic 최악 사례도 40 ms 이하) |

이 호스트에는 Metal 툴체인이 없어서 테스트는 macOS의 CPU 전용 MLX 빌드로 실행했다. 파서는 MLX를 사용하지 않으므로 백엔드 종류는 이 결과에 영향을 주지 않는다.

## 5. 미검증 항목과 범위 밖 항목

- **실제 체크포인트 smoke 미실행:** Qwen3-Coder-30B-A3B-Instruct-4bit와 Devstral-Small-2507-4bit에 도구 두 개를 주고 `/v1/chat/completions`를 스트리밍과 비스트리밍으로 호출하는 검사.
- **Anthropic 경로:** `anthropic_translator.rs`와 `routes/anthropic.rs`는 여전히 `toolu_` id를 새로 만들고 `call.id`를 쓰지 않는다.
- **Python 전용 리터럴:** `{'k': True}`와 `None`은 원시 문자열로 남는다. JSON 문법의 dict만 객체로 바뀐다.
- **워크스페이스 전체 검증:** 로컬에서는 워크스페이스 전체 테스트를 실행하지 않았다. CI는 GB10에서 clippy를 실행한다.

## 6. 학습 포인트

- `serde_json::Deserializer::from_str(..).into_iter::<Value>()`와 `byte_offset()` 조합은 Python `json.JSONDecoder.raw_decode`에 해당한다. 긴 텍스트 안에 JSON이 섞여 있을 때 쓴다.
- 파서를 "첫 불량 항목에서 멈춤"에서 "건너뛰고 계속"으로 바꾸면 제한된 스캔이 2차 시간 스캔이 될 수 있다. 이런 변경에서는 안쪽 탐색마다 상한이 있는지 확인한다.
- 재시도 한도는 성능뿐 아니라 정확성에도 영향을 주는 값이다. 무엇을 시도 한 번으로 세는지에 따라 찾을 수 있는 유효 입력이 달라진다.
