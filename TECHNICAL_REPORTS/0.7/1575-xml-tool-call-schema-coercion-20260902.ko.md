# 기술 보고서: PR #1575 - fix(server): type XML tool-call arguments by the request schema

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 지정 전
**상태**: 완료
**언어**: Rust
**위험도**: Medium

---

## 요약

Qwen3-Coder와 MiniMax M2 tool-call 파서는 요청의 `tools`를 전달받지 못해 인자의 JSON 타입을 원문 텍스트만 보고 추측했다. 그 결과 `string`으로 선언된 파라미터에 담긴 `02134`가 클라이언트에는 숫자 `2134`로, `integer`로 선언된 파라미터에 모델이 쓴 `5.0`은 실수로 전달됐다. 이 PR은 두 파서에 요청 스키마를 연결하고, MiniMax M3와 GLM-4.7, LongCat이 이미 공유하는 coercion 헬퍼로 값 변환을 일원화했으며, 남은 관대한 규칙(불리언, null, 소수부가 0인 정수)을 텍스트 모양이 아니라 선언된 타입에 근거하도록 정리했다.

---

## 1. 문제 정의

### 1.1 배경

mlxcel이 지원하는 XML tool-call 문법 중 두 가지는 마크업 자체에 타입 정보가 없다. Qwen3-Coder는 `<function=NAME><parameter=KEY>VALUE</parameter></function>`, MiniMax M2는 `<invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke>` 형태이며, 두 문법 모두 값은 그냥 텍스트다. 의도된 타입이 존재하는 유일한 자리는 클라이언트가 요청에 실어 보낸 `tools` 배열이고, 이 배열은 OpenAI 호환 chat 라우트가 파싱 시점에 이미 손에 쥐고 있다.

같은 파일의 파서 세 개는 이 문제를 이미 해결한 상태였다. `try_minimax_m3`, `try_glm47`, `try_longcat`은 `tools`를 받아 `minimax_m3_function_schema`로 호출된 함수의 스키마를 찾고 선언된 타입에 맞춰 값을 변환한다. XML 파서 두 개만 그러지 못했는데, 이유는 이들이 `&[fn(&str) -> Option<ToolCallParseResult>]` 타입의 디스패치 테이블 안에 있었기 때문이다. 필요한 데이터가 그 원소 타입을 통과할 수 없으니 파서는 추측을 택할 수밖에 없었다.

### 1.2 기존 문제점

- **string 값이 변조됨**: `coerce_minimax_param`은 `i64` 파싱, `f64` 파싱, 불리언 단어 목록을 차례로 시도한 뒤에야 문자열로 떨어뜨렸다. 미국 우편번호 `02134`는 `i64`로 파싱되어 2134가 되면서 앞의 0이 사라진다. 제품 코드 `1e5`는 `100000.0`이 되고, 텍스트 `true`는 JSON 불리언이 된다. 문자열을 기대한 도구 핸들러는 숫자를 받아 자체 검증에서 실패하거나, 조용히 잘못된 값으로 동작한다.
- **integer 값이 실수 형태로 남음**: 모델이 `integer` 파라미터에 `5.0`을 쓰면 JSON 실수 `5.0`이 그대로 나갔다. 엄격한 도구 구현은 이를 거부하며, 스키마는 이 값이 정수라고 분명히 말하고 있었다.
- **타입 선언이 없는 자리에까지 적용된 추측**: 요청이 무엇을 선언했든 상관없이 `yes`, `on`, `no`, `off`는 불리언으로, `none`, `nil`은 null로 바뀌었다. 선언이 없다면 이는 데이터에 대한 추론이 아니라 영어 단어에 대한 추측이다.
- **coercion 로직 중복**: `coerce_minimax_param`은 `minimax_m3_coerce_leaf`와 같은 일을 더 약하게 구현한 두 번째 사본이었다. 스키마 처리가 개선되어도 한쪽에만 반영됐다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 스키마가 string으로 선언한 자리에 숫자가 도착해 도구 호출이 실패하거나 손상된 값으로 실행됨 | High | 앞자리 0, 지수 표기, 불리언 단어 형태의 문자열 인자에서는 확정적 |
| `integer` 인자를 `5.0`으로 받은 도구가 거부함 | Medium | 간헐적 (모델의 표기 방식에 좌우됨) |
| coercion이 발전할수록 XML 문법 두 개가 스키마 인식 문법 세 개와 더 벌어짐 | Medium | 로직이 중복된 동안에는 시간이 지나면 확정적 |
| 값 파싱 실패가 tool call 자체를 통째로 버리게 됨 | High | 설계로 회피: 모든 경로의 마지막은 원문 문자열 |
| 호출 이름에 네임스페이스를 붙이는 모델에서는 수정이 아무 일도 하지 않음 | High | 그런 모델에서는 확정적 (이번에 조회 정규화로 해결) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

**검토 항목:**
- [x] 입력 검증: 파싱 대상은 전부 신뢰할 수 없는 모델 출력이다. 이번 변경은 인덱싱이나 슬라이싱을 추가하지 않고 값 타이핑만 바꾼다
- [x] 서비스 거부: 기존 `MINIMAX_M2_MAX_CALLS` / `MINIMAX_M2_MAX_PARAMS_PER_CALL` / `QWEN3_CODER_*` 상한과 잘못된 태그에 대한 O(N^2) 방지용 `break` 가드는 그대로다
- [x] panic 없음: 테스트 모듈 밖에 `unwrap`, `expect`, 인덱싱을 추가하지 않았다
- [x] 민감정보 로깅 없음

**발견된 이슈:**

| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| 2^53 부근에서 소수값이 반올림되어 정수가 됨 (`9007199254740991.5`가 정수로 변환) | Medium | 두 번째 커밋에서 수정. 규칙을 텍스트 기반으로 바꿔 `f64` 왕복을 제거 |
| 2^53 부근에서 정수가 1 어긋남 (`9007199254740993.0`이 `...992`로) | Medium | 같은 변경에서 수정 |

두 건 모두 이 PR의 첫 커밋이 만든 결함이며 머지 전 리뷰에서 발견했다. 부수적 개선도 둘 있다. `coerce_minimax_param`은 null 표기 세 가지를 비교하려고 모든 파라미터 값의 소문자 `String` 사본을 할당했지만 `coerce_xml_param`은 `eq_ignore_ascii_case`를 써서 할당이 없다. 또한 `serde_json`은 무한대 파싱을 거부하므로, 직렬화 시 `null`이 되어버릴 비유한 실수를 담은 `Value::Number`가 만들어지는 경로는 존재하지 않는다.

### 2.2 성능 관점

**검토 항목:**
- [x] 알고리즘 복잡도: 파라미터당 비용은 그대로이고, 호출당 `tools` 슬라이스 선형 탐색 1회와 파라미터당 property 조회 1회가 추가된다. MiniMax M3와 GLM-4.7, LongCat이 이미 하고 있는 것과 같다
- [x] 메모리 사용: 파라미터 값마다 `String` 할당 1회가 줄어든다

tool-call 파싱은 completion당 한 번, 모델 응답 하나 크기의 텍스트 버퍼에 대해 실행되므로 어느 쪽도 hot path가 아니다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: API 수준에서는 없다. 기존 추측에 의존하던 클라이언트에게는 동작이 바뀌며, 그 목록은 4.1절에 있다. 공개 함수 `try_qwen3_coder`와 `try_minimax_m2`의 시그니처가 바뀌지만 `tool_calls::parser`와 같은 파일의 테스트 외에는 호출자가 없다.
- **새로운 의존성**: 없음.
- **적용 범위**: 비스트리밍 라우트(`routes/chat.rs`의 `parse_tool_calls(&result.text, tools)`)와 스트림 종료 라우트(`parse_tool_calls(&cb.accumulated, tools_ref)`)가 이미 요청 tools를 넘기고 있어, 한 번의 변경으로 스트리밍과 비스트리밍이 함께 적용된다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: 단위 테스트 26개 추가. `server::tool_calls` 스위트가 350개에서 376개가 되었고 전부 통과한다.
- **코드 복잡도**: 대체로 비슷하다. 37줄짜리 수제 coercion 사다리가 스키마 기반 `coerce_xml_param`과 작은 헬퍼 두 개로 바뀌었고, 디스패처에 변형 2개짜리 enum이 생겼다.
- **기술 부채**: 감소. 중복 coercion 구현 둘 중 하나가 사라졌고, `parse_tool_calls`의 오래된 주석 두 곳을 바로잡았으며, 호출자 범위가 넓어진 공유 헬퍼 다섯 개에 프로젝트 규약이 요구하는 `// Used by:` 주석을 달았다.

---

## 3. 기술적 선택과 그 이유

### 3.1 모든 파서의 시그니처를 넓히는 대신 `FormatParser` enum 도입

**컨텍스트:**

디스패치 테이블은 순서가 곧 의미인 목록이다. `try_functionary_v31`은 `try_qwen3_coder`보다 먼저 시도되어야 하고(둘 다 `<function=`으로 열리며 v3.1은 JSON이 아닌 본문을 거절한다), `try_qwen3_coder`는 `try_functionary_v32`보다 먼저여야 한다. 15개 항목 중 `tools`가 필요한 것은 2개뿐인데, 배열 리터럴은 원소 타입을 하나만 허용한다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| 파서 15개 전부에 `tools` 파라미터 추가 | 테이블 타입이 하나로 통일되고 래퍼가 없다 | 13개 파서가 쓰지도 않는 인자를 갖게 되고, 파일 안의 모든 호출부와 테스트가 바뀌며, 어느 파서가 실제로 스키마를 보는지 알 수 없게 된다 |
| 두 파서를 테이블 밖 `try_minimax_m3` 옆으로 이동 | 테이블을 건드리지 않는다 | Functionary v3.1, v3.2 대비 디스패치 순서가 조용히 바뀐다. 테이블이 존재하는 이유가 바로 그 순서다 |
| **선택: `Plain`과 `WithTools` 변형을 가진 `FormatParser` enum** | 순서가 한 줄 단위로 그대로 보존되고, 스키마 인식 항목이 한눈에 보이며, 쓰지 않는 인자를 가진 파서가 생기지 않는다 | 작은 enum과 `run` 메서드가 늘고, 각 항목이 래퍼 이름을 달고 있다 |

**선택 이유:**

테이블의 의미가 순서인 이상, 그 순서를 문자 그대로 보존하는 선택이 안전하다. enum은 어느 문법이 스키마를 인식하는지도 함께 기록하는데, 이는 다음 포맷을 추가하는 사람에게 실제로 필요한 정보다. 두 번째 대안이 깨뜨렸을 순서 속성은 `parse_qwen3_coder_still_runs_after_functionary_v31` 테스트로 고정했다.

**트레이드오프:**

각 항목에 `Plain(...)` 또는 `WithTools(...)` 래퍼가 붙어 함수 이름만 있을 때보다 약간 시끄럽다. 디스패치 비용은 시도하는 포맷마다 `match` 한 번이며, 각 파서가 수행하는 문자열 스캔에 비하면 무시할 수준이다.

### 3.2 관대한 규칙은 유지하되 선언된 타입이 뒷받침할 때만

**컨텍스트:**

첫 커밋은 기존 추측기의 관대한 규칙(`yes`/`on` → true, `no`/`off` → false, `none`/`nil` → null)을 전부 없애고 `null`은 스키마와 무관하게 null이 되게 했다. 리뷰에서 양쪽 다 경계에서 틀렸음이 드러났다. `{"type": "boolean"}` 아래에서 모델이 쓴 `yes`가 문자열 `"yes"`가 되는데, 이는 대체한 추측보다 나쁘다. 하필 불리언으로 읽을 근거가 가장 확실한 자리이기 때문이다. 반대로 `{"type": "string"}` 아래의 텍스트 `null`은 여전히 JSON null이 되는데, 이는 이 PR이 없애려던 바로 그 타입 위반이다.

**선택 이유:**

중요한 구분은 "관대한가 엄격한가"가 아니라 그 관대함에 선언된 타입이라는 근거가 있는가다. `yes`를 `true`로 읽는 것은 아무 타입도 선언되지 않았다면 영어 단어에 대한 추측이지만, 스키마가 `boolean`이라고 말한다면 타당한 해석이다. 그래서 `coerce_xml_param`은 불리언 단어 목록을 `boolean` 선언이 있을 때만 적용하고, `null`은 평범한 `string` 선언을 제외한 모든 곳에서 JSON null이 된다. string 선언 아래에서는 텍스트가 곧 도구가 요구한 값이기 때문이다. `{"type": ["string", "null"]}` 같은 nullable 선언은 여전히 null을 받는다. 스키마가 둘 다 가능하다고 말하고 있고, 그 단어가 이 문법이 표기할 수 있는 유일한 null이기 때문이다.

**트레이드오프:**

두 동작이 균일한 규칙이 아니라 선언에 따라 갈리므로 머릿속에 담아야 할 것이 늘었다. `coerce_xml_param`의 문서 주석에 둘 다 적었고, 테스트 6개가 선언이 있을 때와 없을 때 양방향으로 고정한다. string이 아닌 파라미터에 문자열 `"null"`을 넣고 싶은 모델은 여전히 표현할 방법이 없는데, 이는 이 PR 이전에도 마찬가지였고 새 이스케이프 문법을 만들 만한 사안은 아니다.

### 3.3 integer 규칙은 `f64`가 아니라 텍스트로 소수부를 떼어낸다

**컨텍스트:**

공유 `M3Type::Integer` 갈래는 `i64`와 `u64` 리터럴만 받았기 때문에 `integer` 타입의 `5.0`은 느슨한 fallback으로 떨어져 실수로 남았다. 첫 커밋은 이를 `f64`로 파싱한 뒤 2^53 상한 아래에서 `fract() == 0.0`을 확인하는 방식으로 고쳤다.

**선택 이유:**

그 검사는 성립할 수 없다. 반올림이 `parse::<f64>()` 안에서, 즉 무엇도 결과를 들여다보기 전에 일어나기 때문이다. `9007199254740991.5`는 `...992.0`으로 파싱되어 소수부 0 검사를 통과하고 정수가 된다. `9007199254740993.0`은 1 어긋난 값으로 돌아오면서 상한도 통과한다. 텍스트 규칙(`.`에서 자르고, 소수부가 전부 0인지 확인하고, 정수부를 정확한 정수 파서로 파싱)은 모든 크기에서 정확하며 상한 자체가 필요 없다. 또한 지수 표기를 거절하는데, `minimax_m3_typed_coerce`가 `anyOf` 대안 중 처음 성공한 것을 택하기 때문에 이 점이 중요하다. `f64` 경로에서는 `anyOf: [integer, string]`에 값 `1e5`가 오면 string 대안이 가져가던 것을 integer가 `100000`으로 채갔다.

**트레이드오프:**

`integer`로 선언된 `1e5`는 더 이상 `100000`으로 정규화되지 않고 fallback을 거쳐 실수 `100000.0`으로 남는다. 이는 string 대안의 식별자를 빼앗지 않기 위한 대가이며, 모델의 도구 인자에서 지수 표기는 소수점보다 훨씬 드물다.

### 3.4 `integer`는 `5.0`을 정규화하고 `number`는 표기를 보존

**컨텍스트:**

integer 갈래를 확장하면서 옆의 `number`가 정수 리터럴을 어떻게 다뤄야 하는지가 함께 문제가 됐다.

**선택 이유:**

`integer` 아래에서는 선언된 타입이 답을 정한다. 값은 정수이므로 `5.0`은 `5`이고 소수점은 서식일 뿐이다. `number` 아래에서는 두 표기가 모두 유효하고 스키마가 선호를 표현하지 않으므로, 파서는 모델이 쓴 그대로를 유지한다(`5`는 `5`, `5.0`은 `5.0`). 이렇게 두면 `number` 타입 정수값에 대한 XML 파서의 기존 출력도 그대로 유지되고, 기존 `f64` 전용 갈래에 있던 2^53 이상 정수의 정밀도 손실도 사라진다.

**트레이드오프:**

`M3Type::Number`는 공유 코드이므로, MiniMax M3가 `number`/`num`/`float`/`double` 파라미터를 정수로 받았을 때 내보내는 값의 타입이 `5.0`에서 `5`로 바뀐다. JSON Schema는 정수를 `number`로 받아들이므로 스키마 위반은 아니지만, 이슈가 요구한 범위 밖의 눈에 보이는 변화라서 PR 본문에 명시했다.

### 3.5 GLM-4.7과 LongCat에는 typed 경로 전체가 아니라 integer 규칙만 적용

**컨텍스트:**

`coerce_kv_value`(GLM-4.7, LongCat)는 `string` 스키마만 존중하고 나머지 타입은 전부 느슨한 fallback으로 보냈다. 이슈는 `integer` 타입 `5.0`이 모든 XML 문법에서 `5`가 되기를 요구했다.

**선택 이유:**

`coerce_kv_value`에 `Integer` 갈래를 추가하면 요구된 수정이 정확히 그만큼 이뤄진다. 대신 이 함수를 `minimax_m3_typed_coerce`로 통째로 태웠다면 `enum`, `anyOf`, `object`, `array`, `boolean` 스키마 처리까지 부수적으로 바뀐다. 이슈가 정당화하는 범위보다 훨씬 넓고, 두 문법의 기존 테스트가 덮지 않는 영역이다. 규칙 자체는 `parse_integer_literal` 하나를 공유하므로 호출부 사이에서 어긋날 수 없다.

**트레이드오프:**

GLM-4.7과 LongCat은 여전히 MiniMax M3보다 스키마에 덜 엄격하다. 그 간극은 이제 `coerce_kv_value` 문서 주석에 명시되어 있고, 좁히는 작업은 별도의 검증 가능한 변경으로 분리된다.

### 3.6 스키마 조회가 네임스페이스 붙은 호출 이름을 정규화한다

**컨텍스트:**

`minimax_m3_function_schema`는 `t.function.name == name` 완전 일치로만 찾았다. 그런데 `filter_by_tools`가 존재하는 이유 자체가 모델이 `functions.get_weather` 같은 이름을 내보내기 때문이며, 이 접두사는 파싱이 끝난 뒤에야 제거된다.

**선택 이유:**

네임스페이스를 쓰는 모델에서는 순서가 이랬다. 스키마 조회 실패 → 모든 값이 느슨한 규칙을 탐 → `filter_by_tools`가 접두사를 떼고 호출을 승인. 클라이언트는 이 PR이 없애려던 바로 그 버그를 품은 성공한 tool call을 받으며, 스키마가 무시됐다는 표시는 어디에도 없다. 이제 조회는 첫 `.` 뒤의 이름으로 한 번 더 시도하며, 이는 `filter_by_tools`와 같은 정규화다. 완전 일치를 먼저 시도하므로 이름이 실제로 `a.b`인 도구는 영향을 받지 않는다.

**트레이드오프:**

네임스페이스가 붙은 이름에서는 선형 탐색이 한 번 더 돈다. 현실적인 도구 개수에서는 측정되지 않으며, 문제가 된다면 호출 루프 밖으로 `HashMap`을 끌어올리는 것이 답이다.

---

## 4. 구현 상세

### 4.1 동작 변화표

| 스키마 | 원문 값 | 변경 전 | 변경 후 |
|-------|--------|--------|--------|
| `{"type":"string"}` | `02134` | `2134` | `"02134"` |
| `{"type":"string"}` | `true` | `true` | `"true"` |
| `{"type":"string"}` | `1e5` | `100000.0` | `"1e5"` |
| `{"type":"string"}` | `null` | `null` | `"null"` |
| `{"type":["string","null"]}` | `null` | `null` | `null` |
| `{"type":"boolean"}` | `yes` / `on` / `TRUE` | `true` | `true` |
| `{"type":"boolean"}` | `maybe` | `"maybe"` | `"maybe"` |
| `{"type":"integer"}` | `5.0` | `5.0` | `5` |
| `{"type":"integer"}` | `5.5` | `5.5` | `5.5` (느슨한 fallback, 호출 유지) |
| `{"type":"number"}` | `5` | `5` | `5` |
| `{"type":"object"}` | `{not json` | `"{not json"` | `"{not json"` |
| `{"anyOf":[integer,string]}` | `1e5` | `"1e5"` | `"1e5"` |
| 없음 | `5` / `true` / `null` | `5` / `true` / `null` | 동일 |
| 없음 | `yes` / `on` / `none` | `true` / `true` / `null` | `"yes"` / `"on"` / `"none"` |
| 없음 | `TRUE` | `true` | `"TRUE"` |
| 네임스페이스 호출 `functions.f`, `{"type":"string"}` | `02134` | `2134` | `"02134"` |

공유 fallback을 재사용하면서 스키마가 없을 때의 차이가 둘 더 생겼다. 텍스트가 `{` 또는 `[`로 시작할 때만 JSON 파싱을 시도하던 것을 이제 항상 시도하므로 JSON 리터럴로 쓰인 값(`[1, 2]`, 따옴표로 감싼 문자열)은 그 리터럴로 파싱된다. 그리고 불리언 단어 비교가 대소문자를 구분하므로 스키마 없는 `TRUE`는 문자열이다. 둘 다 MiniMax M3와 GLM-4.7, LongCat이 이미 하던 동작이며, `boolean` 선언이 있으면 관대한 해석이 되살아난다.

### 4.2 주요 코드 변경

**파일: `src/server/tool_calls/formats.rs`**

```rust
// 변경 전
fn coerce_minimax_param(value: &str) -> serde_json::Value {
    let lower = value.to_lowercase();
    if lower == "null" || lower == "none" || lower == "nil" { return Value::Null; }
    if let Ok(i) = value.parse::<i64>() { return Value::Number(i.into()); }
    // ... f64, 그 다음 "true"/"1"/"yes"/"on", 그 다음 "{"/"[" JSON, 마지막에 String
}

// 변경 후
fn coerce_xml_param(raw: &str, schema: Option<&serde_json::Value>) -> serde_json::Value {
    let declared = schema.and_then(minimax_m3_schema_type);
    let keeps_raw_text =
        declared == Some(M3Type::Str) && !schema.is_some_and(schema_type_admits_null);
    if !keeps_raw_text && raw.eq_ignore_ascii_case("null") {
        return serde_json::Value::Null;
    }
    if declared == Some(M3Type::Boolean)
        && let Some(b) = loose_boolean_word(raw)
    {
        return serde_json::Value::Bool(b);
    }
    minimax_m3_coerce_leaf(raw, schema)
}
```

**변경 이유:** 타입은 텍스트의 생김새가 아니라 스키마에 속한다. `minimax_m3_coerce_leaf`는 `enum`과 `anyOf`/`oneOf`, 배열 item 해석까지 포함한 선언 타입 경로를 이미 구현하고 있고, 마지막이 원문 문자열이므로 파싱 실패가 호출을 버리는 일도 없다. 그 앞에 놓인 두 규칙은 선언이 정당화하는 것들이다.

```rust
// 변경 후: 공유 integer 규칙
fn parse_integer_literal(raw: &str) -> Option<serde_json::Value> {
    if let Some(v) = parse_exact_integer_literal(raw) {
        return Some(v);
    }
    let (int_part, fraction) = raw.split_once('.')?;
    if fraction.bytes().any(|b| b != b'0') {
        return None;
    }
    parse_exact_integer_literal(int_part)
}
```

**변경 이유:** `integer`는 모델이 소수점을 붙여 쓴 정수값을 받아들여야 하고, 그 변환은 정확해야 한다. `f64` 대신 텍스트로 처리하면 반올림 결함 두 가지가 사라지고 지수 표기가 `anyOf`의 integer 대안에 끼어들지 않는다.

**파일: `src/server/tool_calls/parser.rs`**

```rust
// 변경 전
let parsers: &[fn(&str) -> Option<ToolCallParseResult>] = &[ /* 15개 항목 */ ];
for parser in parsers {
    if let Some(mut result) = parser(text) { /* ... */ }
}

// 변경 후
enum FormatParser {
    Plain(fn(&str) -> Option<ToolCallParseResult>),
    WithTools(fn(&str, Option<&[Tool]>) -> Option<ToolCallParseResult>),
}

let parsers: &[FormatParser] = &[ /* 같은 15개 항목, 같은 순서 */ ];
for parser in parsers {
    if let Some(mut result) = parser.run(text, tools) { /* ... */ }
}
```

**변경 이유:** 테이블이 `tools`를 필요로 하는 파서를 담을 수 없었다. 순서는 그대로이며, 그 순서가 테이블이 존재하는 이유다.

---

## 5. 학습 포인트

### 5.1 컬렉션의 원소 타입이 버그의 원인일 수 있다

**개념:**

`try_qwen3_coder`와 `try_minimax_m2`가 타입을 추측한 이유는 추측이 옳다고 판단해서가 아니다. 원소 타입이 `fn(&str) -> Option<...>`인 배열 안에 있었고, 필요한 데이터가 그 타입을 통과할 수 없었기 때문이다. 배열 밖으로 나간 파서 셋은 스키마를 받았고, 남은 둘은 받지 못했다.

**이 PR에서의 적용:**

수정의 대부분은 컨테이너 변경이다. 테이블이 `WithTools` 항목을 담을 수 있게 되자 파서 본문에 필요한 것은 네 줄, 즉 함수를 찾고 파라미터별 스키마를 두 단계 아래로 내려보내는 일뿐이었다.

**같은 패턴이 나타나는 곳:**

- 처음 몇 멤버가 필요로 한 가장 좁은 시그니처에 맞춰진 핸들러 레지스트리
- 컨텍스트 인자를 빠뜨린 트레이트 메서드. 구현체는 전역 상태나 휴리스틱에 손을 뻗는다
- 첫 사용 사례에서 원소 타입이 굳어버린 콜백 목록. 이후 멤버는 우회로를 만든다

신호는 넘겨받았어야 할 정보를 스스로 다시 만들어내는 파서나 핸들러다.

### 5.2 손실이 일어난 뒤에 놓인 검사는 아무것도 검사하지 않는다

**개념:**

첫 커밋은 소수부가 0인 정수를 `raw.parse::<f64>()`로 읽은 뒤 `f.fract() == 0.0 && f.abs() <= 2^53`으로 걸렀다. 두 조건 모두 파싱이 이미 반올림한 값 위에서 평가되므로, `9007199254740991.5`는 걸러져야 할 소수부 검사를 통과하고 `9007199254740993.0`은 1 어긋난 채로 상한을 통과한다. 보호 장치처럼 읽히지만 실제로는 아니다.

**이 PR에서의 적용:**

규칙을 텍스트 기반으로 바꿨다. `.`에서 문자열을 자르고, 소수부 바이트가 전부 `0`인지 확인하고, 정수부를 정확한 정수 파서로 읽는다. 반올림이 없으니 뒤에서 검사할 것도 없다. `integer_literal_rule_is_textual_and_exact`가 이전 형태가 틀렸던 두 값을 모두 고정한다.

**같은 패턴이 나타나는 곳:**

- 축소 캐스팅 이전이 아니라 이후에 놓인 범위 검사
- 이미 float 왕복을 거친 값에 대한 정밀도 단언
- 정규화가 파괴하는 속성을 정규화 이후에 검증하는 문자열 검사

### 5.3 표기 보존도 올바른 coercion의 일부다

**개념:**

coercion은 보통 "값을 타입에 맞춘다"로 이해되지만, 하나의 타입이 여러 표현을 허용한다면 파서가 그중 하나를 골라서는 안 된다. `number` 아래에서는 `5`와 `5.0`이 모두 유효하므로 모델이 쓴 것을 유지하고, `integer` 아래에서는 유효한 표현이 하나뿐이므로 `5.0`이 `5`가 된다.

**이 PR에서의 적용:**

`M3Type::Number`는 `f64` 파싱 전에 `parse_exact_integer_literal`을 먼저 시도하고, `M3Type::Integer`는 더 넓은 `parse_integer_literal`을 쓴다. `qwen3_coder_number_typed_keeps_written_form`과 `minimax_m3_number_typed_keeps_written_form` 테스트는 `is_i64()`와 `is_f64()`로 양쪽을 모두 단언한다. `serde_json`에서는 `assert_eq!(value, 5)`만으로 정수와 실수를 구분할 수 없기 때문이다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|-------|-----|-------|
| `minimax_m3_coerce_leaf` | 스키마 기반 leaf coercion: 타입 파싱, JSON 파싱, 느슨한 리터럴, 원문 문자열 순서 | 이제 네 XML 문법이 모두 도달하는 단일 coercion 경로 |
| `minimax_m3_typed_coerce` | 선언 타입이 맞지 않으면 `None`을 돌려주는 엄격한 스키마 파싱 | 그 `None` 덕분에 `anyOf`가 다음 대안을 시도하고, 잘못된 값에도 호출이 살아남는다 |
| `kv_param_schema` | 함수 스키마의 `properties`에서 파라미터를 찾는다 | `minimax_m3_function_schema`와 짝을 이루는 파라미터 단위 조회 |
| `filter_by_tools` | 등록되지 않은 함수 호출을 버리고 네임스페이스 접두사를 제거한다 | 스키마 조회가 같은 접두사를 정규화해야 하는 이유 |
| 2^53 | `f64`가 모든 정수를 표현하는 최대 크기 | 첫 커밋이 의지했던 상한이자, 최종 규칙이 `f64` 자체를 피한 이유 |

### 관련 PR/이슈

- Issue #1336: 이 PR이 구현한 명세. 4.1절의 스키마 표가 여기서 왔다.
- GLM-4.7 / LongCat key-value 파서와 MiniMax M3 네임스페이스 XML 파서는 이 PR이 재구현하지 않고 재사용한 선행 구현이다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 소스 파일 수 | 2 |
| 추가된 라인 (소스, 이 보고서 제외) | +786 |
| 삭제된 라인 (소스) | -115 |
| 테스트 추가 | 26 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Correctness | 파일 2개 | Qwen3-Coder / MiniMax M2 문법의 스키마 기반 타이핑, 정확한 integer 규칙, 네임스페이스를 정규화하는 스키마 조회 |
| Code Quality | 1 | `coerce_minimax_param` 제거, `parse_tool_calls`의 오래된 주석 2곳 수정, 범위가 넓어진 헬퍼 5개에 `// Used by:` 추가 |
| Tests | 26 | 두 파일에 스키마, 무스키마, 경계값, `anyOf`, 네임스페이스, 디스패처 순서 커버리지 추가 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `272afe8` | fix | fix(server): type XML tool-call arguments by the request schema |
| `f222376` | fix | fix(server): tighten schema-driven XML tool-call coercion |

---

## 8. 후속 조치

### 완료 필요

- [ ] 실제 Qwen3-Coder 체크포인트로 end-to-end 동작 확인 (요청과 기대 `tool_calls[].function.arguments`는 PR 본문에 있다). 이 PR은 단위 테스트로만 검증했고 체크포인트를 올리지 않았다.

### 모니터링 필요

- 두 문법에서 클라이언트로 나가는 tool-call 인자, 특히 숫자처럼 보이는 string 타입 값. 기존 coercion에 맞춰 우편번호를 다시 문자열로 되돌리는 식으로 보정하던 클라이언트는 이제 올바른 문자열을 받으므로 보정을 걷어내야 한다.

### 향후 개선 사항

- GLM-4.7과 LongCat을 string, integer 규칙만이 아니라 `minimax_m3_typed_coerce` 전체 경로로 태울지 결정. 이번에는 의도적으로 범위를 끊었고 `coerce_kv_value` 주석에 남겼다.
- `minimax_m3_typed_coerce`는 `anyOf`/`oneOf` 루프에서 키가 존재하면 곧바로 반환하므로, 아무 대안도 맞지 않는 `anyOf`를 가진 스키마는 `oneOf`나 자신의 `type`을 끝내 시도하지 않는다. 기존 동작이며 이번에 건드리지 않았지만 별도 수정 대상이다.
- 공유 헬퍼의 `minimax_m3_` 접두사는 이제 오해를 부른다. Qwen3-Coder와 MiniMax M2도 호출하기 때문이다. `schema_coerce_leaf` 식의 개명은 별도 `refactor:` PR로 다룬다.
- array 타입 파라미터가 스칼라 하나를 담고 있으면 1원소 배열이 아니라 그 스칼라가 된다. 이 문법들이 반복 요소를 표기할 수 없기 때문이다. `xml_array_typed_bare_scalar_takes_the_item_type`가 이를 문서화하며, 배열로 감쌀지는 제품 결정 사항이다.
- 델타 단위 증분 tool-call 인자 스트리밍은 여전히 범위 밖이다. 스트림 경로는 스트림 종료 시점에 한 번 파싱한다.
- 스키마 검증 오류(`required` 키 누락, `enum` 불일치)는 아직 클라이언트에 전달되지 않는다. 파서는 변환만 하고 거부하지 않는다.

---

## 부록

### A. 테스트 결과

```
cargo test --profile test-fast --features metal,accelerate --lib server::tool_calls
test result: ok. 376 passed; 0 failed; 0 ignored; 7313 filtered out

cargo test --profile test-fast --features metal,accelerate --lib server::muse_atem
test result: ok. 11 passed; 0 failed; 0 ignored; 7678 filtered out

cargo test --profile test-fast --features metal,accelerate --lib server::routes::chat
test result: ok. 48 passed; 0 failed; 0 ignored; 7641 filtered out

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
Finished (no warnings)

cargo fmt --all -- --check
clean
```

이슈가 유지를 요구한 네 케이스는 수정 없이 통과한다: `qwen3_coder_single_call_multiple_params_with_type_coercion`, `minimax_m2_numeric_params`, `minimax_m2_boolean_param`, `minimax_m2_null_param`.

### C. 참고 자료

- `docs/code-guidelines.md`: `coerce_xml_param`, `parse_integer_literal`과 호출자 범위가 넓어진 헬퍼 다섯 개에 적용한 `// Used by:` 규약.
- JSON Schema type 키워드: `integer`는 `number`와 구분되는 별도 타입이며, 두 갈래가 다르게 동작하는 근거다.
