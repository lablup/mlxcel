# 기술 보고서: PR #1575 - fix(server): type XML tool-call arguments by the request schema

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 지정 전
**상태**: 완료
**언어**: Rust
**위험도**: Medium

---

## 요약

Qwen3-Coder와 MiniMax M2 tool-call 파서는 요청의 `tools`를 전달받지 못해 인자의 JSON 타입을 원문 텍스트만 보고 추측했다. 그 결과 `string`으로 선언된 파라미터에 담긴 `02134`가 클라이언트에는 숫자 `2134`로, `integer`로 선언된 파라미터에 모델이 쓴 `5.0`은 실수로 전달됐다. 이 PR은 두 파서에 요청 스키마를 연결하고, MiniMax M3와 GLM-4.7, LongCat이 이미 공유하는 coercion 헬퍼로 값 변환을 일원화했으며, 공유 integer 규칙이 소수부 없는 실수 리터럴도 받도록 확장했다.

---

## 1. 문제 정의

### 1.1 배경

mlxcel이 지원하는 XML tool-call 문법 중 두 가지는 마크업 자체에 타입 정보가 없다. Qwen3-Coder는 `<function=NAME><parameter=KEY>VALUE</parameter></function>`, MiniMax M2는 `<invoke name="NAME"><parameter name="KEY">VALUE</parameter></invoke>` 형태이며, 두 문법 모두 값은 그냥 텍스트다. 의도된 타입이 존재하는 유일한 자리는 클라이언트가 요청에 실어 보낸 `tools` 배열이고, 이 배열은 OpenAI 호환 chat 라우트가 파싱 시점에 이미 손에 쥐고 있다.

같은 파일의 파서 세 개는 이 문제를 이미 해결한 상태였다. `try_minimax_m3`, `try_glm47`, `try_longcat`은 `tools`를 받아 `minimax_m3_function_schema`로 호출된 함수의 스키마를 찾고 선언된 타입에 맞춰 값을 변환한다. XML 파서 두 개만 그러지 못했는데, 이유는 이들이 `&[fn(&str) -> Option<ToolCallParseResult>]` 타입의 디스패치 테이블 안에 있었기 때문이다. 필요한 데이터가 그 원소 타입을 통과할 수 없으니 파서는 추측을 택할 수밖에 없었다.

### 1.2 기존 문제점

- **string 값이 변조됨**: `coerce_minimax_param`은 `i64` 파싱, `f64` 파싱, 불리언 단어 목록을 차례로 시도한 뒤에야 문자열로 떨어뜨렸다. 미국 우편번호 `02134`는 `i64`로 파싱되어 2134가 되면서 앞의 0이 사라진다. 제품 코드 `1e5`는 `100000.0`이 되고, 텍스트 `true`는 JSON 불리언이 된다. 문자열을 기대한 도구 핸들러는 숫자를 받아 자체 검증에서 실패하거나, 조용히 잘못된 값으로 동작한다.
- **integer 값이 실수 형태로 남음**: 모델이 `integer` 파라미터에 `5.0`을 쓰면 JSON 실수 `5.0`이 그대로 나갔다. 엄격한 도구 구현은 이를 거부하며, 스키마는 이 값이 정수라고 분명히 말하고 있었다.
- **스키마 근거가 없는 추측**: `yes`, `on`, `no`, `off`를 불리언으로, `none`, `nil`을 null로 바꿨다. JSON Schema 어디에도 근거가 없는 규칙이며, 정상적인 문자열 인자를 잘못된 타입으로 만들었다.
- **coercion 로직 중복**: `coerce_minimax_param`은 `minimax_m3_coerce_leaf`와 같은 일을 더 약하게 구현한 두 번째 사본이었다. 스키마 처리가 개선되어도 한쪽에만 반영됐다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 스키마가 string으로 선언한 자리에 숫자가 도착해 도구 호출이 실패하거나 손상된 값으로 실행됨 | High | 앞자리 0, 지수 표기, 불리언 단어 형태의 문자열 인자에서는 확정적 |
| `integer` 인자를 `5.0`으로 받은 도구가 거부함 | Medium | 간헐적 (모델의 표기 방식에 좌우됨) |
| coercion이 발전할수록 XML 문법 두 개가 스키마 인식 문법 세 개와 더 벌어짐 | Medium | 로직이 중복된 동안에는 시간이 지나면 확정적 |
| 값 파싱 실패가 tool call 자체를 통째로 버리게 됨 | High | 설계로 회피: 모든 경로의 마지막은 원문 문자열 |

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
| 없음 | n/a | n/a |

부수적 개선이 하나 있다. `coerce_minimax_param`은 null 표기 세 가지를 비교하려고 모든 파라미터 값의 소문자 `String` 사본을 할당했다. `coerce_xml_param`은 `eq_ignore_ascii_case`를 쓰므로 할당이 없다.

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

- **테스트 커버리지**: 단위 테스트 20개 추가. `server::tool_calls` 스위트가 350개에서 370개가 되었고 전부 통과한다.
- **코드 복잡도**: 전체적으로 단순해졌다. 37줄짜리 수제 coercion 사다리가 공유 헬퍼로의 6줄 위임으로 바뀌었고, 대신 디스패처에 변형 2개짜리 enum이 생겼다.
- **기술 부채**: 감소. 중복 coercion 구현 둘 중 하나가 사라졌고, `parse_tool_calls`의 오래된 주석 두 곳을 바로잡았다.

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

### 3.2 선언된 타입을 무시하는 규칙은 `null` 하나뿐

**컨텍스트:**

`coerce_xml_param`은 스키마보다 앞서는 규칙을 정확히 하나만 남겼다. 대소문자를 가리지 않는 `null`이라는 단어는 `string`으로 선언된 파라미터에서도 JSON null이 된다.

**선택 이유:**

이 문법들에는 JSON null을 표기할 다른 방법이 없다. `<parameter=x>null</parameter>`은 모델이 쓸 수 있는 유일한 null이고 두 파서 모두 지금까지 이를 null로 내보냈으므로, 규칙을 없애면 이 표기를 쓰는 모든 모델에 회귀가 된다. 좁게 해석해서 string 타입의 `null`을 네 글자짜리 문자열로 두면, 클라이언트가 타입을 선언하는 순간 null 자체에 도달할 수 없게 된다.

**트레이드오프:**

문자열 `"null"`을 진짜로 보내고 싶은 모델은 이 문법으로 그것을 표현할 수 없다. 이는 이번 PR 이전에도 마찬가지였고, 스키마 인식 문법(MiniMax M3, GLM-4.7)은 string 스키마 아래에서 `"null"`을 문자열로 유지하므로 이 입력 하나에서 두 계열이 갈린다. XML 쪽 동작을 유지한 이유는 바꾸는 쪽이 수정이 아니라 회귀이고, 이슈의 범위도 아니기 때문이다.

### 3.3 `integer`는 `5.0`을 정규화하고 `number`는 표기를 보존

**컨텍스트:**

공유 `M3Type::Integer` 갈래는 `i64`와 `u64` 리터럴만 받았기 때문에 `integer` 타입의 `5.0`은 느슨한 fallback으로 떨어져 실수로 남았다. 이 갈래를 확장하면서 옆의 `number`가 정수 리터럴을 어떻게 다뤄야 하는지가 함께 문제가 됐다.

**선택 이유:**

`integer` 아래에서는 선언된 타입이 답을 정한다. 값은 정수이므로 `5.0`은 `5`이고, 실수 표기는 데이터가 아니라 모델의 서식이다. `number` 아래에서는 두 표기가 모두 유효하고 스키마가 선호를 표현하지 않으므로, 파서는 모델이 쓴 그대로를 유지한다(`5`는 `5`, `5.0`은 `5.0`). 이렇게 두면 `number` 타입 정수값에 대한 XML 파서의 기존 출력도 그대로 유지되는데, 일괄 실수 변환을 택했다면 이 부분이 바뀌었을 것이다.

**트레이드오프:**

실수 허용 범위는 2^53으로 제한된다. 그 너머에서는 `f64`가 모든 정수를 표현하지 못해 변환이 조용히 반올림되므로, 규칙이 거절하고 값은 fallback을 거쳐 실수로 남는다. 이는 빈틈이 아니라 추측하지 않겠다는 의도적 선택이다.

### 3.4 GLM-4.7과 LongCat에는 typed 경로 전체가 아니라 integer 규칙만 적용

**컨텍스트:**

`coerce_kv_value`(GLM-4.7, LongCat)는 `string` 스키마만 존중하고 나머지 타입은 전부 느슨한 fallback으로 보냈다. 이슈는 `integer` 타입 `5.0`이 모든 XML 문법에서 `5`가 되기를 요구했다.

**선택 이유:**

`coerce_kv_value`에 `Integer` 갈래를 추가하면 요구된 수정이 정확히 그만큼 이뤄진다. 대신 이 함수를 `minimax_m3_typed_coerce`로 통째로 태웠다면 `enum`, `anyOf`, `object`, `array`, `boolean` 스키마 처리까지 부수적으로 바뀐다. 이슈가 정당화하는 범위보다 훨씬 넓고, 두 문법의 기존 테스트가 덮지 않는 영역이다. 규칙 자체는 `parse_integer_literal` 하나를 공유하므로 호출부 사이에서 어긋날 수 없다.

**트레이드오프:**

GLM-4.7과 LongCat은 여전히 MiniMax M3보다 스키마에 덜 엄격하다. 그 간극은 이제 `coerce_kv_value` 문서 주석에 명시되어 있고, 좁히는 작업은 별도의 검증 가능한 변경으로 분리된다.

---

## 4. 구현 상세

### 4.1 동작 변화표

| 스키마 | 원문 값 | 변경 전 | 변경 후 |
|-------|--------|--------|--------|
| `{"type":"string"}` | `02134` | `2134` | `"02134"` |
| `{"type":"string"}` | `true` | `true` | `"true"` |
| `{"type":"string"}` | `1e5` | `100000.0` | `"1e5"` |
| `{"type":"integer"}` | `5.0` | `5.0` | `5` |
| `{"type":"integer"}` | `5.5` | `5.5` | `5.5` (느슨한 fallback, 호출 유지) |
| `{"type":"number"}` | `5` | `5` | `5` |
| `{"type":"object"}` | `{not json` | `"{not json"` | `"{not json"` |
| 없음 | `5` / `true` / `null` | `5` / `true` / `null` | 동일 |
| 없음 | `yes` / `on` / `none` | `true` / `true` / `null` | `"yes"` / `"on"` / `"none"` |
| 없음 | `[1, 2]` | `[1, 2]` | `[1, 2]` |

스키마가 없을 때의 차이가 하나 더 있다. fallback이 텍스트가 `{` 또는 `[`로 시작할 때만 JSON 파싱을 시도하던 것을 이제 항상 시도하므로, 스키마 없이 JSON 리터럴로 쓰인 값은 그 리터럴로 파싱된다. MiniMax M3와 GLM-4.7, LongCat이 이미 이렇게 동작하고 있었으므로 이제 네 문법의 동작이 일치한다.

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
    if raw.eq_ignore_ascii_case("null") {
        return serde_json::Value::Null;
    }
    minimax_m3_coerce_leaf(raw, schema)
}
```

**변경 이유:** 타입은 텍스트의 생김새가 아니라 스키마에 속한다. `minimax_m3_coerce_leaf`는 `enum`과 `anyOf`/`oneOf`, 배열 item 해석까지 포함한 선언 타입 경로를 이미 구현하고 있고, 마지막이 원문 문자열이므로 파싱 실패가 호출을 버리는 일도 없다.

```rust
// 변경 후: 공유 integer 규칙
fn parse_integer_literal(raw: &str) -> Option<serde_json::Value> {
    if let Some(v) = parse_exact_integer_literal(raw) {
        return Some(v);
    }
    let f = raw.parse::<f64>().ok()?;
    if f.is_finite() && f.fract() == 0.0 && f.abs() <= INTEGER_FROM_FLOAT_LIMIT {
        return Some(serde_json::Value::Number((f as i64).into()));
    }
    None
}
```

**변경 이유:** `integer`는 모델이 실수로 적은 정수값을 받아들여야 하지만, 변환이 정확한 범위 안에서만 그렇다. `INTEGER_FROM_FLOAT_LIMIT`은 2^53이다.

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

### 5.2 표기 보존도 올바른 coercion의 일부다

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
| 2^53 | `f64`가 모든 정수를 표현하는 최대 크기 | 소수부 없는 실수 규칙의 경계 |

### 관련 PR/이슈

- Issue #1336: 이 PR이 구현한 명세. 4.1절의 스키마 표가 여기서 왔다.
- GLM-4.7 / LongCat key-value 파서와 MiniMax M3 네임스페이스 XML 파서는 이 PR이 재구현하지 않고 재사용한 선행 구현이다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +545 |
| 삭제된 라인 | -121 |
| 테스트 추가 | 20 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Correctness | 파일 2개 | Qwen3-Coder / MiniMax M2 문법의 스키마 기반 타이핑, 공유 integer 규칙의 소수부 없는 실수 허용 |
| Code Quality | 1 | `coerce_minimax_param` 제거, `parse_tool_calls`의 오래된 주석 2곳 수정 |
| Tests | 20 | 두 파일에 스키마, 무스키마, 경계값, 디스패처 순서 커버리지 추가 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `272afe8` | fix | fix(server): type XML tool-call arguments by the request schema |

---

## 8. 후속 조치

### 완료 필요

- [ ] 실제 Qwen3-Coder 체크포인트로 end-to-end 동작 확인 (요청과 기대 `tool_calls[].function.arguments`는 PR 본문에 있다). 이 PR은 단위 테스트로만 검증했고 체크포인트를 올리지 않았다.

### 모니터링 필요

- 두 문법에서 클라이언트로 나가는 tool-call 인자, 특히 숫자처럼 보이는 string 타입 값. 기존 coercion에 맞춰 우편번호를 다시 문자열로 되돌리는 식으로 보정하던 클라이언트는 이제 올바른 문자열을 받으므로 보정을 걷어내야 한다.

### 향후 개선 사항

- GLM-4.7과 LongCat을 string, integer 규칙만이 아니라 `minimax_m3_typed_coerce` 전체 경로로 태울지 결정. 이번에는 의도적으로 범위를 끊었고 `coerce_kv_value` 주석에 남겼다.
- 델타 단위 증분 tool-call 인자 스트리밍은 여전히 범위 밖이다. 스트림 경로는 스트림 종료 시점에 한 번 파싱한다.
- 스키마 검증 오류(`required` 키 누락, `enum` 불일치)는 아직 클라이언트에 전달되지 않는다. 파서는 변환만 하고 거부하지 않는다.

---

## 부록

### A. 테스트 결과

```
cargo test --profile test-fast --features metal,accelerate --lib server::tool_calls
test result: ok. 370 passed; 0 failed; 0 ignored; 7313 filtered out

cargo test --profile test-fast --features metal,accelerate --lib server::muse_atem
test result: ok. 11 passed; 0 failed; 0 ignored; 7672 filtered out

cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings
Finished (no warnings)

cargo fmt --all -- --check
clean
```

이슈가 유지를 요구한 네 케이스는 수정 없이 통과한다: `qwen3_coder_single_call_multiple_params_with_type_coercion`, `minimax_m2_numeric_params`, `minimax_m2_boolean_param`, `minimax_m2_null_param`.

### C. 참고 자료

- `docs/code-guidelines.md`: `coerce_xml_param`과 `parse_integer_literal`에 적용한 `// Used by:` 규약.
- JSON Schema type 키워드: `integer`는 `number`와 구분되는 별도 타입이며, 두 갈래가 다르게 동작하는 근거다.
