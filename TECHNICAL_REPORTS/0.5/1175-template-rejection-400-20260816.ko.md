# 기술 보고서: PR #1175 - fix(server): fail requests a template refuses, map reasoning_effort

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

챗 템플릿이 Jinja의 `raise_exception`으로 값을 거부하면, 예전에는 그 거부가 `render_simple_fallback`에 그대로 삼켜져 채팅 형식도 시스템 메시지도 도구 선언도 없는 프롬프트로 HTTP `200`을 응답했다. 클라이언트 입장에서는 이 답을 정상 답변과 구분할 방법이 없었다. 이번 PR은 이 거부를 템플릿 자신의 메시지를 담은 `400`으로 바꾸고, 별도로 OpenAI 표준 최상위 `reasoning_effort` 필드를 `reasoning_effort` chat-template kwarg로 매핑한다. 이 필드는 이전까지 값을 받기만 하고 조용히 버려지고 있었다. 의도적 거부와 순수 렌더 실패를 가르는 판별 기준은 `raise_exception`이 `minijinja::Error`의 source로 붙이는 비공개 `TemplateRejection` sentinel이며, 오류 체인을 따라가며 이를 복구한다. `ErrorKind::InvalidOperation` 하나만으로는 이 역할을 할 수 없는데, minijinja 자체가 `value/argtypes.rs`와 `format_utils.rs`에서만 스무 곳이 넘는 지점에서 평범한 타입 변환 실패에 같은 kind를 사용하기 때문이다. 이 kind만 보고 판별하면 진짜 엔진 문제까지 `400`으로 바뀌어버린다. 상위 PR 리뷰와 보안 검토는 HIGH 항목 두 건을 찾았고, 둘 다 이번 finalization 이전에 이미 수정됐다. 하나는 prompt-cache 다음 턴 warm-up이 실제 bucket의 `template_sig`와 다른 `reasoning_effort`로 probe를 렌더링하던 불일치였고, 다른 하나는 `CHANGELOG.md` 누락이었다. 이번 finalization 단계에서는 이 PR 스스로 도입한 결함 하나(흔한 무인자 tool-call 표기 두 가지가, 예전에는 우아하게 처리되던 요청을 이제 강제로 실패시키는 가용성 회귀)를 찾아 수정했고, 거부 메시지 truncation 헬퍼의 로그 인젝션 경로를 막았으며, 오래된 문서 서술 두 곳을 바로잡고, 리뷰가 누락으로 지적한 테스트를 추가했다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1164는 Qwen3.8-27B 자격 검증(#1163)에서 분리된 이슈로, `reasoning_effort` 처리의 결함 두 가지를 다룬다. 첫째, 챗 템플릿의 의도적인 `raise_exception` 거부가 요청을 실패시키는 대신 조용한 fallback 프롬프트로 저하됐다. 둘째, OpenAI 표준 최상위 `reasoning_effort` 필드를 받는 코드 경로가 아예 없어서, serde가 값을 아무도 손대기 전에 버렸다. 둘 다 평범한 OpenAI 호환 클라이언트에서 도달 가능한 경로이고, 합쳐 놓고 보면 가장 가능성 높은 호출인 `reasoning_effort: "high"`가 어디에 넣느냐에 따라 아무 효과가 없거나(최상위 필드) 프롬프트를 조용히 저하시키는(`chat_template_kwargs`) 결과를 낳았다.

### 1.2 기존 문제점 (리뷰·보안 검토에서 발견, 이번 finalization에서 처리)

- **가용성 회귀** (최우선 처리 대상으로 지정): `tests/fixtures/muse_glimmer/chat_template.jinja`의 `render_atem` 매크로는 `tool_call.function.arguments`가 mapping이 아니면 예외를 낸다. `normalize_tool_call_arguments`는 `arguments` 문자열이 유효한 JSON으로 파싱되고 그 JSON이 object일 때만 object로 다시 썼으므로, `arguments: ""`(에이전틱 클라이언트가 흔히 돌려보내는 무인자 표기)와 `arguments: "null"`은 문자열로 남았다. 이 PR 이전에는 이 값들이 fallback 답변으로 저하됐지만, 이 PR 이후로는 OpenAI wire format 상 유효한 입력임에도 클라이언트가 대응할 방법이 없는 메시지와 함께 그냥 실패한다. 이 실패는 들러붙는다: 문제의 `tool_calls` 항목이 대화 기록에 그대로 남아 이후 모든 턴에서 재현된다.
- **CHANGELOG 업그레이드 노트가 엉뚱한 사례를 지목**: 노트는 "지원하지 않는 role, 번갈아 나오지 않는 role, 거부하는 tool 선언, 알 수 없는 kwarg 값"을 `200`에서 `400`으로 바뀌는 트래픽으로 나열했다. 이 트리의 체크포인트들에서 실제로 가장 트래픽이 많은 새 `400` 두 가지는 이 목록에 없다: 0번이 아닌 위치의 `system` 메시지, 그리고 `user` 메시지가 전혀 없는 대화다.
- **`reject_if_template_rejection`의 로그 인젝션 경로**: 이 함수는 일부 호출자가 제어할 수 있는 텍스트를 최대 512자까지 단일 라인 평문 `tracing_subscriber::fmt` 레코드에 로깅한다. 이 텍스트에 줄바꿈이나 ANSI escape가 들어 있으면 `--log-file` 출력에 그대로 실린다.
- **오래된 문서 서술 두 건**: `MAX_MESSAGE_CHARS`의 doc comment는 이 상한이 서버 측 로그 라인을 바꾸지 않는다고 적혀 있었는데, 거부 경로가 truncate되지 않은 `WARN` 대신 truncate된 메시지를 `INFO`로 로깅하기 시작하면서 더 이상 사실이 아니게 됐다. `docs/supported-models.md`는 `TemplateRejection`을 "private sentinel"이라 불렀지만, 이 struct는 `pub`이라 크레이트 밖에서도 이름을 참조할 수 있다.
- **테스트 커버리지 누락**: 512자 상한을 고정하는 테스트가 없었다. 나중에 byte slicing으로 리팩터링됐을 때 패닉을 재도입할 수 있는 UTF-8 경계 케이스도 마찬가지였다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 무인자 tool-call 재현이 대화의 이후 모든 턴을 강제로 실패시킴 (가용성 회귀) | Medium-High (이 PR이 아니었다면 그대로 출시됐을 흔한 에이전틱 패턴을 깨뜨림) | 수정 전에는 mapping을 요구하는 모든 템플릿에서 확정적 |
| 오퍼레이터가 뭉뚱그린 CHANGELOG 노트만 보고 업그레이드 영향 범위를 잘못 판단 | Low (문서 문제일 뿐 기능 영향 없음) | 업그레이드 전에 노트를 읽는 오퍼레이터라면 실재 |
| 호출자가 제어하는 텍스트가 가짜 로그 라인을 위조하거나 `--log-file` 출력에서 터미널 상태를 다시 씀 | Low-Medium (로그 무결성 문제, 데이터 노출은 아님) | `--log-file`을 쓰고 악의적 클라이언트에 노출된 배포라면 실재 |
| 오래된 문서 서술이 향후 유지보수자를 오도함 | Low (문서 문제일 뿐) | 해당 없음, 수정 완료 |
| `truncate_chars`의 향후 byte slicing 리팩터링이 UTF-8 경계 패닉을 재도입 | 현재는 Low (그런 리팩터링이 없음), 다만 가드 자체가 없었음 | 이번 단계에서 고정 테스트를 추가하기 전까지는 잠재적 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

이미 수정된 HIGH 항목(warm-up kwargs 불일치, CHANGELOG 누락)은 3.2절에서 다루며, 이번 단계에서는 검증 외에 다시 손대지 않았다. 이번 단계 자체가 찾은 문제는 `truncate_chars` 호출자들의 로그 인젝션 경로다.

**발견된 이슈:**

| 이슈 | 심각도 | 상태 |
|-----|-------|-----|
| warm-up kwargs 불일치: `render_next_turn_history`가 kwargs를 직접 계산하면서 매핑된 `reasoning_effort`를 놓쳐, 자신이 속하지 않는 `template_sig` bucket 아래 prefill 벡터를 저장 | High | 이번 단계 이전에 수정 완료 (`8dac53ea`) |
| `CHANGELOG.md`에 `#1164` 항목 누락 | High | 이번 단계 이전에 수정 완료 (`2bd259b6`) |
| 가용성 회귀: 빈 문자열/`"null"` tool-call arguments가 저하되지 않고 강제로 실패 | Medium-High | 수정 완료 (`3c880883`) |
| 로그 인젝션 경로: truncate된 거부 텍스트의 제어 문자가 필터링되지 않은 채 평문 로그 레코드에 도달 | Medium | 수정 완료 (`e355d4c2`) |

제어 문자 수정은 거부가 아니라 필터링을 택했다. `truncate_chars`의 두 호출자(`TemplateRejection::new`, `truncate_key_for_log`) 모두 요청에 이미 받아들여져 로그·오류 sink로 향하는 텍스트를 다루지, 거부해도 되는 요청 경계가 아니기 때문이다. 이는 `src/server/florence2_worker.rs`의 `validate_task_input`과는 의도적으로 다른 선택이다. 그 함수는 실제 요청 경계에 있으므로 제어 문자를 그냥 거부한다.

### 2.2 성능 관점

측정하지 않았고 필요하지도 않았다. 이번 단계의 변경은 모두 요청 준비 경로(JSON 정규화, 문자열 truncation)에 있으며 hot inference 경로가 아니다. 토크나이징, KV 캐시, 생성 루프 어디에도 손대지 않았다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 이번 단계가 새로 만든 것은 없다. 가용성 회귀 수정은 `arguments: ""`를 예전처럼 우아하게 처리하는 동작을 되살리고, `arguments: "null"`에도 같은 처리를 새로 확장한다. `"null"`은 이전에는 mapping을 요구하지 않는 템플릿에서는 그대로 문자열로 남아 있어도 무해했지만, 이제는 모든 템플릿에서 `{}`로 정규화된다. `"[1,2]"`, 단순 스칼라, 잘린/손상된 JSON은 "인자 없음"으로 읽을 안전한 방법이 없으므로 의도적으로 그대로 둔다.
- **새로운 의존성**: 없음.
- **호환성**: 빈 문자열/`"null"` 정규화는 어떤 템플릿이 로드됐는지와 무관하게 동일하게 적용된다. `arguments`를 아예 살펴보지 않는 템플릿(흔한 경우)은 영향받지 않는다. 제어 문자 필터는 거부 메시지에 제어 문자가 이미 포함돼 있을 때만 정확한 바이트를 바꾸는데, 이 트리의 테스트 스위트가 쓰는 어떤 템플릿 fixture도 그런 메시지를 만들지 않는다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: `chat_request_tests.rs`에 새 단위 테스트 6건(빈 문자열, 공백만 있는 문자열, `"null"`, 손상/잘린 문자열은 여전히 문자열로 남음, 그리고 기존 스칼라/배열 커버리지에서 이제 별도로 처리되는 `"null"`을 "문자열로 남음" 목록에서 뺀 조정)을 추가했고, `muse_atem_roundtrip_tests.rs`에 실제 Muse Glimmer fixture 템플릿으로 재현된 빈 인자 tool-call을 렌더링하는 end-to-end 회귀 테스트 1건을 추가했다. `chat_template.rs`에는 제어 문자 필터와 512자 상한을 검증하는 새 단위 테스트 4건을 추가했는데, 그중 하나는 512번째 문자가 3바이트 UTF-8 문자 연속의 시작점이 되도록 구성해, byte slice 방식 상한이었다면 문자 중간에서 잘렸을 지점을 정확히 겨냥한다.
- **코드 복잡도**: `normalize_tool_call_arguments`에 조기 반환 분기 하나가 늘었고, `truncate_chars`는 기존 iterator 체인에 `.map()` 하나가 늘었다. 그 외 제어 흐름 변경은 없다.
- **기술 부채**: 감소했다. 문서 수정 두 건은 코드와 더 이상 맞지 않는 서술을 없앴고, 새 테스트는 리뷰가 명시적으로 지적한 공백을 메웠다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `TemplateRejection` sentinel, 그리고 `ErrorKind::InvalidOperation` 하나만으로는 안전한 판별 기준이 될 수 없는 이유

상위 PR이 답한 핵심 설계 질문은 이것이다. "템플릿이 이 입력을 의도적으로 거부했다"와 "mlxcel이 이 템플릿을 렌더링하지 못했다"를, 둘 다 지금은 같은 minijinja 오류 kind로 나타나는 상황에서 어떻게 구분하는가.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| `minijinja::ErrorKind::InvalidOperation`을 기준으로 판별 | 새 타입 없음, 코드 최소 | 안전한 판별 기준이 아니다. mlxcel이 직접 등록하는 `raise_exception` 구현(minijinja에는 내장 `raise_exception`이 없어 mlxcel이 직접 만들어 등록한다) 자체가 이 kind를 쓰지만, 엔진도 `value/argtypes.rs`와 `format_utils.rs`에서만 독립적으로 스무 곳이 넘는 지점에서(`\|items`를 non-mapping에 적용, `\|tojson` 타입 불일치, 문자열 포맷팅 오류 등) 같은 kind를 쓴다. 이 kind만 보고 판별하면 진짜 렌더 실패까지 `400`으로 바뀌어버려, 고치려던 것과 정반대 실수가 된다. |
| 오류 메시지 텍스트를 매칭 | 새 타입 없음 | 그 텍스트는 템플릿 작성자가 자유롭게 고른다. 템플릿마다 다르고 문구가 바뀌면 쉽게 깨진다. |
| **선택: 오류의 `source()`에 비공개 `TemplateRejection` sentinel을 붙임** | 정확한 타입 수준 판별, 문자열 매칭 없음. minijinja가 오류를 감싸는 경로(`{% include %}`, `super()`, loop 재귀 모두 교체가 아니라 `with_source`로 붙이는 방식)를 그대로 통과 | 타입 하나, `env.add_function` 등록 하나가 늘어남 |

**선택 이유**: 이 sentinel은 mlxcel이 minijinja `Environment`에 스스로 등록하는 `raise_exception` 함수 본문 안에서만 붙는다. 그래서 이 sentinel의 존재 여부는, 감싸고 있는 `minijinja::Error`가 어떤 `ErrorKind`를 갖든 템플릿 작성자가 어떤 문구를 골랐든 관계없이 정확한 신호가 된다. `template_rejection_message`는 한 단계만 확인하는 대신 `anyhow::Error` 체인 전체를 따라가며(`err.chain().find_map(...)`) 이를 복구하는데, minijinja의 VM이 전파되는 오류에 파일/라인 정보를 교체가 아니라 그 자리에 덧붙이는 방식으로 주석을 달고, 실제로 감싸는 몇몇 경로도 원본을 `with_source`로 붙이기 때문이다.

이 타입의 비공개성은 의도적으로 절반만 적용됐다. `TemplateRejection`은 `pub`이라(감싸는 모듈이 `pub mod chat_template`이므로 `mlxcel::server::chat_template::TemplateRejection`으로 참조 가능) 크레이트 밖에서도 이름을 붙일 수 있지만, 생성자(`fn new`, `pub fn new`가 아니다)와 필드는 비공개다. 즉 외부 코드는 타입 이름을 알 수 있고, 크레이트 자체의 오류 체인을 통해 얻은 `&TemplateRejection`이 있다면 공개 `message()` accessor로 메시지를 읽을 수는 있지만, 처음부터 하나를 만들어낼 수는 없다. mlxcel의 렌더 경로 밖에서는 아무도 가짜 거부를 위조해 `reject_if_template_rejection`을 평범한 텍스트에 대해 발화시킬 수 없다.

### 3.2 warm-up kwargs 불일치 (이번 단계 이전에 발견해 수정, `8dac53ea`)

리뷰가 지적한 두 HIGH 항목 중 더 무게가 실리는 쪽이다. 문체 문제가 아니라 실제로 살아 있는 prompt-cache 정합성 버그였기 때문이다. `render_next_turn_history`(이슈 #1144의 다음 턴 warm-up)는 `extract_request_kwargs` + `merge_server_and_request`로 chat-template kwargs를 직접 계산했다. 이 PR이 병합 로직의 단일 진실 공급원으로 새로 도입한 `resolve_effective_kwargs`를 우회한 것이다. 그 결과 매핑된 최상위 `reasoning_effort`를 전혀 보지 못했다. warm-up probe가 저장되는 bucket은 `build_prompt_cache_request_context`가 만드는데, 이 함수의 `template_sig`는 이미 `resolve_effective_kwargs`를 호출하고 있어 매핑된 값을 포함한다. 그래서 최상위 `reasoning_effort`를 설정한 요청이 이를 실제로 읽는 템플릿(Qwen3.8) 앞에서, 템플릿의 `xhigh` 기본값으로 렌더링된 벡터를, 예컨대 effort가 `low`라고 말하는 signature의 bucket 아래 저장하는 일이 벌어졌다. 다음 턴의 실제 렌더링은 `low`를 쓰고 `low` bucket을 찾았으므로, 공유 head 이후로는 아무것도 일치하지 않아 warm-up 결과가 조용히 버려졌다. 출력을 손상시키지는 않았지만, warm-up 작업 자체가 무의미해졌다. 이는 바로 아래 세 줄에 있는 이 함수 자신의 comment가 이미 `preserve_thinking`에 대해 경계하던 것과 같은 종류의 불일치다. 수정은 `render_next_turn_history`도 `resolve_effective_kwargs`를 호출하게 만들어, 두 코드 경로가 병합 로직을 동일하게 도출하도록 했다.

### 3.3 빈 문자열과 `"null"` tool-call arguments: 어디까지 모호하게 남기고 어디부터는 아닌가

`normalize_tool_call_arguments`의 기존 규칙은 "wire-format `arguments` 문자열이 유효한 JSON으로 파싱되고 그 JSON이 object일 때만 object로 다시 쓴다"였다. 그 외에는 설계상 모두 문자열로 남는다. 스칼라나 배열을 mapping으로 안전하게 읽을 방법이 없기 때문이다. 이번 단계가 찾은 회귀는 이 규칙이 지나치게 좁았다는 데 있다. 빈 문자열은 애초에 유효한 JSON이 아니어서(파싱 자체가 실패한다) 의도가 아니라 기본값으로 "문자열로 남음"에 떨어졌고, 문자열 `"null"`은 JSON `null`로 파싱되어 유효한 JSON이긴 하지만 object가 아니라서 이 역시 "문자열로 남음"에 떨어졌다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| 빈 문자열만 `{}`로 매핑하고 `"null"`은 문자열로 남김 | 변경 범위가 좁고 이슈의 최소 요구사항과 일치 | `"null"`은 빈 문자열 대신 `null` 값을 `JSON.stringify()`하는 클라이언트가 내는, 똑같은 "인자 없음" 의도다. 그대로 두면 같은 템플릿 매크로가 똑같이 흔한 표기에 대해 계속 예외를 낸다 |
| **선택: 빈 문자열/공백만 있는 문자열과 `"null"`을 모두 `{}`로 매핑하고, `"[1,2]"`, 스칼라, 손상된 JSON은 그대로 둠** | 두 표기 모두 명확한 "인자 없음" 신호다. 나머지는 그렇지 않다 | 이슈의 문구 그대로보다는 범위가 조금 넓다. 다만 암묵적으로 넘어가지 않고 명시적으로 결정했다 |

**선택 이유**: 판별 기준은 "이것이 유효한 JSON인가"가 아니라 "이것이 모호하지 않은 무인자 표기인가"다. 빈 문자열과 `"null"`은 둘 다 그렇다. `"[1,2]"`, 단순 스칼라, 잘린 payload는 그렇지 않다. 이들의 의도를 추측하는 것은 같은 PR에서 형제 격인 `reasoning_effort` 매핑이 명시적으로 거부하는 부류의 조용한 값 변환(상위 PR 3.1절의 설계, 여기서는 arguments에 그대로 적용)과 정확히 같은 실수다.

### 3.4 제어 문자 필터링: 거부가 아니라 필터

`truncate_chars`는 기존 길이 상한 앞에 `.map(\|c\| if c.is_control() { ' ' } else { c })` 단계 하나를 추가했다. 터미널 escape sequence를 여는 ANSI `ESC`를 포함한 모든 제어 문자를 버리거나 문자열 전체를 거부하는 대신 평범한 공백 하나로 바꾼다. `truncate_chars`의 두 호출자 중 어느 쪽도 요청 경계가 아니라는 점을 그대로 반영한 선택이다. `TemplateRejection::new`는 템플릿이 이미 만들어낸 메시지의 길이를 제한하고, `truncate_key_for_log`는 이미 렌더링에 받아들여진 kwargs 키의 길이를 제한한다. 실제 요청 경계에서 동작해 거부해도 되는 `src/server/florence2_worker.rs`의 `validate_task_input`이 제어 문자를 그냥 거부하는 것과는 의도적으로 다르다. 버리지 않고 필터링(공백으로 치환)하는 이유는, 값의 길이 자체가 제어 문자의 존재 여부를 흘리는 정보가 되지 않게 하기 위해서다.

---

## 4. 구현 상세

### 4.1 무인자 정규화 (`src/server/chat_request.rs`)

```rust
fn normalize_tool_call_arguments(tool_calls: &mut serde_json::Value) {
    let serde_json::Value::Array(calls) = tool_calls else {
        return;
    };
    for call in calls {
        let Some(args) = call.pointer_mut("/function/arguments") else {
            continue;
        };
        let serde_json::Value::String(s) = args else {
            continue;
        };
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed == "null" {
            *args = serde_json::json!({});
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
            && parsed.is_object()
        {
            *args = parsed;
        }
    }
}
```

### 4.2 제어 문자 필터 (`src/server/chat_template.rs`)

```rust
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars().map(|c| if c.is_control() { ' ' } else { c });
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}\u{2026}")
    } else {
        truncated
    }
}
```

이 함수의 호출자 두 곳(`TemplateRejection::new`, `truncate_key_for_log`) 모두 이 변경 하나로 함께 고쳐진다. 각 호출 지점을 따로 손보는 대신 공유 헬퍼에 필터를 접어 넣은 이유가 여기 있다.

### 4.3 End-to-end 회귀 테스트 (`src/server/muse_atem_roundtrip_tests.rs`)

`atem_replay_with_empty_string_tool_call_arguments_does_not_raise`는 4개 메시지(user, `arguments: ""`를 가진 tool-call 포함 assistant, tool 응답, 후속 user)로 대화를 구성하고 실제 Muse Glimmer fixture 템플릿으로 렌더링한 뒤, 렌더링이 성공하고 템플릿 자신의 `"Onyx ATEM chat template requires..."` 거부 텍스트가 나타나지 않는지 확인한다. 수정 전이었다면 실패했을 테스트다. `render_atem`의 `{%- if args is not mapping -%}{{- raise_exception(...) }}`가 이 입력에 정확히 반응하기 때문이다.

### 4.4 512자 상한과 멀티바이트 경계 테스트 (`src/server/chat_template.rs`)

`template_rejection_message_multibyte_boundary_does_not_panic`는 ASCII 511자 뒤에 CJK(UTF-8 3바이트) 문자 200개를 이어 붙여, 512번째 문자가 멀티바이트 연속의 첫 문자가 되도록 구성한다. 순진하게 `&s[..512]`로 byte slice를 했다면 정확히 그 문자의 3바이트 인코딩 한가운데인 byte 512에서 잘렸을 지점이다. `chars().take()`를 쓰는 한 이 패닉은 구조적으로 불가능하지만, 이 테스트는 그 사실을 고정해 향후 byte slicing으로 리팩터링됐을 때 프로덕션에서 악의적 입력을 만나기 전에 즉시 실패하게 한다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 (상위 기능 커밋 `90f979b7`) | 14 |
| 변경된 파일 수 (warm-up kwargs 수정, 기존 완료, `8dac53ea`) | 2 |
| 변경된 파일 수 (CHANGELOG 누락 수정, 기존 완료, `2bd259b6`) | 1 |
| 변경된 파일 수 (가용성 회귀 수정, 이번 단계, `3c880883`) | 3 |
| 변경된 파일 수 (로그 인젝션 필터, 이번 단계, `e355d4c2`) | 1 |
| 변경된 파일 수 (문서 수정, 이번 단계, `18ed6fe4`) | 2 |
| 추가/삭제 라인 (이번 단계 합계) | +223 / -14 |
| 테스트 추가 (이번 단계) | `chat_request_tests` 6건 + `muse_atem_roundtrip_tests` 1건 + `chat_template::tests` 4건 = 11건 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| 가용성 수정 | 1 | `normalize_tool_call_arguments`가 빈 문자열/`"null"` arguments를 `{}`로 매핑 |
| 보안 강화 | 1 | `truncate_chars`가 제어 문자를 필터링, 두 호출 지점을 한 번에 해결 |
| 문서 정확성 | 3 | `MAX_MESSAGE_CHARS` doc comment, `TemplateRejection` 비공개성 서술, `CHANGELOG.md` 업그레이드 노트 |
| 테스트 커버리지 | 3 | 무인자 단위/end-to-end 테스트, 512자 상한 + UTF-8 경계 + 제어 문자 테스트 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `90f979b7` | fix | fail requests a template refuses, map reasoning_effort (상위 PR) |
| `8dac53ea` | fix | resolve warm-up kwargs through the shared helper (기존 HIGH 수정) |
| `2bd259b6` | docs | add the #1164 CHANGELOG entry (기존 HIGH 수정) |
| `3c880883` | fix | treat empty and "null" tool-call arguments as no-argument calls |
| `e355d4c2` | fix | filter control characters out of truncated rejection text |
| `18ed6fe4` | docs | name the concrete #1164 upgrade-note cases, fix sentinel wording |

---

## 7. 후속 조치

### 완료 필요

- [ ] 없음. 이번 finalization 단계에 요청된 항목은 모두 수정하고 테스트로 고정했다.

### 향후 개선 사항 (알려진 제약으로 기록, 이번 PR에서는 수정하지 않음)

- `src/server/router_front.rs:657`은 분리형(disaggregated) router-front 표면에서 템플릿 거부를 `400`이 아니라 `500`으로 매핑한다. 실재하는 문제다. OpenAI SDK는 `5xx`는 재시도하고 `4xx`는 재시도하지 않으므로, 잘못된 요청 하나가 세 개가 되고 거짓 서버 오류 경보까지 곁들여진다. 다만 이 PR이 손댄 표면과는 다른 곳에 있는 기존 문제이며, #1176으로 추적 중이고 이번 단계에서는 의도적으로 손대지 않았다.
- Responses API의 `reasoning.effort`는 여전히 참고용이고 template kwarg로 매핑되지 않는다. 그래서 이를 거부할 템플릿 앞에서 `effort: "high"`를 보내는 Responses 클라이언트는 chat-completions 필드가 이제 내는 `400` 대신 여전히 조용한 `200`을 받는다. `docs/responses-api.md`에 명시했고, 별도 범위 결정으로 의도적으로 미뤘다.
- CLI에도 같은 형태의 조용한 저하가 `src/commands/generate.rs`의 `apply_user_chat_template`(`.unwrap_or_else(\|_\| user_prompt.to_string())`)에 있다. 오퍼레이터가 출력을 직접 보므로 위험도가 낮고, 이 이슈가 다루는 서버 범위 밖이다.

---

## 부록

### A. 테스트 결과

- `cargo test --release -p mlxcel --lib --features metal,accelerate server::chat_request`: 87건 통과, 실패 0.
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::muse`: 17건 통과, 실패 0 (새 `atem_replay_with_empty_string_tool_call_arguments_does_not_raise` 포함).
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::chat_template`: 104건 통과, 1건 ignored(`models/` 디렉터리가 필요한 로컬 모델 감사 테스트), 실패 0.
- `cargo test --release -p mlxcel --lib --features metal,accelerate server::reasoning_effort_tests`: 19건 통과, 실패 0.
- `cargo fmt --check`: 이번 단계에서 건드린 모든 파일에서 클린.
- `cargo clippy --release -p mlxcel --lib --features metal,accelerate --tests -- -D warnings`: 클린.

### B. 참고 자료

- 이슈 #1164 (사양), 이슈 #1163 (이를 드러낸 자격 검증 실행)
- 이슈 #1176 (`router_front.rs`의 `500` vs `400` 간극을 추적, 이번 단계에서는 명시적으로 범위 밖)
- `src/server/chat_template.rs` (`TemplateRejection`, `template_rejection_message`, `truncate_chars`), `src/server/chat_request.rs` (`normalize_tool_call_arguments`, `reject_if_template_rejection`, `resolve_effective_kwargs`)
- `tests/fixtures/muse_glimmer/chat_template.jinja` (`render_atem`, 이번 단계의 가용성 수정이 충족시키는 mapping 요구사항을 가진 매크로)
- PR #1175 리뷰·보안 코멘트
