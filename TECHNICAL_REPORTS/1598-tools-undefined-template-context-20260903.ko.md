# 기술 보고서: PR #1598 - fix(server): define tools only when the request sends them

**작성일**: 2026-09-03
**작성자**: mlxcel maintainers
**리뷰어**: implementation and security review cycle
**상태**: 완료 (가중치 없이 두 체크포인트의 템플릿 렌더 일치를 검증함. 실제 가중치를 쓰는 HTTP 게이트는 중앙에서 실행)
**언어**: Rust, Markdown
**위험도**: Medium (`tools is defined` 또는 `tools is not none`으로 분기하는 모든 체크포인트에서 렌더된 프롬프트가, 따라서 프롬프트 캐시 접두어가 바뀐다)

---

## 요약

`build_template_context`는 minijinja 컨텍스트에 `tools` 키를 무조건 넣었고, 두 렌더 경로 모두 요청에 도구가 없으면 빈 `Vec<Tool>`을 대신 넣었다. 그래서 `tools is defined`나 `tools is not none`으로 도구 분기를 결정하는 채팅 템플릿은 정의돼 있고 none이 아닌 빈 리스트를 보게 되고, 평범한 대화 요청마다 빈 함수 목록이 붙은 도구 호출 프리앰블을 렌더했다. 실제로 두 계열이 이 모양이다. DeepSeek V3 파생(Youtu-LLM)과 Llama 3.1 / 3.2 / 3.3 / 4이며, 후자에는 프로젝트의 기준 트랜스포머 참조 체크포인트가 들어간다.

PR #1598은 `build_template_context`가 `tools: Option<minijinja::Value>`를 받아 `Some`일 때만 키를 넣도록 바꾸고, "도구 없음"과 명시적 `"tools": []`를 모두 `apply_raw_inner` / `apply_inner` 안에서 `None`으로 정규화한다. 여기서 `none`이 아니라 정의되지 않은 상태여야 하는 것은 취향이 아니라 필수다. minijinja의 `iterable` 테스트는 `none`에 대해 참인데 `none`에 `| length`를 적용하면 오류가 나므로, `none`을 넣으면 Nemotron / MiMo 계열의 렌더가 중단되고 프롬프트가 조용히 단순 폴백으로 대체된다. 결과적으로 로컬 코퍼스의 모든 가드 형태에서 transformers(`tools=None`) 및 llama-server(키 미설정)와 동일해진다.

---

## 1. 문제 정의

### 1.1 배경

HuggingFace에 공개된 채팅 템플릿은 `transformers` 기준으로 작성된다. 거기서는 호출자가 도구를 주지 않으면 `apply_chat_template`이 `tools=None`을 넘긴다. llama-server는 경로는 다르지만 같은 상태에 도달한다. `common_chat_tools_to_json_oaicompat`이 빈 도구 배열에 대해 null JSON 값을 돌려주고, minja의 `chat-template.hpp`는 그 값이 non-null일 때만 컨텍스트 키를 설정한다. 두 엔진 모두 도구 없는 요청을 `tools`가 없거나 `None`인 컨텍스트에서 렌더하고, 템플릿 작성자는 그 전제 위에 가드를 쓴다.

mlxcel은 그렇지 않았다. `src/server/chat_template.rs`는 `tools`를 무조건 넣었고, 두 내부 렌더 함수가 `None`을 `minijinja::Value::from_serialize(Vec::<Tool>::new())`로 바꿨다. 두 호출 지점 옆의 주석은 이 빈 리스트를 `{% if tools is iterable and tools | length > 0 %}`에 대한 우회책으로 설명하면서, minijinja가 단축 평가에도 불구하고 "`none`의 `| length`를 계산하려 든다"고 적어 두었다. 절반만 맞았고 결론은 틀렸다. minijinja 2.24는 `and`를 실제로 단축 평가(`BinOpKind::ScAnd`)로 컴파일하지만, `iterable` 테스트가 `Value::try_iter().is_ok()`라 `none`에 대해 성공하므로 우변에 어차피 도달한다. 이 우회책은 그 가드 하나를 고치는 대신 모든 `is defined` / `is not none` 가드를 망가뜨렸다.

### 1.2 측정된 영향

`{"messages":[{"role":"user","content":"The Fibonacci sequence begins with"}]}`를 `tools` 키 없이 `POST /apply-template`에 보낸 결과다.

| 체크포인트 | 가드 형태 | 수정 전 렌더 | 오라클 (transformers / `tools` 미정의 jinja2) |
|---|---|---|---|
| `models/mlx/youtu-llm-2b-4bit` | `tools is defined and tools is not none` | 76 토큰, "available:"과 "For tool call returns" 사이에 빈 `<\|begin_of_tool_description\|>` 블록 | 8 토큰 |
| `models/mlx/llama-3.2-1b-4bit` | `{%- if not tools is defined %}{%- set tools = none %}{%- endif %}` 이후 `tools is not none` | 97 토큰, `Environment: ipython`과 "Given the following functions" 사용자 프리앰블 포함 | 40 토큰 |

각 체크포인트의 템플릿을 Python jinja2로 직접 렌더했을 때 mlxcel의 긴 출력과 바이트 단위로 일치한 것은 `tools=[]`를 넘겼을 때뿐이었다. 다른 컨텍스트 차이가 아니라 빈 리스트가 원인임을 직접 확인해 준 지점이다. 요청 본문에 `"tools": []`를 명시해도 같은 유령 블록이 나왔는데, `effective_tools`가 빈 슬라이스를 그대로 넘기기 때문이다.

### 1.3 결과

- **프롬프트 팽창.** 한 줄짜리 메시지에서도 프리앰블이 Youtu에서 68토큰, Llama 3.2에서 57토큰을 매 요청마다 잡아먹는다.
- **잘못된 유도.** 호출할 함수가 하나도 없는데 모델에게 함수를 호출해도 된다고 알려 주고 호출 문법까지 준다. Youtu의 한 프롬프트에서는 12토큰 뒤, 다른 프롬프트에서는 5토큰 뒤에 그리디 출력이 mlx-lm 오라클과 갈라졌다. 8토큰짜리 오라클 프롬프트를 raw `/completion`에 넣으면 32/32로 일치했다.
- **캐시 오염.** 프리앰블은 해당 모델의 모든 프롬프트 캐시 키의 `tokens[..prefix_len]`과 캐시가 스냅샷하는 히스토리 경계 렌더에 그대로 들어간다. 즉 영향받는 모델의 모든 엔트리가 모델이 애초에 보면 안 되는 프롬프트로 키잉돼 있었다.
- **조용하다.** 렌더된 프롬프트는 자연스럽고 요청은 성공하므로 로그나 응답 형태에서 아무 신호도 나오지 않는다. 외부 오라클과 비교하거나 `usage.prompt_tokens`를 봐야만 보인다.

---

## 2. 기술적 검토 사항

### 2.1 근본 원인

한 줄이다. `build_template_context`의 `ctx.insert("tools", tools);`와 그 앞의 `match tools { None => Vec::<Tool>::new() }` 두 곳. 로컬 코퍼스의 일곱 가드 계열 중 셋은 키의 *값*이 아니라 *정의 여부*로 분기하는데, 코드에는 "없음"을 표현할 방법이 없었다.

### 2.2 가드 형태 코퍼스

고정된 minijinja 버전으로 `tools`의 각 후보 값에 대해 로컬 템플릿을 조사하면 이 수정이 근거로 삼은 판단표가 나온다.

| 가드 형태 | 대표 체크포인트 | `[]` (수정 전) | `none` | 미정의 (수정 후) |
|---|---|---|---|---|
| `tools is defined and tools is not none` | youtu-llm-2b-4bit | 유령 블록 | ok | ok |
| 미정의일 때 `set tools = none`, 이후 `tools is not none` | meta-llama-3.1-8b-instruct-4bit, llama-3.2-1b-4bit, llama-4-scout-17b-4bit | 유령 블록 | ok | ok |
| 미정의일 때 `set tools = []`, 이후 `tools is iterable and tools \| length > 0` | nemotron-3-nano-omni-30b-a3b-reasoning-4bit, nemotron-h-30b-4bit, mimo-v2-flash-4bit | ok | 렌더 오류 | ok |
| `not tools is defined or tools is none`에서 `[]` 설정, 이후 같은 iterable 가드 | seed-oss-36b-instruct-4bit | ok | ok | ok |
| `if tools` / `tools and tools is iterable and tools is not mapping` | Qwen 2.5 / 3 / 3.5 이후 | ok | ok | ok |
| `tools is defined and tools is not none and tools\|length > 0` | ministral-3b-4bit, mistral-small-4-119b-2603-4bit | ok | ok | ok |
| `tools is defined and tools` | apertus-8b-instruct-2509-4bit, exaone4-1.2b-4bit | ok | ok | ok |

모든 행에서 깨끗한 열은 미정의뿐이다. `none` 열이 있어서 이 수정이 빈 벡터를 `Value::from(())`로 바꾸는 한 글자짜리 변경이 될 수 없다.

### 2.3 호환성/의존성 관점

의존성 변경은 없다. `src/server/chat_request.rs`의 `effective_tools`는 손대지 않았고, 덕분에 같은 파일을 수정 중인 tool_choice 작업(PR #1581)과도 충돌하지 않는다. `tool_choice: "none"`은 예전과 동일한 경로로 템플릿에 "도구 없음"으로 도달한다.

### 2.4 코드 품질 관점

빈 리스트 우회책을 minijinja 단축 평가 탓으로 잘못 돌린 두 주석은 실제 `iterable` / `length` 의미론, 세 가드 형태, 이슈 링크로 교체했다. `build_template_context`에는 어떤 키를 무조건 넣어도 되는지 판단하는 불변식 주석이 추가돼서, 다음에 이 블록에 키를 추가하는 사람이 선례를 복사하는 대신 규칙에 비춰 확인할 수 있다.

---

## 3. 기술적 선택과 그 이유

### 3.1 빈 리스트도 `none`도 아닌 미정의

이 선택은 취향이 아니라 코퍼스가 강제한 것이다. 빈 리스트는 `is defined` 계열 둘을 깨뜨린다. `none`은 Nemotron / MiMo 계열을 깨뜨리는데, 하필 가장 나쁜 방식으로 깨뜨린다. `render_simple_fallback`이 렌더 오류를 잡아 일반적인 `User:` / `Assistant:` 프롬프트로 대체하므로, 오류가 아니라 품질 저하로만 드러난다. 일곱 행 모두를 의도된 경로에 남겨 두는 값은 미정의뿐이고, 이는 두 참조 구현이 만들어 내는 상태이기도 하다.

남는 위험 하나는 짚어 둘 만하다. 정의 여부 가드 없이 `tools | length`를 계산하는 템플릿이 있다면 이제 오류가 난다. 그런 템플릿은 애초에 배포될 수 없다. 같은 이유로 transformers의 `tools=None`에서도 예외가 나기 때문이다. 그리고 `models/` 아래 196개 채팅 템플릿을 정적 감사한 결과, `tools | length`를 계산하는 8개 템플릿은 모두 그 앞에 가드를 두고 있다.

### 3.2 명시적 빈 리스트는 도구 없음이다

`"tools": []`는 정의된 빈 리스트로 통과시키지 않고 미정의로 정규화한다. llama-server의 규칙(컨텍스트를 만들기 전에 빈 배열이 null JSON 값이 된다)이자 OpenAI에서의 해석이다. 그대로 통과시키면 키를 항상 붙여 보내는 클라이언트, 즉 흔한 SDK 형태에 대해 버그가 그대로 남는다.

### 3.3 호출자가 아니라 두 내부 함수에서 정규화

트리의 모든 렌더는 `apply_raw_inner` 아니면 `apply_inner`를 통과한다. chat / messages / responses / apply-template와 두 input_tokens 라우트, Muse ATEM 스트림 경로, 프롬프트 캐시의 히스토리 경계 렌더, 오프라인 `generate` / `chat` 명령이 전부 여기로 모인다. 이 병목에서 정규화하면 전부를 한 곳에서 덮고, 규칙을 잊을 수 있는 호출자가 남지 않는다. 대안인 "각 호출자가 부르는 공용 헬퍼"는 병목이 없앤 호출자별 어긋남을 그대로 되살린다.

### 3.4 `enable_thinking`은 정의된 채로 둔다

이슈가 요구한 감사는 빌더가 넣는 모든 키를 대상으로 했다. 원리상 템플릿이 `is defined`로 검사할 수 있는 다른 무조건 키는 `enable_thinking`이고, 실제로 Youtu 템플릿이 그렇게 읽는다(`enable_thinking is defined and enable_thinking is false`). 그래도 무조건 유지한다. 그 정의 여부 자체가 #686과 #1114에서 테스트로 굳힌 요청별 오버라이드 계약이고, `enable_thinking is defined`만으로 transformers가 타지 않을 분기를 타는 배포 템플릿은 발견되지 않았다. 여기까지 수정을 넓히면 측정된 결함 하나를 테스트가 걸린 계약에 대한 측정되지 않은 변경과 맞바꾸는 셈이 된다.

### 3.5 `tools`는 `RESERVED_KEYS`에 남는다

"이 요청이 어떤 도구를 갖고 있는가"에 대한 서버 관리 답이 빈 도구 집합이므로, `chat_template_kwargs` 항목이 요청이 일부러 비워 둔 키를 정의할 수 있어서는 안 된다. minijinja의 기본 lenient 미정의 모드에서 `test_kwargs_cannot_override_reserved_tools_key`는 수정 없이 계속 통과한다. 미정의 `tools`를 순회하면 아무것도 나오지 않는데, 이는 빈 리스트가 만들던 관측 결과와 동일하다.

---

## 4. 구현 상세

### 4.1 컨텍스트 빌더

```rust
// 수정 전
    tools: minijinja::Value,
    ...
    ctx.insert("tools", tools);

// 수정 후
    tools: Option<minijinja::Value>,
    ...
    if let Some(tools) = tools {
        ctx.insert("tools", tools);
    }
```

### 4.2 두 렌더 경로

```rust
// 수정 전, apply_raw_inner와 apply_inner 양쪽
let tools_val = match tools {
    Some(t) => minijinja::Value::from_serialize(t),
    None => minijinja::Value::from_serialize(Vec::<Tool>::new()),
};

// 수정 후
let tools_val = tools
    .filter(|t| !t.is_empty())
    .map(minijinja::Value::from_serialize);
```

새로 컴파일한 환경에서 프로덕션 컨텍스트를 재현하려고 존재하는 테스트 모듈의 렌더 헬퍼도 같은 표현식을 쓰며, 동일하게 유지해야 한다는 주석을 달았다.

### 4.3 불변식 주석

`build_template_context`의 insert 블록에 조건부/무조건을 가르는 규칙을 적었다. 배포 템플릿이 키의 정의 여부만으로 분기할 수 없을 때에만 무조건 넣는다는 것이다. 각 키와 판단 근거를 함께 적어서, 나중에 추가되는 키는 복사가 아니라 검토를 거치게 했다.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib server::chat_template` | 154 통과, 3 무시 |
| `cargo test ... --lib server::chat_request` | 104 통과 |
| `cargo test ... --lib server::routes` / `server::tool` / `server::prompt_cache` | 387 / 384 / 173 통과 |
| `cargo test ... --lib server::reasoning_effort_tests` / `server::anthropic_translator` / `server::muse_atem_roundtrip_tests` / `server::muse_glimmer_template_tests` | 28 / 33 / 5 / 6 통과 |
| `cargo test ... --lib server::chat_template::tests::local_checkpoint_templates_render -- --ignored` | 통과. 두 체크포인트 모두 transformers 오라클과 바이트 단위 일치 |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | 클린 |
| `cargo fmt --all -- --check` | 클린 |
| `python3 scripts/ci/check_cross_repo_refs.py` | 클린 |
| 로컬 채팅 템플릿 196개 정적 감사 | `tools \| length`를 계산하는 8개 모두 앞에 정의 여부/none 검사를 둠 |
| `cargo test --test apply_template_tools_parity -- --ignored` (실제 가중치, HTTP) | 대기, 중앙에서 실행 |

### 5.1 가중치 없는 게이트가 잡아낸 것

`ChatTemplateProcessor::from_model_path`는 템플릿 파일만 읽으므로, 두 체크포인트를 가중치 없이, GPU 없이, 서버 없이 렌더해서 오라클과 비교할 수 있다. 이 테스트는 크기에 비해 값어치가 크다. 첫 버전은 서버가 만들지 않는 프롬프트를 단언하고 있었는데, Youtu 템플릿이 `enable_thinking`이 정의돼 있고 false일 때 빈 `<think></think>` 블록을 미리 채우고, `server::startup`은 토크나이저의 think 마커에서 그 기본값을 true로 설정하기 때문이었다. 지금은 테스트가 체크포인트별로 그 유도를 그대로 반영한다. 이 테스트가 없었다면 이 불일치는 실제 체크포인트 HTTP 게이트에서야 드러났을 것이다.

### 5.2 추가된 테스트

체크포인트 없이 도는 단위 테스트 넷이 세 가드 형태와 히스토리 경계 렌더를, 타입 경로와 raw JSON 경로 양쪽에서, `None` / `Some(&[])` / 실제 도구 하나에 대해 덮는다. `tests/apply_template_tools_parity.rs`는 실제 가중치로 HTTP를 통해 정확한 프롬프트 문자열과 8 / 40 토큰 수를 단언하고, 도구를 하나 실은 요청은 두 체크포인트 모두에서 여전히 도구 블록을 렌더하는지 확인한다. 두 서버는 한 테스트 안에서 순차로 띄운다. 스위트가 Metal 장치 하나를 공유하고, 체크포인트 둘이 동시에 상주하는 것이 실행을 중단시키는 형태이기 때문이다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 추가 줄 | 557 |
| 삭제 줄 | 27 |
| 커밋 | 2 |

### 카테고리별 변경

| 파일 | 변경 |
|---|---|
| `src/server/chat_template.rs` | `build_template_context`가 `Option<minijinja::Value>`를 받아 조건부 삽입, 두 내부 렌더 함수에서 빈 값을 `None`으로 정규화, 낡은 주석 교체, 불변식 주석 추가, 테스트 5개 추가 |
| `tests/apply_template_tools_parity.rs` | 두 체크포인트를 대상으로 하는 게이트된 HTTP 일치 테스트 신규 |
| `docs/llama-server-compat.md` | "Chat templates, reasoning, and output parsing" 아래 하위 절 신규 |

### 관련 커밋

- `7d5d80d42` fix(server): leave tools undefined when a request sends none
- `4d66e8c1d` test(server): gate the tools-less prompt without loading weights

### 관련 PR/이슈

- Closes #1597
- 인접하지만 일부러 건드리지 않음: #1581 (tool_choice 강제, 이슈 #1319). `src/server/chat_request.rs`를 수정 중이다
- 이 수정이 계약을 그대로 둔 컨텍스트 키들: #686, #1114 (`enable_thinking`), #512 (`thinking`), #775, #819 (`thinking_mode`)

---

## 7. 후속 조치

### 7.1 운영 참고

영향받는 모델은 렌더된 접두어가 바뀌므로, 업그레이드 후 모델당 첫 요청이 프롬프트 캐시에서 한 번 미스가 나고 다시 만든다. 호환성 파괴가 아니라 콜드 미스 한 번이다. 키 다이제스트 버전은 바뀌지 않고 마이그레이션도 필요 없다.

### 7.2 범위 밖, 필요한 체크포인트가 나오면 별도 등록

- minijinja의 `iterable` 테스트와 `length` 필터를 `none` / 미정의에 대해 Python과 맞추는 것. 미정의 `tools`를 `none`으로 기본값 설정한 뒤 iterable 가드를 쓰는 템플릿은 여전히 오류가 나고 폴백된다. 로컬 코퍼스에는 그런 템플릿이 없다.
- `enable_thinking`의 정의 여부. 구체적인 템플릿 불일치가 발견될 때만.
- 수정 이후 Youtu 두 번째 프롬프트에 남는 bf16 동점 드리프트. 이 결함과는 무관하다.

### 7.3 더 넓은 교훈

이 버그는 근거를 확인할 수 있는데도 확인하지 않은 우회책이었다. 주석은 고정된 버전에 두 줄짜리 탐침을 돌리면 반증되는 minijinja 동작("단축 평가에도 `none`의 `| length`를 계산하려 든다")을 단언했고, 그 주석이 정당화한 수정은 깨진 가드 계열 하나를 둘로 바꿨다. 주석이 값 선택을 엔진 동작으로 설명한다면, 가장 싼 수는 엔진을 직접 돌려 보는 것이다.

두 번째 교훈은 게이트를 어디에 두느냐다. 이 두 체크포인트의 바이트 단위 프롬프트는 템플릿 파일이 입력의 전부이므로 가중치 없이, GPU 없이, 서버 없이 검증할 수 있다. 그 게이트를 단위 테스트 층에 두니 단언이 밀리초 단위로 돌았고, 실제 체크포인트를 돌리기 전에 잘못된 기대치를 잡아냈다.
