# 기술 보고서: 채팅 템플릿 컴파일 캐시

## 요약

PR #1518은 issue #1235를 해결하기 위해 각 `ChatTemplateProcessor`에 컴파일된 MiniJinja 채팅 템플릿 환경을 캐시한다. 변경 전에는 typed 렌더, raw JSON 렌더, history 렌더, 프롬프트 캐시 probe 렌더가 모두 매 호출마다 환경을 새로 만들고 필터, 함수, 메서드 콜백을 다시 등록한 뒤 불변 템플릿 소스를 다시 컴파일했다.

이번 구현은 컴파일을 프로세서 소유 `OnceLock` 뒤로 옮기고, MiniJinja의 owned-template API를 사용해 `Environment<'static>`을 저장한다. 요청별 값은 캐시에 들어가지 않는다. 메시지, 도구, kwargs, generation-prompt 모드, thinking alias는 여전히 렌더마다 새로 구성된다.

## 문제

로드된 템플릿 문자열은 프로세서 생명주기 동안 불변이지만, `apply_inner`와 `apply_raw_inner`는 매 호출마다 새 MiniJinja 환경을 만들고 같은 `"chat"` 템플릿을 추가했다. 하나의 chat-completions 요청도 primary prompt, prompt-cache history boundary, next-turn warm-up probe 때문에 렌더러를 여러 번 호출할 수 있다.

그 결과 현대적인 5-20 KB 채팅 템플릿은 호출마다 parse/compile 비용과 필터 등록 비용을 반복해서 지불했다. 실제로 바뀌는 것은 렌더 컨텍스트뿐이었다.

## 구현

- `ChatTemplateProcessor`에 `compiled_template` 캐시를 추가하고, `Arc<OnceLock<_>>`로 clone 간에 공유했다.
- MiniJinja 환경을 한 번 설정하고 `add_template_owned`로 템플릿을 삽입하는 `compile_chat_template_environment`를 추가했다.
- raw와 typed 렌더 경로의 중복 환경 생성을 `render_template` 호출로 대체해 `apply_inner`, `apply_raw_inner`, history 렌더, probe 렌더가 같은 컴파일 결과를 공유하게 했다.
- 불변 템플릿의 parse failure도 캐시해 malformed template이 fallback 시도마다 재파싱되지 않게 했다.
- `raise_exception`은 캐시된 환경의 렌더 시점 callable로 유지하고 요청 컨텍스트는 매번 새로 만들기 때문에 `TemplateRejection` 동작을 보존했다.

## 정확성

캐시는 운영자 또는 모델이 제공한 템플릿과 MiniJinja 환경 설정만 보관한다. 요청 데이터, 도구 선언, kwargs, generation-prompt 모드는 보관하지 않는다. 따라서 요청 격리는 유지하면서 bytecode와 callable 등록만 재사용한다.

테스트 전용 compile counter는 전역이 아니라 프로세서별로 저장된다. 그래서 병렬 테스트 실행 순서에 의존하지 않고 typed/raw 경로가 하나의 compile을 공유하는지, clone이 같은 캐시를 재사용하는지, malformed template의 parse failure가 캐시되는지, 동시 렌더 race가 하나의 compile로 수렴하는지를 검증한다.

## 호환성

집중 byte-identity 테스트는 변경 전 동작을 모사하는 fresh-environment oracle과 캐시 렌더 결과를 직접 비교한다. 도구와 kwargs가 있는 typed render, typed history render, raw JSON render, raw JSON history render를 모두 포함한다. 기존 `server::chat_template::tests`도 통과해 기존 fixture 기반 동작을 보존했다.

#1176의 rejection/fallback 분기도 유지된다. 템플릿이 `raise_exception`으로 거부한 경우는 계속 `TemplateRejection` sentinel을 노출하고, engine failure와 parse failure는 rejection으로 분류되지 않아 요청 계층의 fallback 후보로 남는다.

## 성능 근거

400회 렌더를 수행한 제한된 debug-profile smoke run 결과는 다음과 같다.

- Fresh environment/render loop: 27.248578 ms
- Cached environment/render loop: 7.99117 ms

이는 로컬 단위 측정이며 production throughput benchmark가 아니다. 컴파일/렌더 경계에서 기대한 방향과 대략적인 크기의 개선을 확인하기 위한 근거로만 사용한다.

## 검증

- `cargo fmt` 통과.
- `cargo test --lib cached_template` 통과: 6 passed, 1 ignored.
- `cargo test --lib server::chat_template::tests` 통과: 90 passed, 2 ignored.
- `cargo test --lib server::reasoning_effort_tests` 통과: 28 passed.
- `cargo test --lib history_render` 통과: 5 passed.
- `cargo test --lib cached_template_performance_smoke_reports_delta -- --ignored --nocapture` 통과, 위 timing evidence 산출.
- `cargo clippy --lib --tests -- -D warnings` 통과.
- `git diff --check` 통과.

## 생략한 검증

Wave-runner watchdog guard에 따라 broad cargo workspace tests, serial all-tests, workspace clippy, cold release builds는 의도적으로 실행하지 않았다. 로컬 `models/` 디렉터리가 필요한 model-template audit 테스트는 ignored 상태로 유지했다.

## 위험 및 참고 사항

캐시된 환경은 clone된 프로세서 간에 공유된다. 이는 clone이 같은 불변 템플릿 생명주기를 나타낸다는 전제에서 의도한 동작이다. 향후 프로세서에서 템플릿 소스를 변경하는 코드가 생기면 새 소스와 함께 새 캐시를 할당해야 한다.

Malformed-template 오류는 이제 매 호출마다 새 MiniJinja error를 만들지 않고 캐시된 내부 오류 wrapper를 통해 노출된다. 사용자에게 보이는 parse context는 유지되고 rejection discriminator는 계속 false다. 현재 테스트와 요청 동작 중 parse error를 `minijinja::Error`로 downcast해야 하는 경로는 없다.
