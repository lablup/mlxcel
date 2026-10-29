# PR #2724: subscriber 설치 전 INFO notice 노출과 신규 호출 방지 가드

**작성일**: 2026-10-28 (머지 전 작성, PR은 draft 상태)
**상태**: 3f816e06에서 구현 완료. 구현 리뷰에서 CRITICAL/HIGH 항목 없음. 필터링한 로컬 테스트 통과(`cargo test --lib startup_notice`: 6개 통과). 작성 시점에 `make verify-rocm-held`와 필터 없는 `cargo test --bin mlxcel`은 아직 실행 전이었고, hosted CI는 일부만 끝난 상태(cargo-fmt, cargo-deny 통과, cargo-clippy와 링크 잡은 미완료)
**언어**: Rust
**위험도**: 낮음 (결정되는 설정 값은 바뀌지 않음. 바뀌는 것은 어떤 줄이 로그에 남는지, 어떤 레벨로 남는지, 그리고 공개 `env_fallback_*` 헬퍼 십여 개와 `GrammarSourceArg::load`의 시그니처)

## 요약

PR #2724(#2722 해결)는 #2704가 시작한 작업을 마무리한다. #2704는 서버 시작 경로에서 tracing subscriber 설치 전에 발생한 경고를 `startup_notices: Vec<String>`에 모았다가 `start_server`가 tracing을 설치한 직후 WARN으로 출력하게 했다. 두 가지 빈틈이 남아 있었다. "CLI 플래그가 환경 변수보다 우선한다"는 줄은 여전히 subscriber가 없는 시점의 `tracing::info!` 호출이라 어디에도 기록되지 않았다. 그리고 이 경로에 새 `tracing::` 호출이 추가되어 같은 식으로 조용히 사라지는 것을 막는 장치가 없었다.

이 PR은 문자열 목록을 레벨을 가진 `StartupNotice { level, message }`(Info, Warn, Error)로 바꾸고, 서버 시작 경로의 subscriber 이전 로그 호출을 모두 원래 레벨의 notice로 전환하며, 시작 루트에서 호출 그래프를 따라가 로깅 매크로를 만나면 실패하는 소스 스캔 테스트를 추가한다. 이 가드는 이슈에 없던 누락 세 건(두 바이너리의 superseded-model INFO 줄과 `--grammar` 파싱 오류)을 추가로 찾아냈고, 이것들도 전환했다. 또한 KV 캐시 모드 우선순위 경고의 출력 지점을 하나로 정리했다. `resolve_kv_cache_mode`는 더 이상 직접 로그를 남기지 않고, subscriber가 있는 시점에 실행되는 호출자가 명시적으로 출력한다.

## 1. 문제 정의

### 1.1 우선순위 안내 줄이 사라짐

두 서버 바이너리(`src/bin/mlx_server.rs`의 `mlxcel-server`, `src/commands/serve.rs`의 `mlxcel serve`)는 `start_server`가 `initialize_server_logging`을 호출하기 전에 `build_startup_input`과 `ServerStartupInput::into_startup_config`에서 플래그와 환경 변수를 처리한다. `mlxcel serve`는 CLI 자체의 `init_cli_tracing`(`src/main.rs`)에서 명시적으로 제외되는데, `server::startup`이 자체 subscriber를 설치하고 두 번째 `init()`은 panic을 일으키기 때문이다. 따라서 이 경로의 `tracing` 매크로는 받아 줄 subscriber가 없다.

#2704는 경고만 옮기고 INFO 줄은 그대로 두었다. 이슈 기준으로 `src/server/cli_input.rs`에 우선순위 `tracing::info!`가 14개, `src/cli/turbo_args.rs`에 1개, `src/cli/speculative_args.rs`에 2개 있었고, `thinking_budget.rs`와 `chat_template_kwargs.rs`에도 더 있었다. `--cache-type-k`와 `LLAMA_ARG_CACHE_TYPE_K`를 서로 다른 값으로 함께 설정한 운영자는 어느 쪽이 적용됐는지 알 방법이 없었다.

### 1.2 회귀 방지 장치 부재

#2704는 발견한 경고를 손으로 고쳤다. 버그를 만든 패턴(env fallback 헬퍼에서 `tracing::warn!` 호출, 겉보기에 맞고 컴파일도 됨)은 여전히 가장 자연스럽게 쓰게 되는 코드였고, #2704의 테스트는 이미 알고 있는 notice만 확인했다.

### 1.3 KV 캐시 우선순위 경고 중복

`resolve_kv_cache_mode`는 "분리된 `--cache-type-k/v` 플래그가 레거시 `--kv-cache-mode`보다 우선한다"는 경고를 직접 로그로 남겼고, `into_startup_config`도 같은 내용을 notice로 넣었다. 서버는 `resolve_kv_cache_mode`를 최대 세 번(모델 다운로드 전 조기 검증, 메모리 preflight, `into_startup_config`), `generate`는 두 번(preflight와 실제 실행) 호출한다. 서버에서는 직접 로그가 모두 사라졌고, CLI에서는 여러 번 출력될 수 있었다.

## 2. 변경 요약

### 2.1 `StartupNotice`

새 모듈 `src/server/startup_notice.rs`를 추가하고 `mlxcel::server`에서 re-export한다.

- `StartupNoticeLevel::{Info, Warn, Error}`와 `StartupNotice { level, message }`, 생성자 `info`/`warn`/`error`, `contains`, 메시지만 출력하는 `Display`.
- `ServerStartupInput`, `ServerStartupConfig`, `ServeArgs`는 `Vec<String>` 대신 `Vec<StartupNotice>`를 가진다.
- `log_startup_notices`(`src/server/startup.rs`)는 레벨에 따라 `tracing::info!`, `warn!`, `error!`를 호출한다. 호출 위치는 그대로다. `initialize_server_logging` 직후, router 모드 분기 이전이라 router 서버와 단일 모델 서버 모두 출력한다.

기존 `notices.push(String)`는 모두 `StartupNotice::warn(...)`이 되어 #2704의 경고는 WARN 레벨을 유지한다.

### 2.2 우선순위 줄을 INFO notice로

`cli_input.rs`, `turbo_args.rs`, `speculative_args.rs`, `thinking_budget.rs`, `chat_template_kwargs.rs`의 우선순위 `tracing::info!`는 모두 같은 문구로 `StartupNotice::info(...)`를 넣는다. 우선순위 줄만 있던 헬퍼는 notice 목록이 없었으므로 `notices: &mut Vec<StartupNotice>` 인자를 새로 받는다. 대상은 `env_fallback_cache_type_k/v`, `kv_quant_scheme`, `apc_hash`, `draft_model`, `log_file`, `embedding_model`, `reranker_model`, `lang_bias`, `draft_kind`, `chat_template_kwargs`이며, 두 바이너리 모두 자신의 목록을 넘긴다.

### 2.3 가드가 찾아낸 줄

- `mlx_server.rs`의 `build_startup_input`과 `serve.rs`의 `run_serve_async`에 있던 superseded-model INFO 줄(`superseded_model_notice`)을 INFO notice로 넣는다.
- `into_startup_config`가 호출하는 `GrammarSourceArg::load`(`src/cli/grammar_args.rs`)는 파싱되지 않는 GBNF 문법에 대해 `tracing::error!`를 썼다. 이제 `notices`를 받아 `unparseable_gbnf_notice`가 만든 ERROR notice를 넣는다. 알리는 동작 자체는 같다. llama.cpp b10621과 마찬가지로 서버는 시작되고, 제약이 걸린 요청은 모두 파서의 진단 메시지와 함께 거부된다. 모델 어휘가 필요한 `<token-text>` 종단에서만 실패하는 문법은 notice를 만들지 않는다.

### 2.4 소스 스캔 가드

`src/server/startup_notice_tests.rs`(522줄, 대부분이 스캐너)에 다음을 추가한다.

- `startup_notices_guard_finds_no_tracing_before_the_subscriber`. `cli_input.rs`, 모든 `src/cli/*_args.rs`, `thinking_budget.rs`, `chat_template_kwargs.rs`, `model_source.rs`, `bin/mlx_server.rs`, `commands/serve.rs`를 읽는다. 루트는 이름이 `env_fallback_*`, `into_startup_config`, `build_startup_input`인 모든 함수다. 너비 우선 탐색으로 함수 본문에서 이 파일들에 정의된 함수 이름을 만나면 따라가고, 도달한 `trace!`/`debug!`/`info!`/`warn!`/`error!`/`event!`를 파일, 줄, 함수, 출발 루트와 함께 보고한다. 기대하는 각 파일에서 루트가 발견됐는지, `build_startup_input` 루트가 정확히 두 개인지도 확인하므로 루트 이름을 바꿔 스캔 대상이 조용히 비는 일을 막는다.
- `startup_notices_guard_flags_a_tracing_call_in_an_env_fallback_helper`. 실제 `env_fallback_kv_bits` 소스에 `tracing::warn!`을 끼워 넣고 스캔이 이를 보고하는지 확인한다. 이슈의 인수 조건을 손으로 한 번 확인하는 대신 매 테스트 실행마다 확인하는 셈이다.
- 전이 호출, 주석과 문자열 비우기, `#[cfg(test)]` 모듈 제외를 고정하는 합성 케이스.

스캔 전에 `code_only`가 주석(중첩 블록 주석 포함), 일반 및 raw 문자열 리터럴 내용, char 리터럴을 공백으로 바꾸되 오프셋과 줄바꿈은 유지한다. 그래서 doc 주석 속 매크로 이름이나 포맷 문자열 속 `{`가 중괄호 매칭을 망가뜨리거나 오탐을 만들지 않는다.

### 2.5 KV 캐시 우선순위 경고의 단일 출력 지점

- `resolve_kv_cache_mode`는 더 이상 로그를 남기지 않는다.
- `turbo_args.rs`에 `log_kv_cache_mode_precedence_notice`를 추가했다. `kv_cache_mode_precedence_notice`를 `tracing::warn!`으로 출력하며, `generate`(preflight가 아닌 실제 실행), `inspect`, 프로세스 내 CLI 서버(`resolve_cli_kv_cache_mode`)가 한 번씩 호출한다.
- `bench_decode`는 subscriber가 없으므로 기존 모드 치환 안내 옆에서 `eprintln!`으로 출력한다.
- 서버는 `into_startup_config`의 notice 하나만 유지하고, 조기 검증과 preflight에서의 `resolve_kv_cache_mode` 호출은 아무것도 출력하지 않는다.

### 2.6 테스트

- `startup_notices_are_logged_once_a_subscriber_exists`(`src/server/startup_tests.rs`)는 21개 우선순위 케이스마다 플래그와 다른 값의 환경 변수를 설정하고 subscriber 없이 헬퍼를 실행한 뒤, 캡처용 subscriber를 설치하고 `log_startup_notices`를 호출한다. 각 우선순위 줄이 INFO로 정확히 한 번, 기존 경고는 WARN 한 줄, grammar 오류는 ERROR 한 줄인지 확인한다. 이슈의 인수 조건상 이 테스트는 main에서 실패해야 하며, main에서는 우선순위 줄이 캡처되지 않는다.
- `startup_notices_reach_the_startup_config`는 `into_startup_config`를 거쳐도 순서와 레벨이 유지되는지 확인한다.
- grammar 테스트는 잘못된 GBNF에서 ERROR notice가 하나, token 종단에서는 없음을 확인한다.
- 기존 테스트(`cli_input_tests.rs`, `speculative_args_tests.rs`, `chat_template_kwargs_tests.rs`, `grammar_args_tests.rs`, `main_tests.rs`)의 호출부를 새 시그니처에 맞췄다.

## 3. 기술적 선택과 그 이유

### 3.1 별도 목록 대신 notice별 레벨

경고 목록 옆에 `startup_info_notices`를 따로 두는 방법도 있었다. 그러면 순서가 갈라지고(어느 값이 이겼는지 알리는 INFO와 무시된 값에 대한 WARN이 서로 다른 묶음으로 출력됨), 레벨이 늘 때마다 세 구조체에 필드를 추가해야 한다. 태그가 붙은 단일 목록은 원래 순서를 유지하고, `--grammar`의 ERROR 케이스도 추가 배선 없이 들어갔다.

### 3.2 subscriber 버퍼링이 아닌 수집

프로세스 시작 시 버퍼링 subscriber를 설치하고 나중에 실제 subscriber로 재생하는 방법도 있었다. 스캔 대상 파일 밖의 줄까지 모두 잡을 수 있지만, 바이너리가 재생 계층에 의존하게 되고, `initialize_server_logging`이 지금 만들고 있는 그 설정에서 읽는 `--log-file`과 로그 포맷 옵션과 얽히며, 이중 `init()` 문제도 여전히 신경 써야 한다. 명시적 목록은 데이터 흐름을 함수 시그니처에 드러낸다. 무언가를 알릴 수 있는 헬퍼는 `&mut Vec<StartupNotice>`를 받고, 컴파일러가 모든 호출자에게 이를 넘기도록 강제한다.

### 3.3 런타임 검사 대신 소스 스캔

런타임 테스트는 자신이 만든 입력만 다룰 수 있고, #2704의 빈틈이 살아남은 이유가 바로 그것이다. 소스 텍스트 스캔은 어떤 테스트도 실행하지 않는 분기까지 모든 헬퍼를 다룬다. 대신 `syn`이 아닌 휴리스틱 파서라서 호출 대상을 이름만으로 찾고, 모듈, 로깅으로 확장되는 매크로, `use` 별칭을 모른다. 설계는 과잉 보고 쪽으로 기울어 있고(스캔 파일 안에 같은 이름의 함수가 있으면 따라감), 자가 테스트와 "루트가 존재해야 함" 확인이 아무것도 스캔하지 않는 상태로 조용히 통과하는 것을 막는다.

### 3.4 `resolve_kv_cache_mode`를 순수 함수로

리졸버에서 부수 효과를 없애면 서버가 세 번 호출해도 로그가 세 번 나오지 않는다. 그리고 출력 위치(notice, `tracing`, stderr)를 정하는 일은 subscriber 존재 여부를 아는 호출자에게 넘어간다.

## 4. 남은 항목과 검증되지 않은 부분

- **스캔 범위.** 스캔 파일 밖의 호출 대상은 따라가지 않는다. `into_startup_config`는 다른 모듈의 코드(예: `crate::lora::multi`, `crate::lang_bias`)에 도달한다. 현재 그쪽에 subscriber 이전 `tracing` 호출은 없지만, 생겨도 가드는 잡지 못한다.
- **이름 기반 루트.** subscriber 이전에 실행되지만 `env_fallback_*`, `into_startup_config`, `build_startup_input` 루트 아래에 있지 않은 코드는 스캔되지 않는다. 3f816e06 기준 두 예가 있다. `serve.rs`의 `run_serve_async`(이 PR에서 해당 줄 하나는 손으로 전환했지만 함수 자체는 루트가 아님), 그리고 `mlx_server.rs`의 `main`으로, `start_server` 전에 `tracing::info!("autotune: applied {var}={value} from the tactic cache")`를 호출한다. 이 autotune 줄은 여전히 사라진다. #2722 범위 밖이고, `MLXCEL_AUTOTUNE`이 설정되고 튜닝 항목이 있을 때만 발생한다.
- **이름만으로 호출 대상 해석.** `load`나 `new` 같은 흔한 이름은 무관한 호출에서도 따라갈 수 있다. 오탐만 생길 뿐 누락은 생기지 않는다.
- **테스트 커버리지 공백.** 21개 케이스 인수 테스트는 `env_fallback_cache_type_k`는 실행하지만 `env_fallback_cache_type_v`는 실행하지 않는다. 둘 다 `apply_optional_string_env_fallback`을 거치므로 공백은 동작이 아니라 테스트에 있다.
- **CLI에서의 가시성.** `generate`와 `inspect`는 `RUST_LOG`가 설정됐을 때만 subscriber를 설치하므로(`init_cli_tracing`) KV 캐시 우선순위 경고는 기본 설정에서 여전히 보이지 않는다. PR 이전에도 마찬가지였다. 무조건 보이는 곳은 `bench_decode`(stderr)와 서버(startup notice)뿐이다.
- **API 변경.** 공개 `env_fallback_*` 헬퍼 십여 개와 `GrammarSourceArg::load`에 인자가 추가됐다. 트리 내 호출자는 모두 수정했고, `mlxcel::server`를 쓰는 외부 코드도 같은 수정이 필요하다.
- **남은 검증.** 작성 시점에 `make verify-rocm-held`, 필터 없는 `RUST_MIN_STACK=16777216 cargo test --bin mlxcel`, 나머지 hosted CI 잡은 끝나지 않았다.

## 5. 학습 포인트

- subscriber 설치 전에 호출한 로깅 매크로는 조용한 no-op이고, 호출 자체는 올바르게 보이므로 코드 리뷰에서 잘 걸리지 않는다. 시작 과정에 "로깅이 아직 없는" 단계가 있다면 그 단계에 명시적인 출력 채널을 주고 함수 시그니처의 일부로 만든다.
- 알려진 메시지를 확인하는 테스트는 수정이 맞음을 증명하고, 버그 패턴을 스캔하는 테스트는 다음 버그를 막는다. 이 스캔은 첫 실행에서 세 건을 더 찾았다.
- 텍스트 수준 소스 스캔에는 실제 소스에 실제 위반을 주입하는 자가 테스트와 루트 존재 확인이 필요하다. 그렇지 않으면 이름 변경이나 파서 버그 하나로 가드가 늘 통과하는 테스트가 된다.
- 시작 경로에서 여러 번 호출되는 리졸버 함수는 로그를 남기지 않는 편이 낫다. 출력을 책임지는 단일 호출자가 결정하게 한다.

## 6. 관련 항목

- 이슈 #2722. PR #2704(이슈 #2697)와 `TECHNICAL_REPORTS/2704-env-fallback-startup-warnings-guard-20261028.ko.md`의 후속.
- #2664(CORS origin 환경 변수), #1350(모델 계열별 KV 캐시 모드 치환), #1445(KV 캐시 타입 조기 검증).
