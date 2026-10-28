# PR #2704: env fallback 시작 경고 표시 및 CLI env 가드 공백 해소

**날짜**: 2026-10-28 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 구현 완료, ROCm (gfx1151)에서 로컬 테스트 통과, 작성 시점에 hosted CI는 대기 중
**언어**: Rust
**위험도**: 낮음 (해석되는 값은 바뀌지 않음, 경고가 출력되는 위치와 CLI 가드가 감지하는 범위만 바뀜)

## 요약

PR #2704 (#2697 해결)는 #2622와 #2668 이후 남아 있던, `mlxcel serve`와 `mlxcel-server`가 환경 변수 fallback을 읽는 방식의 세 가지 공백을 메웁니다.

1. `env_fallback_*` 헬퍼와 `ServerStartupInput::into_startup_config`가 내는 경고는 `start_server`가 tracing subscriber를 설치하기 전에 `tracing::warn!`으로 출력되어 아무 데도 기록되지 않았습니다. 이제 이 경고들을 하나의 `startup_notices: Vec<String>`에 모으고, `initialize_server_logging` 직후 WARN 레벨로 한 번 기록합니다. 기존의 임시 목록 두 개(#2664의 `cors_origin_notices`, #1350의 `kv_cache_mode_notices`)를 대체합니다.
2. clap이 충돌로 취급할 인자에 `env` 바인딩이 붙지 않도록 막는 `env_bound_override_args` 가드가 이제 인자 그룹과 `exclusive`에서 clap이 파생하는 충돌도 감지합니다.
3. CORS의 `--flag=value` 표기를 end to end로 테스트합니다. 이를 위해 `long_cli_flag_was_set`과 동일한 매칭 규칙을 적용하는 `long_cli_flag_in`을 새로 추가했습니다.

15개 파일, +1065 / -368, 커밋 두 개(`36e9b8e`, `07a4e40`)입니다.

## 1. 문제 정의

### 1.1 subscriber가 생기기 전에 기록된 경고

두 서버 바이너리는 서빙 전에 인자를 두 단계로 해석합니다. 먼저 바이너리가 `env_fallback_*` 헬퍼들을 호출하고(`src/bin/mlx_server.rs::build_startup_input`, `src/commands/serve.rs`), 이어서 `ServerStartupInput::into_startup_config`가 입력을 검증하고 변환합니다. 그 다음에야 `start_server`가 `initialize_server_logging`을 호출합니다. 앞 단계의 `tracing::warn!`은 subscriber가 없으므로 조용히 버려집니다.

운영자 입장에서는 `LLAMA_ARG_KV_BITS=eight`나 `LLAMA_ARG_ENDPOINT_SLOTS=maybe` 같은 오타가 로그에 아무 흔적 없이 무시되고 서버가 기본값으로 시작했다는 뜻입니다. `into_startup_config` 내부 경고(예: `--cache-ram`과 `--prompt-cache-capacity-bytes` 동시 지정, 범위를 벗어난 `--adaptive-decay`, API 키와 함께 쓴 `--api-prefix`)도 마찬가지였습니다.

이전에 두 번, 각각 한 가지 경우에 대해 이 문제를 고친 적이 있습니다. #1350은 KV 캐시 모드 대체를 위해 `kv_cache_mode_notices`를, #2664는 무시된 CORS origin 변수를 위해 `cors_origin_notices`를 추가했습니다. 둘은 각자의 필드와 각자의 로깅 지점을 가졌습니다.

### 1.2 파생된 충돌을 놓친 env 바인딩 가드

clap은 환경 변수에서 온 값도 "지정됨"으로 셉니다. `env = "VAR"`인 인자가 다른 인자와 충돌하면, 사용자가 하나만 입력했어도 `VAR=... --other`가 파싱 오류가 됩니다(#2609, #2664). `src/cli/mod.rs`의 `env_bound_override_args`가 이런 인자를 나열하고, 두 바이너리의 테스트가 그 목록이 비어 있음을 확인합니다. 이 PR 전에는 `overrides_with`와 `conflicts_with`만 보았습니다. clap의 validator는 `multiple`이 아닌 그룹, 그룹 자체의 `conflicts_with`, `exclusive(true)`에서도 충돌을 파생합니다. 이런 형태로 env 바인딩이 추가되면 가드를 통과한 채 파싱이 깨졌을 것입니다.

### 1.3 테스트되지 않은 CORS `--flag=value`

CORS 테스트 헬퍼는 `argv.contains(&"--cors-origins")`로 플래그 지정 여부를 판단했기 때문에, 바이너리는 `long_cli_flag_was_set`의 prefix 매칭으로 처리하는 `--cors-origins=http://a.example` 형태가 end to end로 검증된 적이 없었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| env fallback 헬퍼 (`src/server/cli_input.rs`, `src/cli/speculative_args.rs`, `src/server/thinking_budget.rs`) | 경고를 내던 모든 헬퍼가 `notices: &mut Vec<String>`을 받아 로깅 대신 메시지를 push합니다: KV bits, group size, skip-last-layer, endpoint slots, settings endpoint, CORS credentials, lang-bias byte fragments, prompt cache enabled / TTL / snapshot TTL, APC enabled, 모든 `Option<usize>` 헬퍼(`apply_optional_usize_env_fallback` 경유), reasoning budget, draft max, draft block size. |
| `into_startup_config` | `mut self`로 바뀌고, 들어온 목록을 `std::mem::take`로 가져온 뒤 자체 경고를 덧붙입니다: `--cache-ram` 우선순위, 잘못된 `--reasoning-budget`, `--timeout` 마이그레이션, API 키와 함께 쓴 `--api-prefix`, reasoning 플래그와 `--chat-template-kwargs` 충돌(`apply_reasoning_template_kwargs`), 범위를 벗어난 `--adaptive-decay`(`sanitize_adaptive_decay`), 분리형과 레거시 KV 캐시 플래그 우선순위, #1350의 KV 모드 대체. |
| KV 캐시 플래그 우선순위 (`src/cli/turbo_args.rs`) | 새 `kv_cache_mode_precedence_notice`가 메시지를 반환합니다. `resolve_kv_cache_mode`는 서버 외 호출자를 위해 여전히 직접 로깅하고, 서버는 같은 메시지를 notice로도 push합니다. |
| 데이터 모델 | `ServerStartupInput`, `ServerStartupConfig`, `ServeArgs`(`#[arg(skip)]`)의 `startup_notices: Vec<String>`이 `cors_origin_notices`와 `kv_cache_mode_notices`를 대체합니다. |
| 로깅 지점 (`src/server/startup.rs`) | `log_startup_notices`(`log_cors_origin_notices`에서 이름 변경)가 `initialize_server_logging` 직후, router 모드 분기 전에 모든 항목을 WARN으로 기록합니다. `start_server` 뒤쪽에 있던 #1350 전용 루프는 제거됐습니다. |
| `--spec-type none` | 두 바이너리 모두 "ignoring the configured draft model"과 "ignoring --draft-kind prompt-lookup" 경고를 목록에 push합니다. |
| CLI 가드 (`src/cli/mod.rs`) | `env_bound_override_args`가 먼저 command 복제본을 build합니다. `arg_conflict_partners`는 `direct_conflicts`(자체 충돌, 그룹 충돌, non-multiple 그룹 형제), `unroll_group_or_arg`(그룹 id를 중첩 그룹 포함 멤버 인자로 전개), `group_conflicts`(`Debug`에서 private 목록 읽기), 공용 `parse_debug_id_list`로 나뉘었습니다. exclusive 인자도 보고합니다. |
| 플래그 감지 (`src/server/cli_input.rs`) | `long_cli_flag_in(args, name)`이 `long_cli_flag_was_set`의 규칙을 노출하고, `long_cli_flag_was_set`은 이를 호출합니다. `mlxcel::server`에서 re-export됩니다. |
| 테스트 | 경고 경로 20개에 대한 end to end 캡처 테스트, `--spec-type none` notice 테스트, 가드 self-test 6개, `--flag=value` CORS 테스트 2개, 기존 테스트의 시그니처 갱신. |

## 3. 기술적 선택과 그 이유

### 3.1 반환값이나 전역 버퍼 대신 out-parameter

이슈는 notice를 반환하거나 push하는 방식 모두 허용했습니다. PR은 경고를 내는 모든 헬퍼에 `&mut Vec<String>`을 택했습니다. 대안은 세 가지였습니다.

- **헬퍼마다 `Option<String>` 반환.** #2664의 CORS 헬퍼가 이미 이 방식(`Result<Option<String>>` 반환)이고, 그대로 유지했습니다. 나머지에 대해서는 반환값이 `let _ =`나 수집 누락으로 쉽게 버려질 수 있고, 일부 헬퍼는 경고를 여러 개 낼 수 있습니다(`env_fallback_kv_skip_last_layer`는 변수 두 개를 시도하고, `apply_reasoning_template_kwargs`는 키마다 경고할 수 있음).
- **프로세스 전역 버퍼 또는 일찍 설치하는 버퍼링 subscriber.** 시그니처를 건드리지 않아도 되지만, 데이터 흐름이 숨겨지고, 병렬 테스트가 상태를 공유하게 되며, 초기 subscriber를 설정된 subscriber(포맷, 로그 파일, 레벨)로 교체해야 하는데 `initialize_server_logging`은 그렇게 설계되어 있지 않습니다.
- **out-parameter (채택).** 필수 인자를 추가하면 목록을 넘기기 전까지 모든 호출 지점이 컴파일 오류가 되므로, 어떤 호출자도 notice를 조용히 버릴 수 없습니다. 목록은 기존 input/config 구조체에 실려 새 배관 없이 `start_server`까지 전달됩니다.

비용은 `mlxcel::server`에서 re-export되는 public 함수들의 넓은 시그니처 변경입니다. 저장소 밖의 호출자는 수정이 필요합니다. 저장소 안의 모든 호출 지점은 갱신됐고, notice에 관심 없는 테스트는 `&mut Vec::new()`를 넘깁니다.

### 3.2 하나의 목록을 router 분기 전에 한 번 기록

`kv_cache_mode_notices`와 `cors_origin_notices`를 `startup_notices` 하나로 합쳐, 필드도 로깅 함수도 다음 경고를 추가할 위치도 하나가 됐습니다. `initialize_server_logging` 직후, router 모드 분기 전에 기록하므로 router 서버도 이 경고를 보고합니다. 이전의 #1350 notice는 router `return`과 `--models-preset`, sampler 순서 검사 이후에 기록됐기 때문에, router 시작은 그 루프에 도달하지 못했고 그 검사에서 실패한 시작은 notice를 보여주지 않고 종료했습니다. 대가는 순서입니다. KV 대체 notice가 이제 `kv_cache_mode = ...` INFO 줄 옆이 아니라 로그 앞부분에 나옵니다. 옛 위치의 주석이 `log_startup_notices`를 가리킵니다.

### 3.3 경고만 옮기고 INFO는 그대로

같은 헬퍼의 "env var is set but the CLI flag takes precedence" 메시지는 `tracing::info!`이고 그대로 두었기 때문에, 여전히 subscriber 전에 출력되어 사라집니다. PR이 이를 명시합니다. 플래그가 변수를 이기는 것은 예상된 동작이고, 잘못된 값이 무시되는 것이 운영자가 봐야 할 경우라는 판단입니다. INFO 줄까지 옮기려면 WARN으로 올리거나 notice마다 레벨을 실어야 했습니다.

### 3.4 스캔 전에 command를 build

clap derive의 `#[arg(group = "x")]`는 멤버십을 인자 쪽에 기록하고, clap은 `Command::build`에서야 이를 `ArgGroup`에 복사하며, 이렇게 생긴 그룹은 `multiple`이 아닙니다. build하지 않은 command를 스캔하면 바이너리가 실제로 쓰는 바로 그 형태를 놓칩니다. 그래서 `env_bound_override_args`는 command를 복제해 먼저 build합니다. build는 멱등이고, 복제하므로 호출자의 command는 바뀌지 않습니다.

### 3.5 clap validator를 그대로 따르고, private 필드는 `Debug`로 읽기

`direct_conflicts`는 clap `clap_builder/src/parser/validator.rs`의 `gather_arg_direct_conflicts`를 따릅니다: 인자 자체의 충돌, 포함하는 각 그룹의 충돌(그룹 id는 멤버로 전개), 포함하는 non-multiple 그룹의 형제들. `arg_conflict_partners`는 역방향(직접 충돌 목록에 이 인자를 포함하는 인자)과 모든 `exclusive` 인자를 더해 `validate_exclusive`를 반영합니다. 스스로 exclusive인 env 바인딩 인자는 `exclusive` 관계로 보고됩니다.

clap은 그룹의 `conflicts` 목록에 대한 getter를 제공하지 않습니다. PR은 인자의 `overrides`에 이미 쓰던 방식대로 그룹의 `Debug` 출력에서 이를 읽습니다. 검색은 `, args: [` 이후부터 시작하므로 그룹 id에 같은 문자열이 있어도 잘못 매칭되지 않습니다. clap이 포맷을 바꾸면 가드는 빈 목록을 반환하고 통과하는 대신, 수정할 함수를 명시한 메시지와 함께 panic합니다. id 목록 파서는 `parse_debug_id_list`로 분리되어 두 리더가 공유합니다. clap 4.6에서 `ArgGroup::is_multiple`은 `&mut self`를 받기 때문에 `group.clone().is_multiple()`를 씁니다.

### 3.6 실제 매칭 규칙으로 테스트

테스트 헬퍼의 `contains` 검사를 그 자리에서 고치는 대신, 바이너리의 규칙을 임의의 argv에 적용할 수 있게 만들고(`long_cli_flag_in`) 운영 코드와 테스트 헬퍼 모두 이를 쓰게 했습니다. 이제 테스트가 바이너리와 같은 코드를 실행하므로, 매칭 규칙이 바뀌어도 테스트가 검사하는 내용과 어긋나지 않습니다.

## 4. 검증

- `env_fallback_warnings_are_logged_once_a_subscriber_exists`는 subscriber 없이 환경 변수 18개에 잘못된 값을 설정하고, 모든 헬퍼와 `into_startup_config`(범위를 벗어난 `--adaptive-decay`와 충돌하는 KV 캐시 플래그 포함)를 실행한 뒤, 캡처용 `tracing_subscriber` 아래에서 `log_startup_notices`를 실행합니다. 메시지 20개가 각각 WARN 줄에 나타나고 WARN 줄 수가 정확히 20개인지 확인하므로, notice 중복도 잡아냅니다.
- 가드 self-test는 그룹 쪽과 인자 쪽에 선언된 non-multiple 그룹, 양방향으로 인자와 충돌하는 그룹, 다른 그룹과 충돌하는 그룹, exclusive env 인자, exclusive 인자 옆의 env 인자를 다룹니다. `src/main_tests.rs`와 `src/bin/mlx_server.rs`의 실제 CLI 가드 검사는 여전히 빈 목록을 보고하므로, 현재 인자 중 확장된 가드에 걸리는 것은 없습니다.
- PR에 보고된 ROCm (gfx1151) 로컬 실행: `cargo clippy --lib --tests --bins -- -D warnings`, `cargo test --lib server::cli_input` (161 통과), `cargo test --lib cli::` (301 통과), `cargo test --lib server::thinking_budget`, `cargo test --bin mlxcel-server` (45 통과), `cargo test --bin mlxcel` (321 통과). 이 보고서 작성 과정에서는 다시 실행하지 않았습니다.

## 5. 남은 공백과 위험

- **INFO 우선순위 메시지는 여전히 사라집니다** (3.3 참고). 운영자에게 필요하다면 notice 타입이 레벨을 가져야 합니다.
- **새 경고를 목록에 넣도록 강제하는 장치가 없습니다.** out-parameter는 기존 헬퍼만 보호합니다. subscriber 이전 경로에 새로 추가된 `tracing::warn!`은 작성자가 규칙을 모르면 다시 버려집니다. lint나, 시작 입력 경로 전체를 캡처 subscriber 아래에서 실행해 `start_server` 전에 아무것도 기록되지 않음을 확인하는 테스트가 있으면 막을 수 있지만, 이 PR은 추가하지 않았습니다.
- **순서가 바뀌면 KV 플래그 우선순위 경고가 두 번 출력될 수 있습니다.** `resolve_kv_cache_mode`는 여전히 직접 `tracing::warn!`을 호출하고, 서버는 같은 메시지를 notice로도 push합니다. 지금은 첫 호출 시점에 subscriber가 없으므로 운영자는 한 번만 봅니다. 나중에 `into_startup_config`가 활성 subscriber 아래에서 호출되면 두 번 나옵니다.
- **메시지 문구가 약간 바뀌었습니다.** 여러 메시지가 수동 따옴표(`"{raw}"`)에서 `{raw:?}`로 바뀌었습니다. 일반적인 값은 출력이 같고, 따옴표나 제어 문자가 있는 값은 이제 escape됩니다. 특이한 입력에 대해 옛 문구를 정확히 매칭하던 로그 수집기는 차이를 보게 됩니다.
- **clap `Debug` 포맷 의존.** `arg_overrides`와 `group_conflicts` 모두 clap이 derive한 `Debug` 레이아웃에 의존합니다. 이를 바꾸는 clap 업그레이드는 가드 테스트에서 panic하며, 이는 의도된 실패 방식이지만 그때 코드 수정이 필요합니다.
- **public API 변경.** `mlxcel::server`에서 re-export되는 `env_fallback_*` 함수들의 시그니처가 바뀌었습니다. 알려진 저장소 외부 호출자는 없습니다.

## 6. 학습 포인트

- tracing subscriber를 늦게 설치하는 Rust 바이너리에서는 그 이전의 모든 `tracing` 호출이 no-op입니다. 초기 설정 코드에는 로깅이 아니라 명시적인 메시지 채널이 필요하다고 보아야 합니다.
- 필수 `&mut Vec<_>` 인자는 많은 호출 지점에 걸쳐 "버리지 말 것"을 컴파일 타임 속성으로 만드는 저렴한 방법입니다.
- clap derive의 그룹 멤버십은 `Command::build` 이후에만 존재하고, clap은 `get_arg_conflicts_with`로 보이지 않는 충돌을 그룹과 `exclusive`에서 파생합니다. clap command를 정적으로 분석하려면 먼저 build하고, builder API가 아니라 validator를 따라야 합니다.
- 테스트 헬퍼가 운영 코드의 규칙을 다시 구현하고 있다면, 운영 규칙을 명시적 입력에 대해 노출하고 테스트에서 그것을 호출하는 편이 낫습니다.

## 7. 참고

- 이슈 #2697 (이번 수정), PR #2704
- 이전 작업: #1350 (KV 모드 대체 notice), #2609와 #2618 (override 쌍의 env 바인딩, 가드 도입), #2622, #2664, #2668 (CORS origin 변수와 notice)
- clap 4.6.1, `clap_builder/src/parser/validator.rs` (`gather_arg_direct_conflicts`, `validate_exclusive`)
