# PR #2622: overrides_with 플래그 쌍에서 clap env 바인딩 금지

**날짜**: 2026-10-26
**상태**: 구현 및 로컬 검증 완료(draft PR, 아직 머지 전)
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2622(이슈 #2618)는 #2612의 후속 작업이다. #2612는 clap 4.6.7이 `VAR=... --반대쪽-플래그`를 파싱 오류로 만들기 때문에 llama-server `--x` / `--no-x` 쌍 여섯 개에서 clap `env` 바인딩을 걷어냈다. 이 PR은 두 가지를 한다. 첫째, clap `Command` 트리를 순회해 `env`와 `overrides_with`를 함께 가진 인자를 모두 나열하는 테스트용 가드 `mlxcel::cli::env_bound_override_args`를 추가하고, 두 바이너리가 그 목록이 비어 있음을 단언한다. 새 쌍이 같은 버그를 조용히 다시 들여오지 못한다. 둘째, #2612가 남긴 엄격성 차이를 없앤다. `LLAMA_ARG_MODELS_AUTOLOAD`도 나머지 다섯 쌍 변수와 같은 `apply_boolish_pair_env` 리더를 거치므로, 잘못된 값이면 같은 메시지로 시작이 실패하고 한 글자 표기 `y`/`t`/`n`/`f`도 받아들인다.

## 1. 문제 정의

#2612 이후 두 가지 빈틈이 남아 있었다.

- **재발 방지 장치가 없었다.** #2609의 결함은 derive 속성에서는 무해해 보이는 조합에서 나왔다. 서로 `overrides_with`인 쌍의 한쪽에 `env = "LLAMA_ARG_X"`를 붙이는 것이다. clap은 환경 변수에서 온 값을 "있음"으로 세고, override 짝을 모두 직접 충돌로 등록하며(`clap_builder` `parser/validator.rs`), 덮어쓰인 쪽을 지우는 것은 명령줄에 실제로 나온 경우뿐이다. 앞으로 누군가 예전 방식으로 쌍을 작성해도 테스트 스위트나 `make verify-llama-compat`이 잡아내지 못했다. 호환 매니페스트 검사기는 clap 정의가 아니라 JSON을 읽으므로 이 속성을 볼 수 없다.
- **규칙이 다른 env 리더 두 개.** #2612 이후 다섯 쌍 변수는 `cli::apply_boolish_pair_env`가 읽었다. clap `BoolishValueParser` 어휘를 쓰고 모르는 값이면 오류를 돌려주며, 두 바이너리는 이를 시작 실패로 바꿨다. 반면 `LLAMA_ARG_MODELS_AUTOLOAD`는 `y`/`t`/`n`/`f`를 거부하는 `parse_env_bool`이 읽었고, 잘못된 값이면 경고만 남긴 채 autoload를 켠 상태로 두었다. llama-server b10621은 잘못된 쌍 값을 시작 시 거부하므로(`docs/llama-server-compat.md`) autoload만 예외였다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 가드 헬퍼 | `src/cli/mod.rs`에 `pub fn env_bound_override_args(&Command) -> Vec<String>`(`#[doc(hidden)]`)을 추가했다. 서브커맨드를 재귀로 돌며, `get_env()`가 있는 인자마다 양방향으로 짝을 모은다. 자신을 override 목록에 올린 인자와, 자신의 목록에 있는 인자다. 결과는 `<command path> <arg id> (env <VAR>) <-> <partners>` 형식이다. |
| override 목록 리더 | `arg_overrides(&Arg)`는 `Arg`의 `Debug` 출력에서 `overrides: [...]` 필드의 id를 작은 따옴표 문자열 스캐너로 뽑는다(`\` 이스케이프 처리). 필드가 없거나 목록이 닫히지 않으면 panic한다. |
| 가드 자체 테스트 | `cli::env_override_guard_tests`: 덮어쓰는 쪽의 env, 덮어쓰이는 쪽의 env, env 없는 쌍과 쌍 없는 env에서 오탐이 없는지, 서브커맨드 순회. |
| 바이너리 가드 테스트 | `src/main_tests.rs`(`mlxcel` `Cli::command()`, 모든 서브커맨드)와 `src/bin/mlx_server.rs`의 `tests` 모듈에 `no_arg_combines_clap_env_with_overrides_with`. 둘 다 빈 목록을 단언하고 실패 시 위반 인자를 출력한다. 허용 목록은 없다. |
| autoload 리더 | `src/server/cli_input.rs`의 `env_fallback_models_autoload`가 `anyhow::Result<()>`를 반환한다. `bool` 필드를 `apply_boolish_pair_env`가 기대하는 `Option<bool>` 형태로 옮기고(`enable_was_set.then_some(*value)`, `disable_was_set`은 `negated`), 값이 결정됐을 때만 되써 넣는다. |
| 공용 오류 | 새 비공개 함수 `invalid_bool_pair_env((var, raw))`가 `"<VAR> has an invalid boolean value \"<raw>\""`를 만든다. `apply_bool_pair_env_bindings`도 이를 써서 여섯 변수가 한 메시지를 공유한다. |
| 호출 지점 | `build_startup_input`(`mlxcel-server`)과 `run_serve_async`(`mlxcel serve`)가 결과를 `?`로 전파한다. |
| 테스트 | `src/server/cli_input_tests.rs`: `models_autoload_env_ignores_an_unparseable_value`가 `models_autoload_env_rejects_an_unparseable_value`로 바뀌었다(정확한 오류 문구, 기본값 유지, 어느 플래그든 잘못된 변수를 건너뜀). 새 `models_autoload_env_accepts_boolish_spellings`(`y`, `t`, `YES`, `n`, `f`, `Off`). 세 테스트 파일의 기존 호출부에는 `.expect(...)`가 붙었다. |

## 3. 기술적 선택과 그 이유

### 3.1 매니페스트 린트가 아니라 실제 `Command`를 보는 Rust 테스트

#2612 보고서는 `--no-*` 항목의 `env_binding: clap`을 거부하는 `make verify-llama-compat` 린트를 제안했다. 이 PR은 대신 두 바이너리가 실제로 만드는 clap `Command`를 검사한다. 매니페스트는 llama-server 호환 플래그만 다루고, 누군가 바인딩을 정확히 기록해야만 의미가 있다. `Command`는 진실의 원천이고, 매니페스트에 절대 오르지 않을 mlxcel 전용 쌍까지 모든 인자를 덮는다. 검사가 평범한 `cargo test` 바이너리 안에서 돌기 때문에 새 verify 타깃도 필요 없다.

### 3.2 `Debug`로 비공개 필드를 읽되, 실패는 크게

clap은 `get_env()`는 공개하지만 `overrides` getter는 없다. 선택지는 같은 지식을 별도 목록으로 다시 적기(어긋나기 쉽다), 런타임에 가짜 env 값을 넣고 파싱 오류를 확인하기(느리고 모든 인자를 실행해야 한다), `Debug` 출력을 읽기였다. PR은 `Debug`를 읽고 취약한 부분을 드러낸다. 필드가 없거나 닫히지 않으면 `cli::arg_overrides`를 가리키는 메시지로 panic하고, 자체 테스트가 가짜 `env` + `overrides_with` 쌍이 여전히 잡히는지 고정한다. 따라서 clap 업그레이드로 필드 이름이나 모양이 바뀌면, 빈 목록을 돌려줘 바이너리 가드를 조용히 통과시키는 대신 테스트가 깨진다.

### 3.3 override를 양방향으로 확인

지금 이 쌍들의 derive 코드는 양쪽 모두에 `overrides_with`를 두지만, clap은 한쪽만 선언해도 충돌을 적용한다. 그래서 헬퍼는 env에 묶인 인자가 덮어쓰는 쪽이든 덮어쓰이는 쪽이든 보고한다. 자체 테스트는 긍정 쪽만 override를 선언한 쌍을 만들고 env 바인딩을 각 쪽에 번갈아 붙여 두 조회 경로를 모두 덮는다.

### 3.4 허용 목록 없음

이슈는 정당한 `env` + `overrides_with` 사례가 없다고 밝힌다. 허용 목록은 가드가 막으려는 바로 그 패턴에 예외를 만들 빌미가 된다. 정말 필요해지면 그때 이유와 함께 추가하면 된다.

### 3.5 엄격한 리더를 하나 더 만들지 않고 autoload를 공용 리더에 맞춤

`models_autoload`는 `default_value_t = true`인 평범한 `bool`이라, 다섯 호환 쌍처럼 `Option<bool>`의 존재 여부를 쓸 수 없다. PR은 별도 엄격 파서를 두지 않고 상태를 공용 리더의 입력으로 옮긴다. 명령줄의 `--models-autoload`는 `Some(현재값)`, `--no-models-autoload`는 `negated = true`가 되고, 리더가 값을 만들었을 때만 되써 넣는다. 그러면 어휘, 오류, 우선순위 규칙이 한 곳에서 나오고, 메시지도 `invalid_bool_pair_env`로 빼 두었으니 여섯 변수가 서로 어긋날 수 없다.

## 4. 검증

PR 본문 기준, 로컬에서 실행:

- `cargo test --profile test-fast --lib cli::`: 291개 통과(가드 자체 테스트 4개 포함).
- `cargo test --profile test-fast --lib server::cli_input`: 155개 통과.
- `cargo test --profile test-fast --bin mlxcel --bin mlxcel-server`: 316 + 39개 통과.
- `cargo test --profile test-fast --test cli_help_consistency`: 27개 통과.
- `cargo clippy ... -D warnings`, `cargo fmt --all -- --check`, `make verify-llama-compat`, `make verify-env-var-docs`: 통과.
- 음성 시연: `mlxcel serve --models-autoload`와 `--cache-prompt`에 `env`를 잠시 다시 붙이자 두 바이너리 가드 테스트가 실패하며 위반 인자 세 개를 이름으로 보고했다. 바인딩을 다시 빼면 둘 다 통과한다.
- 수동: `LLAMA_ARG_MODELS_AUTOLOAD=maybe`이면 `mlxcel-server`와 `mlxcel serve` 모두 공용 오류로 종료한다.

전체 `make verify`는 구현 worktree에서 돌리지 않았고 머지 게이트에 맡겼다. #2618의 세 번째 인수 조건이 이를 요구한다.

## 5. 남은 한계

- **`Debug` 스캔을 `settings` 뒤로 고정 (머지 전 수정).** clap 4.6.7은 `help`와 `long_help`를 `settings`와 `overrides`보다 먼저 출력하므로, 처음 나오는 `overrides: [` 부분 문자열에서 시작하던 첫 버전은 그 문자열을 담은 도움말을 잘못 읽을 수 있었다. 이제 스캔은 `, settings: ArgFlags(` 다음에서 `, overrides: [`를 찾으며, 도움말에 같은 문자열을 넣은 self-test `help_text_that_looks_like_the_overrides_field_is_ignored`는 첫 버전에서 실패한다.
- **`overrides_with`만 검사한다.** `env`와 결합한 `conflicts_with`(예: `LLAMA_ARG_CORS_ORIGINS` 대 `--allowed-origins`)도 "env는 있음으로 센다"는 같은 동작을 하지만, override 쌍이 아니라 문서화된 상호 배제이고 #2618이 범위 밖으로 두었다.
- **전역 인자.** 헬퍼는 build 전 `Command::get_arguments()`를 보고 한 커맨드 안에서만 짝을 맞춘다. 루트에 있는 env 바인딩 전역 인자를 서브커맨드에 선언된 인자가 덮어쓰는 경우는 보고되지 않는다. 지금 그런 인자는 없다.
- **autoload의 공백 처리가 바뀌었다.** `parse_env_bool`은 값을 trim했고 `parse_boolish`는 하지 않는다. `LLAMA_ARG_MODELS_AUTOLOAD=" true"`는 #2612부터 이 PR 전까지 받아들여졌지만 이제는 시작 오류다. 나머지 다섯 변수, 그리고 clap 자체 파서와 같은 동작이다.
- **배포에 보이는 동작 변화.** `LLAMA_ARG_MODELS_AUTOLOAD`를 알 수 없는 값으로 설정한 배포는 전에는 경고와 함께 autoload를 켠 채 시작했지만, 이제는 시작하지 못한다. b10621과 맞추려는 의도된 변화지만 사용자에게 보인다.

## 6. 학습 포인트

- 라이브러리가 린트에 필요한 상태를 숨길 때, 테스트 전용 가드라면 `Debug` 출력을 읽어도 된다. 단 파서가 닫힌 쪽으로 실패하고, 자체 테스트가 알려진 양성 사례로 형식을 고정해야 한다.
- "발견 없음"을 돌려주는 가드는 무언가를 발견할 수 있는지부터 테스트해야 한다. 그렇지 않으면 형식이 바뀌는 순간 영원히 통과한다.
- 모양이 다른 두 필드(기본값 있는 `bool`, `Option<bool>`)가 같은 env 의미를 가져야 할 때, 한쪽을 다른 쪽 리더의 입력으로 옮기면 리더와 오류 메시지가 하나로 유지되어, 파서 두 개를 맞춰 두는 것보다 일관성을 지키기 쉽다.
