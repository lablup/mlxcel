# PR #2612: LLAMA_ARG_* 토글 쌍 변수를 파싱 후에 읽기

**날짜**: 2026-10-26
**상태**: 구현 및 로컬 검증 완료(draft PR, 아직 머지 전)
**언어**: Rust, JSON
**위험도**: 낮음

## 요약

PR #2612(이슈 #2609)는 두 서버 바이너리의 파싱 실패를 고친다. `LLAMA_ARG_MODELS_AUTOLOAD=<아무 값> mlxcel serve --no-models-autoload`는 플래그가 이기는 대신 clap의 "cannot be used with '--models-autoload'" 오류로 종료했다. 원인은 clap 4가 `env`에 묶여 있으면서 `overrides_with` 짝도 가진 인자를 다루는 방식이다. 이슈가 요청한 전수 점검에서 llama-server 호환 쌍 다섯 개에서 같은 결함이 더 나왔다. `--cache-prompt`, `--cont-batching`, `--context-shift`, `--cache-idle-slots`, `--kv-unified`다. 이제 여섯 쌍 모두 `LLAMA_ARG_*` 변수를 파싱이 끝난 뒤에, 그리고 쌍의 어느 쪽도 명령줄에 없을 때만 읽는다. 그 결과 우선순위는 llama-server b10621과 같다. 명시적 플래그는 어느 쪽이든 환경 변수를 이기고, 환경 변수만 있으면 여전히 적용된다.

## 1. 문제 정의

llama-server는 많은 불리언을 `--x` / `--no-x` 쌍과 `LLAMA_ARG_X` 변수로 노출한다. mlxcel은 두 플래그 사이에 clap의 `overrides_with`를 두고, 긍정 쪽에 `env = "LLAMA_ARG_X"` 속성을 붙여 이를 모델링했다. 이 조합은 clap 4.6.7에서 동작하지 않는다.

- clap은 `overrides_with` 짝을 모두 직접 충돌로 등록한다(clap_builder validator의 `gather_arg_direct_conflicts`).
- clap은 환경 변수에서 온 값을 "있음"으로 센다.
- 검증 전에 덮어쓰인 쪽을 지우는 것은 명령줄에 실제로 나온 경우뿐이다.

그래서 변수가 설정된 상태에서 `--no-x`는 "있는" `--x`와 만나고, 검증기가 충돌을 보고한다. 덮어쓰기는 실행되지 않는다. 컨테이너 이미지나 systemd 유닛처럼 변수를 전역으로 설정한 배포에서는 한 번의 실행에서 기능을 끌 수 없었다. 바로 그 경우를 위해 `--no-x` 플래그가 있다. 이 버그는 #2584 이전부터 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `--models-autoload`(두 바이너리) | `src/main.rs`(`ServeArgs`)와 `src/bin/mlx_server.rs`(`ServerArgs`)의 clap 인자에서 `env`를 뺐다. `src/server/cli_input.rs`에 새로 둔 `env_fallback_models_autoload(value, enable_was_set, disable_was_set)`는 두 플래그가 모두 없을 때 `parse_env_bool`로 변수를 읽고, 해석할 수 없는 값이면 경고를 남기고 무시한다. 그 다음 쌍은 여전히 `resolve_compat_toggle`로 결정한다. |
| `Option<bool>` 쌍 다섯 개 | `--cache-prompt`, `--cont-batching`(`CacheCompatArgs`), `--context-shift`(`ContextCompatArgs`), `--cache-idle-slots`, `--kv-unified`(`SlotCompatArgs`)에서 `env`를 뺐다. 각 그룹에 공용 `cli::apply_boolish_pair_env(var, &mut value, negated)` 위에 만든 `apply_env_bindings()`가 생겼다. |
| 공용 리더 | `apply_boolish_pair_env`는 `value.is_some() || negated`이면 바로 반환한다. 아니면 clap `BoolishValueParser`의 어휘(`y/yes/t/true/on/1`, `n/no/f/false/off/0`, 대소문자 무시)로 변수를 해석하고, 그 밖의 값이면 `(var, raw)`를 반환한다. |
| 묶음 함수 | `server::apply_bool_pair_env_bindings(cache, context, slot)`이 세 그룹을 실행하고, 잘못된 값을 `"<VAR> has an invalid boolean value \"<raw>\""` 오류로 바꾼다. |
| 연결 지점 | `mlxcel serve`: `run_serve_async`에서 settings-endpoint 폴백 바로 뒤, 메모리 사전 점검(`--kv-unified`를 읽음)과 Web UI 라우터의 `--models-autoload` 덮어쓰기보다 앞. `mlxcel-server`: `build_startup_input`에서 `env_fallback_cors_credentials` 옆. `cache_args::from_env()`도 빈 argv 파싱 뒤에 `apply_env_bindings()`를 부른다. |
| 호환 매니페스트 | `compat/llama-server/b10621/router-models.json`과 `runtime-and-context.json`의 여섯 항목이 `env_binding: clap`에서 `env_binding: runtime`으로 바뀌고 `env_test: src/server/cli_input_tests.rs`를 기록한다. 문서 주석은 여전히 각 변수 이름을 적고 있어 `make verify-llama-compat`과 `make verify-env-var-docs`가 계속 맞는다. |
| 테스트 | `src/main_tests.rs`: `mlxcel_serve_models_autoload_env_yields_to_either_flag`(true/false/1/0을 두 플래그와 조합, 환경 변수 단독 포함). 기존 `mlxcel_serve_no_models_autoload_reaches_the_resolved_toggle`은 더 이상 변수를 지우지 않고, 쓰이지 않게 된 `ScopedEnv::unset`은 삭제했다. `mlx_server.rs`: `models_autoload_env_does_not_conflict_with_either_flag`(환경 변수 단독으로 `build_startup_input`까지 가는 경우 포함). `src/server/cli_input_tests.rs`: models-autoload 우선순위와 해석 불가 값. 다섯 쌍에 대해서는 환경 변수와 `--no-*`, 거짓 값 환경 변수와 긍정 플래그, 환경 변수 단독, 잘못된 값이 변수 이름과 함께 보고되는지. |

## 3. 기술적 선택과 그 이유

### 3.1 clap을 우회하지 않고 파싱 뒤에 변수를 읽는다

다른 선택지가 두 가지 있었다. `overrides_with`를 빼면 한 명령줄의 `--x --no-x`도 충돌이 되어, 스크립트가 기대는 llama-server의 "마지막 플래그가 이긴다" 동작이 깨진다. `env`를 두고 충돌 오류를 잡으려면 clap 오류 문자열을 비교해야 한다. 파싱 뒤에 변수를 읽으면 명령줄은 clap이 책임지고, 환경 변수 우선순위는 프로젝트가 통제하는 코드에 남는다. 코드베이스에 이미 있던 패턴이기도 하다. `--cors-credentials`, `--slots`, `--warmup`은 처음부터 clap `env`를 쓰지 않았고, 이슈가 본보기로 지목한 것도 `env_fallback_cors_credentials`였다.

### 3.2 필드 타입에 따라 두 가지 감지 방식

다섯 호환 쌍은 긍정 쪽을 `num_args = 0..=1`인 `Option<bool>`로 저장하므로, 파싱 뒤 `Some(_)`이면 이미 "명령줄이 무언가를 말했다"는 뜻이고 부정 쪽은 평범한 `bool`이다. 그래서 `apply_boolish_pair_env`는 argv를 볼 필요가 없다. `models_autoload`는 `default_value_t = true`인 평범한 `bool`이라 파싱 결과로는 "없음"과 "지정함"을 구분할 수 없다. 그래서 CORS 폴백처럼 프로세스 argv에서 `--name` 또는 `--name=`을 찾는 `long_cli_flag_was_set`을 쓴다. 이 시그니처 덕분에 테스트 하네스의 argv를 제어할 수 없는 테스트에서도 함수를 쉽게 구동할 수 있다. `mlx_server` 테스트는 불리언을 직접 넘긴다.

### 3.3 다섯 쌍은 예전 값 범위를 그대로 유지한다

예전에 동작하던 변수 값은 지금도 같은 뜻이어야 한다. 플래그의 선택적 값은 여전히 `BoolishValueParser`를 쓰므로, 환경 변수 리더도 `parse_env_bool`(`y`, `t`, `n`, `f`가 없음)을 재사용하지 않고 그 어휘를 정확히 따른다. 잘못된 값은 clap 바인딩 때와 마찬가지로 변수 이름을 밝히는 시작 오류다. 반면 `models_autoload`는 CORS 쪽을 따른다. `parse_env_bool`을 쓰고, 잘못된 값이면 경고 후 기본값으로 넘어간다. 이 비대칭은 PR에서 의도한 것이지만 알아 둘 만하다(알려진 한계 참고).

### 3.4 `mlxcel serve`에서는 폴백을 일찍 적용한다

이 값들은 `ServerStartupInput`이 만들어지기 전에 쓰인다. 메모리 사전 점검은 `--kv-unified`를 읽고, Web UI 경로는 라우터 모드에서 `--models-autoload`를 덮어쓸 수 있다. `run_serve_async` 앞부분에서 폴백을 적용하면 둘 다 clap이 필드를 채우던 때처럼 환경 변수에서 온 값을 본다. 나중에 적용하면 `LLAMA_ARG_KV_UNIFIED`를 설정한 사용자의 사전 점검 결과가 조용히 달라진다.

### 3.5 호환 매니페스트에 변경을 기록한다

b10621 매니페스트는 각 변수가 어떻게 묶여 있는지 적는다. `clap`을 `runtime`으로 바꾸고 테스트 파일을 적으면 검증기가 정직하게 유지된다. `scripts/ci/check_llama_compat_manifest.py`는 `runtime` 바인딩에 대해 `env_test`가 실제로 있는 파일을 가리키는지 검사한다.

## 4. 검증

PR 본문 기준, 로컬에서 실행했다(호스트 CI 결과는 주장하지 않는다).

- `cargo test --profile test-fast --bin mlxcel --bin mlxcel-server`: 315 + 38 통과.
- `cargo test --profile test-fast --lib cli::`: 287 통과. `--lib server::cli_input`: 154 통과.
- `cargo clippy --profile test-fast --bin mlxcel --bin mlxcel-server --lib --tests -- -D warnings`: 경고 없음.
- `cargo fmt --all -- --check`, `make verify-llama-compat`, `make verify-env-var-docs`: 통과.

#2609의 인수 조건(두 플래그와 참/거짓 값 조합, 환경 변수 단독, 두 바이너리, `src/main_tests.rs`의 `ScopedEnv` 테스트, 호환 검증)은 각각 위의 이름 붙은 테스트나 검사로 확인된다.

## 5. 알려진 한계

- **두 리더의 엄격도가 다르다.** 잘못된 `LLAMA_ARG_MODELS_AUTOLOAD`는 경고 후 무시되고(autoload는 켜진 채), 나머지 다섯 변수의 잘못된 값은 시작을 멈춘다. `LLAMA_ARG_MODELS_AUTOLOAD=y`나 `=t`도 `parse_env_bool`이 거부하지만, 다섯 쌍은 받아들인다. 예전 clap 바인딩이 그 `bool` 필드에서 이런 철자를 받아들였는지는 확인하지 않았다. 받아들였다면 이 변수 하나에 대해 작은 회귀다.
- **`long_cli_flag_was_set`은 원시 argv를 훑는다.** 다른 옵션의 값인데 `--models-autoload`와 같은 토큰, 또는 `--` 구분자 뒤의 토큰도 플래그로 센다. 기존 헬퍼에서 물려받은 동작이고 CORS 폴백도 같다. 현실적인 실행에서는 걸리지 않는다.
- **범위 밖: `--cors-origins`.** `LLAMA_ARG_CORS_ORIGINS`는 `conflicts_with = "allowed_origins"`를 써서 이 변수도 `--allowed-origins`와 충돌한다. 이 쌍은 덮어쓰기 쌍이 아니라 상호 배타로 문서화되어 있어 그대로 두었다.
- **재발 방지 장치가 없다.** 앞으로 새 `--x` / `--no-x` 쌍이 `overrides_with`와 함께 `env`를 다시 붙여도 막을 것이 없다. `--no-*` 철자를 가진 항목의 `env_binding: clap`을 거부하는 검사를 `make verify-llama-compat`에 넣으면 막을 수 있다.
- **종단 간 실행은 하지 않았다.** 테스트는 파싱과 시작 입력 생성까지 구동한다. 변수를 설정한 채 서버를 실제로 띄우지는 않았다.

## 6. 학습 포인트

- clap 4에서 `overrides_with`는 명령줄 등장 순서로 해소되는 충돌로 구현되어 있다. 환경 변수에서 온 값은 충돌에는 참여하지만 해소에는 참여하지 않으므로, 같은 쌍에 `env`와 `overrides_with`를 함께 쓰면 맞물리지 않는다.
- 플래그 필드가 `Option<bool>`이면 파싱 뒤 `Some`이 "사용자가 무언가를 말했다"는 믿을 만한 신호다. 기본값이 있는 `bool`은 그 정보를 잃기 때문에 argv를 직접 봐야 한다.
- 라이브러리에서 값 리더를 꺼내 올 때는 근처의 비슷해 보이는 헬퍼를 재사용하지 말고 라이브러리가 받아들이던 어휘를 그대로 옮긴다. 이 코드베이스의 두 불리언 파서는 철자 네 개가 다르다.
