# PR #2668: CORS origin 환경 변수를 파싱 후에 읽어 clap 충돌 검사를 피하기

**작성일**: 2026-10-27
**상태**: Open (draft), 로컬 검증 완료
**언어**: Rust, Markdown, JSON
**위험도**: 낮음 (시작 인자 처리만 변경, CORS 정책 결정은 그대로)

## 요약

`--cors-origins`와 `--allowed-origins`는 clap에서 서로 배타적이고, 둘 다 clap의 `env` 속성으로 환경 변수에 묶여 있었습니다. clap 4.6.7은 `conflicts_with`를 검사할 때 환경 변수에서 온 값도 지정된 것으로 셉니다. 그래서 한쪽 변수를 설정하고 다른 쪽 플래그를 넘기면 `mlxcel serve`와 `mlxcel-server` 모두 파싱 오류가 났습니다(이슈 #2664). PR #2668은 두 인자에서 clap `env` 바인딩을 없애고, `LLAMA_ARG_CORS_ORIGINS`와 `MLXCEL_ALLOWED_ORIGINS`를 파싱 후에 읽습니다. 명령줄 플래그가 반대쪽 변수보다 우선합니다. 무시된 변수에 대한 안내는 `start_server`까지 전달되어 tracing subscriber가 생긴 뒤에 기록되므로 실제로 보입니다. `src/cli/mod.rs`의 env 바인딩 guard는 이제 `conflicts_with` 쌍의 어느 쪽이든 env에 묶인 인자를 보고합니다.

## 1. 문제

main(4246aa87)의 `target/release/mlxcel serve`에서 양방향으로 재현했습니다.

```
$ LLAMA_ARG_CORS_ORIGINS=http://a.example mlxcel serve --allowed-origins http://b.example ...
error: the argument '--allowed-origins <ORIGINS>' cannot be used with '--cors-origins <ORIGINS>'
$ MLXCEL_ALLOWED_ORIGINS=http://a.example mlxcel serve --cors-origins http://b.example ...
error: the argument '--cors-origins <ORIGINS>' cannot be used with '--allowed-origins <ORIGINS>'
```

같은 쌍이 `src/bin/mlx_server.rs`와 `src/main.rs`에 모두 있습니다. 운영자가 충돌하는 조합을 직접 입력하지 않았는데도, 상속된 환경 변수만으로 시작이 막혔습니다. 기존 guard `env_bound_override_args`(#2618)는 `overrides_with` 쌍만 확인했기 때문에 이 형태를 잡지 못했습니다.

## 2. 변경 내용

| 영역 | 변경 |
|---|---|
| 인자 정의 | 두 바이너리에서 `cors_origins`와 `allowed_origins`의 clap `env`를 제거했습니다. doc comment에 변수를 런타임에 읽는다고 적었고, `--help`는 이 두 플래그에 `[env: ...]`를 더 이상 출력하지 않습니다. |
| 런타임 fallback | `src/server/cli_input.rs`에 `env_fallback_cors_origins`와 `env_fallback_allowed_origins`를 추가했습니다. `src/commands/serve.rs`와 `src/bin/mlx_server.rs`가 파싱 후, 모델 해석 전에 호출하며, 각 플래그 지정 여부는 `long_cli_flag_was_set`으로 판단합니다. |
| 안내 메시지 | helper는 "ignored" 경고를 기록하지 않고 반환합니다. 두 바이너리가 이를 `ServerStartupInput::cors_origin_notices`에 모으고, `into_startup_config`가 `ServerStartupConfig::cors_origin_notices`로 옮기며, `start_server`가 `initialize_server_logging` 직후 router 모드 분기 전에 `log_cors_origin_notices`를 호출합니다. |
| Guard | `env_bound_override_args`에 `arg_conflict_partners`를 추가했습니다. clap의 공개 API `Command::get_arg_conflicts_with`를 양방향으로 사용합니다. 항목 형식은 `<path> <id> (env VAR) conflicts with <ids>`입니다. |
| 문서와 compat | `docs/environment-variables.md`에 런타임 읽기와 우선순위를 설명했습니다. b10621 manifest 항목 `transport-tls-cors.json`은 `env_binding: runtime`과 `env_test`로 바뀌었습니다. |

## 3. 기술적 결정

### 3.1 우선순위

| 명령줄 | `LLAMA_ARG_CORS_ORIGINS` | `MLXCEL_ALLOWED_ORIGINS` | 결과 |
|---|---|---|---|
| `--cors-origins` | 무관 | 설정됨 | 플래그 적용, allow-list 변수는 경고와 함께 무시 |
| `--allowed-origins` | 설정됨 | 무관 | 플래그 적용, b10621 변수는 경고와 함께 무시 |
| 없음 | 설정됨 | 미설정 | 기존처럼 변수 적용 |
| 없음 | 미설정 | 설정됨 | `,`로 나눈 뒤 기존 origin 검증 적용 |
| 없음 | 설정됨 | 설정됨 | 두 변수를 모두 명시한 시작 오류 |
| 두 플래그 모두 | 무관 | 무관 | clap 파싱 오류, 기존과 동일 |

플래그를 주면 그 플래그 자신의 변수는 쓰이지 않으며, 이는 기존 clap 동작과 같습니다. 플래그 없이 두 변수가 모두 설정되면 오류로 처리합니다. 두 규칙은 배타적이고 운영자가 어느 쪽을 원했는지 알 수 없기 때문입니다. 기존 clap 충돌도 이 경우 실패했습니다. UTF-8이 아닌 값은 clap 바인딩과 마찬가지로 시작 오류입니다.

### 3.2 안내를 기록하지 않고 반환하는 이유

fallback은 `start_server`가 tracing subscriber를 설치하기 전에 실행되고, `mlxcel serve`는 `init_cli_tracing`을 건너뜁니다. 그래서 helper 안의 `tracing::warn!`은 어디에도 출력되지 않았습니다. 첫 커밋이 바로 그렇게 동작했고, 구현 리뷰에서 경고 없이 시작되는 것을 재현했습니다. 가장 중요한 경우는 `MLXCEL_ALLOWED_ORIGINS`가 설정된 상태의 `--cors-origins '*'`입니다. allow-list가 wildcard에 밀리는 것은 의도한 우선순위지만, 메시지 없이 일어나서는 안 됩니다. 두 번째 커밋은 텍스트를 반환하고, 시작 구조체를 통해 전달하며(`mlxcel serve`는 skip 처리된 `ServeArgs` 필드 사용), 로깅이 준비된 뒤 기록합니다. 범위를 좁히기 위해 다른 `env_fallback_*` 경고는 그대로 두었습니다.

### 3.3 Guard 확장

clap은 어느 쪽이 선언했든 충돌을 양방향으로 적용합니다. 그래서 `arg_conflict_partners`는 해당 인자가 충돌하는 인자와, 해당 인자와의 충돌을 선언한 인자를 모두 모읍니다. 두 바이너리의 기존 테스트가 guard 목록이 비어 있음을 확인하므로, 충돌 관계의 인자에 env 바인딩이 새로 붙으면 테스트가 실패합니다.

### 3.4 보안

보안 리뷰에서 CRITICAL이나 HIGH 항목은 없었습니다. 플래그가 없으면 동작은 기존과 같습니다. 환경 변수에서 온 allow-list도 플래그와 같은 `parse_allowed_origins` 검증을 거칩니다. `MLXCEL_ALLOWED_ORIGINS=`는 여전히 시작에 실패하고, `LLAMA_ARG_CORS_ORIGINS=`는 여전히 `*`가 아닌 빈 literal이 됩니다. 두 필드가 모두 채워지면 `CorsPolicy::resolve`가 비어 있지 않은 allow-list를 선택하므로 닫힌 쪽으로 실패합니다. `long_cli_flag_was_set`은 `--flag value`와 `--flag=value`를 모두 인식합니다. 두 바이너리 모두 `infer_long_args`나 이 플래그의 alias를 쓰지 않고, `allow_hyphen_values` 인자는 정수형 `--parallel` / `--n-parallel`뿐입니다.

## 4. 검증

- 바이너리별: 양방향으로 플래그가 반대쪽 변수보다 우선하고, 변수 하나만 있으면 여전히 적용되며, 두 플래그는 여전히 충돌하고, 두 변수가 모두 있으면 둘을 명시하며 시작에 실패합니다. `mlxcel-server`의 두 변수 검사는 env lock 없이 `build_startup_input`을 호출하는 다른 테스트와 경합하지 않도록 자식 프로세스에서 실행합니다.
- `src/cli/mod.rs`: 선언하는 쪽과 충돌 대상 쪽 각각의 env 바인딩에 대한 guard 단위 테스트, 그리고 env가 없는 경우의 테스트.
- 안내 메시지: 각 helper가 무시된 변수에 대한 안내를 반환하고(라이브러리와 두 바이너리), `into_startup_config`가 이를 전달하며, `log_cors_origin_notices`가 캡처한 subscriber에 `WARN`을 기록합니다. 안내나 로깅 단계를 stub으로 바꾸면 이 테스트들이 실패합니다.
- main에서 Red: clap `env` 속성을 되돌리면 바이너리별 env 테스트와 두 guard 테스트가 실패합니다(guard가 두 CORS 인자를 모두 보고). guard 확장을 stub으로 바꾸면 새 `cli::` 테스트 두 개가 실패합니다.
- Green: `cargo test --profile test-fast`로 `--lib cli:: server::cli_input server::cors`, `--bin mlxcel`, `--bin mlxcel-server`, `--test cli_help_consistency`, `--test llama_compat_manifest`; `--lib --tests --bins -D warnings` clippy; fmt; `make verify-env-var-docs verify-llama-compat`.

## 5. 남은 과제와 후속 작업 (아직 이슈 미등록)

- **subscriber 이전의 다른 경고**: `src/server/cli_input.rs`의 나머지 `env_fallback_*` helper는 여전히 tracing subscriber가 생기기 전에 `tracing::warn!`을 호출하므로, CORS 안내와 같은 방식으로 경고가 사라집니다. `kv_cache_mode_notices` 주석도 같은 한계를 기록하고 있습니다. 근본적인 해결은 이 PR이 CORS에 한 것처럼 모든 시작 안내를 `start_server`를 통해 전달하는 것입니다.
- **Guard 범위**: `src/cli/mod.rs`의 `arg_conflict_partners`는 인자별 `conflicts_with`만 읽습니다. `ArgGroup`으로 선언한 충돌, `multiple(false)` 그룹 소속, `exclusive` 인자는 보고하지 않으므로, 그런 형태의 env 바인딩은 guard를 통과합니다. 현재 해당하는 인자는 없습니다.
- **테스트의 플래그 감지**: CORS 테스트 helper `apply_cors_origin_env`는 `argv.contains`로 플래그를 감지합니다. 따라서 `--cors-origins=value` 형태는 CORS 우선순위 테스트가 아니라 `long_cli_flag_was_set` 자체의 테스트로만 검증됩니다.
