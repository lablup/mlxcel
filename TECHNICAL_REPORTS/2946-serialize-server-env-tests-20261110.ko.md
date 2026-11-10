# PR #2946: mlxcel-server env 테스트 직렬화와 lock poison 연쇄 실패 제거

**작성일**: 2026-11-10
**상태**: 완료 (PR은 draft로 열려 있고 #2937을 닫음. 이 보고서를 작성할 때 전체 `make verify-test` 실행은 아직 진행 중이었음)

## 요약

이슈 #2937은 M1 Ultra에서 `cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast`를 실행해 8개 실패가 나왔다고 보고했다. 이 명령은 로컬 에이전트 지침에 적힌 것이고 libtest를 멀티스레드로 실행한다. CI와 같은 게이트인 `make verify-test`와 nightly workflow는 의도적으로 `-- --test-threads=1`을 붙인다. #1092와 #1210이 하나의 바이너리 안에서 MLX 기반 테스트 여러 개가 동시에 돌 때 Metal 디바이스 fault가 난다는 것을 추적했기 때문이다. 따라서 보고된 실패 대부분은 설계상 게이트 밖에 있다. 이 PR은 스레드 설정과 무관하게 실패를 읽기 어렵게 만들던 두 결함을 고친다. 하나는 `mlxcel-server` 인자 테스트 사이의 환경 변수 경쟁이고, 다른 하나는 dense batched decode 테스트의 lock poison 연쇄 실패다. `mlxcel-server` 테스트 모듈의 52개 테스트 전부가 이제 poison을 견디는 단일 `cli_env_lock()` guard를 잡고, 파싱 헬퍼와 `ScopedEnv` 쓰기 함수가 guard를 잡고 있는지 assert한다. 수정 전 트리에서는 `--bin mlxcel-server` 멀티스레드 실행 6번 중 5번이 실패했다. 수정 후에는 바이너리 전체 멀티스레드 실행 15번이 모두 통과했고(첫 커밋에서 10번, 리뷰 후속 커밋 후 5번, 매번 51 passed와 1 ignored), 이슈의 네 테스트 필터 실행도 5번 모두 통과했다. tolerance나 snapshot은 바꾸지 않았다. 3개 파일, +154/-85.

## 1. 문제 정의

### 1.1 문서의 명령과 실제 게이트가 다름

로컬 에이전트 지침의 "Build and test" 블록은 `-- --test-threads=1` 없이 `cargo test --workspace --profile test-fast --features metal,accelerate`를 제시한다. `make verify-test`와 nightly workflow는 동시에 실행되는 MLX 기반 테스트가 Metal 디바이스 fault를 일으키므로 바이너리 안의 테스트 실행을 직렬화한다(#1092, #1210). `CONTRIBUTING.md`도 macOS용으로 멀티스레드 형태를 그대로 적어 두었는데, 같은 문서의 Linux/CUDA 줄에는 이미 `--test-threads=1`이 있었다. 어느 문서를 따르든 기여자는 게이트에서는 절대 나오지 않는 실패를 보게 된다. `src/models/popular_dense_batched_decode_tests.rs`의 glm4 케이스, `qwen4_exp_snapshot_tests.rs`, `dots1_tests.rs`, gemma_unified nvfp4의 테스트 하나씩, 그리고 한 번은 `src/imagegen/mage_flow/vae/vae_tests.rs`의 layer_norm 테스트가 그렇다. 이 테스트들은 단독으로 실행하거나 `--test-threads=1`로 실행하면 모두 통과하고, #2931의 변경을 stash한 `main`에서도 같은 묶음이 재현되었다.

### 1.2 `mlxcel-server`의 env 경쟁

이슈는 경쟁하는 인자 테스트 네 개를 지목했다. `legacy_server_mode_*`(두 테스트), `no_prompt_cache_flag_disables_prompt_cache`, `image_model_without_model_leaves_the_chat_path_empty`다. 이 테스트들은 `CLI_ENV_LOCK`을 잡지 않고 argv를 파싱했고, 그 사이 형제 테스트들은 그 lock 아래에서 `ScopedEnv`로 `LLAMA_ARG_*`와 `MLXCEL_*` 변수를 설정했다.

근본 원인은 네 테스트보다 넓다. clap 4는 command를 만들 때 각 `env = "..."` binding을 읽는다(`Arg::env`가 `env::var_os`를 호출). 그래서 `Cli`를 만들거나, help를 렌더링하거나, argv를 파싱하거나, 자체 env fallback이 있는 `build_startup_input`을 호출하는 테스트는 모두 프로세스 env를 읽는다. 점검 결과 이렇게 env를 읽으면서 lock을 잡지 않는 테스트가 40개 있었다. help와 flag introspection 테스트, 그리고 CORS 자식 프로세스 테스트의 부모도 포함된다. 그 자식은 spawn 시점에 부모가 가진 env를 그대로 물려받는다. 지목된 네 테스트는 assertion이 우연히 형제 테스트의 값을 감지한 경우일 뿐이었다. 이 경쟁 때문에 checkout에 크기 0인 `legacy.log` 파일이 남기도 했다. 형제 테스트가 `LLAMA_LOG_FILE=legacy.log`를 잡고 있는 동안 다른 테스트가 logging 설정을 만든 것이다.

### 1.3 Poison 연쇄 실패

기존 `CLI_ENV_LOCK` 12곳 중 7곳은 `.expect("CLI env lock")`로 lock을 잡았고, dense batched decode 테스트는 `test_guard()`를 `.expect("dense test lock")`로 잡았다. 테스트 하나가 mutex를 잡은 채 panic하면, 이후 그 mutex를 잡는 모든 테스트가 `PoisonError`로 실패했다. 실제 실패 하나가 여러 개로 보고되어 원래 실패를 찾기 어려웠다.

## 2. 변경 내용

### 2.1 `src/bin/mlx_server.rs` 테스트 모듈

- **단일 guard.** `cli_env_lock()`은 `MutexGuard`를 감싼 `CliEnvGuard`를 반환하고, `unwrap_or_else(PoisonError::into_inner)`로 poison에서 복구한다. `expect` 형태와 poison 허용 형태가 섞여 있던 손으로 쓴 lock 12곳을 대체한다.
- **lock 보유 플래그.** thread-local `CLI_ENV_LOCK_HELD`는 guard를 잡을 때 설정되고 `Drop`에서 해제된다. `cli_env_lock()`은 먼저 이 플래그가 꺼져 있는지 assert하므로, 한 스레드에서 lock을 두 번 잡으면 deadlock 대신 메시지와 함께 panic한다.
- **assert하는 헬퍼.** `cli_command()`는 `Cli::command()`를, `try_parse_cli()`는 `Cli::try_parse_from()`을 감싸고, `parse_server_args()`는 `try_parse_cli()`를 거친다. 세 함수 모두 `assert_cli_env_lock_held()`를 호출하며, 그 메시지가 고치는 방법(`let _lock = cli_env_lock();`)을 알려준다. 테스트 안의 직접 `Cli::command()`, `Cli::try_parse_from()` 호출은 모두 바뀌었고, 개별 테스트의 `use clap::CommandFactory` import도 사라졌다.
- **assert하는 쓰기 함수.** 리뷰 후속 커밋(`e7ec7451`)이 같은 assertion을 `ScopedEnv::set`과 `ScopedEnv::unset`에도 넣어 경쟁의 쓰기 쪽까지 검사한다. `set_var`와 `remove_var`의 `SAFETY` 주석도 관례 대신 이 assertion을 가리킨다.
- **적용 범위.** 모듈의 `#[test]` 함수 52개 전부가 guard를 정확히 한 번 잡는다. 자식을 spawn하는 부모 `both_cors_origin_envs_without_a_flag_fail_startup`과 ignore된 `both_cors_origin_envs_child`도 포함된다. 부모 테스트의 주석은 다시 썼다. 예전 주석은 다른 테스트가 lock 없이 `build_startup_input`을 호출한다고 했는데 이제 사실이 아니므로, 부모가 spawn하는 동안 lock을 잡는 이유를 설명하도록 바꿨다.
- **drop 순서.** guard 문서는 `_lock`을 어떤 `ScopedEnv`보다 먼저 바인딩하라고 안내한다. 지역 변수는 역순으로 drop되므로 lock이 풀리기 전에 env가 복원된다.

### 2.2 `src/models/popular_dense_batched_decode_tests.rs`

`dense_contract!` 매크로가 `test_guard()`를 `unwrap_or_else(PoisonError::into_inner)`로 잡고, 이유를 주석으로 남겼다. 실패하는 케이스는 이제 혼자 실패한다.

### 2.3 `CONTRIBUTING.md`

macOS 테스트 명령이 직렬 형태로 바뀌었고 `make verify-test`가 실행하는 것과 같은 명령이라고 표시했다. 왜 직렬이어야 하는지 한 문장으로 설명하고 #1092와 #1210을 인용한다. 같은 문서 4단계가 PR 시점에는 테스트를 게이트하지 않는다고 하므로 이것을 "the gate"라고 부르지 않고, `make verify-test`와 nightly workflow가 실행하는 명령이라고 적었다.

## 3. 검증

| 실행 | 수정 전 | 수정 후 |
|------|---------|---------|
| `--bin mlxcel-server` 멀티스레드, 바이너리 전체 | 6번 중 5번 실패(`image_model_without_model_*` 3번, `legacy_server_mode_*`와 `no_prompt_cache_flag_*` 각 1~2번) | 15번 모두 통과(첫 커밋에서 10번, 후속 커밋 후 5번, 매번 51 passed, 1 ignored) |
| 이슈의 네 테스트 필터, 멀티스레드 | 간헐적 실패 | 5번 모두 통과 |
| `--lib popular_dense_batched_decode`, 멀티스레드 | 측정 안 함 | 통과(15 passed) |
| 멀티스레드 실행 후 남는 `legacy.log` | 생성됨 | 5번 실행 동안 생성 안 됨 |

- 반대 방향 검사: 테스트 하나에서 guard를 지우면 그 테스트가 assertion 메시지로 실패한다. 즉 검사가 실제로 동작하며 관례를 문서화하는 데 그치지 않는다.
- `cargo clippy --lib --tests --bin mlxcel-server --features metal,accelerate -- -D warnings`와 `cargo fmt --check`가 깨끗하다.
- 수치 tolerance를 넓히거나 snapshot을 바꾸지 않았다. MLX 기반 실패 중 `--test-threads=1`에서 재현되는 것이 없었으므로, 이슈가 정한 변경 조건이 한 번도 충족되지 않았다.
- 이슈의 마지막 인수 조건인 전체 `make verify-test`는 이 보고서를 작성할 때 worktree에서 실행 중이었다. 그 결과는 여기에 기록하지 않았다.

## 4. 기술적 선택과 그 이유

### 4.1 env 읽기를 없애지 않고 env 테스트를 직렬화

**배경:** clap은 `Cli`를 만들 때마다 env를 읽으므로, 테스트는 `Cli`를 만들지 않는 것 외에는 env 읽기를 피할 수 없다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| build, parse, spawn하는 모든 테스트가 lock을 잡음(채택) | 작고 기계적인 변경, 기존 `CLI_ENV_LOCK` 설계와 일치, 테스트가 실제 clap env binding을 계속 검사 | 바이너리 안에서 52개 테스트가 직렬화됨, 새 테스트가 guard를 잡아야 함 |
| 테스트에서 env binding 제거(예: env를 끈 `Command`로 파싱) | 테스트가 병렬로 실행됨 | `LLAMA_ARG_*` 호환성 표면인 env binding을 더 이상 검사하지 못함 |
| CORS 테스트처럼 env를 바꾸는 테스트를 자식 프로세스로 옮김 | 진짜 격리 | 테스트마다 프로세스 spawn, diff가 훨씬 커짐 |

이 테스트들은 인자 파싱만 하고 빨리 끝나므로 직렬화 비용이 작다. 게이트는 어차피 이미 직렬로 실행한다.

### 4.2 관례 대신 실패하는 assertion

관례로만 지키는 lock이 lock 없는 테스트 40개를 낳았다. 이제 헬퍼가 assert하므로 guard를 잊은 새 테스트는 스레드 설정과 상관없이 첫 실행에서 결정적으로 실패하고, 메시지가 고치는 방법을 알려준다. assertion이 없으면 그런 테스트는 직렬 게이트에서는 통과하고 멀티스레드 실행에서만 경쟁한다. thread-local 플래그는 두 번째 획득을 멈춤 대신 분명한 panic으로 바꾼다. 테스트가 이미 잡은 상태에서 헬퍼 안에서 다시 잡기 쉬우므로 중요하다.

### 4.3 두 lock 모두 poison 허용

env lock은 데이터를 보호하지 않고 배타성만 보장한다. lock을 잡은 채 panic한 테스트는 `ScopedEnv`가 guard 뒤에 바인딩되어 먼저 drop되므로 이미 env를 복원한 상태다. 따라서 guard를 복구해도 안전하고, 실제 실패 하나가 수십 개의 `PoisonError` 실패로 번지는 것을 막는다. dense 테스트 lock도 케이스들을 직렬화할 뿐이므로 같은 논리가 적용된다.

### 4.4 테스트 스레딩이 아니라 문서를 고침

이슈는 멀티스레드 실행을 통과시키려고 tolerance를 넓히거나 snapshot을 바꾸는 것을 명시적으로 배제했다. MLX 기반 테스트를 동시 실행에서도 안전하게 만드는 일은 #1092와 #1210의 문제이고, 프로젝트는 이미 직렬 게이트로 답했다. 그 부분의 수정은 게이트를 정확히 문서화하는 것이다.

## 5. 한계와 남은 일

- **`CLAUDE.md`는 PR에 없다.** 이슈의 첫 인수 조건은 `CLAUDE.md`를 지목하지만, 이 checkout에서 그것은 추적되지 않는 `AGENTS.md`를 가리키는 gitignore된 로컬 symlink다. 그 "Build and test" 줄은 여전히 멀티스레드 명령을 보여주며 트리 밖에서 고쳐야 한다. 추적되는 수정은 `CONTRIBUTING.md`가 담는다.
- **MLX 기반 병렬 전용 실패는 남아 있다.** glm4 dense 케이스, qwen4_exp snapshot, dots1, gemma_unified nvfp4, vae layer_norm 테스트는 멀티스레드 실행에서 여전히 실패할 수 있다. 이 PR은 dense 테스트의 poison 연쇄만 없앴고, 근본적인 동시성 fault는 설계상 범위 밖이다.
- **`build_startup_input` 자체는 assert하지 않는다.** fallback으로 env를 읽지만 `ServerArgs`를 받는데, 테스트에서 모든 `ServerArgs`는 본문 전체에서 guard를 잡고 있는 테스트 안의 assert하는 `parse_server_args`에서 나온다. 다른 방법으로 `ServerArgs`를 만드는 미래의 테스트는 이 검사를 우회한다.
- **guard의 범위.** 이 테스트 모듈 하나만 덮는다. clap으로 `LLAMA_ARG_*`를 읽는 다른 바이너리나 모듈의 테스트는 덮지 않으며, env를 바꾸는 테스트와 같은 프로세스를 공유할 때만 문제가 된다.
- **`make verify-test`는 이 보고서 시점에 확인되지 않았다**(3절 참고).

## 6. 학습 포인트

- clap의 `Arg::env`는 값이 없을 때만이 아니라 command를 만들 때 환경 변수를 읽는다. help 렌더링이나 flag introspection도 env를 읽으므로, "이 테스트는 help 텍스트만 읽는다"는 것이 env와 무관하다는 증거가 아니다.
- 자식 프로세스는 spawn 시점에 부모의 환경을 물려받는다. 테스트를 자식으로 격리해도, 부모가 spawn하는 동안 lock을 잡지 않으면 형제 테스트의 `ScopedEnv`로부터 보호받지 못한다.
- mutex가 데이터가 아니라 배타성을 지킬 때, 테스트에서는 `unwrap_or_else(PoisonError::into_inner)`를 쓰는 편이 낫다. `expect`는 실패 하나를 여러 개로 늘린다.
- 같은 명령이 여러 곳에 문서화되어 있으면 각 사본을 CI가 실제로 실행하는 것과 비교해야 한다. 여기서는 `CONTRIBUTING.md`의 Linux 줄은 이미 직렬이었고 macOS 줄은 그렇지 않았다.
