# 기술 보고서: PR #2027 - mlxcel-server download의 --help, -h, --usage 지원

**날짜**: 2026-09-29
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2027은 `mlxcel-server download --help`(및 `-h`, `--usage`)가 도움말을 출력하는 대신 "unexpected argument" 오류와 함께 종료 코드 2로 실패하던 문제를 수정한다. llama-server b10621이 도움말 옵션을 `-h`/`--help`/`--usage`로 표기하고 clap이 자동 생성하는 도움말 인자는 별칭(alias)을 가질 수 없기 때문에, 상위 `Cli` 명령은 도움말 인자를 수작업으로 선언해 두었다. 이 인자에 `global` 표시가 없어서 `disable_help_flag` 설정만으로는 `download` 서브커맨드에 도움말 플래그가 전혀 전달되지 않았다. `global = true`를 추가해 세 가지 표기 모두 `download`에서 동작하도록 하면서, 변경 전 캡처와의 diff로 `mlxcel-server --help` 자체의 출력이 바이트 단위로 동일하게 유지됨을 검증했다.

## 1. 문제 정의

### 1.1 배경

`mlxcel`과 `mlxcel-server` 두 바이너리 모두 `download` 서브커맨드에서 `DownloadArgs`(`src/downloader/cli.rs`)를 파싱한다. `mlxcel download --help`는 정상 동작했고, `mlxcel-server --help`와 `mlxcel-server help download`도 정상 동작했지만, `mlxcel-server download --help`만은 clap이 "unexpected argument '--help' found" 오류를 내며 종료 코드 2로 실패했다. 상위 `Cli` 구조체는 `disable_help_flag = true`를 설정하고 `-h`/`--help`/`--usage` 인자를 `action = clap::ArgAction::Help`로 직접 선언한다. clap 4의 derive 매크로가 자동 생성하는 도움말 플래그에는 `visible_alias`를 붙일 수 없는데, mlxcel-server가 CLI 호환성을 맞추는 대상인 llama-server의 b10621 릴리스는 이 옵션을 세 가지 표기 모두로 받아들이기 때문이다.

### 1.2 기존 문제

- **`download`에 도움말 플래그 부재**: clap 4.6에서 `disable_help_flag`는 서브커맨드에도 "자동 생성 도움말 인자가 없는" 상태를 전파하지만, 수작업으로 선언한 대체 인자는 명시적으로 `global = true`를 붙여야만 서브커맨드까지 전달된다. 이 표시가 없었기 때문에 `download`에는 비활성화된 기본값도, 수작업 대체 인자도 전혀 존재하지 않았다.
- **드러나지 않는 비대칭**: `mlxcel-server --help`, `mlxcel-server help download`, `mlxcel download --help`는 모두 정상 동작했기 때문에 수동 테스트 중 이 누락을 발견하기 어려웠다. `mlxcel-server download --help`를 직접 호출하는 경우에만 실패가 드러났다.

### 1.3 위험도 평가

낮음. `global = true`는 기존에 이미 존재하는 도움말 인자를 clap이 어디까지 찾아보는지만 바꿀 뿐, 새 인자를 추가하지 않으며, `ArgAction::Help`는 `download`의 나머지 로직이 실행되기 전에 파싱을 즉시 중단시킨다. 유일하게 변하면 안 되는 것은 최상위 `--help` 출력이다. `global` 인자는 도움말에 한 번만 문서화되며 헤딩 배치가 달라질 수 있어, 이 부분은 가정에 그치지 않고 직접 검증했다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 1개 (`src/bin/mlx_server.rs`) |
| 커밋 | 1개 |
| 추가/삭제 줄 수 | 29 / 0 |

- `Cli::help`(수작업으로 선언한 `-h`/`--help`/`--usage` 인자)의 `#[arg(...)]` 속성에 `global = true`를 추가해, 최상위 표기뿐 아니라 `mlxcel-server download --help`/`-h`/`--usage`로도 도달할 수 있게 했다.
- 변경 근거는 필드의 기존 `///` 문서 주석에 덧붙이지 않고 그 바로 아래에 일반 `//` 주석으로 남겼다. clap은 `///` 문서 주석을 해당 인자의 `--help` 설명 텍스트로 그대로 렌더링하므로, 문서 주석을 늘리면 `mlxcel-server --help`의 `-h, --help` 줄 출력이 바뀌게 된다. `//` 주석은 컴파일된 도움말 텍스트에 포함되지 않으면서도 향후 유지보수자에게 동일한 정보를 전달한다.
- 기존 `the_download_subcommand_does_not_take_negative_numbers` 테스트 옆에 `the_download_subcommand_accepts_all_three_help_spellings` 테스트를 추가했다. `["--help", "-h", "--usage"]`를 순회하며 각각에 대해 `Cli::try_parse_from(["mlxcel-server", "download", flag])`가 `ErrorKind::DisplayHelp`를 반환하는지 검증한다. 수정 전 코드에서는 이 테스트가 `ErrorKind::UnknownArgument`로 실패함을 확인했다.

## 3. 기술적 선택과 그 이유

### 3.1 서브커맨드 전용 `ServerDownloadArgs` 래퍼 대신 `global = true` 선택

**배경**: 이슈는 두 가지 접근을 제시했다. 우선 기존에 수작업으로 선언한 도움말 인자에 `global`을 표시하는 방법, 그리고 첫 번째 방법이 `mlxcel-server --help` 출력을 바꿀 경우의 대안으로 `DownloadArgs`를 플래튼(flatten)하고 동일한 도움말 인자를 다시 선언하는 서버 전용 `ServerDownloadArgs` 구조체를 도입하는 방법이다.

**근거**: 설명 텍스트를 문서 주석 밖으로 옮긴 뒤 실행한 바이트 단위 동일성 검증(`diff`를 변경 전 `mlxcel-server --help` 캡처와 비교)이 통과했으므로 `global = true`만으로 충분했다. 대안이었던 래퍼 구조체는 필요하지 않았고, 덕분에 상위 구조체의 네 가지 `#[arg(...)]` 속성을 그대로 다시 유지보수해야 하는 두 번째 구조체를 만들지 않아도 되었다.

### 3.2 렌더링되는 도움말 텍스트를 보호하기 위한 주석 배치

**배경**: 해당 필드의 기존 `///` 문서 주석("Print usage and exit. Declared by hand, with `disable_help_flag`...")은 단순한 설명이 아니라, clap이 모든 `--help` 호출에서 `-h, --help` 줄의 설명으로 그대로 컴파일해 넣는 텍스트다.

**근거**: 처음 구현했을 때는 `global = true`의 근거를 이 문서 주석에 그대로 덧붙였는데, 컴파일과 새 플래그 관련 검사는 모두 통과했지만 최상위 `--help`의 바이트 단위 동일성 diff에서 `-h, --help` 설명 줄이 한 줄 달라졌다는 차이가 발견되었다. 같은 설명을 `#[arg(...)]` 속성 바로 위의 `//` 줄로 옮기자, 향후 유지보수자를 위한 근거는 소스에 그대로 남으면서도 컴파일된 도움말 텍스트에는 반영되지 않게 되었고, 이후 diff는 깨끗했다.

## 4. 검증

- `mlxcel-server --help` 출력을 변경 전 캡처와 바이트 단위로 diff: 차이 없음.
- `mlxcel-server download --help`, `-h`, `--usage`와 최상위 `--help`, `-h`, `--usage`: 모두 종료 코드 0.
- `mlxcel-server download -1`: 여전히 `UnknownArgument`로, 이번 수정의 영향을 받지 않음(서브커맨드별 `allow_negative_numbers` 설정은 `global` 도움말 전파와 무관).
- `cargo test --release --features metal,accelerate --bin mlxcel-server the_download_subcommand`: 2/2 통과(새 테스트와 기존 음수 회귀 테스트).
- `cargo test --release --features metal,accelerate --test cli_help_consistency`: 27/27 통과.
- `cargo clippy --release --features metal,accelerate --bin mlxcel-server --tests -- -D warnings`: 이상 없음.
- `cargo fmt --check`와 `scripts/ci/check_cross_repo_refs.py`: 이상 없음.
- PR #2023(`src/main.rs`만 수정하고 `src/bin/mlx_server.rs`는 건드리지 않음) 머지 이후 `origin/main`으로 리베이스한 뒤 위 검증을 모두 다시 실행해, 두 변경 사이에 상호작용이 없음을 확인했다.

## 5. 관련 작업

- 이슈 #2025: 이번 변경의 출처 이슈.
- 이슈 #1448: 수작업 도움말 인자를 도입한 원래 근거(b10621의 `-h`/`--help`/`--usage` 표기).
- 이슈 #1459 / `allow_negative_numbers`: 서브커맨드로 전파되지 않는 설정으로, `download -1` 회귀 테스트가 이 PR의 수정으로 영향받지 않음을 함께 확인하게 한 배경.
