# 기술 보고서: PR #2028 - mlxcel 및 mlxcel-server의 낡은 문서 주석 예시 수정

**날짜**: 2026-09-29
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2028은 PR #2023을 구현하고 리뷰하는 과정에서 발견한 `--help` 출력의 문서 주석 결함 두 건을 수정한다. `mlxcel inspect --help`의 Examples 블록은 필수 옵션인 `-m`/`--model` 플래그 없이 호출 예시를 보여주고 있었는데, 그대로 실행하면 파싱에 실패한다. `mlxcel-server`의 `download` 서브커맨드 요약은 PR #2023이 `main.rs`의 형제 요약들에 적용한 "마침표 + 짧은 설명" 스타일이 도입되기 이전의, 구두점 없는 한 줄 요약 그대로였다. 두 건 모두 문서 주석만 변경하며 로직에는 영향이 없다.

## 1. 문제 정의

### 1.1 배경

이슈 #1657 / PR #2023은 `src/main.rs`의 `Commands` 열거형 중 다섯 개 서브커맨드에 걸쳐 문서 주석 스타일을 통일했다. 구두점 없는 한 줄 요약은 마침표와 짧은 확장 설명 문단으로 바뀌었고, 여러 서브커맨드에 그대로 복사해 실행할 수 있는 `Examples:` 블록이 추가되었다. 이번에 다룬 두 결함은 해당 PR의 범위 밖에 있었지만 같은 문서 주석 하위 시스템과 같은 검증 표면(`tests/cli_help_consistency.rs` 및 수동 `--help` 점검)을 공유하므로, 이 저장소의 PR 범위 설정 관례에 따라 하나의 이슈와 하나의 PR로 묶었다.

### 1.2 기존 문제

- **항목 1, `src/main.rs`**: `InspectArgs::model`(`#[arg(short, long, value_name = "PATH_OR_REPO_ID")]`, 필수인 비위치 옵션이며, `mlxcel inspect --help`가 출력하는 `Usage: mlxcel inspect [OPTIONS] --model <PATH_OR_REPO_ID>`로 확인됨)는 `-m`/`--model`을 요구한다. `Commands::Inspect` 문서 주석의 `Examples:` 블록은 대신 위치 인자만 사용하는 호출(`mlxcel inspect models/llama-3.2-1b-instruct-4bit`)을 보여주었는데, 그대로 실행하면 `error: unexpected argument 'models/llama-3.2-1b-instruct-4bit' found`로 실패한다.
- **항목 2, `src/bin/mlx_server.rs`**: `Commands::Download`의 문서 주석은 여전히 `/// Download a HuggingFace model repository snapshot`(마침표도, 확장 설명도 없음)이었고, 이는 `mlxcel-server --help`의 플래튼된 출력(`flatten_help = true`)에서 `mlxcel-server download:` 헤딩 아래 그대로 렌더링되었다. PR #2023은 `main.rs`의 동일한 `Commands::Download` 요약을 새 스타일로 끌어올렸지만, #1657이 `main.rs`로만 범위를 한정했기 때문에 `mlx_server.rs`는 의도적으로 그대로 두었다.

### 1.3 위험도 평가

낮음. 두 변경 모두 `Commands` 열거형 변형(variant)의 `///` 문서 주석에 한정되며, 인자 파싱이나 검증, 런타임 동작을 바꾸지 않는다. 가장 큰 위험은 텍스트를 잘못된 도움말 표면에 렌더링하는 것(예: 플래튼된 최상위 블록에 긴 문단이 새어 들어가는 경우)이었는데, 이는 가정에 의존하지 않고 직접 확인했다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 2개 (`src/main.rs`, `src/bin/mlx_server.rs`) |
| 커밋 | 1개 |
| 추가/삭제 줄 수 | 11 / 4 |

- `src/main.rs`: `Commands::Inspect` Examples 블록의 세 호출 모두에 `-m`을 추가했다(`mlxcel inspect -m models/llama-3.2-1b-instruct-4bit` 및 `--max-tokens`, `--cache-type-k`/`--cache-type-v` 변형).
- `src/bin/mlx_server.rs`: `Commands::Download`의 문서 주석을 구두점 없는 한 줄 요약에서 마침표로 끝나는 요약과 짧은 확장 문단으로 바꾸었다. `main.rs`의 `Download` 변형에 이미 있는 문구(`Fetches an owner/name repo-id into the global mlxcel store...`)를 그대로 재사용했고, `main.rs`가 자신의 `Download` 변형을 렌더링하는 방식과 줄바꿈을 맞추기 위해 `#[command(verbatim_doc_comment)]`를 추가했다.
- `mlx_server.rs`의 `Download` 변형에는 새 `Examples:` 블록을 추가하지 않았다. `DownloadArgs`가 공유하는 `after_help`(`src/downloader/cli.rs`)가 `mlxcel-server download --help`에서 이미 완전한 Examples 섹션을 렌더링하기 때문이며, 이는 #2023이 `main.rs`의 `Download` 변형에 자체 Examples 블록을 두지 않은 것과 동일한 근거다.

## 3. 기술적 선택과 그 이유

### 3.1 수정 전 #2023 이후 줄 번호 기준으로 전제 재확인

**배경**: 이슈는 #2023 병합 이전 줄 번호를 기준으로 작성되었고, (당시 열려 있던) #2023이 줄 번호를 이동시킬 것을 명시적으로 언급하며, 구현자는 줄 번호 대신 변형 이름(`Commands::Inspect`, `Commands::Download`)으로 위치를 맞추도록 안내했다.

**근거**: 이 이슈를 실제로 작업할 시점에는 #2023(`main.rs`의 `Commands::Download` 문서 주석을 새 스타일로 변경)과 #2025(같은 파일의 관련 없는 부분에서 `mlx_server.rs`의 도움말 인자에 `global = true` 추가)가 모두 이미 병합되어 있었다. `main.rs`의 `Commands::Inspect`와 `mlx_server.rs`의 `Commands::Download` 모두 변형 이름으로 다시 위치를 찾아 수정 전에 현재 소스 기준으로 재검증했고, 두 결함 모두 이슈에 기술된 그대로 재현됨을 확인했다(`mlx_server.rs`의 `Download` 문서 주석은 #2023이 `main.rs`로만 범위를 한정했기 때문에 영향을 받지 않았고, `Commands::Inspect`의 Examples 블록 내용도 #2023이 열거형의 다른 위치에 삽입한 내용의 영향을 받지 않았다).

### 3.2 `mlx_server.rs`의 `Download` 변형에 `#[command(verbatim_doc_comment)]` 추가

**배경**: `verbatim_doc_comment`가 없으면 clap은 여러 줄로 된 문서 주석의 문단 텍스트를 터미널 너비에 맞춰 다시 흘려보내는데(reflow), 이는 의도적으로 다른 파일의 문구와 맞춰둔 손수 정리한 문단과 다른 줄바꿈을 만들어낼 수 있다.

**근거**: `main.rs`의 `Download` 변형은 동일한 문단에 이미 이 속성을 붙여두고 있다. 여기에도 같은 속성을 추가하면 `mlxcel-server download --help`가 렌더링하는 문단이 `mlxcel download --help`가 동등한 텍스트를 렌더링하는 방식과 바이트 단위로 일치하게 되며, 두 파일이 시간이 지나며 독립적으로 다시 흘려보내져 문구가 서서히 어긋나는 것을 막는다.

### 3.3 확장 문단이 플래튼된 최상위 도움말로 새지 않는지 확인

**배경**: `mlx_server.rs`의 상위 `Cli`는 `flatten_help = true`를 설정하고 있어, 각 서브커맨드의 전체 도움말이 자신의 헤딩 아래 `mlxcel-server --help` 안에 그대로 렌더링된다. 문서 주석에 긴 long about 문단을 덧붙이면, 첫 줄만 쓰이는 짧은 about과 달리 의도하지 않은 위치에 나타날 위험이 있다.

**근거**: clap의 about/long_about 분리 동작이 기대대로일 것이라 가정하는 대신 플래튼된 블록을 직접 확인했다. `mlxcel-server --help`의 `mlxcel-server download:` 헤딩 아래에는 첫 줄인 `Download a HuggingFace model repository snapshot.`만 렌더링되었고, "아래 Examples 섹션을 참고하라"는 언급을 포함한 전체 확장 문단은 `mlxcel-server download --help`에서만 렌더링되었다. 이곳에서는 DownloadArgs의 Examples 블록이 바로 뒤이어 나오므로, 새 문단의 상호 참조가 실제로 나타나는 표면에서는 정확함을 확인했다.

## 4. 검증

- `mlxcel inspect -m models/llama-3.2-1b-instruct-4bit`: "unexpected argument" 오류 없이 깨끗하게 파싱되며 모델 리졸브 단계로 진행하고, 다운로드 404로 종료 코드 1을 반환함. 문서 주석의 예시 경로가 이 환경에서 실제 로컬 체크포인트나 HuggingFace 저장소가 아니므로 예상된 결과.
- `mlxcel inspect --help`: Examples 블록의 세 줄 모두에 `-m`이 표시됨.
- `mlxcel-server --help`: `download` 요약이 이제 마침표로 끝남(`Download a HuggingFace model repository snapshot.`).
- `mlxcel-server download --help`: 확장 문단 전체가 렌더링되고 이어서 기존 공유 Examples 섹션이 나타남.
- `cargo test --release --features metal,accelerate --test cli_help_consistency`: 27/27 통과.
- `cargo clippy --release --features metal,accelerate --bin mlxcel --bin mlxcel-server --tests -- -D warnings`: 이상 없음.
- `cargo fmt --check`와 `scripts/ci/check_cross_repo_refs.py`: 이상 없음.

## 5. 관련 작업

- 이슈 #2024: 이번 변경의 출처 이슈.
- 이슈 #1657 / PR #2023: 이번 PR이 `mlx_server.rs`까지 확장한 "마침표 + 설명" 문서 주석 관례를 확립함.
- 이슈 #2025 / PR #2027: 같은 파일(`src/bin/mlx_server.rs`)에 대한 직전 수정으로, 같은 체인에서 이 PR 바로 앞에 병합되었다. `Cli::help` 인자만 건드리고 `Commands::Download` 문서 주석은 건드리지 않아 겹치지 않음을 확인했다.
