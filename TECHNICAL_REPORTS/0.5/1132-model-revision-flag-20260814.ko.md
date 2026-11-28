# 기술 보고서: PR #1132 - feat(cli): add --revision to the -m/--model resolver

**작성일**: 2026-08-14
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #1132는 이슈 #1113을 해결한다. `mlxcel download`는 `--revision`을 받는데 `-m/--model` 해석기는 받지 않아서, 고정한 리비전을 받아 놓고도 repo-id로 실행할 수 없었다. 이제 `generate`, `run`, `serve`, `inspect`, `mlxcel-server`에서 `--revision <REV>`를 쓸 수 있다.

중요한 부분은 배관이 아니다. 이슈가 정확히 지적했듯 배관은 이미 끝까지 연결되어 있었다. 핵심은 이슈가 명시적으로 요구한 범위 결정이다. **mlxcel 스토어는 리비전 네임스페이스가 없다.** 따라서 리비전을 모든 곳에서 존중할 수 없다. 이 PR은 올바르게 존중할 수 있는 곳에서만 존중하고, 나머지는 조용히 엉뚱한 리비전을 돌려주는 대신 크게 실패한다.

이 결정 과정에서 기존 결함도 드러났다. 같은 이유로 `mlxcel download --revision`은 지금도 잘못된 리비전을 돌려줄 수 있다. 이 PR이 만든 문제가 아니며 후속 과제로 분리했다.

---

## 1. 문제 정의

### 1.1 배경

`resolve_repo_id`는 이미 `revision: Option<&str>`을 받아 `store::hf_cache_snapshot`과 `DownloadOptions`로 넘기고 있었다. 두 공개 진입점만 `None`을 하드코딩했고, 해석기 주석이 그 공백을 의도적으로 기록해 두었다.

> `revision`은 HF 캐시 스냅숏 리비전(브랜치 / 태그 / 커밋)을 고른다. `None`은 `main`이다. CLI 서브커맨드는 현재 `--revision` 플래그를 노출하지 않으므로 `None`을 넘기며, 이는 `mlxcel download`의 기본값과 일치한다.

### 1.2 기존 문제점

- **고정한 리비전을 받을 수는 있어도 실행할 수 없었다.** `mlxcel download owner/name --revision v2`는 되지만, 이어서 `mlxcel run owner/name`은 `main`으로 해석했다.
- **스토어는 두 리비전을 함께 담을 수 없다.** `store::model_dir_under`는 리비전 성분 없이 `<models_root>/<owner>/<name>`을 구성한다.
- **다운로더는 이미 채워진 디렉터리로의 fetch를 조용히 건너뛴다.** `download_repo_blocking`은 요청된 리비전의 파일 목록으로 `snapshot_complete(&local_dir, &wanted)`를 확인하고, 원하는 파일 이름이 모두 있고 크기가 0이 아니면 "all expected files already present ..., skipping"을 찍고 조기 반환한다. 한 저장소의 여러 리비전은 대개 파일 이름이 같으므로, 성공을 보고하면서 엉뚱한 리비전을 돌려준다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 리비전 지정 요청에 다른 리비전이 조용히 응답 | High | 이 PR의 가드가 없으면 High |
| 로컬 경로에 `--revision`이 무시됐는데 고정했다고 믿음 | Medium | Medium |
| "good first issue" 명목으로 스토어 레이아웃 변경까지 끌어들임 | Medium | 범위 분리로 회피 |

---

## 2. 기술 검토

### 2.1 리비전을 존중할 수 있는 위치

재사용 위치들은 리비전을 기록하는지 여부가 다르고, 그 차이가 설계 전체를 결정한다.

| 위치 | 리비전 인식 | 처리 |
|---|---|---|
| `store::hf_cache_snapshot` | 예 (`refs/<rev>` 또는 커밋 이름 스냅숏 디렉터리) | 평소대로 조회 |
| 레거시 `./models/<basename>` | 아니오 | 리비전 지정 요청에서는 건너뜀 |
| mlxcel 스토어 `<owner>/<name>` | 아니오 | 재사용 대상에서 제외. 이미 차 있으면 다운로드 목적지로도 거부 |
| 네트워크 fetch | 예 (`DownloadOptions.revision`) | 요청한 리비전을 받음 |

건너뛰기는 한계를 덮는 조치가 아니다. 리비전 정보가 없는 위치에서의 히트는 잘못된 리비전 히트와 구분할 수 없으므로, 그것을 돌려주면 `--revision`이 막으려는 바로 그 실패가 된다.

### 2.2 다운로드 후 탐색을 별도 함수로 나눈 이유

`locate_cached_snapshot`이 답하는 질문은 "이 위치가 `revision` 요청에 응답해도 되는가"다. 리비전 지정 요청에서 정보가 없는 위치의 답은 전부 아니오다. 다운로드 이후의 질문은 다르다. "방금 받은 바이트가 어디에 떨어졌는가"이고, 이건 알 수 있다. 다운로드 전에 스토어 디렉터리가 `SnapshotState::Absent`임을 확인했으므로 지금 거기 있는 것은 요청한 리비전이다.

두 번째 질문에 첫 번째 함수를 재사용했다면 잠재 버그가 됐다. 다운로드 후 탐색이 스토어를 건너뛰어 미스가 나고, "clean re-download" 복구로 빠져 다시 받고, 또 미스가 나고, 실제로는 완전한 스냅숏에 대해 "still incomplete afterwards" 오류를 돌려준다. `locate_landed_snapshot`은 해당 리비전의 HF 캐시와 스토어 목적지만 보고, 다운로드 목적지가 될 수 없는 레거시 디렉터리는 의도적으로 보지 않는다.

### 2.3 공개 API 형태

`resolve_model_source(value)`는 인자 하나를 유지하며 `None`으로 위임한다. 이슈는 두 공개 진입점 모두에 인자를 추가하자고 제안했지만, 이 함수는 트리 안에 호출자가 없는 편의 래퍼이고, 넓히면 얻는 것 없이 외부 호출자만 깨진다. 트리의 모든 호출부가 이미 쓰는 `resolve_model_source_with_override`만 매개변수를 받는다.

### 2.4 호환성

이전에 동작하던 명령줄은 모두 동일하게 동작한다. 모든 호출부에서 `revision` 기본값이 `None`이고, `None`일 때 해석기의 탐색 순서와 결과는 바뀌지 않는다. 이는 `no_revision_with_existing_local_path_is_unchanged`와 손대지 않은 기존 테스트들이 고정한다.

---

## 3. 기술적 결정

### 3.1 차 있는 스토어에는 받지 않고 거부한다

| 선택지 | 장점 | 단점 |
|---|---|---|
| 그냥 받는다 | 단순하고 흔한 경우에는 "동작"한다 | fetch가 "이미 있음"으로 건너뛰어져 호출자가 조용히 다른 리비전을 받는다 |
| 스토어 스냅숏을 조용히 재사용 | 오류 경로가 없다 | 같은 결과가 확률이 아니라 확정으로 발생 |
| 스토어에 리비전 네임스페이스 도입 | 완전한 일반해 | `list`, `rm`, `download`와 공유하는 디스크 레이아웃 변경. 이슈 기준 범위 밖 |
| **선택: 두 가지 우회책을 명시하며 거부** | 플래그의 약속이 정확해진다. 경로가 반환되면 그것은 그 리비전이다 | 평범한 실행은 되는데 리비전 지정 실행은 실패할 수 있다 |

오류 메시지가 `mlxcel rm <repo>`와 `--models-dir <PATH>`를 함께 알려 주므로 한 번 읽고 다음 행동을 정할 수 있다.

### 3.2 로컬 경로 + `--revision`은 오류

1단계는 존재하는 경로를 그대로 반환하며 리비전을 맞춰 볼 대상이 없다. 오류를 내면 실수를 짚어 주고, 무시하면 사용자는 무언가를 고정했다고 믿게 된다. 플래그의 계약도 균일해진다. 리비전을 존중하거나 거부할 뿐, 조용히 무시하는 경우는 없다.

### 3.3 스토어 레이아웃 작업을 분리한다

이슈는 2번 항목이 규모를 결정하며 스토어 레이아웃을 바꿔야 하면 분리하라고 했다. 바꿔야 하므로 분리했다. 이 PR에는 레이아웃 변경이 없고, 후속 과제가 네임스페이스 도입과 뿌리가 같은 기존 `mlxcel download --revision` 충돌을 함께 가져간다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 12 |
| 신규 테스트 | 5 |
| 스토어 레이아웃 변경 | 0 |
| `--revision` 없을 때 동작 변경 | 0 |

### 영역별 변경

| 영역 | 파일 | 요약 |
|---|---|---|
| 해석기 | `src/downloader/resolver.rs` | `revision` 매개변수, `locate_cached_snapshot`의 리비전 게이팅, 신규 `locate_landed_snapshot`, 오류 두 개, 모듈 문서의 Revisions 절, 낡은 "`--revision` 플래그 없음" 주석 교체 |
| CLI | `src/main.rs`, `src/commands/run.rs`, `src/bin/mlx_server.rs` | `ModelOptions`, `InspectArgs`, `ServeArgs`, `RunArgs`, `ServerArgs`에 `--revision` |
| 호출부 | `src/commands/{generate,chat,serve,inspect}.rs`, `src/bin/mlx_server.rs` | 다섯 곳 모두 연결. REPL 경로를 위해 `ChatOptions`가 값을 보유 |
| 테스트 | `src/downloader/resolver_tests.rs`, `src/commands/{generate,serve}_tests.rs` | 신규 5개, 픽스처가 새 필드 초기화 |
| 문서 | `CHANGELOG.md` | `## [Unreleased]` / `### Added` 항목 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `55033e37` | feat | feat(cli): add --revision to the -m/--model resolver |
| `89e8cc39` | test | test: initialize the new revision field in the arg-struct fixtures |

---

## 5. 검증 및 후속 과제

### 통과

- `cargo test --profile test-fast --features cuda --lib downloader::resolver`: 33 passed, 0 failed. 신규 5개는 로컬 경로 거부, 레거시 디렉터리 건너뛰기, 스토어 건너뛰기, 리비전 없는 로컬 경로의 불변 동작, 차 있는 스토어 거부(메시지가 다운로드 실패가 아님을 함께 단언해 아무것도 받지 않았음을 확인)를 덮는다.
- `cargo test --profile test-fast --features cuda --test cli_help_consistency`: 25 passed. `the_two_server_binaries_accept_the_same_flag_surface`가 포함되어 `mlxcel serve`와 `mlxcel-server`가 새 플래그에서 어긋나지 않게 잡아 준다.
- `cargo test --profile test-fast --features cuda --bin mlxcel serve_`: 13 passed. `... validate_pipeline_parallel`: 5 passed.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`: 통과.
- `cargo fmt --all -- --check`: 통과.
- 빌드된 바이너리 기준: `generate`와 `run`에서 의도한 문구로 `--revision`이 노출되고, `-m /tmp --revision v2`는 로컬 경로 오류로 거부되며, 차 있는 스토어 + `--revision v2`는 두 우회책을 명시하며 네트워크 요청 없이 거부된다.

### clippy 실행에 관한 기록

첫 clippy 실행은 출력을 `grep | tail`로 넘겼는데, 파이프라인의 종료 상태는 cargo가 아니라 마지막 단계의 것이다. 통과한 것처럼 보였지만 실제로는 bin 크레이트 테스트 타깃에서 `E0063` 두 개로 실패하고 있었다. `sample_generate_args`와 `sample_args`가 `ModelOptions`와 `ServeArgs`를 명시적 필드 초기화로 만들기 때문이다. 종료 코드를 직접 받도록 다시 돌리고 픽스처 두 개를 고친 뒤의 결과가 위 기록이다. 출력을 필터링하는 검증 명령 전반에 해당하는 교훈이다.

### 후속 후보

- **mlxcel 스토어에 리비전 네임스페이스 도입.** 위의 모든 제약이 풀리고, 두 번째 리비전 요청이 차 있는 스토어 디렉터리에서 "이미 있음"으로 건너뛰어져 첫 리비전이 반환되는 기존 `mlxcel download --revision` 충돌도 고쳐진다. 이 충돌은 지금 `main`에 존재하며 이 변경과 무관하다.
- 후속 과제가 반영되면 이 PR의 `revision_store_occupied_error`는 죽은 코드가 되므로 함께 제거해야 한다.
