# PR #2533: aux.rs 이름 변경과 Windows 장치 이름 검사

**작성일**: 2026-10-22
**상태**: 완료 (작성 시점에 PR 열림, #2531 종료)

## 요약

`src/lib/mlxcel-xla/src/aux.rs` 하나 때문에 Windows에서는 저장소를 체크아웃할 수 없었다. `AUX`는 확장자와 대소문자에 상관없이 예약된 장치 이름이고, Git for Windows는 기본값 `core.protectNTFS=true` 상태에서 이런 경로를 쓰지 않는다. `git clone`은 체크아웃 도중 멈추면서 모든 파일이 삭제된 것처럼 보이는 인덱스를 남겼고, `git reset --hard main`은 `invalid path`로 실패했다. 이 파일은 `f128d1a5`(#922)부터 트리에 있었지만 CI 작업이 전부 macOS와 Linux에서 돌기 때문에 아무도 몰랐다. 이 PR은 파일과 Rust 모듈 이름을 `auxiliary`로 바꾸고, `scripts/ci/check_windows_reserved_paths.py`를 CI 작업과 `make` 타깃으로 추가했다. 이제 예약 이름이 들어오면 어느 플랫폼에서든 실패한다. Rust 쪽은 순수한 이름 변경이라 동작은 그대로다. 리뷰와 보안 점검에서 CRITICAL, HIGH, MEDIUM 지적은 없었고 LOW 세 건은 리뷰 중에 고쳤다.

## 1. 문제 정의

Windows가 예약한 이름은 `CON`, `PRN`, `AUX`, `NUL`, `COM0`-`COM9`, `LPT0`-`LPT9`, 그리고 `COM`/`LPT` 뒤에 위첨자 `¹`, `²`, `³`가 붙은 것이다. 판정 기준은 이름에서 첫 점 앞부분이다. 그래서 `aux.rs`도 `Aux.md`도 `aux.tar.gz`도 모두 AUX 장치로 취급된다. macOS와 Linux는 이런 이름을 문제없이 받는다. 트리 어디든 이런 경로가 하나만 있으면 크레이트 하나가 아니라 저장소 전체를 Windows에서 체크아웃하지 못한다. 우회법은 sparse checkout뿐인데, 그 상태로는 `mlxcel-xla`를 빌드할 수 없다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/lib/mlxcel-xla/src/aux.rs` → `auxiliary.rs` | `git mv`, 내용 변경 없음 |
| `src/lib/mlxcel-xla/src/lib.rs`와 형제 모듈 10개 | `mod auxiliary;`, `crate::aux::` 임포트 전부 교체, rustfmt 임포트 재정렬 |
| `src/loading/vlm_gemma_xla_tests.rs` | `include_str!` 경로를 새 파일에 맞춤 |
| `scripts/ci/check_windows_reserved_paths.py` (신규) | 추적 중인 경로의 구성 요소 중 예약 stem이 있으면 실패 |
| `scripts/ci/check_windows_reserved_paths_test.sh` (신규) | 임시 저장소 케이스 16개와 실제 트리 |
| `.github/workflows/ci.yml` | `binary-assets` 다음에 `windows-reserved-paths` 작업, `changes` 필터 밖 |
| `Makefile` | `make verify-windows-paths`, Maintenance Targets에 표시 |

## 3. 기술적 선택과 그 이유

### 3.1 파일만이 아니라 모듈 이름까지 바꾼다

`#[path = "auxiliary.rs"] mod aux;` 한 줄이면 체크아웃 문제는 풀린다. 하지만 모듈 이름은 여전히 `aux`로 남고, 나중에 누가 모듈과 파일 이름을 다시 맞추면 문제의 경로가 되살아난다. 그래서 모듈 이름 자체를 바꿨고, `crate::aux::` 호출 지점 12곳이 함께 바뀌었다. `aux_manifest.rs`, `aux_smoke.rs` 같은 형제 파일은 stem이 `aux_manifest`라 예약 이름이 아니므로 그대로 뒀다.

`TECHNICAL_REPORTS/1381-openxla-warning-backlog-20260822.*`에는 아직 `aux.rs`가 나온다. 보고서는 날짜가 박힌 기록이라 손대지 않았다.

### 3.2 판정 규칙

검사기는 `git ls-files -z`가 내놓는 경로를 구성 요소별로 나누고, 각 구성 요소에서 첫 `.` 앞부분을 잘라 뒤쪽 공백을 지운 뒤 casefold해서 고정된 집합과 비교한다. 규칙의 각 부분에는 이유가 있다.

- **마지막 점이 아니라 첫 점.** `aux.tar.gz`도 예약이다. 마지막 점으로 자르면 stem이 `aux.tar`가 되어 놓친다.
- **파일 이름만이 아니라 모든 구성 요소.** 디렉터리 이름이 `nul`이면 그 아래 파일이 전부 깨진다(`a/nul/b.txt`).
- **뒤쪽 공백 제거.** Windows가 공백을 잘라내므로 `aux .txt`도 AUX 장치가 된다. 보안 점검에서 발견해 테스트 케이스와 함께 넣었다.
- **stem 완전 일치.** `aux_manifest.rs`, `console.rs`, `nullable.rs`, `com10.rs`, `data.aux`는 통과해야 하고, 자체 테스트가 하나씩 확인한다.
- **UTF-8 디코딩 명시.** 위첨자 숫자는 ASCII가 아니어서 러너의 로캘에 판정이 흔들리면 안 된다. UTF-8이 아닌 경로가 들어오면 조용히 건너뛰지 않고 예외를 내서 작업을 실패시킨다.

`COM0`과 `LPT0`은 이슈에 적힌 대로 집합에 넣었다. 일부 Windows 버전은 예약하지 않을 수 있지만, 그 결과는 아무도 쓸 것 같지 않은 이름에서의 오탐일 뿐 누락은 생기지 않는다.

### 3.3 CI 배치

작업 구성은 `binary-assets`를 그대로 따랐다. `ubuntu-latest`, `contents: read`, `persist-credentials: false`를 준 `actions/checkout@v7`이고, 비밀값도 `run:` 안의 `${{ }}` 식도 없다. `changes` 필터 밖에 둔 것은 툴체인이 필요 없고 몇 초면 끝나며, 결함이 CI가 쓰는 모든 플랫폼에서 드러나지 않기 때문이다. Rust 변경에만 걸어 두면 `con.md` 같은 문서나 픽스처는 그냥 통과한다. 저장소를 클론하는 Windows 러너로도 잡을 수 있지만, 워크플로에 Windows 작업이 없고 체크아웃 확인만을 위해 하나를 추가하면 정적 검사의 몇 초 대신 실행마다 몇 분이 든다.

### 3.4 make 타깃 이름

타깃 이름은 `verify-windows-reserved-paths`가 아니라 `verify-windows-paths`다. `make help`는 타깃 이름과 설명을 grep해서 섹션을 나누는데, Run Targets 필터가 `(run|serve|generate)`다. "reserved"에 "serve"가 들어 있어서 긴 이름은 Maintenance와 Run Targets 양쪽에 나왔다. 설명에서도 같은 이유로 이 단어를 피했다. `verify-binary-assets`처럼 묶음 타깃 `make verify`에는 넣지 않았다.

### 3.5 리뷰 지적 사항

| 심각도 | 출처 | 지적 | 수정 |
|---|---|---|---|
| LOW | 리뷰 | 실패 메시지와 CI 주석이 COM0-9, LPT0-9만 적고 규칙이 거부하는 위첨자 이름은 빠뜨림 | `975047a2` |
| LOW | 리뷰 | `git ls-files` 출력을 로캘 인코딩으로 디코딩 | `975047a2`: `encoding="utf-8"` |
| LOW | 보안 | stem에 뒤쪽 공백이 남아 `aux .txt`가 통과 | `d3eae581`: `rstrip(" ")`과 테스트 케이스 |
| (포맷) | finalizer | `aux`가 `auxiliary`로 바뀌면서 rustfmt가 임포트 순서를 바꿈 | `23ff2ac4` |

## 4. 검증

| 확인 항목 | 결과 |
|---|---|
| `cargo check --all-targets` (루트 크레이트) | 통과 |
| `cargo check -p mlxcel-xla --all-targets`, `cargo fmt -p mlxcel-xla -- --check` | 통과 |
| `cargo test -p mlxcel-xla` | 363 통과, 0 실패, 19 무시 |
| 이 브랜치에서 `check_windows_reserved_paths.py` | OK |
| 같은 검사기를 `origin/main` 트리에 적용 | 실패, `src/lib/mlxcel-xla/src/aux.rs` 지목 |
| `check_windows_reserved_paths_test.sh` | 케이스 16개와 실제 트리 전부 통과 |
| `git grep -nE '\baux::\|\baux\.rs' -- ':!TECHNICAL_REPORTS'` | 일치 없음 |

## 5. 미검증 항목과 남은 일

- **로컬 빌드 못 함:** `vlm_gemma_xla_tests.rs`의 `diagnostics_thread_overrides_stay_out_of_production_iree_paths`는 `xla-iree` 기능에 묶여 있고 `IREE_DIST`가 필요한데, 이 호스트에는 없다. `include_str!` 대상 경로는 직접 확인했다. CI의 `xla-compile` 작업이 `cargo check --features cuda,xla-iree --all-targets`로 이 테스트를 컴파일한다.
- **Windows 실기 확인 안 함:** 수락 기준의 "Windows에서 `git clone`이 끝나고 `git status`가 깨끗하다"는 Windows 기계가 있어야 확인할 수 있다.
- **범위 밖:** Windows가 만들 수 없는 다른 이름은 검사기를 통과한다. `<>:"|?*` 문자, 백슬래시, 제어 문자, 점이나 공백으로 끝나는 구성 요소, 대소문자만 다른 경로가 여기에 해당한다. 지금 트리에는 하나도 없다. 생기면 이 검사기에 규칙을 더하면 된다.

## 6. 학습 포인트

- CI 러너가 없는 플랫폼의 제약은 그 플랫폼 사용자가 저장소를 써 보기 전까지 드러나지 않는다. `git ls-files`에 대한 정적 검사는 몇 초면 끝나고, 러너 없이도 그 플랫폼을 지켜 준다.
- Windows 장치 이름은 첫 점 앞의 stem을 기준으로, 대소문자 구분 없이, 뒤쪽 공백을 잘라낸 뒤, 경로의 모든 구성 요소에서 판정된다. 파일 이름의 마지막 확장자만 보는 검사로는 대부분 놓친다.
- `make help`가 부분 문자열로 타깃을 분류한다면 새 타깃 이름을 각 섹션의 필터와 대조해 봐야 한다. "reserved" 같은 단어는 엉뚱한 섹션에 들어간다.
