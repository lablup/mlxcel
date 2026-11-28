# 기술 보고서: PR #1397 - Cross-Repository Reference 경계 동적 산출

**작성일**: 2026-08-24

**작성자**: 신정규

**상태**: 완료

**언어**: Python, Bash, YAML, Markdown

**위험도**: Medium

## 요약

PR #1397은 1000 이상의 모든 bare issue 또는 pull-request 번호를 upstream reference로 취급하던 만료된 규칙을 제거한다. Advisory checker는 안전하게 토큰을 사용할 수 있을 때 GitHub에서 현재 `lablup/mlxcel` 번호 경계를 산출하고, offline 또는 unauthenticated 환경에서는 명시적인 수동검토 방식으로 fallback한다. 두 모드를 검증하는 결정적 companion test도 실제 pull-request workflow에서 실행한다.

## 1. 문제 정의

저장소 번호가 classifier의 고정 가정을 넘어섰다. #1023, #1340, #1355, #1385는 유효한 same-repository link지만 기존 `num >= 1000` 분기는 이를 likely upstream으로 표시했다. 측정된 PR #1386 diff에서는 7개, 과거 PR #1385 merge diff에서는 숫자 규칙만으로 28개의 false positive가 발생했다. 검사가 advisory이므로 지속적인 잡음은 실제 unqualified upstream 또는 private-repository reference를 찾기 위한 리뷰 신호를 약화시켰다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| Classifier | 5초 제한 `gh api` 호출로 최신 issue/PR 번호를 조회하고 고정 임계값 제거 |
| Fallback | Non-upstream bare ref를 수동검토 버킷에 계속 표시하고 live classification을 사용하지 못한 이유 출력 |
| Companion test | 임시 저장소에서 same-repo, upstream, qualified, boundary 초과, offline, API 실패, strict mode 검증 |
| CI | Classifier 전에 companion suite를 실행하고 same-repository PR에만 `github.token` 제공 |
| 기여 문서 | Live boundary, offline, fork, manual-review 동작 문서화 |

## 3. 기술적 선택과 그 이유

### 3.1 경계를 재조정하지 않고 동적으로 산출

Issue와 pull-request 번호는 GitHub 저장소 순서를 공유하고 issues endpoint에는 pull request도 포함된다. 생성일 내림차순을 명시해 최신 항목을 조회하면 저장소 성장에 따라 다시 만료될 숫자를 관리하지 않고 하나의 이동 경계를 얻을 수 있다.

### 3.2 실패는 허용하되 조용히 신호를 버리지 않음

토큰 없음, `gh` 실행 파일 없음, timeout, API 실패, 잘못된 응답은 advisory check 자체를 실패시키지 않는다. 대신 classifier가 fallback 이유를 출력하고 모든 non-upstream bare reference를 기존 수동검토 버킷에 넣는다. 명시적인 upstream-name signal은 두 모드 모두 likely-upstream 버킷으로 간다.

### 3.3 Fork code에서 base-repository credential 분리

Workflow는 pull-request가 제어하는 script를 실행한다. 따라서 보안 리뷰에서 `github.token`을 same-repository PR에만 제공하도록 제한했다. Fork PR은 빈 값을 받아 의도적으로 문서화된 offline fallback을 실행한다. 이 방식은 변경된 코드에 base repository token을 노출하지 않으면서 public fork coverage를 유지한다.

### 3.4 격리된 저장소에서 실제 동작 검증

Shell companion은 임시 Git 저장소와 fake `gh` binary를 만들어 network 없이 boundary 및 실패 동작을 결정적으로 검증한다. 의도적인 reference corpus는 기존 `IGNORE_PREFIXES` 경로로 제외하므로 production classifier가 자체 fixture를 보고하지 않는다.

## 4. 검증

- `python3 -m py_compile scripts/ci/check_cross_repo_refs.py`: 통과.
- `bash -n scripts/ci/check_cross_repo_refs_test.sh`: 통과.
- `bash scripts/ci/check_cross_repo_refs_test.sh`: strict mode의 예상 non-zero 결과를 포함한 7개 case 모두 통과.
- 부모 환경에 `GITHUB_TOKEN`이 있는 상태에서도 companion suite가 통과해 no-token case가 상속 credential을 명시적으로 제거함을 확인.
- 합성 PR #1386 replay: #1340과 #1355가 `Likely UPSTREAM` 밖에 유지됨.
- 과거 `fb1e909cc~1...fb1e909cc` replay: 숫자 규칙만으로 발생한 same-repository false positive 28개가 사라지고, 이슈가 허용한 경계와 같이 명시적 upstream 문맥이 있는 세 줄만 upstream signal로 남음.
- Live endpoint 확인에서 현재 PR 번호 1397을 저장소 경계로 반환.
- Hosted `cross-repo refs`는 상속 토큰 fixture 수정 후 companion suite와 production classifier를 모두 성공적으로 실행.
- Formatting, Clippy, cargo-deny, MLX-pin extraction, OpenXLA compile, 저장소 metadata 검사, CLA 통과. 최종 OpenXLA link job은 보고서 작성 시점에 pending이며 merge 전에 terminal 상태에 도달해야 함.
- 정확성, 보안/성능, finalization 리뷰는 fork-token 제한 후 미해결 문제를 발견하지 못함.

## 5. Hosted 검증에서 발견한 통합 문제

첫 hosted run에서는 companion suite의 offline case가 workflow-level `GITHUB_TOKEN`을 상속해 fallback 경로에 진입하지 못했다. 토큰이 없는 로컬 shell에서는 통과했기 때문에 실제 환경에서만 드러난 차이였다. Fixture는 이제 해당 case 내부에서 `GH_TOKEN`과 `GITHUB_TOKEN`을 모두 제거하고, hosted GNU `env` 호환성을 위해 option을 assignment보다 앞에 둔다. 교체된 hosted job은 통과했다.

## 6. 관련 작업

- Issue #1387: same-repository reference의 만료된 숫자 분류.
- PR #1397: 이 보고서가 다루는 구현 및 리뷰 교정.
- PR #1386 및 merge commit `fb1e909cc`: 측정에 사용한 false-positive dataset.
- `scripts/ci/check_crate_versions.py`: classification 정책을 명시적이고 reviewable하게 만드는 관련 선례.
