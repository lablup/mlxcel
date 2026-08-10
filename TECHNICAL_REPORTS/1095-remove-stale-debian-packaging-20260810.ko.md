# 기술 보고서: PR #1095 - 오래된 Debian 패키징 제거

**작성일**: 2026-08-10
**상태**: 완료
**언어**: Debian packaging, shell, Markdown
**위험도**: Low

## 요약

PR #1095는 사용되지 않던 mlxcel의 Debian 및 Launchpad PPA 패키징 경로를 제거한다. 이 저장소는 실제 업로드 이력이 없고 release CI에도 연결되지 않았으며 현재 Ubuntu target에서 필요한 Rust toolchain을 해석할 수 없는 패키지의 changelog와 build metadata를 release마다 계속 관리하고 있었다.

## 1. 문제 정의

추적 중이던 `debian/` tree는 실제로 존재하지 않는 자동 PPA release 경로를 설명했다. Target PPA에는 mlxcel package가 한 번도 게시되지 않았지만 changelog는 release 때마다 갱신되었고, active build dependency는 unversioned `rustc >= 1.85`를 요구했다. 그러나 현재 Ubuntu archive는 더 오래된 unversioned compiler를 제공하거나 새 compiler를 version-suffixed package로만 제공한다. 또한 network가 차단된 Launchpad builder에서는 문서화된 rustup fallback도 사용할 수 없다.

따라서 이 tree는 사용 가능한 산출물 없이 반복적인 release 유지 비용과 잘못된 distribution 설명만 만들었다.

## 2. 기술적 선택과 그 이유

### 2.1 패키징 부활 대신 제거 선택

Issue #1068은 제거와 부활을 상호 배타적인 경로로 제시했다. Linux release binary는 이미 생성되고 있지만, 부활에는 별도의 distribution 결정, 실제 MSRV 확정, offline vendoring 증명, release upload workflow, 성공한 PPA publish가 모두 필요했다. 이 전제들이 존재하지 않으므로 제거 경로를 선택했다.

### 2.2 전체 패키징 경로를 원자적으로 삭제

Control file, rules variant, helper script, packaging 문서, 1,206-line generated Debian changelog를 함께 제거했다. 남아 있던 `CHANGELOG.md`의 repository path 표현은 일반적인 release note 설명으로 바꾸어 유지되는 문서가 삭제된 산출물을 가리키지 않게 했다.

## 3. 변경 요약

| 항목 | 값 |
|------|----|
| 변경 파일 | 18 |
| 추가 라인 | 1 |
| 삭제 라인 | 1,765 |
| 제거한 packaging path | 17 |

- 추적 중이던 `debian/` directory 전체를 제거했다.
- Changelog generator와 PPA version/query helper를 제거해 오래된 release 유지 경로를 없앴다.
- Launchpad 및 GitHub Actions 연동을 주장하던 오래된 문서를 제거했다.
- 기존 binary release workflow 동작은 변경하지 않았다.

## 4. 리뷰 발견 사항

Implementation, security/performance, finalization review에서 Critical, High, Medium 또는 조치 가능한 Low 이슈는 발견되지 않았다. 저장소 밖에 설치된 release helper에는 file-existence guard가 있는 `debian/changelog` branch가 남아 있지만, 이번 삭제 이후에는 해당 branch가 비활성화되어 제거된 경로를 다시 생성할 수 없다.

## 5. 검증

- `test ! -d debian`: 통과.
- `git ls-tree -r --name-only HEAD | rg '^debian/'`: 추적 중인 packaging path 없음.
- `debian/changelog`, `README.packaging`, Launchpad, PPA, `dput` repository scan: 현재 참조 없음.
- Workflow, documentation, script 및 local repository scaffolding scan에서 남은 release step이나 PPA 설명이 없음을 확인했다.
- `git diff --check origin/main...HEAD`: 통과.
- Hosted `Detect changes`, crate-version, kernel-dtype-key, cross-repository-reference 및 CLA check 통과. Rust 또는 dependency 경로를 변경하지 않았으므로 Rust-heavy job은 change detection에 의해 skip되었다.

## 6. 관련 작업

- Issue #1068: Debian packaging 제거 또는 부활 결정과 acceptance criteria.
- Issue #1066: packaging metadata 불일치를 더 명확히 드러낸 Rust toolchain pin 갱신.
