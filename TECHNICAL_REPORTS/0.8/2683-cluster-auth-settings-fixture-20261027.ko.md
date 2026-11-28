# 기술 보고서: PR #2683 - 클러스터 인증 설정 fixture

**작성일**: 2026-10-27  
**상태**: 완료  
**언어**: TypeScript, JSON  
**위험도**: 낮음

## 요약

PR #2683은 에픽 #2574 통합 과정에서 추가된 `cluster_auth` 필드의 읽기 전용 settings 계약을 완성합니다. WebUI는 인증 설정 여부라는 boolean만 영어와 한국어로 표시하며, 공유 fixture는 향후 schema drift를 차단하고 임베디드 번들에 갱신된 카탈로그를 반영합니다.

## 1. 문제 정의

통합 전체 워크스페이스 테스트에서 서버는 `cluster_auth.configured`를 보내지만 `tests/fixtures/settings/read-only-fields.json`은 이전 필드 목록을 유지하고 있음을 확인했습니다. 카탈로그 항목도 빠져 있어 WebUI에서 일반 다국어 시작 설정이 아니라 raw name fallback으로 표시될 수 있었습니다.

## 2. 기술적 선택과 근거

상위 라벨은 `Cluster authentication` / `클러스터 인증`을 사용합니다. 유일한 하위 필드는 기존 API key 인증 어휘와 동일한 `Configured` / `설정됨`을 사용합니다. 값은 공통 boolean 다국어 표시(`Yes`/`No`, `예`/`아니요`)를 그대로 이용합니다. Fixture와 UI 카탈로그 어느 곳에도 secret, credential 값, 인증 결과를 추가하지 않았습니다.

생성된 WebUI asset은 저장소에 고정된 Node.js 26.5.1과 pnpm 11.18.0으로 다시 빌드했고 결정적 출력을 검증했습니다.

## 3. 변경 요약

| 영역 | 변경 |
|---|---|
| 계약 fixture | `cluster_auth`와 유일한 하위 필드 `configured` 추가 |
| 다국어 | 영어·한국어 시작 설정 라벨과 catalog fixture 항목 추가 |
| 표시 회귀 검사 | 구조화된 상태가 다국어 boolean 텍스트와 catalog 라벨을 사용하는지 검증 |
| 배포 번들 | 임베디드 WebUI asset과 manifest 재생성 |

## 부록: 검증

- 읽기 전용 필드 fixture 계약 테스트: 1개 통과.
- WebUI focused unit test: startup label, startup value, catalog drift 총 28개 통과.
- WebUI typecheck, lint: 통과.
- 임베디드 번들 결정성 검증: 통과.
- `cargo fmt --check`, `git diff --check`: 통과.
- 같은 통합 스위트의 두 model-family 실패는 이 PR 범위 밖이며 별도로 처리됩니다.

관련 항목: #2574.
