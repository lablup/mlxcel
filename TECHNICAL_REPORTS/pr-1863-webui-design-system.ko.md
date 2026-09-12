# PR #1863 WebUI 디자인 시스템 및 provider 통합 보고서

**작성일**: 2026-09-13
**상태**: provider 기반 셸과 승인된 4185 macOS 27 스타일 후보 구현 완료; Safari/VoiceOver/native zoom 표적 재확인과 사용 불가 GB10 CI는 로컬 검증 범위 밖
**위험도**: 중간

## 요약

PR #1863은 이제 정적 placeholder나 두 번째 인증 cache 대신 공유 #1842 provider에 WebUI 디자인 셸을 연결합니다. production route는 로그인, 로그아웃, 연결 상태, 선택된 catalog identity, lifecycle label, catalog/operation freshness, 안전한 schema mismatch recovery를 provider snapshot과 action으로 처리합니다. 직접 접근용 `#gallery` artifact route는 deterministic visual baseline을 위해 격리되어 있고, 명시적 로그인 전에는 로컬 API에 접촉하지 않습니다.

승인된 4185 후보는 이전의 떠 있는 macOS 26식 처리를 source-backed macOS 27 방향으로 교체했습니다. 핵심은 flush full-height sidebar, 연속된 58 px sidebar/header edge, hard scroll boundary를 가진 sticky main toolbar, 중립 content surface, 절제된 control group, concentric radius, 44 px toolbar hit target을 유지하는 compact two-row reflow입니다. Apple artwork, fake traffic light, SF Symbol asset은 번들하지 않았습니다.

## 변경 요약

구현은 앱을 `WebUiProvider`로 감싸고, LoginView 제출은 `actions.login`, 로그아웃은 `actions.logout`으로 보냅니다. 세션 키는 provider/client 메모리에만 머물며, 인증 실패는 localized presentation code로 축약해 raw token이나 서버 메시지를 DOM에 반사하지 않습니다. Models, Chat, Activity는 여전히 정직한 단계적 route입니다. signed-out 상태에서는 provider 기반 로그인 표면을 보여주고, authenticated 상태에서는 모델을 로드하거나 추론을 시작하지 않은 채 backend mode, build version, provider state, catalog count, operation count, snapshot sequence만 보고합니다. toolbar의 selected-model pill은 선택된 catalog entry와 lifecycle state가 실제로 있을 때만 그 값에서 파생하고, 없으면 “선택한 모델 없음”으로 유지합니다.

시각 수정은 기존 outer app gutter, floating sidebar tile, hero-card page frame, decorative page gradient를 제거했습니다. compact wrapping 동작은 더 이상 `data-test-text-scale` layout selector에 의존하지 않고 production CSS에 직접 존재하며, browser test suite가 document/panel scroll width, visible compact focus, compact toolbar hit-target geometry를 검증합니다.

## 검증 상태

4185 후보의 최종 로컬 검증은 다음과 같습니다.

- `pnpm --dir webui run typecheck` 통과.
- `pnpm --dir webui run lint` 통과.
- `pnpm --dir webui run unit` 통과: Vitest 8파일, 62개 테스트.
- `pnpm --dir webui run browser` Darwin 통과: strict screenshot 12개, product mock-API journey, axe 검사, overflow 검사, compact hit-target assertion, token-containment 검사를 포함한 Playwright 19개 테스트.
- `mcr.microsoft.com/playwright:v1.63.0-noble` Docker Linux Playwright 통과: 별도 Linux screenshot baseline으로 19개 테스트 통과. 해당 image는 Node v24.20.0을 보고해 project engine v26.5.1과 다르지만, 그 Linux browser 환경에서 Vite와 Playwright가 완료되고 snapshot이 생성되었습니다.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` 통과: WebUI contract fixture 41개와 DTO drift/schema strictness 검사.
- `make verify-webui-bundle` 통과: deterministic checked-in asset 검증 및 bundle digest `abb41bf0eb305acbc21291249afd7ac558b2d341382686dd705a1d1f5cf3fde0`.
- 승인된 4185 source digest: `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`.

## 수용 검증 경계

provider/mock HTTP 테스트는 submit 전 초기 요청 없음, Bearer가 붙은 bootstrap/catalog/operations/events 호출, 모든 route에서 접근 가능한 logout, 401 세션 purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized error, stale auth-failure fencing, token의 DOM/storage/URL 미반사, authenticated route 탐색 중 autoload 및 inference endpoint 미호출을 검증합니다. browser screenshot은 명시적으로 mock API 기반 product shell 증거이며 실제 backend session 증명은 아닙니다. 실제 Safari on macOS 27, VoiceOver, native browser 200% zoom, 이 새 4185 후보에 대한 최종 사용자 스타일 승인은 현재 환경에서 실행하지 않았으며 수동 후속 gate로 남습니다. 이전 수동 Safari/VoiceOver/native-zoom 피드백은 더 오래된 cb489/4184 preview에만 적용됩니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않고, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.
