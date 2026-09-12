# PR #1863 WebUI 디자인 시스템 및 provider 통합 보고서

**작성일**: 2026-09-13
**상태**: provider 기반 셸 구현 완료; Safari/VoiceOver/스타일 수동 승인과 사용 불가 GB10 CI는 로컬 검증 범위 밖
**위험도**: 중간

## 요약

PR #1863은 이제 정적 placeholder나 두 번째 인증 cache 대신 공유 #1842 provider에 WebUI 디자인 셸을 연결합니다. production route는 로그인, 로그아웃, 연결 상태, 선택된 catalog identity, lifecycle label, catalog/operation freshness, 안전한 schema mismatch recovery를 provider snapshot과 action으로 처리합니다. 직접 접근용 `#gallery` artifact route는 deterministic visual baseline을 위해 격리되어 있고, 명시적 로그인 전에는 로컬 API에 접촉하지 않습니다.

## 변경 요약

구현은 앱을 `WebUiProvider`로 감싸고, LoginView 제출은 `actions.login`, 로그아웃은 `actions.logout`으로 보냅니다. 세션 키는 provider/client 메모리에만 머물며, 인증 실패는 localized presentation code로 축약해 raw token이나 서버 메시지를 DOM에 반사하지 않습니다. Models, Chat, Activity는 여전히 정직한 단계적 route입니다. signed-out 상태에서는 provider 기반 로그인 표면을 보여주고, authenticated 상태에서는 모델을 로드하거나 추론을 시작하지 않은 채 backend mode, build version, provider state, catalog count, operation count, snapshot sequence만 보고합니다. toolbar의 selected-model pill은 선택된 catalog entry와 lifecycle state가 실제로 있을 때만 그 값에서 파생하고, 없으면 “선택한 모델 없음”으로 유지합니다.

## 검증 상태

이번 단계의 최종 로컬 검증은 다음과 같습니다.

- `pnpm --dir webui run typecheck` 통과.
- `pnpm --dir webui run lint` 통과.
- `pnpm --dir webui run unit` 통과: Vitest 8파일, 60개 테스트.
- `pnpm --dir webui run browser` 통과: strict Darwin gallery baseline 및 behavior/a11y 검사를 포함한 Playwright 12개 테스트.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` 통과: WebUI contract fixture 41개와 DTO drift/schema strictness 검사.
- `make verify-webui-bundle` 통과: deterministic checked-in asset 검증 및 bundle digest `cd7c54baf8b191aee79e80beb5ff4711ff91596adbb80fa52ea76edccf74f7dc`.

## 수용 검증 경계

새 provider/mock HTTP 테스트는 submit 전 초기 요청 없음, Bearer가 붙은 bootstrap/catalog/operations/events 호출, 401 세션 purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized error, token의 DOM/storage/URL 미반사, authenticated route 탐색 중 autoload 및 inference endpoint 미호출을 검증합니다. 실제 Safari on macOS 27, VoiceOver, native browser 200% zoom, 최종 사용자 스타일 승인은 현재 환경에서 실행하지 않았으며 수동 후속 gate로 남습니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않고, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.
