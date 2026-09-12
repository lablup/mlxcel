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
- 초기 stock Docker Linux 실행은 자체 baseline으로 19개 테스트를 통과했지만 hosted renderer 호환성을 입증하지 못했습니다. 해당 image에는 DejaVu가 없어 Latin 문자도 WenQuanYi로 렌더링됩니다. hosted font package를 고정한 뒤 native arm64 재현에서 리뷰된 hosted image 12개에 대해 20개 테스트를 통과했습니다(기본 19개와 opt-in Chromium font diagnostic). image의 Node는 canonical v26.5.1이 아닌 v24.20.0이므로 build-toolchain 동등성이 아니라 renderer 호환성 증거입니다.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` 통과: WebUI contract fixture 41개와 DTO drift/schema strictness 검사.
- `make verify-webui-bundle` 통과: deterministic checked-in asset 검증 및 bundle digest `abb41bf0eb305acbc21291249afd7ac558b2d341382686dd705a1d1f5cf3fde0`.
- 승인된 4185 source digest: `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`.

## 수용 검증 경계

provider/mock HTTP 테스트는 submit 전 초기 요청 없음, Bearer가 붙은 bootstrap/catalog/operations/events 호출, 모든 route에서 접근 가능한 logout, 401 세션 purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized error, stale auth-failure fencing, token의 DOM/storage/URL 미반사, authenticated route 탐색 중 autoload 및 inference endpoint 미호출을 검증합니다. browser screenshot은 명시적으로 mock API 기반 product shell 증거이며 실제 backend session 증명은 아닙니다. 유지보수자는 2026-09-13에 source digest `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`의 [수정된 4185 디자인을 승인](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5648012196)했으므로 디자인 승인은 완료되었습니다. 실제 macOS 27 Safari, VoiceOver, native browser 200% zoom에서 변경된 layout과 focus 동작의 표적 재확인은 아직 대기 중이며, 디자인 승인이 해당 검사 통과를 의미하지는 않습니다. 이전 수동 Safari/VoiceOver/native-zoom 피드백은 더 오래된 cb489/4184 preview에만 적용됩니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않고, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.

이번 문서 전용 마감 검증은 `50caf9a9`에서 typecheck·lint·unit 62개와 공유 contract/compatibility/version/kernel-key 검사를 다시 통과했고, 소스나 asset을 변경하지 않고 위 두 digest를 각각 다시 계산했습니다. Darwin 및 Docker Linux 브라우저 결과는 앞선 구현 검증의 증거이며, 이번 문서 수정에서는 브라우저 suite를 동시에 실행하지 않았습니다.

## Linux screenshot 교정

Hosted Chromium CDP로 Latin 문자에 DejaVu Sans regular/bold, 한국어 fallback에 WenQuanYi Zen Hei가 쓰임을 확인했습니다. stock amd64 container에서도 불일치가 발생하므로 CPU architecture만으로 설명할 수 없습니다. WebUI CI job은 이제 Ubuntu 24.04와 `fonts-dejavu-core=2.37-8`, `fonts-dejavu-extra=2.37-8`, `fonts-wqy-zenhei=0.9.45-8`을 고정하며 canonical Node·pnpm 버전은 유지합니다. hosted actual image 12개를 각각 시각적으로 리뷰한 뒤 채택했습니다. [Screenshot provenance](../webui/tests/screenshots/README.md)에 원본 run과 hash를 기록했습니다. production font, 승인된 source/asset, screenshot threshold, geometry, rendering flag는 바꾸지 않았습니다.

고정 font를 사용한 native arm64 재현은 Chromium CDP의 hosted font와 일치했고 20개 테스트를 모두 통과했습니다. QEMU 기반 로컬 amd64 시도는 렌더링 전에 Chromium GPU process가 충돌했으므로 통과가 아닙니다. 이전 stock-image 결과로 이 교정된 renderer 증거를 대체해서는 안 됩니다.

`b3c5326491230cddf77d68ce3d90815ff237f7b9`의 [canonical hosted WebUI job](https://github.com/lablup/mlxcel/actions/runs/34723713520/job/103634069494)이 Ubuntu 24.04 amd64, Node 26.5.1, pnpm 11.18.0 및 고정 font로 통과했습니다. Typecheck, lint, unit 62개, browser 20개 전체, deterministic bundle verification 단계가 실제 실행되어 성공했습니다. 이 결과가 불충분했던 stock-container 증거를 대신하는 hosted 검증이며, 사용 불가 GB10 job이나 대기 중인 수동 검사 통과를 주장하지는 않습니다.
