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

provider/mock HTTP 테스트는 submit 전 초기 요청 없음, Bearer가 붙은 bootstrap/catalog/operations/events 호출, 모든 route에서 접근 가능한 logout, 401 세션 purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized error, stale auth-failure fencing, token의 DOM/storage/URL 미반사, authenticated route 탐색 중 autoload 및 inference endpoint 미호출을 검증합니다. browser screenshot은 명시적으로 mock API 기반 product shell 증거이며 실제 backend session 증명은 아닙니다. 유지보수자는 2026-09-13에 source digest `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`의 [수정된 4185 디자인을 승인](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5648012196)했으므로 디자인 승인은 완료되었습니다. 유지보수자는 이후 이 디자인을 다시 확정했고, 별도 질문에 직접 답하여 Safari/VoiceOver의 툴바·compact 메뉴, Cmd+K Tab/Escape 포커스 및 native browser 200% zoom 표적 재검사를 수용했습니다. [수용 기록](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5653425099)은 immutable 4185에 대한 사용자 보고 PASS이며, 독립적인 브라우저 관찰이나 이후 레이아웃 변경의 수용은 아닙니다. 이 결과는 이전 cb489/4184 수동 보고를 보완합니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않고, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.

이번 문서 전용 마감 검증은 `50caf9a9`에서 typecheck·lint·unit 62개와 공유 contract/compatibility/version/kernel-key 검사를 다시 통과했고, 소스나 asset을 변경하지 않고 위 두 digest를 각각 다시 계산했습니다. Darwin 및 Docker Linux 브라우저 결과는 앞선 구현 검증의 증거이며, 이번 문서 수정에서는 브라우저 suite를 동시에 실행하지 않았습니다.

## Linux screenshot 교정

Hosted Chromium CDP로 Latin 문자에 DejaVu Sans regular/bold, 한국어 fallback에 WenQuanYi Zen Hei가 쓰임을 확인했습니다. stock amd64 container에서도 불일치가 발생하므로 CPU architecture만으로 설명할 수 없습니다. WebUI CI job은 이제 Ubuntu 24.04와 `fonts-dejavu-core=2.37-8`, `fonts-dejavu-extra=2.37-8`, `fonts-wqy-zenhei=0.9.45-8`을 고정하며 canonical Node·pnpm 버전은 유지합니다. hosted actual image 12개를 각각 시각적으로 리뷰한 뒤 채택했습니다. [Screenshot provenance](../webui/tests/screenshots/README.md)에 원본 run과 hash를 기록했습니다. production font, 승인된 source/asset, screenshot threshold, geometry, rendering flag는 바꾸지 않았습니다.

고정 font를 사용한 native arm64 재현은 Chromium CDP의 hosted font와 일치했고 20개 테스트를 모두 통과했습니다. QEMU 기반 로컬 amd64 시도는 렌더링 전에 Chromium GPU process가 충돌했으므로 통과가 아닙니다. 이전 stock-image 결과로 이 교정된 renderer 증거를 대체해서는 안 됩니다.

`b3c5326491230cddf77d68ce3d90815ff237f7b9`의 [canonical hosted WebUI job](https://github.com/lablup/mlxcel/actions/runs/34723713520/job/103634069494)이 Ubuntu 24.04 amd64, Node 26.5.1, pnpm 11.18.0 및 고정 font로 통과했습니다. Typecheck, lint, unit 62개, browser 20개 전체, deterministic bundle verification 단계가 실제 실행되어 성공했습니다. 이 결과가 불충분했던 stock-container 증거를 대신하는 hosted 검증이며, 사용 불가 GB10 job 통과나 사용자 보고 수동 결과의 독립적 입증을 주장하지는 않습니다.

## 수동 수용 문서 마감

표적 수동 게이트는 사용자 보고 PASS로 수용했지만, #1838 → #1841 → #1843 순서의 중앙 통합은 아직 대기 중이므로 이슈와 PR은 review 상태를 유지합니다. 이번 문서 전용 변경은 소스·스타일·asset·테스트를 수정하지 않으며 로컬 GPU·브라우저·런타임 테스트를 실행하지 않습니다. 호스트 GPU firmware 장애 복구는 확인되지 않았으며, 사용 불가 GB10 면제는 로컬 호스트 복구나 다른 통합 게이트의 면제가 아닙니다.

## ui-common adoption checkpoint

새 사용자 요청에 따라 @lablup/ui-common alpha.19를 정확히 고정하고 공통 adapter/token bridge에서 component subpath를 사용합니다. [공유 export/예외 표](../docs/webui/ui-common.md)를 따릅니다. CPU typecheck·lint·unit 72개와 현재 worktree의 contract fixture 41개가 통과했으며 공개 NOTICE/LICENSE를 번들에 포함합니다. 이전4185 소스와 사용자 보고 수동 승인은 과거 버전에 한정되며, 변경된 DOM에는 실제 served-CSP·시각·관련 수동 검증이 새로 필요합니다. 루트의 호스트 예약 중 로컬 브라우저/GPU 테스트는 실행하지 않았습니다. 이후 중앙 rebase에서 최신 backend schema/fixture 변경을 보존해야 합니다.

## Rebase 후 migration 검증

`62c4f259`는 병합된 startup/security backend를 통합하면서 Unicode validator, schema 및 contract fixture 44개를 그대로 보존합니다. Frozen dependency install, typecheck, lint, unit 76개와 deterministic bundle 검증이 통과했습니다. Source digest는 `cfbcaada7d5e9734b2d305eacf534e003d8b8ea3916ce163c5a53f294c6bbd78`, bundle digest는 `858f5775f548532b3fa93942c7f063e06b83f427d107b47c82678f3564638f7f`입니다. 독립 correctness/security 재리뷰에서 migration seam 관련 findings는 없습니다.

고정된 Playwright heading-level helper는 명시적 `aria-level`보다 native `h3`를 우선하지만, Chromium AX 직접 관찰에서는 common Tabs 반복 remount 후에도 level 2, 올바른 이름과 non-ignored 상태가 확인됩니다. Browser 및 실제 서버 CSP 테스트가 이 AX tree를 검증하고 StrictMode unit도 remount를 검사합니다. 이는 Chromium 증거이며 native Safari 증거가 아닙니다.

Hosted run 34801765896은 browser 19개를 통과하고 기존 States screenshot만 실패했습니다. 측정 progress와 미상 progress를 기존 cell 하나로 묶고 lifecycle label casing을 보존했으며, 개별 시각 리뷰를 거친 Darwin/Linux States baseline만 출처와 함께 갱신합니다. 다른 baseline과 엄격한 threshold는 바꾸지 않았습니다. 이 문서 시점에서 hosted 전체 재실행, 루트의 실제 secured-CSP 실행 및 변경된 DOM에 대한 사용자 Safari/VoiceOver 표적 재확인은 각각 대기 중입니다. 로컬 브라우저 진단 프로세스는 루트 runtime build 전에 모두 종료했으며 이 유닛은 MLX/GPU 테스트를 실행하지 않았습니다.

### Migration 게이트 결과

[`98fe9df2`의 canonical hosted WebUI job](https://github.com/lablup/mlxcel/actions/runs/34802205815/job/103846995858)은 typecheck, lint, unit 76개, browser 20개 전체와 deterministic bundle 검증을 통과했습니다. Bundle digest는 `858f5775f548532b3fa93942c7f063e06b83f427d107b47c82678f3564638f7f`입니다. `62c4f259`에서 빌드한 실제 model-free production server도 별도로 supplied-server CSP 두 경우(1440 light, 390 dark)를 통과했습니다. 실제 응답 정책, 정책 위반 및 외부 요청 없음, Select 위치와 Escape 포커스, 측정 progress, reduced motion, axe와 browser AX heading semantics를 검사했으며 static hosted 테스트만으로 이 결과를 주장하지 않습니다. 두 head의 production source/asset은 동일합니다.

변경된 DOM의 Safari/VoiceOver/native zoom 재확인은 새 immutable production preview를 대상으로 한 번 요청했으며 아직 대기 중입니다. 과거4185 승인을 재사용하지 않습니다. Download backend 병합 이후 최종 통합 검증도 남아 있으므로 review 상태를 유지하며, 사용 불가 GB10 결과를 통과로 보고하지 않습니다.

### 사용자 결정: 수동 검증을 최종 통합 수용으로 이연

사용자는 현재 원격 환경에서 Safari/VoiceOver 검증을 수행할 수 없어, 변경된 ui-common DOM의 Safari/VoiceOver/native zoom 검사를 전체 구현 완료 후 최종 통합 수용 단계에서 함께 진행하도록 명시적으로 결정했습니다. 상태는 PASS가 아니라 DEFERRED이며 개별 유닛의 머지 차단 조건은 아닙니다. 공통 컨트롤과 페이지별 native 검사를 함께 수행하고, 이전4185 승인을 변경된 DOM에 재사용하지 않습니다. Download backend 병합 이후 최종 통합 검증은 여전히 필요하므로 review 상태를 유지합니다. 이번 변경은 문서 전용이며 immutable4186 preview, production source 및 bundle을 변경하지 않습니다.
