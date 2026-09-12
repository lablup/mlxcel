# PR #1863 WebUI 디자인 시스템 보고서

**작성일**: 2026-09-12
**상태**: 부분 완료 — 컴포넌트 준비, provider 통합 및 수동 수용 검증 대기
**위험도**: 중간

## 요약

PR #1863은 에픽 #1834의 공유 WebUI 디자인 시스템 계층을 구현합니다. semantic CSS 토큰, 절제된 macOS 27 방향의 셸, typed 영어/한국어 문자열, 재사용 가능한 primitive, appearance 저장, `#gallery` 컴포넌트 갤러리를 제공합니다. 글래스는 장식적 chrome에만 쓰고 콘텐츠는 중립 표면에 올리며, Apple 에셋 대신 프로젝트 작성 SVG 아이콘을 사용하고 각 레이아웃·재질 결정에 사용한 공식 Apple reference를 문서화했습니다.

## 문제 정의

후속 페이지는 개별 스타일이나 중복 인증 상태 대신 하나의 공유 재질·간격·지역화·키보드 계약을 사용해야 합니다. 공유 client가 준비되기 전에 컴포넌트 갤러리로 primitive를 검증할 수 있지만, 이는 실제 인증이나 네이티브 브라우저 접근성 수용 검증을 대신하지 않습니다.

## 변경 요약 및 리뷰 보강

수정 사이클에서 production 화면의 과장된 placeholder를 하나의 중립 connection prompt로 교체했고, Gallery를 primary navigation에서 숨기되 direct artifact route로 유지했으며, production route의 빈 inspector chrome을 제거하고 desktop grid를 정리했습니다. 또한 controlled LoginView 및 SchemaMismatchView contract를 추가했고, 브라우저 테스트를 단순 시각 smoke에서 실제 동작 검증으로 확장했습니다. production 화면은 공유 #1842 provider가 머지된 뒤 통합될 때까지 사실을 과장하지 않는 placeholder로 남으며, 이 PR은 #1842 provider state를 복사하거나 별도 cache하지 않습니다.

## 검증 상태

최종 로컬 검증 범위는 `pnpm --dir webui run typecheck`, `pnpm --dir webui run lint`, `pnpm --dir webui run unit`의 Vitest 10개 통과, `pnpm --dir webui run browser`의 Playwright 12개 통과, deterministic bundle verification digest `44b731231da59d5454c3d2956bc50fe22c6fddecffa306d0134f93d06c804ece`, `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python`, `make verify-llama-compat verify-versions verify-kernel-dtype-keys`입니다. 현재 환경에서는 실제 Safari on macOS 27, VoiceOver, native browser 200% zoom 수동 검증을 실행하지 않았고, `docs/webui/design-system.md`에 후속 수동 gate용 정확한 체크리스트와 로컬 preview URL을 기록했습니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않으며, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.


## 수용 검증 경계

브라우저 테스트 12개 중 8개는 390·1024·1440 CSS px 너비에서 Darwin/Linux별로 분리된 screenshot baseline을 검사하며, 나머지는 상호작용과 appearance를 검사합니다. 통합 담당자의 screenshot 리뷰는 구현 기준선 확인이지 사용자 승인이 아닙니다. 최종 시각적 승인은 아직 필요합니다. 수동 피드백용 root preview는 최종 200% text-scale reflow 수정 이전의 고정 snapshot `b3f0cd04`이므로 최종 소스의 검증 증거가 아닙니다.

LoginView는 form과 password field에 `autocomplete="off"`를 요청하고 제출 또는 자체 logout 동작 때 로컬 입력값을 지웁니다. 이 힌트는 브라우저나 password-manager 확장의 자격 증명 저장을 막는 보장이 아닙니다. 세션 상태와 실제 인증은 공유 #1842 provider가 담당해야 하며, #1843 완료 또는 머지 전에 해당 통합이 필요합니다.

이번 단계별 마감 검증은 `b202f221`에서 typecheck, lint, unit 10개와 contract fixture 32개를 포함한 공유 정적 검사를 다시 통과했습니다. 브라우저 테스트 12개 및 플랫폼별 baseline 리뷰는 앞선 리뷰/CI 사이클의 증거이며 이번 문서 전용 수정에서 재실행하지 않았습니다. 실제 Safari, VoiceOver, 네이티브 브라우저 200% zoom, 사용자 screenshot 승인은 여전히 필요하며 GB10 예외로 면제되지 않습니다.

`b202f221`의 [Linux WebUI bundle job](https://github.com/lablup/mlxcel/actions/runs/34695797255/job/103558967545)은 typecheck·lint·unit·browser·generated-bundle verification 단계를 실제 실행하여 통과했습니다. 사용할 수 없는 GB10 job 두 개는 통과가 아니라 queued 상태입니다. 앞의 asset-tree digest와 manifest의 `source_digest_sha256`은 서로 다른 값이며, 후자는 `e5d0a9c8274fe0893d287339ea29f2598b01e493695337f7653b3ba7ecbd939e`입니다. 단계별 마감 검증에서 두 digest를 각각 다시 계산했습니다.
