# PR #1863 WebUI 디자인 시스템 보고서

## 범위

PR #1863은 에픽 #1834의 공유 WebUI 디자인 시스템 계층을 구현합니다. semantic CSS 토큰, 절제된 macOS 27 방향의 셸, typed 영어/한국어 문자열, 재사용 가능한 primitive, appearance 저장, `#gallery` 컴포넌트 갤러리를 제공합니다. 글래스는 장식적 chrome에만 쓰고 콘텐츠는 중립 표면에 올리며, Apple 에셋 대신 프로젝트 작성 SVG 아이콘을 사용하고 각 레이아웃·재질 결정에 사용한 공식 Apple reference를 문서화했습니다.

## 리뷰 보강

수정 사이클에서 production 화면의 과장된 placeholder를 하나의 중립 connection prompt로 교체했고, Gallery를 primary navigation에서 숨기되 direct artifact route로 유지했으며, production route의 빈 inspector chrome을 제거하고 desktop grid를 정리했습니다. 또한 controlled LoginView 및 SchemaMismatchView contract를 추가했고, 브라우저 테스트를 단순 시각 smoke에서 실제 동작 검증으로 확장했습니다. production 화면은 공유 #1842 provider가 머지된 뒤 통합될 때까지 사실을 과장하지 않는 placeholder로 남으며, 이 PR은 #1842 provider state를 복사하거나 별도 cache하지 않습니다.

## 검증 상태

최종 로컬 검증 범위는 `pnpm --dir webui run typecheck`, `pnpm --dir webui run lint`, `pnpm --dir webui run unit`의 Vitest 9개 통과, `pnpm --dir webui run browser`의 Playwright 12개 통과, deterministic bundle verification digest `2a94d3fe3f243f065b55772e61b368b01b5cb23d894065267c959f41717409f3`, `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python`, `make verify-llama-compat verify-versions verify-kernel-dtype-keys`입니다. 현재 환경에서는 실제 Safari on macOS 27, VoiceOver, native browser 200% zoom 수동 검증을 실행하지 않았고, `docs/webui/design-system.md`에 후속 수동 gate용 정확한 체크리스트와 로컬 preview URL을 기록했습니다. 필수 GB10 runner가 down 상태이므로 CUDA/GB10 검증은 통과로 주장하지 않으며, 합의된 진행 방식은 unavailable required GB10 job skip 및 local CI 통과입니다.
