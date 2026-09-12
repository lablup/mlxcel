# PR #1863 WebUI 디자인 시스템 보고서

## 범위

PR #1863은 에픽 #1834의 공유 WebUI 디자인 시스템 계층을 구현합니다. semantic CSS 토큰, 절제된 macOS 27 방향의 셸, typed 영어/한국어 문자열, 재사용 가능한 primitive, appearance 저장, `#/gallery` 컴포넌트 갤러리를 제공합니다. 글래스는 장식적 chrome에만 쓰고 콘텐츠는 중립 표면에 올리며, Apple 에셋 대신 프로젝트 작성 SVG 아이콘을 사용하고 각 레이아웃·재질 결정에 사용한 공식 Apple reference를 문서화했습니다.

## 리뷰 보강

첫 리뷰 사이클에서 영구 모바일 rail을 native dialog 기반 내비게이션 sheet로 교체했고, modal focus containment/restoration, roving tabs, storage error handling, high-contrast system/on/off preference, axe 기반 브라우저 검사, screenshot 비교 baseline, 정직한 placeholder state, 인증 및 schema mismatch surface, same-origin/offline 검증을 추가했습니다. production 화면은 공유 #1842 provider가 통합될 때까지 사실을 과장하지 않는 placeholder로 남습니다.

## 검증 상태

로컬 자동 검증 범위는 TypeScript, lint, unit test, Playwright 접근성/키보드/viewport/screenshot 검사, deterministic bundle rebuild verification입니다. 현재 환경에서는 실제 Safari on macOS 27 및 VoiceOver 수동 검증을 실행하지 않았고, `docs/webui/design-system.md`에 후속 수동 gate용 정확한 체크리스트와 로컬 preview URL을 기록했습니다. CUDA/GB10 검증은 이 디자인 시스템 PR 범위가 아니며 여기서 통과로 주장하지 않습니다.
