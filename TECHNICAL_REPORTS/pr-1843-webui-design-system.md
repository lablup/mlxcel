# PR #1843 WebUI design system report

## English

This PR implements the shared WebUI design-system layer for epic #1834. It replaces the temporary scaffold with semantic CSS tokens, a responsive macOS 27-inspired shell, typed English/Korean strings, reusable primitives, appearance persistence and a component gallery at `#/gallery`. The implementation reserves glass for decorative chrome, keeps content on neutral surfaces, avoids proprietary Apple assets and records the official Apple references that informed each layout and material decision.

Validation planned in this branch covers TypeScript, lint, unit tests, Playwright accessibility/keyboard/viewport checks and deterministic bundle rebuild verification. Actual Safari and VoiceOver manual verification remain a downstream/manual gate because the current browser plugin session has no controllable browser surface.

## 한국어

이 PR은 에픽 #1834의 공유 WebUI 디자인 시스템 계층을 구현합니다. 임시 스캐폴드를 semantic CSS 토큰, macOS 27 방향의 반응형 셸, typed 영어/한국어 문자열, 재사용 가능한 primitive, appearance 저장, `#/gallery` 컴포넌트 갤러리로 교체했습니다. 글래스는 장식적 chrome에만 쓰고 콘텐츠는 중립 표면에 올리며, Apple 독점 에셋은 번들하지 않고 각 레이아웃·재질 결정에 사용한 공식 Apple reference를 문서화했습니다.

이 브랜치의 검증 범위는 TypeScript, lint, unit test, Playwright 접근성/키보드/viewport 검사, deterministic bundle rebuild verification입니다. 현재 브라우저 플러그인 세션에서 제어 가능한 브라우저 surface가 없어 실제 Safari 및 VoiceOver 수동 검증은 후속 수동 gate로 남습니다.
