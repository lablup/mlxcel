# PR #1865: WebUI 보안 정책 및 라우터 하니스

## 개요

PR #1865는 에픽 #1834와 이슈 #1837을 위해 번들 WebUI의 서버 측 보안 기반을 구현한다. 이번 변경은 라우터 모드 하니스와 #1838이 연결할 단일 앱/스타트업 빌더가 함께 사용할 수 있는 WebUI 보안 정책과 미들웨어를 추가한다. 정책은 health와 정적 shell 공개 동작을 유지하되, 모델 관리, 런타임 작업, 설정/속성, 이벤트를 기존 API 키 체계로 보호되는 관리자급 private surface로 취급한다.

## 구현

새 `src/server/webui/security.rs` 미들웨어는 요청을 기존 라우터 스택에 넘기기 전에 Host, Origin, Fetch Metadata, query parameter, bearer credential, 제어 요청 본문 크기, 제어 동시성, SSE 연결 수를 검증한다. 보안 라우터 accessor는 이 레이어를 Trace, CORS, 기존 auth 바깥에 설치하므로 악의적인 브라우저 메타데이터와 query에 심어진 secret이 하위 레이어에서 로그에 남거나 반사되기 전에 거부된다. 정책은 reverse proxy 배포를 위한 명시적 WebUI/API path prefix를 지원하고, literal credential query 이름뿐 아니라 percent-encoded 이름도 거부한다.

`src/server/webui/security/policy.rs`는 CSP, referrer, nosniff, permissions-policy header, 허용 Host/Origin 검증, path prefix 검증, control/SSE semaphore limit을 중앙화한다. `src/server/webui/security/startup.rs`는 startup resolver 계약을 추가한다. WebUI 비활성화는 정책 없음으로 끝나고, loopback interactive startup은 재시작 단위 credential을 생성할 수 있으며, noninteractive mode는 설정된 key를 요구하고, non-loopback WebUI는 명시적 key와 TLS를 요구하며, browser WebUI 경로에서 Unix socket은 거부된다. 생성 credential은 clone할 수 없고 Debug에서 redaction되며 terminal secret 노출은 값을 소비하는 방식으로 한 번만 가능하다.

## 검증

로컬 검증은 formatting, targeted security test, adversarial secured router test, scoped clippy, no-default-feature compilation, WebUI contract fixture, llama compatibility, workspace version consistency, kernel dtype key static check를 포함했다. 테스트는 누락/잘못된 key, public health 동작, private route 거부, hostile/null Origin, cross-site Fetch Metadata, public/private path의 credential query 거부, percent-encoded credential 이름, DNS rebinding Host 거부, preflight allowlist, legacy `GET /models?reload`, encoded private path, response body drop까지 유지되는 SSE permit, 선언 및 streamed body limit, startup 실패 모드, 엄격한 allowed Origin, custom path prefix classification을 다룬다.

## 보류 범위

production CLI/startup mounting은 의도적으로 #1838에 남겼다. frontend credential 처리, logout cleanup, browser UX는 형제 WebUI client/design 이슈가 맡는다. 이 PR은 서버 정책 PR이므로 CUDA/GB10, 실제 checkpoint, Safari, VoiceOver 검증을 주장하지 않는다. PR 게시 시점에 GB10 runner는 down 상태였고, 이번 변경은 inference path를 수정하지 않는다.
