# PR #1865: WebUI 보안 정책 및 라우터 하니스

## 개요

PR #1865는 에픽 #1834와 이슈 #1837을 위해 번들 WebUI의 서버 측 보안 기반을 구현한다. 이번 변경은 라우터 모드 하니스와 #1838이 연결할 단일 앱/스타트업 빌더가 함께 사용할 수 있는 WebUI 보안 정책과 미들웨어를 추가한다. 정책은 health와 정적 shell 공개 동작을 유지하되, 모델 관리, 런타임 작업, 설정/속성, 이벤트를 기존 API 키 체계로 보호되는 관리자급 private surface로 취급한다.

## 문제 정의

인증된 private route라도 현재 UI action만 열거하면 관리자 리소스 제한을 우회할 수 있다. 리뷰에서 기존 속성/설정 별칭과 후속 catalog/download/removal 경로의 이 누락을 발견했다. DTO 왕복 검사 역시 Rust 직렬화의 자기 일관성만 확인할 뿐 정식 오류 계약을 검증하지 못했다.

## 구현

새 `src/server/webui/security.rs` 미들웨어는 요청을 기존 라우터 스택에 넘기기 전에 Host, Origin, Fetch Metadata, query parameter, bearer credential, 제어 요청 본문 크기, 제어 동시성, SSE 연결 수를 검증한다. 보안 라우터 accessor는 이 레이어를 Trace, CORS, 기존 auth 바깥에 설치하므로 악의적인 브라우저 메타데이터와 query에 심어진 secret이 하위 레이어에서 로그에 남거나 반사되기 전에 거부된다. 정책은 reverse proxy 배포를 위한 명시적 WebUI/API path prefix를 지원하고, literal credential query 이름뿐 아니라 percent-encoded 이름도 거부한다.

`src/server/webui/security/policy.rs`는 CSP, referrer, nosniff, permissions-policy header, 허용 Host/Origin 검증, path prefix 검증, control/SSE semaphore limit을 중앙화한다. `src/server/webui/security/startup.rs`는 startup resolver 계약을 추가한다. WebUI 비활성화는 정책 없음으로 끝나고, loopback interactive startup은 재시작 단위 credential을 생성할 수 있으며, noninteractive mode는 설정된 key를 요구하고, non-loopback WebUI는 명시적 key와 TLS를 요구하며, browser WebUI 경로에서 Unix socket은 거부된다. 생성 credential은 clone할 수 없고 Debug에서 redaction되며 terminal secret 노출은 값을 소비하는 방식으로 한 번만 가능하다.

## 리뷰 수정과 기술적 선택

`security/request.rs`는 정규화된 `/ui-api/v1` 네임스페이스의 GET/HEAD/OPTIONS 이외 요청에 기본적으로 제어 제한을 적용하여 미래 경로도 보호한다. 기존 route inventory에서 확인한 `/models`, `/slots`, `/props`, `/settings`, `/v1/settings`, `/lora-adapters`, `/v1/cache/reset` 관리자 변경에도 요청률·동시 응답·본문 제한을 적용한다. 추론·토큰화·응답 취소·스트림 제어는 데이터 경로 제한을 유지한다. prefix 경계로 `/ui-api/v10`을 v1과 혼동하지 않는다.

실제 HTTP 오류 JSON 전체를 독립적으로 스키마 검증한 정식 fixture 8개와 비교한다. 완전한 token 형식을 검증한 `request_id`만 정규화하며, 코드·메시지·재시도 여부·추가 필드·생략과 null의 차이는 정확히 일치해야 한다. 음성 대조는 누락 필드, 추가/null 필드, 잘못된 코드, 메시지/재시도 변경, 잘못된 동적 ID를 거부한다. OpenAPI와 생성 TypeScript 변경은 없다.

## 검증

`cargo test --profile test-fast --features metal,accelerate security`에서 선택된 테스트 24개가 통과했다(실패·무시 0개). 관리자 경로 9종의 HTTP 요청률/용량/본문 사례 27개를 포함한다. `make verify-webui-contract`는 fixture 40개를 검증했다(마감 수정 전 32개). 로컬 검증은 formatting, targeted security test, adversarial secured router test, scoped clippy, no-default-feature compilation, WebUI contract fixture, llama compatibility, workspace version consistency, kernel dtype key static check를 포함했다. 테스트는 누락/잘못된 key, public health 동작, private route 거부, hostile/null Origin, cross-site Fetch Metadata, public/private path의 credential query 거부, percent-encoded credential 이름, DNS rebinding Host 거부, preflight allowlist, legacy `GET /models?reload`, encoded private path, response body drop까지 유지되는 SSE permit, 선언 및 streamed body limit, startup 실패 모드, 엄격한 allowed Origin, custom path prefix classification을 다룬다.

## 변경 요약

분류기를 별도 모듈로 옮겨 production 보안 파일을 500줄 이하로 유지했다. 회귀 검사는 관리자 경로 9종의 실제 HTTP 제한, prefix 분류, 전체 오류 비교를 다룬다. 새 분류기 회귀 테스트는 수정 전 `POST /props`에서 실패했다(1개 실패, exit 101). 따라서 수정 후 통과만 확인하는 자기 충족 테스트가 아니다.

## 고정 본문 제한 정합성

최종 계약 대조에서 미들웨어 기본값은 256 KiB지만 기존 OpenAPI `LimitSummary.json_body_bytes` 계약은 2,097,152바이트임을 발견했다. 후속 변경은 production 상수만 2 MiB로 맞추며 스키마·fixture·라우트 동작·추론을 변경하지 않는다. 범용 보안 래퍼 하니스에서 정확히 2,097,152 및 2,097,153바이트인 유효 JSON을 구성하고, Content-Length 선언과 Content-Length 없는 다중 청크 본문을 각각 검사한다. 경계 리터럴은 런타임 상수를 복사하지 않고 정식 OpenAPI 상수와 독립적으로 대조한다. 핸들러는 `Request<Body>`를 직접 소비하여 extractor 제한이 미들웨어 동작을 가리지 않도록 한다.

수정 후 선언/청크 모두 정확히 2,097,152바이트는 204로 허용하고 2,097,153바이트는 413으로 거부한다. 기존 큰 데이터 경로 회귀 테스트도 변경 없이 통과했다. 상수 수정 전 두 경계 회귀 테스트 모두 실패했다. 유효한 2,097,152바이트 선언/청크 요청이 204 대신 413을 받았다(2개 실패, exit 101).

직전 `92d77dbd` 스냅샷은 루트가 실행한 전체 workspace 게이트(11,184개 통과, 실패 0개, 무시 361개)와 workspace Clippy를 통과했다. 이 광범위 검증은 상수 수정 이전 결과이며, 후속 변경에 대한 새 GPU/전체 workspace 실행을 주장하지 않는다.

## 보류 범위

production CLI/startup mounting은 의도적으로 #1838에 남겼다. frontend credential 처리, logout cleanup, browser UX는 형제 WebUI client/design 이슈가 맡는다. 이 PR은 서버 정책 PR이므로 CUDA/GB10, 실제 checkpoint, Safari, VoiceOver 검증을 주장하지 않는다. 사용자가 GB10 runner의 down 상태를 확인한 후 사용 불가능한 필수 GB10 CI 생략을 명시적으로 승인하고 로컬 CI 통과를 요구했다. 이번 변경은 inference path를 수정하지 않는다.
