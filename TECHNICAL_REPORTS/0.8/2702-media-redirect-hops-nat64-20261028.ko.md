# PR #2702: 모든 media redirect hop 검사와 local-use NAT64 거부

**날짜**: 2026-10-28 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

PR #2649는 원격 media fetch에 SSRF 공백 두 개를 남겼고, 그 보고서도 두 항목을 검증되지 않은 것으로 적어 두었습니다. HTTP proxy 뒤에서는 public origin이 내부 hostname으로 redirect해도 mlxcel이 그 hop을 검사하지 않았습니다. reqwest는 proxy의 host만 resolve하고, hop의 이름은 proxy가 resolve했기 때문입니다. 또 잘 알려진 `64:ff9b::/96` 밖의 NAT64 주소는 public으로 취급되어, `64:ff9b:1::a9fe:a9fe`가 로컬 translator를 거쳐 `169.254.169.254`에 도달할 수 있었습니다. 이 PR(이슈 #2693)은 redirect 추적을 reqwest 밖으로 옮깁니다. media client는 `Policy::none()`으로 만들어지고, 새 파일 `src/server/media_net_redirect.rs`의 `send_following_redirects`가 proxy 여부와 상관없이 모든 hop에 대해 요청을 보내기 전에 `ensure_media_url_allowed` 전체 검사를 실행하며, 체인 전체가 하나의 10초 deadline을 공유합니다. RFC 8215 local-use 블록 `64:ff9b:1::/48`은 기본으로 거부됩니다. 새 환경 변수 `MLXCEL_MEDIA_NAT64_PREFIXES`는 RFC 6052 prefix 목록을 받고, 그 안의 주소는 내장된 IPv4 주소로 판정하며, 잘못된 항목이 있으면 시작이 실패합니다. 보안 검토에서 fail-closed 규칙 두 개가 추가되었습니다. `0.0.0.0/8` 전체를 거부하고, `64:ff9b::/96`을 덮는 /96보다 짧은 설정 prefix는 parse 단계에서 거부합니다.

## 1. 문제

#2649 이후 media fetch에는 네 가지 검사가 있었습니다. 요청 URL의 사전 검사(`ensure_media_url_allowed`: 형태, 로컬 resolve, 모든 주소), client의 DNS resolver로 설치된 `PolicyResolver`, redirect policy callback, 그리고 연결 후 peer 주소 검사입니다. proxy가 요청을 나르면 이 중 둘이 동작하지 않습니다.

- `media_redirect_policy()`는 reqwest redirect callback이었습니다. 이 callback은 동기 함수이므로 다음 URL에 대해 `ensure_media_url_shape`만 다시 실행할 수 있었습니다. IP literal 대상(`http://169.254.169.254/`)은 막았지만 hostname 대상은 `PolicyResolver`에 맡겼습니다.
- `http_image_client()`는 `.proxy()`나 `.no_proxy()`를 호출하지 않으므로, reqwest 0.12.28은 시스템 proxy matcher(`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY`, macOS와 Windows 시스템 설정)를 설치합니다. proxy를 거치면 reqwest는 proxy의 host만 resolve합니다. `http://intranet.internal/`로 가는 hop은 proxy가 resolve했고 `PolicyResolver`에는 도달하지 않았습니다. 연결 후 검사는 proxy의 주소만 보았습니다.

그 결과 환경에 proxy가 설정된 배포라면, `302 Location: http://localhost:PORT/`로 답하는 public URL 하나로 proxy가 내부 대상을 가져오게 만들 수 있었습니다. main 54d63b0a에서 먼저 실패하도록 작성한 테스트가 이를 그대로 보여 줍니다. 기록용 proxy의 로그에 두 번째 항목 `http://localhost:42731/x.png`가 남았고, fetch는 proxy의 502를 돌려받았습니다.

NAT64 공백은 `is_public_address`에 있었습니다. 이 함수는 `64:ff9b::/96`과 `2002::/16`(6to4)의 내장 IPv4만 해석했고, 나머지 IPv6 주소는 일반 규칙으로 판정했습니다. RFC 8215는 `64:ff9b:1::/48`을 local-use NAT64용으로 예약하며, 이 블록은 전역에서 도달할 수 없습니다. 운영자는 자기 global unicast 공간에서 떼어 낸 network-specific prefix(NSP)로 변환할 수도 있습니다. 두 경우 모두 public으로 통과했습니다.

## 2. 기술적 결정

### proxy를 쓸 때만이 아니라 모든 fetch에서 redirect를 직접 따라간다

client는 이제 `reqwest::redirect::Policy::none()`을 쓰고, `send_following_redirects(client, first)`가 체인을 따라갑니다. 따라갈 3xx를 받을 때마다 `Location`을 `response.url()`에 join하고, `ensure_media_url_allowed(next)`(형태, 로컬 resolve, resolve된 모든 주소)를 실행한 뒤에야 다음 요청을 보냅니다. 거부된 hop의 host로는 CONNECT도, absolute-form 요청 줄도 나가지 않습니다.

이슈는 proxy가 적용될 때만 수동 추적으로 바꾸는 방법을 검토했고 기각했습니다. reqwest는 proxy matcher를 노출하지 않고, 판정은 URL 단위입니다. `NO_PROXY`, scheme별 변수, CGI `REQUEST_METHOD` 억제, macOS/Windows 시스템 설정이 모두 영향을 주므로 한 체인 안에 proxy hop과 직접 hop이 섞일 수 있습니다. 이 판정을 재현하려면 `hyper_util` matcher를 직접 소유하고 redirect 경로를 두 개 유지해야 하며, 그중 하나는 proxy 배포에서만 실행됩니다. 수동 경로 하나는 환경 변수 proxy, 시스템 proxy, 나중에 설정될 proxy에 똑같이 동작하고, 기존의 proxy 없는 redirect 테스트가 계속 이 경로를 검증합니다.

`PolicyResolver`와 연결 후 검사는 그대로 남습니다. 직접 hop에서 resolver는 `check_hop` 자신의 lookup과 실제 연결 사이의 틈을 닫으며, 이제 hostname에 대한 심층 방어 계층이 됩니다. PR의 mutation 실행이 이 계층 구조를 보여 줍니다. `check_hop`을 제거하자 proxy 거부 테스트 두 개가 실패했지만, 직접 경로의 disclosure 테스트는 `PolicyResolver`가 그 hop을 혼자 거부했기 때문에 계속 통과했습니다.

### 체인 전체에 deadline 하나

예전에는 reqwest의 client 수준 `.timeout(10 s)`가 redirect 체인과 본문 읽기를 함께 덮었습니다. 수동 루프에서 hop마다 같은 timeout으로 보내면 요청 다섯 번에 최악 50초가 됩니다. helper는 `deadline = Instant::now() + budget`을 한 번만 잡고, 모든 hop을 `.timeout(deadline - now)`로 보냅니다. reqwest의 요청별 timeout은 돌려준 응답의 본문 읽기까지 덮으므로, 최종 본문도 같은 deadline에 묶입니다. hop 검사의 로컬 lookup도 `tokio::time::timeout_at(deadline, ...)`으로 제한되어, 공격자가 고른 이름의 느린 resolver가 체인을 늘릴 수 없습니다. `send_following_redirects_within(client, url, budget)`은 이를 위한 테스트 seam입니다. 두 번째 hop이 응답하지 않는 origin에 300ms budget을 주면 `is_timeout()`이 참인 `Transport`가 돌아옵니다.

시간이 다 된 lookup은 timeout이 아니라 `Refused(Unresolvable)`로 보고됩니다. 클라이언트는 어느 쪽이든 일반 문장 `MEDIA_HOST_REFUSED`를 보므로 정보가 새지는 않지만, 운영자의 경고 줄은 timeout이었던 일을 "거부"로 적습니다. 리뷰는 이를 LOW로 기록했습니다.

### reqwest policy와 같은 체인 의미

루프는 중요한 부분에서 제거된 policy의 관찰 가능한 동작을 유지합니다.

- 301, 302, 303, 307, 308만 따라갑니다. 그 밖의 상태는 3xx(예: 304)라도 그대로 호출자에게 돌려줍니다.
- 요청은 최대 `MAX_MEDIA_REQUESTS = 5`번입니다. 다섯 번째 응답도 따라갈 redirect이면 결과는 `Refused(TooManyRedirects { url: next })`로, 예전 `attempt.previous().len() >= 5`와 같습니다. 횟수 검사가 `check_hop`보다 먼저 실행되므로 여섯 번째 대상은 resolve도 연결도 되지 않습니다.
- 모든 hop은 본문 없는 GET으로 보냅니다. media fetch는 GET만 쓰므로 307과 308에서 잃는 것이 없습니다.
- `Referer` header를 보내지 않습니다. reqwest의 자동 policy는 이를 붙였지만 의존하는 곳이 없었고, 이전 URL을 다음 host에 보내지 않는 쪽이 더 안전한 기본값입니다.
- 따라갈 3xx인데 `Location`이 없거나, UTF-8이 아니거나, join되지 않으면 새 오류 `BadRedirect { status }`가 되고 `{kind} URL returned redirect {status} without a usable Location header`로 표시됩니다. 예전에 reqwest는 이런 응답을 그대로 돌려주었고, `error_for_status()`는 3xx를 거부하지 않으므로 redirect의 본문을 읽은 뒤 디코딩할 수 없는 media로 뒤늦게 실패했습니다.

중간 응답은 읽지 않고 버립니다. 읽지 않은 본문이 남아 있으면 연결이 닫히고, redirect 본문이 비어 있으면 연결이 pool로 돌아갈 수 있습니다. `drop(response)`의 주석은 리뷰 중에 정확히 이 내용으로 고쳐졌습니다. 어느 쪽이든 모든 hop을 검사하므로 안전합니다.

### 두 호출 지점의 오류 매핑

`MediaSendError`의 variant는 `Refused(MediaUrlError)`, `Transport(reqwest::Error)`, `BadRedirect` 세 가지입니다. `transport_or_refusal`은 reqwest 오류에 대해 여전히 `MediaUrlError::find_in`을 실행하므로, `PolicyResolver`가 낸 거부도 `Refused`로 돌아옵니다. `fetch_remote_bytes_with_limit_cancel`에서는 새 함수 `media_send_failure`가 거부를 `{kind} URL rejected: {MEDIA_HOST_REFUSED}`로 바꾸고 상세 정보는 `debug`로 남깁니다. 사전 검사의 거부와 같은 문장입니다. 따라서 hop 거부와 첫 URL 거부를 구분할 수 없고, #2649의 단일 문장 규칙이 유지됩니다. `fetch_remote_video`는 variant마다 기존의 `warn`과 `None` 형태를 유지합니다. `media_send_failure`는 기존 closure `rejected`를 호출하지 않고 같은 표시 로직을 반복합니다. 문장은 같고 테스트가 이를 확인합니다(LOW, 그대로 둠).

`media_client_builder()`를 `http_image_client()`에서 분리해, 테스트가 운영 client 설정(timeout, `Policy::none()`, `PolicyResolver`)을 그대로 만들고 자기 `.proxy(..)`를 붙이게 했습니다. `http_image_client()`는 여기에 추가 CA 인증서를 더해 공유 client를 만듭니다.

### NAT64: 기본은 fail-closed, 운영자가 지정한 것만 해석

`64:ff9b:1::/48` 안의 주소는 IPv4 주소가 어디 있는지 스스로 말해 주지 않습니다. RFC 8215는 블록 안에서 모든 RFC 6052 길이(/48, /56, /64, /96)를 허용하고, 길이마다 IPv4 바이트의 위치가 다릅니다. 가능한 모든 길이로 해석하는 방법은 통하지 않습니다. public 주소 93.184.216.34의 /96 합성인 `64:ff9b:1::5db8:d822`는 /48로 해석하면 `0.0.0.0`이 되어 거부됩니다. 운영자 NSP는 더 어렵습니다. 일반 global IPv6와 전혀 구분할 수 없습니다. 그래서 PR은 설정을 추가하고 기본값을 안전한 쪽으로 정했습니다.

- 아무것도 설정하지 않으면 `64:ff9b:1::/48` 안의 모든 주소를 거부합니다. 이 블록은 전역에서 도달할 수 없으므로 정상적인 public origin이 있을 수 없습니다.
- `MLXCEL_MEDIA_NAT64_PREFIXES`는 `addr/len` 형식의 prefix 목록이며 `len`은 {32, 40, 48, 56, 64, 96} 중 하나입니다. 목록의 prefix 안에 있는 주소는 `Nat64Prefix::embedded_ipv4`가 RFC 6052 2.2절에 따라 꺼낸 IPv4 주소로 판정하며, 96보다 짧은 길이에서는 octet 8(`u` octet)을 건너뜁니다. `u` octet이 0이 아니면 내장 주소가 없는 것으로 보고 거부합니다. 표준을 따르는 translator는 그런 주소를 만들지 않습니다. `64:ff9b:1::/96`을 지정하면 IPv6 전용 배포에서 DNS64로 합성된 public host에 다시 접근할 수 있고, NSP를 지정하면 그 주소가 public으로 통과하는 대신 해석됩니다.
- parse는 공백을 제거하고 빈 항목을 건너뜁니다. 해석되지 않거나, 다른 길이를 쓰거나, 길이 뒤의 비트를 세우거나, /96 prefix에서 `u` octet을 세운 값이 있으면 변수 이름과 항목을 담은 오류로 시작이 실패합니다. 오타 때문에 translator가 조용히 열려 있으면 안 됩니다. 두 서버 진입점(`src/bin/mlx_server.rs`, `src/commands/serve.rs`)은 `configure_private_media_urls` 옆에서 `configure_media_nat64_prefixes_from_env()`를 호출하고 오류를 `anyhow`로 바꿉니다.

`is_public_address_with(addr, nat64)`의 IPv6 규칙 순서는 다음과 같습니다. loopback, unspecified, multicast 거부; IPv4-mapped는 IPv4로 판정; 나머지 `::/96` 거부; 주소를 포함하는 첫 번째 설정 prefix가 결정; `64:ff9b::/96`은 예전처럼 해석; `64:ff9b:1::/48` 거부; 6to4 해석; ULA, link-local, documentation 거부. 설정 prefix를 well-known 규칙보다 앞에 두어 운영자의 목록이 우선하게 했습니다.

`is_public_address_with`는 순수 함수이고 테스트는 이것을 호출합니다. `is_public_address(addr)`는 `with_configured_prefixes`를 통해 `std::sync::RwLock<Vec<Nat64Prefix>>`에 담긴 프로세스 전역 목록을 읽으므로, `PolicyResolver`, 사전 검사, 연결 후 검사가 모두 같은 설정을 봅니다. lock은 시작할 때 한 번 쓰고, 검사하는 주소마다 한 번 읽으며, poisoning은 `PoisonError::into_inner`로 복구합니다.

이슈에 없던 추가 사항이 하나 있습니다. `u` octet이 세워진 /96 prefix는 parse에서 거부합니다. RFC 6052는 그 octet이 0이어야 한다고 정하고, 그런 prefix 아래의 주소는 어차피 모두 거부되므로, 받아들이면 운영자의 실수만 가려집니다.

### 리뷰 수정: 잘못된 길이는 fail-closed여야 한다

보안 검토에서 MEDIUM 수준의 fail-open 경로가 나왔습니다. 설정 prefix는 well-known 규칙보다 먼저 검사되는데, `parse`는 `64:ff9b::/96`을 더 짧은 길이로 덮는 prefix(예: `64:ff9b::/64`)를 받아들였습니다. 그 항목 아래에서 `64:ff9b::a9fe:a9fe`는 `0.0.0.169`로 해석됩니다. IPv4 규칙은 `0.0.0.0`만 거부했으므로 주소가 통과했고, /96 translator는 이를 `169.254.169.254`로 보냈습니다. 커밋 0bd4fad0이 규칙 두 개를 추가합니다.

- IPv4 규칙이 `0.0.0.0`뿐 아니라 `0.0.0.0/8`(RFC 791 "this network") 전체를 거부합니다. translator의 실제 길이보다 짧게 지정되고 남는 비트가 0인 목록은 이제 `0.0.0.x`로 해석되어 거부됩니다. /96 translator에 대해 `64:ff9b:1::/64`를 지정한 경우도 여기에 포함됩니다.
- `Nat64Prefix::parse`는 `64:ff9b::/96`을 포함하는 /96보다 짧은 prefix를 거부합니다. RFC 6052 2.1절은 well-known prefix를 /96으로만 정의합니다.

같은 커밋에서 proxy 문서에 DNS rebinding 문장을 추가했습니다.

## 3. 변경 요약

- `src/server/media_net_redirect.rs`(신규, 151줄): `MEDIA_FETCH_TIMEOUT`, `MAX_MEDIA_REQUESTS`, `MediaSendError`, `send_following_redirects`, `send_following_redirects_within`, 그리고 hop의 lookup을 체인 deadline으로 제한하는 `check_hop`.
- `src/server/media_net_nat64.rs`(신규, 197줄): `Nat64Prefix`(`parse`, `contains`, `embedded_ipv4`), `parse_media_nat64_prefixes`, 프로세스 전역 prefix 목록, `configure_media_nat64_prefixes_from_env`, `with_configured_prefixes`.
- `src/server/media_net.rs`: 두 모듈을 `#[path]`로 mount(`media_net_test_guard.rs`와 같은 방식, 파일은 533줄); `is_public_address`를 전역을 읽는 함수와 순수 함수 `is_public_address_with`로 분리; 설정 prefix와 local-use 규칙; `0.0.0.0/8` 거부; `media_redirect_policy()` 삭제; 검사 2~4와 proxy 단락의 module doc 재작성.
- `src/server/media.rs`: `media_client_builder()` 분리; 두 fetch 경로가 `send_following_redirects` 호출; labeled-image 경로는 `media_send_failure`로, video 경로는 인라인으로 오류 매핑.
- `src/server/mod.rs`, `src/bin/mlx_server.rs`, `src/commands/serve.rs`: `configure_media_nat64_prefixes_from_env`의 re-export와 시작 시 연결.
- `docs/environment-variables.md`: `MLXCEL_MEDIA_NAT64_PREFIXES` 행(`make verify-env-var-docs`가 강제). `docs/llama-server-compat.md`: 검사 2~4, proxy 단락(split-horizon DNS와 DNS rebinding을 proxy egress policy의 잔여 위험으로 명시), `0.0.0.0/8`과 local-use 블록을 넣은 거부 목록.
- 테스트: `src/server/media_net_proxy_tests.rs`(신규, 436줄)는 loopback에 기록용 forward proxy, origin, 내부 listener를 띄우고, proxy 뒤의 hostname·literal redirect 거부, proxy를 거쳐 200에 도달하는 체인, 요청 다섯 번 제한, `Location` 누락, 따라가지 않는 3xx, deadline 두 경우를 다룹니다. `media_net_tests.rs`에는 NAT64 테스트 11개가 추가되었고, 192.0.2.33에 대한 RFC 6052 2.4절 예시 여섯 개, 운영자 NSP 해석, parse 거부, 환경 변수 목록 trim, 리뷰 수정 테스트 두 개가 포함됩니다. `media_url_disclosure_tests.rs`는 직접 경로 redirect 테스트를 `media_client_builder()`와 helper를 쓰도록 다시 작성했습니다.

## 4. 트레이드오프와 잔여 위험

- **proxy는 이름을 다시 resolve한다.** mlxcel은 자기 resolver로 각 hop을 검사하고, 실제 연결을 위해 proxy가 같은 이름을 다시 resolve합니다. DNS 시야가 host와 다른 proxy(split horizon)나, mlxcel에는 public 주소를 주고 proxy에는 private 주소를 주는 낮은 TTL 이름(rebinding)은 여전히 proxy 자신의 egress policy에 달려 있습니다. 이는 이후 hop뿐 아니라 첫 URL에도 해당하며, 문서에 이제 그렇게 적혀 있습니다.
- **loopback proxy는 자기의 성공한 fetch를 거부한다.** 연결 후 검사는 proxy의 주소를 봅니다. proxy가 `127.0.0.1`에 있으면 완전히 성공한 proxy fetch도 `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1` 없이는 거부되고, 이 변수는 새 hop 검사를 포함한 모든 검사를 끕니다. 운영자가 설정한 proxy를 연결 후 검사에서 제외하는 일은 이슈의 범위 밖이었고 바뀌지 않았습니다.
- **직접 hop마다 lookup 한 번 추가.** 직접 hop은 `check_hop`에서 한 번, 연결 시 `PolicyResolver`에서 한 번 resolve됩니다. 두 lookup 모두 제한되어 있고 체인 deadline을 공유합니다. proxy 없이는 public 이름을 resolve하지 못하는 host는 main에서도 이미 사전 검사에서 실패했으며, 이 점은 바뀌지 않았습니다.
- **잘못된 길이의 NAT64 목록은 남는 비트가 0일 때만 fail-closed다.** `0.0.0.0/8` 규칙은 translator 길이보다 짧고 남는 비트가 0인 목록을 막으며, 이것이 흔한 경우입니다(/96 translator에 대한 `64:ff9b::/64`, `64:ff9b:1::/64`). 모든 불일치를 막지는 못합니다. 실제보다 긴 목록(/64 translator의 NSP를 /96으로 지정)은 합성된 주소를 포함하지 않으므로, 그 주소는 일반 규칙으로 넘어가 public으로 통과합니다. NSP에 대해서는 PR 이전 동작과 같습니다. 실제보다 짧으면서 남는 network 비트가 0이 아닌 목록은 network 바이트를 IPv4 결과에 섞어 넣습니다. translator가 `2001:db9:64:1200::/56`인데 `2001:db9:64::/48`을 지정하면, `169.254.169.254`의 합성 주소는 `18.169.254.169`로 해석되어 통과합니다. 두 경우 모두 운영자가 길이를 잘못 지정해야 생기고, 주소만으로는 감지할 수 없습니다. 지정한 길이는 translator의 길이와 같아야 합니다.
- **기존 IPv6 공백.** `fec0::/10`(폐기된 site-local), Teredo `2001::/32`, RFC 2765 `::ffff:0:0:0/96`, `100::/64`는 main과 마찬가지로 여전히 public으로 취급됩니다. 일반적인 스택에서는 어느 것도 도달할 수 없습니다.
- **DNS 시간 차이와 public transport 오류.** #2649 이후와 마찬가지로 NXDOMAIN과 거부된 private 응답은 같은 문장이지만 지연 시간은 다르고, 연결에 실패한 public host는 여전히 reqwest의 transport 문장을 돌려줍니다.

## 5. 검증

**main 54d63b0a에서 먼저 실패.** 새 테스트를 main의 API(`Proxy::all` 뒤의 `media_redirect_policy()`와 `.send()`, 그리고 `is_public_address`)에 맞춰 작성했습니다. proxy redirect 테스트는 `Ok(("http://localhost:42731/x.png", 502))`를 돌려받았고 proxy 로그는 `["http://media.test/start.png", "http://localhost:42731/x.png"]`였습니다. `local_use_nat64_forms_are_refused`와 `an_operator_nat64_prefix_form_of_a_refused_address_is_refused`는 첫 주소에서 실패했습니다. `Nat64Prefix`, deadline, `BadRedirect` 테스트는 main에 없는 API를 씁니다.

**mutation 검사.** `check_hop`을 건너뛰고 hop마다 전체 budget을 주자 proxy 테스트 세 개가 실패했습니다(`behind_a_proxy_a_redirect_onto_an_internal_hostname_never_leaves`, `behind_a_proxy_a_redirect_onto_an_internal_literal_never_leaves`, `hops_that_each_fit_the_budget_still_share_it`). 리뷰 수정 규칙 두 개를 끄자 `this_network_ipv4_block_is_refused_whole`(`0.1.2.3 must be refused`)와 `a_prefix_shorter_than_96_covering_the_well_known_prefix_is_rejected`가 실패했습니다.

**테스트 묶음**(GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `gpu-lock` 아래에서 모듈 하나씩, 0bd4fad0 기준): `server::media` 115개 통과, `server::routes::embeddings` 23개, `server::routes::rerank` 20개. `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, `make verify-env-var-docs` 모두 통과했습니다.

**실제 서버**(GB10, 945d43d7에서 실행; 리뷰 커밋은 redirect 경로를 건드리지 않음). `colsmol-256m-merged`를 올린 `mlxcel-server`, `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS` 해제, `HTTP_PROXY`와 `HTTPS_PROXY`는 `127.0.0.1:19652`의 기록용 forward proxy, 기록용 listener는 `127.0.0.1:19651`:

| 경우 | 결과 |
|---|---|
| `MLXCEL_MEDIA_NAT64_PREFIXES=64:ff9b:1::/50` | 시작 실패(exit 1), 오류에 변수와 항목 표시 |
| `https://postman-echo.com/redirect-to?url=http://localhost:19651/x.png` | 400, `image URL rejected: media URL host is not allowed or could not be resolved` |
| `http://[64:ff9b:1::7f00:1]:19651/x.png` | 400, 같은 문장 |
| `data:` PNG | 200, embedding 목록 |

proxy는 `CONNECT postman-echo.com:443`만 기록했고, listener에는 연결이 없었으며, debug 로그는 redirect 경우의 거부를 `host="localhost" address=127.0.0.1`로 기록했습니다. 이슈의 계획에서 두 가지를 바꿨습니다. httpbin은 이제 redirect 대상이 내부 주소이면 403으로 답하고 302를 보내지 않으므로 `httpbin.org/redirect-to` 대신 `postman-echo.com/redirect-to`를 썼고, listener는 18391 대신 19651 포트를 썼습니다.

**검증하지 못한 것.** proxy를 거쳐 끝까지 성공하는 체인(마지막 200)은 테스트 opt-in 아래의 `a_redirect_chain_through_the_proxy_reaches_its_final_response`로만 확인했습니다. 위의 loopback proxy 잔여 위험 때문에 실제 서버 실행에서는 opt-in 없이 이를 보일 수 없습니다. 실제 NAT64 translator가 없었으므로 NAT64 검증은 단위 테스트(RFC 6052 예시 포함)와 literal host 경우뿐입니다. 테스트 묶음과 실제 서버 실행은 origin/main merge(fdfe7f6f) 이전에 이루어졌습니다. 이 merge로 들어온 #2687, #2688, #2700, #2701은 `src/server/media*`, 시작 지점, 여기서 바꾼 문서 중 어느 것도 건드리지 않습니다.

## 6. 학습 포인트

- reqwest의 redirect policy는 동기 함수이고 resolve 전에 실행되며, proxy를 거치면 client는 대상을 아예 resolve하지 않습니다. 이름을 resolve해야 하는 보안 검사는 그 callback 안에 둘 수 없습니다. redirect를 끄고(`Policy::none()`) 비동기 루프에서 따라가는 것만이 전송 방식과 상관없이 모든 hop을 보내기 전에 resolve하고 검사할 수 있는 자리입니다.
- client 수준 timeout을 수동 루프로 바꿀 때는 timeout이 아니라 deadline을 넘겨야 합니다. 하나의 `Instant`에서 계산한 요청별 `.timeout(remaining)`은 최악의 경우를 원래 한도로 유지하고, hop별 lookup을 `timeout_at`으로 제한하면 남은 틈도 닫힙니다.
- RFC 6052 주소는 자기 prefix 길이를 담고 있지 않습니다. 길이를 알 수 없으면 거부하고, 운영자가 알려 주면 그 길이로만 해석하며, 잘못된 값은 시작을 실패시켜야 합니다.
- 해석에 잘못된 길이가 들어갈 수 있다면, 잘못된 답이 어떤 모양인지 확인해야 합니다. 0으로 채워진 불일치는 `0.0.0.0/8`에 떨어지므로, 그 블록 전체를 거부하면 한 종류의 잘못된 설정이 fail-open에서 fail-closed로 바뀝니다. 남는 비트가 0이 아닌 불일치는 막지 못합니다.
- 새 검사를 제거하는 mutation 테스트는 다른 계층이 여전히 통과할 때 더 많은 것을 알려 줍니다. 여기서는 `PolicyResolver`가 hop을 잡았기 때문에 직접 경로 테스트가 계속 통과했고, 이것이 각 경우를 실제로 어느 계층이 맡는지 보여 줍니다.
