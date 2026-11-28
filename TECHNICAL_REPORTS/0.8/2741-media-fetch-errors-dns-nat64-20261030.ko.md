# PR #2741: 미디어 fetch 오류, DNS 상한, NAT64 prefix 강화

**날짜**: 2026-10-30 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: b0c76bed에서 구현 완료(커밋 420c3a7c, c7c65f85, d1596463과 060b5051 시점 main의 merge). PR 본문에 따르면 대상 `server::media` 테스트, clippy, fmt, 문서 검증기 두 개가 GB10(spark-102)에서 통과함. 작성 시점에 호스팅 fast check는 통과했고 `cargo-clippy`, `CPU-only link`, `WebUI installed artifact`, `OpenXLA feature compile`은 대기 중이었으며, worktree에서 사전 머지 로컬 게이트가 실행 중이었음
**언어**: Rust, Markdown
**위험도**: 낮음 (시작 동작 변경 하나: `MLXCEL_MEDIA_NAT64_PREFIXES`가 이제 `2000::/3`과 `64:ff9b:1::/48` 밖의 prefix를 거부함. 미디어 DNS 조회가 이제 32개 permit 풀을 공유함)

## 요약

PR #2741(#2728 해결)은 #2718의 보안 리뷰가 원격 미디어 URL fetch(`src/server/media.rs`, `src/server/media_net*.rs`)에 남긴 LOW 잔여 문제 세 가지를 고친다. 첫째, 클라이언트에 보이는 미디어 오류가 더 이상 redirect hop을 드러내지 않는다. reqwest의 오류 텍스트는 실패한 URL을 담는데, redirect 이후 그 URL은 origin의 `Location` 헤더가 정한 값이다. 그래서 이 PR은 오류가 만들어지는 곳에서 `reqwest::Error::without_url()`로 URL을 지우고, `TooManyRedirects` 메시지에서도 hop을 뺀다. 둘째, 미디어 DNS 조회에 상한이 생겼다. 새 `media_net_dns` 모듈은 프로세스 전체에서 공유하는 semaphore permit 32개 중 하나를 쥔 채 `spawn_blocking`에서 `getaddrinfo`를 실행하고, permit은 resolver가 반환할 때까지 유지된다. 따라서 타임아웃으로 버려진 조회도 계속 집계된다. 셋째, 등록된 NAT64 prefix는 global-unicast 검사보다 우선하므로, `Nat64Prefix::parse`는 global unicast `2000::/3`과 RFC 8215 local-use 블록 `64:ff9b:1::/48` 밖의 prefix를 거부한다.

기본 설정의 main에서 세 항목 어느 것도 클라이언트가 사설 주소에 도달하게 하지는 않는다. 항목 1은 origin 서버가 고른 텍스트를 서버 자신의 오류 응답에 반사한다. 항목 2는 Tokio blocking pool에 대한 서비스 거부다. 항목 3은 운영자가 global이 아닌 prefix를 설정해야 성립한다.

## 1. 문제 정의

이슈 #2728이 세 잔여 문제를 모았고 main 59c07652에서 확인했다. 4.1절의 실패 우선 실행은 main 8385f40b에서 했다.

### 1.1 redirect hop URL이 클라이언트 오류에 노출되었다

reqwest 오류의 `Display`는 `for url (...)`을 덧붙인다. 미디어 클라이언트가 redirect를 따라가면 그 URL은 origin의 `Location` 헤더가 지정한 hop이다. 세 경로가 이를 embeddings, rerank, image-generation 라우트가 반환하는 400에 넣었다. hop에서의 전송 실패(`Failed to fetch image URL <url>: error sending request for url (<hop>)`), 2xx가 아닌 최종 응답(`image URL returned error status <url>: HTTP status client error (404 Not Found) for url (<hop>)`), 본문 읽기 오류다. 구현 중 네 번째 경로가 드러났다. `MediaUrlError::TooManyRedirects`가 `media URL redirect chain exceeded 5 hops at <hop>`을 출력했다. 클라이언트가 보낸 URL은 클라이언트가 봐도 되지만 hop은 그렇지 않다. hop에는 origin이 반사시키고 싶은 무엇이든(다른 경로로 알아낸 내부 호스트 이름, 토큰, 오류를 읽는 사람을 오도하려는 텍스트) 담을 수 있다.

### 1.2 미디어 DNS 조회가 blocking pool 전체를 점유할 수 있었다

`media_net.rs`의 `lookup_bounded`는 `tokio::net::lookup_host`를 5초 `DNS_LOOKUP_TIMEOUT`으로 감쌌다. `lookup_host`는 Tokio blocking pool에서 `getaddrinfo`를 부르고, `getaddrinfo`는 취소할 수 없다. 타임아웃은 future를 버릴 뿐 스레드를 멈추지 않는다. 그래서 nameserver가 응답하지 않는 호스트를 지정한 요청들이 시스템 resolver가 포기할 때까지 blocking pool 전체(기본 512 스레드)를 차지할 수 있었고, 서버의 다른 모든 `spawn_blocking` 사용자가 굶었다. 사전 검사와 hop별 검사(`ensure_media_url_allowed`), 연결 시점의 `PolicyResolver`(`resolve_allowed_addresses`)가 모두 이 함수를 거쳤다.

### 1.3 NAT64 prefix 설정이 global이 아닌 prefix를 받아들였다

`Nat64Prefix::parse`는 문법, RFC 6052 길이, host 비트, well-known 겹침, 예약된 `u` 옥텟을 검사했지만 prefix의 위치는 보지 않았다. #2718 이후 설계상, 등록된 prefix 안의 주소는 `2000::/3` global-unicast 검사보다 먼저 그 주소가 담은 IPv4로 판정된다. 따라서 link-local(`fe80::/10`)이나 unique-local(`fc00::/7`) prefix를 등록하면 그 안의 주소가 공인 IPv4를 담기만 하면 통과했고, 그 prefix가 실제 translator가 아니면 해당 사설 대역이 다시 열렸다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `media_net_redirect.rs` | `MediaSendError::Transport`를 만드는 유일한 곳인 `transport_or_refusal`이 hop을 debug로 기록하고(`hop = ?err.url()`) `err.without_url()`을 저장한다. 감싸진 `MediaUrlError` 거부는 여전히 `Refused`로 반환한다. |
| 1 | `media.rs` | `fetch_remote_bytes_with_limit_cancel`이 2xx가 아닌 응답 오류를 `{kind} URL returned error status {url}: {err.without_url()}`로 만들고, 본문 스트림 오류는 `with_context` 전에 `reqwest::Error::without_url`을 거친다. 클라이언트의 `{url}`은 남는다. video 경로는 로그만 남기므로 바뀌지 않았다. |
| 1 (리뷰) | `media_net.rs` | `MediaUrlError::TooManyRedirects`는 `media URL redirect chain exceeded 5 hops`를 출력한다. hop은 variant에 남고, `log_detail`에 이를 debug로 기록하는 분기가 추가되었다. |
| 2 | `media_net_dns.rs` (신규) | `lookup_bounded`는 5초 `DNS_LOOKUP_TIMEOUT`을 유지하고 `MAX_CONCURRENT_MEDIA_DNS_LOOKUPS = 32`, 즉 `OnceLock`에 든 프로세스 전체 `Arc<Semaphore>`를 추가한다. `lookup_bounded_with(resolve, host, port, timeout, permits)`는 하나의 deadline 안에서 `OwnedSemaphorePermit`을 얻고, 이를 `resolve`를 부르는 `spawn_blocking` closure로 옮긴 뒤 같은 deadline 안에서 handle을 기다린다. 포화(debug 로그), 닫힌 semaphore, resolver 오류, join 오류, 타임아웃은 모두 `MediaUrlError::Unresolvable`을 반환한다. resolver는 시스템 resolver인 `(host, port).to_socket_addrs()`다. |
| 2 | `media_net.rs` | 기존 `lookup_bounded`와 `DNS_LOOKUP_TIMEOUT`이 빠져나가고, 모듈은 `dns::lookup_bounded`(`cfg(test)`에서는 `lookup_bounded_with`도)를 import한다. 호출부는 그대로다. |
| 3 | `media_net_nat64.rs` | 기존 검사 뒤에 `parse`가 `is_global_translator_space(network, len)`를 부른다. 상위 3비트가 `001`(`2000::/3`)이거나, `len >= 48`이면서 앞 48비트가 `64:ff9b:1::`(`64:ff9b:1::/48`)이거나, 정확히 `64:ff9b::/96`이어야 한다. 아니면 `"<entry>" lies outside 2000::/3 and 64:ff9b:1::/48; a link-local, unique-local or other non-global prefix would let synthesized addresses reach private hosts` 오류를 내고, `configure_media_nat64_prefixes_from_env`가 이를 변수 이름을 담은 시작 실패로 바꾼다. |
| 문서 | `docs/environment-variables.md`, `docs/llama-server-compat.md` | `MLXCEL_MEDIA_NAT64_PREFIXES` 행에 범위 규칙과 동작 변경(ULA translator prefix는 더 이상 받지 않으며, global 또는 `64:ff9b:1::/48` 공간으로 재번호해야 함)을 적었다. compat 페이지의 주소 정책 문단도 같은 규칙을 적는다. |
| 테스트 | 테스트 파일 3개 | 새 테스트 6개(4.1절). disclosure 테스트에 loopback origin `serve_canned`와, 라우트가 400을 만드는 경로인 `fetch_labeled_image`를 거치는 `redirected_fetch_outcome` helper가 추가되었다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 10개 (신규 `media_net_dns.rs` 포함 소스 5, 테스트 3, 문서 2) |
| 추가 / 삭제 줄 | main 대비 +414 / -26 |
| 추가된 테스트 | 6개 (`media_url_disclosure_tests.rs` 4, `media_net_resolver_tests.rs` 1, `media_net_tests.rs` 1) |
| 새 의존성 | 없음 |
| 커밋 | 420c3a7c (항목 1~3), c7c65f85 (`TooManyRedirects` hop), d1596463 (본문 읽기 guard 테스트, DNS 테스트 안정화), b0c76bed (main merge) |

## 3. 기술적 선택과 그 이유

### 3.1 출력된 텍스트가 아니라 오류가 만들어지는 곳에서 URL을 지운다

**컨텍스트.** hop은 `reqwest::Error`를 포맷하는 모든 코드를 통해 클라이언트에 닿는다. 최종 메시지를 정리하는 방식(예: 패턴으로 `for url (...)` 제거)은 reqwest의 문구에 의존하고 새 포맷터를 놓친다.

**선택.** `transport_or_refusal`은 `MediaSendError::Transport`를 만드는 유일한 곳이므로, 여기서 `without_url()`로 지우면 이후의 모든 출력이 안전해진다. 이 함수를 거치지 않는 두 오류(redirect 루프가 끝난 뒤 생기는 2xx 외 상태와 본문 읽기)는 각자 생성 지점에서 지운다. hop은 사라지지 않고 지우는 지점 옆의 debug 로그로 가므로, 운영자는 여전히 체인이 어디로 갔는지 볼 수 있다.

**트레이드오프.** 자기 redirect 체인을 디버깅하는 클라이언트는 보는 정보가 줄어든다. 의도한 결과이고, hop은 서버 로그에 있다.

### 3.2 `TooManyRedirects`는 hop을 데이터로 갖되 출력하지 않는다

이슈는 경로 세 개를 적었다. 인수 조건은 더 넓었고("클라이언트에 보이는 어떤 미디어 오류도 클라이언트가 보낸 URL 외의 URL을 담지 않는다"), 그 조건에 맞춰 쓴 테스트(`too_many_redirects_omits_hop_url`)가 네 번째를 찾았다. variant는 `log_detail`이 기록할 수 있도록 `url` 필드를 유지하고 `#[error]` 문자열에서는 뺀다. `Unresolvable { host }`가 이미 쓰던 패턴과 같다. 필드는 로그용이고 `Display`는 클라이언트용이다.

### 3.3 async resolver가 아니라 `spawn_blocking`을 semaphore로 감싼다

| 선택지 | 장점 | 단점 |
|---|---|---|
| 타임아웃을 둔 `tokio::net::lookup_host` 유지 | 변경 없음 | 타임아웃이 스레드가 아니라 future를 버림. 이것이 버그 |
| async resolver (hickory-resolver) | 취소 가능, blocking 스레드 없음 | 새 의존성. 운영자가 의존하고 주소 정책이 따르는 `/etc/hosts`와 nsswitch를 우회함 |
| **선택: 32 permit semaphore 아래에서 `spawn_blocking`으로 시스템 resolver 실행** | resolver 의미 유지. 버려진 조회까지 포함해 스레드 수 제한. 의존성 없음 | 클라이언트 하나가 풀을 채울 수 있음(5절) |

permit은 blocking closure 안으로 옮겨지므로(`let _permit = permit;`) 호출자의 future가 버려질 때가 아니라 `getaddrinfo`가 반환할 때 해제된다. 상한에 필요한 성질이 바로 이것이다. 호출자 수를 세면 버려진 스레드가 이전과 똑같이 쌓인다. permit 획득과 조회는 하나의 deadline(`timeout_at`)을 공유하므로 호출자는 전체적으로 `DNS_LOOKUP_TIMEOUT`보다 오래 기다리지 않는다. 포화된 풀은 거부되었거나 해석되지 않는 호스트와 같은 클라이언트 텍스트인 `Unresolvable`을 반환하므로, 포화가 새로운 oracle이 되지 않는다.

32는 Tokio 기본 blocking 스레드 512보다 훨씬 작아서 미디어 이름 해석이 파일 I/O나 다른 `spawn_blocking` 사용자를 굶길 수 없고, 정상 서버에 필요한 조회 동시성(각 조회는 보통 수 밀리초 안에 끝남)보다는 훨씬 크다.

### 3.4 테스트를 위해 resolver와 semaphore를 주입한다

`lookup_bounded_with`는 blocking resolver closure와 `Arc<Semaphore>`를 인자로 받고, `lookup_bounded`는 `system_resolve`와 프로세스 전체 풀을 넘긴다. 그래서 테스트는 다른 테스트가 공유하는 전역 풀을 건드리지 않고 네트워크 없이, 2초 동안 잠들며 동시 호출 수를 세는 resolver와 자체 4 permit semaphore를 쓸 수 있다.

### 3.5 NAT64 prefix가 놓일 수 있는 위치

규칙은 `2000::/3` 또는 `64:ff9b:1::/48` 안에 완전히 들어가는 prefix를 받는다. local-use 검사는 `len >= 48`을 요구하므로 local-use 블록보다 넓은 `64:ff9b:100::/40`은 거부된다. well-known `64:ff9b::/96`은 둘 다의 밖에 있지만 특별히 받는다. 기존 테스트가 이를 parse하고, IANA 레지스트리에서 globally reachable이며, 주소 정책은 등록 여부와 무관하게 항상 이를 decode하므로 등록해도 달라지는 것이 없다.

이 검사는 상위 비트에 대한 범위 검사이지 IANA special-purpose 레지스트리 조회가 아니다. `2001:db8:64::/96`은 여전히 parse된다(이슈가 계속 동작해야 하는 prefix로 적었고 테스트도 쓴다). 이로 인해 남는 부분은 5절에 적었다.

## 4. 검증

### 4.1 실패 우선 증거

테스트는 수정 전에 작성되어 main 8385f40b에서 실행되었다.

```
redirect_hop_transport_error_omits_hop_url: the redirect hop URL leaked: ... Failed to fetch image URL http://127.0.0.1:39655/start.png: error sending request for url (http://127.0.0.1:40097/hop-secret-path)
redirect_hop_error_status_omits_hop_url: the redirect hop URL leaked: ... image URL returned error status http://127.0.0.1:45529/start.png: HTTP status client error (404 Not Found) for url (http://127.0.0.1:40087/hop-secret-path)
too_many_redirects_omits_hop_url: ... image URL rejected: media URL redirect chain exceeded 5 hops at http://127.0.0.1:46355/hop-secret-path
non_global_nat64_prefixes_are_refused: non-global NAT64 prefixes were accepted: ["fe80::/64 ...", "fd00:64::/96 ...", "fc00::/96 ...", "fec0::/96 ...", "64:ff9b:100::/40 ..."]
dns_lookups_are_bounded_by_the_permit_pool: error[E0425]: cannot find function `lookup_bounded_with` in this scope
```

`redirect_hop_body_read_error_omits_hop_url`(4096 바이트를 약속하고 `partial`만 보낸 뒤 닫는 hop)는 본문 읽기 strip이 있든 없든 통과한다. reqwest가 현재 본문 스트림 오류를 URL 없이 만들기 때문이다. 방어적 strip을 위한 회귀 guard이지 실패 우선 증거가 아니다.

`dns_lookups_are_bounded_by_the_permit_pool`은 2초 동안 잠드는 resolver에 대해 permit 4개, 타임아웃 200ms로 조회 100개를 실행한다. resolver 동시 실행 최대치가 4를 넘지 않는지, 모든 조회가 200ms에 여유 1초를 더한 시간 안에 자기 호스트를 담은 `Unresolvable`을 반환하는지(여전히 2초 수면보다 짧으므로 resolver를 기다린 조회는 실패함), 버려진 resolver 호출이 끝나면 permit이 모두 돌아오는지, 그 뒤의 조회가 성공하는지를 확인한다. 이슈는 500ms resolver를 지정했지만, d1596463이 이를 2초로 늘려 부하가 걸린 runner에서도 여유 시간이 "타임아웃됨"과 "기다림"을 구분하게 했다.

`non_global_nat64_prefixes_are_refused`는 `64:ff9b:1::/96`, `64:ff9b:1:abcd::/64`, `2001:db8:64::/96`이 여전히 parse되는지, `parse_media_nat64_prefixes("64:ff9b:1::/96, fd00:64::/96")`이 `MLXCEL_MEDIA_NAT64_PREFIXES`와 잘못된 항목을 담은 오류로 실패하는지도 확인한다.

### 4.2 로컬 테스트

최종 head에서 GB10, `--profile test-fast --features cuda`: `server::media_net::` 36개 통과, `server::media::tests::url_disclosure::` 9개 통과, `server::media::tests::net_proxy::` 10개 통과, `server::media` 전체 127개 통과. `cargo clippy --lib --tests -D warnings`, `cargo fmt --check`, `make verify-env-var-docs`, `make verify-llama-compat`이 깨끗했다.

### 4.3 수동 확인

`MLXCEL_ALLOW_PRIVATE_MEDIA_URLS`를 설정하지 않은 `mlxcel-server -m llama-nemotron-embed-vl-1b-v2`에서, 도달할 수 없는 `http://1.1.1.1:81/hop-secret-path`로 redirect하는 `/v1/embeddings` image URL은 hop 없이 400 `Failed to fetch image URL <submitted URL>: error sending request`를 반환한다. `MLXCEL_MEDIA_NAT64_PREFIXES=fe80::/64`로 시작하면 `"fe80::/64" lies outside 2000::/3 and 64:ff9b:1::/48; ...`로 실패한다.

### 4.4 호스팅 CI

작성 시점에 fast check(fmt, cargo-deny, env var docs, llama-compat manifest, license headers 등)는 통과했고, `cargo-clippy`, `CPU-only link`, `WebUI installed artifact`, `OpenXLA feature compile`은 대기 중이었다. CUDA, ROCm, Windows job은 변경 필터에 의해 건너뛰었다.

## 5. 남은 공백과 위험

- **클라이언트 하나가 DNS 풀을 채울 수 있다.** nameserver가 응답하지 않는 URL을 많이 보내는 클라이언트는 한 번에 최대 5초 동안 permit 32개를 모두 쥘 수 있고, 그동안 다른 클라이언트의 호스트 이름 미디어 fetch는 기다리다가 `Unresolvable`로 실패한다. 이 PR 전에는 같은 트래픽이 blocking pool 전체를 소진해 무관한 작업까지 해쳤으므로 영향 범위는 줄었지만, 호스트 이름 기반 미디어 fetch는 여전히 영향을 받는다. 요청별 공정성이나 negative cache는 별도 변경이다. IP literal URL은 사전 조회를 건너뛴다.
- **deadline에 permit 대기 시간이 포함된다.** 경쟁 상황에서는 조회가 resolver 시간을 5초보다 적게 받는다. 의도한 상한이지만, 부하 아래의 느리되 정상인 resolver가 이전에는 성공했을 곳에서 `Unresolvable`을 낼 수 있다.
- **호스트 이름 fetch 하나가 permit을 여러 번 쓴다.** 사전 검사와 연결 시점의 `PolicyResolver`가 각각 해석하고, redirect hop마다 사전 검사를 반복한다. 따라서 풀이 바쁠 때 호스트 이름 기반 fetch의 실질 동시성은 32보다 낮다.
- **본문 읽기 strip은 테스트로 증명할 수 없다.** reqwest는 현재 본문 스트림 오류에서 URL을 뺀다. strip은 앞으로의 reqwest 변경에 대비한 것이고, guard 테스트가 그 변경을 잡는다.
- **NAT64 규칙은 IANA 레지스트리가 아니라 범위 검사다.** 주소 정책이 다른 경우에는 거부하는 `2000::/3` 안의 prefix(문서용 `2001:db8::/32`, protocol-assignment 블록 `2001::/23`, 6to4 `2002::/16`)는 여전히 parse된다. 이런 prefix를 등록하면 그 안의 주소는 거부되지 않고 담긴 IPv4로 판정된다. 이 블록들은 link-local이나 unique-local처럼 운영자의 사설 호스트로 라우팅되지 않으므로 운영자 판단에 맡겼다.
- **동작 변경.** translator가 unique-local prefix(예: `fd00:64::/96`)를 쓰는 배포는 이제 시작에 실패한다. 해결책은 translator를 global 또는 `64:ff9b:1::/48` 공간으로 재번호하는 것이다. 사설 대상이 필요한 배포에는 `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`이 여전히 전면 opt-out으로 남아 있다.

## 6. 학습 포인트

### 6.1 오류 값은 요청 컨텍스트를 담고, 요청 컨텍스트는 공격자가 고를 수 있다

reqwest가 오류에 URL을 기록하는 데는 이유가 있지만, redirect 이후 그 URL은 원격 서버에서 온다. 신뢰 경계를 넘는 오류는 상대가 통제하는 필드가 있는지 확인해야 한다. `reqwest::Error::without_url()`은 이를 위해 있다. 오류를 만드는 곳에 적용하면 이후의 모든 포맷터가 보호되지만, 최종 문자열을 정리하는 방식은 그렇지 않다.

### 6.2 타임아웃은 blocking 작업을 취소하지 않는다

`spawn_blocking`에 위임하는 future를 감싼 `tokio::time::timeout`은 기다리기만 멈춘다. 스레드는 계속 돈다. 자원을 제한하려면 permit을 호출자의 수명이 아니라 스레드의 수명에 묶어야 하고, 그 방법은 permit을 blocking closure 안으로 옮기는 것이다.

```rust
let permit = tokio::time::timeout_at(deadline, permits.acquire_owned()).await??;
let lookup = tokio::task::spawn_blocking(move || {
    let _permit = permit; // 호출자가 포기할 때가 아니라 resolve가 반환할 때 해제
    resolve(&owned_host, port)
});
tokio::time::timeout_at(deadline, lookup).await
```

### 6.3 검사보다 우선하는 항목은 설정 시점에 그 검사를 받아야 한다

등록된 NAT64 prefix는 global-unicast 검사보다 먼저 평가되므로, 설정 parser가 그 검사가 강제했을 내용을 대신 강제해야 한다. 이후의 정책 검사를 우회하는 allow-list나 override는 모두 마찬가지다.

### 6.4 나열된 지점만이 아니라 인수 조건에 맞춰 테스트를 쓴다

이슈는 누출 경로 세 개를 적었다. 더 넓은 조건에 맞춰 쓴 테스트가 네 번째(`TooManyRedirects`)를 찾았다. 반대로 수정 없이도 통과하는 테스트는 d1596463이 본문 읽기 경우에 한 것처럼 회귀 guard라고 밝혀야 버그의 증거로 읽히지 않는다.

## 7. 참고 자료

- 이슈 #2728: Media URL fetch error text, DNS thread bound and NAT64 prefixes
- PR #2718과 이슈 #2713: 이 잔여 문제들을 기록한 보안 리뷰
- 이슈 #2693: NAT64 prefix decoding (`MLXCEL_MEDIA_NAT64_PREFIXES`)
- 이슈 #2644: 연결 시점 주소 검사인 `PolicyResolver`
- RFC 6052 (IPv4/IPv6 translator의 IPv6 주소 지정), RFC 8215 (local-use IPv4/IPv6 translation prefix)
- `docs/environment-variables.md` (`MLXCEL_MEDIA_NAT64_PREFIXES`), `docs/llama-server-compat.md` (원격 미디어 URL 주소 정책)
