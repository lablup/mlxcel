# PR #2771: 미디어 DNS 도메인별 상한, hop scheme 비공개, NAT64 레지스트리 검사

**날짜**: 2026-10-31 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: a955fe97에서 구현 완료(커밋 fdc89ca2, 72d5fc54와 main merge). PR 본문에 따르면 대상 `server::media` 테스트, clippy, fmt, 문서 검증기 두 개가 GB10(spark-102)에서 통과했고, 실제 `mlxcel-server`로 NAT64 시작 거부와 hop scheme 비공개를 확인함. 작성 시점에 `license/cla`를 제외한 호스팅 check는 모두 대기 중이었고, worktree에서 사전 머지 로컬 게이트가 실행 중이었음
**언어**: Rust, Markdown
**위험도**: 낮음 (시작 동작 변경 하나: `MLXCEL_MEDIA_NAT64_PREFIXES`가 IANA special-purpose 블록 네 곳 안의 prefix를 거부함. 미디어 DNS 조회 한 번의 최악 시간이 5초에서 7초로 늘어남)

## 요약

PR #2771(#2764 해결)은 #2741이 원격 미디어 SSRF 방어(`src/server/media_net*.rs`)에 남긴 잔여 문제 다섯 가지를 고친다. 핵심은 공정성이다. #2741은 미디어 DNS 조회를 permit 32개 풀로 제한했지만, nameserver가 응답하지 않는 도메인의 서브도메인을 대량으로 지정하는 클라이언트 하나가 permit 32개를 모두 잡아 다른 모든 클라이언트의 hostname fetch를 `Unresolvable`로 실패시킬 수 있었다. 이제 조회는 전역 permit보다 먼저 자기 도메인 키(host의 마지막 두 label)의 permit 4개 중 하나를 잡으므로, 풀을 채우려면 그런 도메인이 최소 8개 필요하다. 나머지 네 가지 변경은 permit 대기에 별도 2초 상한을 두어 resolver가 5초를 온전히 쓰게 하고, pre-flight DNS 응답을 30초 동안 pin해 hop마다 두 번이 아니라 한 번만 조회하게 하며, redirect 대상의 scheme(origin의 `Location`이 정한 값)을 클라이언트에 되돌려주지 않고, `2001::/23`, `2001:db8::/32`, `2002::/16`, `3fff::/20` 안의 NAT64 prefix를 거부한다.

main에서 다섯 항목 어느 것도 클라이언트가 사설 주소에 도달하게 하지는 않는다. 항목 1은 다른 클라이언트의 미디어 fetch에 대한 서비스 거부(이슈에서 MEDIUM 우선순위)이고, 항목 2부터 5는 LOW다.

## 1. 문제 정의

이슈 #2764는 #2741의 보안 리뷰와 `TECHNICAL_REPORTS/2741-media-fetch-errors-dns-nat64-20261030.en.md` 5절의 잔여 문제를 모았다. 4.1절의 failing-first 실행은 main 5c8ef80a 기준이다.

### 1.1 클라이언트 하나가 DNS permit 풀을 채울 수 있었음

`lookup_bounded_with`는 프로세스 전체의 `MAX_CONCURRENT_MEDIA_DNS_LOOKUPS = 32` permit 중 하나를 잡아 `spawn_blocking` closure로 옮겼으므로, 버려진 조회도 `getaddrinfo`가 반환할 때까지 permit을 유지했다. 이것이 blocking thread를 제한하는 장치지만, 동시에 nameserver가 응답하지 않는 host를 지정하는 클라이언트(공격자 도메인 하나의 임의 서브도메인은 비용이 없다)가 system resolver의 타임아웃 동안 permit 전부를 잡을 수 있다는 뜻이기도 했다. 그러면 다른 모든 클라이언트의 hostname fetch가 5초를 기다린 뒤 `Unresolvable`로 실패했다. IP literal URL은 영향이 없었다.

### 1.2 permit 대기가 resolver 시간을 깎았음

`DNS_LOOKUP_TIMEOUT`(5초) 하나의 deadline이 permit 대기와 resolver 호출을 함께 덮었다. 경합 중에는 정상이지만 느린 resolver가 남은 시간만 받았으므로, 성공했을 조회가 실패했다.

### 1.3 hop마다 두 번 조회했음

pre-flight 검사 `ensure_media_url_allowed`가 host를 조회하고, connect 시점의 `PolicyResolver`(`resolve_allowed_addresses`)가 다시 조회했다. `check_hop`은 redirect hop마다 pre-flight를 반복한다. 따라서 hostname hop 하나가 permit 두 개를 썼고, 부하 중 실효 동시성은 32보다 낮았으며, connect가 pre-flight가 승인한 것과 다른 응답을 쓸 수 있었다(그 응답도 필터링되었으므로 우회가 아니라 일관성 문제였다).

### 1.4 hop의 scheme이 클라이언트에 전달되었음

`check_hop`은 hop에 `ensure_media_url_allowed(next.as_str())`를 실행했고, `MediaUrlError::Scheme { scheme }`은 `media URL scheme {scheme:?} is not supported`로 렌더링된다. redirect 이후 scheme은 origin의 `Location`에서 오므로, origin이 서버가 반환하는 400에 원하는 토큰을 넣을 수 있었다. `Malformed { url }`도 같은 방식으로 hop URL 전체를 노출할 수 있지만, hop은 이미 파싱된 상태이므로 이 경로는 이론적이다. #2741이 transport 오류와 `TooManyRedirects`에서 막은 것과 같은 종류의 노출이다.

### 1.5 NAT64 prefix를 레지스트리가 아닌 범위로 검사했음

#2741은 `Nat64Prefix::parse`가 `2000::/3` 또는 `64:ff9b:1::/48` 안의 prefix만 받게 했다. `2000::/3` 안에서는 문서용 prefix `2001:db8::/32`, IETF protocol-assignment 블록 `2001::/23`, 6to4 `2002::/16`, 문서용 prefix `3fff::/20`이 여전히 통과했다. 주소 정책은 네 블록의 주소를 모두 거부하지만 등록된 prefix가 정책보다 우선하므로, 이 중 하나를 등록하면 해당 주소들이 거부되는 대신 내장된 IPv4로 판정되었다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `media_net_dns.rs` | 새 `DnsPool { global, domains: Mutex<DomainTable> }`. `lookup_bounded_with`는 `domain_key(host)`(소문자로 바꾼 마지막 두 label, 끝의 점 무시, label이 둘 이하이면 host 자체)를 계산하고, 그 키의 semaphore(`MAX_CONCURRENT_LOOKUPS_PER_DOMAIN = 4`)에서 permit을 잡은 뒤 전역 permit을 잡아, 둘 다 `let _permits = (domain_permit, global_permit);`로 blocking closure에 옮긴다. |
| 1 | `media_net_dns.rs` | `DnsPool::domain`은 삽입이 `prune_at`에 도달할 때 idle 키(strong count 1이고 permit이 모두 남은 키)를 정리한다. `prune_at`은 1024에서 시작해 정리 후마다 `max(1024, 2 * len)`이 된다. |
| 2 | `media_net_dns.rs` | `DnsLimits { permit_wait, lookup_timeout, per_domain }`와 `DnsLimits::DEFAULT`(2초, 5초, 4). 두 permit은 하나의 `permit_wait` deadline 안에서 잡고(`acquire_by`, 포화 시 debug 로그), resolver는 그다음 `tokio::time::timeout(lookup_timeout, ..)` 아래에서 실행된다. |
| 3 | `media_net_pin.rs` (신규) | `PinStore`: 소문자 host에서 `(Vec<SocketAddr>, 만료 시각)`으로의 map, TTL 30초, 최대 1024 host(만료된 항목 먼저, 그다음 가장 오래된 항목 제거). connector가 URL의 port를 넣도록 port는 0으로 지운다. |
| 3 | `media_net.rs` | `ensure_media_url_allowed`는 모든 주소가 통과한 뒤 응답을 pin한다. `resolve_allowed_addresses`는 사설 주소가 거부되는 경우 살아 있는 pin을 읽고 여전히 `filter_allowed_addresses`를 적용하며, pin이 없으면 기존대로 조회한다. `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`이면 pre-flight는 아무것도 조회하지 않고, pin을 쓰지도 읽지도 않는다. |
| 4 | `media_net.rs`, `media_net_redirect.rs` | 새 variant `RedirectScheme { scheme }`("media URL redirect target uses an unsupported scheme; only http and https are followed")과 `RedirectMalformed { url }`("media URL redirect target is not a valid absolute URL"), 둘 다 `log_detail`에 debug arm이 있다. `MediaUrlError::for_redirect_target`이 `Scheme`과 `Malformed`를 변환하고, `check_hop`이 hop의 pre-flight 결과에 적용한다. 클라이언트 자신의 URL은 기존 메시지를 유지한다. |
| 5 | `media_net_nat64.rs` | `SPECIAL_PURPOSE_BLOCKS`에 네 블록과 RFC를 나열하고, `parse`는 network가 그 안에 있는 prefix를 블록 이름과 함께 거부한다(`"2001:db8:64::/96" lies inside 2001:db8::/32 (documentation, RFC 3849), ...`). |
| 문서 | `docs/environment-variables.md`, `docs/llama-server-compat.md` | `MLXCEL_MEDIA_NAT64_PREFIXES` 행과 주소 정책 절에 새 거부 규칙과 동작 변경을 적었다. compat 문서에는 DNS 상한(전역 32, 도메인당 4, 2초 더하기 5초)과 multi-label public suffix 한계를 설명하는 문단이 추가되었고, pin과 hop scheme 비공개도 언급한다. |
| 테스트 | 테스트 파일 4개 | 새 테스트 11개(4.1절). `system_resolve`의 `cfg(test)` `test_answers` hook이 정확한 host별로 준비된 응답을 돌려주고 조회 횟수를 센다. 기존 NAT64 테스트 11개는 `media_net_tests.rs`(573줄)에서 새 `media_net_nat64_tests.rs`로 옮겼고, RFC 6052 decode 예시는 이제 `parse_unchecked_for_test`로 `2001:db8::` prefix를 만든다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 11 (새 `media_net_pin.rs` 포함 소스 5, 새 `media_net_nat64_tests.rs` 포함 테스트 4, 문서 2) |
| 추가 / 삭제 라인 | main 대비 +1087 / -263 (삭제 중 192줄과 `media_net_nat64_tests.rs`의 대부분은 옮긴 NAT64 테스트) |
| 추가된 테스트 | 11 (`media_net_resolver_tests.rs` 9, `media_net_nat64_tests.rs` 1, `media_url_disclosure_tests.rs` 1) |
| 새 의존성 | 없음 |
| 커밋 | fdc89ca2 (항목 1-5), 72d5fc54 (리뷰에서 나온 amortized pruning과 redirect variant 단위 테스트), a955fe97 (main merge) |

## 3. 기술적 선택과 그 이유

### 3.1 공정성 키를 클라이언트나 public suffix가 아닌 도메인으로

| 선택지 | 장점 | 단점 |
|---|---|---|
| 클라이언트별 공정성 | 실제 남용자를 겨냥 | fetch 계층(`fetch_labeled_image`, chat, embeddings, rerank, image route가 사용)에 클라이언트 식별자가 없고, 이를 전달하려면 더 큰 변경이 필요 |
| Public suffix list 키 | 정확한 등록 도메인(`example.co.uk`) | 어차피 근사인 상한을 위해 의존성과 목록 관리가 추가됨 |
| **선택: host의 마지막 두 label** | 의존성 없음, 한 도메인의 임의 서브도메인이 키 하나로 모임 | multi-label public suffix 아래 host들이 suffix의 키(`co.uk`)를 공유 |

이 상한은 공격을 없애는 것이 아니라 비용을 바꾼다. 전역 permit 32개를 모두 잡으려면 이제 nameserver가 응답하지 않는 등록 도메인이 하나가 아니라 8개 필요하다. 알려진 한계는 `media_net_dns.rs` 모듈 문서와 `docs/llama-server-compat.md`에 적혀 있다.

도메인 permit을 전역 permit보다 먼저 잡는다. 따라서 전역 permit을 기다리는 조회는 이미 자기 도메인에 집계되고, 한 도메인이 전역 풀 앞에 4개 넘게 줄을 세울 수 없다. 두 permit 모두 blocking closure로 옮겨지므로, 버려진 조회도 `getaddrinfo`가 반환할 때까지 집계된다는 #2741의 성질이 유지된다.

### 3.2 quadratic 경로 없이 도메인 테이블 정리하기

semaphore들은 std `Mutex` 아래의 `HashMap<String, Arc<Semaphore>>`에 있다. `Arc::strong_count == 1`이고 permit이 모두 남아 있으면 idle이다. 대기 중인 `acquire_owned`가 `Arc` clone을 쥐고 있으므로 대기자가 있는 키는 지워지지 않는다. 첫 버전은 테이블이 1024 키에 도달할 때마다 정리했다. 리뷰의 보안 점검은 1024개가 넘는 키를 바쁜 상태로 유지하는 호출자(전역 풀이 포화된 동안 각 키를 최대 2초 permit 대기 동안 유지)가 새 키마다 Tokio worker 위에서 mutex를 쥔 채 전체 `retain`을 돌게 만든다는 점을 찾았다. 6000번 삽입에 정리가 6476번 일어났다. 72d5fc54는 정리 후마다 임계값을 `max(1024, 2 * 남은 len)`으로 바꿔 scan을 크기가 두 배가 될 때마다 한 번으로 줄였고(같은 7500개 바쁜 키에서 최대 8번), 급증이 끝나면 idle 키는 여전히 정리된다.

### 3.3 permit 대기와 resolver의 deadline 분리

deadline 하나가 가장 단순한 상한이지만, 서버가 바쁠 때 정확히 정상 조회를 불리하게 만든다. 분리하면 최악 지연이 2초 늘어나는(5초에서 7초) 대신, resolver가 실제로 실행되면 5초를 온전히 받는다. 세 숫자를 `DnsLimits`로, 풀을 `&DnsPool`로 전달하므로 테스트는 같은 코드를 300ms, 500ms 상한과 전용 풀로 실행하고 프로세스 전체 도메인 map을 공유하지 않는다.

### 3.4 DNS 캐시 대신 pre-flight 응답 pin

pin은 일반 DNS 캐시가 아니다. `ensure_media_url_allowed`만 쓰고, 모든 주소가 통과한 뒤에만 쓰며, pre-flight는 pin을 읽지 않는다. 따라서 모든 요청은 여전히 자기 URL과 각 hop을 새로 조회한다. connect 시점의 `PolicyResolver`가 pin을 읽고 다시 필터링하므로, pin이 도달 가능한 범위를 넓힐 수 없다. 결과적으로 hop당 조회가 한 번(permit 두 개가 아니라 하나)이 되고, 연결은 검사를 거친 주소 집합으로 간다. rebinding 이름에서는 거부될 두 번째 응답 때문에 실패하는 대신 승인된 응답을 쓴다. `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`이면 pre-flight가 조회하지 않으므로 pin할 것이 없고 경로는 그대로다.

TTL 30초와 1024 host로 저장소 크기를 제한한다. pin 키는 소문자 host이고 port는 0이어서, connector가 새 조회 결과에 하는 처리와 같다.

### 3.5 hop에서만 비공개, 클라이언트 자신의 메시지는 유지

`Scheme`과 `Malformed` 자체를 바꾸지 않고, hop 거부가 만들어지는 유일한 곳인 `check_hop`에서 `for_redirect_target`을 적용한다. `ftp://...`를 보낸 클라이언트는 자신이 보낸 scheme과 거부 이유를 여전히 알 수 있고, origin이 정한 텍스트만 debug 로그로 간다. 두 variant 모두 `media_url_rejected`를 거쳐 같은 400으로 렌더링되고, `MediaUrlError`를 exhaustive하게 match하는 코드가 없으므로 variant 추가가 호출부에 영향을 주지 않았다.

### 3.6 전체 IANA 레지스트리가 아닌 짧은 블록 표

네 블록은 주소 정책이 `2000::/3` 안에서 이미 거부하는 블록이다. 허용되는 RFC 6052 길이는 모두 /32 이상이고 나열된 블록은 모두 /32 이하이므로, prefix가 블록과 겹치는 것은 정확히 그 network 주소가 블록 안에 있을 때다. 블록마다 mask 비교 한 번이면 된다. `2001::/23` 안의 globally reachable 예외(AMT, AS112-v6, ORCHIDv2)도 translator 공간이 아니므로, 설정에서 /23 전체를 거부해도 실제로 잃는 것은 없다.

## 4. 검증

### 4.1 Failing-first 증거

새 acceptance 테스트를 변경 없는 main 소스(5c8ef80a)에서 실행했다. DNS 테스트는 main과 호환되는 형태(`lookup_bounded_with(.., Duration, Arc<Semaphore>)`, 같은 시나리오와 숫자)로, connect 테스트는 테스트 전용 `system_resolve` 카운터를 추가해 실행했다.

| 테스트 | main에서의 결과 |
|---|---|
| `one_domain_cannot_fill_the_dns_pool` (`a{i}.slow.test` 조회 40개, 2초 sleep하는 resolver, 전역 8, 도메인당 4) | `slow.test`가 풀을 잡아 `ok.other.test`가 500ms 타임아웃 후 `Unresolvable`로 실패 |
| `permit_wait_does_not_eat_resolver_time` (전역 1, 대기 300ms, 타임아웃 500ms, 첫 조회가 200ms 점유, 둘째 조회는 400ms에 해석) | 둘째 조회가 `Unresolvable`로 실패 |
| `connect_reuses_the_preflight_answer` (hostname origin이 hostname hop으로 한 번 redirect) | 조회 횟수 `(origin, hop)`가 기대값 `(1, 1)` 대신 `(2, 2)` |
| `redirect_to_a_non_http_scheme_omits_the_scheme` (`Location: secret-token-123://x/y`) | 클라이언트 텍스트가 `image URL rejected: media URL scheme "secret-token-123" is not supported; only http and https are fetched` |
| `special_purpose_nat64_prefixes_are_refused` | `2001:db8:64::/96`이 파싱됨 |
| `pruning_is_amortized_while_more_than_the_idle_cap_of_keys_stay_busy` (바쁜 키 1500 + 6000) | 이 PR 첫 버전에서 정리 6476번, 요구치는 최대 8번 |

#2741의 `dns_lookups_are_bounded_by_the_permit_pool`은 main에서 통과했고 `DnsLimits`에 맞춰 고친 뒤에도 통과한다. 나머지 새 테스트(`the_domain_key_is_the_last_two_labels`, `idle_domain_keys_are_pruned_and_busy_ones_kept`, `idle_keys_are_still_pruned_after_a_busy_surge_drains`, `a_redirect_hop_refusal_never_carries_the_origin_chosen_text`, `pins_expire_and_the_oldest_is_evicted_past_capacity`)는 새 helper를 직접 검사한다. redirect scheme 테스트는 origin이 loopback에 있으므로 private URL opt-in을 켠 채 실행한다. scheme 검사는 opt-in을 확인하기 전에 `ensure_media_url_shape_with`에서 실행되므로 거부 결과는 opt-in과 무관하다.

### 4.2 로컬 테스트

GB10에서 `--profile test-fast --features cuda`로 `--test-threads=1` 실행 시 `server::media` 138개 통과. resolver와 disclosure 테스트는 직렬로 5번, `--test-threads=8`로 3번 더 통과했다. `cargo clippy --lib --tests -D warnings`, `cargo fmt --check`, `make verify-env-var-docs`, `make verify-llama-compat`도 깨끗했다.

### 4.3 실제 서버

`gpu-lock` 아래 127.0.0.1:20600에서 `mlxcel-server -m llama-nemotron-embed-vl-1b-v2`:

- `MLXCEL_MEDIA_NAT64_PREFIXES=2001:db8:64::/96`이면 `invalid MLXCEL_MEDIA_NAT64_PREFIXES entry: "2001:db8:64::/96" lies inside 2001:db8::/32 (documentation, RFC 3849), ...`로 시작에 실패하고, `64:ff9b:1::/96`이면 시작한다.
- origin이 `302 Location: secret-token-123://x/y`로 응답하는 `/v1/embeddings` image URL은 HTTP 400 `input[0]: failed to read image: image URL rejected: media URL redirect target uses an unsupported scheme; only http and https are followed`를 반환한다.

### 4.4 호스팅 CI

작성 시점에 `license/cla`를 제외한 호스팅 check는 모두 대기 중이었다.

## 5. 남은 공백과 위험

- **Multi-label public suffix는 키를 공유한다.** `co.uk`, `com.au` 같은 suffix 아래 host는 모두 slot 4개를 공유하므로, 그런 suffix 위의 공격자는 같은 suffix의 다른 host를 기다리게 만들 수 있다. 문서화되었고, 문제가 되면 public suffix list 키가 해법이다.
- **풀은 여전히 채울 수 있다.** nameserver가 응답하지 않는 저렴한 도메인 8개면 permit 32개를 한 번에 최대 7초 동안 잡을 수 있다. 클라이언트별 공정성에는 fetch 계층의 클라이언트 식별자가 필요하고, 타임아웃된 이름의 negative cache는 범위 밖이었다.
- **지연.** 조회 한 번의 최악 시간은 이제 7초다. redirect hop은 여전히 체인 공유 10초 deadline 안에 있으므로, 느린 hop 조회는 거기서 먼저 잘린다.
- **pin 저장소 eviction scan.** 저장소에 살아 있는 pin이 1024개이면 새 host마다 mutex를 쥔 채 최대 1024개 항목에 `retain`과 `min_by_key`를 실행한다. 상수 상한이므로 3.2절에서 고친 증가형 문제는 아니지만, 삽입마다 발생한다.
- **pin은 DNS TTL보다 최대 30초 더 살 수 있다.** pre-flight는 항상 새로 조회해 pin을 덮어쓰므로, 이는 요청의 pre-flight와 connect 사이, 또는 같은 host에 대한 동시 요청이 교차할 때만 의미가 있다. pin된 응답은 다시 필터링되므로 영향은 승인된 주소 중 어느 것을 쓰는지에 그치고, 허용 범위에는 미치지 않는다.
- **`RedirectMalformed`는 단위 테스트로만 검증된다.** `check_hop`에 도달한 hop은 이미 파싱되었으므로, 현재 이 variant를 만드는 end-to-end 경로는 없다.
- **동작 변경.** `2001::/23`, `2001:db8::/32`, `2002::/16`, `3fff::/20` 안의 translator prefix를 등록한 배포는 이제 시작에 실패하며, translator를 다른 global 공간이나 `64:ff9b:1::/48`로 다시 번호 매겨야 한다. 문서용 prefix가 운영 트래픽을 나를 일은 없어야 하므로 영향은 테스트 환경에 한정될 것으로 본다.

## 6. 학습 포인트

### 6.1 전역 상한 아래에는 공정성 키가 필요하다

semaphore 하나는 자원을 제한하지만 tenant 하나가 전부를 가져갈 수 있게 둔다. 먼저 잡고 전역 permit과 함께 유지하는 더 작은 키별 semaphore를 더하면 "호출자 하나가 모두를 굶긴다"가 "호출자가 모두를 굶기려면 독립된 키 N개가 필요하다"로 바뀐다. 키를 무엇으로 고르느냐가 N의 비용을 정한다. 여기서는 서브도메인이 아니라 등록 도메인이다.

### 6.2 공격자가 바쁘게 유지할 수 있는 map의 정리는 amortize해야 한다

"map이 K를 넘으면 정리"는 공격자가 K개 넘는 항목을 살려 둘 수 있으면 quadratic이 된다. 삽입마다 아무것도 지우지 못하는 전체 scan이 일어나기 때문이다. 정리 후마다 임계값을 남은 크기의 배수로 옮기면 scan 횟수가 키 수의 로그로 제한된다. 늘어나는 vector가 쓰는 것과 같은 기법이다.

```rust
if table.map.len() >= table.prune_at {
    table.map.retain(|_, s| Arc::strong_count(s) > 1 || s.available_permits() < per_domain);
    table.prune_at = MAX_IDLE_DOMAIN_KEYS.max(2 * table.map.len());
}
```

### 6.3 두 번 검사하지 말고 검사한 응답을 재사용한다

검사와 그 검사가 보호하는 동작이 각각 이름을 조회하면 둘이 다른 답을 얻을 수 있고, 조회마다 제한된 자원을 쓴다. 검사한 응답을 동작에 넘기면(심층 방어로 다시 필터링하며) 검사하지 않은 데이터를 내주는 캐시를 추가하지 않고도 비용과 불일치를 함께 없앤다.

### 6.4 상류가 정하는 필드는 모두 반사 경로다

#2741은 오류에서 hop URL을 지웠고, 이 PR은 scheme이 같은 출처를 가진 별개 필드라는 것을 찾았다. 원격 상대가 정한 데이터로 오류를 만들 때는 눈에 띄는 URL만이 아니라 variant의 필드를 하나씩 점검해야 한다.

## 7. 참고 자료

- 이슈 #2764: Fair DNS budget, hop scheme redaction, NAT64 registry
- PR #2741과 이슈 #2728: 이 PR이 확장하는 DNS permit 풀, hop URL 비공개, NAT64 범위 규칙. `TECHNICAL_REPORTS/2741-media-fetch-errors-dns-nat64-20261030.en.md` 5절에 잔여 문제가 나열되어 있음
- 이슈 #2644: connect 시점 주소 검사 `PolicyResolver`
- 이슈 #2693: redirect hop 검사와 NAT64 prefix decoding
- RFC 2928 (IETF protocol assignments), RFC 3056 (6to4), RFC 3849와 RFC 9637 (IPv6 문서용 prefix), RFC 6052 (IPv4-embedded IPv6 주소), RFC 8215 (local-use translation prefix)
- `docs/environment-variables.md` (`MLXCEL_MEDIA_NAT64_PREFIXES`), `docs/llama-server-compat.md` (원격 미디어 URL 주소 정책)
