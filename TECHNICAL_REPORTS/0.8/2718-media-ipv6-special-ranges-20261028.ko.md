# PR #2718: media URL 정책에서 특수 용도 IPv6 대역 거부

**날짜**: 2026-10-28 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (SSRF allow-list를 좁히는 변경이므로, 이전에 허용되던 주소가 이제 거부될 수 있음)

## 요약

원격 media URL 정책(#1451, #2644, #2649, #2693)은 `image_url`, `audio_url`, video URL과 모든 redirect hop이 어떤 resolve 결과 주소나 literal 주소에 도달할 수 있는지 판정합니다. 이 정책의 IPv6 분기는 unique-local, link-local, `2001:db8::/32`, 그리고 거부 대상 IPv4 주소의 변환 표기만 거부했습니다. 그래서 site-local `fec0::/10`, IPv4-translated `::ffff:0:0:0/96`, discard-only `100::/64`, dummy prefix `100:0:0:1::/64`, SRv6 SID `5f00::/16`, `2000::/3` 밖의 예약 공간, Teredo, benchmarking, 폐기된 ORCHID, 새 문서용 prefix `3fff::/20`이 모두 public으로 통과했습니다. #2702 보고서는 이를 기존부터 있던 공백으로 기록해 두었습니다. 이 PR(이슈 #2713)은 목록이 아니라 구조적 규칙 하나로 이 공백을 닫습니다. IPv4-mapped decode, 등록된 NAT64 prefix, 잘 알려진 `64:ff9b::/96` 판정이 끝난 뒤에는 `2000::/3` 밖의 모든 IPv6 주소를 거부합니다. `2000::/3` 안에서는 `2001::/23` IETF protocol-assignment 블록을 거부하되 IANA special-purpose registry가 globally reachable로 표시한 항목은 제외하고, 문서용 prefix 두 개도 거부합니다. 주소 판정 함수는 별도 파일 `src/server/media_net_address.rs`로 옮겼습니다.

또 #2702 리뷰의 LOW 항목 두 개를 닫습니다. `{kind} URL rejected:` 문자열은 이제 helper 하나, `media_url_rejected`가 사전 검사, redirect hop, 연결 후 peer 검사의 거부를 모두 만듭니다. redirect hop의 정책 lookup이 체인 deadline을 넘기면 이제 `MediaSendError::DeadlineExceeded`가 되고, client에는 일반적인 host 거부 문구 대신 host를 밝히지 않는 timeout으로 전달됩니다.

## 1. 문제 정의

### public으로 통과하던 IPv6 블록

main(752db4f9)의 `is_public_address_with`는 IPv6를 다음 순서로 처리했습니다. loopback, unspecified, multicast 거부; IPv4-mapped는 IPv4로 판정; 나머지 `::/96` 거부; 운영자가 등록한 NAT64 prefix는 내장 IPv4로 판정; `64:ff9b::/96` decode; `64:ff9b:1::/48` 거부; 6to4 decode; 마지막으로 `fc00::/7`, `fe80::/10`, `2001:db8::/32`만 거부. 나머지 IPv6 주소는 모두 public이었습니다.

빠진 블록 중 globally reachable한 것은 없지만, 일부 host에서는 각각 도달 가능합니다. 아직 `fec0::/10`을 routing하는 사내망, 로컬 Teredo interface가 있는 host, SID가 로컬 기능을 가리키는 SRv6 배포가 그렇습니다. 이런 host에서는 client가 보낸 media URL이나 redirect `Location`이 이 주소들을 가리키면 서버가 실제로 가져왔습니다. 752db4f9에서 먼저 실패하도록 돌린 결과가 범위를 보여 줍니다. 새 table test의 거부 대상 17개 행이 모두 public으로 통과했습니다.

### fetch 경로의 두 가지 렌더링 문제

- `fetch_remote_bytes_with_limit_cancel`은 거부 문구를 지역 closure에서 만들었고, `media_send_failure`는 같은 `"{kind} URL rejected: {refusal}"`과 debug log 호출을 한 번 더 만들었습니다. #2644의 정보 노출 규칙은 사전 검사 거부와 hop 거부가 똑같이 렌더링되는 것에 기대는데, format 문자열이 두 벌이면 어긋날 수 있습니다.
- `check_hop`은 hop lookup 도중 체인 deadline이 끝나면 `Refused(Unresolvable)`로 매핑했습니다. client는 `MEDIA_HOST_REFUSED`를 받았고, 운영자 log는 timeout을 "refused"로 기록했습니다.

## 2. 기술적 선택과 그 이유

### 열거 대신 global-unicast 관문

새 규칙은 한 줄, `if segments[0] & 0xe000 != 0x2000 { return false; }`입니다. IANA가 global unicast로 할당한 것은 `2000::/3`뿐이고(RFC 4291, RFC 3587), 이 지점에 도달한 그 밖의 주소는 로컬, 특수 용도, 예약 공간 중 하나입니다. 이슈는 `2000::/3` 밖의 블록을 하나씩 나열하는 방식을 기각했습니다. registry는 예약 공간을 잘라 계속 늘어나기 때문입니다. `5f00::/16`(RFC 9602, 2024)과 `100:0:0:1::/64`(RFC 9780, 2025)가 모두 예약 공간에서 나왔고, 목록은 다음 블록을 놓칩니다. 관문은 앞으로 나올 블록도 기본으로 거부합니다. 기존 `unique_local`, `link_local` 조건은 중복이 되어 삭제했습니다.

### 관문의 위치

관문은 `2000::/3` 밖에서 내장 IPv4를 decode하는 모든 규칙 뒤, 6to4 앞에서 실행됩니다.

1. IPv4-mapped decode(`::ffff:0:0/96`)와 `::/96` 거부가 먼저 실행되므로 `::ffff:127.0.0.1`은 여전히 `127.0.0.1`로 판정됩니다. IPv4-translated 블록 `::ffff:0:0:0/96`은 mapped 표기와 segment 5가 달라 decode되지 않고 관문에서 거부됩니다.
2. 운영자가 등록한 NAT64 prefix가 여전히 우선합니다. `MLXCEL_MEDIA_NAT64_PREFIXES`에 등록한 `64:ff9b:1::/96`은 `2000::/3` 밖에 있고, `a_listed_local_use_prefix_is_judged_by_its_embedded_address`는 수정 없이 통과합니다.
3. 잘 알려진 `64:ff9b::/96`은 관문보다 먼저 decode됩니다. 이전에는 public이던 `64:ff9b::/32`의 나머지는 이제 관문에서 거부됩니다. 보안 검토가 이 순서를 확인했습니다.
4. 6to4(`2002::/16`)는 `2000::/3` 안에 있으므로 decode 위치가 관문 아래로 내려가도 동작이 바뀌지 않습니다.

관문 위의 명시적 `64:ff9b:1::/48` 규칙은 이제 관문에 포함됩니다(`0x0064`는 `2000::/3` 밖). 등록된 prefix 없이는 이 블록을 판정할 수 없는 이유를 주석이 설명하므로 그대로 두었습니다.

### `2000::/3` 안에서는 registry를 그대로 반영

`ietf_protocol_assignment_not_global(segments)`는 `2001::/23`(`segments[0] == 0x2001 && segments[1] < 0x0200`)에서 true를 반환하되, registry의 globally reachable 항목은 제외합니다. anycast 주소 `2001:1::1`, `2001:1::2`, `2001:1::3`(RFC 7723, RFC 8155, RFC 9665), AMT `2001:3::/32`(RFC 7450), AS112-v6 `2001:4:112::/48`(RFC 7535), ORCHIDv2 `2001:20::/28`(RFC 7343), DETs `2001:30::/28`(RFC 9374)입니다. nightly의 `Ipv6Addr::is_global`과 같은 방식인데, stable에서는 쓸 수 없으므로 IPv4 분기와 마찬가지로 직접 구현했습니다. 최종 식은 이 helper의 결과, `2001:db8::/32`, `3fff::/20`(`segments[0] == 0x3fff && segments[1] & 0xf000 == 0`)을 거부합니다.

테스트는 각 예외의 양쪽 경계를 고정합니다. `2001:4::1`, `2001:4:113::1`, `2001:40::1`은 거부되어야 하고, `2001:200::1`(`/23` 바로 다음 주소)과 `3fff:1000::1`(`/20` 바로 다음 주소)은 public이어야 합니다. 경계 행은 구현 리뷰에서 나와 70d3829e에 추가되었고, 이후 수정이 예외 mask를 넓히면 테스트가 실패합니다.

### Teredo는 통째로 거부

Teredo `2001::/32`는 bit 32..64에 서버 IPv4를, 마지막 32 bit에 XOR로 가린 client IPv4를 담습니다. 6to4처럼 이를 decode하면 내장 주소 두 개를 판정해서, 어떤 media origin도 보내지 않는 트래픽을 허용하게 됩니다. registry도 이 블록을 globally reachable로 표시하지 않습니다. Teredo는 별도 처리 없이 `/23` 규칙에 걸립니다.

### 판정 함수를 `media_net_address.rs`로 이동

`media_net.rs`는 이 변경으로 판정 함수가 더 커지기 전에 이미 533줄이었고, 판정 함수는 등록된 NAT64 목록을 읽는 것 말고는 순수 함수라 주변의 resolver, URL 형태 검사, 환경 변수 코드와 깔끔하게 분리됩니다. `is_public_address`, `is_public_address_with`, `embedded_ipv4`, 새 helper가 `src/server/media_net_address.rs`(185줄)로 옮겨졌고, `media_net_nat64.rs`, `media_net_redirect.rs`와 같은 `#[path]` 방식으로 mount됩니다. `media_net.rs`는 이제 425줄입니다. production용으로는 `is_public_address`만 re-export하고, `is_public_address_with`와 `Nat64Prefix`는 submodule 밖의 production 호출자가 없으므로 `#[cfg(test)]` 아래에서 re-export합니다.

### 거부 문구는 한 곳에서

`media_url_rejected(kind, url, err)`는 `err.log_detail(url)`(host와 주소를 `debug`로 기록)을 호출하고 `anyhow!("{kind} URL rejected: {err}")`를 반환합니다. 사전 검사 closure와 `media_send_failure`의 `Refused` 분기가 모두 이를 호출하고, peer 검사도 같은 closure를 거칩니다. 동작 변화 없는 refactor이며, `a_hop_refusal_renders_like_the_preflight_refusal`이 `BlockedAddress`와 `Unresolvable`에 대해 두 경로가 같은 문구를 만드는지 확인합니다.

### hop lookup 중의 deadline 만료는 timeout

`check_hop`은 이제 `check_hop_with(host, deadline, check)`에 위임하고, 이 함수는 임의의 정책 future를 `timeout_at(deadline, ...)`으로 감쌉니다. `check`가 돌려준 결과는 `Unresolvable`을 포함해 `Refused`로 남고, deadline 만료는 `DeadlineExceeded { host }`가 됩니다. future를 인자로 받으므로 테스트에서 끼워 넣을 지점이 생깁니다. `std::future::pending()`으로 느린 resolver 없이 50 ms 안에 deadline 경로를 실행합니다.

client 문구는 `Failed to fetch {kind} URL {url}: the fetch deadline ran out while resolving a redirect target`입니다. client가 보낸 URL만 담고, hop host는 `debug` log에만 남습니다. #2644의 규칙은 거부된 host와 resolve되지 않는 host를 똑같이 렌더링해서 어떤 내부 이름이 존재하는지 응답으로 드러나지 않게 하는 것입니다. deadline은 hop의 이름이 아니라 체인에 속하므로 이 규칙을 깨지 않고, 같은 체인의 transport timeout은 이미 따로 보고되고 있었습니다. `fetch_remote_video`는 같은 debug 줄과 `warn`을 남기고 다른 오류 분기처럼 `None`을 반환합니다.

## 3. 변경 요약

- `src/server/media_net_address.rs`(신규, 185줄): global-unicast 관문, `ietf_protocol_assignment_not_global`, 문서용 prefix를 포함한 주소 판정 함수와 블록별 RFC 인용.
- `src/server/media_net.rs`(+7/-115): 새 submodule mount와 re-export 조정(`check_hop_with`를 `#[cfg(test)]` 아래에 추가).
- `src/server/media_net_redirect.rs`: `MediaSendError::DeadlineExceeded { host }`; `check_hop`을 `check_hop`과 `check_hop_with`로 분리.
- `src/server/media.rs`: `media_url_rejected`, `REDIRECT_DEADLINE_EXCEEDED`, `log_redirect_deadline`; `media_send_failure`와 `fetch_remote_video`가 새 variant를 처리.
- `docs/llama-server-compat.md`: 거부 목록에 IPv6 global unicast로 취급하는 것은 `2000::/3`뿐이라고 적고, 새로 거부하는 블록과 globally reachable 예외를 나열. 이 문서의 한국어판은 없습니다.
- 테스트: `media_net_tests.rs`의 `ipv6_special_purpose_ranges_are_refused`(20행, 실패한 행마다 블록과 RFC를 보고)와 `ipv6_globally_reachable_neighbours_stay_public`(14개 주소); `media_net_proxy_tests.rs`의 `check_hop_with` 테스트 두 개; `media_url_disclosure_tests.rs`의 렌더링 테스트 두 개.

## 4. 트레이드오프와 남은 문제

- **특이한 배포의 동작 변화.** site-local, Teredo 등 `2000::/3` 밖의 주소에서 media를 가져오던 배포는 이제 400을 받습니다. `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`이 이전 도달 범위를 되돌립니다(다른 모든 주소 검사도 꺼집니다). DNS64가 `2000::/3` 밖으로 합성하는 IPv6-only 배포는 이전처럼 그 prefix를 `MLXCEL_MEDIA_NAT64_PREFIXES`에 등록하면 됩니다.
- **`lookup_bounded` 자체의 timeout은 여전히 거부로 읽힙니다.** `ensure_media_url_allowed` 안의 lookup별 5초 제한은 첫 URL과 hop 모두에서 `Unresolvable`을 보고합니다. 체인 deadline 만료만 `DeadlineExceeded`가 됩니다. 이슈는 이것을 범위 밖으로 두었습니다.

5a4d137b에 대한 보안 검토는 CRITICAL, HIGH 항목 없이 LOW 세 개를 남겼습니다.

- **transport, status 오류 문구의 hop URL.** redirect 뒤의 `Transport` 또는 `error_for_status` 오류는 reqwest의 `Display`를 그대로 렌더링하고, 여기에 hop URL이 들어갑니다. 그 hop은 이미 정책을 통과했고 host는 origin의 `Location`에서 왔으므로 내부 이름을 드러내지는 않지만, `err.without_url()`을 쓰면 hop URL이 client 문구에 아예 나타나지 않습니다.
- **blocking resolver thread가 timeout 뒤에도 남습니다.** `lookup_host`는 Tokio blocking pool에서 `getaddrinfo`를 실행합니다. 5초 제한과 체인 deadline은 future를 drop할 뿐 thread를 멈추지 않으므로, 응답이 멈추는 DNS 이름을 담은 요청이 많으면 시스템 resolver가 포기할 때까지 blocking thread를 붙잡을 수 있습니다. 이슈는 여기서 `lookup_bounded`를 바꾸지 않기로 했습니다.
- **등록된 NAT64 prefix가 로컬 공간에 있을 수 있습니다.** 등록된 prefix는 설계상 관문보다 우선하므로, 운영자가 link-local이나 unique-local prefix를 등록하면 그 안에서 public IPv4를 내장한 주소에 도달할 수 있습니다. parse 단계에서 `fe80::/10`, `fc00::/7` prefix를 거부하면 이 설정 실수를 막을 수 있습니다.

## 5. 검증

**752db4f9에서 먼저 실패.** 새 `media_net_tests.rs` 테스트만 추가한 상태에서 `server::media_net::tests::ipv6_`는 1 passed, 1 failed였고, 이슈 표의 17개 행이 모두 public으로 통과했습니다. 예외 경계 3행은 나중(70d3829e)에 추가되어 이 실행에는 포함되지 않았습니다. 렌더링과 deadline 테스트는 752db4f9에 `media_url_rejected`, `check_hop_with`, `DeadlineExceeded`가 없어 컴파일되지 않으며, 그 커밋에서 deadline 경우는 `Refused(Unresolvable)`를 반환했습니다.

**테스트 묶음** (GB10, `--profile test-fast --features cuda`, `--test-threads=1`, `gpu-lock` 아래): `server::media` 121 passed, `server::routes::embeddings` 23, `server::routes::rerank` 20. 구현 리뷰와 보안 검토가 각각 `server::media`를 다시 실행했습니다(121 passed). `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`, `make verify-env-var-docs`, `make verify-llama-compat`은 깨끗했습니다. 70d3829e(테스트 행과 RFC 인용만 변경)의 커밋 메시지도 같은 121을 보고합니다.

**실제 서버** (`target/test-fast/mlxcel-server`, `colsmol-256m-merged`, `127.0.0.1:19651`, `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS` 미설정):

| `/v1/embeddings` 요청 | 결과 |
|---|---|
| `image_url` `http://[fec0::1]/x.png` | 400, `image URL rejected: media URL host is not allowed or could not be resolved`; debug log에 `address=fec0::1` |
| `tests/fixtures/test_image.png`의 `data:` URL | 200, shape (79, 128)인 embedding 하나 |

**검증하지 못한 것.** 테스트 환경의 어떤 host도 site-local, Teredo, SRv6 공간으로 가는 route가 없었으므로, 거부는 단위 테스트 표와 `fec0::1` literal로 보였을 뿐 실제 fetch가 막히는 것으로 보이지는 않았습니다. deadline 경로는 단위 테스트의 멈춘 future로 확인했고, 실제 redirect 뒤의 느린 resolver로는 확인하지 않았습니다. 기록된 테스트 묶음과 실제 서버 실행은 5a4d137b에서 했고, origin/main merge(844241d0) 이전입니다. 이 merge는 #2711(ROCm imagegen parity 테스트)을 가져왔으며 이 PR의 파일은 건드리지 않습니다. 이 보고서를 쓰는 동안 merge 커밋에 대한 merge 전 테스트 실행이 진행 중이었습니다.

## 6. 학습 포인트

- 주소 공간이 계속 잘려 나가는 allow-list에서는 global로 확정된 할당(`2000::/3`)을 관문으로 삼고 그 안의 예외만 나열합니다. global이 아닌 것을 나열한 목록은 IANA가 예약 공간에서 다음 블록을 떼어 낼 때 낡아 버립니다.
- 주소 판정 함수에서 규칙의 순서는 계약의 일부입니다. 내장 주소를 decode하는 규칙과 운영자 override는 모두, 그렇지 않으면 그 대역 전체를 거부할 관문보다 먼저 실행되어야 합니다. 그 순서를 고정하는 테스트(`2000::/3` 밖의 등록 NAT64 prefix, IPv4-mapped loopback)가 이후 refactor에서 순서가 바뀌는 것을 막습니다.
- 판정 함수가 registry의 예외 행을 반영할 때는 각 예외 mask의 양쪽을 모두 테스트합니다. 바로 바깥 주소가 거부되는지 확인하지 않으면 mask의 1 bit 실수가 예외를 조용히 넓힙니다.
- client 문구를 일반적으로 유지해야 하더라도 "거부"와 "timeout"은 별도 오류 variant로 둡니다. 그래야 운영자 log가 실제로 일어난 일을 말할 수 있고, 정책 future를 `check_hop_with`에 인자로 넘기면 느린 resolver 없이 timeout을 테스트할 수 있습니다.
