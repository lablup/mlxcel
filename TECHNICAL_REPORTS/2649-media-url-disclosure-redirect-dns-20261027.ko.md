# PR #2649: media URL 오류에서 내부 DNS 정보 노출 중단

**날짜**: 2026-10-27
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: 중간

## 요약

원격 media URL에 대한 SSRF 보호는 호출자에게 어떤 내부 이름이 존재하고 그 이름이 어디로 resolve되는지 알려 주고 있었습니다. 거부된 host는 `media URL host {host} resolves to {address} ... set MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1 ...`로, 존재하지 않는 host는 `media URL host {host} does not resolve`로 표시되었고, 두 문장 모두 네 개의 route에서 400 응답 본문으로 그대로 전달되었습니다. 또한 public origin이 fetch를 내부 hostname으로 redirect할 수 있었고, reqwest는 post-connect 검사가 응답을 버리기 전에 그 host로 실제 GET을 보냈습니다. 이제 두 거부 모두 `media URL host is not allowed or could not be resolved`(`MEDIA_HOST_REFUSED`) 한 문장으로 표시되고, 상세 정보는 `debug` 로그로 옮겨졌습니다. 공유 media client는 이제 모든 이름을 policy resolver로 resolve하므로, hostname redirect hop은 DNS 단계에서 거부되고 내부 host에는 어떤 요청도 도달하지 않습니다. 같은 작업에서 private IPv4 주소의 NAT64·6to4 표기를 거부하고, 모든 lookup을 5초로 제한했습니다 (이슈 #2644).

## 1. 문제

`src/server/media_net.rs`의 `MediaUrlError::BlockedAddress { host, address }`와 `MediaUrlError::Unresolvable { host }`는 공격자가 넣은 host, 내부에서 resolve한 주소, opt-in 환경 변수를 `Display` 텍스트에 담았습니다. 네 곳이 그 텍스트를 그대로 클라이언트에 돌려줍니다.

- `fetch_labeled_image`는 `POST /v1/embeddings`, rerank, 이미지 생성이 쓰며, 오류 체인 전체를 400 메시지에 넣습니다.
- 오디오 입력이 있는 chat은 `AudioAcquisitionError::Source`를 거쳐 같은 공유 fetch에 도달하고, chat route는 이를 `to_string()`으로 표시합니다.

chat 이미지와 video는 fetch 오류를 삼키므로 노출되지 않았습니다. 후보 이름마다 POST 한 번(`http://intranet-admin.internal/x.png`)이면 그 이름의 존재 여부와, 존재할 경우 내부 주소를 알 수 있었습니다. 이미지나 오디오를 받는 모델이 올라간 배포라면 어디서든 내부 DNS를 정찰할 수 있었고, 기본 no-auth 설정에서는 key도 필요 없었습니다.

두 번째 결함은 redirect policy에 있었습니다. reqwest의 redirect callback은 동기 함수라서 `media_redirect_policy`는 각 hop에 `ensure_media_url_shape`(scheme, credential, IP literal)만 다시 실행했습니다. `http://localhost:PORT/`로 가는 hop은 shape 검사를 통과했고, reqwest는 연결해서 GET을 보냈으며, 그 뒤에야 post-connect `remote_addr` 검사가 응답을 거부했습니다. 이어지는 400 응답은 내부 peer 주소를 노출했습니다. "public origin은 fetch를 내부로 돌릴 수 없다"는 module doc의 설명은 IP literal 대상에만 맞는 말이었습니다.

노출 문제에 대해서는 코드베이스에 선례가 있었습니다. 이슈 #1612는 설정된 `--media-path` 후보를 클라이언트용 텍스트에서 빼고 `debug`로 기록하게 했습니다(`src/server/media_root.rs`). 원격 경로는 이 규칙에서 벗어나 있었습니다.

## 2. 기술적 결정

### 하나의 일반 문장, 상세 정보는 debug로

`BlockedAddress`와 `Unresolvable`은 이제 둘 다 `#[error("{MEDIA_HOST_REFUSED}")]`로 표시됩니다. 두 variant는 `host`와 `address` 필드를 그대로 갖지만, 이를 읽는 곳은 `MediaUrlError::log_detail(url)`뿐입니다. 이 함수는 URL, host, 주소, opt-in 안내를 담은 `debug` 이벤트를 남깁니다. `media.rs`의 fetch 경계(`fetch_remote_bytes_with_limit_cancel`의 지역 closure `rejected`, 그리고 `fetch_remote_video`)에서 호출합니다. 수정이 오류 타입과 하나의 공유 fetch에 있으므로, 노출되던 네 route는 handler 변경 없이 모두 이를 물려받습니다.

두 경우를 하나로 합친 것은 의도적입니다. "resolve되지 않음"과 "거부됨"이 다르게 읽히면, host와 주소를 지워도 응답은 여전히 내부 이름의 존재 여부를 알려 줍니다.

다른 variant는 바뀌지 않았습니다. `Malformed { url }`은 호출자 자신의 입력을 되돌려 줄 뿐이고, `TooManyRedirects { url }`은 policy가 이미 public shape로 받아들인 hop을 가리킵니다.

### reqwest 오류 안에서 거부 찾기

redirect policy나 resolver가 낸 거부는 `send()`에 `reqwest::Error`로 도착하고, 그 source chain 안에 `MediaUrlError`가 들어 있습니다. 이 PR 전에는 이 경로가 일반 transport 분기로 가서 reqwest의 문구로 `Failed to fetch {kind} URL {url}: {err}`를 출력했습니다. pre-flight 거부와는 다른 메시지였고, `log_detail`에도 도달하지 않았습니다. `MediaUrlError::find_in`은 `source()`를 따라가며 각 단계를 downcast합니다. policy 거부를 찾으면 호출자는 상세 정보를 로그로 남기고 pre-flight 거부와 같은 `{kind} URL rejected: {MEDIA_HOST_REFUSED}` 텍스트를 돌려줍니다. 이것이 없으면 resolver 거부는 연결 실패로 보입니다. 클라이언트는 pre-flight와 다른 메시지를 보게 되고(그래서 "redirect에서 거부됨"을 다른 결과와 구별할 수 있고), 운영자는 거부된 host를 알려 주는 debug 줄을 받지 못합니다.

### redirect 차단: client에 붙인 DNS resolver

이슈는 두 가지 방법을 제시했습니다. 각 hop의 연결 뒤에 peer를 다시 검사하거나, authority가 바뀌는 redirect를 거부하는 것입니다. 이 PR은 더 강한 세 번째 방법을 택했습니다. `media_net::PolicyResolver`는 `reqwest::dns::Resolve`를 구현하고 `ClientBuilder::dns_resolver`로 공유 client에 설치됩니다. client가 연결하는 모든 이름에 대해(최초 요청이든 redirect hop이든) `resolve_allowed_addresses`로 resolve하고, `is_public_address`를 통과한 주소만 남기며(`filter_allowed_addresses`), 남는 주소가 없으면 `BlockedAddress`로, 응답이 비어 있으면 `Unresolvable`로 resolution을 실패시킵니다. 연결 자체가 열리지 않으므로 내부 host는 아무것도 받지 않습니다.

다른 방법을 택하지 않은 이유는 다음과 같습니다.

- hop별 post-connect 검사는 거부하기 전에 이미 내부 host에 연결해서 request line과 header를 보냅니다. 응답은 막아도 요청은 막지 못합니다.
- authority가 바뀌는 redirect 거부는 public URL이 CDN host로 redirect하는 흔한 정상 흐름을 깨뜨리고, hop 사이에 DNS가 바뀌는 같은 authority의 이름에는 아무 효과가 없습니다.

resolver는 첫 hop에서 pre-flight lookup과 실제 연결 사이의 time-of-check 간격도 닫습니다. client가 연결하는 주소가 방금 policy가 승인한 바로 그 주소이기 때문입니다.

`PolicyResolver`는 `private_addresses_allowed()`를 생성 시점에 고정하지 않고 resolution마다 읽습니다. client는 프로세스당 한 번 `OnceCell`로 만들어지고 테스트는 테스트마다 opt-in을 바꿉니다. 한 번만 읽으면 첫 fetch가 본 값이 고정됩니다.

public 주소와 private 주소가 섞인 응답은 두 DNS 검사가 다르게 다룹니다. 원래 host에 대한 pre-flight 검사는 주소 중 하나라도 private이면 이름 전체를 거부합니다. redirect hop도 보는 resolver는 응답을 public 주소로 걸러내고 그 주소로만 연결을 진행합니다. 어느 경로도 private 주소로 연결하지 않습니다.

### IP literal과 post-connect peer를 지키는 것

`http://169.254.169.254/` 같은 literal은 resolver를 거치지 않으므로, IP literal redirect 대상을 막는 것은 여전히 `media_redirect_policy`의 hop별 shape 검사뿐이며 그대로 유지됩니다. post-connect `remote_addr` 검사도 남아 있으며, 이제는 다층 방어이자 proxy 주소를 보는 검사입니다. 이 검사는 이제 원래 요청의 host 대신 `response.url().host_str()`(마지막 hop의 host)와 peer를 짝지으므로, redirect 뒤에도 debug 줄이 실제로 연결한 host를 가리킵니다. module doc과 `docs/llama-server-compat.md`는 기존의 세 검사 대신 네 검사(pre-flight, resolver, hop별 shape, post-connect)를 설명합니다.

### NAT64와 6to4

`is_public_address`는 IPv4-mapped와 IPv4-compatible IPv6 표기는 이미 풀어서 판단했지만, `64:ff9b::/96`(RFC 6052 NAT64 well-known prefix)과 `2002::/16`(RFC 3056 6to4)은 일반 global IPv6로 보았습니다. 클라우드의 IPv6 전용 subnet이 같은 VPC의 IPv4 자원에 닿기 위해 쓰는 NAT64 gateway 뒤에서는, `64:ff9b::a00:5`라는 AAAA 레코드가 `10.0.0.5`에 도달합니다. 이제 두 prefix 모두 내장된 IPv4 주소(NAT64는 96..128 bit, 6to4는 16..48 bit, `embedded_ipv4`로 추출)를 IPv4 규칙으로 판단합니다. `64:ff9b::808:808`, `2002:808:808::1` 같은 public 주소는 계속 허용됩니다. 이 문제는 #2644 이전부터 있었지만 같은 policy 영역이며, PR의 보안 리뷰에서 발견되었습니다.

### lookup 시간 제한

pre-flight lookup과 resolver는 `lookup_bounded`를 함께 씁니다. `tokio::net::lookup_host`를 5초(`DNS_LOOKUP_TIMEOUT`, client의 connect timeout과 같음) `tokio::time::timeout`으로 감싼 함수이며, 오류나 timeout은 `Unresolvable`이 됩니다. 이전에는 공격자가 고른 이름의 authoritative server가 느리면 시스템 resolver가 재시도하는 동안 요청이 묶여 있었습니다.

### proxy는 계속 사용

media client는 여전히 `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY`를 따릅니다. `MLXCEL_EXTRA_CA_CERTS`가 존재하는 이유인 기업 proxy 배포가 이에 의존합니다. reqwest 0.12.28(`socks` feature 없음)에서 client는 proxy 자신의 host만 resolve하고 연결하므로, resolver와 post-connect 검사는 proxy endpoint에 적용됩니다. private 주소의 proxy는 원래도 opt-in이 필요했고, 이는 바뀌지 않았습니다.

## 3. 변경 요약

- `src/server/media_net.rs`: `MEDIA_HOST_REFUSED`, `BlockedAddress`와 `Unresolvable`의 일반 `Display`, `MediaUrlError::log_detail`과 `MediaUrlError::find_in`, `PolicyResolver`, `resolve_allowed_addresses`, `filter_allowed_addresses`, `DNS_LOOKUP_TIMEOUT`을 쓰는 `lookup_bounded`, `is_public_address`의 NAT64·6to4 처리와 `embedded_ipv4`. module doc을 네 검사, 클라이언트가 받는 내용, proxy 주의 사항 중심으로 다시 썼고 redirect policy doc을 바로잡았습니다.
- `src/server/media_net_test_guard.rs`(신규): 테스트 전용 opt-in guard를 `media_net.rs`에서 옮겼습니다. 변경 후 `media_net.rs`는 506줄입니다.
- `src/server/media.rs`: 두 fetch 경계의 `log_detail`, `send()` 오류에 대한 `find_in`, 마지막 응답의 host를 쓰는 post-connect 검사, 공유 `http_image_client`의 `.dns_resolver(Arc::new(PolicyResolver))`. video 경로는 `warn` 줄을 유지하며, 이제 일반 텍스트만 출력합니다.
- `docs/llama-server-compat.md`: 원격 media divergence 항목이 네 검사, 일반 클라이언트 텍스트, proxy 동작, NAT64/6to4 표기를 설명합니다.
- 테스트(신규 파일): `media_url_disclosure_tests.rs`(labeled-image 텍스트, chat-audio 텍스트, 그리고 모든 연결을 기록하는 listener를 두고 실제 공유 client로 loopback origin에서 `localhost`로 redirect하는 경우), `media_net_resolver_tests.rs`(resolver 거부, 허용, resolution마다 opt-in 읽기, trailing-dot 이름, 섞인·전부 private·빈·opt-in 응답 집합에 대한 `filter_allowed_addresses`), `routes/embeddings_media_url_tests.rs`(이미지를 받는 stub과 실제 embeddings handler). `media_net_tests.rs`에 추가: 두 거부의 Display 동일성, post-connect 오류 텍스트, NAT64/6to4 거부, literal로 검사되는 숫자 IPv4 표기(`0x7f.1`, `2130706433`, `0177.0.0.1`, `127.1`).

## 4. 트레이드오프와 남은 문제

- **운영자는 400에서 안내를 잃습니다.** 응답은 더 이상 `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS`를 언급하지 않습니다. 내부 media가 로드되기를 기대했는데 일반 오류를 받은 운영자는 `RUST_LOG=mlxcel::server::media_net=debug`를 켜야 host, 주소, opt-in 안내를 볼 수 있습니다.
- **proxy 배포에서 hostname redirect는 proxy에 의존합니다.** proxy를 거치면 hostname redirect 대상은 `PolicyResolver`가 아니라 proxy가 resolve합니다. pre-flight 검사는 요청한 host를 여전히 로컬에서 resolve하고, hop별 shape 검사는 literal 대상을 여전히 거부하지만, 내부 이름으로 가는 hop을 거부하는 것은 proxy 자체의 egress policy에 달려 있습니다.
- **DNS 시간 차이.** private 주소로 resolve되는 이름과 NXDOMAIN을 반환하는 이름은 같은 텍스트를 내지만, 지연 시간까지 같지는 않을 수 있습니다. 응답 시간을 재는 호출자는 여전히 둘을 구별할 수 있습니다. 어느 쪽도 주소를 드러내지는 않습니다.
- **public host의 transport 오류.** public으로 resolve되지만 연결이나 TLS가 실패하는 이름은 여전히 `Failed to fetch {kind} URL {url}: {reqwest error}`를 반환합니다. 이 텍스트는 호출자가 고른 public 목적지를 설명할 뿐 내부 주소를 담지 않습니다.
- **well-known 이외의 NAT64 prefix.** local-use prefix `64:ff9b:1::/48`(RFC 8215)과 운영자가 정한 NAT64 prefix는 주소만으로 식별할 수 없어 계속 허용됩니다.
- **첫 host에 대한 두 번의 lookup.** pre-flight 검사와 resolver가 각각 원래 host를 resolve하므로, fetch마다 DNS 질의가 이전보다 하나 늘었습니다. 둘 다 5초로 제한됩니다.

## 5. 검증

**revert 증명** (PR 작성 세션의 scratchpad에 있는 `revert-evidence.txt`에 기록). 수정이 커밋되지 않은 patch 상태일 때, 노출·redirect 테스트 여섯 개(`server::routes::embeddings::tests::media_url::`, `server::media::tests::url_disclosure::`, `server::media_net::tests::a_blocked*`)가 `cargo test --profile test-fast --features cuda -p mlxcel --lib -- --test-threads=1`에서 통과했습니다. 테스트 파일은 그대로 두고 `media.rs`와 `media_net.rs`만 되돌리자(358줄 patch) 여섯 개 모두 실패했습니다. 대표적인 실패로, route 테스트는 `input[0]: failed to read image: image URL rejected: media URL host localhost resolves to 127.0.0.1, which is a loopback, ... set MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1 ...`를 받았고, redirect 테스트는 `redirect was followed to http://localhost:36649/x.png (status 200 OK)`로 panic했습니다. 원래 결함이 그대로 관찰된 것입니다. patch를 다시 적용하자 여섯 개 모두 다시 통과했습니다. PR 본문과 commit 2a9a062e에 따르면 NAT64 테스트도 `media_net.rs`를 되돌리면 실패합니다(`64:ff9b::7f00:1 must be refused`).

**실제 서버** (`real-server.txt`에 기록, 그리고 rebase 전 최종 head 73fbd1c9에서 한 번 더 실행). GB10에서 `colsmol-256m-merged`를 올린 `mlxcel-server`, `RUST_LOG=info,mlxcel::server::media_net=debug`, opt-in 미설정, 내부 서비스 역할을 하는 기록용 TCP listener.

| `/v1/embeddings`에 보낸 `image_url` | 결과 |
|---|---|
| `http://localhost:18391/x.png` | 400, 일반 텍스트 |
| `http://intranet-admin.internal/x.png` | 400, 같은 텍스트 |
| `http://10.0.0.7/x.png` | 400, 같은 텍스트 |
| `https://httpbin.org/redirect-to?url=http://localhost:18391/x.png` | 400, 같은 텍스트 |
| `http://[64:ff9b::7f00:1]:18393/x.png` (최종 head 실행) | 400, 같은 텍스트 |
| `tests/fixtures/test_image.png`의 `data:` URL | 200 |

기록용 listener에는 `LISTENING` 줄만 남았으므로, public redirect를 포함해 어떤 경우도 listener에 도달하지 않았습니다. 서버의 debug 로그는 각 경우의 거부된 host를 남겼고, httpbin redirect에 대해서는 `host="localhost" address=127.0.0.1`을 기록해 redirect hop의 resolution 단계에서 거부되었음을 보여 줍니다. `MLXCEL_ALLOW_PRIVATE_MEDIA_URLS=1`에서는 `http://localhost:18392/x.png`(resolver 경유)와 `http://127.0.0.1:18392/y.png`(literal) 모두 200을 반환했고 listener에 두 GET이 기록되었으므로, opt-in은 두 경로 모두에서 동작합니다.

**테스트 suite.** rebase 전 head에서 `server::media_net`, `server::media::`, `server::routes::embeddings`, `server::routes::rerank`(119개 통과), `cargo clippy --lib --tests -D warnings`와 `cargo fmt --check` 통과.

**rebase.** 이 보고서 작성 전에 branch를 main 11463e50 위로 rebase했습니다. `git range-diff`로 세 commit의 patch가 동일함을 확인했고, 새로 들어온 upstream commit(#2645, #2646, #2648)은 `src/server/`와 `Cargo.lock`을 건드리지 않습니다. 위의 실제 서버 실행과 revert 실행은 rebase 이전의 것이며, rebase 이후에 다시 실행하지 않았습니다.

**검증하지 않은 것.** HTTP proxy를 거치는 테스트나 실행은 없었으므로, proxy 동작은 reqwest 0.12.28의 `connect_via_proxy`를 읽은 결과에 근거합니다. 5초 lookup 제한은 느린 DNS 서버로 테스트하지 않았습니다. 실제 NAT64 gateway는 없었고, NAT64 검증은 unit test와 literal host를 쓴 실제 서버 사례뿐입니다.

## 6. 학습 포인트

- 오류에서 식별자를 지워도, 결과가 둘로 구별된다면 충분하지 않습니다. "거부됨"과 "resolve되지 않음"이 같게 읽혀야 응답이 존재 여부 oracle이 되지 않습니다.
- reqwest의 redirect policy는 동기 함수이고 resolution 전에 실행되므로 hostname에 대한 주소 판단을 할 수 없습니다. `ClientBuilder::dns_resolver`는 redirect hop을 포함해 client가 연결하는 모든 이름을 보는 지점이며, 여기서 거부하면 socket이 열리기 전에 요청이 멈춥니다.
- reqwest 안에서 발생한 policy 오류(resolver나 redirect policy)는 감싸진 채로 돌아옵니다. 클라이언트에 보낼 텍스트를 정하기 전에 `source()`를 따라가며 downcast해야, 감싼 오류의 transport 문구가 policy 오류가 숨기려던 결과 차이를 다시 드러내지 않습니다.
- IPv6 주소는 IPv4-mapped 말고도 여러 형태로 IPv4 목적지를 담을 수 있습니다. NAT64(`64:ff9b::/96`)와 6to4(`2002::/16`)는 모두 내장된 IPv4 주소로 라우팅되므로, 그 주소에 IPv4 규칙을 적용해야 합니다.
