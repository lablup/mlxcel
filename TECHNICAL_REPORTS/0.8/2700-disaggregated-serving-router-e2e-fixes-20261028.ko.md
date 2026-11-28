# PR #2700: disaggregated serving과 router e2e suite 통과시키기

**날짜**: 2026-10-28
**상태**: 검증 완료, PR은 draft로 열려 있음 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, Markdown
**위험도**: 중간 (모든 cluster transport가 쓰는 connection pool과 decode node의 stop token 집합을 건드림, wire와 설정 변경은 없음)

## 요약

Issue #2695는 `#[ignore]`로 CI 밖에 있는 real-process disaggregated test가 main에서 실패한다는 사실을 확인했다. `disaggregated_serving_e2e`는 1개 중 1개, `disaggregated_router_e2e`는 7개 중 4개가 실패했다. 원인은 서로 독립적인 네 가지였고, 하나는 test 코드, 셋은 product 코드에 있었다. 이 PR은 네 가지를 모두 고치고, 각각 main에서 실패하는 regression test를 붙인다. Test secret helper가 spawn된 role과 다른 bytes로 client key를 만들었고, connection pool이 peer가 종료된 idle socket을 재사용했고, router가 `usage.prompt_tokens_details`를 빠뜨렸고, decode node가 checkpoint의 `config.json` EOS를 무시했다. 또한 MiniCPM "byte_fallback" e2e test 두 개가 실제로는 byte-fallback token을 한 번도 만들지 않았다는 점을 찾아, 각 test에 `tokenizer.json`으로 전제를 검증하는 case를 추가했다.

두 e2e 파일 어디에서도 assertion을 약하게 만들지 않았다. 기존 assertion에 대한 유일한 변경은 진단용이다. `post_completion`이 이제 router의 error body를 돌려주므로 실패한 assertion에 그 내용이 찍힌다. Head `db334ac9`에서 모든 suite가 통과한다: `disaggregated_serving_e2e` 2/2, `disaggregated_router_e2e` 7/7 (soft-skip 없음), `disaggregated_e2e` 51/51, lib suite 1565/0.

## 1. 문제 정의

### 1.1 배경

Disaggregated serving 경로는 요청 하나를 여러 process에 나눠 처리한다. Router가 HTTP 요청을 받아 cluster TCP transport로 prefill node에 prompt를 보내고, prefill node는 KV state를 decode node에 넘기며, decode node가 나머지 생성 결과를 stream으로 돌려준다. 기준 계약은 single-node serving이므로, router 응답이 single-node 응답과 다르면 router 쪽 버그다. Real-process e2e test가 정확히 이것을 검사하지만 checkpoint와 GPU가 필요해 수동으로만 돈다. 최근 cluster 인증 작업 (#2650, #2674, #2684)이 그 아래의 transport를 바꿨는데, 이 suite는 그동안 실행되지 않았다.

### 1.2 기존 문제점

- **Handshake secret 불일치 (test 버그).** `TestClusterSecretFile::new`는 random 32 bytes를 만들어 그 64자 hex 인코딩을 secret 파일에 쓰고, `auth()`의 key는 원래 32 bytes로 만들었다. Spawn된 role은 `--cluster-secret-file`을 `ClusterSecret::from_file`로 읽는데, 이 함수는 파일 내용을 그대로 key로 쓴다 (문서화된 `openssl rand -hex 32` 계약). 그래서 in-test client와 node가 서로 다른 HMAC key를 유도했고, `disaggregated_two_process_handoff_matches_single_node`가 "secret mismatch"로 실패했다. Router test는 모든 참여자가 같은 파일을 읽는 spawn된 process라서 영향이 없었다.
- **Prefill node가 죽은 뒤 router가 500을 반환.** `ConnectionPool::acquire`는 `peer_addr()`가 성공하면 idle connection을 살아 있다고 판단했다. Peer process가 종료되어 socket이 CLOSE_WAIT에 있어도 이 호출은 성공하고, 그 socket에 대한 첫 write도 성공한다. Router는 `transport.send`가 `Err`를 돌려줄 때만 failover하므로, 장애 이후 첫 요청은 죽은 socket에 쓰여진 채 handoff timeout (기본 600초)까지 첫 token을 기다렸다. 수정 전 코드를 `MLXCEL_DECODE_TIMEOUT=30`으로 다시 돌리자 body가 `{"error":{"message":"timed out waiting for the prefill first token"}}`였고, 수정 전에 추정 원인이 맞다는 것을 확인했다. 수정 전 코드에서 이 test는 3번 중 2번 실패했으며, 장애 후 첫 요청을 어느 prefill node가 받느냐에 따라 결과가 갈렸다.
- **Router가 `usage.prompt_tokens_details`를 빠뜨림.** #1473 이후 single-node chat과 completions는 prompt cache가 켜져 있으면 (기본값) `prompt_tokens_details.cached_tokens`를 보고한다. Router의 usage builder 네 개 (chat과 completions, 각각 non-stream과 stream usage chunk)는 이 필드를 한 번도 내보내지 않았고, non-stream completion body는 `usage`에서만 달랐다.
- **Decode node가 설정된 EOS를 무시.** Llama format의 MiniCPM-2B는 `Llama3Model`로 load되는데, 이 model의 `eos_token_ids()`는 Llama 3 id인 `[128001, 128009]`를 반환하고 둘 다 이 checkpoint의 122753개 vocabulary 밖에 있다. 실제 EOS (`</s>` = 2)는 `config.json`에서 scheduler의 `config_eos`로만 들어온다. Single-node admission과 prefill handoff는 둘 다 `merge_config_stop_tokens`를 호출하지만 decode 쪽 `ingest_handoff_as_active`는 호출하지 않았고, router는 빈 `stop_token_ids`를 보낸다. 결과적으로 decode node는 `</s>`를 지나 budget이 끝날 때까지 생성했다. Router `completion_tokens`는 16, single node는 1이었다.
- **Byte fallback에 도달하지 않는 byte-fallback test.** MiniCPM "byte_fallback" test 두 개는 "café" prompt를 썼지만 `▁café`와 `▁Café` (69435)는 vocabulary에 있는 단일 piece다. Test 이름이 가리키는 `<0xXX>` byte piece의 token 수 경로는 end to end로 한 번도 실행되지 않았다.

### 1.3 위험성

| 고치지 않았을 때의 위험 | 영향 | 가능성 |
|------|--------|------------|
| Prefill node가 죽으면 그 node로 가는 다음 요청이 failover 대신 600초 대기 후 HTTP 500이 됨 | 높음 | 높음 (router가 idle pooled connection을 가진 상태에서 node가 종료되면 언제나) |
| EOS를 `config.json`만 선언하는 checkpoint가 decode node에서 `max_tokens`까지 생성해 텍스트가 길어지고 `completion_tokens` / `finish_reason`이 틀림 | 높음 | 중간 (Llama 3가 아닌 vocabulary를 가진 llama-format export) |
| Router `usage`가 single node와 달라 `prompt_tokens_details`를 읽는 client가 깨짐 | 낮음 | 높음 |
| 수동 e2e suite가 계속 실패 상태라 같은 영역의 이후 regression이 보이지 않음 | 중간 | 높음 |

## 2. 기술적 선택과 그 이유

### 2.1 `ClusterSecret::from_file`이 아니라 test helper를 고침

**컨텍스트:** Client와 node가 key에 합의하지 못했다. 어느 쪽을 바꿔도 맞출 수는 있었다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `from_file`이 파일을 hex decode하도록 변경 | Helper는 그대로 | 운영자 계약이 바뀜 (`openssl rand -hex 32 > cluster.secret`이 지금과 다른 key를 만듦), 배포된 모든 secret 파일이 깨짐 |
| **선택: helper가 자기가 쓴 bytes 그대로 key를 만듦** | Product 코드 변경 없음, helper가 구조적으로 올바름 | 없음 |

Helper는 파일에 쓰는 바로 그 문자열로 `ClusterSecret::from_bytes(hex.into_bytes())`를 만들고, #2692에서 계획한 router 인증 fixture가 재사용할 수 있도록 `secret()`를 노출한다. `to_hex` doc comment는 이제 hex 문자열을 child에 넘기면 새 secret을 넘기는 것이라고 명시한다. 원래 helper가 빠진 함정이 바로 이것이다. 후속 commit (`86e0a899`)은 unix에서 `create_new`와 mode 0600으로 파일을 만들어, spawn된 role이 `chmod 600` 경고를 더 이상 남기지 않게 한다.

### 2.2 Liveness 검사: 직접 호출하는 non-blocking `MSG_PEEK`

**컨텍스트:** Pool은 idle stream의 peer가 닫았는지 알아야 한다. Byte를 소비하면 안 되고 (#2684 이후 stream은 암묵적 sequence number를 쓰는 AEAD 봉인 frame을 나른다), pool mutex를 잡은 채로 block되면 안 된다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `peer_addr().is_ok()` 유지 | 비용 없음 | CLOSE_WAIT를 보지 못함, 이것이 버그 |
| Tokio를 통한 `TcpStream::peek(...).now_or_never()` (issue의 제안) | Reactor API 사용 | Runtime이 FIN을 관찰하기 전까지 `None`을 반환하므로 결과가 reactor timing에 좌우됨 |
| Health monitor 기반 eviction | Acquire마다 syscall 없음 | 3초 probe 간격만큼 공백이 생기고, 그 사이 요청은 여전히 유실됨 |
| **선택: `socket2::SockRef::from(&stream).peek(&mut [MaybeUninit<u8>; 1])`** | 이미 non-blocking인 fd에 `recv(MSG_PEEK)` 한 번, 결정적, 아무것도 소비하지 않고 codec과 tokio readiness를 건드리지 않음 | `acquire`마다 syscall 하나, 읽지 않은 byte 뒤의 FIN은 보지 못함 |

`ProtectedStream::peer_closed`는 `Ok(0)` (정상 종료)과 `WouldBlock` / `Interrupted`를 제외한 모든 error (예: reset)를 "닫힘"으로, `WouldBlock`이나 읽지 않은 byte가 있는 경우를 "살아 있음"으로 본다. `acquire`는 닫힌 idle stream을 버리고 (`active`에 집계된 적이 없음) 다음 것을 시도한 뒤 `connect_with_retry`로 넘어간다. Node가 없어졌다면 connect가 거부되고, `send`가 `Err`를 반환하고, `start_handoff`에 이미 있던 router failover가 동작한다. 이 항목 때문에 router 코드는 바뀌지 않았다.

기각한 `now_or_never` 경로는 다시 이 코드를 보는 사람에게 중요하다. 겉보기에 같고 대부분 통과하기 때문에, flaky failover test가 이런 식으로 되돌아올 수 있다.

### 2.3 Router 자신의 prompt cache 설정에 따른 `cached_tokens: 0`

**컨텍스트:** Single node는 prompt cache store가 있을 때만 `prompt_tokens_details`를 내보내고, `resolve_prompt_cache_store`는 `config.prompt_cache.is_enabled()`일 때 그 store를 만든다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `ResultFrame`에 cached token 수 추가 | Handoff prefill이 나중에 cache를 쓰게 되면 정확 | 오늘은 항상 0인 값을 위해 wire를 바꿈 |
| **선택: `state.config.prompt_cache.is_enabled()`이면 0을 보고, 아니면 생략** | Single node와 같은 조건, wire 변경 없음 | 읽는 설정이 worker node가 아니라 router process의 것 |

0은 placeholder가 아니라 정확한 값이다. Handoff prefill은 `already_cached_tokens: 0`으로 설정하고 prompt cache를 조회하지 않는다. Completion response와 usage chunk 생성은 `completion_response`와 `completion_usage_chunk`로 옮겼고, chat builder는 `cache_enabled`를 받는다. 그래서 네 가지 형태 모두 공통 규칙 `router_usage_json` 하나를 통해 unit test된다.

### 2.4 Decode intake에서 `config_eos`를 merge하고 `Llama3Model`은 그대로 둠

**컨텍스트:** Model은 이 checkpoint에 대해 틀린 EOS id를 보고하고, scheduler는 이미 올바른 id를 갖고 있다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `Llama3Model::eos_token_ids`가 `config.json`을 읽도록 변경 | Model 자체의 답을 고침 | 모든 llama-format model에 영향, e2e 수리 범위를 벗어남 |
| Router가 stop id를 전달 | 없음 (router는 checkpoint를 모름) | Router가 model 설정을 load해야 함 |
| **선택: `ingest_handoff_as_active`에서 `merge_config_stop_tokens(sampling, &self.config_eos)`** | Single-node admission과 prefill handoff가 이미 하는 호출과 동일, 한 줄 | Decode node가 prefill node와 같은 checkpoint를 load할 때만 정확함 (topology가 이미 그것을 요구함) |

Merge는 `merged_eos_token_ids`와 `RowSampler::new`보다 먼저 실행되고, `merge_config_stop_tokens`는 없는 id만 덧붙이므로 이미 EOS를 담은 요청은 바뀌지 않는다.

### 2.5 Vocabulary로 검증하는 byte-fallback case

"café" prompt를 교체하는 대신 (chat 쪽은 item 4의 regression이다), 각 test는 이제 `MiniCpmCase` 항목을 순회한다. `byte_fallback`로 표시된 case는 single-node 기준 출력에 `tokenizer.json`의 byte가 아닌 어떤 piece에도 없는 문자를 포함해야 한다. `minicpm_vocab_chars`가 그 문자 집합을 만들고 (`model.byte_fallback == true`도 확인), `assert_emits_byte_fallback`가 변경 없는 parity assertion보다 먼저 전제를 검사한다. 새 case는 completions의 " Шарик" (`Ш`, 6 tokens)과 chat의 "Чай" (`Ч`, 4 tokens)다. 이후 tokenizer나 model 변경으로 case가 byte fallback을 더 이상 실행하지 않게 되면, test는 조용히 통과하지 않고 그 사실을 말하는 메시지와 함께 실패한다.

## 3. Module별 구현

- `src/distributed/tcp_frame.rs`: 새 `ProtectedStream::peer_closed`.
- `src/distributed/connection_pool.rs`: `acquire`가 `peer_addr` 대신 `peer_closed`를 쓰고, stream을 버릴 때 debug log를 남김.
- `src/distributed/cluster_auth.rs`: `to_hex` doc만 변경.
- `src/server/router_front.rs`: `router_usage_json`, `completion_response`, `completion_usage_chunk` 추가, `chat_completion_json`과 `chat_chunk_usage`가 `cache_enabled`를 받음, 호출 지점 네 곳이 `state.config.prompt_cache.is_enabled()`를 전달.
- `src/server/batch/scheduler/handoff.rs`: `ingest_handoff_as_active`의 `merge_config_stop_tokens` 호출.
- `tests/common/mod.rs`: `TestClusterSecretFile`이 쓴 bytes로 key를 만들고, 파일을 0600으로 생성하고, `secret()`를 추가.
- `tests/disaggregated_router_e2e.rs`: MiniCPM test 두 개의 case별 loop, vocabulary 전제 검사, `post_completion`과 failover assertion 메시지 두 곳의 error body. 공백을 무시하면 파일 diff는 +196/-58이고, 나머지는 loop 안으로 들어가며 생긴 들여쓰기 변경이다.
- `src/server/batch/serving_handoff_parity_tests.rs`: `make_request_with`, `collect_text_and_done`, MiniCPM EOS test, 그리고 qwen3 형제 test가 `models/mlx/<name>` 구조의 host에서도 checkpoint를 찾도록 `repo_model_dir`에 fallback 추가.
- `docs/CONTINUOUS_BATCHING.md`: failover, `prompt_tokens_details`, 설정된 EOS 동작.

## 4. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 13 |
| 추가 라인 | +1057 |
| 삭제 라인 | -324 |
| 새 test | 7개 (integration 1, lib unit 5, `#[ignore]` real-model lib 1)와 새 e2e case 2개 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-------|---------|
| Product 수정 | 3 | Pool의 죽은 peer 감지, router `prompt_tokens_details`, decode 쪽 설정 EOS |
| Test 수정 | 2 | Secret helper key 생성 (그리고 0600 mode), 실제로 byte fallback에 도달하는 case |
| Test 진단 | 1 | `post_completion`이 error body를 반환 |
| 문서 | 2 | `docs/CONTINUOUS_BATCHING.md`, `to_hex` doc |

### 새 test

| Test | 지키는 것 |
|------|--------|
| `test_secret_file_auth_matches_server_resolution` | Helper key가 `ClusterSecret::from_file` key와 같음 (ignore 아님, CI에서 실행) |
| `acquire_discards_idle_connection_whose_peer_closed` | Peer 종료 후 pool이 낡은 stream 대신 `Err`를 반환 |
| `peer_closed_probe_is_non_consuming` | Probe가 읽지 않은 frame을 그대로 두고, 다 읽은 뒤 닫힌 peer는 닫힘으로 보고 |
| `send_to_peer_that_closed_its_pooled_connection_fails` | Transport 수준의 쌍둥이 test: `send`가 `Err`를 반환해 호출자가 failover |
| `router_usage_reports_zero_cached_tokens_when_prompt_cache_enabled` | Router usage builder 네 개 전부 |
| `router_usage_omits_prompt_tokens_details_when_prompt_cache_disabled` | 같은 네 개, cache 꺼짐 |
| `serving_handoff_parity_stops_on_config_eos_minicpm` | Decode continuation이 `</s>`에서 멈추고, prefill 수와 decode 수의 합이 single node와 같음 |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|---------|
| `af4723d0` | fix | fix(distributed): make disaggregated serving and router e2e pass |
| `86e0a899` | test | test: create the test cluster secret file owner-only |
| `db334ac9` | merge | Merge remote-tracking branch 'origin/main' (image generation port만 포함, 이 PR의 파일과 겹치지 않음) |

## 5. 호환성

- Wire format, CLI, 설정 변경 없음.
- Router의 prompt cache가 켜져 있으면 (기본값) router 응답에 `usage.prompt_tokens_details: {"cached_tokens": 0}`이 추가된다. 추가만 있는 변경이고 single node와 같다.
- Decode node가 이제 checkpoint에 설정된 EOS에서 멈춘다. Model의 기본 id에 이미 포함된 checkpoint는 바뀌는 것이 없다.
- `ProtectedStream::peer_closed`는 `socket2`를 쓴다 (`all` feature로 이미 의존성에 있음). `SockRef`는 cross-platform이지만 동작은 Linux에서만 실행해 봤다.

## 6. 검증

수정이 없는 tree에서의 실패 확인 (GB10, `--profile test-fast`, `MLXCEL_REQUIRE_MODELS=1`, `MLXCEL_SDPA_DETERMINISTIC=1`, `gpu-lock` 아래):

- Secret test: 32-byte 서명 두 개가 다름.
- Pool과 transport test: "acquire handed out an idle connection whose peer had closed it", "send to a peer that closed its pooled connection returned Ok". Review 중 `acquire`를 `peer_addr().is_ok()`로 되돌리자 둘 다 다시 실패함.
- MiniCPM handoff test: decode id `[69435, 2, 1345, 72, 2, ...]` (16 tokens, 텍스트 " Café</s>  .</s> ...") 대 기준 `[69435]`.
- Router failover e2e: HTTP 500, "timed out waiting for the prefill first token" (수정 전 3번 중 2번).

수정 후: failover e2e는 5번 중 5번 통과, 새 socket test 세 개는 25번 반복 모두 통과했다. MiniCPM handoff test는 기준 " Café" (1 token, `stop`)와 handoff " Café"를 보고하고, decode `completion_tokens`는 0 (첫 decode step이 EOS)이므로 1 + 0 = 1이다.

Orchestrator의 merge 전 gate (`db334ac9`):

| Gate | 결과 |
|------|--------|
| fmt, version, env-var docs, kernel dtype keys, kernel port dispatch, binary assets, Windows paths 검사 | 모두 exit 0 |
| Contract test | 27 + 4 통과 |
| clippy `-D warnings` | 경고 없음 |
| lib (`--profile test-fast`) | 1565 통과, 0 실패 |
| `remote_runtime` / `server_remote` / `multi_stage_smoke` | 6 / 5 / 1 통과 |
| `disaggregated_serving_e2e --include-ignored` | 2 통과 |
| `disaggregated_router_e2e --include-ignored` | 7 통과 (51.7초), soft-skip 없음 |
| `disaggregated_e2e` | 51 통과 |

## 7. 남은 위험과 후속 조치

- **Failover 지연은 reconnect backoff만큼이다.** Router transport는 기본 `PoolConfig` (connect 시도 6회, 100 ms에서 두 배씩 늘어나는 지연)를 쓰므로, 죽은 node로 가는 첫 요청은 `send`가 error를 내고 failover가 돌기까지 약 3.1초의 backoff를 기다린다. 600초 대기보다는 낫지만, router role에 더 짧은 retry budget을 줄지 검토할 만하다.
- **요청을 받은 뒤에 죽는 node**는 여전히 그 요청에 handoff timeout 전체를 소모시킨다. 여기서의 감지는 idle connection 재사용 경우만 다루며, `docs/CONTINUOUS_BATCHING.md`에도 그렇게 적혀 있다.
- **읽지 않은 byte가 FIN을 가린다.** `peer_closed`는 대기 중인 data가 있으면 peer가 그 뒤에 닫았더라도 "살아 있음"으로 보고한다. Pooled outbound stream에 그런 byte를 남길 수 있는 것은 `rpc_call` 응답 (`distributed::bench`에서 사용)뿐이고, disaggregated handoff 경로는 그러지 않는다. 기존 문제, LOW.
- **Send 실패 시 `active`가 샌다.** `acquire` 뒤 `send`가 error를 내면 stream이 `release` 없이 버려져 `active`가 증가한 채로 남는다. `stats()`에만 쓰인다. 기존 문제, LOW.
- **`prompt_tokens_details`는 router의 설정을 따른다.** Router의 prompt cache가 켜져 있으면 worker node가 꺼져 있어도 필드를 보고한다. 값은 항상 0이므로 형태만의 문제다.
- **Router chat 내용이 앞 공백 하나만큼 다르다.** MiniCPM "café" chat case에서 router는 " Café", single node는 "Café"를 반환한다. Chat byte-fallback test는 usage만 비교하고 content는 비교하지 않아 이를 잡지 못한다. #2695 범위 밖이며 아직 issue로 등록되지 않았다.
- **`Llama3Model::eos_token_ids`는 여전히 Llama 3 id로 고정돼 있다.** 이제 모든 serving 경로가 `config_eos`를 merge하지만, 다른 vocabulary를 가진 llama-format checkpoint에 대해 model 자체의 EOS 목록을 믿는 코드는 여전히 틀린다.
- **CI에서 돌지 않는다.** Real-process suite는 여전히 `#[ignore]`이고 checkpoint (`qwen3-0.6b-4bit`, `minicpm-2b-sft-4bit-llama-format-mlx`, `gemma-3-1b-it-4bit`)와 GPU가 필요하다. CI에서 도는 것은 `test_secret_file_auth_matches_server_resolution`과 lib unit test뿐이다.

## 8. 학습 포인트

### 8.1 CLOSE_WAIT는 밖에서 보면 살아 있는 것처럼 보인다

원격 process가 종료되면 그쪽 kernel이 FIN을 보내고 local socket은 CLOSE_WAIT로 간다. `getpeername`은 여전히 성공하고, `write`도 local send buffer에 복사만 하므로 성공한다. 상대 kernel이 응답으로 보내는 RST는 그 다음 호출에서야 드러난다. 재사용이 안전한지 알고 싶은 pool은 읽어 봐야 하고, non-blocking socket에서 소비하지 않는 `recv(MSG_PEEK)`는 부작용 없이 답한다. `0`은 EOF, `EWOULDBLOCK`은 대기 중인 data가 없고 아직 연결되어 있다는 뜻이다.

```rust
let mut probe = [std::mem::MaybeUninit::<u8>::uninit(); 1];
match socket2::SockRef::from(&self.stream).peek(&mut probe) {
    Ok(0) => true,  // peer closed
    Ok(_) => false, // unread bytes, alive
    Err(e) => !matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted),
}
```

### 8.2 Reactor 수준의 peek는 reactor가 본 것에 의존한다

Tokio의 `peek`는 reactor가 추적하는 readiness를 기다린다. `now_or_never`로 한 번만 poll하면 event loop가 FIN을 등록하기 전까지 `None`을 돌려주므로, 같은 socket 상태가 scheduling에 따라 "살아 있음"으로도 "닫힘"으로도 읽힌다. fd를 직접 호출하면 probe 결과가 kernel 상태만의 함수가 된다.

### 8.3 Test의 assertion뿐 아니라 전제도 증명하라

Byte-fallback test는 이름, 문서, review 모두에서 byte-fallback coverage로 취급되었지만 그 prompt는 byte piece를 한 번도 만들지 않았다. 전제 ("기준 출력에 vocabulary가 piece로 낼 수 없는 문자가 있다")를 assertion으로 적어 두면, 그런 조용한 어긋남이 명확한 메시지를 가진 실패로 바뀐다.

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|---------|-------------|-----------|
| CLOSE_WAIT | 상대의 FIN을 받은 뒤 local 쪽이 닫기 전의 TCP 상태 | 죽은 connection에서 `peer_addr`와 첫 write가 성공하는 이유 |
| `MSG_PEEK` | Data를 queue에서 제거하지 않고 반환하는 `recv` flag | AEAD sequence number를 유지하는 비소비 liveness probe |
| Byte fallback | Vocabulary에 없는 문자를 SentencePiece가 `<0xXX>` piece로 내는 것 | MiniCPM token 수 parity test가 실행해야 하는 경로 |
| `config_eos` | `read_eos_token_ids`가 `config.json`에서 읽은 EOS id | Llama format MiniCPM-2B의 유일하게 올바른 EOS 출처 |

### 관련 PR/이슈

- Issue #2695: 이 PR이 고치는 네 가지 실패.
- #2650, #2674, #2684: cluster handshake, role 인증, frame별 AEAD. #2674가 `TestClusterSecretFile`을 도입했고, #2684의 AEAD stream 때문에 liveness probe가 byte를 소비하면 안 된다.
- PR #2671: 또 다른 수동 suite의 GB10 remote-pipeline parity 실패를 고침.
- Issue #2692: `TestClusterSecretFile::secret()`를 재사용할 router 인증 fixture.
- #387, #398, #708, #1473: router token 수, chat usage, MiniCPM fixture 재지정, single-node `prompt_tokens_details`.
