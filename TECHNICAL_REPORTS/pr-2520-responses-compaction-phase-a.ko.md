# PR #2520: Responses 대화 압축, phase A

**Date**: 2026-10-21
**Status**: Completed (PR open, #2397 참조, phase B 예정)

## 요약

PR #2520은 Responses API에 대화 압축(compaction)을 추가한다. #2397의 phase A다. 클라이언트가 `POST /v1/responses/compact`(또는 `/responses/compact`)를 호출하면 세 가지를 돌려받는다. instruction 메시지, 가장 최근의 온전한 turn들, 그리고 `compaction` item 하나다. 이 item에는 버려진 기록을 모델이 요약한 내용이 담긴다. `[...output, 새 메시지]`를 `/v1/responses`로 보내면 버려진 turn 대신 요약이 재생된다. 요약은 `previous_response_id` chain을 따라 계속 재생된다. 나머지 기능은 phase B에서 다룬다. `context_management`를 통한 자동 압축, 남은 context에 맞춘 출력 상한, `mlx.compaction.*` 스트리밍 이벤트가 그것이다. phase B는 이슈를 닫는 별도 PR로 낸다. 리뷰 중에 maintainer가 PR을 검토하고 수정 커밋 두 개를 push했다. 리뷰와 보안 점검에서 HIGH 두 건이 나왔고 모두 고쳤다. 변경 규모는 19개 파일, +2337/-16이며, 이 중 약 980줄이 테스트다.

## 1. 문제 정의

`/v1/responses`로 긴 agent 세션을 진행하면 결국 모델 context를 넘어 실패한다. OpenAI Responses 프로토콜과 Codex는 이를 압축으로 해결한다. 명시적인 compact 호출로 압축하고, 버려진 기록 대신 불투명한 `compaction` item을 보낸다. 이 PR 전의 mlxcel은 이 endpoint를 mount하지 않았다. compaction item을 다시 보내는 클라이언트는 400을 받았다. upstream mlx-vlm은 Blaizzy/mlx-vlm#2408에서 이 기능을 추가했고, 후속 PR은 #2418, #2458, #2501이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `responses_compaction.rs`(신규) | 분할 지점 선택, `CompactionBackend` trait, `compact_history` |
| `responses_compaction/capsule.rs`(신규) | capsule encode/decode, 재생, instruction 중복 제거 |
| `routes/responses_compact.rs`(신규) | handler와 backend: render, token 계산, 요약 생성 |
| `routes/responses_compact_media.rs`(신규) | 압축 1회 단위 이미지 cache. fetch 전에 허용 여부를 먼저 검사한다 |
| `types/responses_request.rs`, `responses_response.rs`, `responses_translator.rs`, `responses_store.rs` | `compaction` 입력·출력 item, 재생 hook, chain 저장 |
| `app.rs`, `routes/mod.rs`, `availability_tests.rs` | 두 route mount, 모델 없는 서버의 501 동작 일치 |
| 테스트 | `responses_compaction_tests.rs`(448), `responses_compact_route_tests.rs`(532) |
| 문서 | `docs/responses-api.md`, `docs/server-features.md`, `CHANGELOG.md` |

## 3. 기술적 선택과 그 이유

### 3.1 capsule은 암호화하지 않고 인코딩만 한다

`encrypted_content`는 `{"v":1,"summary":...,"retained":[...]}`를 base64url로 인코딩한 값이다. capsule이 요약하는 대화는 클라이언트가 이미 갖고 있다. 그래서 암호화해도 보호할 것이 없다. HMAC key를 쓰려면 재시작 뒤에도 남는 key 저장소가 따로 필요하다. 대신 서버는 capsule을 신뢰할 수 없는 입력으로 다룬다.
- decode 전에 인코딩된 크기를 검사한다.
- `v`는 1이어야 한다.
- `retained` 안에 다른 capsule이 들어 있으면 거부한다.
- retained item은 일반 입력 parser, 이미지 gate, 제한을 그대로 거친다.

위조된 capsule에는 클라이언트가 원래 `input`으로 보낼 수 있는 role만 담을 수 있다.

### 3.2 선택 로직은 call과 그 output을 나누지 않는다

instruction(system, developer 메시지)은 원래 role로 유지한다. 분할은 두 위치에서만 허용한다.
- 열린 function call이 없을 때, user 메시지 바로 앞
- 마지막으로 열린 call을 닫는 output 바로 뒤(mlx-vlm#2501)

끝에 남은 미완료 tool group은 통째로 유지한다. 일부 output만 도착한 병렬 call도 마찬가지다. 분할 지점은 binary search로 고른다. 유지 부분이 `keep_tokens`에 들어가는 가장 앞쪽 지점이 선택된다. `keep_tokens` 기본값은 context의 1/16이고 최대 8192다. 자르는 위치를 앞으로 옮기면 유지 부분은 늘어나기만 하므로 탐색은 단조적이다. 최신 user 메시지는 항상 유지한다. 선택 로직만으로는 결과가 짧아진다고 보장할 수 없다. 그래서 사후 검사를 둔다. 압축한 context가 원래보다 짧지 않으면 압축은 실패하고 context는 바뀌지 않는다.

### 3.3 요약 생성은 비스트리밍 worker 경로를 재사용한다

요약 prompt는 버려진 앞부분에 고정된 인계 지시문을 붙여 만든다. temperature 0, reasoning off로 생성한다. 실행 경로는 비스트리밍 `/v1/responses`와 같다. busy slot 검사, slot 계산, metrics, provider 호출을 그대로 쓴다. token 계산에는 `/v1/responses/input_tokens` 경로를 쓴다. context를 넘는 경우는 HTTP 400, type `exceed_context_size_error`, code `context_length_exceeded`를 반환한다. 해당 경우는 세 가지다. 기록이 window 하나를 넘을 때, 요약 prompt가 들어가지 않을 때, 결과가 짧아지지 않을 때다. 이슈는 "413 계열"을 제안했다. 그러나 서버의 기존 overflow 응답과 upstream이 모두 400을 쓰므로 400을 유지했다.

### 3.4 재생과 chain(maintainer 커밋 d71e90ad)

가장 최근 capsule이 적용되고, 그보다 오래된 capsule은 버린다. 저장된 response는 instruction, capsule, 이후 turn을 기준 입력으로 저장한다. 그래서 이후 `previous_response_id` hop에서도 요약이 다시 재생된다. 예전에는 hop 하나만 지나면 요약을 잃었다. `conversation` transcript에는 새 turn만 덧붙인다. 완전히 같은 system/developer 메시지는 한 번만 남긴다. 재생할 때 body 제한을 넘는 결과는 compact 호출이 거부한다. 따라서 제한에 가까운 이미지를 유지하는 경우에도, 반환 즉시 쓸 수 없는 결과는 나오지 않는다.

### 3.5 리뷰 지적 사항

| 심각도 | 지적 | 수정 |
|---|---|---|
| HIGH | `MediaCache`는 모델이 이미지를 받는지 확인하기 전에 클라이언트가 지정한 이미지 URL을 fetch했다. 이미지 개수 제한도 없었다. 텍스트 전용 모델에서 임의 URL fetch가 다시 가능해졌다(#1451이 막은 구멍). 요청 하나로 64 MB 이미지 수천 개를 메모리에 붙잡아 둘 수도 있었다. | `95baf786`(maintainer): 첫 fetch 전에 이미지 지원 여부, tool, 이미지 개수를 검사 |
| HIGH | instruction 중복 제거가 `Vec::contains`를 써서 capsule 기록을 재생할 때마다 O(k²)였다 | `95baf786`: `HashSet` |
| MEDIUM | chain turn에서 다시 보낸 system 메시지가 중복되었다 | `87db02b0`, `d71e90ad` |
| MEDIUM | token을 셀 때마다 모든 이미지 URL을 다시 읽었다 | `87db02b0`: 압축 1회 단위 cache. route 테스트 기준 URL당 fetch가 5회에서 1회로 줄었다 |

## 4. 검증

| 항목 | 결과 |
|---|---|
| `responses_compact` | 28개 통과 |
| `server::routes::` | 493개 통과, 1개 ignored |
| `server::responses`, `server::types` | 각각 74, 107개 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings`, `cargo fmt --check` | 문제 없음 |

모든 실행은 macOS에서 `MLXCEL_BUILD_METAL=0 --profile test-fast --lib`로 했다. route 테스트는 scripted backend로 실제 handler를 실행한다. 테스트는 다음을 확인한다.
- compact output의 구조
- 재생한 prompt에 요약은 들어 있고 버려진 turn은 없는지
- 3-hop chain
- conversation에 덧붙는 turn 수
- body 제한 경계
- 이미지 URL당 fetch 1회

## 5. 미검증 항목과 남은 항목

- **미실행:** `mlx-community/Qwen3-30B-A3B-4bit`와 OpenAI Python SDK를 쓰는 통합 검증(`client.responses.compact` 후 `create`). 이 호스트에는 Metal도 CUDA도 없다. compact `output`의 입력 형태 message item을 SDK가 받아들이는지도 확인하지 못했다.
- **phase B:** `context_management` 자동 압축, 남은 context에 맞춘 출력 상한(mlx-vlm#2458), `mlx.compaction.started`·`mlx.compaction.completed` 스트리밍 이벤트(mlx-vlm#2418).
- **알려진 한계(문서화함):**
  - retained item은 `output`과 capsule 안에 모두 들어간다. 이슈가 이렇게 정했다. 그래서 `[...output, new]`를 보내면 같은 item이 두 번 전송된다. 클라이언트는 instruction item과 compaction item만 보내도 된다.
  - 가장 최근의 완료된 tool group이 요약에 흡수될 수 있다.
  - 요약은 instruction 바로 뒤의 assistant turn으로 재생된다. user/assistant 교대를 엄격히 요구하는 template은 이를 거부할 수 있다.
  - output을 받지 못한 call이 있으면 그 뒤로는 분할할 수 없다.
- **남은 MEDIUM(보안 리뷰):**
  - 중복 제거가 두 번 겹친다. 전체 context를 보는 두 번째 단계는 capsule 뒤의 system/developer reminder가 앞의 메시지와 똑같으면 그것도 지운다. 저장된 snapshot에는 그 메시지가 남는다.
  - 읽기에 실패한 URL은 token을 셀 때마다 다시 시도한다.
  - 이미지가 있는 요청은 token을 셀 때마다 worker slot을 차지한다.
- **규모:** 나눈 뒤에도 phase A는 +2337줄로 1,000줄 가이드라인을 넘는다. 이 중 약 980줄이 테스트이고 130줄이 문서다.

## 6. 학습 포인트

- 나중 단계의 gate를 위해 미리 fetch하는 cache를 두면 fetch가 gate보다 먼저 실행된다. 허용 여부 검사는 cache된 경로 안이 아니라 첫 fetch 전에 실행해야 한다.
- chain response를 저장할 때는 클라이언트가 보낸 그대로가 아니라 다음 hop이 context를 다시 만드는 데 필요한 형태로 저장해야 한다. 그렇지 않으면 요약은 hop 하나만 지나고 사라진다.
