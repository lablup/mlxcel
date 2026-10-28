# PR #2708: #2671, #2680, #2684가 남긴 remote pipeline 잔여 문제 정리

**날짜**: 2026-10-28 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: 구현 완료, 00bb4723 시점의 보안/성능 리뷰에서 CRITICAL, HIGH, MEDIUM 이슈 없음, LOW 4건 중 2건은 7c8d340a에서 수정, 7c8d340a 기준 GB10 (spark-102) 로컬 게이트 통과, 작성 시점에 hosted CI 진행 중
**언어**: Rust, Markdown
**위험도**: 중간 (zero-config assignment protocol 버전이 1에서 2로 오르고, handshake 버전은 4로 유지된 채 remote stage 응답 형식이 바뀌므로 이 PR 전후로 빌드된 바이너리는 하나의 pipeline을 구성하지 못함)

## 요약

PR #2708 (#2694 해결)은 #2671, #2680, #2684가 remote pipeline에 남긴 일곱 가지 잔여 문제를 반영합니다. `src/distributed/`의 frame/pipeline 코드, Gemma4 stage 모델, 하나의 검증 절차를 공유하므로 한 PR로 묶었습니다. 보안 수정은 없고, 관측성, 토큰당 비용, bring-up 원자성, 테스트 커버리지에 관한 변경입니다.

가장 큰 변경은 토큰당 경로입니다. 이제 final stage가 logits를 stage를 거슬러 relay하지 않고, forward activation에 실린 `reply_nonce`를 사용해 `RunEntry` exchange의 `RunResult`로 coordinator에 직접 보냅니다. 디코딩 토큰 하나가 4개 대신 3개의 frame이 되고, 약 405 KB인 logits 응답이 wire를 한 번만 지납니다. frame을 제자리에서 seal하는 변경(전송마다 payload 복사 1회 감소)과 합쳐, 2-stage loopback qwen3-0.6b-4bit pipeline의 디코딩 처리량은 main (433560c4)의 median of medians 136.73 tok/s에서 156.94 tok/s로 약 15% 올랐고, 모든 반복에서 greedy 토큰이 동일했습니다.

나머지 항목: protected stream은 AEAD tag 검증 뒤에만 frame kind 바이트를 해석합니다. `--pp-peer`는 자기 역할을 설명하는 TOML 헤더를 씁니다. zero-config bring-up에 `Activate`/`Abort` 단계가 추가되어 한 피어의 commit 실패가 다른 피어에서 stage를 시작시키지 않습니다. sliding layer가 없는 Gemma4 stage는 sliding mask를 만들지 않습니다. 새 stage 테스트가 앞부분이 trim된 KV cache에서의 parity를 고정합니다.

## 1. 문제 정의

각 항목은 이슈 작성 시 main @ 54d63b0a에서, PR을 정리할 때 #2705 이후 main에서 다시 확인했습니다.

### 1.1 변조된 kind 바이트가 rejected-frame 로그를 빠져나감

`ProtectedStream::read_frame`은 body를 읽고 open하기 전에 `MessageKind::try_from(header[0])?`로 kind 바이트를 파싱했습니다. 헤더는 AEAD associated data이므로, 다른 유효한 값으로 바뀐 kind 바이트는 `open`에서 실패해 `FrameRejected`가 되고, 호출 측이 rate limit이 걸린 진단 예산 안에서 로그를 남깁니다. 반면 유효하지 않은 값(0, 또는 5~255)으로 바뀐 kind 바이트는 인증 전에 일반 `unknown message kind` 오류를 반환했고, 연결은 로그 한 줄 없이 닫혔습니다. 같은 변조가 공격자가 고른 값에 따라 보이기도 하고 안 보이기도 했습니다.

### 1.2 frame 경로의 피할 수 있는 토큰당 비용 두 가지

`TcpTransport::send`는 `serialize_message`로 payload를 만들고, `write_frame`은 seal 전에 두 번째 버퍼를 할당해 헤더와 payload를 복사했습니다. 토큰당 logits 응답의 경우 전송마다 405 KB를 복사합니다.

또한 stage가 2개 이상이면 final stage가 logits를 reverse `ActivationMessage`로 upstream에 돌려보내고, 각 stage가 이를 다시 upstream으로 relay하며, entry stage가 request와 micro-batch를 키로 하는 `PendingEntryReply` map을 사용해 coordinator용 `RunResult`로 바꿨습니다. 그래서 디코딩 토큰 하나가 logits를 두 번(qwen3 기준 각각 약 405 KB의 base64 JSON) 4개의 frame으로 실어 날랐습니다: `RunEntry`, forward activation, reverse activation, `RunResult`. relay하는 stage마다 이 payload에 대해 JSON decode, AEAD open, JSON encode, AEAD seal을 수행했습니다.

### 1.3 피어 TOML 헤더가 coordinator를 설명함

`--pp-peer`는 받은 cluster를 `render_deterministic_toml`로 썼고, 그 헤더는 파일이 `mlxcel-server --pp-auto`로 생성되었으며 bring-up 명령을 다시 실행하면 바이트 단위로 같은 파일이 나온다고 말합니다. 이는 coordinator의 명령이자 coordinator의 보장입니다. 피어를 다시 실행해도 아무것도 재현되지 않고, 다음에 commit되는 assignment가 파일을 덮어씁니다.

### 1.4 commit 단계 실패 시 commit한 피어가 계속 실행됨

`push_stage_assignments`는 수락한 모든 피어에 `Commit`을 동시에 보냅니다. `Committed`로 응답한 피어는 즉시 `CommittedAssignment`를 전달하고, TOML을 쓰고, stage를 시작했습니다. 이후 다른 피어가 commit에 실패하면 coordinator는 `AssignmentPushFailed`로 종료했고, commit한 피어들은 결코 구성되지 않을 cluster의 stage를 계속 실행했으며 멈추라는 신호도 받지 못했습니다. 2단계 설계는 offer 단계에서만 원자성을 보장했습니다.

### 1.5 쓰이지 않는 Gemma4 sliding mask

`text_prefill_masks`는 `sliding_attn`이 `None`일 때도 `window < l <= 4096`이면 sliding-window prefill mask를 만들었습니다. `Gemma4StageModel::execute_hidden`은 모든 layer가 `full_attention`인 stage에 `None`을 넘기므로, 그런 stage는 어떤 layer도 읽지 않는 `[l, l]` mask를 할당했습니다.

### 1.6 trim된 cache에서의 stage parity 커버리지 부재

`gemma4_stage_tests.rs`는 prefill, window 초과 prefill, continuation prefill을 다뤘지만 `KVCache::trim_front`가 `live_start`를 전진시킨 경우(#430, `--max-kv-size`)는 없었습니다. trim 상태에서 global layer는 `offset`개가 아니라 `live_len`개의 key를 보며, dense 모델과 stage 모델 모두 maskless causal 경로에 머물러야 합니다. 기존 회귀 테스트는 dense 전용이었습니다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `tcp_frame.rs` | protected stream에서 kind 바이트를 `FrameOpener::open` 뒤에 파싱합니다. 어떤 값으로 변조된 kind든 `FrameRejected`이고, 인증을 통과한 알 수 없는 kind는 `unknown message kind {n} from an authenticated peer`입니다. plain stream과 헤더만 보는 oversize 검사는 그대로입니다. |
| 2 | `tcp_frame.rs`, `frame_protection.rs`, `tcp_transport.rs` | `encode_message_frame`은 앞에 9바이트 헤더, 뒤에 `MAX_TAG_LEN` (32)바이트를 예약한 버퍼에 메시지를 인코딩합니다. `finish_frame`이 헤더를 쓰고 제자리에서 seal하며, 유일한 seal 경로로서 `write_prepared_frame` (`TcpTransport::send`)과 `write_frame(kind, &[u8])`이 사용합니다. `serialize_message`는 테스트 전용이 되었습니다. |
| 3 | `activation_transfer.rs`, `remote_service.rs`, `runtime.rs` | `ActivationMessage::reply_nonce` (`#[serde(default)]`)가 `RunEntry` nonce를 stage들을 통해 전달합니다. final stage는 coordinator의 인증된 identity로 `RunResult`를 보냅니다. `PendingEntryReply`, 그 map, 응답의 `pending_entry_replies` 필드를 제거했습니다. coordinator는 마지막 stage에서 온 `RunResult`만 받습니다. |
| 4 | `cluster_init.rs`, `startup.rs`, `mod.rs` | `render_peer_assignment_toml`이 private `render_toml_body`를 `render_deterministic_toml`과 공유하며, `--pp-peer`가 이것으로 파일을 씁니다. |
| 5 | `stage_assignment.rs` | `Committed` 뒤에 `AssignmentFrame::Activate`와 `Abort { detail }`이 옵니다. `ASSIGNMENT_PROTOCOL_VERSION`은 2입니다. 피어는 `Activate`에서만 assignment를 전달합니다. |
| 6 | `gemma4.rs` | `sliding_attn`이 `None`이면 `text_prefill_masks`가 `(None, None)`을 반환합니다. |
| 7 | `gemma4_stage_tests.rs` | `two_stage_split_matches_dense_after_front_trim_bitwise`와, 가운데 stage가 full attention인 3-stage 분할 테스트를 추가했습니다. |
| 문서 | `docs/distributed.md` | assignment protocol (3단계, 버전 2), 피어 TOML 헤더, 응답 권한, 토큰당 3-frame 흐름. 이 절들에 대응하는 한국어 문서는 없습니다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 22 |
| 추가 / 삭제 라인 | +1685 / -236 |
| 추가된 테스트 | 24 (`#[test]`와 `#[tokio::test]`) |
| 커밋 | 8deca793 (일곱 항목), 00bb4723 (`RunResult`의 coordinator identity, push 실패 메시지), 7c8d340a (`Abort` 로그 제한, 짧은 prepared frame), 7dd6aa62 (main 머지) |

main 머지(7dd6aa62)는 #2706과 #2707을 가져오며, 이 PR의 22개 파일 중 어느 것도 건드리지 않습니다.

## 3. 기술적 선택과 그 이유

### 3.1 먼저 인증하고, 그다음 kind를 해석

protected stream에서 `MessageKind::try_from`을 `open` 뒤로 미뤄, 두 호출 지점의 기존 rejected-frame 로깅을 호출 측 변경 없이 재사용합니다. 기각된 대안은 헤더만 보고 유효하지 않은 kind를 모두 `FrameRejected`로 처리하는 것이었습니다. 이는 인증되지 않은 헤더를 인증 실패로 기록하는 것이며, oversize 검사는 의도적으로 그렇게 하지 않습니다(`protected_stream_refuses_oversize_header_before_body`). 또한 실제 버전 불일치 상황을 변조 메시지 뒤에 숨기게 됩니다. 인증을 통과한 알 수 없는 kind를 별도 오류로 두면 두 원인을 구분할 수 있고, 이 오류는 secret 보유자만 만들 수 있습니다.

oversize 검사는 여전히 body를 읽기 전에 헤더에서 수행되므로, 변조된 길이로 큰 읽기나 암호 연산을 유도할 수 없습니다.

### 3.2 한 번 크기를 잡고, 하나의 seal 경로

`MAX_TAG_LEN`은 가장 큰 `FrameSuite::tag_len`인 32이고, `const` 블록이 `FrameSuite::ALL`의 모든 suite가 이 안에 들어오는지 검사합니다. 이 검사가 없으면 tag가 더 긴 suite가 나중에 추가될 때 `seal`이 재할당하면서 복사가 조용히 되살아납니다. borrowed payload용 `write_frame`은 이제 얇은 앞단(prepared frame에 복사한 뒤 `write_prepared_frame`)이어서, stream chunk, RPC, stage assignment frame도 메시지와 같은 `finish_frame`을 거칩니다. 이 경로의 복사 횟수는 그대로이고, payload를 직접 만드는 `send`만 복사 한 번을 줄입니다.

`write_prepared_frame`은 `pub(crate)`이고, `finish_frame`은 헤더보다 짧은 버퍼에 대해 slice panic 대신 오류를 반환합니다(리뷰 후 7c8d340a에서 수정).

범위 밖: `TransportMessage::Control` 내부의 JSON-to-`Bytes` 복사는 `Transport` trait 변경이 필요합니다.

### 3.3 logits는 final stage에서 coordinator로 한 번만

| 선택지 | 장점 | 단점 |
|---|---|---|
| reverse relay 유지 | 프로토콜 변경 없음 | logits가 `num_stages`번 wire를 지남, entry stage에 요청별 응답 상태 |
| logits용 raw `TensorData` frame | base64 없음 | nonce나 request id가 없음, 토큰당 약 100 KB를 위해 `exchange`에 두 번째 상관 경로 필요 |
| **채택: forward activation의 nonce, final stage의 `RunResult`** | logits 전송 1회, 어떤 stage에도 요청별 응답 상태 없음, 기존 exchange 상관 규칙 재사용 | coordinator가 명령을 보낸 stage가 아닌 stage의 응답을 받아야 함 |

coordinator 측은 #2637의 상관 규칙을 유지하고 하나를 추가합니다. `ExchangeMatch`가 `result_peer`를 가지며, `RunEntry`에서는 `stage_peers.last()`, 그 외에는 명령을 받은 피어입니다. `RunResult`는 `result_peer`에서 와야 하고, `Ack`, `State`, `Error`는 여전히 명령을 받은 피어에서 와야 합니다. 가운데나 entry stage에서 온 `RunResult`, 오래된 nonce, request나 micro-batch가 맞지 않는 응답은 버려지고 대기는 절대 deadline까지 이어집니다. single-stage pipeline은 첫 stage가 곧 마지막 stage이므로 동작이 같습니다.

stage 측에서는 reverse activation을 경고와 함께 버리고, `reply_nonce` 없는 forward activation이 final stage에 도달하면 logits를 경고와 함께 버립니다. nonce가 없으면 coordinator가 frame을 매칭할 수 없으므로, 보내봐야 coordinator가 버릴 frame이 될 뿐입니다. in-process `stage_worker`는 자체 reverse 경로를 유지합니다.

### 3.4 설정된 주소가 아니라 coordinator의 인증된 identity로 응답

처음 구현(8deca793)은 `RunResult`를 `self.coordinator.configured()`로 보냈습니다. unspecified IP로 기재된 coordinator(`0.0.0.0:19000`, control 주소가 `--host 0.0.0.0`을 따를 때 `--pp-auto`가 쓰는 값이며, 피어가 그 TOML 사본으로 실행될 때 재사용됨)는 접속할 호스트를 가리키지 않으므로, 다른 머신의 final stage는 자기 loopback에 접속했고 모든 디코딩이 timeout되었습니다. 00bb4723은 coordinator의 가장 최근 명령의 인증된 출처(`coordinator_identity`)를 기록해 다른 모든 응답과 마찬가지로 그곳으로 `RunResult`를 보냅니다. 설정된 주소는 명령을 받기 전의 fallback입니다. 모든 `RunEntry` 앞에는 모든 stage로 가는 `PrepareSequence` broadcast가 있으므로, final stage는 logits를 만들기 전에 명령을 본 상태입니다. coordinator를 `[::]:port`로 설정한 `final_stage_sends_run_result_to_authenticated_coordinator_identity` 테스트는 수정 전 `Elapsed(())`로 실패했습니다.

### 3.5 모두 commit하면 activate, 아니면 abort

피어는 `Committed`를 쓴 뒤 같은 연결에서 `commit_timeout` 동안 commit을 보류하고, `Activate`에서만 assignment를 전달합니다. `Abort`, 다른 frame, 연결 종료, timeout은 `shared.committed`를 되돌리고 피어를 listening loop로 돌려보내며, commit 이전의 폐기 경로와 같습니다. coordinator는 commit 단계의 stream을 유지합니다. 모든 피어가 commit하면 전원에게 `Activate`를 보내고, 하나라도 commit에 실패하면 commit한 모든 피어에게 `Abort { detail }`(실패 요약, 문자 경계에서 1 KiB로 자름)을 push deadline 안에서 best effort로 보낸 뒤 기존처럼 실패합니다.

기각된 대안은 `Committed`에서 activate하고 백그라운드에서 `Abort`를 기다리는 방식이었습니다. 그 시점에는 stage가 이미 weight를 로드하고 transport를 bind하고 있어, abort가 일부 시작된 stage를 되돌려야 합니다.

남는 구간은 모듈 문서와 `docs/distributed.md`에 적었습니다. 두 `Activate` 쓰기 사이에 coordinator가 죽으면 이미 activate된 피어는 계속 실행되며, 이는 bring-up 성공 직후 crash와 같습니다. `Activate` 쓰기 실패는 reason이 `committed but not activated`로 시작하는 `AssignmentPushFailed` 항목으로 보고되고, 이때 오류 메시지는 stage가 하나도 시작되지 않았다고 말하는 대신 나머지 피어가 stage를 실행 중이니 멈추라고 안내합니다(00bb4723).

피어는 받은 `Abort` detail을 1 KiB로 제한하고 `escape_debug`로 escape해 기록하므로, 인증된 coordinator라도 64 KiB detail이나 추가 로그 줄을 피어 로그에 넣을 수 없습니다(7c8d340a).

### 3.6 버전 변경

`ASSIGNMENT_PROTOCOL_VERSION`이 1에서 2로 올라, 버전이 섞인 coordinator와 피어는 이전 coordinator가 보내지 않을 `Activate`를 피어가 기다리는 대신 `assignment protocol mismatch`로 즉시 실패합니다. handshake `PROTOCOL_VERSION`은 4로 유지합니다. #2705가 같은 미출시 주기에 4로 올렸고, 어떤 태그된 릴리스(최신은 v0.8.0-beta.1)도 버전 3이나 4를 포함하지 않습니다. 응답 형식 변경(`pending_entry_replies` 제거, `reply_nonce` 추가)은 이 미출시 버전에 함께 실립니다.

## 4. 검증

### 4.1 수정 전 실패 증거

새 동작 테스트는 각 수정 전에 변경되지 않은 코드에서 실행했습니다.

- 항목 1: `protected_stream_rejects_tampered_invalid_kind_as_frame_rejected`는 `unknown message kind: 255`로 실패했고, `protected_stream_reports_authenticated_unknown_kind_distinctly`도 구분되지 않는 같은 오류로 실패했습니다.
- 항목 3: `final_stage_sends_run_result_to_coordinator_not_upstream`은 timeout(`Elapsed(())`), `run_result_from_last_stage_completes_forward`는 `remote stage 10.1.1.1:1 timed out after 5s`, `run_result_from_entry_stage_is_dropped_in_multi_stage_pipeline`은 stage 0의 `RunResult`가 exchange를 완료해서 실패했고, `final_stage_sends_run_result_to_authenticated_coordinator_identity`는 응답이 설정된 주소로 가서 timeout되었습니다.
- 항목 5: `peer_discards_committed_assignment_on_abort_and_activates_on_the_next`는 `the peer delivered an assignment that was never activated`, `coordinator_aborts_committed_peers_when_another_commit_fails`는 `the real peer delivered an assignment the coordinator aborted`로 실패했습니다.
- 항목 6: `text_prefill_masks_without_sliding_layer_builds_no_mask`는 `a stage without a sliding layer must not allocate a sliding mask`로 실패했습니다.

항목 7은 커버리지이므로 이전 코드에서도 통과합니다. 민감도는 stage 경로에 offset 크기의 global prefill mask를 주는 임시 변형(커밋하지 않았고 파일은 바이트 단위로 복원)으로 확인했습니다. 이때 테스트는 `trim_front(4)` 뒤의 3-token continuation에서 최대 절대 logit 차이 5.46875e-2로 실패했습니다. 시나리오에서 `offset != live_len` (12 대 8)인 Gemma4 stage 테스트는 이것이 유일합니다.

### 4.2 로컬 게이트

GB10 spark-102, `--profile test-fast --features cuda,test-utils`, 테스트 바이너리는 GPU lock 밖에서 빌드하고 `gpu-lock` 아래에서 `--test-threads=1`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`로 실행, 7c8d340a 기준:

| 대상 | 결과 |
|---|---|
| `cargo fmt --check`, `cargo clippy --lib --tests -D warnings` | clean |
| `--lib distributed::` | 1494 통과, 4 ignored |
| `--lib models::gemma4` | 127 통과, 15 ignored |
| `zero_config_cluster_init --include-ignored` | 7/7 (CLI rehearsal이 실제 바이너리로 `Activate`까지 진행) |
| `pipeline_remote_real_models --include-ignored` | 10/10 (`glm_moe_dsa` 제외) |
| `pipeline_server_remote_real_models --include-ignored` | 7/7 (`glm_moe_dsa` 제외) |
| `pipeline_ci_multi_stage_real_models --include-ignored` | 4/4, logits가 이제 가운데 stage를 건너뛰는 3-host parity 테스트 포함 |
| `disaggregated_serving_e2e` / `disaggregated_router_e2e` / `disaggregated_e2e` | 2/2, 7/7, 51/51 |

제외한 `glm_moe_dsa` 테스트는 418.6 GB `glm5-4bit` 체크포인트가 필요하며 128 GB GB10에서는 로드할 수 없습니다.

### 4.3 디코딩 처리량

`pipeline_remote_decode_bench`: qwen3-0.6b-4bit, loopback 2 stage, 128 토큰, 실행마다 warm-up 후 7회 측정. main 433560c4 (A)와 이 브랜치 (B)를 조용한 GB10에서 A, B 순으로 5라운드 번갈아 실행했고, 각 실행은 `gpu-lock` 아래에서 돌렸습니다. 두 바이너리 모두 벤치의 고정 포트를 19990으로 옮겨 빌드했습니다(로컬 전용, 커밋하지 않음). 모든 반복에서 greedy 토큰이 동일했습니다.

| 라운드 | main (tok/s) | branch (tok/s) |
|---|---|---|
| 1 | 120.67 | 149.33 |
| 2 | 140.70 | 142.75 |
| 3 | 131.38 | 159.04 |
| 4 | 140.29 | 156.94 |
| 5 | 136.73 | 162.01 |
| **Median of medians** | **136.73** | **156.94** |

브랜치가 약 15% 빠르고, 가장 느린 라운드(142.75)가 main의 가장 빠른 라운드(140.70)보다 높습니다. 이슈의 수용 기준은 "run-to-run 편차를 넘어 main보다 느리지 않을 것"이었습니다. 측정은 loopback 전용이며, 두 노드 링크 측정(Thunderbolt 또는 10/25 GbE)은 범위 밖으로 실행하지 않았습니다. 없어진 전송이 토큰당 405 KB logits frame 하나 전체이므로 실제 링크에서는 절대 시간 절감이 더 클 것으로 보이지만, 측정되지는 않았습니다.

## 5. 남은 공백과 위험

- **버전이 섞인 빌드가 버전 신호 없이 실패합니다.** handshake 버전이 그대로이므로 #2705와 이 PR 사이의 main 빌드는 이 PR 빌드와 handshake를 완료합니다. 이전 coordinator는 새 `Ack`/`State`를 decode하지 못하고(필수였던 `pending_entry_replies` 필드가 없음), 새 coordinator는 이전 entry stage가 relay한 `RunResult`를 버립니다. 둘 다 exchange timeout으로 끝납니다. 태그된 릴리스는 영향이 없고 버전 혼용은 원래 지원하지 않지만, 장애가 버전 불일치가 아니라 느리거나 죽은 stage처럼 보입니다.
- **entry가 아닌 stage의 오류는 여전히 timeout됩니다.** 가운데나 final stage의 실행 오류는 여전히 nonce 0으로 upstream에 보내져 coordinator timeout으로 끝납니다(#2694에서 범위 밖으로 둠). 이제 모든 stage에 `reply_nonce`와 `coordinator_identity`가 있으므로 final stage가 해당 exchange의 `Error`를 직접 보낼 수 있습니다. coordinator도 `RunEntry`에 대해 `result_peer`에서 온 `Error`를 받아야 합니다.
- **`coordinator_identity`는 마지막 명령 기준입니다(LOW, 미해결).** coordinator가 unspecified IP로 설정되면 `PeerMatcher`는 해당 포트의 인증된 identity를 모두 받아들이므로, 다른 호스트의 그 포트에서 listen하는 secret 보유자가 아무 명령이나 보내 이후 logits를 자신에게 돌릴 수 있습니다. secret과 수동 작성 설정이 필요합니다(zero-config 피어는 `localize_cluster`에서 구체적인 주소를 받음). `PrepareSequence` 시점에 요청별로 응답 identity를 바인딩하면 막을 수 있습니다.
- **transport read loop는 인증된 알 수 없는 kind에서 여전히 조용히 연결을 닫습니다(LOW, 미해결).** `tcp_transport.rs`는 닫기 전에 `FrameRejected`만 로그로 남깁니다. 새 구분 오류는 stage assignment listener의 로그에는 나타나지만 transport의 로그에는 나타나지 않아, 버전이 다른 멤버가 그쪽에서는 여전히 보이지 않습니다.
- **activation 구간.** 문서화한 대로, `Activate` 쓰기 사이에 coordinator가 crash하면 앞선 피어가 계속 실행됩니다.
- **두 노드 링크 수치**는 #2656과 이 변경 모두에 대해 아직 없습니다.

## 6. 학습 포인트

### 6.1 AEAD associated data는 `open`이 성공할 때까지 신뢰하지 않는다

헤더는 인증되지만 암호화되지 않으므로 미리 파싱하고 싶어집니다. `open` 전에 헤더로 내린 결정(여기서는 알 수 없는 kind 거부)은 인증된 경로와 다르게 동작하는 비인증 오류 경로를 만들고, 그 차이는 밖에서 관찰됩니다. 인증 전에 둘 것은 자원 사용을 제한하는 검사(길이 상한)뿐이며, 그 검사도 변조 탐지가 아니라 그 자체로 분류해야 합니다.

### 6.2 한 번 예약하고, 제자리에서 seal

```rust
let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload_len + MAX_TAG_LEN);
frame.resize(FRAME_HEADER_LEN, 0); // header written later by finish_frame
frame.extend_from_slice(payload);   // or encode the message directly here
finish_frame(sealer, kind, &mut frame)?; // header + in-place encrypt + tag append
```

`finish_frame_seals_without_reallocating` 테스트는 seal 전에 `frame.as_ptr()`를 기록하고 모든 suite에서 값이 바뀌지 않는지 확인해, "복사 없음"을 주석이 아니라 검사되는 속성으로 만듭니다. `FrameSuite::ALL`에 대한 `const` 검사가 suite가 추가되어도 이를 유지합니다.

### 6.3 nonce로 상관시키고, identity로 라우팅

응답을 명령을 받은 stage 밖으로 옮기는 데 새로운 신뢰가 필요하지 않았습니다. coordinator는 이미 요청마다 새 nonce, request id, micro-batch로 응답을 매칭하므로, 어떤 인증된 피어가 `RunResult`를 보낼 수 있는지 바꾸는 것은 matcher의 필드 하나 변경입니다. 반대로 설정된 주소는 호스트를 가리키지 않을 수도 있는(`0.0.0.0`) 접속 대상이므로, 응답은 coordinator 트래픽의 인증된 출처를 따릅니다.

### 6.4 commit은 activation이 아니다

프로세스 간 2단계 commit은 "prepared" 시점에 되돌릴 수 없는 일이 일어나지 않을 때만 원자적입니다. `Committed`에서 stage를 시작하면 commit 단계가 원자적이지 않습니다. 세 번째 메시지를 추가하면 되돌릴 수 없는 단계가 coordinator가 결과를 아는 시점 뒤로 옮겨지고, 훨씬 작은 구간(`Activate` 쓰기 사이의 crash)만 남으며 그것은 숨기지 않고 문서화됩니다.

## 7. 참고 자료

- Issue #2694: Remote pipeline residuals from #2671, #2680, #2684
- PR #2671 (#2654 해결): remote pipeline stage를 dense kernel 경로에 유지 (Gemma4 stage mask)
- PR #2680 (#2658 해결): `--pp-peer` 프로세스에 stage assignment push
- PR #2684 (#2656 해결): 연결별 AEAD로 모든 cluster frame 보호, JSON 문서의 base64 tensor 인코딩
- Issue #2637: cluster transport 인증과 응답 상관 (이 PR이 확장한 응답 상관 규칙)
- PR #2705 (#2692 해결): handshake `PROTOCOL_VERSION` 4, 인증 전 수락 단계 강화
- PR #2700: `ProtectedStream::peer_closed`, disaggregated serving/router e2e 수정
- Issue #430: `--max-kv-size`에서 단조 offset 기준으로 크기를 잡던 Gemma prefill mask (`KVCache::trim_front`)
