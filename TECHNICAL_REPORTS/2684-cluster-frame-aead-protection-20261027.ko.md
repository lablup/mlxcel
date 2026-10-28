# PR #2684: 모든 cluster frame을 connection별 AEAD로 보호

**날짜**: 2026-10-27
**상태**: 검증 완료, PR은 draft로 열려 있음 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, Markdown
**위험도**: 중간 (보안에 민감한 transport 코드, wire 호환성 깨짐: handshake `PROTOCOL_VERSION` 2에서 3으로, stage protocol tensor bytes의 JSON 인코딩 변경)

## 요약

Issue #2656은 cluster TCP transport에 frame 단위 보호를 넣되, 방식은 측정값과 미리 정한 결정 규칙으로 고르라고 요구했다. 이 PR은 기존 handshake에서 connection마다 방향별 key 두 개를 유도하고 (cluster secret과 양쪽 hello nonce에 대한 HKDF-SHA256), handshake 이후의 모든 frame을 AES-256-GCM으로 봉인한다. Nonce는 방향별 암묵적 sequence number이고, 평문 `kind || len` header는 associated data다. 변조, 재전송, 순서 변경, 주입, 다른 connection에서 옮겨 온 frame은 열리지 않으며 connection이 닫힌다. 모든 송신 경로와 단일 수신 reader가 하나의 `ProtectedStream`을 거치고, `--pp-auto` stage assignment channel도 별도 key label로 같은 codec을 쓴다. 새 설정 항목은 없고 끄는 옵션도 없다.

PR은 또한 stage protocol JSON에서 대용량 tensor bytes를 숫자 배열 대신 base64 문자열로 보내도록 바꾼다. End-to-end 결과는 이 두 번째 변경이 좌우한다. 2-stage qwen3-0.6b-4bit decode가 main의 68.7 tok/s에서 133.2 tok/s로 오른다 (+94 퍼센트). 보호만 따로 재면 decode throughput을 약 4.0 퍼센트 깎는다 (138.8에서 133.2 tok/s). 이는 issue의 2 퍼센트 조항을 넘는다. 규칙의 대안인 frame별 HMAC-SHA256은 측정한 모든 payload 크기에서 AES-256-GCM보다 느렸으므로, 대안을 택했다면 throughput은 더 떨어지고 기밀성도 잃었다. 최종 선택은 AES-256-GCM이다.

## 1. 문제 정의

### 1.1 배경

PR #2650 (issue #2637)은 모든 cluster TCP connection 앞에 상호 HMAC-SHA256 handshake를 두었고, #2655 / PR #2674가 이를 보강해 disaggregated role까지 넓혔다. Handshake는 connect 시점에 connection을 한 번 인증할 뿐이었다. 그 뒤의 frame은 MAC도 암호화도 없는 평문 `[kind u8][len u64 BE][payload]`였다.

### 1.2 기존 문제점

- **인증된 stream 위의 frame 변조**: cluster segment의 경로상 공격자 (ARP 또는 DHCP spoofing, 장악된 switch나 host)는 handshake 이후 frame을 주입, 변조, 삭제, 재전송, 재정렬할 수 있었다. `RunEntry`, `Shutdown` 등 control command가 평문으로 오갔다.
- **Handshake relay**: handshake가 이후 traffic을 보호하는 어떤 것과도 묶여 있지 않아, 중계자가 실제 peer의 handshake를 server에 전달한 뒤 인증된 stream을 고칠 수 있었다.
- **기밀성 없음**: activation, logits, KV handoff에 실린 prompt, command가 wire에서 그대로 읽혔다.
- **비대한 stage protocol**: issue 범위가 아니라 측정 중에 발견했다. serde_json은 `Vec<u8>`을 숫자 배열로 인코딩해 byte당 약 3.6 문자를 쓰므로, qwen3-0.6b-4bit에서 token 하나를 decode할 때 같은 304 KB f16 logits를 담은 1.1 MB frame이 두 번 오갔다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 경로상 공격자의 command 주입 또는 activation 변조 | 높음 | 낮음 (문서가 이미 신뢰 구간으로 유지하라고 한 segment에서 경로상 위치가 필요) |
| Activation, logits, KV handoff를 통한 prompt와 출력 노출 | 높음 | 낮음 |
| Latency에 민감한 pipeline에서 frame별 암호 연산으로 인한 decode 저하 | 중간 | 측정 없이는 높음, 그래서 issue가 결정 규칙을 먼저 정했다 |

## 2. 기술적 선택과 그 이유

### 2.1 Suite: AES-256-GCM, 그리고 결정 규칙의 실제 결과

Issue의 규칙: 기밀성을 주는 방식을 택하려면 (a) 8 KiB에서 추가되는 p50 one-way latency가 후보 1 (frame별 HMAC) 대비 20 us 또는 5 퍼센트 중 큰 값 이내이고, (b) end-to-end decode tokens per second 감소가 baseline 대비 2 퍼센트 이하여야 한다. 아니면 후보 1을 택하고 payload가 평문으로 남는다고 문서화한다.

| 조항 | 측정값 | 결과 |
|------|--------|------|
| (a) 8 KiB p50, HMAC 대비 | AES-256-GCM 7.4 us, HMAC-SHA256 12.2 us, 보호 없음 5.5 us | 통과: AES-GCM이 비교 대상보다 4.8 us 빠름 |
| (b) main 대비 e2e decode, PR 그대로 | main 68.7 tok/s, branch 133.2 tok/s | 문자 그대로는 통과: +94 퍼센트 |
| (b) 보호만 분리한 e2e decode | compact 인코딩에 codec 없음 138.8 tok/s, AES-256-GCM 적용 133.2 tok/s | 미충족: -4.0 퍼센트 |

문자 그대로의 해석이 통과하는 이유는 compact 인코딩이 같은 PR에 묶여 있기 때문이다. 보호 자체의 비용으로 판단하면 조항 (b)는 실패한다. 이 경우 규칙이 정한 결과는 후보 1인데, 후보 1은 모든 지점에서 AES-256-GCM보다 느리다: 8 KiB에서 12.2 대 7.4 us, 1 MiB에서 1.008 대 0.422 ms, 16 MiB에서 27.2 대 17.9 ms. 이를 택하면 throughput은 더 낮아지고 기밀성도 버리게 되어, 대안이 존재하는 이유 자체와 맞지 않는다. 그래서 PR은 끄는 옵션 없이 AES-256-GCM을 유지하고, 보호만의 4 퍼센트 비용을 PR body와 `docs/distributed.md`에 기록했다.

`docs/distributed.md`를 읽는 reviewer는 그 문구 ("the rule's 2 percent clause holds with nothing to deviate from")가 묶인 수치를 근거로 한다는 점에 유의해야 한다. 보호만의 수치는 바로 다음 문장에 있다. 이 보고서는 4 퍼센트를 이 기능의 정직한 비용으로 본다.

ChaCha20-Poly1305는 다른 AEAD로 측정했고 이 host에서는 모든 크기에서 HMAC보다도 느렸다 (8 KiB에서 14.0 us). GB10 (AES와 PMULL을 갖춘 ARM Neoverse)과 Apple silicon 모두 AES 명령어가 있으므로 기본값은 AES-256-GCM이다. 세 suite 모두 코드에 남아 있는데, benchmark가 실제 transport에서 이를 측정하기 때문이다. `TcpTransportConfig::frame_suite`는 그 용도로만 존재하고 CLI는 이를 바꾸지 않는다.

### 2.2 Key schedule, nonce, framing

| 선택 | 배제한 대안 | 이유 |
|------|-------------|------|
| HKDF-SHA256, IKM = cluster secret, salt = client nonce \|\| server nonce, info = `mlxcel-cluster-frame-v1/<channel>/{c2s,s2c}` | Handshake HMAC key 재사용 | Connection별, 방향별 key: 한 connection에서 녹화한 frame은 다른 connection에서 검증되지 않고, 중계자는 secret 없이 key를 유도할 수 없다. HKDF extract는 nonce salt로 HMAC을 keying하고 handshake는 secret으로 keying하므로 두 용도가 겹치지 않는다. |
| 암묵적 64-bit sequence를 nonce로 사용 (`0x00000000 \|\| seq`), wire에 싣지 않고 wrap하지 않음 | Wire상의 명시적 sequence 또는 random nonce | 추가 byte가 0이고, 재전송, 재정렬, 삭제 후 주입이 replay window 없이 모두 tag 실패가 된다. 방향별 key가 key당 counter 유일성을 보장한다. `u64::MAX`에서는 sealer가 거부하고 connection이 닫힌다. |
| 평문 `kind \|\| len` header를 AAD로 | Header까지 암호화 | 수신 측은 할당이나 암호 연산 전에 과대한 `len`을 거부해야 한다. Header는 여전히 인증된다. |
| 양쪽 hello에 suite byte, 두 authenticator가 모두 덮음, 불일치 시 reject 사유 `0x02` | 협상 | Downgrade 경로가 없다: 각 측은 자기 설정 suite로 유도하고, 불일치는 첫 frame 실패가 아니라 타입이 있는 설정 오류가 된다. |

Frame당 wire overhead는 16-byte GCM tag다. Frame trace에서 `wire = payload + 25` (9-byte header와 tag)로 확인된다.

### 2.3 Socket과 함께 소유되는 단일 보호 경로

`ProtectedStream` (`src/distributed/tcp_frame.rs`)은 `TcpStream`과 그 `FrameSealer`, `FrameOpener`를 함께 가진다. `write_frame`과 `read_frame`만이 framing 코드이고, `send`, `send_stream`, RPC 요청, `handle_connection`에서 쓰는 RPC 응답, assignment channel이 모두 이를 호출한다. Main에는 손으로 짠 frame writer가 세 벌 있었는데 (`write_frame`, `send_stream`, RPC 응답) 모두 사라졌다. `ConnectionPool`은 이제 `ProtectedStream`을 저장하므로 재사용된 pooled connection은 자기 sequence를 이어 간다. 실패하거나 취소된 send는 connection을 pool에 돌려주지 않고 버린다. `seal` 안에서 sequence가 이미 증가했기 때문이다.

### 2.4 HKDF info의 channel label

첫 review에서 stage transport와 assignment channel이 같은 handshake label과 HKDF info를 쓴다는 점이 발견됐다. 두 listener 사이로 우회된 connection이 handshake를 통과했다 (그 뒤 frame 형식이 달라 parse에 실패했다). Commit `4f42196f`가 HKDF info에 `FrameChannel` (`transport`, `assignment`)을 넣었다. `client_handshake` / `server_handshake`는 signature를 유지하고 `Transport`를 쓰며, `client_handshake_on` / `server_handshake_on`이 channel을 받는다. Assignment channel의 기존 `u32` + JSON framing은 같은 codec의 `Control` frame으로 바뀌었다.

### 2.5 같은 PR에 묶인 compact tensor bytes

`pipeline::wire_bytes`는 `RunEntry` (`input_ids`, `attention_mask`), `RunResult` (`logits`), `ActivationMessage` (`tensor_data`, `attention_mask`, `position_ids`)의 대용량 `Vec<u8>` field용 serde `with` module을 추가한다. JSON에서는 base64 (byte당 1.33 문자), human-readable이 아닌 format에서는 raw bytes다. Decode token 하나당 logits frame 두 개가 각각 약 1.1 MB에서 405 KB로, forward activation이 7.5 KB에서 3.0 KB로 줄었다. 동기는 AEAD 비용 (봉인할 byte 감소)이었지만, 더 큰 효과는 main 자체의 매 step에서 JSON 숫자 배열 encode/parse 시간을 없앤 것이다. 묶은 것은 의도된 trade-off다. PR이 순이익인 이유이면서, 동시에 headline 수치로는 end-to-end 결정 조항을 판단할 수 없는 이유이기도 하다.

### 2.6 TLS 1.3 (후보 3)은 prototype하지 않음

rustls 0.23에는 external PSK 지원이 없어, 후보 3은 node별 임시 certificate, runtime dependency로서의 `rcgen`, connection당 두 번째 handshake, TLS exporter에 묶인 HMAC handshake가 필요했다. Record layer는 후보 2와 같은 AEAD primitive에 16 KiB record framing을 더하므로 후보 2보다 빠를 수 없고, 규칙은 3이 느리지 않은 경우가 아니면 2를 선호한다. 측정이 아니라 논증이다.

## 3. 측정

모두 GB10 host 한 대 (spark-102), loopback, 조용한 상태, `gpu-lock` 아래, 2026-10-27.

### 3.1 Frame microbenchmark

`tests/cluster_frame_bench.rs`: 127.0.0.1의 인증된 `TcpTransport` 두 개, server가 각 `TensorData` message를 두 번째 인증 connection으로 echo, one-way = round trip / 2 (양쪽 frame 작업 포함). p50만 표시하며 p95, p99는 `docs/distributed.md`에 있다.

| Mode | 8 KiB | 64 KiB | 1 MiB | 2 MiB | 16 MiB |
|------|------:|-------:|------:|------:|-------:|
| 보호 없음 (main layout) | 5.5 us | 11.1 us | 411 us | 1.040 ms | 16.39 ms |
| HMAC-SHA256 | 12.2 us | 64.1 us | 1.008 ms | 2.115 ms | 27.22 ms |
| AES-256-GCM (선택) | 7.4 us | 27.4 us | 422 us | 0.926 ms | 17.93 ms |
| ChaCha20-Poly1305 | 14.0 us | 75.4 us | 1.192 ms | 2.479 ms | 30.16 ms |

2 MiB에서 AES-256-GCM이 보호 없음보다 낮은 것은 loopback noise이며, 이 harness의 해당 크기 noise floor가 10 퍼센트 근처임을 보여 준다. Issue가 요구한 two-node link (Thunderbolt 또는 10/25 GbE)는 측정하지 않았다.

### 3.2 End-to-end decode

`tests/pipeline_remote_decode_bench.rs`: qwen3-0.6b-4bit, 두 stage와 coordinator가 한 process에서 loopback으로 연결, greedy 128 token, run당 warm-up과 7회 반복, 5 round interleave.

| Binary | Run별 median의 median | Run별 범위 |
|--------|----------------------:|-----------|
| main 4144ecfb, 보호 없음, 배열 인코딩 | 68.3 tok/s | 65.4 ~ 72.7 |
| main cfa14e7b (측정 시점 HEAD), 같은 flag | 68.7 tok/s | 60.7 ~ 69.7 |
| branch, compact 인코딩, codec 비활성 (일회용 build) | 138.8 tok/s | 136.5 ~ 146.7 |
| branch 최종, compact 인코딩, AES-256-GCM | 133.2 tok/s | 122.2 ~ 145.1 |

보호만의 차이는 -4.0 퍼센트로, 7.2 ~ 7.5 ms step 중 약 0.3 ms이며 분포가 겹친다. Codec 비활성 행은 commit하지 않은 환경 변수 hook을 넣은 binary에서 나왔고, commit된 코드에는 그런 스위치가 없다.

### 3.3 첫 결과의 원인 분석

Compact 인코딩 이전의 첫 interleave run은 71.3 대 63.5 tok/s (-11.0 퍼센트)였다. Frame trace에서 token당 frame 네 개, 그중 둘이 같은 logits를 담은 1.1 MB JSON 숫자 배열임을 확인했다 (final stage가 logits를 entry stage로 돌려보내고, entry stage가 coordinator로 전달). 그 2.2 MB에 대한 in-process AES-256-GCM은 token당 0.54 ms (seal 275 us, open 268 us, HMAC-SHA256은 1.77 ms)로 14 ms step의 약 4 퍼센트다. Framing만 있는 build (같은 handshake와 `ProtectedStream`, null codec)는 63.5 tok/s로, 보호 65.6, main 69.1과 비교해 빠르지 않았다. 따라서 plumbing 비용은 측정되지 않을 만큼 작고, -11 퍼센트의 나머지는 run 간 noise였다 (main만 해도 round에 따라 66 ~ 75 tok/s). 같은 flag로 다시 build한 main baseline이 orchestrator binary의 hash를 재현해 build flag 차이도 배제했다.

## 4. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 28 |
| 추가 line | +2993 |
| 삭제 line | -391 |
| 새 unit test | 21 (frame_protection 8, tcp transport proxy와 oversize 7, cluster_handshake 3, wire_bytes 3) |
| 새 ignored benchmark | 2 (`cluster_frame_bench`, `pipeline_remote_decode_bench`) |
| 새 직접 dependency | `subtle = "2"` (rustls를 통해 이미 transitive) |

### Module별 변경

| Module | 변경 |
|--------|------|
| `frame_protection.rs` (신규) | `FrameSuite`, `FrameChannel`, `SessionKeys::derive`, `FrameSealer` / `FrameOpener`, 타입이 있는 `FrameRejected`와 `SequenceExhausted` |
| `tcp_frame.rs` (신규) | 유일한 `write_frame` / `read_frame`을 가진 `ProtectedStream`, `serialize_message` / `deserialize_message`를 `tcp_transport.rs`에서 이동 |
| `cluster_handshake.rs` | hello의 suite byte, `REJECT_FRAME_SUITE`, `FrameSuiteMismatch`, `SessionKeys` 반환, `PROTOCOL_VERSION` 3, `*_handshake_on` 변형 |
| `cluster_auth.rs` | `ClusterAuth`가 `hkdf_extract`용 secret 보관 |
| `connection_pool.rs`, `tcp_transport.rs` | pool이 `ProtectedStream` 저장, 중복 frame writer 세 벌 제거, 거부된 frame을 payload 없이 `handshake_warn_allowed`로 로그 |
| `stage_assignment.rs` | `FrameChannel::Assignment`에서 공용 codec의 `Control` frame 사용, rate-limit된 거부 로그 |
| `pipeline/wire_bytes.rs` (신규), `remote_service.rs`, `activation_transfer.rs` | JSON 안의 base64 tensor bytes |
| `bench_activation.rs` | round-trip mode, `FrameProtectionMode`, `measure_frame_protection_candidates` |
| `mlx_server.rs`, `router_front.rs`, `docs/distributed.md` | "암호화하지 않음" 같은 낡은 서술 제거, threat model 절, 측정값과 결정 기록 |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `6a1620d6` | test | add remote pipeline decode throughput bench |
| `348063d0` | feat | protect every cluster frame with per-connection AEAD |
| `1bbf41b5` | perf | encode stage protocol tensor bytes as base64 in JSON |
| `5a4864ee` | docs | record the final frame protection measurements |
| `32ccd72f` | docs | drop stale no-encryption claims after frame protection |
| `ec5fdc91` | docs | fix stale per-frame claim in router comment |
| `4f42196f` | fix | separate frame key channels and harden review findings |
| `69a32b28` | merge | origin/main을 feature branch에 병합 |

## 5. 호환성

- **Breaking**: 이 PR이 들어간 build와 들어가지 않은 build는 cluster를 구성할 수 없다. 인증된 peer는 명시적인 version 메시지로 handshake에 실패한다 (`PROTOCOL_VERSION` 3). base64 tensor 인코딩은 같은 version 상향 뒤에 있는 별도 wire 변경이다. Main의 모든 distributed role이 이미 `auth`를 설정하므로, 버전이 섞이면 stage message를 parse하기 전에 version 검사에서 걸린다. 버전 혼용은 원래 지원하지 않았다.
- **설정 변경 없음**: 기존 secret 소스 (`--cluster-secret-file`, `MLXCEL_CLUSTER_SECRET_FILE`, `MLXCEL_CLUSTER_SECRET`, `[cluster] secret_file`)가 유일한 입력이다. 새 flag나 환경 변수는 없다.
- **Single-node serving**: 변화 없음. `auth`가 없으면 평문 frame layout을 쓴다.

## 6. 검증

- Test-first: `src/distributed/tcp_transport_frame_tests.rs`의 proxy test 다섯 개 (`tampered_frame_is_rejected`, `replayed_frame_is_rejected`, `reordered_frames_are_rejected`, `frame_from_other_connection_is_rejected`, `activation_bytes_are_not_visible_on_the_wire`)는 main에 있는 API만 쓰고 trailer 길이를 calibration frame에서 알아내므로, 수정 전 transport에서 먼저 돌렸다. 다섯 개 모두 그곳에서 실패했고 (변조된 frame이 `inuegrity`로 도착, 나머지는 `recv()`에 도달, 패턴이 읽힘) branch에서는 통과한다. 두 개가 더 `ProtectedStream`과 transport에서 oversize header를 다룬다.
- 최종 head의 `distributed::` lib module 전체: 1433 passed, 0 failed, 4 ignored (CUDA build, `--test-threads=1`, `gpu-lock` 아래).
- GB10 실제 모델 remote pipeline parity (`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`): qwen3와 qwen3.5 test 통과. llama, gemma3, gemma4 remote parity는 main에서도 독립적으로 실패하므로 (#2654) 실행하지 않았다.
- `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --tests -- -D warnings`와 `cargo fmt --check` 통과.
- Commit `4f42196f`는 측정 이후에 들어왔다. Key 유도 (info 문자열 변경), 로그, test만 건드리고 frame별 seal/open 경로와 인코딩 hot path는 그대로이므로 수치를 다시 재지 않았다.

## 7. 잔여 위험과 후속 조치

### 잔여 위험 (`docs/distributed.md`에 문서화)

- 경로상 공격자는 여전히 frame을 버리거나 지연시킬 수 있다. Frame 길이, 종류, timing이 보이므로 관찰자가 decode step 수를 셀 수 있다.
- Secret을 가진 host는 정식 member다. Frame별 보호는 member를 제약하지 않는다.
- `len`이 `max_frame_size` (기본 256 MiB) 이내인 frame은 tag 검사 전에 할당되고 읽힌다. Handshake를 마친 peer만 도달할 수 있고 0으로 채운 buffer는 lazy하게 commit되며, main과 같은 할당이다.
- `hkdf::Prk`와 ring의 key schedule은 지워지지 않는다 (ring의 한계). 유도된 raw key bytes는 key 생성 직후 0으로 지운다.

### 후속 조치

- [ ] Two-node link (Thunderbolt 또는 10/25 GbE) 측정. Issue의 acceptance 목록이 요구하지만 benchmark는 양쪽을 loopback에 bind하므로, 두 번째 host의 transport를 상대로 round-trip mode의 `measure_activation_transfer`를 돌려야 한다.
- [ ] 중복 logits 전송 제거: final stage가 logits를 entry stage로 돌려보내고 entry stage가 coordinator로 전달한다. Final stage의 직접 응답이나 JSON 대신 binary frame을 쓰면 남은 token당 byte와 그에 따른 AEAD 작업이 대략 절반이 된다.
- [ ] Main에서부터 있던 문제: TCP `rpc_call` 응답이 pooled outbound stream으로 돌아오는데 아무도 그것을 읽지 않는다. 응답 경로는 보호되지만 쓰이지 않는다.

## 8. 학습 포인트

### 8.1 Reliable stream 위의 암묵적 sequence AEAD

순서가 보장되는 reliable byte stream에서는 양쪽이 추적하는 방향별 counter가 wire byte도 replay window도 없이 유일한 nonce와 재전송/재정렬 탐지를 동시에 준다. Counter가 nonce의 일부이므로 재전송, 삭제 후 주입, 재정렬 같은 모든 이탈은 그냥 tag 실패가 된다. 이 방식이 요구하는 제약: key는 connection과 방향마다 유일해야 하고 (양쪽 handshake nonce를 salt로 쓰는 HKDF), counter는 write 전에 증가해 부분적으로 쓰인 frame이 같은 stream에서 재시도되지 않아야 하며, send가 실패하거나 취소된 stream은 pool로 돌려보내지 말고 버려야 한다.

```rust
// FrameSealer::seal, AEAD 분기 (frame_protection.rs)
let (header, payload) = frame.split_at_mut(FRAME_HEADER_LEN);
let tag = key
    .seal_in_place_separate_tag(aead_nonce(seq), aead::Aad::from(&*header), payload)
    .expect("AEAD seal cannot fail for an in-range payload");
frame.extend_from_slice(tag.as_ref());
self.seq += 1;
```

### 8.2 Hardware AES는 MAC보다 빠를 수 있다

"MAC만" 하는 것이 "암호화와 MAC"보다 싸다는 직관이 이 hardware에서는 맞지 않았다. ring의 AES-GCM은 ARMv8 AES와 PMULL 명령어를 쓰고 SHA-256 경로는 이 core에서 더 느려, AES-256-GCM이 측정한 모든 payload 크기 (8 KiB ~ 16 MiB)에서 HMAC-SHA256보다 1.5 ~ 2.4배 빨랐다. "더 가볍다"는 이유로 정한 보안 대안은 대상 hardware에서 측정한 뒤에 선호해야 한다.

### 8.3 암호를 최적화하기 전에 원인을 분해하라

첫 end-to-end 수치 (-11 퍼센트)는 AEAD 문제처럼 보였다. Frame trace는 token당 byte가 두 번 전송되는 JSON 숫자 배열에 지배된다는 것을 보여 줬고, null codec build가 framing 비용과 암호 비용을 분리했다. 가장 효과가 큰 수정은 cipher가 아니라 serialization에 있었다. 같은 trace는 묶인 PR의 구성 요소를 따로 재야 하는 이유도 보여 준다. Headline +94 퍼센트는 보호의 비용에 대해 아무것도 말해 주지 않는다.

### 핵심 키워드

| 키워드 | 관련성 |
|--------|--------|
| HKDF (RFC 5869) | Extract-then-expand key 유도. 여기서는 공유 secret 하나와 handshake nonce 두 개를 방향별, channel별 key 네 개로 바꾼다 |
| AEAD, associated data | AES-256-GCM이 평문 header를 AAD로 인증하므로 `len`을 복호화 전에 검사하면서도 위조할 수 없다 |
| Nonce 유일성 | GCM은 (key, nonce) 쌍이 반복되면 깨진다. 암묵적 counter와 connection별 방향 key가 반복이 없음을 보장한다 |
| serde `with` module, `is_human_readable` | 하나의 annotation으로 JSON에서는 base64, binary format에서는 raw bytes를 쓴다 |

### 관련 PR과 이슈

- Issue #2656: 이 설계와 결정 규칙
- Issue #2637 / PR #2650: 이 PR이 기반으로 삼는 connect 시점 cluster handshake
- Issue #2655 / PR #2674: pending-handshake pool, rejection record, disaggregated role 인증
- Issue #2654: GB10에서 원래부터 실패하는 llama/gemma3/gemma4 remote parity
