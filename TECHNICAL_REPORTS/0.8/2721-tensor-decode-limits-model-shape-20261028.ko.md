# PR #2721: 텐서 디코드 한도를 모델 shape에 맞게 조이기

**날짜**: 2026-10-28 (머지 전 작성, PR은 draft 상태로 열려 있음)
**상태**: aaa4daa7에서 구현 완료. 구현 리뷰가 HIGH 항목 하나(오래된 테스트 기대값)를 찾았고 a91c64a8에서 고침. 보안 및 성능 리뷰는 CRITICAL, HIGH, MEDIUM 이슈를 찾지 못함. aaa4daa7에서 GB10(spark-102) 로컬 게이트 통과. 2f137091에서 main을 머지함(#2719, #2720 포함). 작성 시점에 머지된 트리에 대한 사전 머지 게이트와 호스팅 CI가 진행 중
**언어**: Rust, Markdown
**위험도**: 중간 (프레임 포맷은 그대로지만, peer가 보내는 파이프라인 디코드가 모두 이전에는 받아들이던 프레임을 거부하고, inbound activation 폭을 정할 수 없으면 stage 로드가 실패하고, 빈 shape의 원소 수가 0에서 1로 바뀌고, `reassemble_layer_payload`와 `LayerTransferHeader`의 시그니처가 바뀜)

## 요약

PR #2721(#2716 해결)은 파이프라인 수신 측이 자신이 기대하는 것을 명시하게 만든다. 이전에는 원격 stage나 코디네이터가 디코드하는 모든 텐서가 송신 측이 쓴 헤더로만 제한되었다. 그래서 인증은 되었지만 침해된 peer가 약 21 MiB 프레임을 보내 4 GiB로 압축 해제하게 하고(그다음 MLX 배열로 한 번 더 복사됨), stage executor에 임의의 rank, batch, 시퀀스 길이, 폭을 가진 텐서를 넘길 수 있었다. 이 PR은 헤더 단계의 한도 검사(`TensorDecodeLimits`, `deserialize_tensor_with_limits`가 `TensorHeader::decode` 직후, payload를 자르거나 복사하거나 압축 해제하거나 할당하기 전에 검사)를 추가하고, activation 경로가 COMPRESSED와 QUANTIZED 플래그를 무조건 거부하게 하며, 한도를 로드된 모델에서 도출한다. hidden state는 `[1, n <= 2^20, width]`여야 하고 `width`는 stage의 inbound activation 폭이다. entry token id는 int32 `[1, n <= 2^20]`, mask는 rank 2에서 4이며 모든 축이 2^20 이하, 코디네이터가 받는 logits는 `[1, <= input_seq_len, vocab_size]`이다.

폭은 단순히 `hidden_size`가 아니다. per-layer input을 쓰는 Gemma 4 체크포인트(E2B, E4B)는 남은 모든 레이어의 per-layer input을 stage 경계 activation에 붙여 보낸다. 그래서 이 PR은 stage executor trait에 `inbound_activation_width`를 추가하고 Gemma 4에서 이를 override하며, `unpack_stage_hidden`도 같은 메서드를 쓰게 해서 패킹 폭 계산이 한 곳에만 있게 했다.

#2716의 나머지 다섯 항목은 같은 패스에서 고친 낮은 심각도의 wire 결함이다. 스트리밍 KV 레이어 헤더에 `original_bytes`가 생기고 홀수 바이트 수에서 원소 수를 잘못 세던 문제가 사라지며, 양자화되지 않은 레이어는 정확히 그 바이트 수로 디코드되어야 하고, 빈 shape는 MLX처럼 원소 1개인 스칼라가 되고, 디코더가 곧 압축 해제할 payload를 미리 복사하지 않으며, `MAX_DECODED_TENSOR_BYTES`가 `u64`가 되어 32비트 대상에서도 const 평가가 된다. 모든 항목은 인증된 클러스터 멤버가 있어야 하므로, 인증되지 않은 구멍을 막는 수정이 아니라 침해된 peer에 대한 방어 강화이다.

## 1. 문제 정의

#2701(이슈 #2691)은 텐서 wire 포맷에서 peer가 정하는 모든 크기를 검사된 값으로 바꿨지만, 남은 한도는 송신 측이 스스로 선언한 값이었다. #2716은 #2701 보고서 7절의 잔여 문제 여섯 가지를 정리했고, 모두 main 752db4f9에서 다시 확인했다.

### 1.1 Activation 디코드가 송신 측 헤더를 믿었다 (중간)

`deserialize_tensor`의 유일한 프로덕션 호출자는 파이프라인 activation 경로(`deserialize_wire_tensor`에서 `ActivationMessage::deserialize_activation`)였다. 정직한 송신자는 COMPRESSED나 QUANTIZED를 설정하지 않지만 디코더는 둘 다 따랐다. 헤더가 자기 일관적이면 압축 해제 상한은 `min(헤더 한도, MAX_DECODED_TENSOR_BYTES)`였고, `decompress`는 블록의 최대 256배까지 받아들인다. 그래서 약 16 MiB LZ4 블록(base64 후 약 21 MiB, 256 MiB TCP 프레임 상한보다 훨씬 작음)이 4 GiB 버퍼를 만들고, `reconstruct_mlx_array`가 이를 MLX로 다시 복사했다. 프레임 하나에 약 8 GiB를 할당하는 셈이다. 텐서가 executor에 도달하기 전에 rank, batch, 시퀀스 길이, 폭을 검사하는 곳이 없었다. peer가 보내는 디코드 지점은 entry stage의 `input_ids`와 attention mask, non-entry stage의 hidden state와 mask, 그리고 코디네이터의 logits였고, logits는 전체 디코드가 끝난 뒤에야 `validate_logits_shape`가 검사했다.

### 1.2 스트리밍 KV 원소 수 (낮음)

`prepare_layer_payload`는 `keys.len() / 2 + values.len() / 2`로 원소 수를 셌다. key와 value의 바이트 수가 모두 홀수인 1바이트 dtype이면 텐서마다 원소 하나씩 모자라므로, 압축 해제를 `original_num_elements * 2`로 제한하던 level None 수신 측이 정상 레이어를 2바이트 초과로 거부했다. 양자화기는 이어 붙인 데이터에서 `(k + v) / 2`로 세므로, 같은 오차가 Int8과 Int4에서 헤더와 양자화 레이아웃을 어긋나게 했다.

### 1.3 압축되지 않은 level None 길이 미검사 (낮음)

`reassemble_layer_payload`는 압축되지 않은 level None 레이어에 대해 헤더와 아무것도 비교하지 않고 `wire_data.to_vec()`를 반환했다.

### 1.4 빈 shape 의미가 서로 달랐다 (낮음)

`TensorHeader::num_elements`와 `validate_data_size`는 `[]`를 원소 0개로 봤지만, wire 경로 끝의 `validate_raw_tensor`는 MLX처럼 1부터 곱한다. 0바이트짜리 `[]` 헤더는 `deserialize_tensor`를 통과하고 나중에야 실패했으며, 인코더는 실제 2바이트 float16 스칼라를 거부했다. `validate_data_size`의 overflow 메시지에는 줄 이음 백슬래시가 빠져 공백 18개가 들어가 있었다.

### 1.5 압축 해제 전 복사 (낮음)

`deserialize_tensor`는 압축 해제나 역양자화 전에 `to_vec()`으로 payload를 복사했으므로, 플래그가 있는 프레임마다 전체 크기 복사가 한 번 더 일어났다.

### 1.6 32비트 이식성 (낮음)

`pub const MAX_DECODED_TENSOR_BYTES: usize = 1 << 32`는 32비트 `usize`에서 const 평가 overflow이므로, 원리적으로도 그 대상에서 크레이트가 컴파일되지 않았다.

## 2. 변경 요약

| 항목 | 영역 | 변경 |
|---|---|---|
| 1 | `tensor_serialize.rs`, `distributed/mod.rs` | `DimBound { Exact, AtMost }`, `ShapeBound { Axes, Uniform { rank, max_dim } }`, `unrestricted()`와 `check_header`를 갖는 `TensorDecodeLimits { allow_compressed, allow_quantized, dtypes, shape }`. `deserialize_tensor_with_limits`는 `TensorHeader::decode` 직후 `check_header`를 호출하고, `deserialize_tensor`는 `unrestricted()`로 이를 감싼다. 오류 메시지는 플래그, dtype 목록, 또는 축과 그 값, 한도를 명시한다. 모두 `deserialize_tensor` 옆에서 re-export된다. |
| 1 | `pipeline/activation_transfer.rs` | `deserialize_activation_with_limits`는 호출자가 무엇을 넘기든 `allow_compressed`와 `allow_quantized`를 끈다. `deserialize_activation`이 이를 감싸므로 in-process 경로도 두 플래그를 거부한다. |
| 1 | `pipeline/wire_tensor.rs` | `MAX_STAGE_TOKENS_PER_FRAME = 1 << 20`, `hidden_states()`, `input_ids()`, `attention_mask()`, `logits(input_seq_len)`을 갖는 `StageWireLimits { activation_width, vocab_size }`, 그리고 `deserialize_wire_tensor_with_limits`. 폭이나 vocab이 `None`이면 그 축은 `AtMost(u64::MAX)`가 된다. `deserialize_wire_tensor`는 제한 없는 in-process 진입점으로 남는다. |
| 1 | `pipeline/stage_executor/`, `models/gemma4.rs`, `loading/mod.rs` | `StageExecutor`와 `FamilyStageExecutor`에 기본값 `None`인 `inbound_activation_width()`. `LoadedStageExecutor`는 family override 또는 `loading::read_hidden_size`(`text_config` 먼저, 그다음 최상위)를 저장하고, 둘 다 없으면 로드에 실패한다. `Gemma4StageModel::inbound_activation_width`는 per-layer 필드가 0이 아니면 `hidden_size + (num_hidden_layers - layer_range.start) * hidden_size_per_layer_input`을 반환하고, `unpack_stage_hidden`이 이 값으로 자른다. |
| 1 | `pipeline/remote_service.rs` | `RemoteStageService::load`가 executor의 폭으로 `PeerDecodeLimits { hidden_states, input_ids, attention_mask }`를 한 번 만들고, peer가 보내는 네 디코드 지점(`handle_run_entry`의 id와 mask, `handle_activation`의 hidden state와 mask)에서 쓴다. |
| 1 | `pipeline/runtime.rs` | `RemotePipelineRuntime`은 `RunResult` logits를 자신의 `vocab_size`와 `logits(max(input_seq_len, 1))` 아래에서 디코드하고, 정확한 "입력 길이 또는 1" 규칙을 위해 `validate_logits_shape`를 유지하며, 제한된 디코드가 실패하면 이제 `cleanup_request`를 호출한다. |
| 1 | `kv_cache_transfer/streamed.rs` | `reassemble_layer_payload`가 `max_layer_bytes`를 받고, 압축 해제 전에 `header.original_bytes > max_layer_bytes`를 거부한다. doc comment에 수신 측 도출식 `2 * n_kv_heads * head_dim * element_size * token_budget`을 적었다. |
| 2 | `kv_cache_transfer/{mod,streamed,parallel}.rs` | `LayerTransferHeader.original_bytes`(`serde(default)` 없음, 이 헤더에는 프로덕션 수신자가 없음)를 두 송신자가 채운다. `num_elements = raw_bytes.len() / 2`. Int8과 Int4는 홀수 길이 레이어를 거부한다. Int8과 Int4 수신 측은 압축 해제나 역양자화 전에 `original_num_elements * 2 == original_bytes`를 요구한다. |
| 3 | `kv_cache_transfer/streamed.rs` | 새 `reassemble_unquantized`: level None은 압축 여부와 무관하게 정확히 `original_bytes`를 내야 한다(압축이면 압축 해제 상한을 그 값으로, 아니면 복사 전에 길이 비교). |
| 4 | `tensor_protocol.rs`, `tensor_serialize.rs` | `TensorHeader::shape_num_elements(&[u64])`가 1부터 곱하므로 `[]`는 원소 1개이다. `num_elements`와 `validate_data_size`가 이를 공유한다. 공백 18개 메시지를 고쳤다. |
| 5 | `tensor_serialize.rs`, `kv_cache_transfer/streamed.rs` | payload를 `Cow<[u8]>`로 다룬다. 압축 해제와 역양자화는 빌린 슬라이스를 읽고, 플래그가 없는 payload만 `into_owned`로 한 번 복사된다. 양자화된 KV 경로도 압축되지 않은 wire 바이트를 같은 방식으로 읽는다. |
| 6 | `tensor_protocol.rs` | `MAX_DECODED_TENSOR_BYTES: u64`와 `usize::MAX`로 포화하는 `max_decoded_tensor_bytes() -> usize`. `tensor_serialize.rs`와 `streamed.rs`에서 쓴다. |
| 문서 | `docs/distributed.md` | "What the secret protects" 목록에 수신 측 한도를 설명했다. |

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 25개 (소스 14, 테스트 10, 문서 1) |
| 추가 / 삭제 줄 | main 대비 +1362 / -101 |
| 추가된 테스트 | 7개 파일의 `#[test]`/`#[tokio::test]` 함수 21개(새 파일 `tensor_serialize_limits_tests.rs`, `gemma4_stage_width_tests.rs` 포함). 기존 테스트 3개 조정(`test_header_num_elements_empty_shape_is_zero`를 `..._is_one`으로 변경, `forward_rejects_logits_with_wrong_vocab_dimension`, `validate_logits_shape` 케이스 목록) |
| 커밋 | 27ac0f02 (여섯 항목), a91c64a8 (logits 테스트 기대값), aaa4daa7 (peer 한도 사전 계산, 문서, `validate_logits_shape` 케이스), 2f137091 (main 머지) |

main 머지(2f137091)로 #2719와 #2720이 들어왔다. #2720은 `RemoteStageService::handle_activation`의 실패 경로를 다시 작성했다. git은 `remote_service.rs`를 충돌 없이 머지했고, 충돌은 두 PR이 모두 파일 끝에 테스트를 추가한 `remote_service_tests.rs`에서만 났다. 해결은 #2716의 한도 테스트 두 개와 #2720의 오류 라우팅 테스트 두 개를 모두 남겼다. 두 변경이 어떻게 맞물리는지는 5절에서 다룬다.

## 3. 기술적 선택과 그 이유

### 3.1 payload가 아니라 헤더를 검사한다

한도 검사는 디코드된 `TensorHeader`에 대해, payload 슬라이스가 생기기 전에 산술 없이 수행된다. 거부 비용이 0인 지점은 여기뿐이다. 그 뒤에는 디코더가 이미 peer가 주장한 크기를 복사하거나 압축 해제하거나 할당했을 수 있다. 테스트는 헤더만 있는 버퍼(`data_len`은 설정, payload 바이트 없음)를 보내 이 순서를 고정한다. 제한된 디코드는 "buffer too short"가 아니라 한도 오류로 실패해야 한다. "buffer too short"는 payload 길이를 먼저 봤다는 뜻이다.

구조는 세 층으로 나뉜다. `tensor_serialize.rs`는 파이프라인을 모르는 일반 어휘(`DimBound`, `ShapeBound`, `TensorDecodeLimits`)를 갖는다. `wire_tensor.rs`는 파이프라인 정책(`StageWireLimits`와 네 빌더)을 갖는다. `remote_service.rs`와 `runtime.rs`는 모델 사실(폭, vocab, 입력 길이)을 갖는다. 다른 종류의 텐서(KV, 가중치)를 받는 미래의 수신자도 첫 층을 자신의 정책과 함께 재사용할 수 있다.

### 3.2 증폭 상한을 조정하지 않고 증폭 자체를 없앤다

activation에 더 엄격한 압축 해제 상한을 두는 대신, `deserialize_activation_with_limits`는 넘겨받은 한도와 무관하게 COMPRESSED와 QUANTIZED를 거부하고, 이 거부는 `deserialize_activation`을 거치는 in-process 경로에도 적용된다. `serialize_activation`은 두 플래그를 설정한 적이 없으므로 정직한 프레임은 바뀌지 않는다. 두 플래그가 사라지면 activation 디코드가 할당하는 가장 큰 버퍼는 실제로 받은 바이트 수로 제한되므로, 256:1 비율은 옮겨지는 것이 아니라 사라진다.

### 3.3 컨텍스트 윈도가 아닌 프레임 단위 토큰 상한

`MAX_STAGE_TOKENS_PER_FRAME`은 `MAX_HANDOFF_PROMPT_TOKENS`와 같은 2^20이다. 컨텍스트 윈도도 검토했지만 버렸다. rope scaling을 쓰는 체크포인트는 정당하게 `max_position_embeddings`를 넘어 동작하므로, 컨텍스트 윈도 한도는 정직한 긴 프롬프트를 거부하게 된다. 토큰 상한은 터무니없는 shape를 막기 위한 것이고, 정확한 검사는 폭과 vocab 축이 맡는다.

### 3.4 폭은 executor에서 오고, 알 수 없는 폭은 로드를 실패시킨다

`config.json`의 `hidden_size`만 썼다면 정직한 Gemma 4 E2B/E4B 프레임을 모두 거부했을 것이다. `pack_hidden_for_downstream`이 남은 모든 레이어의 per-layer input을 hidden state에 붙이기 때문이다. trait 메서드는 family가 실제 inbound 폭을 밝힐 수 있게 하고, 기본값 `None`은 "`hidden_size`를 쓰라"는 뜻이다. PR의 family 감사 결과, 나머지 어댑터(llama, mistral, mixtral, qwen3, VLM 포함 qwen35, qwen3_next, gemma3, glm4 계열, glm_moe_dsa, deepseek_v3, gpt_oss, jamba, llama4, nemotron_h)는 모두 평범한 `[1, L, hidden_size]`를 보낸다.

`read_hidden_size`가 `text_config`를 먼저 읽는 이유는 VLM과 Llama 4 설정이 언어 모델을 그 아래에 두고, 최상위 `hidden_size`는 다른 tower를 설명할 수 있기 때문이다. family도 설정도 폭을 주지 못하면 `LoadedStageExecutor`는 아무 폭이나 조용히 받아들이는 대신 로드를 거부한다. 첫 조작된 프레임이 오는 런타임이 아니라 운영자가 보는 시작 시점에 닫힌 쪽으로 실패하는 것이다.

Gemma 4 공식은 한도와 `unpack_stage_hidden`이 모두 호출하는 `pub(crate)` 메서드 하나에만 있으므로, 패킹 레이아웃이 바뀌어도 수신 측 한도가 뒤처질 수 없다. `per_layer_input_inbound_width_matches_upstream_packing`은 이 값을 상위 stage가 실제로 내보내는 폭과 비교한다.

### 3.5 logits는 헤더 한도를 받고 정확한 규칙도 유지한다

코디네이터의 한도는 `[1, <= input_seq_len, vocab]`이며, 실제 규칙(시퀀스 축은 입력 길이 또는 1)보다 느슨하다. 헤더 검사는 할당을 막고, `validate_logits_shape`는 디코드 후 정확한 규칙을 계속 적용한다. 그 결과 `validate_logits_shape`는 이제 1과 입력 길이 사이에 엄격히 놓인 시퀀스 길이에서만 동작하며, aaa4daa7이 새 케이스로 이를 다룬다. PR은 제한된 디코드가 실패할 때 `cleanup_request`도 호출하게 해서, 거부된 `RunResult`가 요청의 stage 상태를 남기지 않게 했다.

### 3.6 한도는 stage마다 한 번 만든다

첫 구현은 프레임마다 `TensorDecodeLimits`를 만들어 토큰마다 작은 `Vec` 할당이 두세 번 생겼다. aaa4daa7은 이를 `RemoteStageService::load`에서 `PeerDecodeLimits`로 미리 계산한다. 코디네이터의 logits 한도는 요청의 입력 길이에 의존하므로 forward 호출마다 만들지만, payload 복사에 비하면 무시할 만하다.

### 3.7 빈 shape는 오류가 아니라 스칼라

`[]`를 거부하는 방안도 검토했지만 버렸다. wire 경로는 이미 `[]`를 스칼라로 읽는 `validate_raw_tensor`에서 끝나고 MLX도 그렇다. `TensorHeader`에 같은 의미를 적용하면 헤더와 최종 검사가 일치하고, 인코더가 실제 스칼라를 보낼 수 있으며, 0바이트 `[]`는 나중이 아니라 `deserialize_tensor`에서 일찍 실패한다.

### 3.8 스트리밍 KV는 정확한 바이트 수를 갖는다

`LayerTransferHeader`에 `original_bytes`를 넣으면 수신 측은 level None에 필요한 숫자(정확한 디코드 길이)와 Int8, Int4의 교차 검사(`elements * 2 == bytes`)를 얻는다. 이 헤더에는 프로덕션 수신자가 없으므로 `serde(default)`는 넣지 않았다. 프로덕션 분리형 handoff는 압축 없는 `kv_cache_serde`를 쓰고 각 slab을 이미 `ExpectedBlockGeometry`에 고정한다. `max_layer_bytes`가 상수가 아니라 매개변수인 이유는 기대 geometry를 아는 쪽이 수신 측뿐이기 때문이다.

## 4. 검증

### 4.1 실패 우선 증거

새 테스트는 모두 main에 없는 API를 대상으로 하므로, 브랜치에서 해당 가드를 752db4f9의 동작으로 되돌리는 mutation 검사로 각 테스트가 실패함을 보였다.

| 테스트 | Mutation | 실패 |
|---|---|---|
| `limits_refuse_flags_dtype_and_shape_before_the_payload`, `hidden_states_shape_is_checked_before_the_payload`, `entry_tensor_limits_bound_ids_and_masks`, `logits_limits_bound_sequence_and_vocab` | `check_header` 제거 | 한도 오류 대신 "buffer too short" |
| `activation_decode_refuses_compressed_and_quantized_frames` | 플래그 강제 해제 중단 | LZ4 확장 오류, 플래그 언급 없음 |
| `run_entry_with_out_of_bounds_ids_answers_limit_error`, `activation_outside_stage_limits_never_executes` | 호출 지점에 제한 없는 한도 | 한도 오류 없음, 또는 executor 실행 |
| `per_layer_input_inbound_width_matches_upstream_packing` | 폭이 per-layer input 무시 | 256 != 304 |
| `read_hidden_size_reads_text_config_then_top_level` | 최상위 먼저 | 1152 != 2560 |
| `reassemble_rejects_original_bytes_above_max_layer_bytes` | 한도 제거 | `decompress`에 도달 |
| `reassemble_round_trips_odd_byte_layer_compressed` | 텐서별 절반 계산 | 4095 != 4096 |
| `prepare_rejects_odd_byte_layer_for_quantization` | 홀수 검사 제거 | Ok |
| `reassemble_rejects_quantized_header_bytes_mismatch` | 바이트 검사 제거 | Ok, 512바이트 |
| `reassemble_rejects_uncompressed_length_mismatch` | 길이 검사 제거 | Ok, 511바이트 |
| `test_header_num_elements_empty_shape_is_one`, `float16_scalar_round_trips_and_empty_scalar_is_rejected` | `[]`는 원소 0개 | 0 != 1, 인코더가 스칼라 거부 |
| `decode_does_not_copy_the_payload_before_decompressing` | 압축 해제 전 `to_vec()` | assertion |
| `test_max_decoded_tensor_bytes_helper` | 상수를 다시 `usize`로 | E0308 |

### 4.2 로컬 게이트

aaa4daa7에서 GB10 spark-102, `--profile test-fast`, GPU 락 밖에서 빌드하고 `gpu-lock` 아래 `--test-threads=1`로 실행, 실제 모델 대상은 `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`:

| 대상 | 결과 |
|---|---|
| `cargo fmt --check`, `cargo clippy --features cuda,test-utils --lib --tests -D warnings` | 깨끗함 |
| `--lib distributed::` | 1512 통과, 4 무시 |
| `mlxcel-core from_bytes` | 6 통과 |
| `distributed_integration`, `pipeline_e2e` | 20, 30 통과 |
| `pipeline_remote_real_models --ignored` | 10 통과 (RDMA/Thunderbolt 포함 llama, gemma3, gemma4, glm4, gpt_oss, qwen3, qwen3.5; `glm_moe_dsa` 건너뜀) |
| `pipeline_server_remote_real_models --ignored` | 7 통과 (`glm_moe_dsa` 건너뜀) |
| `pipeline_ci_multi_stage_real_models`, `--ignored` 유무 각각 | 2 + 2 통과, 3호스트 parity 테스트 포함 |
| `disaggregated_e2e`, `disaggregated_serving_e2e`, `disaggregated_router_e2e` | 51, 2, 7 통과 |

gemma4 실행은 `gemma-4-e2b-it-4bit`(`hidden_size_per_layer_input = 256`)를 쓰므로, 실제 체크포인트에서 정직한 패킹 프레임이 per-layer 폭 한도를 통과함을 확인했다. 이것이 "정직한 프레임은 거부되지 않는다"는 인수 조건이다. `glm_moe_dsa`는 418.6 GB `glm5-4bit` 체크포인트가 필요해 128 GB 머신에서 로드할 수 없다. 32비트 빌드는 실행할 수 없으므로(mlxcel-core가 32비트에서 빌드되지 않음), 6번 항목은 32비트 컴파일이 아니라 상수가 이제 평가된다는 점과 헬퍼 테스트로 검증했다. 성능 측정은 하지 않았다. 토큰당 경로의 변화는 디코드마다 헤더 검사 하나, 플래그 있는 프레임의 복사 하나 감소뿐이다.

위 게이트는 main 머지 전에 실행했다. 머지된 트리(2f137091)에 대한 사전 머지 게이트는 이 보고서를 쓰는 시점에 진행 중이었다.

## 5. 남은 공백과 위험

- **새 한도에 거부된 forward activation은 여전히 코디네이터 타임아웃으로 끝난다 (LOW).** #2720을 머지한 뒤 `handle_activation`은 실행 실패와 하위 stage 전송 실패를 `report_activation_failure`로 보내며, 이 함수는 `RunEntry` exchange의 `Error`를 코디네이터에 보낸다. 그러나 `handle_activation` 맨 위의 두 제한 디코드(hidden state와 mask)는 `mark_request_started` 전에 여전히 `?`로 빠져나간다. 서비스 루프는 `dropping frame`을 로그로 남기고, 코디네이터는 `stage_timeout`을 다 기다린다. #2720 보고서가 기존 `deserialize_wire_tensor` 실패에 대해 적은 공백과 같지만, #2716은 여기에 도달하는 새 경로를 많이 더한다(한도를 벗어난 모든 shape, dtype, 플래그). entry stage는 영향이 없다. `handle_run_entry`는 거부된 `input_ids`나 mask에 즉시 `Error`로 답한다. `activation_outside_stage_limits_never_executes`는 두 결과를 모두 허용하므로 이 문제가 고쳐져도 알아채지 못한다. 수정 방법은 두 디코드를 `match`로 받아 오류와 함께 `report_activation_failure`를 호출하는 것이며, 이때 이전 단계에서 만든 요청의 캐시도 정리된다. 테스트는 그 뒤 activation의 nonce를 가진 `Error`를 요구해야 한다.
- **`DimBound::AtMost`는 0을 허용한다 (LOW).** `[1, 0, width]` hidden state와 `[1, 0]` input id가 한도를 통과해 executor에 도달한다. PR 이전부터 있던 동작이며, 시퀀스 축에 최소 1을 두면 막힌다.
- **mask 한도는 모든 축을 2^20으로 제한한다 (LOW).** key 축이 캐시된 토큰 1,048,576개를 넘는 명시적 mask는 거부된다. 현재 원격 파이프라인으로 mask를 보내는 서버 경로는 없다.
- **`validate_data_size`의 Int4 분기는 `num_elements.div_ceil(2) as usize`를 쓴다 (LOW).** 32비트 대상에서 잘릴 수 있다. PR 이전부터 있던 코드이고, activation 경로는 어차피 Int4를 거부한다.
- **`max_layer_bytes`는 실제 수신자로 검증되지 않았다.** `reassemble_layer_payload`에는 여전히 프로덕션 호출자가 없고, `LayerTransferHeader`에는 압축 플래그가 없어 미래의 스트리밍 수신자가 헤더로 `compressed` 인자를 정할 수 없다. 둘 다 #2716의 범위 밖이며, 그 수신자를 연결할 때 정해야 한다.
- **프로덕션 handoff는 여전히 `CacheIngestLimits::default()`(16 GiB 프레임)를 넘긴다.** #2716의 범위 밖이고 증폭은 없으며, 별도 후속 작업이다.
- **크레이트 내부의 깨지는 API 변경.** `reassemble_layer_payload`에 매개변수가 생겼고, `LayerTransferHeader`에 필수 필드가 생겼고, `MAX_DECODED_TENSOR_BYTES`의 타입이 바뀌었다. 트리 안의 호출자는 모두 고쳤지만, 이 public 항목을 쓰는 트리 밖 사용자는 같은 수정이 필요하다.

## 6. 학습 포인트

### 6.1 한도는 수신 측의 것이다

송신 측 헤더에서 나온 크기 검사는 헤더가 자기 일관적임만 보장할 뿐, 수신 측이 그만큼 큰 텐서를 원했는지는 말해 주지 않는다. 이 PR이 적용한 패턴은 고정 크기 헤더를 디코드하고, 수신 측이 자신의 상태(모델 폭, vocab, 입력 길이)로 아는 값과 비교한 뒤에야 payload를 건드리는 것이다.

```rust
let (header, header_len) = TensorHeader::decode(buf)?;
limits.check_header(&header)?; // 플래그, dtype, shape; 슬라이스나 할당 전에
```

### 6.2 패킹된 레이아웃의 공식은 하나만

송신 측이 텐서에 추가 feature를 붙이고 수신 측이 이를 풀면, 그 폭은 프로토콜의 사실이다. 언패커와 한도가 모두 호출하는 메서드 하나에서 계산하면, 나중에 레이아웃이 바뀌어도 정직한 프레임을 모두 거부하거나 크기가 틀린 프레임을 받아들이는 수신 측이 생기지 않는다.

### 6.3 적대적 경로에서는 아무도 쓰지 않는 기능을 없앤다

COMPRESSED와 QUANTIZED 플래그는 일반 텐서 포맷의 일부였지만, activation 송신자는 아무도 쓰지 않았다. 그 경로에서 두 플래그를 거부하자 플래그 검사 두 개로 증폭 문제 전체가 사라졌고, 이는 압축 해제 크기에 어떤 상한을 두는 것보다 단순하고 강하다.

### 6.4 조건부 소유권에는 `Cow`

디코더는 바이트를 변환할 때만 소유한 바이트가 필요하다. `Cow::Borrowed`로 시작해 압축 해제나 역양자화 후 `Cow::Owned`로 바꾸고 마지막에 `into_owned()`를 호출하면, 두 코드 경로 없이도 플래그 없는 payload는 한 번만 복사되고 변환 전에는 복사가 없다.

### 6.5 32비트를 넘는 크기 상수는 `u64`로

`1 << 32`인 `usize` 상수는 32비트 대상에서 컴파일 오류다. 상수를 `u64`로 선언하고 사용 지점에서 포화 헬퍼로 변환하면 64비트에서는 값이 정확하고 다른 대상에서도 빌드가 유효하다.

## 7. 참고 자료

- 이슈 #2716: tighten tensor decode limits to the model shape
- PR #2701 (이슈 #2691): 텐서 wire 디코드 크기 제한. 그 보고서 7절이 #2716의 출처
- PR #2720 (이슈 #2712): non-entry stage 실패를 `report_activation_failure`로 코디네이터에 보냄
- PR #2719 (이슈 #2715): 라우터 parity 잔여 문제, 2f137091에서 이 브랜치에 머지됨
- PR #2708 (이슈 #2694): 마지막 stage가 `reply_nonce`와 함께 `RunResult`를 코디네이터에 보냄
- #2650 (클러스터 secret), PR #2674와 #2684 (인증된 handshake, 프레임별 AEAD): 여기의 모든 항목이 인증된 멤버를 필요로 하는 이유
- #2654, #2671: 관련 분산 전송 강화 작업
