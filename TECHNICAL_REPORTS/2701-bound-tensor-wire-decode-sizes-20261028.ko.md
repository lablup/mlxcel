# PR #2701: tensor wire decode 크기 제한과 int16 decode 수정

**날짜**: 2026-10-28
**상태**: 검증 완료, PR은 draft로 열려 있음 (merge 전에 PR branch에서 작성한 보고서)
**언어**: Rust, C++
**위험도**: 낮음 (peer가 보낸 데이터를 decode하는 경로의 hardening, wire format과 cluster `PROTOCOL_VERSION`은 변경 없음, crate-public Rust signature 세 개 변경)

## 요약

Issue #2691은 #2671의 보안 리뷰에서 발견된, stage가 cluster peer로부터 받은 tensor를 decode하는 방식의 결함 세 가지를 모은 것이다. Quantized 또는 LZ4 압축 tensor header를 이용하면 송신자가 수신자의 할당 버퍼 크기를 정할 수 있었고 (8바이트 또는 13바이트 payload로 4 GiB, 더 큰 shape이면 process abort), 잘못된 quantized payload는 실패하지 않고 0으로 decode되었으며, C++ `from_bytes` bridge는 INT16과 UINT16을 원소당 1바이트로 decode했다. 이 PR은 peer가 정하는 모든 크기를 할당 전에 검증되는 checked 값으로 바꾼다. `TensorHeader::num_elements`는 overflow 시 실패하고, `dequantize_int8` / `dequantize_int4`는 payload layout이 정확히 일치해야 하며, `decompress`는 호출자 한도를 받아 할당 전에 두 prefix와 256:1 확장 상한을 검사하고, `deserialize_tensor`는 decode된 모든 tensor를 shape과 대조한다. Bridge에는 16비트 정수용 typed branch가 추가되고, bridge가 여전히 decode하지 못하는 FLOAT64와 COMPLEX64는 `validate_raw_tensor`가 거부한다. 정상 트래픽의 wire 형태는 바뀌지 않으며, 유일한 동작 변화는 잘못된 tensor가 이제 오류를 반환한다는 점이다.

## 1. 문제 정의

### 1.1 배경

Cluster frame은 #2650 / #2674의 handshake로 인증되고, #2684 이후로는 frame마다 AES-256-GCM으로 봉인된다. 이 계층들은 누가 frame을 보낼 수 있는지를 정할 뿐, frame이 수신자에게 무엇을 요구할 수 있는지는 제한하지 않는다. 침해된 member나 버그가 있는 peer는 여전히 임의의 header 필드로 tensor decoder에 도달한다.

Decoder는 정상 송신자가 만들 값이 아니라 header의 flag를 그대로 따른다. Pipeline activation은 `SerializeOptions::default()` (quantization 없음, 압축 없음)로 직렬화되므로 정상 stage 트래픽은 `deserialize_tensor`의 압축/quantized branch를 타지 않는다. 그러나 해당 flag bit를 켠 peer는 activation 경로에서 그 branch에 도달한다. `deserialize_wire_tensor`가 `ActivationMessage::deserialize_activation`을 거쳐 `deserialize_tensor`를 호출하기 때문이다. 공격 표면이 정상 실행에서는 한 번도 실행되지 않는 코드에 있었던 이유가 이것이다.

### 1.2 기존 문제점

- **Shape으로 크기를 정하는 quantized 출력** (MEDIUM): `num_elements`는 shape을 `checked_mul`로 접은 뒤 `unwrap_or(u64::MAX)`를 썼다. `deserialize_tensor`는 결과를 `as usize`로 캐스팅해 `dequantize_int8` / `dequantize_int4`에 넘겼고, 두 함수는 payload가 짧거나, group 수나 크기가 0이거나, shape이 요구하는 길이보다 짧으면 `vec![0u8; num_elements * 2]`를 반환했다. Shape `[1 << 31]`과 8바이트 payload를 가진 QUANTIZED int8 header는 4 GiB의 0으로 `Ok`를 반환했고, shape `[1 << 40]`은 할당 실패로 process를 abort시켰으며, 포화된 `u64::MAX`는 `num_elements * 2`에서 overflow했다. 잘못된 payload가 조용히 틀린 데이터가 되었다.
- **Unquantized payload를 shape과 대조하지 않음**: `deserialize_tensor` 안에서는 검사가 없었다. `serialize_tensor`는 송신 시 같은 규칙을 강제했지만 수신 시에는 아무도 강제하지 않았다.
- **LZ4 prefix를 신뢰** (MEDIUM): `decompress`는 바깥 u64 길이를 읽고 버린 뒤 `lz4_flex::decompress_size_prepended`를 호출했다. lz4_flex 0.11.6에서 이 함수는 한 바이트도 decode하기 전에 peer가 보낸 안쪽 u32로 `vec![0; n]`을 실행한다. `0xFFFF_FFFF`를 선언한 13바이트 payload 하나로 frame마다 4 GiB가 할당되었다.
- **16비트 정수 오decode** (LOW): `from_bytes`에는 32/64비트 정수와 16비트 float용 typed case만 있었고 나머지는 `case 1: default:`로 갔다. 이 경로는 `uint8_t*`로 array를 만들고 각 바이트를 cast한다. INT16 (code 6)과 UINT16 (code 2)은 버퍼 앞쪽 절반으로 틀린 값을 만들었고, FLOAT64 (11)와 COMPLEX64 (13)도 같은 방식으로 잘못 decode되었다. `validate_raw_tensor`가 바이트 수를 먼저 검사하므로 over-read는 없었지만 tensor는 틀렸다. Int16 MLX array 하나가 stage 경계를 넘는 것만으로 발생했다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| Member가 frame마다 수 GiB 할당을 강제 (메모리 압박, OOM kill) | 높음 | 낮음 (인증된 peer 필요) |
| 조작된 shape으로 수신 process abort | 높음 | 낮음 |
| 잘못된 quantized payload가 0으로 수용되어 틀린 activation이나 KV entry로 전파 | 중간 | 낮음 |
| int16 / uint16 tensor가 stage 경계에서 조용히 손상 | 중간 | 낮음 (stage activation은 보통 float16, bfloat16, float32) |

## 2. 기술적 선택과 그 이유

### 2.1 정확한 layout을 먼저 검증하고 할당

**컨텍스트:** 기존 dequantizer는 saturating 연산으로 하한 (`payload.len() >= required_len`)만 검사했고, 실패하면 항상 0 버퍼로 대체했다.

**고려한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| 하한 검사 유지, 0 대신 `Err` 반환 | diff가 작음 | 뒤에 붙은 여분 바이트를 여전히 허용, 출력 크기는 여전히 header만으로 결정 |
| 출력을 고정 크기로 제한 | 상수 하나 | 정상적인 큰 tensor도 실패할 수 있고, 작은 frame이 여전히 상한만큼 요구 가능 |
| **선택: layout 정확 일치** (`group_size >= 1`, `num_groups == ceil(n / group_size)`, `len == 8 + 2 * num_groups + data_len`, 모두 checked) | 출력이 실제 수신 바이트에 묶임: int8은 payload의 최대 2배, int4는 최대 4배 | 인코더가 만들지 않는 여분 바이트가 있는 payload는 모두 거부 |

**선택 이유:** payload 길이가 `num_elements`로부터 계산한 layout과 정확히 같으면, `num_elements * 2` 할당은 peer가 실제로 보낸 크기의 고정 배수가 되므로 dequantization에는 별도 크기 상한이 필요 없다. 공용 `validate_layout`이 두 dequantizer가 쓰는 `QuantizedLayout` (`num_groups`, `group_size`, `data_start`)을 반환하고, loop의 slice index는 검증된 영역 안에 머문다. Int4 경로는 `num_elements` 바이트 크기의 중간 `nibbles` vector를 더 이상 만들지 않고 packed 바이트에서 nibble을 직접 읽는다.

**트레이드오프:** `dequantize_*`가 `Vec<u8>`에서 `Result<Vec<u8>>`로 바뀌어 모든 호출자가 오류를 전파해야 했다. PR 리뷰의 호출자 조사에서 호출자는 세 곳 (`tensor_serialize.rs`, `kv_cache_transfer/quantized.rs`, `kv_cache_transfer/streamed.rs`)이었고 모두 수정되었다.

### 2.2 LZ4: 버퍼 할당 전 검사, 이후 `decompress_into`

`decompress(data, max_len)`의 순서는 다음과 같다.

1. 12바이트 prefix를 요구한다 (`split_first_chunk`, `unwrap` 없음).
2. 바깥 u64와 안쪽 lz4_flex u32가 같아야 한다 (`LZ4 size prefix mismatch`).
3. `declared <= max_len`이어야 한다 (`decompressed size {declared} exceeds limit {max_len}`).
4. `declared <= body.len() * 256` (saturating)이어야 한다.
5. 정확히 `declared` 바이트를 할당하고 `lz4_flex::block::decompress_into`로 decode하며, 다른 길이를 쓴 block은 거부한다.

`decompress_size_prepended` 대신 block 본문에 `decompress_into`를 호출하는 것이 할당을 우리 통제 아래로 가져오는 핵심이다. 라이브러리 함수는 우리 검사가 실행되기 전에 안쪽 prefix로 할당한다. 256:1 상한은 LZ4 block format에서 나온다. Match 길이 255바이트마다 최소 1바이트를 쓰므로, 이 상한은 정상 데이터를 거부하지 않으면서 할당을 수신한 block에 묶는다. `decompress_round_trips_64_mib_of_zeros` 테스트는 가장 압축이 잘 되는 입력으로 이를 고정하고, `max_len - 1`이 거부되는지도 확인한다.

Prefix 일치를 요구하는 데에는 알아 둘 부수 효과가 있다. `compress`는 안쪽 prefix를 u32를 저장하는 `compress_prepend_size`로 쓴다. 4 GiB 이상의 입력은 안쪽 prefix가 잘리는데, 이 PR 전에는 수신자가 잘린 tensor를 조용히 decode했고 이제는 prefix mismatch를 보고한다. 그만큼 큰 정상 tensor는 현재 압축되지 않으므로 실제로 발생한 적은 없다.

### 2.3 Header에서 유도한 한도, 그 위에 4 GiB 상수

`deserialize_tensor`는 `min(max_decoded_payload_len(header), MAX_DECODED_TENSOR_BYTES)`를 넘긴다. Header 상한은 unquantized tensor의 경우 정확한 데이터 크기 (Int4는 `div_ceil(2)`), quantized의 경우 가능한 가장 큰 유효 layout이다. `int8_payload_max_len(n) = 8 + 3n`, `int4_payload_max_len(n) = 8 + 2n + ceil(n / 2)`이며 모든 `group_size >= 1`을 포함한다. 같은 두 helper가 streamed KV 경로의 `reassemble_layer_payload`도 제한하므로 상한 정의는 한 곳에만 있다.

`MAX_DECODED_TENSOR_BYTES = 1 << 32`는 안쪽 prefix의 u32 한계와 같다. 위 2단계가 이미 `declared <= u32::MAX`를 강제하므로 현재 이 상수가 단독으로 걸리는 경우는 없다. 그래도 `MAX_METADATA_LEN` 옆에 이름 있는 명시적 상한으로 둔 이유는, 나중에 prefix 형식이 바뀌어도 한도가 조용히 풀리지 않게 하고 호출자가 인용할 단일 값을 주기 위해서다. 실제 방어선은 header 상한과 256:1 규칙의 조합이다. Header가 일관되면 stage는 약 16 MiB frame으로 여전히 약 4 GiB까지 할당할 수 있으며, 보안 리뷰는 이를 issue의 상한 내에서 수용된 것으로 기록했다.

### 2.4 `deserialize_tensor` 끝에서 `validate_data_size` 재사용

`serialize_tensor`는 이미 shape과 dtype에 맞지 않는 버퍼를 거부했다. 압축 해제와 dequantization 후 같은 `validate_data_size`를 호출하면 decode가 encode와 대칭이 된다. `deserialize_tensor`를 나가는 모든 tensor는 quantized 여부와 무관하게 shape과 정확히 일치한다. 두 번째 검사를 새로 쓰지 않고 helper를 재사용해 "올바른 크기"의 정의 (packed Int4 규칙 포함)를 하나로 유지한다. 인코더가 동일한 규칙을 강제하므로 인코더가 만드는 모든 tensor는 계속 round-trip된다.

### 2.5 int16은 C++에서 수정, FLOAT64 / COMPLEX64는 Rust에서 거부

`from_bytes`는 cxx를 통해 `UniquePtr<MlxArray>`를 반환하며 오류 채널이 없다. 잘못 decode되는 dtype에 대해 선택지는 두 가지였다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| Bridge signature를 `Result` 반환으로 변경 | 거부 지점이 한 곳 | peer 데이터를 보지 않는 checkpoint loader를 포함해 모든 `from_bytes` 호출자 수정 필요 |
| **선택: code 2와 6에 typed branch 추가, 11과 13은 `validate_raw_tensor`에서 거부** | int16 / uint16이 기존 `make_array_typed` helper로 bit-exact decode됨, FLOAT64 / COMPLEX64는 peer 데이터가 지나는 유일한 경로에서 거부 | 향후 `validate_raw_tensor` 없이 `from_bytes`에 도달하는 호출자가 생기면 FLOAT64 / COMPLEX64는 여전히 틀림 |

Fallthrough는 이제 명시적 (`case 0: case 1: case 5: default:`)이며 1바이트 dtype만 도달할 수 있다는 주석이 붙었다. Peer가 도달 가능한 모든 `from_bytes` 호출은 `validate_raw_tensor`를 먼저 실행하는 `reconstruct_mlx_array`를 거친다 (KV cache layer, paged block, `deserialize_wire_tensor`). 나머지 호출자는 고정되었거나 파일에서 유도된 dtype으로 로컬 checkpoint를 읽는다. `mlx_dtype_element_size`는 code 11과 13의 폭을 계속 알고 있으므로 다른 곳의 크기 계산은 바뀌지 않는다.

### 2.6 Wire와 protocol 변경 없음

모든 검사는 기존 layout `[u64 LE len][u32 LE len][LZ4 block]`과 `[u32 num_groups][u32 group_size][f16 scales][values]` 위에서 수신자 측에서 실행된다. `PROTOCOL_VERSION`은 #2684가 둔 값 그대로이며, 인코더가 만드는 모든 tensor에 대해 main의 peer와 이 branch의 peer가 상호 운용된다.

## 3. 변경 후 decode 경로

```
frame (AEAD 해제, #2684)
  -> base64 tensor 필드 (wire_bytes.rs, 256 MiB frame 상한으로 제한, 변경 없음)
  -> deserialize_tensor
       header.num_elements()?            overflow -> Err
       usize::try_from                   `as` 캐스팅 없음
       [compressed]  decompress(payload, min(header 상한, 4 GiB))
                       prefix 일치, <= max_len, <= 256 x block, 정확한 decode 길이
       [quantized]   dequantize_int8/int4  할당 전 layout 정확 일치
       validate_data_size(dtype, shape, payload)
  -> reconstruct_mlx_array
       validate_raw_tensor                dtype 11/13 -> Err, 정확한 바이트 수
  -> ffi::from_bytes                      u16/i16 typed branch
```

Streamed KV 경로 (`reassemble_layer_payload`)는 `header.original_num_elements`에서 유도한 한도로 같은 압축 해제와 dequantize 단계를 거친다. Level None은 `n * 2`, 그 외에는 int8 / int4 payload 상한이다.

## 4. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 16 (소스 8, 테스트 8) |
| 추가된 줄 | +850 |
| 삭제된 줄 | -147 |
| 추가된 테스트 | 29 (`mlxcel` lib 28, `mlxcel-core` 1) |

### 모듈별 변경

| 모듈 | 변경 |
|------|------|
| `tensor_protocol.rs` | `num_elements() -> Result<u64>` (`shape product overflow for ...`), `pub const MAX_DECODED_TENSOR_BYTES = 1 << 32` |
| `tensor_quantize.rs` | `validate_layout` / `QuantizedLayout`, `dequantize_int8` / `dequantize_int4`가 0 채움 경로 없이 `Result` 반환, `int8_payload_max_len` / `int4_payload_max_len`, 중간 nibble vector 없는 int4 decode |
| `tensor_compress.rs` | `decompress(data, max_len)`, `LZ4_MAX_EXPANSION = 256`, 정확한 길이 검사를 동반한 `decompress_into` |
| `tensor_serialize.rs` | checked 원소 수, `max_decoded_payload_len`, decode된 모든 tensor에 `validate_data_size` |
| `kv_cache_transfer/streamed.rs` | header에서 유도한 LZ4 한도, dequantize 오류를 context와 함께 전파 |
| `kv_cache_transfer/quantized.rs` | 새 `Result`에 `?` |
| `kv_cache_serde/types.rs` | `validate_raw_tensor`가 dtype 11과 13 거부 |
| `mlxcel-core/cpp/mlx_cxx_bridge.cpp` | `uint16_t` / `int16_t` typed branch, 명시적 1바이트 fallthrough |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `a056d30f` | fix | fix(distributed): bound tensor wire decode sizes and fix int16 decode |
| `bedb4be9` | merge | origin/main을 feature branch로 merge |

## 5. 호환성

- **Wire**: 변경 없음. `PROTOCOL_VERSION` 증가 없음. 인코더가 내보내는 모든 tensor에 대해 main과 branch peer가 섞여도 상호 운용된다.
- **Rust API (crate-public)**: `tensor_compress::decompress`에 `max_len` 인자가 추가되고, `TensorHeader::num_elements`와 `tensor_quantize::dequantize_int8` / `dequantize_int4`가 `Result`를 반환한다. Workspace 내 모든 호출자 (src, tests, benches, examples)가 수정되었다.
- **동작**: 이전에 0으로 decode되거나, 여분 바이트와 함께 decode되거나, 잘못된 크기로 decode되던 tensor는 이제 오류를 반환한다. FLOAT64 / COMPLEX64 raw tensor는 wire 재구성에서 거부된다. Activation이나 KV cache를 직렬화하는 경로 중 이 dtype을 내보내는 곳은 없다.

## 6. 검증

**main 54d63b0a 대상 failing-first.** 새 Rust signature는 main에서 컴파일되지 않으므로, Rust 테스트는 기존 본문을 유지한 signature 전용 shim (`decompress`는 `max_len` 무시, `dequantize_*`는 기존 결과를 `Ok`로 감쌈, `num_elements`는 포화된 값을 감쌈)에 대해 실행했다. 새 테스트 24개 중 20개가 실패했다. `deserialize_rejects_quantized_shape_larger_than_payload`는 4294967296바이트의 0으로 `Ok`를 반환했고 (128 GB 호스트에서 실제로 할당됨), `deserialize_rejects_quantized_shape_overflow`는 `capacity overflow`로 panic했으며, int16 테스트는 `[0, 0, 128, 0, 255, 0, ...]` 바이트를 받았다. 통과한 4개 (`dequantize_accepts_empty_tensor`, `payload_max_len_bounds_every_quantized_payload`, `test_header_num_elements_empty_shape_is_zero`, `deserialize_round_trips_compressed_int4_quantized_tensor`)는 회귀 방지용으로 유지했다. `mlxcel-core`에서는 `from_bytes_round_trips_16bit_integers_bit_exactly`가 수정 전 bridge에서 실패했다. `dequantize_rejects_huge_element_count_without_allocating` (`1 << 40` 원소, 2 TiB)은 기존 코드에 대해 실행하지 않았고, `reassemble_*` 테스트 3개는 수정과 함께 추가되었다.

**Branch head a056d30f** (`--profile test-fast --features cuda`, `gpu-lock` 아래 `--test-threads=1`): fmt와 clippy `--lib --tests -D warnings` 통과, `distributed::tensor_` 445 통과, `kv_cache_transfer::` 62, `kv_cache_serde::` 43, `pipeline::wire_tensor` 9, 새 lib 테스트 28개 모두 통과. GB10 원격 실모델 parity (`MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`) 6개 통과: llama, gemma3, gemma4, qwen3, qwen3.5 parity와 llama drain/shutdown 전환.

**bedb4be9의 merge 전 gate** (main merge 이후):

| 대상 | 결과 |
|------|------|
| verify 스크립트 (fmt, versions, env-var docs, kernel dtype keys, kernel port dispatch, binary assets, Windows paths) | 모두 exit 0 |
| contract 테스트 | 27 + 4 통과 |
| clippy, build | 통과 |
| `distributed::` lib 모듈 | 1461 통과, 0 실패, 4 ignored |
| `pipeline_remote_runtime_*` 실모델 parity | 6 통과 |
| `pipeline_server_remote_coordinator_*` 대 dense baseline | 5 통과 |
| `pipeline_multi_stage_two_host_logical_smoke` | 1 통과 |
| `disaggregated_e2e` | 51 통과 |
| `mlxcel-core` `from_bytes` 테스트 | 6 통과 |

성능 측정은 하지 않았다. 압축/quantization이 없는 activation 경로는 O(1) 검사만 늘었고 복사는 추가되지 않았다. 유효한 입력에 대해 LZ4와 dequantize는 이전과 같은 양을 할당하며, int4 decode는 임시 버퍼 하나가 줄었다.

## 7. 후속 조치

- **약 16 MiB로 4 GiB 할당은 여전히 가능**하다. Header shape이 그 크기와 일관될 때다 (256:1 상한 곱하기 u32 prefix 한계). 모델 설정 (hidden size, 최대 sequence 길이)에서 유도한 더 엄격한 상한은 후속 작업이 필요하다. Issue가 정한 상한은 명세대로 충족했다.
- **Streamed KV level None 상한**은 `original_num_elements * 2`인데, `prepare_layer_payload`는 `keys.len() / 2 + values.len() / 2`로 센다. Key와 value 바이트 수가 모두 홀수인 1바이트 dtype이면 상한이 최대 2바이트 부족해 유효한 layer가 거부된다. `reassemble_layer_payload`는 현재 production 호출자가 없으므로 (`parallel.rs`와 `benchmark.rs`는 `prepare_layer_payload`만 사용) 깨지는 것은 없다. Streamed 수신자가 연결되면 다시 볼 것.
- **`reassemble_layer_payload`의 비압축 level None**은 payload 길이를 `original_num_elements`와 비교하지 않는다. 기존부터 있던 문제로 범위 밖이며, 출력이 wire 바이트의 복사본이므로 증폭은 없다.
- **빈 shape 의미 차이**: `validate_data_size`는 `[]`를 원소 0개로, `validate_raw_tensor`는 원소 1개 scalar로 본다. 인코더가 scalar를 거부하므로 회귀는 없지만, 향후 scalar wire tensor는 이 불일치에 걸린다.
- **FLOAT64 / COMPLEX64는 `from_bytes` 안에서 여전히 오decode**된다. 향후 호출자가 `validate_raw_tensor`를 건너뛰면 문제가 된다. C++ 주석에 불변 조건이 기록되어 있다.
- **기존 복사**: `deserialize_tensor`는 압축 해제 전에 압축 payload를 복사한다.
- **`MAX_DECODED_TENSOR_BYTES = 1 << 32`**는 32비트 `usize` target에서 컴파일되지 않는다. mlxcel이 배포하는 target은 모두 64비트이므로 결함이 아니라 참고 사항이다.

## 8. 학습 포인트

### 8.1 할당 전에 검증하고, 할당 크기를 수신 바이트의 함수로 만들기

버퍼 크기를 정하는 header 필드는 metadata가 아니라 입력이다. 견고한 패턴은 header에서 정확한 기대 길이를 유도하고, payload가 그와 같을 것을 요구한 다음, payload의 고정 배수인 출력만 할당하는 것이다. 하한 검사와 대체값 (`unwrap_or(u64::MAX)`, 실패 시 `vec![0; n]`)을 조합하면 parse 오류가 증폭된 할당과 틀린 데이터로 동시에 바뀐다.

```rust
let expected_len = num_groups
    .checked_mul(2)
    .and_then(|s| s.checked_add(8))
    .and_then(|d| d.checked_add(data_len))
    .ok_or_else(|| anyhow!("... overflows ..."))?;
if payload.len() != expected_len { bail!("... length mismatch ..."); }
let out_len = num_elements.checked_mul(2).ok_or_else(|| anyhow!("... overflows ..."))?;
let mut out = Vec::with_capacity(out_len); // payload의 최대 2배
```

### 8.2 라이브러리 편의 함수는 검사 전에 할당할 수 있다

`lz4_flex::decompress_size_prepended`는 자기 prefix를 읽고 그 값으로 할당한다. 다른 prefix를 검사하는 wrapper로 감싸도 할당이 호출 내부에서 일어나므로 소용없다. Frame을 직접 나누고 직접 크기를 정한 버퍼로 하위 수준 `block::decompress_into`를 호출해야 할당이 검사 뒤로 간다. 신뢰할 수 없는 입력에 쓰이는 모든 "size-prepended" API에 같은 질문을 던져야 한다.

### 8.3 Decoder는 정상 송신자가 켜지 않는 flag도 따른다

정상 stage는 activation을 압축하거나 quantize하지 않으므로, 취약한 branch는 실제 실행에서 커버리지가 없었다. Failing-first 테스트는 직접 만든 wire 버퍼 (`hand_built_wire`, `framed`)로 그 branch에 직접 도달했다. Decoder를 감사할 때는 인코더가 내보내는 조합만이 아니라 header가 표현할 수 있는 모든 flag 조합을 열거해야 한다.

### 8.4 오류 채널 없는 FFI 함수에는 문지기가 필요하다

`from_bytes`는 cxx를 통해 실패를 반환할 수 없으므로, decode할 수 없는 dtype은 peer 데이터가 지나는 유일한 경로에서 한 계층 위에서 거부된다. 불변 조건은 두 주석 (C++ fallthrough와 `validate_raw_tensor`)에 기록되어 있어 향후 호출자가 볼 수 있다.

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| absmax group quantization | group별 f16 scale과 int8 / int4 값 | `dequantize_*`가 이제 정확히 검증하는 layout |
| LZ4 block format | literal run과 (offset, match 길이) sequence | 확장을 약 255:1로 제한 |
| `decompress_into` | 호출자 버퍼에 쓰는 lz4_flex block decoder | 수신자가 버퍼 크기를 정할 수 있게 함 |
| `cxx` bridge | Rust / C++ FFI 생성기 | `from_bytes`에 `Result` 반환이 없음 |

### 관련 PR/이슈

- Issue #2691: 세 결함과 인수 조건 (이 PR이 닫음).
- Issue #2654 / PR #2671: 결함을 발견한 보안 리뷰.
- PR #2684: frame별 AEAD와 base64 tensor bytes. base64 decoder는 확인했고 영향 없음.
- PR #2650, PR #2674: 이 경로를 인증된 peer로 한정하는 cluster 인증.
- PR #2695 (열림): main의 disaggregated serving과 router e2e 실패를 고치는 PR. PR 본문은 이를 streamed KV 경로에 사용 가능한 통합 테스트 대상이 없던 이유로 든다. bedb4be9에서 `disaggregated_e2e`는 51/51 통과했다.
