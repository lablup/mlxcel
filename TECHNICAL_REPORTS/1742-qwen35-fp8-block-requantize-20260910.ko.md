# 기술 보고서: PR #1742 - feat(qwen3_5): 벤더 fine-grained FP8 체크포인트 로딩

**날짜**: 2026-09-10
**작성**: mlxcel maintainers
**검토**: 구현 리뷰 사이클
**상태**: 완료. 인수 조건 하나는 의도적으로 미검증으로 남김 (bf16 원본 체크포인트를 받지 않음. 인수 조건에 적힌 `metal,accelerate` 명령은 이 호스트에서 실행 불가라 실행하지 않음)
**언어**: Rust
**위험도**: 중간 (Qwen3.5 경로에 로드 시점 변환이 새로 들어감. `weight_scale_inv` 사이드카가 없는 체크포인트의 동작은 그대로)

---

## 요약

Qwen3.5 계열의 벤더 FP8 릴리스(`Qwen/Qwen3.8-27B-FP8` 및 형제 모델)는 변환 대상 프로젝션을 `<name>.weight`의 raw `E4M3` 바이트와 `<name>.weight_scale_inv`의 128x128 블록당 `bfloat16` 역스케일, 두 텐서로 나눠 저장한다. 이 작업의 출발점이 된 이슈는 mlxcel이 예상 못 한 사이드카에서 실패할 것이라고 봤다. 실패하지 않는다. MLX의 safetensors 리더는 `F8_E4M3`을 `uint8`로 매핑하고(`mlx/io/safetensors.cpp`의 `dtype_from_safetensor_str`), 그래서 바이트는 온전히 들어오며, `UnifiedLinear`는 `.scales` 키가 없으니 dense 경로를 타고, 모델은 0.3초 만에 로드돼 생성까지 한다. 다만 모든 프로젝션이 자기 블록 스케일만큼 틀려 있다.

이것이 이 PR을 기능 추가가 아니라 정확성 수정으로 만드는 이유이고, 동시에 증거를 어디에 대야 하는지를 결정한다. 여기서 "로드에 성공했다"는 아무것도 증명하지 않는다. 이미 성공하고 있었기 때문이다.

`models::fp8_block`은 각 쌍을 디바이스에서 복원한 뒤 MLX 네이티브 `mxfp8`(group size 32, 8 bits)로 재양자화한다. 공개된 27B 체크포인트에서 이 변환은 로드에 약 30초를 더하고, 할당기 피크를 최종 상주 크기의 1.04배로 유지하며, 정상적인 출력을 낸다.

---

## 1. 체크포인트에 실제로 들어 있는 것

이슈 본문이 아니라 `models/mlx/qwen3.8-27b-fp8`을 직접 읽어 확인한 값이다.

```
config.json quantization_config:
  {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
   "weight_block_size": [128, 128], "modules_to_not_convert": [... 882개 ...]}
```

| 항목 | 실측 |
|---|---|
| 인덱스의 텐서 수 | 1606 |
| `F8_E4M3` 텐서 | 407 |
| `*.weight_scale_inv` 사이드카 | 407 |
| 사이드카가 없는 `F8_E4M3` 텐서 | 0 |
| 자기 weight와 같은 샤드에 있는 사이드카 | 407 / 407 |
| 사이드카 dtype | 전부 `BF16` |
| 형상이 `ceil(R/128) x ceil(C/128)`이 아닌 사이드카 | 0 |
| 너비가 32의 배수가 아닌 weight | 0 |
| 두 축 모두 블록 정렬이 아닌 weight | 0 |

마지막 줄이 테스트 설계를 결정한다. 이 체크포인트의 실제 텐서는 두 축 모두 정확히 128의 배수라서, 실물로는 pad-and-slice 분기가 한 번도 실행되지 않는다. 그래서 단위 테스트는 어느 축도 정렬돼 있지 않은 130x160 텐서를 쓴다.

변환 대상은 디코더 레이어당 열 개의 프로젝션(`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`)과 번들된 `mtp.*` 헤드의 일곱 개다. 나머지는 전부 `BF16`으로 남는다. `embed_tokens`, `lm_head`, 모든 norm, `conv1d`, `A_log`, `dt_bias`, 저랭크 `linear_attn.in_proj_a` / `in_proj_b`, 그리고 27블록짜리 비전 타워 전체가 여기 해당한다. 882개짜리 `modules_to_not_convert` 목록은 참고용이다. 실제로 사이드카를 가진 텐서만 변환하므로 이 목록은 파싱할 일이 없다.

### 1.1 스케일 방향이 함정이다

`weight_scale_inv`라는 이름은 역수를 뜻하고, 그래서 나눗셈을 유도한다. 실제로는 곱셈이다. Rust 경로와 무관하게 numpy로 실제 텐서 세 개를 확인했다.

| 텐서 | `decode`만 | `decode * scale_inv` | `decode / scale_inv` |
|---|---|---|---|
| `layers.3.self_attn.q_proj` `[12288, 5120]` | max 448, std 92.7 | max 0.320, std 0.0172 | max 4.08e6 |
| `layers.1.mlp.down_proj` `[5120, 17408]` | max 448, std 76.8 | max 0.441, std 0.0108 | max 5.31e6 |
| `layers.1.linear_attn.in_proj_qkv` `[10240, 5120]` | max 448, std 90.9 | max 0.333, std 0.0154 | max 3.82e6 |

곱셈은 트랜스포머 프로젝션의 정상 분포에 떨어지고, 나눗셈은 일곱 자릿수만큼 벗어난다. 아울러 raw decode의 최댓값이 모든 텐서에서 정확히 448, 즉 E4M3의 최댓값이다. 배포자가 블록마다 E4M3 범위를 꽉 채우도록 스케일을 잡았다는 뜻이고, 그래서 블록 스케일은 장식이 아니라 값의 크기를 실제로 결정한다.

---

## 2. 변환

`requantize_block_fp8_weights(weights, block)`은 사이드카 키를 정렬 순서로 훑으면서 쌍마다 다음을 수행한다.

```
indices  = astype(raw_uint8, uint32)
decoded  = take(lut_256_f32, indices, axis=0)          # E4M3 디코드, 디바이스에서
padded   = pad(decoded, rows -> rb*128, cols -> cb*128, 0.0)
blocked  = reshape(padded, [rb, 128, cb, 128])
scaled   = blocked * reshape(astype(scale_inv, f32), [rb, 1, cb, 1])
restored = slice(reshape(scaled, [rb*128, cb*128]), [0,0], [R, C])
(q, s)   = quantize(restored, group_size=32, bits=8, mode="mxfp8")
```

`<name>.weight`는 패킹된 `uint32` 평면(`[R, C/4]`)으로 바뀌고, `<name>.scales`(`uint8`, `[R, C/32]`)가 추가되며, `<name>.weight_scale_inv`는 제거된다. `.biases` 평면은 만들어지지 않고, MLX가 혹시라도 그것을 돌려주면 코드가 진행을 거부한다. `mxfp8`은 block-float 모드라 affine 영점이 나왔다는 것은 모드가 잘못 라우팅됐다는 뜻이기 때문이다.

명시할 만한 지점이 셋 있다.

**디코드는 디바이스에서 돈다.** `sanitize.rs`에는 이미 호스트 측 `f8_e4m3_to_f32`가 있고, `tensor_view_to_array`의 기존 `F8_E4M3` 분기가 바이트 단위 루프로 그것을 쓴다. E4M3 바이트가 26 GB인 상황에서 그 루프는 체크포인트 전체를 CPU로 통과시키고 네 배 크기의 f32 `Vec`을 만든다. 같은 함수를 256칸 `f32` 테이블로 올려 `take`로 게더하면 디코드가 GPU에 남고 호스트 사본은 1 KB로 끝난다.

**bf16이 아니라 f32.** 이슈의 의사코드는 bf16으로 디코드한다. 대신 f32를 쓴다. E4M3 값도 bf16 스케일도 f32에서 정확하므로 곱은 정확히 반올림된 참값이 되고, 뒤이은 `mxfp8`의 그룹 최댓값 탐색이 bf16으로 뭉개진 값이 아니라 참 최댓값을 본다. 대가는 임시 버퍼가 E4M3 바이트의 2배가 아니라 4배가 되는 것인데, 아래 실측이 그 정도는 감당된다는 것을 보여준다.

**피크 메모리는 구조적으로 묶여 있고, 실측했다.** 각 쌍은 루프가 다음으로 넘어가기 전에 `eval`되고, E4M3 바이트와 사이드카는 quantize 호출 전에 버려진다. `eval`이 없으면 MLX는 모든 복원을 첫 forward까지 lazy로 들고 있고, 그때의 임시 버퍼는 텐서 하나의 f32 사본이 아니라 체크포인트 전체분이 된다.

### 2.1 무엇을 거부하고, 왜 이름 붙은 오류인가

아래는 전부 받아들이면 실패 없이 텐서 크기만 어긋나게 만드는 조건이다. 이 PR이 없애려는 실패 양상이 정확히 그것이다.

| 조건 | 결과 |
|---|---|
| `[128, 128]`이 아닌 `weight_block_size` | 이름 붙은 오류 |
| `fmt`가 있고 `e4m3`이 아님 | 이름 붙은 오류 |
| `quant_method: "fp8"`인데 `weight_block_size` 없음 (per-tensor FP8) | 이름 붙은 오류 |
| weight 랭크가 2가 아님 | 이름 붙은 오류 |
| weight를 타일링하지 못하는 사이드카 형상 | 이름 붙은 오류 |
| 대응하는 `.weight`가 없는 사이드카 | 이름 붙은 오류 |
| 너비가 32의 배수가 아님 | 이름 붙은 오류 |
| 이미 float dtype으로 디코드된 weight | 이름 붙은 오류 |
| 사이드카가 아예 없음 | 텐서를 그대로 통과 |

마지막 줄 덕분에 Qwen3.5 경로에서 이 사전 패스를 조건 없이 돌릴 수 있다.

---

## 3. 어디에 연결했나

Qwen3.5의 세 진입점 전부다. 셋 다 `Qwen35Model::from_weights`에 도달하고, 그대로 두면 셋 다 스케일 안 걸린 바이트 위에 dense 레이어를 세운다.

- `Qwen35Model::load` (텍스트): 전체 config에서 감지하고, `text_config`가 역직렬화되기 전에 유효 `quantization`을 병합하며, `load_text_weights` 뒤 `sanitize_moe_weights` 앞에서 변환한다.
- `load_qwen3_5_vlm_with_variant` (VLM, 공개 체크포인트가 타는 경로): `load_vlm_weights_common` 직후, 가중치 맵이 텍스트와 비전으로 **쪼개지기 전에** 변환한다. 그래야 모든 사이드카가 자기 weight 옆에 남아 있다. 비전 타워는 애초에 사이드카가 없다.
- `try_load_special_model_from_weights` / `qwen35_text_config` (어댑터): 소유 사본에 같은 두 단계를 적용한다.

병합되는 블록은 `{"group_size": 32, "bits": 8, "mode": "mxfp8"}`이다. `Qwen35Config`는 `group_size`와 `bits`만 읽고, `UnifiedLinear::from_weights`가 `infer_quantization_mode(has_biases: false, 32, 8)`을 호출해 `"mxfp8"`을 얻는다. 모델 생성자에 새로 무엇을 꿰어 넣을 필요가 없었고, dense 텐서는 계속 dense 분기를 탄다. 그 분기가 config가 아니라 텐서별 `.scales` 존재 여부로 갈리기 때문이다.

---

## 4. 검증

모든 명령은 Linux / aarch64 / GB10(sm_121)에서 `--features cuda`와 `MLX_ENABLE_TF32=0`을 명시해 실행했다. MLX의 기본값은 `MLX_ENABLE_TF32=1`이고 이는 가수 10비트라서 f32 증거가 되지 못한다.

### 4.1 단위 수준의 증명

`cargo test --profile test-fast --features cuda --lib models::fp8_block` - 10개 통과.

핵심 테스트 `fp8_block_requantize_matches_direct_path`는 디바이스 경로를 **두 평면 모두 비트 단위로** 호스트에서 만든 레퍼런스와 비교한다. 레퍼런스는 각 블록 스케일을 타일 전체로 확장해 원소별로 곱한 뒤 같은 `quantize_weights_with_mode(..., 32, 8, "mxfp8")`을 통과시켜 만든다. 두 경로가 공유하는 것은 E4M3 디코드 테이블뿐이다. 블록 짝짓기, 패딩, 브로드캐스트 축 순서, 뒤쪽 슬라이스는 전부 크래시가 아니라 틀린 답으로 나타나는 종류의 결함이고, 비트 비교만이 그것을 잡는다. 텐서는 130x160으로 어느 축도 정렬돼 있지 않아 패딩과 슬라이스가 둘 다 실제로 실행된다.

같은 테스트가 결과를 dequantize해 그룹 단위로 복원값과 대조하기도 한다. E4M3은 유효 4비트를 실으므로, 그룹 최댓값과 같은 binade에 있는 원소는 최대 `group_max / 16`만큼 반올림된다. 모든 원소가 이 한계 안에 들었다.

보조 케이스: 블록 정렬된 256x128 텐서(패딩 없는 분기, 역시 비트 단위 비교), 256가지 E4M3 인코딩 전부를 디바이스 LUT로 통과시켜 호스트 테이블과 비트 대조하고 `f32_to_f8_e4m3`로 왕복까지 확인, 위 표의 일곱 가지 거부 조건, dense 텐서 무변경, 사이드카가 없는 맵에 대한 항등성.

### 4.2 실제 체크포인트

`Qwen/Qwen3.8-27B-FP8`, 29 GB:

```
mlxcel generate -m models/mlx/qwen3.8-27b-fp8 -p "Write a limerick about a Rust compiler." -n 220 --show-reasoning
```

```
Model loaded in 51.195s (resident: 23.36 GB, peak: 24.28 GB).
...
There once was a coder in Rust,
Whose compiler was grumpy and just,
It said, "You must borrow,
Or your code will not go,
'Cause lifetimes will make you be dust."
[Generated 207 tokens in 34.85s = 5.94 tok/s]
```

모델은 자기 각운 구조를 스스로 점검하는 주제에 맞는 추론 흔적을 낸 뒤 형식이 맞는 limerick을 쓰고, 220토큰 중 207에서 스스로 멈췄다. 별도의 64토큰 실행은 32.096초에 로드됐고 상주 23.36 GB, 피크 24.37 GB였다.

**피크 메모리**: 24.37 / 23.36 = **1.04배**로, 이슈가 요구한 1.5배 안이다. 이 수치는 MLX 할당기 값(`mlxcel_core::memory::snapshot`의 active와 peak)이고, 여기서는 그쪽이 맞는 계측기다. `/usr/bin/time -v`로 본 프로세스 RSS 피크는 6.9 GB인데, 이 호스트에서 MLX의 CUDA 할당이 RSS에 잡히지 않기 때문이다. 주장한 값이 아니라 측정한 값이다.

### 4.3 인접 스위트

`--lib models::qwen3_5` (28 통과, 11 ignored), `--lib models::sanitize` (71 통과), `--lib loading::special` (6 통과). `cargo fmt --all -- --check`와 `cargo clippy --workspace --all-targets --features cuda -- -D warnings` 통과.

---

## 5. 검증하지 못한 것

**bf16 원본 대비 로짓 일치.** 이슈는 첫 생성 위치의 top-5 로짓이 `Qwen/Qwen3.8-27B`와 mxfp8 허용오차 안에서 일치할 것을 요구한다. 그 체크포인트는 55.6 GB이고 검증 호스트에 의도적으로 받지 않았다. 이 항목은 체크하지 않은 채로 두며, 대체 증거를 제시하지도 않는다. 같은 아키텍처의 다른 양자화본과 비교하거나 이 PR 자신의 역양자화와 비교하는 것은 다른 질문에 답하는 일이기 때문이다. 대신 증명한 것은 복원 공식 자체를 단위 수준에서 비트 단위로, 스케일 방향을 실제 텐서에 numpy로, 그리고 모델 규모에서 정상 생성으로 확인한 것이다.

**`cargo test --workspace --profile test-fast --features metal,accelerate`.** 이 호스트에서 실행 불가다. Linux, aarch64, CUDA이고 Metal도 Accelerate도 없다. 통과가 아니라 미실행이다. 변경 범위에 대해서는 위의 CUDA 대응 명령을 실행했다.

**`qwen3_5_moe` FP8.** 해당 릴리스를 구할 수 없었다. 감지는 config를 보므로 범위에 들어가고, 블록 스케일을 가진 융합 3차원 `experts.gate_up_proj`는 잘못 변환되는 대신 이름 붙은 오류로 거부된다(랭크 2 필요). 전문가별 2차원 텐서라면 변환되어 기존 양자화 switch-layer 로더로 들어가겠지만, 그것은 추론이지 측정이 아니다.

**MTP.** 드래프터는 `mlxcel-core` 안에 자기 가중치 로더를 갖고 있어 `crate::models::fp8_block`에 닿지 못하므로, 번들된 `mtp.*` FP8 텐서는 드래프터용으로 변환되지 않는다. 이 계열의 MTP 추측 디코딩은 이미 미지원이고 VLM 로더는 `mtp.*`를 통째로 버리므로 후퇴하는 것은 없다.

---

## 6. 후속으로 낼 만한 것

- FP8 블록 복원을 `mlxcel-core`로 옮겨 MTP 드래프터를 비롯한 코어 측 로더가 쓸 수 있게 한다. 이 계열의 MTP 지원을 다시 볼 때 필요하다.
- `requantize_block_fp8_weights`는 이미 계열 비의존적이다. DeepSeek V3(UE8M0)와 Mistral 4 FP8도 블록 형상만 확인되면 재사용할 수 있다. `mistral4.rs`는 지금 `weight_scale_inv` 키를 그냥 버리는데, 이는 이 PR이 Qwen3.5에서 제거한 것과 같은 조용한 오스케일링이다.
- 변환은 27B 체크포인트 기준 로드마다 약 30초다. 변환본을 디스크에 캐시하면 없앨 수 있지만, 원본 크기의 산출물이 하나 더 생긴다.
