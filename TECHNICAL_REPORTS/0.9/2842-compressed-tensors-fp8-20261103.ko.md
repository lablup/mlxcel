# 기술 보고서: PR #2842 - feat: compressed-tensors float-quantized FP8 체크포인트 로딩

**작성일**: 2026-11-03
**상태**: 리뷰 중 (열린 PR, #2411 종료)
**언어**: Rust, Markdown
**위험도**: 낮음

---

## 요약

이 PR은 Qwen3.5 로더가 compressed-tensors `float-quantized` FP8 체크포인트를 재구성하도록 만든다. llm-compressor의 `FP8-dynamic` 릴리스가 쓰는 레이아웃이며 `RedHatAI/Qwen3.5-4B-FP8-dynamic`이 그 예이다. 이전에는 벤더의 `quant_method: "fp8"` 블록 레이아웃만 인식해서, 이런 체크포인트는 raw `uint8` 바이트로 모델에 도달하고 per-channel 또는 per-block `weight_scale`은 사용되지 않았다. 변경은 탐지와 requantization 모듈(`fp8_ct`), Qwen3.5 텍스트·VLM·special 로더가 공유하는 라우트 선택기(`fp8_checkpoint`), 그리고 기존 블록 재구성을 정사각 128x128 타일에서 독립적인 `[block_rows, block_cols]` 범위와 edge padding으로 일반화하는 작업을 추가한다. 두 FP8 레이아웃 모두 MLX 네이티브 `mxfp8`로 수렴하므로 forward pass는 그대로다. M3 Ultra에서 변환된 체크포인트는 `Qwen/Qwen3.5-4B`와 decided 위치 116개 중 0개에서 불일치했고, 스케일을 무시한 arm은 첫 위치에서 non-finite logits를 냈다.

---

## 1. 문제 정의

### 1.1 배경

llm-compressor의 `FP8-dynamic` export는 `quantization_config: {quant_method: "compressed-tensors", format: "float-quantized"}`를 선언한다. 변환된 모든 `Linear`는 raw E4M3 바이트를 `<name>.weight`에, `<name>.weight_scale`을 함께 저장한다. 스케일은 출력 채널당 하나(`strategy: "channel"`, `[out, 1]`로 기록)이거나 `block_structure: [rows, cols]` 타일당 하나다. 실제 가중치는 `e4m3_decode(byte) * weight_scale[channel 또는 tile]`이다. `quantization_config.ignore`에 나열된 모듈(vision tower, 모든 `linear_attn` projection, `lm_head`, `re:^mtp.*`)은 bf16으로 남고 스케일이 전혀 없다.

MLX에는 float8 dtype이 없으므로 이 바이트들은 로더에 `uint8`로 도착한다. `src/models/fp8_block.rs`의 기존 탐지는 `quant_method`가 정확히 `fp8`이 아니면 `Ok(None)`을 반환했고, 따라서 compressed-tensors 선언은 어떤 라우트와도 매칭되지 않았다.

### 1.2 기존 문제점

- **문제 1**: E4M3 바이트가 `uint8` 가중치로 모델에 도달하고 `.weight_scale`은 결코 적용되지 않았다. 각 projection이 자기 per-channel 배율만큼 어긋났고, 로더 어디에서도 바이트와 스케일을 짝짓지 않았기 때문에 조용히 진행됐다.
- **문제 2**: `.weight_scale`과 `.input_scale`이 예상치 못한 키로 드러나서, 엄격한 weight map에서는 로드 자체가 실패할 수 있었다.
- **문제 3**: `fp8_block.rs`의 벤더 블록 재구성이 양축에 `SUPPORTED_FP8_BLOCK = 128`을 하드코딩했다. compressed-tensors는 타일을 `block_structure`에 담으므로 정사각이 아니거나 크기가 다른 타일은 재사용할 수 없었다.
- **문제 4**: 같은 FP8 pre-pass가 세 Qwen3.5 로더(텍스트, VLM, special)에 각각 배선돼 있었다. 두 번째 레이아웃을 순진하게 추가하면 로더마다 두 곳씩 분기가 중복됐을 것이다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 잘못된 출력이 정상으로 받아들여짐 (스케일 짝짓기 불일치) | 높음 | 낮음, 이 PR 이후로는 거부 경로가 막음 |
| 기존 벤더 FP8 및 ModelOpt NVFP4 체크포인트 회귀 | 높음 | 낮음, 라우팅이 `.weight_scale_2`와 `num_bits`로 분리됨 |
| 새 재구성으로 peak 로드 메모리 증가 | 중간 | 낮음, 텐서 하나씩 `eval` 수행 |
| 다른 compressed-tensors 계열은 여전히 미지원 | 중간 | 확실함, 다른 모델 계열은 범위 밖 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

변경은 체크포인트 설정과 텐서 메타데이터를 파싱한다. 체크포인트의 코드를 실행하지 않으며, 모든 스케일·형상 불일치는 panic이나 조용한 재해석이 아니라 타입이 있는 로드 에러가 된다.

**체크리스트:**
- [x] 입력 검증: `type`, `num_bits`, `dynamic`, `strategy`, `block_structure`, 스케일 형상을 모두 이름이 붙은 에러로 검증
- [ ] 인증/인가: 해당 없음
- [ ] 데이터 암호화: 해당 없음
- [x] 로깅 (민감 정보 제외): 텐서 수, 레이아웃, 변환 수, 경과 시간만 기록

**발견 사항:**

| 이슈 | 심각도 | 상태 |
|------|--------|------|
| 손상된 체크포인트가 조용히 잘못 스케일될 수 있음 | 높음 | 수정됨: 지원하지 않는 레이아웃은 이름을 붙여 로드 실패 |
| 불일치 채널 스케일이 잘못된 축으로 브로드캐스트될 수 있음 | 높음 | 수정됨: 곱하기 전에 형상을 `rows`와 대조 |

### 2.2 성능 관점

변환은 텐서 하나씩 수행한다. 각 쌍마다 raw 바이트와 스케일을 map에서 제거하고, 디바이스 E4M3 lookup table로 디코드하고, 스케일을 곱한 뒤, 두 입력을 drop하고, `mxfp8`로 requantize한 다음, packed plane과 scales에 `eval`을 호출해 강제로 실체화한 후 다음 텐서로 넘어간다. 따라서 peak 메모리는 f32 재구성 하나와 packed 결과 하나를 담고, 체크포인트 전체의 네 복사본을 담지 않는다.

체크포인트 전체를 bf16으로 dequantize하는 대안(mlx-vlm의 방식)은 체크포인트 대비 가중치 메모리를 두 배로 만들고 forward dtype까지 바꾼다.

**성능 영향:**

| 영역 | 이전 | 이후 | 개선 |
|------|------|------|------|
| 지원 체크포인트 레이아웃 | 벤더 `fp8`만 | 벤더 `fp8`에 compressed-tensors `float-quantized` 추가 | 속도 변화 아님, 수동 변환 단계 제거 |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` 로드 | 미지원 | 1.8 s 로드, 7.75 GB peak | 신규 기능 |
| Forward pass | 변경 없음 (`mxfp8`) | 변경 없음 (`mxfp8`) | 없음 |

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 벤더 라우트와 ModelOpt NVFP4 라우트는 동작이 유지된다. `reconstruct_block_fp8_tensor`는 인자 두 개가 늘었지만 호출자가 같은 모듈에 하나뿐이고 함께 수정됐다.
- **새 의존성**: 없음.
- **호환성**: Qwen3.5 텍스트, Qwen3.5 VLM, special 로더가 배선됐다. compressed-tensors FP8을 낼 수 있는 다른 모델 계열은 이 PR의 범위가 아니다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: `src/models/fp8_ct_tests.rs`에 새 테스트 9개(785줄), `fp8_block_tests.rs`의 호출 지점 갱신 1건. `fp8` 필터는 기존 블록 테스트를 포함해 총 29개 통과를 보고한다.
- **코드 복잡도**: 좁은 공개 표면을 가진 모듈 두 개가 추가됐다. 탐지와 변환이 분리됐고 거부 경로가 명시적이다.
- **기술 부채**: 이 레이아웃에 대해서는 감소했다. 두 재구성 함수에 남은 중복은 스케일 짝짓기 부분뿐이며, 의도적으로 공유하지 않았다.

---

## 3. 기술적 선택과 그 이유

### 3.1 bf16으로 dequantize하지 않고 `mxfp8`로 requantize

**컨텍스트:**
체크포인트는 양자화된 가중치를 저장한다. 로더는 이를 bf16으로 복원하거나, 모델이 이미 실행할 줄 아는 네이티브 MLX 레이아웃으로 requantize할 수 있다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| bf16으로 dequantize (mlx-vlm 방식) | 단순하고 정확함 | 체크포인트 대비 가중치 메모리 두 배, forward dtype 변경 |
| **선택: `mxfp8`로 requantize** | 체크포인트와 같은 메모리 등급, 벤더 경로의 측정된 오차 한계 재사용, forward 변경 없음 | 양자화 오차가 한 단계 추가됨 |
| raw 바이트 유지, matmul마다 dequantize | 로드 시 작업 없음 | 새 forward 커널 필요, 영향 범위 큼 |

**선택 이유:**
`mxfp8`(`group_size` 32, `bits` 8)은 `requantize_block_fp8_weights`가 벤더 레이아웃에 이미 생산하는 대상이며, 프로젝트가 측정하고 pin으로 고정한 오차 한계를 갖는다. 같은 대상을 재사용하면 모델 생성자와 forward pass에 새 코드가 필요 없고, config 병합도 두 라우트가 동일하다.

**트레이드오프:**
재구성은 원래의 E4M3 오차 위에 mxfp8 float-quantization 오차를 더한다. 테스트는 결과를 E4M3 반 스텝에 기존 mxfp8 한계를 더한 값으로 제한하고, logit trace는 decided 위치에서 불일치가 없음을 보인다.

### 3.2 지원하지 않는 레이아웃은 잘못 스케일하지 않고 이름을 붙여 거부

**컨텍스트:**
compressed-tensors는 per-tensor, per-group, per-channel, per-block 스케일과 여러 비트 폭을 선언할 수 있다. 각각은 서로 다른 형상의 sidecar를 가중치와 짝짓는다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 모든 float-quantized config에 최선 노력 변환 | 더 많은 체크포인트 로드 | 채널 짝짓기를 텐서 스케일에 적용하면 유창하지만 틀린 출력 |
| **선택: channel과 block만 허용, 나머지는 로드 실패** | 거부 집합에서는 틀린 출력이 불가능 | 일부 체크포인트는 미지원으로 남음 |

**선택 이유:**
잘못 스케일된 projection은 추론 시 에러를 내지 않는다. 그럴듯한 텍스트를 잘못된 값으로 낸다. 이 실패 모드는 나중에 탐지하기 비싸다. 따라서 탐지는 올바르게 짝지을 수 없는 모든 경우(`weights.strategy`, `weights.num_bits`, `weights.dynamic`, `weights.type`, `block_structure`)에 정확한 필드 이름을 담은 `Err`를 반환한다. 한 체크포인트 안에 channel과 block 레이아웃이 그룹별로 섞인 경우도 포함한다.

**트레이드오프:**
`strategy: "tensor"`와 `"group"` 체크포인트, 다른 비트 폭, dynamic 가중치, `mixed-precision` export는 명시적으로 범위 밖이며 빠르게 실패한다.

### 3.3 모듈 이름이 아니라 텐서 단위로 라우팅

**컨텍스트:**
config의 `ignore` 목록은 HF 모듈 경로로 표현된다. 로더에서 이를 매칭하면 prefix 규칙을 재구현하고 릴리스마다 동기화해야 한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| `ignore` 목록으로 변환 대상 결정 | config를 그대로 반영 | prefix 매칭 중복, 릴리스마다 취약 |
| **선택: `.weight`가 `uint8`이고 `.weight_scale`이 있으며 `.weight_scale_2`가 없을 때만 변환** | 데이터 기반, ModelOpt와 구조적으로 분리 | 체크포인트가 실제로 바이트를 저장한다는 데 의존 |

**선택 이유:**
ignore된 projection은 bf16으로 도착하고 스케일이 없으므로 첫 조건에서 걸러진다. ModelOpt NVFP4 삼중항은 `.weight_scale`을 갖지만 항상 `.weight_scale_2`와 함께 있으므로 세 번째 조건이 이들을 제외하고 `sanitize.rs`에 남긴다. 우연히 스케일을 가진 bf16 텐서도 E4M3로 디코드되지 않고 건너뛴다.

**트레이드오프:**
라우트는 config의 의도보다 텐서 dtype과 키 존재를 신뢰한다. 테스트는 NVFP4 삼중항과 미리 디코드된 가중치 두 경우로 이 경계를 고정한다.

### 3.4 세 로더를 위한 단일 라우트 모듈

**컨텍스트:**
FP8 pre-pass는 Qwen3.5 텍스트 로더, VLM 로더, special 로더에 나타난다. 각각 config 병합과 가중치 pre-pass가 모두 필요하다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 로더마다 레이아웃 분기 | 국소적 | 일관성을 유지할 분기 지점이 여섯 개, config와 가중치를 다르게 배선하기 쉬움 |
| **선택: `fp8_checkpoint_quantization`과 `requantize_fp8_weights`** | 로더당 탐지 하나와 디스패치 하나, 두 레이아웃이 동시에 적용될 수 없음 | 간접 계층 하나 추가 |

**선택 이유:**
`fp8_checkpoint_quantization`은 벤더 탐지를 먼저, compressed-tensors 탐지를 다음에 실행한다. 둘은 `quant_method`로 상호 배타적이므로 최대 하나만 `Some`을 반환한다. `Fp8Checkpoint` enum이 라우트를 config 병합과 가중치 pass 양쪽으로 전달하므로 두 경로가 어긋나지 않는다.

**트레이드오프:**
"명시적 `quantization`이 항상 이긴다" 규칙을 한 곳에 두기 위해 `merge_quantization_block`을 `merge_fp8_block_quantization`에서 추출했다.

### 3.5 블록 재구성을 포크하지 않고 일반화

**컨텍스트:**
compressed-tensors 블록 스케일은 `block_structure: [rows, cols]`를 쓰고, 벤더 경로는 양축 128x128을 가정한다.

**선택 이유:**
`reconstruct_block_fp8_tensor`는 이제 `block_rows`와 `block_cols`를 따로 받고, 축별 블록 수를 계산하고, 온전한 타일로 padding한 뒤 슬라이스로 padding을 제거한다. 벤더 호출자는 `(block, block)`을 전달한다. E4M3 디코드는 `decode_fp8_bytes_to_f32`로 추출해 채널 라우트와 공유하므로, 두 라우트는 바이트를 동일하게 디코드하고 스케일 적용 방식만 다르다.

**트레이드오프:**
`reconstruct_block_fp8_tensor`에 인자가 두 개 늘었다. 호출자가 같은 모듈에 하나뿐이고, 기존 round-trip 테스트를 두 번째 범위를 넘기도록 수정했다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[이전]
config ──> qwen_fp8_block_quantization ──> requantize_block_fp8_weights
              (compressed-tensors에 대해 None 반환)
              │
              └──> 3개 로더 (text, VLM, special), 각자 지역 분기

[이후]
config ──> fp8_checkpoint_quantization
              ├─> qwen_fp8_block_quantization      (quant_method: "fp8")
              └─> compressed_tensors_fp8_quantization (quant_method: "compressed-tensors")
                        │
                        ▼
                  Fp8Checkpoint
                  ├─> merge_fp8_quantization   (config)
                  └─> requantize_fp8_weights   (weights)
                            ├─> requantize_block_fp8_weights
                            └─> requantize_compressed_tensors_fp8_weights
```

### 4.2 주요 코드 변경

**파일: `src/models/fp8_ct.rs`**

탐지는 선언 전체를 검증하고 스케일 레이아웃을 반환하거나 필드 이름을 담아 실패한다:

```rust
pub fn compressed_tensors_fp8_quantization(
    full_config: &Value,
) -> Result<Option<Fp8CtQuantization>, String> {
    let Some(quant_config) = full_config.get("quantization_config").or_else(|| {
        full_config.get("text_config").and_then(|text| text.get("quantization_config"))
    }) else {
        return Ok(None);
    };
    // quant_method는 "compressed-tensors"여야 하고, format은 "float-quantized"
    // (또는 없음, 이 경우 각 그룹의 format이 결정).
    // 각 config_groups.*.weights는 type "float", num_bits 8, static이어야 하고,
    // strategy "channel" 또는 2차원 block_structure를 가져야 한다.
}
```

`requantize_compressed_tensors_fp8_weights`는 텐서 단위로 라우팅하고 peak 메모리를 제한한다:

```rust
let is_e4m3_bytes = weights
    .get(&weight_key)
    .is_some_and(|weight| mlxcel_core::array_dtype(weight) == dtype::UINT8);
if !is_e4m3_bytes { continue; }

let restored = match layout {
    Fp8CtScaleLayout::Channel => reconstruct_channel_fp8_tensor(&weight_key, &raw, &scale, &lut)?,
    Fp8CtScaleLayout::Block { rows, cols } =>
        reconstruct_block_fp8_tensor(&weight_key, &raw, &scale, rows, cols, &lut)?,
};
drop(raw);
drop(scale);

let quantized = mlxcel_core::quantize_weights_with_mode(
    &restored, MXFP8_GROUP_SIZE, MXFP8_BITS, MXFP8_MODE,
);
mlxcel_core::eval(&packed);
mlxcel_core::eval(&scales);
weights.insert(weight_key, packed);
weights.insert(format!("{prefix}.scales"), scales);
weights.remove(&format!("{prefix}{CT_FP8_INPUT_SCALE_SUFFIX}"));
```

**파일: `src/models/fp8_checkpoint.rs`**

라우트 선택기는 config 병합과 가중치 pass를 같은 분기에 묶는다:

```rust
pub enum Fp8Checkpoint {
    VendorBlock(Fp8BlockQuantization),
    CompressedTensors(Fp8CtQuantization),
}

pub fn fp8_checkpoint_quantization(full_config: &Value) -> Result<Option<Fp8Checkpoint>, String> {
    if let Some(detected) = qwen_fp8_block_quantization(full_config)? {
        return Ok(Some(Fp8Checkpoint::VendorBlock(detected)));
    }
    Ok(compressed_tensors_fp8_quantization(full_config)?.map(Fp8Checkpoint::CompressedTensors))
}
```

**파일: `src/models/fp8_block.rs`**

블록 재구성은 독립 범위를 받고 디코드를 공유한다:

```rust
pub fn reconstruct_block_fp8_tensor(
    name: &str, raw: &MlxArray, scale: &MlxArray,
    block_rows: usize, block_cols: usize, lut: &MlxArray,
) -> Result<UniquePtr<MlxArray>, String> {
    let (decoded, rows, cols) = decode_fp8_bytes_to_f32(name, raw, lut)?;
    let row_blocks = (rows + block_rows_i32 - 1) / block_rows_i32;
    let col_blocks = (cols + block_cols_i32 - 1) / block_cols_i32;
    // [row_blocks * block_rows, col_blocks * block_cols]로 padding하고,
    // [row_blocks, block_rows, col_blocks, block_cols]로 reshape해 스케일을 적용한 뒤
    // 슬라이스로 되돌린다.
}
```

**변경 이유:** 하나의 재구성 구현이 두 FP8 레이아웃을 담당하므로 padding과 슬라이스 로직을 한 번만 테스트한다.

**파일: `src/models/qwen3_5.rs`, `src/loading/vlm_qwen.rs`, `src/loading/special.rs`**

각 로더는 벤더 전용 탐지와 변환된 가중치에 대한 `if let`을 라우트 선택기와 두 공유 호출로 교체한다. 호출 지점이 줄어들고, 벤더 체크포인트의 동작은 그대로다.

**파일: `docs/supported-models.md`**

두 번째 FP8 레이아웃과 스케일 형상, ModelOpt NVFP4와의 `.weight_scale_2` 구분, 아래 검증 결과를 기록한다.

### 4.3 데이터 모델 변경

없음. weight map 키는 변환된 텐서에 대해서만 바뀐다. `<name>.weight`가 packed `mxfp8` plane이 되고, `<name>.scales`가 추가되며, `<name>.weight_scale`과 `<name>.input_scale`이 제거된다. 벤더 라우트가 이미 생산하는 것과 같은 형태다.

---

## 5. 학습 포인트

### 5.1 키 서명이 분리된 형제 양자화 레이아웃

**개념:**
두 양자화 방식이 구조적으로 비슷한 데이터(E4M3 바이트와 스케일)를 저장하면서 스케일을 다르게 짝지을 수 있다. 로더는 체크포인트마다 어떤 짝짓기가 적용되는지 결정해야 하고, 두 라우트가 분리됨을 증명할 수 있어야 한다.

**이 PR에서의 적용:**
벤더 라우트는 128x128 블록당 `.weight_scale_inv`를 쓴다. compressed-tensors 라우트는 채널당 또는 `block_structure` 타일당 `.weight_scale`을 쓴다. ModelOpt NVFP4도 `.weight_scale`을 쓰지만 항상 `.weight_scale_2`와 함께다. `.weight_scale_2`가 없어야 한다는 조건이 compressed-tensors 라우트를 ModelOpt와 구조적으로 분리하고, `quant_method`가 벤더 라우트와 분리한다.

**일반적인 사용 사례:**
- 이름 등록 없이 여러 벤더의 양자화 출력을 지원
- 공유 대상 표현 옆에 새 체크포인트 형식을 추가

**예시 코드:**
```rust
weights.keys().any(|key| {
    key.ends_with(".weight_scale") && !weights.contains_key(&format!("{key}_2"))
})
```

### 5.2 메모리가 제한된 다단계 텐서 변환

**개념:**
재구성, 스케일링, requantization은 각각 최소 텐서 크기의 배열을 할당한다. 모든 텐서의 중간값을 한꺼번에 들고 있으면 peak 메모리가 단계 수만큼 곱해진다.

**이 PR에서의 적용:**
pass는 텐서를 하나씩 처리하고, 변환 전에 입력을 map에서 제거하며, 다음으로 넘어가기 전에 출력 쌍마다 `eval`을 호출한다. 코드 주석이 의도를 밝힌다. 가장 큰 projection의 f32 재구성이 유일한 임시값이고, 다음 것을 만들기 전에 해제된다.

**일반적인 사용 사례:**
- 통합 메모리 호스트에서 대형 체크포인트를 로드하고 변환
- dequantize, 변환, requantize를 수행하는 모든 파이프라인

**예시 코드:**
```rust
let restored = reconstruct(...)?;
drop(raw);
drop(scale);
let quantized = quantize_weights_with_mode(&restored, ...);
mlxcel_core::eval(&packed);
mlxcel_core::eval(&scales);
```

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|--------|------|-------------------|
| `compressed-tensors` | llm-compressor가 쓰는 safetensors 양자화 메타데이터 형식 | 이 PR이 추가하는 레이아웃의 이름 |
| `float-quantized` | float8 가중치와 별도 스케일을 나타내는 compressed-tensors 형식 태그 | 탐지 게이트 |
| `block_structure` | 하나의 스케일이 담당하는 그룹별 `[rows, cols]` 타일 | 블록 재구성 경로를 결정 |
| `mxfp8` | 32개 값당 공유 E8M0 지수 하나를 쓰는 MLX 네이티브 8비트 float | 두 FP8 라우트의 requantization 대상 |
| E4M3 | FP8 체크포인트가 쓰는 8비트 float 형식 | 디코드 대상 raw 바이트 인코딩 |
| `.weight_scale_2` | ModelOpt NVFP4 삼중항에 존재하는 2차 스케일 | compressed-tensors 라우트를 구분 |

### 관련 기술/프레임워크
- **MLX**: Apple의 배열 프레임워크. 디바이스 측 변환에 `quantize_weights_with_mode`와 `take`를 쓴다. 공식 문서: https://ml-explore.github.io/mlx/
- **llm-compressor**: `FP8-dynamic` 체크포인트를 생산한다. https://github.com/vllm-project/llm-compressor
- **compressed-tensors**: 메타데이터 스키마. https://github.com/neuralmagic/compressed-tensors
- **mlx-vlm**: bf16으로 dequantize하는 상류 참조 구현, `mlx_vlm/utils.py`의 `dequantize_fp8_weight`. https://github.com/Blaizzy/mlx-vlm

### 관련 PR/이슈
- 이슈 #2411: 이 PR이 종료하는 기능 요청. logit trace와 스케일 무시 arm을 포함한 수용 기준을 나열한다.
- PR #1742: 이 PR이 확장하는 벤더 fine-grained FP8 블록 경로를 추가했다.
- PR #1772: 공유 오차 한계가 의존하는 mxfp8 블록 스케일 반올림을 위해 MLX pin을 올렸다.
- PR #2071: ROCm에서 FP8 블록 체크포인트를 포함해 mxfp8를 end to end로 증명했다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 수 | 10 |
| 추가 줄 수 | +1524 |
| 삭제 줄 수 | -79 |
| 새 테스트 | `src/models/fp8_ct_tests.rs`에 9개 |
| 새 모듈 | 2개 (`fp8_ct`, `fp8_checkpoint`) |
| 테스트 명령 | `cargo test --profile test-fast -p mlxcel --lib fp8` (29 통과) |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-----|------|
| 기능 | 3 | 탐지 및 requantization, 라우트 선택, 로더 배선 |
| 리팩터링 | 2 | 블록 재구성 일반화와 E4M3 디코드 공유, config 병합 공유 |
| 테스트 | 2 | 새 compressed-tensors 스위트, 벤더 블록 호출 지점 갱신 |
| 문서 | 1 | `docs/supported-models.md`에 두 번째 레이아웃 항목 |
| 모듈 등록 | 1 | `src/models/mod.rs` export |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `68bca8f3` | feat | load compressed-tensors float-quantized FP8 checkpoints |

---

## 8. 후속 조치

### 완료 필요

- [ ] 차단 항목 없음. PR은 열려 있고 리뷰를 기다린다.
- [ ] 머지 CI 확인, 특히 `fp8` 필터를 넘어선 전체 워크스페이스 테스트 스위트.

### 모니터링 필요

- 4B 참조 외의 compressed-tensors FP8 체크포인트에서 로드 시간과 peak 메모리를 관찰한다. 이 PR은 한 호스트에서 한 체크포인트만 측정했다.
- `strategy: "tensor"` 또는 `"group"`, 다른 비트 폭을 쓰는 compressed-tensors 릴리스를 관찰한다. 이들은 의도적으로 실패하며 새 짝짓기가 필요하다.

### 향후 개선 사항

- 라우트를 다른 모델 계열로 확장한다. `fp8_checkpoint_quantization`을 호출하는 곳은 Qwen3.5 텍스트, VLM, special 로더뿐이다.
- `mixed-precision` 경로를 검토한다. 현재 `None`을 반환하고 dense 로딩으로 넘어가는데, 이는 영향받는 텐서가 bf16으로 로드될 때만 올바르다.
- CUDA와 ROCm에서 변환을 측정한다. logit trace와 로드 수치는 Metal/M3 Ultra 전용이다.

---

## 부록

### A. 테스트 결과

| 검사 | 결과 |
|------|------|
| `cargo test --profile test-fast -p mlxcel --lib fp8` | 29 통과 (탐지, 채널 및 블록 변환, NVFP4 삼중항 비간섭) |
| `cargo clippy --lib --tests -- -D warnings` | 통과 |
| `cargo fmt --check` | 통과 |

새 스위트가 다루는 범위: 중첩된 `text_config`를 포함한 config 탐지, 블록 및 채널 레이아웃, `[out, 1]` 스케일 표기, 양축 부분 블록 padding, NVFP4 삼중항 격리, 미리 디코드된 bf16 가중치, 홀수 입력 폭 거부, 불일치 채널 스케일 거부, 스케일이 없을 때 항등.

### B. Logit Trace

`docs/architecture.md`의 256 위치를 `examples/logit_trace`로 teacher-forced(청크당 4 토큰, 64 청크, topk 8, prefill 512)하고, `RedHatAI/Qwen3.5-4B-FP8-dynamic`을 `Qwen/Qwen3.5-4B`와 `scripts/compare_logit_traces.py --decided 2`로 비교:

| 지표 | 값 |
|------|-----|
| Top-1 불일치 | 9/256 (3.516%) |
| Decided 위치 불일치 (참조 top-two gap >= 2.0) | 0/116 (0.000%) |
| 불일치의 순위 | 2~3위, 모두 참조 gap 0.5 미만 |
| 후보 perplexity | 16.5992 |
| 참조 perplexity | 16.7090 |

스케일 무시 arm(raw 바이트만 디코드, per-channel 스케일 미적용)은 trace를 전혀 내지 못한다. `chunk 0 position 0 produced a non-finite NLL`. 체크포인트의 `weight_scale` 값은 1.4e-05에서 1.54e-03에 걸쳐 있어, 이를 빼면 변환된 각 행이 649배에서 71392배로 스케일된다.

End-to-end 확인: `mlxcel generate -m RedHatAI/Qwen3.5-4B-FP8-dynamic -p "The capital of France is" -n 300 --temp 0`가 "The capital of France is **Paris**."를 답하며, 7.75 GB peak, 1.8 s 로드다.

### C. 참고 자료

- 이슈 #2411, "Load compressed-tensors float-quantized FP8 checkpoints"
- `src/models/fp8_ct.rs` 모듈 문서, bf16 dequantization을 거부한 이유를 기록
- `src/models/fp8_block.rs` 모듈 문서, mxfp8 오차 한계와 ROCm tie-rounding 주의를 기록
- mlx-vlm `dequantize_fp8_weight`: 상류의 bf16 참조 구현
