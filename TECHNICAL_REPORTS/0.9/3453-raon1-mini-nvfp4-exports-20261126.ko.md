# PR #3453: Raon-1 mini MLX 및 ModelOpt NVFP4 export 로딩 (이슈 #3437)

**작성일**: 2026-11-26

**상태**: PR 열림(#3437을 닫음). M3 Ultra에서 검증. 17개 파일, +1618 / -100.

**위험도**: 중간(공용 ModelOpt NVFP4 감지기 확장, 신뢰할 수 없는 양자화 plane의 로드 시 검증 추가)

## 요약

KRAFTON은 Raon-1 mini의 NVFP4 export를 두 가지로 배포한다. 벤더 MLX 레이아웃(`KRAFTON/Raon-1-mini-mlx`)과 NVIDIA ModelOpt 레이아웃(`KRAFTON/Raon-1-mini-nvfp4`)이다. #3225는 MLX export를 이름을 붙여 거부했고, mlxcel의 ModelOpt 감지기는 두 번째 것을 인식하지 못했다. 이 PR은 둘 다 로드한다. 상주 메모리는 30.56 GB에서 약 16.4 GB로 줄고, decode는 33.4에서 약 39 tok/s로 오른다. bf16 원본 대비 top-1 일치율은 87~90%, perplexity 증가는 2.5~3.6%이며, 두 export는 결정 위치마다 서로 일치한다.

## 1. 문제 정의

### 1.1 배경

두 export 모두 routed expert와 대부분의 dense linear를 NVFP4(E2M1 값, 입력 16개마다 E4M3 scale)로 양자화하고 f32 텐서 단위 scale을 하나 더 두며, `lm_head`는 MXFP8 g32로 유지한다. 차이는 표기 방식이다.

- **MLX export**: MLX `quantization {nvfp4, group 16}`과 모듈별 override. expert는 expert별 `global_scale`을 가진다. `raon_global_scaled_linears`에 나열된 dense linear는 출력에 적용하는 텐서별 `global_scale`을 가진다(`bf16(f32(y) * g)`, 벤더의 `GlobalScaledLinear`).
- **ModelOpt export**: `quant_method: modelopt_fp4`, `quant_algo: MIXED_PRECISION`, 모듈별 `quantized_layers` 맵(`NVFP4`, `W4A16_NVFP4`, `MXFP8`). expert별 텐서를 쌓아야 하고, `weight` / `weight_scale` / `weight_scale_2` 세 개가 한 묶음이다.

### 1.2 기존 문제점

- `is_modelopt_nvfp4`는 `quant_method == "modelopt"`와 `config_groups`를 요구해서, 이 export는 감지되지 않았고 triplet도 repack되지 않았다.
- Raon 로더는 head와 dense linear에 원본 bf16 + MXFP8 레이아웃을 가정했다.
- 체크포인트의 양자화 plane을 그대로 믿었다. `weight`보다 행이 적은 `scales` 텐서가 오면 GPU 커널이 scales 버퍼 밖을 읽을 수 있었다.

## 2. 기술적 선택과 그 이유

### 2.1 plane마다 레이아웃을 텐서 shape로 추론

각 linear의 포맷을 모듈별 config 항목에 맡기지 않고, Raon 로더(`raon1_mini_quant.rs`)가 `weight`와 `scales` shape로 레이아웃을 정한다. 그래서 MXFP8 head를 NVFP4나 MXFP4로 잘못 읽지 않고, 원본 체크포인트, 두 export, tiny 테스트 모델을 같은 코드가 로드한다. config 항목은 여전히 교차 검증한다. `raon_global_scaled_linears`는 `global_scale`을 가진 텐서와 일치해야 하고, 그 목록 밖 dense linear에 붙은 `"global_scale": true`는 조용히 버리지 않고 거부한다.

### 2.2 기존 NVFP4 global-scale 경로 재사용

- expert별 `global_scale`은 기존 config 플래그 해석을 거쳐 `gather_qmm`의 native NVFP4 tensor scale로 들어간다. Inkling 2가 이미 쓰는 경로다.
- dense 텐서별 `global_scale`은 `UnifiedLinear`의 출력 배율이다.
- ModelOpt triplet은 기존 `sanitize_modelopt_nvfp4_weights` repack을 거친다. Raon은 repack된 expert별 plane을 쌓는다.

### 2.3 MXFP8 변환은 Raon만

| 선택지 | 장점 | 단점 |
|---|---|---|
| 모든 계열에서 MXFP8 레이어를 MLX `mxfp8` plane으로 변환 | 규칙이 하나 | 한 포맷만 가정하는 로더가 잘못 읽는다 |
| **`MXFP8_LAYER_FAMILIES`(Raon)에서만 변환** | Raon은 shape로 레이아웃을 읽고, 다른 계열은 이름 붙은 오류 | 다른 계열은 명시적으로 opt-in해야 한다 |

첫 버전은 모든 계열에서 변환했고, pr-reviewer가 HIGH로 지적해 `474d6fee6`에서 좁혔다. 전부 NVFP4인 mixed-precision export는 여전히 모든 텍스트 계열에서 로드된다. 그 밖의 레이어별 알고리즘, NVFP4 g16이나 MXFP8 g32가 아닌 MLX quantization 항목, `kv_cache_scheme`은 이름 붙은 오류로 실패한다.

### 2.4 신뢰할 수 없는 plane은 로드 시 모두 검증

보안 리뷰가 위의 범위 밖 읽기를 HIGH로 찾았다. `check_scales_match_weight`는 이제 모든 Raon 양자화 plane(dense, routed expert, RDA, MLA, head)에서 `scales`(와 `biases`)가 마지막 축을 뺀 모든 축에서 `weight`와 맞는지 확인한다(`f1989648a`, `f6ad29ee5`). dense `global_scale`은 float 원소 하나여야 한다. 공용 `UnifiedLinear` 생성자의 같은 검사는 mlx-lm 양자화 계열 전체에 해당하고 이 PR 이전부터의 문제라 후속으로 남겼다.

## 3. 검증

Trace 기준: bf16 원본에 대한 mlxcel 자체 trace. 이 trace는 Transformers reference와 일치한다(#3441, #3445). 전체 수치는 `docs/benchmark_results/raon-nvfp4-m3ultra-2026-11-26.md`.

| Export | 토큰 | Top-1 일치율 | Perplexity 변화 |
|---|---|---|---|
| MLX | 512 | 88.67% | +2.48% |
| ModelOpt | 512 | 86.91% | +2.59% |
| MLX | 4096 | 89.77% | +3.63% |
| ModelOpt | 4096 | 89.92% | +3.49% |

불일치는 모두 reference top-2 gap 2.375 이하 위치에서만 나왔다.

| 체크포인트 | 로드 | 상주 메모리 | Decode |
|---|---|---|---|
| 원본 | 1.4 s | 30.56 GB | 33.35 tok/s |
| MLX export | 0.7 s | 16.34 GB | 39.88 tok/s |
| ModelOpt export | 67~75 s | 16.39 GB | 38.76 tok/s |

- 두 export 모두 `--reasoning-effort none`에서 `42`, 기본 effort에서는 정상적인 thinking 후 `42`.
- 벤더 런타임(`raon_mlx_model.py --no-dflash`, `RAON_EXACT_KERNELS=0`, greedy 64): 기본 effort에서 동일. thinking off에서는 50번째 토큰에서 logit 0.25 차이의 근소한 동률(34.25 대 34.0)로 갈라지며, 더 넓은 mlxcel forward는 벤더와 일치한다.
- 단위 테스트: 각 레이아웃으로 양자화한 tiny 모델을 디스크에 쓰고 `Raon1MiniModel::load`로 로드하면 두 effort에서 dequantize한 쌍둥이와 일치한다(gather 경로의 expert별 scale, f32 steering 벡터 포함). global-scaled linear는 dequantize-matmul 결과에 scale을 곱한 값과 같다. 감지기는 이 config를 받고 모르는 알고리즘을 거부한다. MXFP8 head는 바이트 view다. 잘못된 scales와 global scale은 거부된다. 범위 테스트 345개 통과, 로컬 `make verify` 전체 게이트 통과.

## 4. 후속 조치

- 공용 `UnifiedLinear::from_weights_with_mode`와 `from_stacked_parts`에 scales 행 검사 추가.
- ModelOpt export는 NVFP4 repack이 CPU에서 돌아 로드에 약 70초 걸린다.
