# 기술 보고서: PR #2085 - ROCm에서 큰 bf16 qmm을 hipBLASLt로 보내고 dense prefill을 경로로 제한

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료. head `c15ff36d`, origin/main `7278397a` 기준, 머지 대기 중.

**언어**: C++/HIP (ROCm overlay: `QuantizedMatmul` dispatch, export된 경로 predicate), C++ (cxx bridge), Rust (dense prefill 적격성, 테스트), Markdown

**위험도**: 중간 (RDNA 3.5 tier에서 128 row 이상인 모든 bf16 affine GEMM의 기본 kernel이 바뀌며, 이는 그 tier에서 bf16 scale 4-bit checkpoint의 prefill 전체에 해당합니다. 다른 ROCm tier는 환경 변수로 ceiling을 지정하지 않는 한 기존 dispatch를 유지하고, Metal과 CUDA에서는 새 bridge 호출이 true를 반환하므로 적격성이 바뀌지 않습니다)

## 요약

이슈 #2081(epic #1801의 일부)은 gfx1151의 `make verify-rocm`에 남은 마지막 실패였습니다. `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`의 bf16 케이스입니다. 이 테스트는 `prefill_dense_gemm_eligible`이 projection을 받아들이는 곳마다 mlxcel의 dense prefill 경로(`dequantize` 후 `matmul`)가 `quantized_matmul`과 같은 바이트를 반환하는지 확인합니다. ROCm에서 bf16 `quantized_matmul`은 fork의 fused `qmm_wmma_dense_kernel`을 실행했고, 이 kernel은 rocWMMA tile을 통해 자체 K 순서로 누적합니다. 반면 dense 쪽은 hipBLASLt를 실행했습니다. 1,048,576개 출력 중 499개가 달랐습니다. f16은 `quantized_matmul`도 dequantize 후 hipBLASLt를 호출하기 때문에 통과했습니다.

이슈는 세 가지 선택지를 제시했습니다. 두 경로를 바이트 단위로 같게 만들기, ROCm bf16 케이스를 적격성에서 제외하기, 바이트 동일성을 ULP 한계로 대체하기입니다. PR은 선택지 1을 택했고, 어느 kernel도 고치지 않고 경로를 바꾸는 방식으로 구현했습니다. op 수준 측정에서 WMMA kernel은 128 row 이상에서 측정한 모든 bf16 shape에서 더 느린 쪽이었으므로, 그런 GEMM을 dequantize + hipBLASLt로 보내면 두 경로가 같아지고 실제 prefill도 빨라집니다. 기본 경로를 이전 경로(`MLX_ROCM_WMMA_QMM=1`)와 비교하면 Gemma 3 4B 4-bit의 prefill은 2048 토큰에서 961에서 2815 tok/s로 올라갑니다.

dispatch 결정은 overlay의 함수 하나, `select_qmm_route`에 모였고, 이 함수가 `QuantizedMatmul::eval_gpu`와 export된 predicate `rocm::quantized_matmul_runs_dequant_gemm` 양쪽에 답을 줍니다. Rust의 적격성 판단은 bridge를 통해 이 predicate에 묻기 때문에, ROCm에서 dense 경로는 `quantized_matmul`이 이미 같은 GEMM을 실행하는 곳에서만 실행됩니다. `814f9efb`에서 실행한 unit의 전체 `make verify-rocm`은 epic #1801 실행 중 처음으로 완전히 통과한 ROCm gate였습니다.

## 1. 문제 정의

### 1.1 실패한 assertion

```
panicked at src/lib/mlxcel-core/src/layers.rs:6369:17:
assertion `left == right` failed: dtype 12 bias false: dense GEMM must match qmm bytes
```

이슈는 첫 실패 케이스(bf16, bias 없음, x `[1, 1024, 2048]`, 4-bit affine weight `[1024, 2048]`, group 64)를 측정했습니다. 1,048,576개 출력 중 499개(0.048%)가 다릅니다. 465개가 1 ULP, 10개가 2 ULP, 24개가 3에서 35 ULP이며, 모두 크기가 0.01 미만인 출력에서 나왔습니다. 이 범위에서는 상쇄 때문에 bf16 ULP 거리가 커집니다. 가장 큰 절대 오차는 0.25(41.5 대 41.75, 1 ULP)로, 최대 |out| 76.5에 비하면 작습니다. 0에 가까운 원소 하나는 부호가 바뀝니다(-8.5e-6 대 1.6e-5, bit 거리로 28308 ULP). bias 케이스는 실행되지도 않았습니다. 이 실패는 #1806(PR #2030)이 제거한 NVFP4 abort 뒤에 가려져 있었고, 그 이후 모든 전체 ROCm gate에서 실패했습니다.

### 1.2 Metal에서는 전제가 성립했고 ROCm에서는 성립하지 않은 이유

적격성 규칙은 #1994/#2001(PR #2002)에서 왔습니다. affine mode, f16 또는 bf16 입력과 같은 dtype의 scale, 2-D weight, `min_rows` 이상의 row, 32 x 32 출력 tile 512개 초과입니다. 바이트 동일성 주장은 Metal에서 실행한 in-tree sweep에 근거합니다. Metal에서는 두 경로가 같은 반올림으로 dequantize하고 tiling만 다릅니다. ROCm에서는 두 경로가 서로 다른 kernel에 도달합니다.

- Dense 쪽: `affine_dequantize` 후 `matmul`, 즉 hipBLASLt.
- qmm 쪽, bf16: 장치에 native WMMA가 있고 low-CU iGPU가 아니면 `qmm_wmma_dense_kernel`. 조건은 bf16 x/scales/biases, group 64, 4/6/8 bit, `N % 16 == 0`, `K % 64 == 0`입니다. dequantize는 `affine_dequantize`와 같지만 rocWMMA 16 x 16 x 16 tile을 통해 자체 K 순서로 f32 누적합니다.
- qmm 쪽, f16: WMMA 경로가 없으므로 `affine_dequantize`와 `dequant_rocblas_gemm`, 역시 hipBLASLt입니다.

`MLX_ROCM_WMMA_QMM=0`으로 테스트가 통과했기 때문에, 원인은 dequantize 차이가 아니라 reduction 순서 차이로 확인되었습니다. 이슈는 #2079에서 고친 CPU stream OpenBLAS 오기록 가능성도 배제했습니다. 테스트의 모든 op는 기본 GPU stream에서 실행됩니다.

### 1.3 수정 전 production 노출

`hardware.rs`의 `prefill_dense_gemm_min_rows_default`는 Apple M1에서만 임계값을 반환합니다. ROCm에서는 운영자가 `MLXCEL_PREFILL_DEQUANT_MIN_M`을 설정해야만 dense 경로가 실행되므로, 이 불일치는 기본 경로 버그가 아니라 opt-in 뒤에 숨은 정확성 구멍이었습니다.

## 2. 경로 변경을 통한 선택지 1을 고른 이유

이슈는 세 선택지 중 하나를 고르라고 요구했고, assertion을 단순히 `cfg`로 끄는 것은 금지했습니다. 선택지 1에는 조건도 붙였습니다. gfx1151에서 bf16 prefill 처리량이 퇴보하지 않아야 하며, op 수준과 실제 모델 양쪽에서 비교하라는 것입니다.

unit은 선택하기 전에 두 kernel을 측정했습니다. bf16, 4-bit g64, `[1, M, K]` 입력 하나와 `[N, K]` weight, 각 arm 40회 호출을 두 번 번갈아 실행한 평균입니다(`dense`는 매 호출마다 다시 dequantize합니다).

| M | K x N | qmm ms | dense ms | dense / qmm | differ |
|---:|---|---:|---:|---:|---:|
| 16 | 4096 x 4096 | 0.388 | 0.337 | 0.87 | 79 / 65,536 |
| 16 | 4096 x 14336 | 1.329 | 1.649 | 1.24 | 261 / 229,376 |
| 64 | 4096 x 4096 | 0.413 | 0.488 | 1.18 | 305 / 262,144 |
| 64 | 14336 x 4096 | 1.472 | 1.995 | 1.36 | 1,217 / 262,144 |
| 128 | 4096 x 4096 | 0.805 | 0.570 | 0.71 | 572 / 524,288 |
| 128 | 14336 x 4096 | 2.519 | 2.479 | 0.98 | 2,309 / 524,288 |
| 256 | 4096 x 14336 | 4.022 | 2.274 | 0.57 | 4,153 / 3,670,016 |
| 512 | 4096 x 14336 | 9.187 | 3.769 | 0.41 | 4,242 / 7,340,032 |
| 1024 | 4096 x 1024 | 1.403 | 0.320 | 0.23 | 1,164 / 1,048,576 |
| 1024 | 4096 x 14336 | 15.775 | 4.644 | 0.29 | 12,251 / 14,680,064 |
| 2048 | 14336 x 4096 | 51.491 | 10.046 | 0.20 | 29,524 / 8,388,608 |

전체 sweep 50개 cell(M 2에서 2048, K x N은 4096 x 4096, 4096 x 1024, 4096 x 14336, 14336 x 4096, 1024 x 3072)에서 WMMA kernel은 64 row 이하의 일부 shape에서만 이겼고, 128 row 이상의 모든 shape에서는 dense가 빨랐습니다(PR 기준 128 row에서 1.0x에서 3.1x, 1024와 2048에서 2.4x에서 5.1x). 모든 cell에서 바이트도 달랐으므로 불일치는 테스트 shape에 한정된 것이 아닙니다.

정확성과 속도가 같은 방향을 가리켰기 때문에 이 측정으로 선택이 정해졌습니다.

- **경로 변경을 통한 선택지 1**은 큰 bf16 GEMM을 더 빠르면서 dense 경로와 바이트가 같은 경로로 보냅니다. 변경 후 128 row 이상의 모든 cell에서 `quantized_matmul`과 dense의 차이는 0개이고, 128 row 미만은 바뀌지 않습니다.
- **WMMA kernel을 고치는 선택지 1**(hipBLASLt의 reduction 순서에 맞추기)은 kernel이 지는 바로 그 shape에서 느린 kernel을 유지했을 것이고, kernel을 hipBLASLt 내부 reduction 순서에 묶었을 것입니다. 그 순서는 라이브러리 버전 사이에서 안정된 목표가 아닙니다.
- **선택지 2**(WMMA 장치의 ROCm bf16을 적격성에서 제외)는 테스트를 정직하게 만들었겠지만 bf16 prefill을 느린 kernel에 남겨 두었고, opt-in dense 경로가 ROCm의 bf16 모델에 도움이 될 일도 없었을 것입니다.
- **선택지 3**(ULP 한계)은 dense 경로의 전제가 되는 보장을 포기합니다. 측정된 분포로는 깔끔한 한계를 정하기도 어렵습니다. 2 ULP를 넘는 출력 24개와 bit 거리 28308 ULP의 부호 반전 하나가 있으므로, 0 근처 출력에 맞춘 절대 항이 필요합니다. 이슈는 테스트를 통과시키려고 고른 한계를 명시적으로 경고했습니다. 이 선택지도 bf16 prefill을 느린 kernel에 남겨 둡니다.

128 row 미만에서는 kernel이 일부 shape에서 여전히 이기므로 그대로 두고, 대신 그곳에서는 dense 경로를 거부합니다(3.3절). 이 부분은 경로 변경이 닿지 않는 곳에만 적용한 선택지 2입니다.

## 3. 변경 요약

`fix/issue-2081-rocm-bf16-dense-gemm`의 세 커밋:

- **`a73370f2`** `fix(rocm): route large bf16 qmm to hipBLASLt and gate dense prefill`: 경로 함수, ceiling, export된 predicate, bridge 호출, 적격성 변경, ROCm 테스트 블록, 결과 페이지, 문서.
- **`814f9efb`** `fix(rocm): tighten the dense-prefill route check and its test`: 리뷰 후속 작업. predicate가 한 row GEMM(`matmul`이 hipBLASLt가 아니라 gemv로 보냄)을 거부하고, device index를 받아 bridge가 device 0이 아니라 기본 GPU를 확인합니다. route-guard 테스트는 shape마다 예상 적격성을 확인하고, 실제로 다른 shape가 하나 이상 거부되는지 확인하며, 경로와 무관한 assertion을 skip보다 먼저 실행하고, `MLX_NO_HIPBLASLT`를 경로 override로 취급합니다.
- **`c15ff36d`** `docs(rocm): note the dequant cache footprint and more route overrides`: 보안 리뷰 후속 작업. `LOCAL_FIXES.md` 항목 29에 dequantize된 weight LRU의 메모리 점유를 적고, 테스트의 override 목록에 `MLX_ROCM_FORCE_LOW_CU`와 `MLX_ROCM_FORCE_WARP_SIZE`를 추가합니다.

영역별 파일:

- Overlay: `patches-rocm/mlx/backend/rocm/quantized/qmm.hip` (`QmmRoute`, `QmmRouteInputs`, `wmma_qmm_env`, `wmma_qmm_max_m`, `select_qmm_route`, `quantized_matmul_runs_dequant_gemm`, dispatch 재배선), `rocm.h` (선언), `no_rocm.cpp` (false를 반환하는 stub), `LOCAL_FIXES.md` 항목 29.
- Bridge: `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{h,cpp}` (`quantized_matmul_matches_dense_gemm`), `src/lib/mlxcel-core/src/lib.rs` (ffi 선언).
- Rust: `src/lib/mlxcel-core/src/layers.rs` (적격성, doc comment, 테스트), `src/lib/mlxcel-core/src/hardware.rs` (doc comment만).
- 문서: `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`, `docs/environment-variables.md`, `docs/mlxcelverse/upstream/README.md` (항목 29는 아직 패키징하지 않음).

### 3.1 하나의 경로 결정

PR 전에는 `QuantizedMatmul::eval_gpu`가 세 개의 분리된 조건으로 경로를 inline에서 결정했습니다. WMMA 블록, dequant GEMM 블록, 그 안의 fp8 블록입니다. PR은 이를 `select_qmm_route(const QmmRouteInputs&, rocm::Device&)`로 옮겼고, 이 함수는 `WmmaDense`, `DequantFp8Gemm`, `DequantGemm`, `Other` 중 하나를 반환합니다. `eval_gpu`는 이제 반환값으로 분기합니다. WMMA shape 조건은 이전과 같은 논리곱입니다. 동작이 바뀌는 곳은 WMMA 분기 안의 조건 하나입니다.

```cpp
const bool hipblaslt_is_faster =
    env != 1 && dequant && in.M >= wmma_qmm_max_m(d) && !fp8();
```

이 조건이 참이면 WMMA kernel에 맞는 GEMM도 `DequantGemm`으로 갑니다. `wmma_qmm_max_m`은 `MLX_ROCM_WMMA_QMM_MAX_M`이 양의 정수면 그 값을, 아니면 `Rdna35` tier에서 128을, 그 밖에서는 `INT_MAX`를 반환합니다. 세 가지 조건이 ceiling이 측정하지 않은 경로로 GEMM을 보내지 못하게 막습니다.

- `dequant`가 참이어야 합니다. dequant 경로가 사용 가능하고 켜져 있으며(`MLX_ROCM_QMM_DEQUANT_GEMM`이 `0`이 아님) 그 shape에서 선호되거나 강제되어야 합니다. ceiling은 GEMM을 `Other`로 보내지 않습니다.
- `!fp8()`: fallback이 e4m3 경로(fp8을 지원하는 hipBLASLt가 있는 RDNA 4)가 되는 곳에서는 이전처럼 fused kernel이 남습니다. fp8 lambda는 경로를 결정하는 곳에서만 평가되므로, 그 지점에 도달하지 않는 장치는 probe되지 않습니다.
- `env != 1`: `MLX_ROCM_WMMA_QMM=1`은 ceiling을 없앱니다. low-CU iGPU가 아닌 모든 장치에서 이는 이전 dispatch와 같습니다(low-CU iGPU에서는 예전처럼 kernel을 강제로 켭니다).

### 3.2 Export된 predicate

`rocm::quantized_matmul_runs_dequant_gemm(device_index, M, N, K, x_dtype, scales_dtype, biases_dtype, group_size, bits)`는 batch 차원이 없는 transposed affine GEMM 하나에 대해 `eval_gpu`가 만드는 `QmmRouteInputs`를 다시 구성합니다. `should_use_dequant_gemm_path`와, qmv가 지원하지 않는 bit 폭에 대한 `force_dequant_gemm` 항도 포함합니다. 그리고 같은 `select_qmm_route`를 호출합니다. `QmmRoute::DequantGemm`이면서 `is_hipblaslt_available()`일 때만 true를 반환합니다.

- `M < 2`는 거부합니다. `matmul`은 transposed weight에 대한 단일 row를 hipBLASLt가 아니라 gemv로 보냅니다.
- `DequantFp8Gemm`은 거부합니다. fp8 GEMM은 activation과 weight를 e4m3로 반올림하므로 bf16 matmul과 같을 수 없습니다.
- hipBLASLt가 없으면 `dequant_rocblas_gemm`과 `matmul`의 rocBLAS 경로가 서로 다른 rocBLAS 호출로 fallback하므로, hipBLASLt 경우만 주장합니다.

두 호출자가 같은 함수를 거치므로, 입력을 같은 방식으로 재구성하는 한 predicate는 dispatch와 어긋날 수 없습니다. 환경 변수 override(`MLX_ROCM_WMMA_QMM`, `MLX_ROCM_WMMA_QMM_MAX_M`, `MLX_ROCM_QMM_DEQUANT_GEMM` 등)는 두 답을 함께 움직입니다.

### 3.3 Bridge와 Rust 적격성

bridge의 `quantized_matmul_matches_dense_gemm`은 ROCm bridge 빌드(`MLXCEL_BRIDGE_ROCM_BACKEND`)이고, 런타임 백엔드가 ROCm이며, 기본 장치가 GPU인 경우가 아니면 true를 반환합니다. ROCm에서는 마지막 두 축 앞에 1보다 큰 축이 있는 x를 거부합니다. `QuantizedMatmul`은 그 축들에 대해 batch를 돌리고, batched GEMM은 같은 row에 대해 `matmul`이 실행하는 단일 GEMM이 아니기 때문입니다. 그다음 기본 GPU의 index로 overlay predicate에 묻습니다. `no_rocm.cpp`는 ROCm이 없는 fork 빌드를 위해 predicate를 false로 stub하며, 그런 빌드는 백엔드 확인 때문에 이 지점에 도달하지 않습니다.

`prefill_dense_gemm_eligible`은 tile 규칙을 유지하고, 더 싼 row와 tile 검사가 먼저 반환한 뒤 마지막 조건으로 bridge 호출을 추가합니다. Metal과 CUDA에서는 호출이 true를 반환하므로 적격성은 이전 규칙과 정확히 같습니다. doc comment에는 acceptance criteria가 요구한 #2081의 근거(불일치 개수, ULP 분포, kernel 경로)가 들어갔고, 이 변경이 강제하는 불변식도 적혀 있습니다. dense 경로는 최적화이므로, 실행되는 곳에서는 `quantized_matmul`이 반환했을 바이트를 반환해야 합니다.

### 3.4 테스트

Metal 시절의 테스트 본문은 의도가 그대로입니다. 경로와 무관한 assertion(정확히 512 tile에서의 narrow-N 거부와 min-rows 거부)이 이제 먼저 실행됩니다. `[1, 1024, 2048]`에서의 바이트 확인은 두 dtype 모두 적격성을 여전히 확인하지만, 측정된 gfx1151 경로가 아닌 ROCm 장치에서 거부된 shape는 실패하지 않고 로그를 남긴 뒤 skip합니다.

ROCm 전용 helper `prefill_dense_gemm_rocm_route_guard`는 tile 규칙은 통과하지만 다른 경로에 도달하는 shape에서 보장 자체를 확인합니다.

| dtype | rows | N | gfx1151에서의 경로 | 예상 적격성 |
|---|---:|---:|---|---|
| bf16 | 64 | 8448 | WMMA kernel (ceiling 미만) | 거부 |
| f16 | 64 | 8448 | dequantize + hipBLASLt | 허용 |
| bf16 | 256 | 4096 | dequantize + hipBLASLt (ceiling 이상) | 허용 |

64 row에서 N 8448은 2 x 264 = 528 tile로 512 tile 하한을 살짝 넘으므로, tile 규칙이 경로 확인을 가리지 않습니다. 모든 ROCm 장치에서 helper는 허용된 shape의 바이트가 같은지 확인합니다. 측정된 경로(`rocm_measured_route()`: gfx1151이고 경로를 움직이는 일곱 개 변수가 모두 unset)에서는 각 예상 적격성과, 실제로 다른 shape가 하나 이상 있었는지도 확인합니다. 그래서 경로 확인을 제거하면 guard가 통과할 수 없습니다. 경로 확인을 우회하면 bf16 64 row에서 실패합니다. 마지막으로 x `[2, 512, 2048]`은 batch 축 때문에 ROCm에서 거부되어야 합니다.

## 4. 결과

### 4.1 모델 prefill

gfx1151에서 `mlxcel-bench-decode --prompt-tokens {512, 2048} -n 8 --warmup-tokens 4 --ignore-eos`, 바이너리 하나로 `MLX_ROCM_WMMA_QMM=1`("before", 이 장치의 이전 dispatch)과 기본값("after")을 cell마다 ABBA 3회 비교했고, sampler가 다른 GPU 프로세스가 없었음을 확인했습니다. Prefill tok/s, 평균(범위):

| 모델 | Scales | pp512 before | pp512 after | pp2048 before | pp2048 after |
|---|---|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | 1146 (1046-1199) | 2289 (2197-2359) | 961 (948-972) | 2815 (2805-2824) |
| Qwen3-0.6B-4bit | bf16 | 4581 (4230-5024) | 7633 (7147-8012) | 3059 (2894-3219) | 4236 (4027-4566) |
| Qwen3-30B-A3B-4bit | bf16 | 311 (305-315) | 326 (321-334) | 283 (282-284) | 297 (296-299) |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | 969 (879-1032) | 1008 (994-1026) | 1149 (1143-1153) | 1132 (1124-1137) |

평균으로 계산하면 Gemma 3 4B는 512 토큰에서 2.0x, 2048에서 2.9x, Qwen3-0.6B는 67%와 38% 빨라집니다. Qwen3-30B-A3B는 두 길이 모두 약 5%인데, attention projection만 이 경로를 타고 expert는 `gather_qmm`을 거치기 때문입니다.

Llama 3.1 8B는 대조군입니다. scale이 f16이라 어느 GEMM도 바뀐 분기에 도달하지 않으며(f16은 WMMA kernel을 타지 않습니다), 그 변동(512에서 범위가 겹치는 +4.0%, 2048에서 -1.5%)을 설명할 코드 경로가 없습니다. bf16의 개선폭은 이 실행 간 노이즈를 기준으로 읽어야 합니다.

Decode는 모든 cell에서 변하지 않습니다. M이 1이고, 이는 WMMA kernel을 탄 적이 없습니다. MLX peak 메모리는 Qwen3-0.6B 512 토큰(1.04에서 1.46 GB)을 빼면 0.1 GB 이내입니다. dequantize 경로가 weight matrix마다 bf16 사본을 할당하기 때문이고, 이는 `LOCAL_FIXES.md` 항목 28(PR #2084)이 제한합니다.

### 4.2 Production 노출: `MLXCEL_PREFILL_DEQUANT_MIN_M`

ROCm에서 dense 경로는 `MLXCEL_PREFILL_DEQUANT_MIN_M`이 설정되어야만 실행됩니다. 이번 변경 후 그곳에서 허용되는 projection은 `quantized_matmul`이 이미 dequantize + hipBLASLt를 실행하는 것뿐이므로, 변수를 켜도 바이트가 바뀔 수 없습니다. 속도도 의미 있게 바뀌지 않습니다. 두 경로가 같은 GEMM을 실행하고, `quantized_matmul`은 dequantize된 weight cache와 함께 실행합니다. 같은 바이너리로 `MLXCEL_PREFILL_DEQUANT_MIN_M=1024`와 unset을 pp2048에서 ABBA 3회 비교한 결과: gemma-3-4b-it-4bit 2871 대 2852 tok/s, Qwen3-0.6B-4bit 4174 대 4185.

따라서 production 상황은 다음과 같습니다.

- **기본 설정(변수 unset)**: ROCm에서 dense 경로는 전후 모두 실행되지 않습니다. 사용자에게 보이는 효과는 bf16 prefill의 더 빠른 `quantized_matmul` 경로입니다.
- **변수 설정**: PR 전에는 운영자가 허용된 projection에서 `quantized_matmul`과 다른 bf16 출력(#2081의 불일치)을 얻을 수 있었습니다. PR 후에는 이 변수가 바이트와 속도 모두에 영향이 없습니다. `docs/environment-variables.md`와 `hardware.rs` doc comment는 ROCm에서 설정해도 이득이 없다고 적고 있고, ROCm 기본값은 꺼진 채로 유지됩니다.

## 5. 기술적 선택과 그 이유

- **세 선택지 중에서 고르기 전에 kernel을 측정했습니다.** 선택지 1에 붙은 처리량 조건이 결정적 근거가 되었습니다. 바이트 동일성을 깬 kernel이 128 row 이상에서는 더 느린 쪽이기도 했습니다.
- **kernel을 다시 쓰지 않고 경로를 바꿨습니다.** 이미 배포된 두 경로 사이에서 shape를 옮기는 것은 새 수치 코드가 필요 없고, 불투명한 라이브러리의 reduction 순서를 맞추는 대신 구조적으로(양쪽이 같은 GEMM) 동일성을 보장합니다.
- **dispatch와 predicate에 함수 하나를 씁니다.** dispatch 조건을 다시 적은 별도 predicate는 누군가 한쪽을 처음 고치는 순간 어긋납니다. `select_qmm_route`를 공유하면 Rust 적격성은 dispatch 자체에 묻는 질문이 됩니다.
- **hipBLASLt 경우만 주장합니다.** predicate는 한 row GEMM, fp8 경로, hipBLASLt 없는 fallback, batched 입력을 거부합니다. 각각은 상위 수준의 경로 이름이 같아도 두 쪽이 다른 코드에 도달하기 때문입니다.
- **ceiling을 측정한 tier로 제한했습니다.** 기본 128 row는 `Rdna35`에만 적용됩니다. RDNA 3, RDNA 4, CDNA는 `MLX_ROCM_WMMA_QMM_MAX_M`을 설정하지 않는 한 모든 row 수에서 kernel을 유지하고, fp8 fallback도 제외되므로, 측정하지 않은 장치가 기본값으로 처리량을 잃지 않습니다.
- **이전 dispatch를 정확히 재현하는 스위치를 남겼습니다.** `MLX_ROCM_WMMA_QMM=1`은 이 장치에서 PR 이전 경로를 복원하며, 그 덕분에 바이너리 하나로 prefill 표의 두 열을 모두 만들 수 있었습니다.
- **ROCm에서 `MLXCEL_PREFILL_DEQUANT_MIN_M` 기본값은 꺼 둡니다.** 경로가 같으므로 ROCm에서 dense 경로가 더할 것이 없다는 것을 4.2절에서 측정했습니다.

## 6. 검증

PR 본문 기준, gfx1151(Radeon 8060S):

- 대상 테스트가 기본값과 `MLX_ROCM_WMMA_QMM=0`, `MLX_ROCM_WMMA_QMM=1`, `MLX_NO_HIPBLASLT=1`, `MLX_ROCM_FORCE_LOW_CU=1`에서 통과합니다.
- `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings` 통과.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt` 통과, `cargo test --features rocm --test dead_doc_pointers` 통과.

Orchestrator 검증:

- unit이 `814f9efb`에서 전체 `make verify-rocm`을 실행했습니다(gfx1151). 146개 test suite, 11,785개 통과, 0개 실패, 378개 ignored, smoke OK. epic #1801 실행 중 처음으로 완전히 통과한 ROCm gate이며, 직전 baseline 실패가 바로 이 테스트였습니다(PR #2084의 `3c9edea0` gate는 정확히 이 target에서만 실패했습니다).
- head 커밋 `c15ff36d`는 `LOCAL_FIXES.md` 문장 하나와 테스트 override 목록의 환경 변수 이름 두 개만 추가합니다. 그 커밋에서 대상 테스트, clippy, `verify-rocm-overlay`, `verify-fmt`가 통과했습니다.
- Orchestrator는 머지 후 새 빌드에서 최종 gate를 실행합니다.

## 7. 남은 위험과 검증하지 않은 부분

- **경로는 graph를 만들 때 판단합니다.** `prefill_dense_gemm_eligible`은 graph를 만들 때 predicate에 묻고, `dequant_rocblas_gemm`은 eval 시점에 hipBLASLt 사용 가능 여부를 다시 읽습니다. 그 사이에 stream capture가 시작되면 두 쪽이 서로 다른 rocBLAS fallback으로 갈 수 있습니다. 이 백엔드는 HIP graph가 꺼져 있으므로(`use_hip_graphs()`가 false 반환), 이 상황은 decode capture 중에 `MLXCEL_PREFILL_DEQUANT_MIN_M`이 설정되어 있어야 생깁니다. predicate는 hipBLASLt launch 자체가 throw하지 않는다고도 가정합니다. throw하면 각 쪽이 자기 rocBLAS fallback을 탑니다.
- **Dequantize된 weight cache의 메모리 점유.** 이 경로의 LRU(matrix 8개 또는 256 MB, `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE`, `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`)는 prefill 동안 모델의 projection 사이에서 hit가 없으므로, 각 projection은 prefill chunk마다 다시 dequantize되고, 마지막 항목들(최대 256 MB)은 prefill 후에도 살아 있습니다. f16 checkpoint는 이미 그렇게 동작했고, 이제 RDNA 3.5의 bf16 checkpoint도 그렇습니다. `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0`으로 끌 수 있습니다. 임시 사본은 projection 크기에 비례해 커지며, 호스트에 4B보다 큰 dense bf16 scale checkpoint가 없어서 그런 모델의 peak 메모리는 측정하지 않았습니다.
- **다른 RDNA tier는 측정하지 않았습니다.** ceiling은 `Rdna35` tier 전체(`gfx1150`에서 `gfx1152`)에 적용되지만 gfx1151에서만 측정했습니다. low-CU gfx1152는 강제하지 않는 한 어차피 WMMA kernel을 건너뜁니다. RDNA 3, RDNA 4, CDNA는 이전 dispatch를 유지합니다. gfx1151이 아닌 ROCm 장치에서는 테스트의 바이트 확인이 거부된 shape를 실패 대신 skip하므로 커버리지가 약합니다.
- **64와 128 사이의 row.** 공개된 표에는 그 사이 row 수의 cell이 없습니다. 128은 모든 shape에서 dense가 이긴 첫 측정 row 수입니다.
- **Metal과 CUDA는 이 호스트에서 실행하지 않았습니다.** Rust 적격성과 테스트는 그 경로에서도 바뀌었지만, bridge가 그곳에서 true를 반환하고 ROCm 블록은 ROCm에서만 실행됩니다.
- **Upstream 반영.** 항목 29는 fork용으로 패키징하지 않았습니다. ceiling은 장치 하나에서만 측정했고, export된 predicate는 mlxcel의 dense prefill 확인을 위해 존재합니다. fork PR은 ceiling만 담고, RDNA 3 또는 RDNA 3.5 장치 하나 이상의 측정을 추가해야 합니다.

## 8. 학습 포인트

- **바이트 동일성 주장은 수학이 아니라 kernel에 대한 주장입니다.** #2001의 sweep은 Metal에서 유효했습니다. 다른 백엔드에서는 같은 두 op가 다른 kernel에 도달했고, 전제에는 tile 규칙이 아니라 백엔드 수준의 확인이 필요했습니다.
- **정확성 수정과 성능 문제가 같은 knob을 공유하면 둘 다 먼저 측정해야 합니다.** 여기서는 테스트를 고친 근거가 이슈가 묻지 않았던 2.9x prefill 격차도 없앴습니다.
- **dispatch에 대한 predicate는 dispatch에 직접 묻게 만들어야 합니다.** 조건을 Rust에 옮겨 적지 않고 같은 경로 선택을 실행하는 함수를 export했기 때문에, 환경 변수 override나 이후 tier가 경로를 옮겨도 적격성이 정직하게 유지됩니다.
- **guard 테스트에는 반드시 잡아야 할 mutation을 줘야 합니다.** route-guard는 측정된 경로에서 실제로 다른 shape가 하나 이상 있는지 확인하므로, 경로 확인을 지우면 테스트가 조용히 통과하지 않고 실패합니다.

Refs: #2081, #1801, #1994, #2001, #2002, #1806, #2030, #2079, #2084.
