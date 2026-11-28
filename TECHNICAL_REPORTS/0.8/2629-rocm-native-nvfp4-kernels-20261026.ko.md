# PR #2629: ROCm 네이티브 NVFP4 qmm, gather_qmm 커널

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 중간

## 요약

ROCm에는 NVFP4 커널이 없어 ModelOpt NVFP4 체크포인트를 로드할 때 affine 4비트로 다시 패킹했고, MLX 네이티브 NVFP4 플레인은 거부했습니다(#1806 정책). 이제 ROCm 오버레이가 `QuantizedMatmul`과 `GatherQMM`에서 NVFP4(E2M1 값, 16개 그룹마다 E4M3 블록 스케일, expert별 선택적 float32 글로벌 스케일)를 네이티브로 실행하며, `quant_mode_support(Rocm, Nvfp4)`는 `Native`입니다(이슈 #2617, LOCAL_FIXES 63번).

## 1. 문제 정의

`QuantizedMatmul::eval_gpu`는 그룹 크기 32, 64, 128만 처리했고, `qdequant.hpp`에는 E4M3 디코드가 없었으며, `GatherQMM`은 글로벌 스케일 입력을 읽지 않았고, non-affine gather 경로는 그룹 32를 요구했습니다. NVFP4 플레인을 로드하는 테스트 6개는 ROCm에서 `expect_native_nvfp4_load`로 건너뛰는 분기를 탔고, `switch_layers.rs`의 `validate_native_global_scale`은 Metal과 CUDA만 허용했습니다.

## 3. 기술적 선택과 그 이유

기존 tiled, warp-shared 템플릿에 다섯 번째 모드를 끼워 넣는 대신 전용 파일 `quantized/nvfp4_qmv.hip`에 `nvfp4_qmv_kernel`과 `nvfp4_dequantize_kernel`을 두었습니다. `nvfp4_qmv_kernel`은 블록당 최대 8행을 계산하고 expert가 바뀔 때만 가중치 그룹을 다시 읽으므로, M = 1 디코드, 업스트림 Metal #4712가 겨냥한 6에서 16행의 speculative-decode 검증 폭, 정렬된 gather prefill을 한 커널로 처리합니다. 행 수가 기준을 넘으면 `select_qmm_route`가 NVFP4를 기존 dequantize + hipBLASLt/rocBLAS 경로로 보내며, 이 경로는 전치되지 않은 가중치도 처리합니다.

E2M1 디코드는 mxfp4의 scaled half bits 방식을 재사용하고(업스트림 Metal #4709), 새 E4M3 디코드는 분기 없이 subnormal까지 정확합니다. 글로벌 스케일은 `mlx/ops.cpp`의 `fp_quantize`/`fp_dequantize`와 CUDA `fp_gather_qmv`를 따라 블록 스케일에 `global_scale[expert] / (448 * 6)`을 곱하며, 출력마다 한 번 적용합니다. MLX는 `QuantizedMatmul`에 글로벌 스케일 입력을 넘기지 않으므로(dense 글로벌 스케일은 그래프 수준에서 적용), non-affine 모드의 네 번째 입력은 이제 무시하지 않고 거부합니다.

커널은 16 lane 열 그룹, 폭 16 shuffle만 쓰고 barrier가 없으며 `WARP_SIZE`를 읽지 않으므로 wave32와 wave64에서 같은 결과를 냅니다. wavefront 제한은 추가하지 않았고, 이는 측정이 아닌 분석에 근거합니다.

리뷰에서 호스트 측 형상 검사를 추가했습니다. MLX의 `validate_quantized_input`은 배치 접두와 마지막 축만 비교하므로, 스케일 플레인이 크면 스케일 크기로 grid를 잡던 dequantize 커널이 출력 밖에 쓸 수 있었습니다. `check_nvfp4_planes`는 스케일 플레인이 `w.shape[:-1] + [w.shape[-1] / 2]`이고 1바이트 원소인지 확인합니다. 16의 배수 검사는 양자화된 축(`transpose ? K : N`)에 적용하고, grid x는 UINT32_MAX / 16으로 제한합니다.

## 7. 변경 요약

- `overlays/rocm/mlx/backend/rocm/quantized/nvfp4_qmv.hip`(신규): qmv, dequantize 커널과 레이아웃, 플레인 검사.
- `overlays/rocm/mlx/backend/rocm/quantized/qmm.hip`: `QuantizedMatmul`과 `GatherQMM`의 NVFP4 분기, 경로 선택, gather 검사 순서.
- `overlays/rocm/mlx/backend/rocm/quantized/qdequant.hpp`: E4M3 디코드, fp4 디코드 이동.
- `mlxcel-core/src/hardware.rs`: `(Rocm, Nvfp4)`가 `Native`, 표 테스트 갱신. `layers.rs`, `ffi_tests.rs`, `src/models/sanitize.rs`, `sanitize_tests.rs`: ROCm 변환을 가정하던 테스트가 해당 경로를 명시적으로 구성.
- `src/models/switch_layers.rs`: `validate_native_global_scale`이 ROCm 허용.
- `tests/rocm_nvfp4_quant.rs`, `tests/rocm_nvfp4_quant/gather.rs`(신규): CPU dense 기준과 비교하는 수치 테스트.
- 문서: 설치 문서의 ROCm 표, 환경 변수, 지원 모델.

## 검증

rebase 후 오버레이를 다시 빌드한 gfx1151에서 `rocm_nvfp4_quant` 7개 모두 통과(CPU `x @ dequantize(w).T` 대비 최악 상대 오차 f32 2.1e-6, f16 4.9e-4, bf16 5.4e-3. qmv M = 1, 다중 행, dequantize + GEMM, 배치, 비전치, 정렬 및 비정렬 gather의 글로벌 스케일 유무, 부분 열 타일, 범위 밖 expert의 0 출력, 잘못된 스케일 플레인 거부 포함). 이전에 건너뛰던 테스트 6개가 네이티브 분기로 통과합니다. `rocm_mxfp4_quant`, `rocm_hip_grid_clamp gather_qmv`, 다섯 가지 계약 검사, clippy 모두 깨끗합니다.

미검증: 실제 NVFP4 체크포인트는 실행하지 못했습니다(캐시에 없고 다운로드는 범위 밖). wave64는 실행하지 않았습니다. 정렬된 NVFP4 gather prefill은 8행마다 expert 가중치를 다시 읽으므로, expert-batched 경로가 있던 기존 affine 변환보다 느릴 수 있습니다. `nvfp4_qmv_kernel`은 대부분의 기존 gather 커널처럼 `lhs_indices`를 활성값 배치 크기와 비교하지 않습니다.
