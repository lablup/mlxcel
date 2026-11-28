# PR #2623: ROCm eager complex64와 bool 변환을 MLX와 일치시킴

**날짜**: 2026-10-26
**상태**: 완료
**언어**: C++ (HIP), Rust
**위험도**: 낮음

## 요약

ROCm eager 변환 커널은 complex64를 bool로 바꿀 때 `&&`를 쓰고, bool을 complex64로 바꿀 때 true를 `1+1i`로 만들었습니다. MLX CUDA와 CPU는 `real != 0 || imag != 0`와 `1+0i`를 쓰며, ROCm fused 경로도 LOCAL_FIXES 59번(#2613) 이후 이를 따릅니다. 이번 변경으로 ROCm 오버레이의 두 `CastOp` 특수화가 같은 의미를 갖게 되어 eager와 compiled 결과가 일치합니다(이슈 #2616, LOCAL_FIXES 60번).

## 1. 문제 정의

eager ROCm 경로에서 `astype(z, bool)`은 `1+0i`와 `0+1i`에 대해 false를 반환했고, `astype(true, complex64)`는 `1+1i`를 반환했습니다. 59번 항목이 fused 커널의 `complex64_t`를 CUDA에 맞춘 뒤로는 같은 그래프가 ROCm에서 eager로 실행할 때와 `mx::compile`로 실행할 때 서로 다른 값을 냈습니다.

## 3. 기술적 선택과 그 이유

기준은 MLX CUDA `mlx/backend/cuda/device/cast_op.cuh:22-39`와 CPU `complex64_t::operator bool`(`mlx/types/complex.h:50`)입니다. 둘 다 실수부나 허수부 중 하나라도 0이 아니면 참으로 보고, bool에서 complex로의 변환은 실수 1로 정의하므로 ROCm 오버레이도 독자 의미 대신 이를 따릅니다.

테스트는 변환 대상을 eager로 미리 만든 입력으로 넘깁니다(`mlx_cxx_compile_probe.cpp`의 프로브 그래프 12, 13). 그래서 eager 모드는 fused 그래프가 아니라 `copy.hpp`의 단독 변환 커널을 실행합니다. 그래프 13은 두 성분을 `real + 10 * imag` 하나의 float로 묶어, 예전 `1+1i`는 11로, 올바른 `1+0i`는 1로 드러나게 합니다.

## 7. 변경 요약

- `overlays/rocm/mlx/backend/rocm/copy/copy.hpp`: complex64에서 bool로는 `||`, true는 `make_hipFloatComplex(1.0f, 0.0f)`.
- `overlays/rocm/LOCAL_FIXES.md`: 60번 항목.
- `mlxcel-core/cpp/mlx_cxx_compile_probe.cpp`: eager 입력을 쓰는 프로브 그래프 12, 13.
- `mlxcel-core/src/compiled_dtype_cast_tests.rs`: `eager_and_compiled_complex64_to_bool_follow_mlx`(1+0i, 0+1i, 0+0i가 true, true, false)와 `eager_and_compiled_bool_to_complex64_follow_mlx`(true가 1+0i). 두 테스트 모두 compiled 결과가 eager와 같은지도 확인합니다.

## 검증

수정 전 gfx1151에서 새 테스트 두 개가 모두 실패했습니다. complex에서 bool로의 eager 결과는 `[1, 1, 0]` 대신 `[0, 0, 0]`이었고, bool에서 complex로는 compiled가 1일 때 eager가 11이었습니다. 수정과 오버레이 재빌드 후 `compiled_dtype_cast_tests` 10개가 모두 통과했습니다. 계약 검사(`verify-llama-compat`, `verify-env-var-docs`, `verify-kernel-port-dispatch`, `verify-kernel-dtype-keys`, `verify-rocm-overlay`)와 `mlxcel-core` lib/tests clippy도 깨끗합니다. ROCm 오버레이만 바뀌었으며 CUDA, Metal, CPU 동작은 그대로입니다.
