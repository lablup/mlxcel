# 기술 보고서: PR #2298 - MLX 핀 업데이트 및 오버레이 베이스 커밋 선언

- **날짜**: 2026-10-14
- **상태**: 완료; 구현 검토 및 전체 컨트랙트 게이트 통과
- **위험 수준**: 낮음 (모든 업스트림 변경 3-way 병합 완료, 32개 오버레이 드리프트 0 검증, 로컬 ROCm 오버레이 및 단위 테스트 전원 통과)

## 개요

PR #2298은 `src/lib/mlx-cpp/CMakeLists.txt`에서 고정된 업스트림 `ml-explore/mlx` 커밋을 `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`에서 `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`로 업데이트합니다 (14개 업스트림 커밋). `src/lib/mlx-cpp/patches/` 및 `src/lib/mlx-cpp/patches-cuda/` 하위의 32개 전체 오버레이 파일에 기계 판독 가능한 `// Modified from upstream MLX <sha> <path>` 헤더를 선언하고, 각 베이스 커밋을 업스트림 히스토리와의 diff를 통해 정밀하게 식별하였습니다. 해당 구간에서 충돌하거나 드리프트가 발생한 9개 오버레이는 mlxcel의 커스텀 커널 및 런타임 변경점을 온전히 보존하면서 새 핀으로 3-way 병합되었습니다. 업스트림 수정사항인 `dfe17bafb` (#3912)가 온전히 유지됨을 확인하였습니다. ROCm 오버레이는 새 핀으로 리타깃되었고 신규 업스트림 `fast::GatedDeltaUpdate` 프리미티브에 대한 폴백 스텁이 추가되었으며, `CONTRIBUTING.md`에 CUDA/Metal/CPU 오버레이 수명 주기와 3-way 병합 절차가 문서화되었습니다.

## 1. 문제 정의

### 1.1 배경
`mlxcel` 저장소는 `configure_file(... COPYONLY)`를 사용하여 업스트림 `ml-explore/mlx`에 파일 단위 소스 오버레이를 적용합니다. 업스트림 `GIT_TAG`를 갱신할 때 오버레이를 명시적으로 리베이스하지 않으면 업스트림의 소스 변경사항이 조용히 덮어씌워져 유실될 위험이 있습니다. 기존에는 ROCm 오버레이만 체계적으로 베이스와 드리프트를 추적하고 있었으며, CUDA, Metal, CPU 오버레이에는 복사 또는 리베이스 시점의 커밋이 명시되지 않았습니다.

### 1.2 문제점
- **오버레이 베이스 선언 부재**: 32개 오버레이 중 29개 파일에 기계 판독 가능한 `Modified from upstream MLX` 헤더가 없어 자동 드리프트 감지가 불가능했습니다.
- **오래된 리듀서 오버레이**: 3개의 CUDA reduction 오버레이(`all_reduce.cu`, `col_reduce.cu`, `row_reduce.cu`)가 `v0.31.1` (`ce45c525`)에 머물러 있어 업스트림의 const-correctness 및 64비트 계획 개선사항이 누락되어 있었습니다.
- **업스트림 핀 갱신 필요**: 업스트림 MLX에 양자화 연산의 글로벌 스케일 지원(#4507), Gated Delta Net 커널(#4020), 32의 배수가 아닌 양자화 차원 지원(#3912) 등 주요 개선사항이 추가되었습니다.
- **ROCm 빌드 심볼 정합성**: 신규 추가된 `fast::GatedDeltaUpdate` 프리미티브에 대해 ROCm 백엔드 스텁이 없으면 정적 링킹 시 정의되지 않은 심볼 에러가 발생합니다.

## 2. 구현 상세

### 2.1 1단계: 기계 판독 가능한 베이스 선언
`src/lib/mlx-cpp/patches/` 및 `src/lib/mlx-cpp/patches-cuda/` 하위의 32개 전체 오버레이 파일에 다음 형태의 헤더 주석을 선언하였습니다:
```cpp
// Modified from upstream MLX <sha> <upstream/relative/path>
```
각 SHA는 `ml-explore/mlx` 업스트림 히스토리와의 diff를 통해 확정되었으며, 32개 SHA 모두 업스트림 저장소에서 유효하게 확인됩니다.

### 2.2 2단계: 드리프트 오버레이 3-way 리베이스
기존 베이스와 `59d600b5e` 사이에서 변경된 9개 오버레이를 `59d600b5e`로 3-way 병합하였습니다:
1. `patches/mlx/backend/metal/quantized.cpp`: 업스트림 글로벌 스케일 로딩을 통합하고 3개의 `MLXCEL_QMV_WIDE` 런타임 오프스위치 헝크(#1186, #1187)를 유지.
2. `patches/mlx/backend/cuda/device/qmm_naive.cuh`: Volta sm_70 `MmaPath` 템플릿 매개변수화(#1543) 유지.
3. `patches/mlx/backend/cuda/quantized/qmm/qmv.cu`: `global_scale` 파라미터 전달을 추가하고 multirow qmv(#725), 행 윈도우 캡(#906), pre-Ampere 누적 최적화(#1539)를 온전히 유지.
4. `patches/mlx/backend/cuda/quantized/quantized.cpp`: 65535 그리드 청킹(#648), 입력 연속성 검증, 정렬된 MoE 프리필 grouped-GEMM 경로(#629) 유지.
5. `patches-cuda/fast.cpp`: `gated_delta_update` 폴백 구현 추가.
6. `patches-cuda/ops.cpp`: `gather_qmm` GPU 디바이스 유효성 검사 및 `GatherQQMM` 인수 목록 갱신.
7. `patches/mlx/backend/cuda/reduce/{all_reduce,col_reduce,row_reduce}.cu`: 업스트림 `const_cast` 제거 및 대규모 입력에 대한 64비트 all-reduce 계획 반영.

드리프트 검증 결과 32개 오버레이 전원에 대해 베이스와 `59d600b5e` 간의 커밋 차이가 0임을 확인하였습니다. 업스트림 `dfe17bafb` (#3912)는 `fp_quantized.h` 및 `quantized.h`가 오버레이되지 않아 새 핀에서 온전히 반영됨을 확인했습니다.

### 2.3 3단계: 핀 갱신 및 ROCm 오버레이 리타깃
- `src/lib/mlx-cpp/CMakeLists.txt`의 `GIT_TAG`를 `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`로 갱신.
- `patches-rocm/UPSTREAM`의 `retargeted_to_mlx_pin`을 `59d600b5e`로 갱신.
- `patches-rocm/mlx/backend/rocm/primitives.cpp`에 `fast::GatedDeltaUpdate::use_fallback` (`true` 반환) 및 `NO_GPU_MULTI(GatedDeltaUpdate)` 추가 (LOCAL_FIXES.md 항목 6에 기록).
- `patches-rocm/CORE_RESIDUAL.diff` 갱신.

### 2.4 4단계: 문서화
`CONTRIBUTING.md`에 `Reconciling CUDA, Metal, and CPU source overlays` 서브섹션을 추가하여 다음 내용을 기술하였습니다:
- `// Modified from upstream MLX <sha> <path>` 헤더 규약.
- 오버레이 드리프트 자동 탐지 셸 루프.
- `git merge-file`을 활용한 3-way 병합 절차.
- 업스트림에 반영된 오버레이의 제거 지침.

## 3. 검증 결과

다음 검증 항목들이 모두 정상 통과되었습니다:
1. `bash scripts/ci/mlx_pinned_commit.sh` 실행 시 `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` 출력.
2. `cargo test -p mlxcel-mlx-pin` 전체 27개 테스트 통과 (26개 단위 테스트 + 1개 크로스 파서 통합 테스트).
3. 32개 오버레이 전체 드리프트 검사 시 0 `NO BASE`, 0 `DRIFT` 출력.
4. `make verify-rocm-overlay` 실행 시 29개 점검 항목 정상 통과 (109개 백엔드 + 15개 코어 파일, 45개 LOCAL_FIXES 항목, 4개 잔차 일치 확인).
5. 컨트랙트 게이트 통과: `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-python-tooling verify-fmt`.
6. `git diff --check origin/main..HEAD` 공백 에러 및 충돌 마커 없음 확인.
7. AMD `gfx1151` 환경에서 탐욕적 텍스트 생성 테스트 결과 핀 업데이트 전후 토큰 단위 동일 출력 확인.
