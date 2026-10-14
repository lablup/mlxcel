# 기술 보고서: MLX 계층을 `src/lib/mlxcelverse` 구조로 재편 (`overlays/` 및 `kernels/`)

- **PR**: #2305
- **이슈**: #1816
- **날짜**: 2026-10-14
- **분류**: 리팩토링 / 빌드 시스템 및 아키텍처

---

## 1. 요약 (Executive Summary)

기존 `src/lib/mlx-cpp/` 디렉터리는 백엔드별 업스트림 MLX 소스 오버라이드와 mlxcel 자체 커스텀 커널이 점진적으로 추가되면서 `patches/`, `patches-cuda/`, `patches-rocm/`, `turbo/`의 네 하위 디렉터리가 혼재된 형태를 띠고 있었다. 이로 인해 어떤 파일이 CMake 구성 시점에 업스트림 MLX 체크아웃 위로 덮어써지는 전체 파일 오버레이이고 어떤 파일이 mlxcel 자체 C++ 브리지 라이브러리로 직접 컴파일되는지 디렉터리 트리 구조만으로는 구분하기 어려웠으며, CUDA 오버레이는 `patches/mlx/backend/cuda/`와 `patches-cuda/` 두 경로로 나뉘어 있었다.

PR #2305는 `src/lib/mlx-cpp/`를 `src/lib/mlxcelverse/`로 재편하여 역할을 명확히 분리한 2계층 구조를 도입한다:
- `src/lib/mlxcelverse/overlays/{cpu,metal,cuda,rocm}/mlx/...`: CMake 구성 단계에서 `mlx_apply_source_overlays`를 통해 `${CMAKE_BINARY_DIR}/_deps/mlx-src` 위로 복사되는 업스트림 MLX 전체 파일 오버레이.
- `src/lib/mlxcelverse/kernels/`: `src/lib/mlxcel-core/build.rs`에 의해 `mlx_cxx_bridge`로 컴파일되는 mlxcel 자체 C++/Metal/CUDA/HIP 융합 커널, 런처 및 백엔드 포트 디스패치.

기존 `src/lib/mlx-cpp/` 하위 파일 전체를 `git mv`로 이동하여 Git 커밋 이력을 보존했고, 계층 구조와 오버레이 규칙을 설명하는 4개의 신규 `README.md`(`src/lib/mlxcelverse/README.md`, `src/lib/mlxcelverse/overlays/{cpu,metal,cuda}/README.md`)를 추가했으며, 빌드 스크립트·CI 워크플로·정적 검증 게이트·ROCm 오버레이 동기화 도구·문서·테스트 픽스처를 일관되게 갱신했다.

---

## 2. 디렉터리 구조 변환 내역

| 기존 경로 | 신규 경로 | 역할 |
|---|---|---|
| `src/lib/mlx-cpp/CMakeLists.txt` | `src/lib/mlxcelverse/CMakeLists.txt` | 고정된 업스트림 MLX를 가져오고, 소스 오버레이를 적용하며 `libmlx`를 빌드하는 최상위 CMake 프로젝트 |
| `src/lib/mlx-cpp/patches/mlx/backend/cpu/` | `src/lib/mlxcelverse/overlays/cpu/mlx/backend/cpu/` | CPU 백엔드 오버레이 (`quantized.cpp`, 스칼라 양자화 matmul의 float 누적, #2248) |
| `src/lib/mlx-cpp/patches/mlx/backend/metal/` | `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/` | Metal 백엔드 오버레이 (4개 파일: `compiled.cpp`, `device.cpp`, `kernels/utils.h`, `quantized.cpp`) |
| `src/lib/mlx-cpp/patches/mlx/backend/cuda/` + `src/lib/mlx-cpp/patches-cuda/*.cpp` | `src/lib/mlxcelverse/overlays/cuda/mlx/` | 단일 루트로 통합된 CUDA 백엔드 오버레이 (`backend/cuda/**` 24개 파일, 및 MLX 코어 파일 `mlx/{dtype,fast,ops}.cpp`) |
| `src/lib/mlx-cpp/patches-rocm/` | `src/lib/mlxcelverse/overlays/rocm/` | `NripeshN/mlx@rocm-support`에서 벤더링된 ROCm 백엔드 오버레이 (`mlx/backend/rocm/**` 109개 파일, 코어 파일 15개, 기록 문서 `UPSTREAM`, `LOCAL_FIXES.md`, `CORE_RESIDUAL.diff`) |
| `src/lib/mlx-cpp/turbo/` | `src/lib/mlxcelverse/kernels/` | mlxcel 자체 융합 커널, 런처 및 백엔드 포트 디스패치 (31개 파일) |

### `CMakeLists.txt` 오버레이 적용 로직

`src/lib/mlxcelverse/CMakeLists.txt`의 `mlx_apply_source_overlays(mlx_source_dir)` 함수는 각 백엔드 오버레이를 `overlays/<backend>/`에서 가져와 적용한다:
- `overlays/metal/`: Metal JIT 타입 프로모션 수정, 커맨드 버퍼 예산 제어, SIMD 그룹 매트릭스 헬퍼, `MLXCEL_QMV_WIDE` 런타임 오프스위치.
- `overlays/cpu/`: CPU 양자화 matmul float 누적 (Metal, CUDA, ROCm, CPU 전용 등 모든 빌드에 공통 적용).
- `overlays/cuda/mlx/backend/cuda/`: 커스텀 grouped GEMM, 양자화 커널, reduce 연산, binary 연산 (무조건 복사되며 non-CUDA 빌드에서는 무해).
- `overlays/cuda/mlx/*.cpp`: `dtype.cpp`, `fast.cpp`, `ops.cpp`는 bf16 비전 가중치의 Metal 정밀도를 보존하기 위해 `MLX_BUILD_CUDA`가 `ON`일 때만 복사.
- `overlays/rocm/`: `MLX_BUILD_ROCM`이 `ON`일 때 109개 ROCm 백엔드 파일과 15개 MLX 코어 파일을 복사하며 캐시된 소스 트리의 불필요한 오래된 파일 정리 수행.

---

## 3. 빌드, 도구 및 검증 파이프라인 갱신

1. **Rust 빌드 스크립트 및 FFI 브리지 (`src/lib/mlxcel-core/`)**:
   - `build.rs`: CMake 프로젝트 경로를 `Config::new("../mlxcelverse")`로 갱신, C++ 커널 런처 소스를 `../mlxcelverse/kernels/*.cpp`로 갱신, include 경로를 `../mlxcelverse/kernels`로 갱신, `cargo:rerun-if-changed` 감시 목록을 신규 경로로 갱신.
   - `build_support/mlx_pin.rs`: CMakeLists 경로 상수를 `../mlxcelverse/CMakeLists.txt` 및 `src/lib/mlxcelverse/CMakeLists.txt`로 갱신.
   - `Cargo.toml`: ROCm 주석 경로를 `../mlxcelverse/overlays/rocm`으로 갱신.
   - `src/lib/mlxcel-core/cpp/*.{cpp,h}`: `#include` 경로를 `"../../mlx-cpp/turbo/..."`에서 `"../../mlxcelverse/kernels/..."`로 변경.
2. **CI 워크플로 및 정적 검증 게이트**:
   - `.github/workflows/ci.yml` 및 `.github/workflows/release.yml`: 경로 필터(`mlxcelverse/**`, `overlays/metal/**`, `overlays/cuda/**`, `overlays/cpu/**`, `kernels/**`), MLX 핀 검증 단계, 하드웨어 컨버터 경로 검증 갱신.
   - `Makefile`: `verify-rocm-overlay` 문서 및 사전 조건 갱신.
   - `scripts/ci/mlx_pinned_commit.sh`: 기본 CMake 경로를 `src/lib/mlxcelverse/CMakeLists.txt`로 갱신.
   - `scripts/ci/{check_cuda_arch_lists.py,check_kernel_dtype_keys.py,check_kernel_port_dispatch.py}` 및 테스트 스크립트: 런처 대상 경로와 프로브 위치를 `src/lib/mlxcelverse/kernels/`로 갱신.
   - `scripts/mlxcelverse/{rocm_overlay.py,rocm_overlay_test.sh,sync_from_fork.sh}`: 기본 오버레이 경로를 `src/lib/mlxcelverse/overlays/rocm`으로 갱신.
3. **테스트 및 문서**:
   - `src/lib/mlxcel-mlx-pin/tests/cross_parser.rs`, `src/lib/mlxcel-core/src/cache/paged_pool_offset_tests.rs`, `src/execution/memory_estimate_inflight_tests.rs`의 경로 인용 갱신.
   - `src/lib/mlxcelverse/README.md` 및 `src/lib/mlxcelverse/overlays/{cpu,metal,cuda}/README.md` 신규 추가, `CONTRIBUTING.md`, `NOTICE`, `docs/*` 갱신 (기존 벤치마크 로그, ADR, 이전 기술 보고서는 원본 유지).

---

## 4. 검증 결과

- **오버레이 바이트 동등성**: `mlx_apply_source_overlays`를 통해 `_deps/mlx-src`로 복사되는 소스 파일들이 재편 전후로 코드상 동일함을 검증 (Metal 4개, CPU 1개, CUDA 27개, ROCm 125개).
- **단위 테스트 및 컨트랙트 게이트**:
  - `bash scripts/ci/mlx_pinned_commit_test.sh` 및 `cargo test -p mlxcel-mlx-pin` (27/27 통과).
  - `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-python-tooling verify-fmt` (Python 도구 테스트 123개 및 cargo fmt 포함 전체 통과).
  - `bash scripts/mlxcelverse/rocm_overlay_test.sh` (29/29 통과) 및 `python3 scripts/mlxcelverse/rocm_overlay.py drift --no-fetch` (드리프트 0개 통과).
  - `python3 scripts/insert_apache_header.py --check` (라이선스 헤더 검사 통과).
  - `python3 scripts/ci/check_cross_repo_refs.py` (통과).
  - `python3 scripts/ci/check_cuda_arch_lists.py` (7개 CUDA 아키텍처 목록 검증 통과).
- **ROCm 하드웨어 스모크 테스트**:
  - `scripts/ci/rocm_smoke.sh` (`Qwen3-0.6B-4bit`, AMD Radeon 8060S `gfx1151`): 32개 토큰을 0.13초 만에 생성 완료 (`255.73 tok/s`).
