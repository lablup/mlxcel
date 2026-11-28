# 기술 보고서: Issue #2108 - CPU 전용 Linux build에서 kv_inplace_write link 실패

**날짜**: 2026-10-05

**상태**: GB10에서 구현 및 검증 완료 (CPU 전용 build와 CUDA build). origin/main `33c45053` 기준, 머지 대기 중.

**언어**: C++ (bridge preprocessor guard), Rust (`build.rs`, build-policy predicate와 unit test), Markdown (설치 문서)

**위험도**: 낮음 (GPU translation unit은 미리 정의되는 macro 하나를 제외하면 바뀌지 않습니다. CPU 전용 build에는 어떤 코드 경로로도 도달할 수 없는, 예외를 던지는 `eval_gpu`가 생깁니다)

## 요약

`mlxcel-core/build.rs`는 모든 backend에서 `src/lib/mlx-cpp/turbo/kv_inplace_write.cpp` (#1959)를 compile합니다. 이 파일의 `InplaceSliceWrite::eval_gpu`는 `mlx::core::copy_gpu_inplace`를 호출하는데, MLX는 이 함수를 GPU backend가 있을 때만 build되는 `mlx/backend/gpu/copy.cpp`에서만 정의합니다. 그래서 `cuda`, `metal`, `rocm` feature가 모두 없는 Linux build는 link 단계에서 실패했습니다. 이번 수정은 MLX가 Metal, CUDA, ROCm 중 하나를 build할 때만 `build.rs`가 정의하는 `MLXCEL_BRIDGE_GPU_BACKEND`를 추가하고, `copy.h` include와 `eval_gpu` 본문을 그 뒤에 둡니다. 정의되지 않으면 `eval_gpu`는 `eval_cpu`와 마찬가지로 예외를 던집니다.

## 1. 문제

이 버그는 #2093을 검증하던 중 발견되었습니다. 당시 CPU 전용 test binary를 link하려면 `--unresolved-symbols=ignore-in-object-files`가 필요했습니다. CI는 이를 잡을 수 없었습니다. `ubuntu-latest` job은 lint만 하고, 모든 GB10 job은 `--features cuda`를 넘기며, 남은 `cargo check` 단계는 link하지 않습니다.

이 branch에서 main의 `kv_inplace_write.cpp`를 되돌려 놓고 `CARGO_TARGET_DIR=target/cpu cargo test -p mlxcel-core --profile test-fast --lib --no-run`을 실행해 재현했습니다. `InplaceSliceWrite::eval_gpu`에서 나온 `undefined reference to mlx::core::copy_gpu_inplace(...)` 오류 두 개로 link가 실패했고, 다른 미해결 symbol은 없었습니다. bridge가 참조하는 GPU 전용 MLX symbol은 `copy_gpu_inplace` 하나뿐입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `build_support/metal_backend.rs` | `resolve_metal_backend` 옆에 순수 predicate `gpu_backend_enabled(metal_backend, cuda, rocm)` 추가 |
| `build.rs` | 그 predicate로 `MLXCEL_BRIDGE_GPU_BACKEND`를 정의. Metal은 Cargo feature가 아닌 resolve된 `metal_backend` 값을, CUDA와 ROCm은 `build_mlx`가 `MLX_BUILD_CUDA`, `MLX_BUILD_ROCM`에 그대로 대응시키는 feature를 사용 |
| `kv_inplace_write.cpp` | `#include <mlx/backend/gpu/copy.h>`와 `eval_gpu` 본문을 `#ifdef MLXCEL_BRIDGE_GPU_BACKEND` 아래에 둠. 아니면 `eval_gpu`가 `std::runtime_error`를 던짐 |
| `build_support/metal_backend_tests.rs` | test 세 개: CPU 전용은 false, backend 하나만 켜도 true, `MLXCEL_BUILD_METAL=OFF`이고 다른 backend가 없는 macOS는 false |
| `docs/installation.md` | 로컬 명령과 측정 비용을 적은 "CPU-only link check" 절 추가 |

## 3. 설계 결정

### 3.1 Metal, ROCm define을 재사용하지 않고 새 define을 둔 이유

`MLXCEL_BRIDGE_METAL_BACKEND`와 `MLXCEL_BRIDGE_ROCM_BACKEND`는 backend 전용 header를 막습니다. `copy_gpu_inplace`는 세 GPU build 모두에 있는 backend 중립 GPU helper이므로 조건은 합집합이어야 하고, CUDA에는 bridge define 자체가 없었습니다. macOS는 Cargo feature 없이도 Metal을 켜고 `MLXCEL_BUILD_METAL=OFF`는 feature가 있어도 끄므로 (#1988), predicate는 Metal을 resolve된 값으로 판단합니다.

### 3.2 파일을 계속 compile하는 이유

issue는 `.file()` 줄을 빼는 방법을 기각했습니다. `cpp/mlx_cxx_ext.cpp`가 header를 include하고 cxx FFI symbol `inplace_slice_write`는 계속 link되어야 하기 때문입니다. runtime에는 모든 Rust 호출 지점 (`cache.rs`의 decode write와 rotating warmup write, `cache/paged.rs`의 slab write)이 이미 `ffi::default_device_is_gpu()`를 요구하고, `inplace_slice_write` 자체도 GPU가 아닌 기본 device를 거부합니다. 따라서 CPU 전용 build에서 예외를 던지는 `eval_gpu`에는 도달할 수 없습니다.

### 3.3 GPU build의 source는 그대로

include는 원래 위치에 guard로 감싼 채 남았고, 원래 본문은 `#else` 쪽에 그대로 있습니다. define이 설정되면 preprocess된 translation unit은 main과 같습니다. 개발 호스트에서 build할 수 없는 Metal과 ROCm에 대한 근거가 이것입니다.

## 4. 검증

| 검사 | 결과 |
|---|---|
| CPU 전용 `cargo test -p mlxcel-core --profile test-fast --lib --no-run` (별도 `target/cpu`) | link 성공. CPU 전용 MLX tree 포함 cold 2분 24초, bridge 파일을 고친 뒤 다시 link하는 데 6.0초 |
| main의 `kv_inplace_write.cpp`로 같은 명령 | 실패: `undefined reference to mlx::core::copy_gpu_inplace` (2곳) |
| CPU 전용 `cargo build --release` | link 성공, cold 9분 48초 |
| CPU 전용 `metal_backend_build_policy` test | 7개 통과 |
| CPU 전용 `inplace` test | 4개 통과 (CPU 경로는 `slice_update`를 유지) |
| CPU 전용 `cargo clippy -p mlxcel-core --release --lib --tests -- -D warnings` | 경고 없음 |
| CUDA 검증 | 5절 참조 |

## 5. CUDA 회귀 검사

| 검사 | 결과 |
|---|---|
| `cargo build --release --features cuda --bin mlxcel-server` | build 및 link 성공 (cold 17분 41초) |
| CUDA `kv_inplace_write.o` | 여전히 `mlx::core::copy_gpu_inplace`를 참조하고 CPU 전용 오류 문자열이 없으므로 GPU 본문이 compile됨. CPU 전용 object는 그 반대 |
| `cargo test --release -p mlxcel-core --features cuda --lib`, filter `inplace`, `gpu-lock` 아래 `--test-threads=1` | 4개 통과 |
| 같은 binary, filter `cache::` | 541개 통과 |
| 같은 binary, filter `metal_backend_build_policy` | 7개 통과 |
| `cargo clippy --release --features cuda --lib --tests -- -D warnings` | 경고 없음 |

## 6. 검증하지 못한 부분

GB10 호스트에서는 Metal과 ROCm을 build할 수 없었습니다. Metal: 기본 macOS build에서는 `metal_backend`가 true이므로 define이 설정되고 source는 main과 같습니다. ROCm: `CARGO_FEATURE_ROCM`이 define을 설정하고 `build_mlx`가 `MLX_BUILD_ROCM=ON`으로 MLX를 build하며, 그 GPU backend가 `copy_gpu_inplace`를 제공하므로 이 경우도 source는 main과 같습니다.

## 7. CI

이 PR은 CI 단계를 추가하지 않았습니다. 같은 실행에서 #2111이 `.github/workflows/ci.yml`을 수정합니다. 제안하는 단계는 자체 persistent target dir (`ci.yml:657-661`의 패턴, 예: `$HOME/.cargo-target/mlxcel-cpu-link-ci`)를 쓰는 GB10 job에서 `cargo test -p mlxcel-core --profile test-fast --lib --no-run`을 실행하는 것입니다. 로컬에서 측정한 warm 실행 비용은 bridge C++ 변경 뒤 6.0초, `mlxcel-core/src/lib.rs` 변경 뒤 6.6초에 checkout 시간을 더한 정도이고, 처음 (cold) 실행은 약 2.5분입니다. 비용이 충분히 작으므로 #2111에서 이 단계를 추가할 것을 권장합니다. 그 단계가 들어가기 전까지는 `docs/installation.md`에 명령을 문서화해 두었습니다.
