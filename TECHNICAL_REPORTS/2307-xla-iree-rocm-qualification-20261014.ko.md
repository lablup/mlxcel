# PR #2307: OpenXLA/IREE 엔진 ROCm/HIP 환경 적격성 검증

- **Date**: 2026-10-14
- **Status**: Implemented, reviewed, formatted, and verified
- **Risk**: Medium (빌드 설정, HIP HAL 런타임 링크, 타깃 검증)

## Summary

PR #2307은 이슈 #851에 정의된 전 단계를 거쳐 AMD ROCm/HIP 환경(`MLXCEL_XLA_DEVICE=hip`, `gfx1151` / AMD Radeon 8060S Graphics, ROCm 7.2.2)에서 OpenXLA/IREE 실행 백엔드(`--features xla-backend,xla-iree`) 적격성을 검증합니다. 자동화된 IREE 툴체인 설정(`scripts/iree/setup-rocm.sh`, `make iree-rocm`), 런타임 RPATH 및 HIP HAL 드라이버 링크, `iree_status_fprint` 기반 다중행 진단 메시지, 아키텍처 및 정밀도 검증, `#574` 기반 퓨전 특성 분석, 종합 적격성 보고서를 제공합니다.

## Problem

본 PR 이전에는 mlxcel의 OpenXLA/IREE 실행이 CPU(`local-task`, `local-sync`), Apple Silicon(`metal`), NVIDIA CUDA(`cuda`)에서만 지원되었습니다. AMD ROCm 호스트에서 `MLXCEL_BACKEND=xla`를 실행할 경우 다음 문제들이 발생했습니다:
- 공식 배포 prebuilt IREE 바이너리에는 HIP HAL 드라이버가 포함되지 않아 `IREE_HAL_DRIVER_HIP=ON`으로 빌드된 고정 버전의 `libiree_runtime_unified.a` 소스 빌드가 필요했습니다.
- IREE의 HIP 드라이버는 런타임에 `dlopen`을 통해 `libamdhip64.so`를 동적으로 로드하므로, 수동 `LD_LIBRARY_PATH` 지정이나 명시적인 ELF `DT_RPATH` 설정이 요구되었습니다.
- 잘못된 타깃 SKU나 디바이스 권한 누락 시 이전에는 잘리거나 불분명한 오류 메시지가 출력되어 원인 파악 및 조치가 어려웠습니다.
- AMD 하드웨어 환경에서 수치적 일치성, 연속 배칭, HTTP 서빙, 메모리 안정성을 검증하는 종합 검증 체계가 부재했습니다.

## Design and Implementation

- **단계 0 및 1: 툴체인 프로비저닝 (`scripts/iree/setup-rocm.sh`, `Makefile`)**: `IREE_HAL_DRIVER_HIP=ON` 및 `IREE_BUILD_COMPILER=OFF` 옵션으로 IREE `3.12.0rc20260721` (`dc9601f88654749456c7cee4ae87e13de2654e1e`)을 `~/.cache/mlxcel/iree-rocm-3.12.0rc20260721` 경로에 복제 및 빌드합니다. `iree-compile`용 일치 버전의 `iree-base-compiler` 휠을 다운로드하고 체크섬을 검증합니다. `rocminfo`와 `amdgpu-arch`를 통해 단일 `gfx*` 아키텍처를 자동 감지합니다.
- **링크 및 RPATH (`build.rs`, `src/lib/mlxcel-xla/build.rs`, `src/lib/mlxcel-core/build_support/rocm_rpath.rs`)**: `IREE_ROCM_HOME`을 수용하여 `libiree_hal_drivers_hip_registration_registration.a`를 링크하고, `-Wl,--disable-new-dtags,-rpath,/opt/rocm/lib`를 전달하여 `DT_RPATH`를 삽입함으로써 `LD_LIBRARY_PATH` 없이 `dlopen("libamdhip64.so", ...)`이 해석되도록 합니다.
- **오류 진단 (`src/lib/mlxcel-xla/csrc/xla_iree.c`)**: 512바이트 고정 버퍼 대신 `iree_status_fprint(stderr, s)`를 사용하여 전체 트레이스백을 출력하고, 구체적인 환경 변수 조치 가이드를 담은 단계별 어노테이션(`[runtime-init]`, `[module-load]`, `[weight-upload]`)을 추가합니다.
- **검증 및 정밀도 (`src/lib/mlxcel-xla/src/emitter/builder.rs`, `src/lib/mlxcel-xla/src/iree.rs`)**: `MLXCEL_XLA_ROCM_TARGET`에서 단일 SKU `gfx*` 식별자를 강제하고, `hip` 디바이스의 기본 정밀도를 `f16`으로 설정하며, `MLXCEL_XLA_PRECISION`의 공백을 제거하고 유효하지 않은 값을 거부합니다.
- **특성 분석 및 적격성 검증 (`scripts/xla/fusion_spike.sh`, `docs/benchmark_results/xla_iree_rocm_qualification.md`)**: `hip` 환경에서 패킹된 int8 역양자화를 측정(1,209개 flow 디스패치, 업스트림 퓨전 미지원)하고, 데이터 타일링 실패가 CUDA `#574`와 일치함을 확인하였으며, 벤치마크 및 토큰 일치성 지표를 문서화했습니다.

## Review Findings and Fixes

1. **포맷팅 검사 (`cargo fmt --check`)**: `builder.rs`, `iree.rs`, `phi4_audio.rs`의 줄바꿈을 조정하여 `cargo fmt --check`를 통과하도록 수정했습니다.
2. **정밀도 오버라이드 공백 처리 (`builder.rs`)**: `precision_env_override`에서 공백을 trim하여 `MLXCEL_XLA_PRECISION=" f16 "`이 `Precision::F16`으로 올바르게 매칭되도록 수정하고, 프로세스 환경 변수를 변경하지 않고 테스트할 수 있도록 순수 함수 `resolve_precision_checked_with_override`를 추출하고 단위 테스트를 추가했습니다.
3. **적격성 지원 판정 명시 (`xla_iree_rocm_qualification.md`)**: 이슈 #851 요구사항에 맞추어 Section 6 ("Support Decision & Support Boundaries")을 추가하고 최종 판정(**Conditionally Supported**) 및 프로덕션 vs 실험적 지원 경계를 명확히 정의했습니다.
4. **ROCm rpath 타깃 OS 분기 (`rocm_rpath.rs`)**: 환경 변수 `IREE_ROCM_HOME`이 설정된 상태에서 macOS 빌드를 수행할 때 지원되지 않는 `-Wl,--disable-new-dtags` 링커 플래그가 주입되지 않도록 `if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") { return; }` 가드를 추가했습니다.
5. **`MLXCEL_XLA_IREE_COMPILE` 비어있지 않음 검증 (`iree.rs`)**: 빈 문자열 또는 공백만 포함된 환경 변수가 설정되었을 때 알 수 없는 오류 대신 명확한 설정 안내가 노출되도록 `trim()` 후 빈 문자열 여부를 검증하도록 개선했습니다.
6. **타깃 아키텍처 식별자 정규화 (`builder.rs`)**: `validate_rocm_target`에서 `.to_ascii_lowercase()`를 적용하여 대문자 입력(`GFX1151`, `gfx90A`)도 안정적으로 파싱되도록 개선하고 단위 테스트를 추가했습니다.
7. **다운로드 신속 실패 및 임시 파일 정리 (`setup-rocm.sh`, `validate_arch.sh`)**: `setup-rocm.sh`의 curl에 `-f` 플래그를 추가하여 HTTP 오류 시 즉시 실패하도록 하고, `validate_arch.sh`에서 `--oracle`이 전달되지 않았을 때 생성되는 `mktemp` 임시 파일을 스크립트 종료 시 정리하도록 exit trap을 등록했습니다.

## Verification

- `cargo fmt --check`: 차이점 없이 통과했습니다.
- `cargo test -p mlxcel-xla --lib`: 282개 통과, 0개 실패, 11개 무시되었습니다.
- `cargo clippy -p mlxcel-xla --lib --tests -- -D warnings`: 경고 없이 통과했습니다.
- `bash scripts/xla/validate_arch.sh --structural-only`: 통과했습니다.
- 하드웨어 검증 결과가 `docs/benchmark_results/xla_iree_rocm_qualification.md`에 기록되었습니다(`gfx1151`에서 64/64 토큰 일치성 통과, B8 연속 배칭 82.06 tok/s, HTTP SSE 스트리밍 정상 동작, 20회 반복 테스트에서 VRAM 누수 0 확인).
