# PR #2516: PR #2505가 남긴 세 가지 계약 위반 수리

**작성일**: 2026-10-21
**상태**: 완료
**위험도**: High (계약/FFI)
**언어**: Rust, C++

---

## 요약

PR #2505가 CI 동결 상태에서 병합되면서 저장소 전역 계약 위반 세 가지를 남겼다. 이 PR은 세 가지를 하나의 변경으로 수리한다: `mlx_cxx_kernels.cpp`의 커널 포트 디스패치 위반 두 곳은 Metal 전용 포트 테이블과 `-> Result<()>` 브리지 계약으로 `mlxcel::select_kernel_port`를 경유하도록 라우팅되어(포트 없는 백엔드가 abort 대신 에러를 받음), 새 환경변수 `MLXCEL_INKLING_MOE_KERNEL`은 연산자 롤백 토글로 문서화되고, `inkling/mlp.rs`와 `switch_layers.rs`의 fmt 위반은 기계적으로 수정되었다. 모든 계약 타깃이 이제 exit 0.

---

## 1. 문제 정의

### 1.1 배경

PR #2505 (`perf(inkling): decode via fast SDPA and fused MoE kernels`)는 GitHub Actions가 동결된 상태(2026-10-20 15:03 UTC 이후 저장소 전역에 실행 없음)에서 병합되어 계약 표면을 아무것도 검사받지 못했다. 다음 PR 헤드에서 돌린 풀 게이트가 세 가지 위반을 한 번에 잡았다 (CT_EXIT=2).

### 1.2 기존 문제점

- `make verify-kernel-port-dispatch` 실패: `mlx_cxx_kernels.cpp:1660`과 `:1773`이 `mlxcel::select_kernel_port`를 경유하지 않고 커널 홀더에 직접 도달해서, 포트 없는 백엔드가 조용히 Metal arm을 실행하게 된다(이 패턴이 막으려는 크래시 클래스, #1801/#1803/#1885/#2018/#2147).
- `make verify-python-tooling` 실패: 새 환경변수 `MLXCEL_INKLING_MOE_KERNEL` (`inkling/mlp.rs:142`)이 `docs/environment-variables.md`에도 문서화되지 않고 allowlist에도 없었다.
- `cargo fmt --all -- --check` 실패: `inkling/mlp.rs` 4곳, `switch_layers.rs` 1곳.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| `noexcept` extern을 건너는 C++ throw가 프로세스를 종료 | High (라우팅된 커널 위의 모든 비-Metal MoE 요청) | 이 수정 전 비-Metal 호스트에서 결정론적 |
| 연산자가 롤백 토글을 발견할 수 없음 | Low | High |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

클린. 새 포트 테이블 두 개는 함수 로컬 `static const`(magic-static 초기화 후 불변), 캡처 없는 람다를 일반 함수 포인터로, 공유 가변 상태 없음. `select_kernel_port`가 도달 가능한 모든 throw를 체계적으로 감사한 결과, 브리지 선언된 모든 런처가 `-> Result<...>`임을 확인 (fused_xielu, bitlinear_matmul, ssm_update_kernel, mamba1_selective_scan, fused_add3_layer_norm, fused_moe_forward, fused_moe_expert/geglu, 그리고 여기서 고친 두 개). C++ 내부 헬퍼는 Result 선언 엔트리나 `has_kernel_port` 가드 아래에서만 도달 가능. C++ 시그니처는 `void` + `noexcept` 없음을 유지해서 cxx 트램펄린이 throw를 `Err`로 변환.

### 2.2 성능 관점

무시 가능. `select_kernel_port` 성공 경로는 한 번 해석된 static 백엔드 enum 읽기 한 번 + switch, 기존 직접 호출이 수행하던 홀더 접근과 동일. `std::string` 생성은 throw 경로에만 존재. 부수 이득: 비-Metal 호스트가 이제 Metal 홀더를 아예 만지지 않음(기존 직접 호출은 `fast::metal_kernel(...)`을 무조건 등록했고, 이것이 크래시 원인이자 낭비였음).

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 라우팅된 커널은 Metal 전용. 올바르게 게이트된 Metal 호스트에서는 동작 불변. 비-Metal 호스트는 처음부터 그래프 경로를 탐(이전: abort).
- **New Dependencies**: 없음.

### 2.4 코드 품질 관점

- 테스트: inkling 필터 92 passed / 0 failed (CUDA). Metal 게이트 패리티 테스트는 설계된 skip 경로를 탐. 모든 계약 타깃을 이 트리에서 재검증 exit 0.
- fmt: 순수 줄바꿈 재배치 5 hunk, 의미 변화 없음.

---

## 3. 기술적 선택과 그 이유

### 3.1 `-> Result<()>` 브리지 선언 (필수, 스타일 아님)

**컨텍스트:** `select_kernel_port`는 entry point와 fallback을 명명하는 포트가 없으면 throw한다. `void noexcept` extern이면 throw가 FFI 경계를 건너 프로세스를 종료시킨다.

**선택:** 두 브리지 선언을 `-> Result<()>`로 변경 (`fn inkling_moe_route_kernel`, `unsafe fn inkling_moe_down_combine_kernel` 유지; cxx는 unsafe+Result 허용, 선례 `fused_moe_forward`). C++ 시그니처는 `void` 유지(저장소 내 `try_eval` 패턴과 일치)해서 cxx가 호출을 감싸고 throw된 MLX 예외를 Rust `Err`로 변환. `kernel_port.h` 용어로 문서화(런처 시그니처 변경 후 `verify-kernel-port-dispatch` 게이트가 요구).

### 3.2 호출 지점의 Err 폴백

Route: `is_ok().then_some((idx, wk, g))`, `None` arm이 `route_weights`(커널 이전 그래프 라우팅 체인)로 폴백. null UniquePtr은 `then_some`이 버리는 즉시 구축 튜플로 이동 — 역참조 없음, 누수 없음. Down-combine: `.is_ok().then(|| reshape(&out, ...))`는 lazy라서 `Err` 시 `reshape`가 평가되지 않음. `forward_with_expert_scales` + `moe_weighted_sum`으로 폴백. 직접 호출 테스트는 `kernel_port.h:81-82` 계약대로 gate-then-expect.

### 3.3 Down-combine을 실제 fast-route 성공에 게이트 (리뷰 채택)

초기 down-combine 디스패치는 env+백엔드 게이트(`route_kernel`)에 게이트되어 실제 fast-route 성공과 무관했다. 게이트는 참인데 런처가 Err인 열화 상태에서 거부될 런처를 시도하게 된다. 리뷰에서 채택: `fast_route_ok`(튜플을 소비하는 match 이전에 캡처)에 게이트. 올바르게 게이트된 호스트에서 동작 불변, 열화 상태는 즉시 그래프 경로로. 두 런처 doc 주석은 생산 경로가 폴백을 통해 `Err`를 소비함을 명시("only from a direct call" 표현 폐기).

### 3.4 문서화, allowlist 아님

`MLXCEL_INKLING_MOE_KERNEL`은 연산자 대면 롤백 토글(0 비활성, unset/기타 활성, 기본 on)로 `MLXCEL_SSM_KERNEL`과 같은 범주. 따라서 allowlist가 아니라 `docs/environment-variables.md` 문서화(`MLXCEL_FUSED_MOE` 계열 옆). doc 행의 값/기본값/효과를 코드 대조로 검증.

---

## 4. 구현 상세

**파일: `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`** — `inkling_route_ports()`와 `inkling_down_combine_ports()` (`.metal` 설정, `.cuda = nullptr, .rocm = nullptr` 명시, `turbo4_delegated_ports` 템플릿). 두 직접 홀더 호출을 `mlxcel::select_kernel_port`로 교체.

**파일: `src/lib/mlxcel-core/src/lib.rs`** — 두 브리지 선언 `-> Result<()>`. 런처 doc은 Err-생산경로-폴백-소비 계약 명시.

**파일: `src/models/inkling/mlp.rs`** — route 호출을 `route_weights` 폴백으로 재구성. down-combine은 gather 경로 폴백. `down_parts`는 `fast_route_ok`에 게이트. 직접 호출 테스트 gate-then-expect.

**파일: `docs/environment-variables.md`** — "Hardware and kernel diagnostic variables" 섹션에 `MLXCEL_INKLING_MOE_KERNEL` 행.

**파일: `src/models/switch_layers.rs`** — 공백만 재배치.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| make verify-kernel-port-dispatch | exit 0 (런처 12개 파일 라우팅, exempt 0, 음성 커버리지 케이스 통과) |
| make verify-python-tooling | exit 0 (139 tests OK) |
| make verify-fmt / cargo fmt --all -- --check | exit 0 / clean |
| 결합 계약 타깃 | exit 0 |
| scoped clippy (lib+tests, -D warnings) | exit 0 |
| inkling 필터 (CUDA, gpu-lock, TF32=0, --test-threads=1) | 92 passed / 0 failed (Metal 게이트 패리티 테스트는 설계대로 skip) |
| 풀 게이트 (161 bins, HEAD 60c744eb) | build 0, clippy 0, doc 0, CT_EXIT=0. 160/161 (mllama_parity 7/1 = 알려진 #2467 flake, main 컨트롤로 환경적임이 입증) |
| 리뷰 채택 | `a2ca2104`: down-combine은 `fast_route_ok`에 게이트, 두 런처 doc 정밀화. 재검증 fmt 0 / clippy 0 / inkling 92/0 |
| 보안 리뷰 | CLEAN. FFI 감사 clean, 폴백 safe, 호출당 비용 무시 가능 |

라우팅된 커널 자체는 이 CUDA 호스트에서 실행 불가(Metal 전용). 그래프 경로 테스트는 통과, 실커널 검증은 Metal 호스트 작업으로 남는다.

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 5 (+후속 커밋 2) |
| 라인 | 약 +120/-30 |

### 관련 커밋

| 해시 | 타입 | 메시지 |
|------|------|---------|
| `60c744eb` | fix | fix: repair the three contract breakages left by PR #2505 |
| `a2ca2104` | fix | fix(inkling): gate down-combine on fast-route success, precise Err docs |

---

## 7. 후속 조치

### 모니터링 필요

- 다음 릴리스 주기의 첫 Metal 호스트 런처에서 라우팅된 커널(route parity, down-combine) 확인.
- CI 동결이 지속되는 동안 추가 계약 드리프트 감시: 각 유닛의 풀 게이트가 대체.

### 향후 개선 사항

- `mod gemma3n_math` (src/lib/mlxcel-xla/src/emitter/mod.rs:59) dead-code 정리 (#2513 리뷰에서 나온 것, 별도 표면).

---

## 부록

### A. 리뷰 판정

pr-reviewer: 6개 초점 영역 전부 PASS, CRITICAL/HIGH 없음. LOW 하나의 doc 표현 관찰(`a2ca2104`에 채택). pr-security-checker: CLEAN, CRITICAL/HIGH/MEDIUM 없음. 선택적 LOW 효율 nit 하나(down_parts 강화, `a2ca2104`에 채택). FFI 감사 결과 `select_kernel_port` 도달 가능 throw가 이미 전부 Result 선언임을 확인.
