# 기술 보고서: PR #1169 - test: acquire the env lock in backend seam tests for xla-backend builds

**작성일**: 2026-08-16
**작성자**: AI Code Reviewer
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

`src/backend/tests.rs`의 테스트 네 개가 크레이트 전역 env 락을 잡지 않은 채 `select_backend()`를 호출했다. 옵트인 `xla-backend` / `experimental-backend` 기능에서는 이 함수가 `MLXCEL_BACKEND`를 읽으므로(`src/backend/mod.rs:316`), 전체 스위트를 병렬 실행하면 같은 락을 쥔 채 `MLXCEL_BACKEND=xla`를 일시적으로 설정하는 `muse_glimmer_startup_rejects_xla_backend_selection`과 인터리브되어 백엔드 테스트가 엉뚱한 백엔드를 받을 수 있다. 이 PR은 PR #1157이 확립한 패턴 그대로 네 테스트 모두에서 락을 잡는다. 단일 `#[cfg(test)]` 모듈에 26줄 추가, 삭제 0줄인 테스트 전용 변경이며 CI가 빌드하는 기능 조합에서는 동작이 바뀌지 않는다.

보안 검토에서 원래 이슈가 예상하지 못한 두 번째 근거가 드러났다. 락으로 감싼 호출 경로들이 `MLXCEL_BACKEND`와 무관하게 자체적으로 환경변수에 접근하므로, 이 락은 애초의 경쟁 상태와 별개로 Rust 2024의 `setenv`/`getenv` 규칙 아래서도 옳다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1159는 PR #1157(#1155 구현)의 보안 리뷰에서 발행됐다. 그 PR이 `src/server/muse_glimmer_startup_guard_tests.rs`에 크레이트 전역 env 락을 도입하면서, 이 잔여 경쟁 상태는 범위 밖으로 명시하고 기록만 남겼다. env를 변경하는 쪽은 더 앞선 #1116에서 들어왔고, 백엔드 테스트는 가드 테스트 강화보다 먼저 존재했다.

### 1.2 기존 문제점

- **문제 1**: `mlx_session_threads_the_token_bias_through`(변경 전 `src/backend/tests.rs:104`)가 락 없이 `select_backend()`를 호출했다. `xla-backend`에서 경쟁적으로 `MLXCEL_BACKEND=xla`가 보이면 `Session::Xla`가 반환되어 `unreachable!("select_backend defaults to MLX without MLXCEL_BACKEND=xla")` 갈래에 진입한다.
- **문제 2**: 같은 노출이 44, 62, 81행의 형제 테스트 세 개에도 적용된다. 이슈는 수정을 지시하는 대신 검토 대상으로 제시했다.
- **문제 3** (보안 검토에서 발견): 락으로 감싼 경로들이 자체적으로 환경변수를 읽고 쓴다. `create_session`은 `boundary_v_layers_from_env()`(`src/lib/mlxcel-core/src/cache/turbo/boundary.rs:83`, `std::env::var` 두 번)에 도달하고, 실물 체크포인트에서의 `load_model`은 `maybe_disable_cuda_graphs_for_model`(`src/loading/mod.rs:509`)을 통해 `set_var("MLX_USE_CUDA_GRAPHS")`를 호출할 수 있다. 이 크레이트는 edition 2024이고, 거기서 `set_var`가 `unsafe`인 이유가 바로 동시 읽기가 unsound하기 때문이다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| 옵트인 기능 빌드에서 헛실패가 나고 실제 백엔드 회귀로 오독됨 | Low | Low (`xla-backend`/`experimental-backend`를 빌드하는 CI 워크플로 없음) |
| 어떤 `Err`든 받아들이는 테스트가 엉뚱한 백엔드를 조용히 검증 | Low | Low (동일한 기능 게이팅) |
| edition 2024에서 동시 `setenv`에 대해 동기화되지 않은 `getenv` | Low | Low (테스트 바이너리 한정) |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

보안 표면이 없다. 변경분 전체가 `#[cfg(test)]` 안이고 `test_support`는 `src/lib.rs:51-52`에서 `cfg(test)`로 게이팅되므로 프로덕션 코드가 `env_lock()`에 닿을 수 없다. 보안 검토는 모든 심각도에서 지적 사항 0건을 반환했고, 각 가드 아래 호출 트리를 전부 추적해 동시성 쟁점을 정리했다.

- `select_backend()`는 `std::env::var` 한 번과 ZST 생성이 전부다.
- `load_model`은 존재하지 않는 경로에 대해 `get_model_type`에서 빠져나오며, 락도 네트워크도 없다.
- `create_session`은 구조체 생성(`KVCache::new_with_mode`)과 thread-local 스트림 준비까지만 도달한다.
- 그 서브트리의 유일한 `static Mutex`(`sparse_v_count_file`, `src/lib/mlxcel-core/src/cache/turbo/sparse_v.rs:335`)는 어텐션 경로에서만 닿고 캐시 생성 경로에서는 닿지 않는다.

재진입은 구조적으로 불가능하며, `mlxcel-core`는 별도 테스트 바이너리에서 자체 `ENV_LOCK`을 갖기 때문에 크레이트 간 락 상호작용도 없다.

**발견된 이슈:**
| 이슈 | 심각도 | 상태 |
|------|--------|------|
| 없음 | - | - |

참고용으로 두 건을 기록하되 조치하지 않았다. 설명 주석이 실패 지점으로 `unreachable!` 갈래를 지목하지만 실제로는 한 줄 앞의 `.expect("session creation must succeed")`가 먼저 발화할 것이다(둘 다 요란하게 실패하며, 이 서술은 이슈 본문에서 온 것이다). 그리고 두 테스트에서 새 근거 주석이 기존 테스트 목적 주석에 빈 줄 없이 이어 붙었다.

### 2.2 성능 관점

무시할 수준이다. 가드가 걸린 테스트는 프로세스 기동을 포함해 각각 10ms 미만에 끝나고, 가드 아래에서 네트워크 I/O나 sleep, 스레드 join을 하는 코드가 없다. 테스트 바이너리 내 다른 `env_lock` 호출 지점과의 경합 증가도 유의미하지 않다.

### 2.3 호환성/의존성 관점

- **호환성 파괴**: 없음
- **신규 의존성**: 없음
- **호환성**: 프로덕션 경로에 영향 없음. 기본 기능 조합에서는 `select_backend`가 환경변수를 읽지 않고 단일 MLX 변형으로 상수 접힘되므로 락은 no-op이다

### 2.4 코드 품질 관점

- **테스트 커버리지**: 개수는 그대로. 기존 네 테스트가 옵트인 기능 빌드에서 결정적으로 동작하게 됐다
- **코드 복잡도**: 테스트당 바인딩 하나와 설명 주석 하나
- **기술 부채**: 감소. 크레이트에 남아 있던 마지막 무방비 `select_backend()` 호출 지점이 모두 덮였다

---

## 3. 기술적 선택과 그 이유

### 3.1 이슈에 명시된 하나가 아니라 네 호출 지점 전부에 가드 적용

**고려한 대안:**

| 선택지 | 장점 | 단점 |
|--------|------|------|
| `mlx_session_threads_the_token_bias_through`만 처리 | 변경 최소, 이슈 제목과 문자 그대로 일치 | 동일한 노출을 가진 테스트 세 개가 남아 같은 이슈가 재발 |
| **채택: `select_backend()` 호출 지점 네 곳 모두 처리** | 개별 사례가 아니라 결함 유형을 닫음 | 변경분이 약간 커짐 |

**선택 이유:** 이슈가 44, 62, 81행도 같은 가드가 필요한지 명시적으로 물었다. 반사적으로 붙이지 않고 각각의 실패 양상을 따로 판단했다. `select_backend_resolves_to_mlx_under_default_features`는 경쟁 시 `matches!(backend, Backend::Mlx(_))` 어서션에서 바로 실패한다. `mlx_backend_creates_a_session_and_advertises_batched_serving`은 첫 어서션에서 실패하는데, `XlaBackend::supports_batched_serving()`이 `cfg!(feature = "xla-iree")`(`src/backend/xla.rs:117`)여서 `xla-backend`만으로는 false이기 때문이다. 흥미로운 쪽은 `seam_delegates_to_real_mlx_loader_on_missing_dir`이다. 이 테스트는 실패하지 **않는다**. `XlaBackend::load_model`이 `load_unsupported()`로 라우팅되어 마찬가지로 `Err`를 반환하기 때문이다(`src/backend/xla.rs:75`). 즉 경쟁이 나면 docstring이 "seam이 실제 MLX 로더에 도달함을 증명한다"고 주장하는 동안 XLA 스캐폴드를 조용히 검증하게 된다. 이 조용한 통과 사례가 가드를 붙일 가장 강한 근거다.

### 3.2 가드를 명명된 변수에 바인딩

**선택 이유:** `let _ = env_lock();`은 `MutexGuard`를 그 문장 끝에서 즉시 떨어뜨려 문제의 경쟁을 그대로 되살린다. 네 곳 모두 함수의 첫 바인딩으로 `let _env_guard = ...`를 쓰므로, 역순 드롭에 따라 가드가 보호 대상인 backend/session/capability 값들보다 마지막에 해제된다.

### 3.3 백엔드 전용 락을 새로 만들지 않고 기존 락 재사용

**선택 이유:** 경쟁이 서로 다른 두 모듈의 테스트 사이에서 일어나므로 락은 `src/test_support/env_lock.rs`의 크레이트 전역 락이어야 한다. 이 락은 이미 `unwrap_or_else(|p| p.into_inner())`로 poisoning에서 복구하므로, 패닉난 보유자가 새로 가드를 붙인 테스트들로 연쇄 실패를 일으킬 수 없다.

---

## 4. 구현 상세

### 4.1 주요 코드 변경

**파일: `src/backend/tests.rs`**
```rust
// Before
#[test]
fn mlx_session_threads_the_token_bias_through() {
    let mut bias = TokenBiasMap::new();
    bias.insert(5, -2.0);
    let session = select_backend()

// After
#[test]
fn mlx_session_threads_the_token_bias_through() {
    // Under the opt-in `xla-backend` feature, `select_backend()` reads
    // `MLXCEL_BACKEND`, and `muse_glimmer_startup_rejects_xla_backend_selection`
    // transiently sets that var to "xla" while holding this same lock. Without
    // taking it here too, a parallel full-suite run could hand this test an
    // XLA session and hit the `unreachable!` arm below.
    let _env_guard = crate::test_support::env_lock::env_lock();
    let mut bias = TokenBiasMap::new();
    bias.insert(5, -2.0);
    let session = select_backend()
```

**변경 이유:** 가드는 `select_backend()`보다 먼저 잡혀야 하고, 그 결과 backend나 session을 들여다보는 구간 전체보다 오래 살아야 한다. 네 테스트 각각에 락이 필요한 이유를 주석으로 달았다. 기본 기능 조합에서는 이 락이 불필요해 보여서, 주석이 없으면 나중 독자가 지울 만한 코드이기 때문이다.

---

## 7. 변경 요약

### 통계
| 항목 | 값 |
|------|-----|
| 변경 파일 | 1 |
| 추가 줄 | +26 |
| 삭제 줄 | -0 |
| 추가 테스트 | 0 (기존 테스트 4개를 결정적으로 만듦) |

### 카테고리별 변경

| 카테고리 | 개수 | 요약 |
|----------|------|------|
| 테스트 정확성 | 4 | 각 테스트에서 `select_backend()` 전에 env 락 획득 |

### 관련 커밋
| 해시 | 유형 | 메시지 |
|------|------|--------|
| `5de2bc60` | test | acquire the env lock in backend seam tests for xla-backend builds |

---

## 8. 후속 조치

### 완료 필요
- [ ] 없음. 수용 기준 세 가지 모두 충족

### 향후 개선 사항
- `seam_delegates_to_real_mlx_loader_on_missing_dir`은 어떤 `Err`든 받아들이고 메시지를 버리므로, 그 자체로는 MLX 로더의 오류와 다른 백엔드의 오류를 구분하지 못한다. 락은 이 테스트를 결정적으로 만들 뿐 어서션을 강하게 만들지는 않는다. `load_model` 호출 전에 `assert_eq!(backend.name(), "mlx")`를 두면 docstring의 주장을 강제할 수 있다. 선재 속성이라 이 PR 범위 밖으로 뒀다.
- `xla-backend`/`experimental-backend`를 빌드하는 CI 워크플로가 없어 이 갈래들은 로컬에서만 실행된다. 주기적인 옵트인 기능 빌드가 있으면 수동 실행 없이도 이 유형의 결함을 잡을 수 있다.

---

## 부록

### A. 테스트 결과

기본 기능 조합:

| 명령 | 결과 |
|------|------|
| `cargo fmt --check` | clean |
| `cargo check --lib --tests` | clean |
| `cargo test --lib backend::tests` | 5 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |

이슈의 두 번째 수용 기준은 `--features metal,accelerate,xla-backend`를 지목했다. `metal`과 `accelerate`는 Apple 대상이라 여기서 쓴 Linux/CUDA 장비에서는 빌드되지 않는다. 따라서 검증은 `--features xla-backend`로 했다. 이것이 `select_backend()`가 실제로 환경변수를 읽게 만드는 기능이자 경쟁이 성립하는 조건이다. `IREE_DIST` 없이 빌드되며, 그 요구는 `xla-iree`에만 해당한다.

| 명령 | 결과 |
|------|------|
| `cargo test --features xla-backend --lib backend::tests` | 7 passed, 0 failed (XLA 게이팅 테스트 2개 포함) |

이 조합은 `Session::Xla` 갈래를 컴파일하므로, `mlx_session_threads_the_token_bias_through`의 `unreachable!` 갈래가 컴파일에서 사라지지 않고 살아 있는 코드가 된다.

경합 실행. 두 모듈을 한 바이너리에서 8스레드로, 5회 반복:

```
for i in 1 2 3 4 5; do
  cargo test --features xla-backend --lib -- \
    backend::tests server::startup::muse_glimmer_startup_guard_tests --test-threads=8
done
```

5회 모두 `test result: ok. 12 passed; 0 failed`(백엔드 테스트 7개 + startup guard 테스트 5개). 이 실행은 변경 자체가 들여오는 주된 위험, 즉 네 테스트가 env를 변경하는 가드 테스트와 같은 크레이트 전역 뮤텍스를 두고 경합하게 되는 부분도 함께 덮으며, 교착이 없음을 보여준다.

범위에 관한 단서: 이는 기준을 문자 그대로, 즉 해당 기능을 켠 병렬 실행에서 테스트가 통과함을 검증한 것이다. 원래 실패를 재현한 것은 아니다. 이 경쟁은 타이밍 의존적이고 여기서 실패하는 모습이 관측된 적은 없으며, 이슈가 이를 안정적으로 재현되는 결함이 아니라 잠복 인터리빙으로 서술한 것과 일치한다.

### C. 참고 자료
- 이슈 #1159(명세), PR #1157(`env_lock` 패턴의 출처이자 이 이슈를 발행한 리뷰 지적), #1116(env 변경 도입)
- `src/backend/mod.rs:316`(`MLXCEL_BACKEND` 읽기), `src/test_support/env_lock.rs:60`(`env_lock`), `src/server/muse_glimmer_startup_guard_tests.rs:150`(경합하는 변경자)
- PR #1169의 리뷰, 보안, 검증 코멘트
