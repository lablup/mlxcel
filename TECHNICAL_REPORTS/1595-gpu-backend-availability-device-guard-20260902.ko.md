# 기술 보고서: PR #1595 - fix(core): GPU backend 존재 여부와 default device를 분리

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 보안 리뷰 사이클
**상태**: 완료 (open PR. 기존 CI와 Apple Silicon 검증 green, 최종 hardening은 NVIDIA GB10/CUDA에서 독립 검증. 부록 참조)
**언어**: Rust, C++
**위험도**: Medium (모든 generation·test 경로가 거쳐 가는 process-global MLX default device를 건드리지만, 변경 자체는 rename + deprecated shim, 새 query 함수, RAII guard로 구성된 additive 변경이고 수백 회의 parallel test run으로 검증됨)

---

## 요약

`mlxcel_core::is_gpu_available()`는 "MLX의 default device가 지금 GPU인가"에 답했지, "GPU backend가 존재하는가"에 답하지 않았다. 그래서 GPU가 있는 머신에서도 누군가 `set_default_device(false)`를 호출하는 순간 바로 `false`가 됐다. 이 혼동은 두 가지를 낳았다. 하나는 `initialize_runtime()`이 운영자의 `MLXCEL_DEVICE=cpu` override를 적용한 *뒤에* 이 플래그를 읽어, 하드웨어를 확인한 것처럼 요청을 그대로 되돌려주고 있었다는 점이고, 다른 하나는 실제 결함을 가리고 있었다는 점이다. 테스트 모듈 두 곳이 `std::sync::Once` 안에서 default device를 CPU로 옮기고 다시는 되돌리지 않았고, 그 결과 `--test-threads=1`에서 그 모듈들 뒤에 정렬되는 모든 real-checkpoint gate가 GPU 대신 CPU backend를 측정했다. PR #1420은 이미 공유 test guard에 무조건적인 GPU pin을 걸어 두 번째 증상만 덮어 놓은 상태였다. PR #1595는 bridge 계층에서 두 질문을 분리하고(device용 `default_device_is_gpu()`, hardware용 신규 `gpu_backend_available()`), 새는 `Once`들을 RAII guard로 교체했으며, 두 차례의 리뷰가 guard 하나만으로는 부족하다는 것을 밝혀낸 뒤에는 process-wide lock을 하나 추가해, "device를 복원하는 것"과 "그 device를 동시에 건드리는 다른 코드를 배제하는 것"이 더 이상 서로 다른 두 개의 보장이 아니게 만들었다.

---

## 1. 문제 정의

### 1.1 `is_gpu_available()`가 서로 다른 두 질문을 섞고 있었다

이 함수의 구현은 `mlx::core::default_device() == mlx::core::Device::gpu` 하나뿐이었다. 지금 dispatch가 어디로 가는지에 대해서는 맞는 진술이지만, 이것을 "GPU가 있는가"의 의미로 쓰던 트리 안의 모든 호출자는, default device가 하드웨어와 무관한 어떤 이유로든 CPU를 가리키고 있을 때마다 틀린 답을 받았다. 그중에는 이전 호출자가 의도적으로 CPU를 요청했을 뿐인 경우도 포함된다.

### 1.2 `initialize_runtime()`이 hardware 판정 대신 요청을 그대로 반영했다

`initialize_runtime()`은 `MLXCEL_DEVICE=cpu` override를 적용한 *뒤에* `is_gpu_available()`을 읽었는데, 그 시점에는 이미 default device가 CPU로 옮겨진 뒤였다. 그 결과 `RuntimeSetup.device`는 호스트에 대해 확정된 사실이 아니라, 운영자가 방금 요청한 것을 마치 런타임이 확인한 것처럼 되돌려주는 값이었다. "이 호스트에 실제로 GPU가 있었는데 쓰지 않기로 한 것인가"를 묻는 운영자나 하위 테스트는 이 구조체에서 답을 얻을 수 없었다.

### 1.3 `Once` 누수: PR #1420이 실제로 쫓던 대상

테스트 모듈 두 곳, `multimodal::host_preprocessor_tests`와 동일한 패턴의 `src/vision/merge_tests.rs`가 `std::sync::Once` 안에서 MLX의 process-wide default device를 CPU로 옮겼다. `Once`는 초기화 코드를 정확히 한 번만 실행하고 그 뒤로는 다시 실행하지 않는다는 성질을 정확히 노린 선택이었지만, device는 다시 옮겨지지 않았다. MLX의 default device는 명시적 stream 없이 dispatch되는 모든 op가 향하는 프로세스 전역 값 하나이고, libtest는 `--test-threads=1`에서 한 바이너리의 테스트 모듈들을 이름 순서로 실행한다. 그래서 새는 모듈 중 하나 뒤에 정렬되는 모든 real-checkpoint gate, `rerank::real_checkpoint_tests`의 reranker 점수들을 포함해, GPU 대신 CPU backend를 조용히 측정하고 있었다. 이것이 정확히 PR #1420이 쫓던 재현 사례다: `rerank::real_checkpoint_tests`는 4-bit Qwen3 reranker를 0.99가 아니라 0.35로 점수 매기고 bf16 Qwen3-VL reranker에서 NaN 이미지 점수를 냈지만, 같은 테스트를 단독으로 돌리면 깨끗이 통과했다.

### 1.4 PR #1420의 수정은 누수를 제거한 게 아니라 가렸다

PR #1420은 공유 `mlx_test_guard()` 테스트 헬퍼 맨 앞에 무조건적인 `set_default_device(true)` pin을 추가했다. 이것은 guard가 실행될 때마다 증상을 고쳐 주므로 reranker gate들은 더 이상 실패하지 않게 됐지만, 동시에 guard가 그 누수를 관찰하거나 보고할 수 없게 만들었다. default device를 옮기는 실제 회귀가 생겨도 이후 호출마다 조용히 재수리될 뿐, assertion도, 로그 한 줄도, 이후 기여자에게 뭔가 잘못됐다는 신호도 없었다. 원인이 되는 `Once`들은 그대로 남아 있었다.

### 1.5 수용 기준 (이슈 #1421)

- `gpu_backend_available()`는 GB10 CUDA와 Apple Silicon에서 true, CPU-only에서 false여야 하고, `set_default_device(false)` 뒤에도 변하지 않아야 한다.
- `default_device_is_gpu()`가 트리 안의 모든 `is_gpu_available()` 호출을 대체하고, deprecated shim은 경고와 함께 컴파일되어야 한다.
- `MLXCEL_DEVICE=cpu`에서 `initialize_runtime()`은 `device == Cpu`와 override 플래그를 보고해야 하고, `cargo run -- generate`는 여전히 CPU에서 돌아야 한다.
- `host_preprocessor_tests.rs`의 테스트(또는 그 뒤에 정렬되는 형제 테스트)가 모듈 실행 전에 기록해 둔 값과 `default_device_is_gpu()`가 같은지 확인해야 하고, pin을 줄인 상태에서 `--test-threads=1`로 통과해야 한다.
- `cargo test -p mlxcel --lib -- --test-threads=1 multimodal:: rerank::real_checkpoint_tests`가 pin을 줄인 상태로 통과해야 한다.
- `RuntimeSetup` 소비자와 startup 경고가 새 필드를 사용해야 한다.

### 1.6 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 향후 호출자가 하드웨어 쿼리처럼 읽히는 이름 아래 다시 availability/default-device 혼동을 들여옴 | High — 이 결함의 재발 | 완화됨. deprecated shim이 남은 모든 호출부에서 컴파일 경고를 강제하고, 두 함수의 doc comment가 서로를 교차 참조 |
| `DefaultDeviceGuard::gpu()`가 기존 무조건 pin처럼 CPU-only 빌드에서 throw | High(발생 시) | 제거됨. `gpu()`는 `set_default_device(Device::gpu)` 호출 전에 `gpu_backend_available()`을 확인하고, 아니면 no-op |
| RAII guard가 테스트별로는 정확히 복원하지만, parallel `cargo test --lib`에서 두 테스트 모듈이 여전히 하나의 process-global device 위에서 interleave됨 | High — 리뷰 중 실제로 발생 (2절 참조) | 세 번째 커밋의 process-wide lock으로 수정 |
| driver 없는 CUDA 호스트가 `Cpu`를 보고하면서도 MLX 자신은 존재한다고 믿는 GPU로 여전히 dispatch | High — 리뷰 중 실제로 발생 (2절 참조) | 운영자 override뿐 아니라 모든 CPU 확정을 적용하도록 수정 |

---

## 2. 기술적 검토 사항

최초 리뷰 사이클은 최초 구현(`7bd6e18b`)에 대해 두 차례 진행됐고, 두 차례 모두 결함을 찾아 각각 별도 커밋으로 고쳤다(첫 커밋에 합쳐 넣지 않았다). 머지 전 최종 구현 hardening에서 신뢰성 결함 두 건을 추가로 수정했다.

### 2.1 구현 리뷰 (`d1f2a38c`)

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| `initialize_runtime`이 `gpu_backend_available()`로부터 `device`를 계산했지만 MLX에는 운영자 override만 적용해, backend 답변(GPU backend 없음, 또는 driver 없는 CUDA 호스트)에서 나온 CPU 확정이 실제로 설치되지 않은 채 보고됨 | HIGH | 수정됨 |
| `vision::merge_tests`의 guard들이 어떤 lock으로도 직렬화되지 않아, guard가 막으려는 바로 그 누수 형태로 두 테스트가 interleave 가능 | HIGH | 수정됨 |
| `host_preprocessor_tests::cpu_device`가 `(MutexGuard, DefaultDeviceGuard)`를 반환. tuple 필드는 선언 순서로 drop되므로 device가 복원되기 전에 모듈 lock이 풀림 | HIGH | 수정됨 |
| `mlx_test_guard`의 assertion이 실제 누수와, 동시에 실행 중인 다른 테스트가 정당하게 들고 있는 guard를 구별하지 못함 | HIGH | `default_device_guards_held()`로 수정됨 |
| `DefaultDeviceGuard::drop`이 `gpu()`류의 availability 체크 없이 `set_default_device(previous_is_gpu)`로 복원 | LOW | 보고만 됨 — 도달 불가능하지만 주석을 남길 가치는 있음(이번 마무리 작업에서 반영, 4.4절 참조) |
| `RuntimeSetup`이 `#[non_exhaustive]` 없는 `pub` 구조체라, `cpu_override` 추가는 이를 직접 리터럴로 생성하는 외부 코드에는 breaking change | LOW | 보고만 됨 — 크레이트의 기존 스타일이며 이 PR에서 바꾸지 않음 |
| `the_default_device_is_restored_after_the_export_tests`는 단독 실행 시 동어반복(그 순간의 현재 값을 `OnceLock`이 그대로 기록) | LOW | 보고만 됨 — cross-module assertion 부분은 여전히 실질적인 검증 역할을 함 |

이 커밋의 측정 효과: 대표적인 parallel `cargo test --lib` filter가 수정 전 40회 중 31회 실패, 수정 후 40회 중 0회 실패. 805개 테스트짜리 더 넓은 parallel filter도 12회 연속 clean.

### 2.2 보안·성능 리뷰 (`3741c8fd`)

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| 모듈별 lock은 process-global device를 직렬화하지 못함: `multimodal::host_preprocessor`, `vision::merge`, `mlxcel_core::streams::tests`가 각자 자기 mutex를 가져, 여전히 하나의 MLX default device 위에서 서로 interleave | HIGH | `lock_default_device()`로 수정됨 |
| 결과 1: `the_default_device_is_restored_after_the_export_tests`가 동시에 실행 중인 `vision::merge` guard가 정당하게 들고 있던 CPU default를 읽고 이를 누수로 오보 (2모듈 parallel filter 300회 중 4회 실패) | HIGH | 수정됨(수정 후 300회 중 0회) |
| 결과 2: `mlx_test_guard`를 들고 있는 어떤 gate든 — 이슈 #1421이 보호하려는 바로 그 `rerank::real_checkpoint_tests`를 포함해 — 동시에 다른 guard가 device를 CPU에 쥐고 있는 동안 real checkpoint를 CPU backend에서 측정할 수 있었고, 하필 그 순간 `default_device_guards_held()` 예외가 leak assertion을 꺼 버림 | HIGH | 수정됨 |
| `gpu_backend_available()`가 어떤 backend에서든 throw하는지 | — | 점검 결과 결함 없음: `no_gpu`는 0 반환, Metal은 try/catch로 감쌈, CUDA는 noexcept `cudaGetDeviceCount` 호출 |
| `DefaultDeviceGuard::drop`이 throw하는 `set_default_device(true)`를 만나는지 | — | 점검 결과 결함 없음 |
| `LIVE_DEVICE_GUARDS`의 atomic ordering | — | 점검 결과 안전함(쓰기 측 AcqRel, 읽기 측 Acquire) |
| `gpu_backend_available()`의 startup 비용 | — | 점검 결과 무시할 만함 — "cheap" 표현에 대한 후속 작업은 4.4절 참조 |
| startup 라인 변경으로 생긴 새 로그 라인 | — | 점검 결과 boolean 하나와 고정 문자열만 출력, 새 trust boundary 없음 |
| MLX는 `default_device_`를 평범한 non-atomic 전역 변수로 보관하는데, 이 PR 이후 유일하게 lock을 안 쥐는 mover는 `initialize_runtime()`의 startup 전용 `set_default_device(false)` | MEDIUM | Hardening 완료 — 실제 startup 전환은 의도적으로 lock 없이 유지하되, CPU-only 및 반복 테스트의 중복 write를 생략하도록 수정했고 미래 mover를 위한 제약을 문서화함(4.4절 참조) |
| `models::bert_tests::mlx_test_guard` / `models::modernbert_tests::mlx_guard`는 공유 헬퍼로의 순수 위임 | MEDIUM | 보고만 됨 — 조치 불필요, 정보성 |
| `mlx_cxx_bridge.cpp`의 `gpu_backend_available` 주석이 "runtime init 전에도 안전하다"고 말하는 건 사실이지만 "저렴하다"는 뉘앙스로 읽힘. Metal에서는 첫 호출이 Metal device singleton을 생성함(metallib load) | LOW | 보고만 됨(이번 마무리 작업에서 반영, 4.4절 참조) |
| `tests/sampling_*_kill_switch.rs`가 default-device 체크에서 backend 체크로 바뀌었는데, 해당 바이너리들은 아무것도 default device를 옮기지 않아 이 구분이 현재로선 도달 불가능 | LOW | 보고만 됨, 정보성 |

측정 효과: 2모듈 filter 300회 parallel run(수정 후 0회 실패), mlxcel-core `streams::` filter 200회(0회), `models::`/`rerank::`/`embeddings::`/`vision::`/`multimodal::`에 걸친 2,432개 테스트 filter 60회(0회 실패, hang 없음).

### 2.3 최종 구현 hardening

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| `initialize_runtime()`이 MLX default가 이미 CPU인데도 CPU 확정 때마다 `Device::cpu`를 다시 썼다. 이 함수는 테스트에서 반복 호출되고 pinned MLX는 default device를 평범한 non-atomic 전역에 저장하므로, CPU-only 실행에서 불필요한 process-global write가 발생했다 | MEDIUM | 현재 default가 아직 GPU일 때만 CPU 확정을 적용하도록 수정. driver 없는 CUDA의 GPU→CPU 전환은 그대로 보장됨 |
| CUDA multi-GPU 테스트 `trivial_op_runs_on_non_default_gpu_index`가 평가 뒤 GPU 0을 수동 복원해, 그 줄 전에 panic이 나면 공유 lock을 쥐고도 GPU 1이 후속 테스트에 누수됐다 | MEDIUM | GPU 1 선택 전에 `DefaultDeviceGuard`로 이전 device를 capture하도록 수정. 이제 unwinding 중에도 lock 해제 전에 복원됨 |

---

## 3. 기술적 선택과 그 이유

### 3.1 같은 이름 아래 의미를 바꾸는 대신, rename + 새 함수

**컨텍스트.** `is_gpu_available()`의 호출자는 availability/default-device 구분의 양쪽에 걸쳐 있었다. `streams.rs`는 stream 선택 헬퍼를 위해 "지금 default device가 GPU인가"를 정말로 원했고, `initialize_runtime()`과 sampling kill-switch 테스트, sampling microbenchmark는 "애초에 GPU backend가 존재하는가"를 원했다.

**고려한 대안.**

| 옵션 | 장점 | 단점 |
|---|---|---|
| `is_gpu_available()`의 의미를 제자리에서 hardware availability로 바꿈 | 새 함수 이름을 배울 필요 없음 | 원래의, 나름대로 옳았던 default-device 의미에 의존하던 `streams.rs` 같은 호출자를 컴파일러 신호 없이 조용히 깨뜨림 |
| **선택: `default_device_is_gpu()`로 rename, `gpu_backend_available()` 신규 추가, `is_gpu_available()`은 deprecated shim으로 유지** | 모든 호출자가 계속 컴파일됨. deprecation 경고가 각 지점에서 두 새 이름 중 어느 쪽이 맞는지 의식적으로 고르게 강제함. shim의 본문 자체가 옛 default-device 의미를 자신의 doc comment에서 명시 | 함수 이름 하나가 아니라 두 개를 유지해야 하고, shim을 지우기까지 한 릴리스의 deprecation 기간이 필요 |

**선택 이유.** 두 의미 모두 트리 어딘가에서 정당하게 필요하다. 결함은 둘 중 하나가 틀렸다는 게 아니라, 이름 하나가 둘 다를 대신하고 있었다는 것이었다. 조용한 의미 변경은 같은 커밋에서 `initialize_runtime()`을 고치면서 `streams.rs`를 깨뜨렸을 것이고, diff의 어디에도 그 사실이 드러나지 않았을 것이다. deprecated shim은 남아 있는 모든 모호한 호출부를 사람이 직접 판단해야 하는 컴파일 타임 경고로 바꾸는데, 이는 주석이나 changelog 항목보다 강한 보장이다.

**트레이드오프.** shim은 제거되기 전까지 한 릴리스 동안 트리에 남는다(후속 조치로 추적, 8절).

### 3.2 `cfg!(feature = "cuda")`가 아니라 unclamped `device_count(Device::gpu)`

**컨텍스트.** `gpu_backend_available()`는 서로 다른 두 종류의 호스트에서 `false`를 답해야 한다. GPU 코드가 아예 컴파일되지 않은 CPU-only 빌드, 그리고 GPU 코드는 컴파일됐지만 구동할 driver가 없는 CUDA 빌드다. 빌드 타임 `cfg!(feature = "cuda")` 체크는 첫 번째 경우만 볼 수 있다.

**선택 이유.** `mlx::core::device_count(Device::gpu)`는 이미 모든 MLX backend에 걸쳐 `#ifdef` 없이 portable하다: pinned MLX의 `no_gpu::device_count()`는 0을 반환하고, Metal은 1을 반환하며, CUDA backend는 `cudaGetDeviceCount`가 보고하는 값을 반환한다 — 실패하면 throw 없이 0. 기존의 더 오래된 bridge 함수 `gpu_device_count()`가 정확히 같은 이유로 이미 같은 하위 primitive를 무조건 호출하고 있다. `gpu_backend_available()`는 이를 재사용하되, `gpu_device_count()`가 하는 것처럼 결과를 `>= 1`로 clamp하지 않는다. 그 clamping은 `Device::gpu`가 항상 선택 가능한 인덱스이도록 하기 위해 존재하는 것이고, availability query에는 정확히 잘못된 동작이기 때문이다. `cfg!(feature = "cuda")` 체크는 driver 없는 GB10 호스트에 앉은 CUDA 빌드에서도 `true`를 보고했을 것이고, 이는 `initialize_runtime()`의 CPU-resolution 버그(2.1절, 3.4절)가 이 함수에 그런 동작이 없어야 성립하는 바로 그 실패 모드다.

### 3.3 test-only leak 체크에서 `debug_assert!`가 아니라 `assert!`

**컨텍스트.** 이슈 본문은 `mlx_test_guard`의 leak assertion에 `debug_assert!`를 제안했다. test-only 코드이고 `debug_assert!`는 release 바이너리에서 비용이 없다는 근거였다.

**선택 이유.** 이 저장소가 돌리는 모든 gate — `make verify-test`, `make verify-test-cuda`, `make test-fast`, 그리고 이번 마무리 작업 자체가 쓰는 `--profile test-fast` — 는 `release`를 상속하는 `--profile test-fast`로 빌드되고, 따라서 `debug_assert!`는 완전히 컴파일에서 빠진다. 여기서 `debug_assert!`를 썼다면 이 체크가 실행되어야 할 모든 곳에서 `cfg`로 제거되고, 아무도 만들지 않는 debug 빌드에서만 동작했을 것이다. `mlx_test_guard` 자체가 `#[cfg(test)]` 코드이므로, 어차피 평범한 `assert!`도 production 바이너리에서는 비용이 없다 — 트리의 다른 곳에서 `debug_assert!`를 쓰는 근거인 debug/release 구분 자체가, 애초에 모든 non-test 빌드에서 컴파일이 빠지는 코드에는 적용되지 않는다.

### 3.4 운영자 override뿐 아니라 모든 CPU 확정을 적용

**컨텍스트.** `initialize_runtime()`은 `resolve_device()`라는 순수 함수를 통해 `gpu_backend_available()`과 운영자의 `MLXCEL_DEVICE` 요청으로부터 `RuntimeSetup.device`를 계산한 뒤, 그 결정 중 실제로 MLX의 default device에 적용할 것을 골라야 한다. 최초 구현(`7bd6e18b`)은 운영자의 명시적 `MLXCEL_DEVICE=cpu` override만 적용했다.

**CUDA `gpu::is_available()`의 특이점.** MLX 자신의 default device는 `mlx::core::gpu::is_available()`로부터 시작되는데, CUDA backend는 이 호출에 무조건 `true`로 답한다 — `device_count(Device::gpu)`처럼 driver에게 실제로 물어보지 않는다. 그래서 driver가 제대로 동작하지 않는 호스트의 CUDA 빌드는, `gpu_backend_available()`(실제로 `cudaGetDeviceCount`를 확인하는)이 정확히 사용 불가로 판정하는 GPU를 이미 default로 가리킨 채로 시작한다. `initialize_runtime()`이 운영자의 명시적 override만 적용했다면, 이 호스트는 (`resolve_device()`가 `gpu_backend_available() == false`를 보므로) `RuntimeSetup.device == Cpu`로 확정되면서도 MLX 자신은 이미 default로 삼은 GPU에 커버되지 않은 모든 op를 계속 dispatch했을 것이다 — 구조체가 런타임이 실제로 실행되는 곳에 대해 거짓인 사실을 보고하는 셈이다.

**수정 내용과 GPU 방향에 대응이 필요 없는 이유.** `initialize_runtime()`은 이제 현재 default가 아직 GPU인 모든 CPU 확정에 `set_default_device(false)`를 적용한다. 그 확정이 운영자 override에서 왔든 backend가 "여기 GPU 없음"이라고 답한 데서 왔든 상관없다. 현재-device 체크는 driver 없는 CUDA에서 필요한 전환을 유지하면서 CPU-only 빌드와 앞선 CPU 초기화 뒤의 중복 write를 피한다. GPU 방향은 의도적으로 비대칭이다: `device == Gpu`는 `gpu_backend_available()`가 true일 때만 도달 가능하고, 이는 `device_count(Device::gpu) > 0`을 요구하며, 어떤 backend에서든 이것은 `gpu::is_available()`도 true임을 함의한다 — 즉 그 분기에서는 MLX의 초기 default가 GPU이므로 적용할 것이 없다. GPU 방향을 pin하지 않는 데는 두 번째 이점도 있다: `mlx_test_guard`의 leak assertion이 닿을 수 없는 곳에 이 호출을 둔다는 점이다. 그 assertion은 GPU가 이미 default라고 기대할 뿐, 모든 런타임 초기화마다 거기에 pin되는 무언가를 기대하지 않는다.

### 3.5 RAII guard, process-wide lock 하나, live-guard backstop — 하나가 아니라 세 겹

**컨텍스트.** 새는 `Once`에 대한 뻔한 수정은 `Drop`이 device를 복원하는 RAII guard다. 이것은 필요하지만, 리뷰는 두 번에 걸쳐 이것만으로는 충분하지 않다는 것을 밝혀냈다.

**guard 하나만으로는 왜 부족한가.** `DefaultDeviceGuard`는 자신이 옮긴 device를 복원하지만, 복원한다는 것과 배제한다는 것은 다르다. 다른 스레드가 기존 guard가 span을 쥐고 있는 도중에 같은 process-global device를 옮기는 것을 막아 주지 않는다. `scripts/run_quality_gate.sh`가 돌리고 `docs/adding-models.md`가 기여자에게 권하는 `--test-threads=1` 없는 `cargo test --lib`에서는, 서로 *다른* guard를 들고 있는 두 테스트 모듈이 여전히 하나의 MLX default device 위에서 자유롭게 interleave될 수 있다.

**모듈별 lock이 왜 부족했는가(그리고 이것은 처음부터 설계된 게 아니라 리뷰에서 잡아낸 것).** 동시 mover를 배제하려는 첫 시도는 `multimodal::host_preprocessor_tests`, `vision::merge_tests`, `mlxcel_core::streams::tests` 각각에 자기만의 private mutex를 줬다. private lock은 그 모듈 자신의 테스트끼리만 서로 배제할 뿐, `vision::merge`가 자기 private lock을 쥔 채 span 중간에 있는 동안 `multimodal::host_preprocessor`가 device를 읽는 것을 막지 못한다. 보안 리뷰의 parallel run이 정확히 이것을 잡아냈다: `the_default_device_is_restored_after_the_export_tests`가 동시에 실행 중인 `vision::merge` guard가 정당하게 쥐고 있던 CPU default를 읽고 이를 leak으로 오보한 것이다 — 실제 회귀가 아니라 multi-lock 설계가 만든 false positive였다.

**선택한 설계.** `mlxcel_core::streams::lock_default_device()`는 하나의 process-wide device를 위한 단 하나의 process-wide lock이다. default device를 옮기는 모든 코드 — 새는 `Once`에서 guard로 바뀐 두 테스트 모듈, `mlxcel-core::streams` 테스트, 그리고 자신이 직렬화하는 테스트들의 전체 span 동안의 `mlx_test_guard` — 가 device를 건드리기 전에 이 lock 하나를 쥔다. 그래서 4.4절에서 의도적으로 예외 처리한 하나의 startup 호출을 제외하면, 트리 어디에서도 측정과 동시적인 이동이 더 이상 겹칠 수 없다. `DefaultDeviceGuard` 자신은 의도적으로 이 lock을 **쥐지 않는다**. 그래야 guard들이 이미 lock을 쥐고 있는 span 안에서 계속 자유롭게 중첩될 수 있다 — `mlx_test_guard`는 lock을 쥔 뒤 그 span 안에서 자기 자신과 데드락 없이 guard를 만든다. lock 순서는 고정되고 문서화되어 있다: `lock_default_device()`는 (존재한다면) 각 모듈의 기존 직렬화 mutex 뒤에만 취해지고 반대 순서로는 취해지지 않으며, device lock을 쥐는 코드가 반대 방향에서 그 직렬화 mutex를 쥐는 일도 없으므로 이 쌍은 데드락될 수 없다.

**live-guard backstop은 남는다.** `default_device_guards_held()`(두 번째 커밋에서 추가, 세 번째 커밋에서 유지)는 lock 없이 만들어진 guard에 대한 예외로 남는다. 이제 lock이 트리 안의 모든 호출자를 커버하므로 실질적으로 backstop은 주 메커니즘이라기보다 심층 방어에 가깝지만, 실제 gate가 전부 쓰는 `--test-threads=1`에서는 무관한 테스트가 실행되는 동안 guard가 살아 있을 수 없으므로 비용이 없고, 진짜 leak은 여전히 크게 실패한다.

---

## 4. 구현 상세

### 4.1 bridge: rename 하나, 신규 함수 하나

`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` / `.h`:

```cpp
// 변경 전 (mlx_cxx_bridge.cpp)
// Check whether the current default device is GPU
bool is_gpu_available() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// 변경 후
bool default_device_is_gpu() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// 신규
bool gpu_backend_available() {
    return mlx::core::device_count(mlx::core::Device::gpu) > 0;
}
```

`default_device_is_gpu`의 본문은 옛 `is_gpu_available`과 바이트 단위로 동일하다 — bridge 계층에서는 순수한 rename이고, 의미 분리는 첫 함수의 동작을 바꾸는 대신 두 번째 함수를 추가하는 것만으로 표현된다. `gpu_backend_available`은 전처리기 `#ifdef` 없이 모든 backend에 걸쳐 portable한데, `device_count`가 이미 MLX 내부에서 backend별로 dispatch되기 때문이다.

### 4.2 Rust shim과 deprecation

`src/lib/mlxcel-core/src/lib.rs`:

```rust
#[deprecated(note = "use default_device_is_gpu or gpu_backend_available")]
pub fn is_gpu_available() -> bool {
    ffi::default_device_is_gpu()
}
```

`is_gpu_available`의 옛 default-device 의미로 남아 있던 모든 호출은 이제 `default_device_is_gpu()`를 직접 거친다. shim은 순수하게 외부 소비자의 빌드가 한 릴리스 동안 경고와 함께 계속 컴파일되도록 하기 위해 존재한다.

### 4.3 `initialize_runtime`: 확정한 다음 적용, 그 순서로

`src/execution/runtime.rs`:

```rust
// Availability is a backend fact and the override is an operator request;
// resolve them separately so the setup can say which one put the runtime
// on the CPU (issue #1421).
let (device, cpu_override) =
    resolve_device(requested_device, mlxcel_core::gpu_backend_available());
// Apply every CPU transition, not just the operator's override. ...
if !device.uses_gpu() && mlxcel_core::default_device_is_gpu() {
    mlxcel_core::set_default_device(false);
}
```

`resolve_device`(3.4절)는 요청된 device와 backend 답변을 받아 `(device, cpu_override)`를 반환하는 순수 함수다. 네 가지 입력 조합 모두 `runtime_tests.rs`의 `resolve_device_separates_backend_availability_from_the_cpu_override`가 환경 변수도 실제 device도 없이 커버한다. 새 `RuntimeSetup.cpu_override: bool` 필드 덕분에 서버 startup 로그(`src/server/startup.rs`)와 CLI runtime 출력(`src/commands/generate.rs`, `src/commands/chat.rs`)이 device가 왜 `Cpu`인지 — 요청됐기 때문인지, 유일한 선택지였기 때문인지 — 두 경우 똑같이 읽히는 맨 `CPU` 대신 보고할 수 있다.

### 4.4 `DefaultDeviceGuard`, `lock_default_device`, live-guard 카운터

`src/lib/mlxcel-core/src/streams.rs`:

```rust
impl DefaultDeviceGuard {
    pub fn capture() -> Self { /* 옮기지 않고 기록만 */ }
    pub fn cpu() -> Self { /* 기록 후 CPU로 이동 */ }
    pub fn gpu() -> Self {
        let guard = Self::capture();
        if ffi::gpu_backend_available() {
            ffi::set_default_device(true);
        }
        guard
    }
}

impl Drop for DefaultDeviceGuard {
    fn drop(&mut self) {
        ffi::set_default_device(self.previous_is_gpu);
        LIVE_DEVICE_GUARDS.fetch_sub(1, Ordering::AcqRel);
    }
}
```

`gpu()`는 이동하기 전에 `gpu_backend_available()`을 확인하는데, pinned MLX는 GPU가 없는 backend에서 `set_default_device(Device::gpu)`를 throw하기 때문이다. PR #1420의 무조건 pin이 만약 CPU-only 테스트 바이너리에서 실행됐다면 정확히 이 방식으로 프로세스를 종료시켰을 것이다. 반면 `Drop`은 그런 체크가 필요 없다. `previous_is_gpu`가 `true`일 수 있는 유일한 경우는 그것이 기록될 당시 GPU backend가 존재했을 때뿐이고, 이는 `set_default_device(Device::gpu)`가 throw하지 않는 것과 같은 조건이므로, 복원 경로는 throw하는 경우에 도달할 수 없다. 이번 마무리 작업은 이 근거를 `Drop` impl에 명시적 주석으로 추가한다(리뷰 발견 사항, 2.1절, LOW로 보고되고 리뷰 시점에는 주석 없이 남아 있었음).

`lock_default_device()`(세 번째 커밋에서 추가)는 3.5절에서 설명한 단일 process-wide 직렬화 지점이다.

```rust
pub fn lock_default_device() -> DefaultDeviceLock {
    DefaultDeviceLock {
        _guard: DEFAULT_DEVICE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    }
}
```

poison된 lock은 전파하지 않고 복구한다. 그래야 panic 하나가 그 뒤에 실행되는 모든 테스트로 lock-poisoning 실패를 연쇄시키지 않고 혼자 실패한다.

### 4.5 새는 `Once`들을 교체

`src/multimodal/host_preprocessor_tests.rs`(동일한 패턴이 `src/vision/merge_tests.rs`에도 적용됨):

```rust
// 변경 전
static CPU_DEVICE: Once = Once::new();
fn ensure_cpu_device() {
    CPU_DEVICE.call_once(|| {
        mlxcel_core::set_default_device(false);
    });
}

// 변경 후
fn cpu_device() -> (DefaultDeviceGuard, DefaultDeviceLock) {
    let lock = lock_default_device();
    DEFAULT_DEVICE_BEFORE.get_or_init(mlxcel_core::default_device_is_gpu);
    let device = DefaultDeviceGuard::cpu();
    (device, lock)
}
```

tuple 순서가 결과를 좌우하는데, 첫 시도에서는 이 순서가 틀려 있었다. tuple 필드는 선언 순서로 drop되므로, `(DefaultDeviceLock, DefaultDeviceGuard)`였다면 device guard가 복원되기 전에 lock이 풀려, 기다리던 형제 테스트가 lock을 가져가면서 아직 옮겨진 채인 CPU device를 자신의 baseline으로 기록했을 것이다. 수정된 순서 `(DefaultDeviceGuard, DefaultDeviceLock)`는 lock을 풀기 전에 device를 복원한다. 새 테스트 `the_default_device_is_restored_after_the_export_tests`는 모듈 안에서 device를 옮기는 모든 테스트 뒤에 정렬되고, `DEFAULT_DEVICE_BEFORE`가 기록한 값과 `default_device_is_gpu()`가 같은지 확인한다 — `mlx_test_guard`가 확인하는 것과 같은 불변식을, 이 모듈 안에서 로컬로도 확인하는 것이다.

### 4.6 `mlx_test_guard`: 무조건 pin에서 실질적인 assertion으로

`src/models/embedding_test_support.rs`:

```rust
// 변경 전 (PR #1420)
mlxcel_core::set_default_device(true);

// 변경 후
let device = mlxcel_core::streams::lock_default_device();
assert!(
    mlxcel_core::default_device_is_gpu()
        || !mlxcel_core::gpu_backend_available()
        || crate::execution::runtime::cpu_override_requested()
        || mlxcel_core::streams::default_device_guards_held() > 0,
    "MLX's default device is the CPU although a GPU backend is available, \
     MLXCEL_DEVICE=cpu is not set, and no DefaultDeviceGuard is held: an \
     earlier test moved the default device and never restored it. ..."
);
```

이 assertion은 지금 default device가 CPU여도 되는 네 가지 정당한 이유(GPU backend 없음, 명시적 운영자 override, 동시에 쥐어진 guard)와 실제 실패 모드(누수)를 나열한 논리합이다. 자신이 직렬화하는 테스트들 전체 span 동안 `lock_default_device()`를 쥐고 있어서, 곧 측정할 checkpoint와 동시에 interleave될 수 있는 mover가 없다.

### 4.7 이번 마무리 작업에서 추가한 주석 수정

리뷰 발견 사항 세 개는 행동을 바꾸지 않으므로 보고만 된 채 남아 있었다. 이번 마무리 작업은 문서로 이를 닫는다.

- `DefaultDeviceGuard::drop`(위 4.4절) — 복원에 `gpu_backend_available()` 체크가 필요 없는 이유.
- `gpu_backend_available()`의 bridge 주석(`mlx_cxx_bridge.cpp`) — 기존 문구는 이 호출이 "runtime 초기화가 끝나기 전에도 안전하다"고 했는데, 이는 사실이지만 저렴하다는 뉘앙스로 읽힌다. Metal에서는 첫 호출이 `metal::Device` singleton을 생성한다(metallib load). 주석은 이제 이를 명시하면서, 어차피 모든 array 할당이 그 singleton을 생성하게 되므로 이 query는 비용을 새로 더하는 게 아니라 그저 더 일찍 당기는 것뿐이라는 점, 그리고 이후 호출들(Metal에서, 그리고 CUDA backend의 캐시된 `cudaGetDeviceCount`)은 magic-static 체크에 불과하다는 점을 함께 밝힌다.
- `initialize_runtime`의 `set_default_device(false)` 호출(4.3절) — 보안 리뷰는(MEDIUM) 세 번째 커밋 이후 이것이 lock을 쥐지 않는 유일한 in-tree default-device mover라고 지적했다. Production에서는 worker가 시작되기 전에 실행되고, 최종 hardening은 이를 조건부로 만들어 CPU-only 테스트의 반복 초기화가 MLX의 평범한 전역 변수를 다시 쓰지 않게 했다. 주석은 두 제약을 모두 기록하고 앞으로 새 mover가 따라야 할 규칙도 명시한다: `lock_default_device()`를 쥐어라.

---

## 5. 변경 요약

### 통계 (`git diff --stat origin/main...HEAD`, 세 커밋)

| 항목 | 값 |
|---|---|
| 변경 파일 | 21 |
| 추가 라인 | 686 |
| 삭제 라인 | 78 |
| 커밋 수 | 3 |

### 변경 규모별 파일

| 파일 | +/- |
|---|---|
| `src/lib/mlxcel-core/src/streams.rs` | +306 |
| `src/execution/runtime_tests.rs` | +61/- |
| `src/execution/runtime.rs` | +69/- |
| `src/models/embedding_test_support.rs` | +89/- |
| `src/multimodal/host_preprocessor_tests.rs` | +80/- |
| `src/vision/merge_tests.rs` | +37/- |
| `src/lib/mlxcel-core/src/lib.rs` | +26/- |
| `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` | +17/- |
| `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.h` | +14/- |
| `src/lib/mlxcel-core/src/ffi_tests.rs` | +8/- |
| `src/server/startup.rs` | +8 |
| `src/commands/generate.rs` | +8 |
| `src/commands/chat.rs` | +6 |
| `examples/gumbel_sampling_microbench.rs` | +6/- |
| `src/models/bert_tests.rs`, `src/models/modernbert_tests.rs` | 각 +9/- |
| `examples/rejection_sampling_microbench.rs` | +4/- |
| `tests/sampling_gumbel_kill_switch.rs`, `tests/sampling_rejection_kill_switch.rs` | 각 +2/-1 |
| `CHANGELOG.md`, `docs/environment-variables.md` | +1, +1/-1 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `7bd6e18b` | fix(core) | separate GPU backend availability from the default device |
| `d1f2a38c` | fix(core) | apply the resolved device and serialize the device guards |
| `3741c8fd` | fix(core) | serialize every default-device mover on one process lock |

### 카테고리별 변경

| 카테고리 | 요약 |
|---|---|
| Bridge (C++) | `is_gpu_available` → `default_device_is_gpu` rename; `gpu_backend_available` 신규 추가 |
| Core (Rust) | `mlxcel-core`의 deprecated shim; `streams.rs`의 `DefaultDeviceGuard`, `lock_default_device`, `default_device_guards_held` |
| Runtime | `initialize_runtime`이 backend availability로부터 device를 확정하고, 모든 CPU 확정을 적용하며, `RuntimeSetup.cpu_override` 추가 |
| Tests | 새는 `Once` 두 곳을 RAII guard + lock으로 교체; `mlx_test_guard`의 무조건 pin을 assertion으로 축소; 새 restore-invariant 테스트 |
| CLI / server | startup·runtime 출력이 backend 답변 옆에 override를 함께 보고 |
| Docs | `CHANGELOG.md`, `docs/environment-variables.md`가 rename, 새 필드, startup 라인을 서술 |

---

## 6. 후속 조치

### 필요(추적 중, 이 PR을 막는 것은 아님)

- 외부 호출자가 `default_device_is_gpu()` 또는 `gpu_backend_available()`로 옮겨 갈 시간을 준 뒤, 한 릴리스 후 deprecated `is_gpu_available()` shim을 제거한다.

### 리뷰에서 보고됐고 의도적으로 열어 둔 것

- `RuntimeSetup`은 `#[non_exhaustive]` 없는 `pub` 구조체다. `cpu_override` 추가는 이를 직접 리터럴로 생성하는 외부 코드에는 이미 breaking change다. 이는 크레이트의 기존 스타일이고 이 PR에서 바꾸지 않았다. 향후 별도의, 의도적으로 범위를 좁힌 변경으로 `RuntimeSetup`(과 크레이트가 export하는 다른 public struct들)에 `#[non_exhaustive]`를 추가하는 검토가 있을 수 있다.
- `the_default_device_is_restored_after_the_export_tests`는 단독 실행 시 동어반복이다(그 순간의 현재 값을 `OnceLock`이 그대로 기록). 이 테스트의 실제 가치는 cross-module 케이스에서 나오는데, 이는 2.2절의 parallel-run 검증이 확인하고 있다.
- `models::bert_tests::mlx_test_guard` / `models::modernbert_tests::mlx_guard`는 공유 `embedding_test_support::mlx_test_guard`로의 순수 위임이다. 조치는 필요 없고, 호출부를 감사하는 사람을 위해 기록해 둔다.
- `tests/sampling_*_kill_switch.rs`는 default-device 체크에서 backend-availability 체크로 바뀌었다. 해당 바이너리들은 현재 아무것도 default device를 옮기지 않으므로 이 구분은 맞지만 지금은 도달 불가능하다.

### 남아 있는 환경별 검증 공백

- 최종 hardening 단계에서는 기존 리뷰의 Apple Silicon suite를 다시 실행하지 않았다. 해당 결과는 부록 A에 그대로 기록되어 있다.
- 동작하는 driver가 없는 CUDA 빌드 환경은 확보하지 못했다. 이 환경의 `gpu_backend_available() == false` 동작과 startup GPU→CPU 전환은 pinned MLX 소스 분석 및 별도의 순수 runtime resolution 테스트를 근거로 한다.

---

## 부록

### A. 테스트 결과 (이 호스트, Apple Silicon, `--profile test-fast --features metal,accelerate`)

| 체크 | 결과 |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --profile test-fast --features metal,accelerate --lib --tests -- -D warnings` | clean |
| `cargo test -p mlxcel-core --lib streams::` | 17 passed |
| `cargo test --lib execution::runtime`(`MLXCEL_DEVICE=cpu` 있음/없음 각각) | 각 19 passed |
| `cargo test --lib multimodal::host_preprocessor` | 24 passed, 1 ignored |
| `cargo test --lib vision::merge` | 4 passed |
| `cargo test --lib models::gemma3_embedding` | 11 passed |
| `cargo test --lib models::bert` | 28 passed |
| `cargo test --lib models::modernbert` | 19 passed |
| `cargo test --lib -- --test-threads=1 multimodal:: rerank::real_checkpoint_tests`(PR #1420 재현) | 262 passed, 26 ignored(reranker gate는 체크포인트 부재로 soft-skip) |
| `cargo test --test sampling_gumbel_kill_switch` / `sampling_rejection_kill_switch` | 각 1 passed |
| parallel-run 검증(리뷰) | 2모듈 filter 300회, `streams::` 200회, `models::`/`rerank::`/`embeddings::`/`vision::`/`multimodal::`에 걸친 2,432개 테스트 filter 60회 — 전부 0 실패, hang 없음 |
| CI(GitHub) | `7bd6e18b`, `d1f2a38c`에서 green: cargo-fmt, cargo-clippy, cargo-deny, OpenXLA feature compile, crate versions, cross-repo refs, kernel dtype keys, llama-compat manifest, CLA. CUDA sm_70 compile과 MLX pin extraction은 이 저장소 CI의 해당 체크에서 path-skip |

### B. `MLXCEL_DEVICE=cpu` generate smoke test

```
target/test-fast/mlxcel generate -m mlx-community/Phi-3.5-mini-instruct-4bit -p "Say hello." -n 3
```

override 없을 때: `Runtime device: Apple GPU (Metal)`, 13.27 tok/s.

`MLXCEL_DEVICE=cpu`일 때: `Runtime device: CPU`, 그리고 `Running on the CPU because MLXCEL_DEVICE=cpu asked for it (GPU backend available: true).`, 0.44 tok/s.

두 줄 다 새 `RuntimeSetup.cpu_override` 필드가 만든다. 두 번째 줄은 override가 `true`*이면서 동시에* 실제로 GPU backend가 사용 가능했을 때만 나올 수 있는데, 이것이 정확히 1.2절과 3.4절이 보고 가능하게 만들려 했던 구분이다.

### C. 기존 Apple Silicon 리뷰에서 실행하지 않은 것

기존 리뷰 호스트에서는 `--features cuda`를 컴파일하거나 실행할 수 없었다. 당시 로컬 체크포인트는 `mlx-community/Phi-3.5-mini-instruct-4bit`와 `mlx-community/Qwen3-4B-4bit`뿐이어서 reranker real-checkpoint gate는 점수를 계산하지 않고 soft-skip됐다.

### D. 최종 hardening 검증 (NVIDIA GB10, compute capability 12.1, CUDA, 2026-09-03)

| 체크 | 결과 |
|---|---|
| `cargo fmt --all -- --check` 및 `git diff --check` | clean |
| `cargo clippy --workspace --all-targets --profile test-fast --features cuda -- -D warnings` | clean(이 diff 밖의 기존 C++ compiler warning은 남아 있음) |
| `cargo test --profile test-fast --features cuda -p mlxcel-core --lib streams:: -- --test-threads=1` | 최종 hardening patch 전후 각 18 passed |
| `cargo test --profile test-fast --features cuda --lib execution::runtime -- --test-threads=1` | 19 passed |
| `MLXCEL_DEVICE=cpu cargo test --profile test-fast --features cuda --lib execution::runtime -- --test-threads=1` | 19 passed |
| `cargo test --profile test-fast --features cuda --lib -- --test-threads=1 multimodal:: rerank::real_checkpoint_tests` | 262 passed, 26 ignored; 로컬에 있는 real reranker checkpoint가 정상 실행됨 |
| CPU-only `mlxcel-core` backend-availability 회귀 테스트 | 1 passed |
| CPU-only root `execution::runtime` 테스트 | 19 passed |
