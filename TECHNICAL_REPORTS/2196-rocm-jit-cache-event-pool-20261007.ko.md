# 기술 보고서: PR #2196 - ROCm JIT 모듈 캐시와 HIP 이벤트 풀 잠금

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 헤드 `c8bf6cf3`(origin/main `84d3a7bc` 위), PR 열림, 머지 대기. #2183을 닫음(#1801의 일부).

**언어**: C++ 및 HIP(ROCm 오버레이 `jit_module.cpp`, `jit_module.h`, `event.hip`, 브리지 `mlx_cxx_bridge.cpp`/`.h`), Rust(`src/lib/mlxcel-core/src/lib.rs`, `rocm_faults.rs`, 신규 `tests/rocm_jit_module_concurrency.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`)

**위험도**: 낮음에서 중간. 이 변경은 ROCm 실행 경로의 프로세스 전역 두 곳에 잠금을 추가할 뿐, 커널, 키, 결과는 바꾸지 않습니다. 비용은 직렬화입니다. hiprtc 컴파일은 이제 한 번에 하나씩 실행되고, 실행되는 동안 다른 스레드의 캐시 조회를 막습니다. 또 모든 `HipEvent`의 생성과 소멸이 뮤텍스를 잡습니다. 이 비용은 측정하지 않았습니다. Metal과 CUDA 코드 경로는 건드리지 않았고, 브리지 픽스처는 ROCm 밖에서 예외를 던지는 스텁으로 컴파일됩니다.

## 요약

이슈 #2183은 `rocm::get_jit_module`이 컴파일된 모듈을 잠금 없는 프로세스 전역 `std::unordered_map`에 메모이즈한다고 보고했습니다. 잠금 없는 `find` 다음에 `try_emplace`가 오고, 그 `JitModule` 생성자는 hiprtc로 수십에서 수백 밀리초 동안 컴파일합니다. PR은 핀 시점 upstream CUDA의 잠금을 그대로 옮깁니다. 누수시키는(leaked) 맵과 누수시키는 `std::shared_mutex`, 공유 잠금 아래의 조회, 그리고 미스일 때 고유 잠금, 두 번째 `find`, 키가 여전히 없을 때만 `try_emplace`입니다. 잠금 없는 접근자 `get_jit_module_cache()`는 제거했고, `JitModule::get_kernel`은 커널별 configured 플래그를 새 `kernels_mtx_` 아래에서 설정합니다.

새 동시성 테스트는 같은 실행 경로에서 잠금 없는 두 번째 프로세스 전역을 드러냈습니다. `event.hip`의 `HipEventPool`입니다. JIT 캐시를 잠그고 이벤트 풀은 main 그대로 둔 상태에서 테스트 파일은 20회 중 20회 실패했습니다(`invalid resource handle` 10회, SIGABRT 8회, SIGSEGV 2회). 일회용 프로브로 원인을 분리했습니다. 이미 컴파일된 커널을 각자 스트림에서 실행하는 8개 스레드는 실패했고, 같은 일을 40번 하는 스레드 1개는 통과했습니다. 모든 `HipEvent`가 풀에서 핸들을 받으므로, 이는 멀티스레드 ROCm 서빙에서 실제로 도달 가능한 크래시이며, 이슈가 다룬 JIT 경쟁보다 더 노출되어 있습니다. PR은 풀을 `std::mutex`로 잠급니다.

테스트가 증명하는 범위는 정직하게 적었습니다. JIT 잠금을 제거하고 이벤트 풀은 잠근 상태에서 20회 중 0회 실패했으므로, JIT 캐시에 대해서 이 테스트는 결정적 재현이 아니라 크래시 가드입니다. 이벤트 풀에 대해서는 매 실행 실패를 재현합니다. 오케스트레이터의 `c8bf6cf3` `make verify-rocm`은 모든 단계를 통과했습니다. 12,006 통과, 0 실패, 384 무시, smoke OK입니다.

## 1. 문제 정의

### 1.1 JIT 캐시 경쟁(#2183)

당시 main에서 `get_jit_module_cache()`는 함수 정적 `std::unordered_map<std::string, JitModule>`을 반환했고, `get_jit_module`은 잠금 없이 `map.find(key)` 후 `map.try_emplace(key, device(mlx_device), name, builder, cache)`를 호출했습니다. 커스텀 커널은 `use_disk_cache = false`를 넘기므로 모든 `fast::hip_kernel`은 프로세스마다 첫 실행에서 컴파일됩니다. 두 가지 실패 형태가 생깁니다.

- 두 스레드가 같은 키를 놓치고, 둘 다 컴파일하고, 둘 다 삽입하려 합니다.
- 테이블을 키우는 삽입이 rehash를 일으키는 동안 다른 스레드가 `find` 안에서 버킷을 순회합니다. 이는 정의되지 않은 동작이며, 크래시나 댕글링 참조로 이어질 수 있습니다.

`Compiled` 프리미티브, `fast::hip_kernel` 포트, `compute_dynamic_offset`의 모든 실행이 캐시 히트를 포함해 `get_jit_module`을 거칩니다. 서버는 여러 OS 스레드에서 평가하며, 각 스레드는 자체 thread-local 스트림을 쓰고 전역 eval 잠금은 없습니다. `BatchScheduler`, `EmbeddingWorker`, `RerankWorker`, `AudioWorker`입니다. `-m <chat> --embedding-model <emb>`는 두 모델을 동시에 서빙하므로, 임베딩 요청이 모듈을 삽입하는 동안 채팅 스케줄러가 `find` 안에 있을 수 있습니다. 새 키는 새 템플릿 인자 조합이 처음 쓰일 때마다 프로세스 수명 후반에도 생깁니다.

`JitModule::get_kernel`도 `kernels_`의 configured 플래그를 잠금 없이 썼습니다. ROCm 호출자 중 `configure_kernel`을 넘기는 곳이 없어 그 경쟁은 무해했지만, CUDA 오버레이는 libtest 기본 스레드 수에서 같은 지점으로 이미 크래시한 적이 있습니다(#1566).

### 1.2 새 테스트가 찾은 이벤트 풀 경쟁

`HipEventPool`은 함수 정적 `std::map<std::pair<int,int>, std::vector<HipEventHandle>>`을 통해 `hipEvent_t` 핸들을 재활용합니다. 모든 `HipEvent`는 생성 시 핸들을 받고 소멸 시 돌려주며, 이는 평가, 대기, 완료 핸들러가 도는 어느 스레드에서든 일어납니다. 잠금은 없었습니다. 같은 벡터에 대한 동시 `push_back`과 `pop_back`은 핸들 하나를 두 이벤트에 주거나 파괴된 핸들을 record에 줄 수 있고, 이는 `hipEventRecord(event_, stream) failed: invalid resource handle`, 힙 손상 abort, segfault로 나타납니다.

이 문제는 이슈에 없었습니다. JIT 잠금만 넣은 PR 첫 버전이 자기 테스트를 돌렸을 때 몇 라운드 안에 실패하면서 드러났습니다. PR은 범위를 넓히기 전에 일회용 프로브로 두 원인을 분리했습니다(4절).

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `patches-rocm/mlx/backend/rocm/jit_module.cpp` | `get_jit_module`이 누수시키는 `static auto* modules` 맵과 `static auto* mtx = new std::shared_mutex`를 가짐. 조회는 `std::shared_lock`, 미스 경로는 `std::unique_lock` 아래에서 `try_emplace` 전에 두 번째 `find`. `get_jit_module_cache()` 제거. `get_kernel`은 configured 플래그 확인과 설정 구간에서 `kernels_mtx_`를 잡음. `#include <shared_mutex>`. |
| `patches-rocm/mlx/backend/rocm/jit_module.h` | `get_jit_module_cache()` 선언 제거, `JitModule`에 `std::mutex kernels_mtx_` 추가. |
| `patches-rocm/mlx/backend/rocm/event.hip` | `HipEventPool::create`와 `release`가 캐시 주변에서 `std::mutex`를 잡음. 캐시 미스 시 `hipEventCreateWithFlags`는 잠금 밖에서 실행. `cache_for`는 "호출자가 mutex()를 잡는다"는 계약과 함께 private으로. |
| 브리지(`mlx_cxx_bridge.cpp`/`.h`, `lib.rs`, `rocm_faults.rs`) | 테스트 전용 픽스처 `rocm_jit_race_probe_array(input, variant)`와 래퍼 `mlxcel_core::rocm_faults::jit_race_probe_array`. 커널 `mlxcel_jit_race_probe_v<variant>`, `out[i] = inp[i] * 2 + 1`, `n <= 1024` 스레드의 블록 하나, ROCm 밖에서는 예외. |
| `tests/rocm_jit_module_concurrency.rs`(신규) | `same_kernel_first_launch_from_many_threads`와 `distinct_kernels_first_launch_while_others_hit_cache`. 각자 thread-local 스트림을 쓰는 8개 스레드, barrier, 16 라운드, 정확한 출력 검사. |
| `patches-rocm/LOCAL_FIXES.md` | 항목 35(JIT 캐시)와 36(이벤트 풀). |

커밋은 두 개입니다. 테스트와 LOCAL_FIXES를 포함한 수정(`6b5a4f7c`), 그리고 리뷰에서 찾은 항목 35의 문구 수정(`c8bf6cf3`)입니다. 맵 이름은 `cache`가 아니라 `modules`이고, 고유 잠금 아래의 컴파일이 다른 스레드의 캐시 히트도 막는다는 점을 이제 명시합니다. 파일 9개, 378줄 추가, 18줄 삭제입니다.

## 3. 설계

### 3.1 JIT 캐시: 새 패턴이 아니라 upstream의 패턴

이슈는 upstream의 잠금만 요구했고 새로운 설계는 원하지 않았으며, PR은 이를 줄 단위로 따릅니다(핀 81ba1c6a의 `mlx/backend/cuda/jit_module.cpp`, 그리고 CUDA 오버레이).

- **이중 확인 조회.** 흔한 경우인 히트는 공유 잠금만 잡으므로, 컴파일된 커널의 동시 실행끼리는 직렬화되지 않습니다. 미스는 고유 잠금을 잡고 `find`를 반복합니다. 두 잠금 사이에 다른 스레드가 키를 삽입했을 수 있기 때문입니다. 키마다 `JitModule`은 하나만 생성되고, 모든 호출자가 같은 객체의 참조를 받습니다.
- **누수시키는 저장소.** `static auto* modules = new std::unordered_map<...>`는 파괴되지 않습니다. upstream의 이유는 메인 스레드가 정적 객체를 정리한 뒤에도 사용자 코드가 JIT 코드를 실행할 수 있다는 것입니다. 이 포크에서는 이점이 하나 더 있습니다. 정적 정리 중에 `hipModuleUnload`가 실행되지 않는데, 장애가 난 디바이스에서는 그 시점의 모든 HIP 호출이 실패합니다(LOCAL_FIXES 항목 7). 호출자는 잠금이 풀린 뒤에도 `JitModule&`를 들고 있으므로 맵에서 반환된 참조가 프로세스 내내 유효해야 하는데, `unordered_map`의 rehash는 노드를 옮기지 않으므로 동시 삽입 아래에서도 참조 안정성이 유지됩니다.
- **`get_jit_module_cache()` 제거.** 헤더에 공개되어 있었지만 호출하는 곳이 없었습니다. 남겨 두면 잠금 없이 맵에 닿는 길이 남으므로, 감싸지 않고 제거했습니다.
- **오류 처리는 그대로.** 컴파일이나 로드 실패는 아무것도 삽입되기 전에 생성자 밖으로 예외를 던지고, 스택 해제가 고유 잠금을 풀며, 다음 호출자가 재시도합니다.
- **configured 플래그.** `kernels_`는 생성자에서(쓰기 잠금 아래) 한 번 채워지고 이후 구조가 바뀌지 않으므로 `kernels_.find`는 잠금 없이 둡니다. configured 비트의 확인과 설정만 `kernels_mtx_`를 잡으며, 이는 #1566 이후 CUDA 오버레이와 같습니다. 잠금은 `configure_kernel` 동안 유지되므로, 앞으로 생길 호출자의 콜백은 `get_kernel`에 재진입하면 안 되고, 코드 주석이 이를 적어 둡니다.

부수 효과로, 컴파일이 직렬화되면서 두 `StderrSuppressor`가 서로의 fd 2를 바꿔치기하는 일도 없어집니다.

### 3.2 JIT 캐시에서 기각한 방안: 키별 잠금

키별 `std::once_flag`나 future를 쓰면 서로 다른 키의 컴파일이 병렬로 돌고, 무관한 키가 컴파일되는 동안에도 캐시 히트가 진행됩니다. 이슈는 upstream과 달라지고 hiprtc 컴파일은 프로세스당 키마다 한 번뿐이라는 이유로 이를 기각했습니다. 그 선택의 비용은 이제 항목 35에 적혀 있습니다. 한 스레드가 고유 잠금 아래에서 컴파일하는 동안, 히트를 포함한 다른 모든 스레드의 `get_jit_module` 호출이 기다립니다. 서버에서 새 템플릿 조합의 첫 `fast::hip_kernel` 실행은 다른 워커들의 실행을 컴파일 한 번 동안 멈출 수 있습니다. 그 멈춤은 측정하지 않았습니다.

### 3.3 이벤트 풀: upstream의 첫 스레드 규칙이 아니라 뮤텍스

upstream CUDA의 `CudaEventPool`(핀 시점 `mlx/backend/cuda/event.cu`)은 같은 경쟁을 다르게 피합니다. 풀을 처음 쓴 스레드에서만 이벤트를 캐시하고, 다른 스레드는 매번 새 이벤트를 만들고 파괴합니다. PR은 이를 따르지 않았고, upstream과 갈라지는 곳은 여기 한 곳입니다.

- mlxcel 서버에서 디코드는 풀을 처음 건드린 스레드가 아니라 워커 스레드(배치 스케줄러와 모델별 워커)에서 돕니다. 첫 스레드 규칙을 쓰면 핫 경로가 이벤트마다 `hipEvent_t`를 만들고 파괴하게 되어, 풀링이 가장 필요한 곳에서 사실상 꺼집니다.
- `cache_for(...)`의 push와 pop 주변의 `std::mutex`는 모든 스레드에서 풀링을 유지합니다. 임계 구역은 맵 조회와 벡터 push 또는 pop 하나입니다. 미스 시 `hipEventCreateWithFlags`는 잠금을 푼 뒤 실행되므로, 느린 드라이버 호출이 다른 스레드를 붙잡지 않습니다.
- 평가 스레드가 하나라면 잠금은 경합이 없습니다.

하지 않은 것: 뮤텍스와 첫 스레드 규칙, 또는 둘 중 하나와 main을 비교한 벤치마크는 없습니다. 뮤텍스를 고른 근거는 측정이 아니라 구조적인 것(디코드가 도는 곳에 캐시를 유지)입니다. 프로파일에서 풀 뮤텍스 경합이 보이면, 공유 폴백을 둔 스레드별 캐시가 다음 단계입니다.

## 4. 검증

### 4.1 20회 실행 표

gfx1151에서 빌드마다 `tests/rocm_jit_module_concurrency.rs`를 20회 실행했습니다.

| 빌드 | 실패한 실행 | 실패 종류 |
|---|---:|---|
| 두 잠금 모두(이 PR) | 20회 중 0회 | 없음 |
| JIT 캐시 잠금 제거, 이벤트 풀 잠금 | 20회 중 0회 | 없음 |
| JIT 캐시 잠금, `event.hip`은 main 그대로 | 20회 중 20회 | `invalid resource handle` 테스트 실패 10회, SIGABRT 8회, SIGSEGV 2회 |

읽는 법:

- **이벤트 풀에 대해 이 테스트는 재현입니다.** 잠금 없는 풀에서 모든 실행이 몇 라운드 안에 실패했습니다.
- **JIT 캐시에 대해 이 테스트는 재현이 아니라 크래시 가드입니다.** 이 호스트에서 JIT 잠금을 제거해도 실패가 한 번도 나지 않았습니다. 경쟁 자체는 코드상 실재하고(잠금 없는 `find`와 rehash하는 `try_emplace`는 정의되지 않은 동작), 테스트는 이슈가 설명한 인터리빙을 그대로 실행하지만, 이를 잡아내지는 못했습니다. JIT 잠금의 회귀는 이 테스트를 통과할 수 있습니다. 이슈의 수용 기준은 이 경우를 명시하라고 요구했고, PR은 명시합니다.

### 4.2 이벤트 풀 분리

범위를 넓히기 전에, 이미 컴파일된 커널만 실행하는(JIT 미스도 삽입도 없는) 일회용 프로브를 두 방식으로 돌렸습니다.

- 각자 스트림을 쓰는 스레드 8개: 같은 `invalid resource handle`로 실패.
- 스레드 1개, 40회 실행: 통과.

이로써 JIT 캐시가 원인에서 빠집니다. 실패에는 동시성이 필요하지만 컴파일은 필요 없고, 그 경로에서 공유되는 가변 상태는 이벤트 풀뿐입니다. 풀을 잠근 뒤 8 스레드 캐시 히트 프로브는 32 라운드를 통과했습니다.

### 4.3 게이트

- `cargo test --profile test-fast --features rocm --test rocm_jit_module_concurrency --test rocm_custom_kernel_jit_key --test rocm_gpu_faults`: 7 통과.
- 새 테스트와 mlxcel-core lib에 대한 `-D warnings` `cargo clippy`: 깨끗함.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`와 `cargo test --test dead_doc_pointers`: 통과.
- 유닛의 `6b5a4f7c`(`84d3a7bc` 위로 rebase) `make verify-rocm`: 테스트 스위트 154개, 12,006 통과, 0 실패, 384 무시. `c8bf6cf3`은 LOCAL_FIXES 문구만 바꾸며, 그 위에서 `verify-rocm-overlay`가 통과합니다.
- 오케스트레이터의 `c8bf6cf3`(origin/main `84d3a7bc` 최신 상태) `make verify-rocm`: 모든 단계 통과, 12,006 통과, 0 실패, 384 무시, smoke OK.

검증하지 않은 것: Metal과 CUDA(이 호스트에 없음, 해당 경로의 코드는 바뀌지 않음), 그리고 두 잠금의 성능 비용.

## 5. LOCAL_FIXES 항목 35와 36

두 항목 모두 `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`의 "Fixes to the fork's kernels" 아래에 들어갑니다.

- **35, JIT 모듈 캐시 잠금.** 잠금 없는 `find`/`try_emplace`, 거기에 닿는 실행 지점과 서버 스레드, 이제 쓰는 upstream 잠금, 바뀌지 않은 `"<device index>:<name>"` 키(이름은 항목 33이 제공), 예외 시 재시도 동작, 컴파일 중 캐시 히트가 멈추는 것을 포함한 직렬화 비용, 항목 7과 관련된 정리 시점 이점, 제거된 접근자, `kernels_mtx_` 채움, 그리고 테스트를 기록합니다.
- **36, HIP 이벤트 풀 잠금.** 잠금 없던 풀, JIT 캐시를 잠근 상태의 20회 중 20회 실패, 8 스레드 대 1 스레드 분리, 생성은 잠금 밖에 둔 뮤텍스, upstream의 첫 스레드 규칙을 쓰지 않은 이유, 32 라운드 통과를 기록합니다.

두 항목 모두 포크 정책 문구로 끝납니다. "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there." `docs/mlxcelverse/upstream/` 아래에는 아무것도 추가하지 않았습니다. 항목 35는 upstream과의 동등성을 되찾는 것이라 제안할 것이 없고, 항목 36은 upstream 설계 대신 포크가 고른 선택입니다.

## 6. 기술적 선택과 그 이유

- **upstream의 JIT 잠금을 그대로 복사.** upstream이 CUDA에서 이미 해결했습니다. 이를 맞추면 오버레이를 핀과 diff로 비교할 수 있고, 포크에서 새 동시성 설계를 피할 수 있습니다.
- **잠금 없는 접근자는 감싸지 말고 제거.** 잠금이 걸린 맵의 참조를 공개하는 것은 호출자가 생기기를 기다리는 잠금 우회입니다.
- **캐시를 누수.** 정리가 시작된 뒤에도 참조가 유효하고, 종료 시 장애 난 디바이스에 HIP 호출을 하지 않습니다.
- **범위를 이벤트 풀까지 확장.** 새 테스트는 이것 없이 통과할 수 없었고, 이쪽이 더 노출된 크래시입니다. JIT 미스 없이 두 스레드가 동시에 평가하기만 하면 됩니다. 별도 PR로 나눴다면 매 실행 실패하는 테스트 파일이 남았을 것입니다.
- **풀에는 첫 스레드 캐싱 대신 뮤텍스.** 디코드가 도는 서버 워커 스레드에서 풀링을 유지합니다. 측정하지 않았고, 핫 경로가 어디서 도는지로 정당화합니다.
- **테스트를 실패할 때까지 조정하지 않고 크래시 가드라는 한계를 보고.** 이 호스트에서 JIT 경쟁을 재현하려면 오버레이 내부에 타이밍 훅이 필요하고, 이슈는 대신 정직한 집계를 요구했습니다.

## 7. 이 작업에서 등록한 후속 이슈

모두 이 PR 작업이나 리뷰 중에 발견했고 main에 이미 있던 문제입니다.

- **#2197, `rocm::device()` 맵.** `device()`는 프로세스 전역 `std::unordered_map<int, Device>`에 잠금 없는 `find`/`try_emplace`를 하며, 모든 실행에서 읽힙니다. 잠재적이며 멀티 GPU에서만 발생합니다. 모든 삽입은 `all_streams()` 잠금 아래의 스트림 생성에서 오고, GPU가 하나면 유일한 키가 어떤 스레드도 스트림을 갖기 전에 삽입됩니다. 스트림이 GPU 1에 놓이는 순간(#486, #488) 도달 가능해집니다. 이 이슈는 이벤트 풀의 맵이 파괴되는 정적 객체라서 정적 정리 중에 해제되는 `HipEvent`가 파괴된 맵에 push하는 문제도 다룹니다. JIT 캐시처럼 누수시키면 고쳐집니다.
- **#2198, rocBLAS 핸들 초기화와 스트림 재바인딩.** `Device::get_rocblas_handle()`는 핸들을 만들기 전에 initialized 플래그를 세우므로 두 번째 스레드가 null 핸들을 받을 수 있습니다. 또 모든 rocBLAS GEMM이 공유 핸들 하나에 잠금 없이 `set_rocblas_stream`을 호출하므로, 스레드 A의 재바인딩과 A의 GEMM 사이에 스레드 B가 재바인딩하면 A의 GEMM이 B의 스트림에 올라갑니다. 두 모델을 동시에 서빙하면 GPU 하나에서도 도달 가능하며, 재바인딩 경쟁은 첫 GEMM뿐 아니라 모든 rocBLAS GEMM에서 열려 있습니다.
- **#2200, hipBLASLt 공유 상태.** 디바이스마다 32 MB 워크스페이스 하나를 모든 스트림이 공유하고(워크스페이스가 필요한 동시 GEMM 두 개가 서로의 출력을 망가뜨림), `ensure_workspace`는 잠금 없이 해제하고 재할당하며, 초기화 경로는 핸들과 워크스페이스가 준비되기 전에 `initialized`를 공개하고, 캐시된 파이프 matmul이 실패하면 다른 스레드가 아직 `hipblasLtMatmul`에 넘기고 있을 수 있는 `GemmPipe`를 파괴하고 지웁니다. hipBLASLt는 bf16과 fp16 GEMM의 기본 경로이므로 GPU 하나에서도 도달 가능합니다.

리뷰에서 나온 작은 메모 하나: 이동된(moved-from) `HipEvent`는 풀에 null 핸들을 돌려줍니다. 오늘은 이동하는 곳이 없습니다.

## 8. 남은 위험

- **JIT 잠금에는 실패하는 테스트가 없습니다.** 누군가 잠금을 약화해도 gfx1151에서 이 테스트는 아마 통과할 것입니다.
- **잠금 비용 미측정.** 모든 `get_jit_module`의 공유 잠금, 컴파일 중 히트의 멈춤, 모든 이벤트의 풀 뮤텍스 모두 벤치마크하지 않았습니다. GEMM에 비해 작을 것으로 예상하지만, 예상일 뿐입니다.
- **잠금 없는 전역이 더 남아 있습니다.** #2197, #2198, #2200이 열려 있습니다. 이들이 들어가기 전까지 멀티 모델 ROCm 서빙은 여전히 rocBLAS와 hipBLASLt 상태에서 경쟁할 수 있습니다.
- **호스트 하나.** 모든 실행이 gfx1151 한 대에서 이뤄졌습니다. 타이밍에 의존하는 경쟁은 다른 ROCm 타깃에서 다르게 행동할 수 있습니다.

## 9. 학습 포인트

- **멀티 모델 서버는 하나의 `Device`를 별도 스트림의 워커 스레드들이 공유하며, ROCm에서 이 구성이 동시에 실행된 적이 없었습니다.** 실행 경로의 모든 프로세스 전역과 `Device`별 지연 필드는 제출 스레드가 하나라고 가정하고 작성되었습니다. 실제로 8개 스레드를 동시에 돌린 첫 테스트가 몇 라운드 안에 두 번째 경쟁을 찾았습니다.
- **잠금 하나를 고칠 때마다 다음 잠금 없는 전역이 드러납니다.** JIT 캐시를 잠그자 실패가 이벤트 풀로 옮겨 갔고, 이 PR의 리뷰가 디바이스 맵(#2197)을, #2197의 리뷰가 rocBLAS(#2198)를, #2198의 리뷰가 hipBLASLt(#2200)를 찾았습니다. 핫 경로의 전역 하나가 잠금 없이 되어 있다면, 보고된 하나만 고치지 말고 이웃도 그렇다고 가정하고 경로 전체를 감사하십시오.
- **범위를 넓히기 전에 분리하십시오.** 이미 컴파일된 커널에 대한 8 스레드 대 1 스레드 프로브가 실행 두 번으로 JIT 캐시를 원인에서 제외했고, 덕분에 범위 변경을 리뷰에서 방어할 수 있었습니다.
- **수정마다 있을 때와 없을 때의 실패를 따로 세십시오.** 20회 표는 테스트가 실제로 어느 잠금을 지키는지 보여 줍니다. "JIT 잠금 제거" 행이 없었다면 통과하는 테스트가 JIT 수정의 증거처럼 보였을 것입니다.
- **upstream과 갈라지려면 이 코드베이스에 묶인 이유가 필요합니다.** upstream의 첫 스레드 이벤트 캐싱은 제출자가 하나인 모델에 맞고, mlxcel의 워커 스레드 디코드는 그렇지 않으며, 이것이 뮤텍스를 고른 근거의 전부입니다.
