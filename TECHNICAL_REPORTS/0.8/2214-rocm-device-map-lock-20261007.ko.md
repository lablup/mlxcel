# 기술 보고서: PR #2214 - rocm::device()의 디바이스 맵에 락 적용

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 head `05368e26`, main `454500b9` 기준, PR 열림, 머지 대기. #2197 종료(#1801의 일부).

**언어**: C++ 및 HIP(ROCm 오버레이 `device.cpp`, `device.h`, `event.hip`), Rust(신규 `tests/rocm_device_map_concurrency.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`)

**위험도**: 낮음. 이 경합은 현재 존재하는 모든 호스트에서 잠재 상태이고, 변경은 primitive당 경합 없는 shared lock 한 번을 더할 뿐이며, 커널, 인자, 결과는 바뀌지 않는다. 새로운 동작은 `Device` 객체와 HIP 이벤트 풀을 의도적으로 누수시키므로 종료 시 소멸자가 더 이상 실행되지 않는다는 점 하나다. Metal과 CUDA 코드 경로는 건드리지 않는다.

## 요약

이슈 #2197은 JIT 모듈 캐시와 HIP 이벤트 풀에 락을 건 #2196의 리뷰에서 나왔다. 리뷰는 `rocm::device()`에서 같은 패턴을 찾았다. 이 함수는 HIP 인덱스마다 `Device` 하나를 함수 static `std::unordered_map<int, Device>`에 두고, 평가를 수행하는 어느 스레드에서든 모든 primitive 실행마다 락 없이 읽으며, miss 시에는 락 없이 `try_emplace`를 했다. `record_stream_error`, `clear_all_encoders`, `Device::clear_encoders`에도 같은 빈틈이 있었다.

이 경합은 지금은 도달할 수 없다. 모든 insert는 스트림 생성에서 나오고 `mlx::core::new_stream`이 이를 직렬화하며, GPU가 하나인 호스트에서는 키가 0뿐이고 어느 스레드가 평가하기 전에 삽입된다. GPU 1의 첫 스트림이 GPU 0의 평가와 겹칠 때 도달 가능해지는데, #486과 #488의 멀티 GPU 작업이 이를 만든다. 이 PR은 그보다 먼저 고친다.

맵은 이제 누수시킨 `DeviceTable`(`std::shared_mutex`와 맵)이며, 익명 네임스페이스의 두 헬퍼로만 접근할 수 있다. `device()`는 shared lock으로 조회하고, 없는 `Device`는 재확인 후 unique lock 아래에서 만든다. 이벤트 풀의 맵과 뮤텍스도 누수시킨다. 포크는 upstream CUDA의 즉시(eager) 생성 대신 디바이스별 지연 생성을 유지한다.

GPU가 하나인 이 호스트에서 테스트는 insert 경합을 재현하지 못한다. 수정을 적용하면 10회 중 0회, `device.cpp`를 main으로 되돌려도 10회 중 0회 실패했다. 테스트는 조회 경로, 인덱스당 `Device` 하나라는 성질, 깨끗한 종료를 지키는 용도다. 두 번째 GPU 테스트는 GPU가 2개 미만이면 건너뛰며 어디에서도 실행된 적이 없다. 테스트를 쓰면서 별개의 버그도 드러나 #2213으로 등록했다. 멀티스레드 평가에서 호스트가 쓴 8바이트 스칼라가 앞서 다른 스칼라가 들고 있던 값을 읽는다. #2213이 고쳐질 때까지 테스트는 전체 길이 상수를 쓴다.

## 1. 문제 정의

### 1.1 공유 상태 목록

main `84d3a7bc` 기준이며 별도 표기가 없으면 모두 `device.cpp`에 있다.

| 상태 | 이전의 보호 | 문제 |
|---|---|---|
| `get_devices()` 뒤의 함수 static `unordered_map<int, Device>` | 없음 | `device()`가 `find`를 하고, miss 시 `hipSetDevice`, `ensure_device_flags`, 락 없는 `try_emplace`를 수행했다. |
| `record_stream_error` | 없음 | 이벤트 오류 경로에서 같은 맵을 락 없이 읽었다. |
| `clear_all_encoders()` | 없음 | 맵을 락 없이 순회했다. |
| `Device::clear_encoders()` | 없음 | `get_command_encoder`와 `find_encoder`가 잡는 `encoders_mtx_` 없이 `encoders_`를 비웠다. |
| `HipEventPool` 맵과 뮤텍스(`event.hip`) | #2196에서 락이 걸렸으나 소멸 가능 | 정적 소멸 중에 해제되는 `HipEvent`가 이미 소멸된 맵에 push한다. 맵 소멸자는 풀에 있는 모든 핸들에 `hipEventDestroy`를 호출했고, 이는 오류가 난 디바이스에서 실패한다(LOCAL_FIXES 항목 7). |

`device()`는 `gpu::eval`이 실행하는 모든 primitive(`get_command_encoder`, `finalize`, `synchronize`, `new_stream`)에서 호출되고, qmm, SDPA, flash attention, conv, compiled, fp8 convert의 `eval_gpu`에서도 직접 호출된다. 서버의 scheduler, embedding, rerank, audio 워커는 각자 스레드 로컬 스트림에서 평가한다.

### 1.2 도달 가능성

- 모든 insert는 스트림 생성에서 나오고, `mlx::core::new_stream`이 그 주변에서 `all_streams()` unique lock을 잡으므로 어떤 호스트에서도 첫 insert 두 개가 겹치지 않는다.
- GPU가 하나인 호스트에서는 키가 0뿐이며, 프로세스의 첫 스트림이 어떤 스레드가 평가하기 전에 삽입한다. 이후 호출은 더 이상 바뀌지 않는 맵에 대한 `find`일 뿐이다.
- 경합에는 GPU 두 개가 필요하다. GPU 1의 첫 스트림(insert)이 GPU 0에서 평가 중인 다른 스레드(같은 버킷을 훑는 `find`)와 겹칠 때다. 이는 데이터 레이스이며 정의되지 않은 동작이다.
- 프로덕션에서 인덱스 1을 쓰는 곳은 없다. 0이 아닌 `--main-gpu`는 거부되고, `new_stream_on_gpu`, `set_default_gpu_device`, `new_thread_local_stream_on_gpu`는 테스트 밖에 호출자가 없다. #486과 #488의 멀티 GPU 작업이 이를 도달 가능하게 만든다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `device.cpp` | `device_table()`과 `find_device()` 뒤의 누수된 `DeviceTable`. `get_devices()` 제거. shared lock 조회와 unique lock 생성을 쓰는 `device()`. 같은 락을 쓰는 `record_stream_error`, `clear_all_encoders`. `Device::clear_encoders`는 `encoders_mtx_` 아래에서 swap하고 락 밖에서 소멸. |
| `device.h` | `device()`와 `clear_all_encoders()`의 주석. |
| `event.hip` | `HipEventPool`의 맵과 뮤텍스를 누수. |
| `tests/rocm_device_map_concurrency.rs`(신규) | 새 프로세스 테스트 두 개(4.1절). |
| `patches-rocm/LOCAL_FIXES.md` | 항목 39. |

커밋 1개, 파일 5개, 533줄 추가, 21줄 삭제.

## 3. 설계

### 3.1 디바이스 테이블

`DeviceTable`은 `std::shared_mutex`와 `unordered_map<int, Device>`를 가진다. `new`로 할당하고 소멸시키지 않는다. `device_table()`과 `find_device()`는 익명 네임스페이스에 있어 `device.cpp` 밖에서는 맵에 닿을 수 없고, `get_devices()`와 그 전방 선언은 사라졌다(`grep -rn get_devices src/lib/mlx-cpp/patches-rocm` 결과 없음).

`find_device(index)`는 shared lock을 잡고 인덱스를 찾아 `Device*` 또는 null을 돌려준다. 호출자가 결과를 쓰기 전에 락은 풀린다. 어떤 항목도 지우지 않으므로 `Device&`는 프로세스가 끝날 때까지 유효하다.

### 3.2 `device()`

- **hit.** `find_device` 후 반환. shared lock 한 번, HIP 호출 없음. 모든 primitive가 타는 경로다.
- **miss.** unique lock을 잡고 다시 찾아, 여전히 없을 때만 `hipSetDevice(index)`, `ensure_device_flags(index)`, `try_emplace(index, index)`를 실행한다. 한 인덱스를 동시에 처음 쓰는 여러 스레드는 `Device` 하나를 만든다. 생성은 요청한 인덱스만 건드린다.
- **실패.** 생성자나 HIP이 예외를 던지면 아무것도 삽입되지 않고, 언와인드가 락을 풀며, 다음 호출이 다시 시도한다.

### 3.3 `Device` 안으로 들어가는 호출자

`record_stream_error`는 `find_device`를 호출한 뒤, 테이블 락을 푼 상태에서 결과에 `find_encoder`를 호출한다. `clear_all_encoders`는 shared lock 아래에서 `Device*` 값을 벡터에 복사하고 락을 푼 다음 각각에 `clear_encoders()`를 호출한다. 규칙은 `device()`의 생성 구간을 제외하고는 `Device` 안으로 들어가는 동안 테이블 락을 잡지 않는다는 것이다. 따라서 `record_stream_error`나 `device()`에 닿는 encoder 소멸자도 재귀 불가능한 락에서 교착되지 않는다. 락 순서는 테이블, 그다음 `encoders_mtx_`이며, 테이블 락을 잡은 채 다른 것을 기다리지 않는다.

`Device::clear_encoders()`는 `encoders_mtx_` 아래에서 `encoders_`를 지역 맵으로 swap하고, 락이 풀린 뒤 지역 맵이 encoder를 소멸시킨다. 진행 중인 작업을 기다리고 HIP 자원을 해제하는 `~CommandEncoder`가 그 뮤텍스를 잡은 채 실행되지 않는다.

### 3.4 이벤트 풀

`HipEventPool::cache_for`는 이제 `new`로 할당한 `std::map`에서, `mutex()`는 `new`로 할당한 `std::mutex`에서 반환한다. 정적 소멸 중에 해제되는 `HipEvent`가 소멸된 맵에 push할 수 없고, 종료 시 `hipEventDestroy`도 실행되지 않는다. 풀의 핸들은 프로세스와 함께 사라진다. JIT 모듈 캐시가 받은 처리(항목 35, 36)와 같고, #2196 리뷰가 요구한 것이다.

### 3.5 지연 생성과 즉시 생성

pin 81ba1c6a의 upstream CUDA(`mlx/backend/cuda/device.cpp`의 `device(int)`)는 magic-static 초기화 하나에서 `gpu::device_count()`개 인덱스마다 `Device`를 만들어 `std::vector<Device>`에 채우고 누수시키므로, 이후 조회는 읽기 전용이다. 이는 pin에서 직접 확인했다. 포크는 이슈가 든 이유로 디바이스별 지연 생성을 유지한다. `Device::Device`가 `hipSetDevice`로 디바이스를 바인딩하고, 멀티 GPU 호스트에서 요청하지 않은 GPU에 컨텍스트나 큐를 만들면 TB5 링크 너머의 외장 GPU 큐가 멈춘다(`ensure_device_flags` 참고). 즉시 생성은 이 PR의 범위 밖이다.

### 3.6 기각된 대안

- **모든 디바이스 즉시 생성**(upstream): 3.5절 참고.
- **키별 락**: 이슈에 따라 범위 밖.
- **CUDA 오버레이 수정**: 이슈에 따라 범위 밖.

## 4. 검증

### 4.1 테스트

`tests/rocm_device_map_concurrency.rs`(`#![cfg(feature = "rocm")]`)에는 부모 테스트 두 개가 있고, 각각 `#[ignore]` 자식을 새 프로세스로 10번 띄운다. 따라서 프로세스의 첫 스트림(인덱스 0 insert)은 모든 스레드가 배리어에서 풀려난 뒤에 일어난다.

- `streams_created_while_others_evaluate`: 배리어 뒤의 스레드 8개, 16라운드. 매 라운드 모든 스레드가 새 스레드 로컬 스트림을 만들어 설치하고 `arange * 2 + 1`을 네 번 평가하므로, `device()` 조회가 다른 스레드의 스트림 생성 및 평가와 겹친다. 모든 원소를 정확히 비교한다.
- `second_gpu_first_stream_while_first_gpu_evaluates`: `gpu_device_count() >= 2`가 아니면 건너뛴다. 스레드 8개가 GPU 0에서 계속 평가하는 동안 스레드 하나가 GPU 1의 첫 스트림을 만들어 설치하고, 그 위에서 `ones`를 채우고 같은 검사를 한다.

각 자식은 종료 코드 0, 완료 마커 출력, 사용한 인덱스마다 정확히 한 번의 `[mlx-rocm] bound HIP device N:` 출력(몇 스레드가 경쟁하든 인덱스당 `Device` 하나), 그리고 `releasing a HIP handle failed` 줄이 없어야 한다. 마지막 줄은 정적 소멸 때 `hipEventDestroy`나 핸들 해제가 실패했을 때 나타나는 출력이다. 상수는 호스트가 쓴 전체 길이 배열이다(4.3절).

### 4.2 테스트가 보여줄 수 있는 것과 없는 것

이 호스트(gfx1151 한 장)에서는 1.2절의 예측대로 insert 경합이 실행될 수 없다. 이 테스트는 동시 스트림 생성 아래의 조회 경로에 대한 크래시 방어, 종료 동작과 인덱스당 `Device` 하나라는 성질의 검사이지 재현이 아니다. 테스트의 모듈 문서도 그렇게 적었다.

### 4.3 결과

gfx1151(GPU 1개, ROCm 7.15)에서 빌드당 `tests/rocm_device_map_concurrency.rs`를 10회, 회당 자식 10개로 실행했다.

| 빌드 | 실패한 실행 |
|---|---:|
| `device.cpp`를 main으로 되돌림(수정 제거) | 10회 중 0회 |
| 이 PR | 10회 중 0회 |

두 빌드가 구별되지 않으므로 이 수치는 수정이 효과가 있는지에 대해 아무것도 말하지 않는다. 새 코드가 GPU 1개 경로를 깨뜨리지 않는다는 것만 말한다. 두 번째 테스트는 건너뛰기 줄을 출력했고(이 호스트는 GPU 1개로 보고한다), 따라서 인덱스 0 조회 중의 인덱스 1 insert는 어디에서도 실행된 적이 없다. 멀티 GPU 호스트는 없었다. `releasing a HIP handle failed`에 대한 종료 검사는 수정 적용 상태에서 통과하며, 수정 없이 실패한다는 주장은 하지 않는다.

### 4.4 발견: #2213

이슈의 원래 테스트 본문은 `arange * 2 + 1` 검사에 `multiply_scalar`와 `full_like`를 썼다. 그 버전은 수정을 적용하고도 10회 중 9회, 수정 없이는 10회 중 10회 실패했고 출력에는 잘못된 상수가 들어 있었다. 원인은 디바이스 테이블이 아니다.

프로브(배리어 뒤 스레드 8개, `arange * k + c`를 4번 평가하는 16라운드, 한 프로세스에서 모든 원소를 정확히 비교)로 다음을 분리했다.

- 호스트가 쓴 8바이트 스칼라가 앞서 다른 스칼라가 들고 있던 값(다른 스레드의 상수, 또는 같은 스레드의 이전 평가 상수)을 읽는다. 8바이트 이하 스칼라는 `allocator.cpp`의 `SmallSizePool`에서 나온다.
- 라운드마다 새 스트림을 만드는 대신 스레드당 스트림 하나로 끝까지 돌려도 실패했다(5회 중 3회, 스레드별로 값이 다르면 5회 중 5회). 스트림 생성은 요인이 아니다.
- 같은 상수를 호스트가 쓴 전체 길이 배열(각 1 KB)로 바꾸면 10회 중 10회 통과했고, 호스트 데이터가 전혀 없는 연산(`a + a + arange(1, 257)`)도 10회 중 10회 통과했다.

원인은 아직 분리되지 않았다. 8바이트 슬롯 하나가 살아 있는 두 배열에 주어지거나(할당자 경합), GPU가 재사용된 슬롯의 오래된 값을 읽는다(일관성 문제). 둘을 가르는 측정은 아직 없다. #2213(우선순위 high)을 지금 구현하는 중이며, 이슈에 구분 실험이 적혀 있다. 그때까지 #2197 테스트는 #2213을 가리키는 주석과 함께 호스트가 쓴 전체 길이 상수를 쓰고, #2213의 완료 조건에 이 테스트를 `multiply_scalar`와 `full_like`로 되돌리는 항목이 있다.

### 4.5 성능

hit 경로는 primitive당 경합 없는 shared lock 한 번을 더하고 HIP 호출은 더하지 않는다. 측정하지 않았으며, 이 보고서는 처리량에 대해 아무 주장도 하지 않는다.

### 4.6 게이트

- `tests/rocm_jit_module_concurrency.rs`와 `tests/rocm_gpu_faults.rs`: 통과. `cargo clippy --features rocm --test rocm_device_map_concurrency -- -D warnings`: 경고 없음. `grep -rn get_devices src/lib/mlx-cpp/patches-rocm`: 결과 없음.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: 통과(LOCAL_FIXES 39개 항목).
- 오케스트레이터의 `make verify-rocm`, `05368e26`(main `454500b9` 위): 스위트 159개, 12,116 통과, 0 실패, 389 무시, smoke 정상.

검증하지 못한 것: Metal과 CUDA(이 호스트에서 쓸 수 없다. 변경은 `patches-rocm/` 아래 파일과 `rocm` 게이트 테스트만 건드린다).

## 5. LOCAL_FIXES 항목 39

"Device table locked and leaked, event pool leaked, encoders cleared outside the locks" 항목은 락이 없던 지점, 도달 가능성 논거, 새 구조와 락 순서, upstream 비교와 즉시 생성을 기각한 이유, 누수와 그것이 정적 소멸에 주는 효과, GPU 1개에서의 테스트 결과, #2213 메모를 기록한다. 포크 정책 문구로 끝난다: "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2197)." `docs/mlxcelverse/upstream/` 아래에는 아무것도 추가하지 않는다.

## 6. 기술적 선택과 그 이유

- **hit 경로는 shared lock, 생성에만 unique lock.** 모든 primitive가 테이블을 읽고, 인덱스를 처음 쓸 때만 쓴다.
- **unique lock 아래 재확인.** 함께 miss한 여러 스레드가 `Device` 하나를 만든다.
- **`Device` 안으로 들어가는 동안 테이블 락을 잡지 않는다.** encoder 소멸자와의 락 역전과 재귀 경우를 없앤다.
- **테이블과 이벤트 풀을 누수.** upstream CUDA와 JIT 캐시와 같고, 오류가 난 디바이스에서 종료 시점의 HIP 호출을 없앤다.
- **지연 생성 유지.** 포크는 요청받지 않은 GPU를 건드리면 안 된다.
- **도달 가능해지기 전에 고친다.** 그러지 않으면 멀티 GPU 작업이 가리켜 주는 테스트 없이 정의되지 않은 동작의 경합을 만난다.

## 7. 후속 이슈

- **#2213, 멀티스레드 평가에서의 스칼라 풀 상수.** 시리즈의 다음 항목. 원인을 분리하고, 고치고, `tests/rocm_scalar_pool_concurrency.rs`를 추가하고, #2197 테스트를 스칼라 상수로 되돌린다.
- **멀티 GPU 호스트가 생기면 두 번째 GPU 테스트를 실행한다.**

## 8. 남은 위험

- **insert 경합은 한 번도 실행되지 않았다.** 수정은 실패에서 통과로 바뀌는 테스트가 아니라 도달 가능성 논거와 락 구조에 의존한다.
- **두 번째 GPU 테스트는 어디에서도 실행된 적이 없다.** 하네스에 실수가 있다면 GPU 2개 호스트에서 처음 드러난다.
- **누수된 `Device` 객체.** 종료 시 소멸자가 실행되지 않는 것이 목적이지만, 소멸자에 기대던 정리 작업은 이제 명시해야 한다.
- **hit 경로 비용 미측정.** primitive당 경합 없는 shared lock 한 번은 작을 것으로 예상한다.
- **호스트 하나.** 모든 실행은 gfx1151 한 대에서 했다.

## 9. 학습 포인트

- **방어 테스트는 재현이 아니다.** 10회 중 0회와 10회 중 0회를 있는 그대로 보고했다. 이 테스트의 가치는 크래시와 종료 검사에 있다.
- **현실적인 피연산자로 테스트를 쓴다.** 첫 초안에서 스칼라 연산을 쓴 덕에 디바이스 테이블 변경과 무관한 버그를 찾았다.
- **대조군 없이 실패를 테스트 대상 변경의 탓으로 돌리지 않는다.** 수정 없이 10회 중 10회 실패한 것이 재현처럼 보였지만 아니었다.

## 10. 시리즈 전체

이 PR은 ROCm 실행 경로의 공유 전역 상태를 고치는 한 갈래 작업의 다섯 번째 단계이며, 락을 하나 고칠 때마다 다음 전역 상태가 드러났다.

| 단계 | 이슈 / PR | 고친 것 |
|---|---|---|
| 1 | #2196 | JIT 모듈 캐시와 HIP 이벤트 풀 |
| 2 | #2202 | `Device`의 rocBLAS 핸들을 락 아래에서 한 번만 생성 |
| 3 | #2208 | hipBLASLt 핸들, 스트림별 워크스페이스, 스레드 로컬 파이프 캐시 |
| 4 | #2214(이 PR) | 디바이스 맵, 그리고 이벤트 풀 누수 처리 |
| 5 | #2213 | 멀티스레드 평가에서의 8바이트 스칼라 할당(진행 중) |

#2196 리뷰가 디바이스 맵을 찾았다. #2208의 테스트 작업은 bf16 잔차(#2206)를, #2214의 테스트 작업은 #2213을 찾았다. 어느 경우든 스레드 전반에서 모든 원소를 정확히 검사하는 동시성 테스트가 아무도 들여다보지 않은 다음 공유 전역 상태에 닿았다.
