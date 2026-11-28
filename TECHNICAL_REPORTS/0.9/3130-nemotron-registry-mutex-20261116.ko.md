# 기술 리포트: PR #3130 - NemotronH decode handle registry를 mutex로 보호

**작성일**: 2026-11-16
**상태**: 완료 (#3124 해결)
**언어**: C++, Rust
**위험도**: Low

---

## 요약

#3124는 Metal 테스트 묶음을 한 프로세스에서 돌릴 때 프로세스가 간헐적으로 죽는 문제였다. 원인은 Metal 상태가 아니라 mlxcel C++ bridge의 data race였다. NemotronH decode handle registry가 lock 없는 전역 `std::unordered_map`이었고, handle 카운터도 원자적이지 않았다. 이 PR은 registry에 mutex를 두고, 종료 시 static destructor가 돌지 않도록 registry를 leak한다.

---

## 1. 문제 정의

모든 `NemotronHModel`은 만들어질 때 `mlx_cxx_nemotron.cpp`의 registry에 handle을 등록하고 `Drop`에서 지운다. `server::batch` 테스트 모듈 여러 개(`prompt_count_tests`, `request_memory_tests`, `scheduler_tests`)가 병렬 libtest 스레드에서 작은 NemotronH 모델을 만들기 때문에 insert, erase, rehash가 경쟁했다. M3 Ultra의 crash report에는 같은 heap 손상이 세 가지 형태로 나타났다.

- 해시 테이블 순회 중 `nemotron_free_model`에서 SIGSEGV.
- 관련 없는 할당에서 malloc freelist 검사가 SIGTRAP.
- `test result: ok`가 출력된 뒤 `exit` 중 map의 static destructor에서 "pointer being freed was not allocated" SIGABRT. 이슈의 "lockf exit 70" 사례다.

서버가 NemotronH 모델을 동시에 로드하거나 내릴 때도 프로덕션에서 같은 위험이 있다.

수정 전 `server::batch::`는 기본 스레드 수로 41회 중 2회 크래시했다. 같은 필터를 `--test-threads=1`로 돌리거나 model-owned cache 모듈을 어느 설정으로 돌려도 크래시는 없었다(90회).

---

## 2. 기술적 선택과 그 이유

### 2.1 mutex 하나, 모델 소멸은 lock 밖에서

register와 free가 lock을 잡는다. free는 lock 안에서 `unique_ptr`만 꺼내고 lock을 놓은 뒤 모델을 소멸시킨다. MLX 배열 해제가 다른 스레드를 기다리게 하지 않는다.

### 2.2 decode는 lookup할 때만 lock

`nemotron_decode_step`은 lock 안에서 모델 포인터만 복사하고 step은 lock 없이 돈다. `unordered_map`은 insert나 rehash 때 원소를 옮기지 않고, handle을 가진 Rust 모델이 호출 내내 borrow되어 있어 step 중에 handle이 해제될 수 없으므로 안전하다. step 전체에 lock을 잡으면 모델 간 decode가 이유 없이 직렬화된다.

### 2.3 registry를 leak

registry는 함수 안의 `*new Registry()`로, 다른 bridge 캐시가 이미 쓰는 방식이다. static 객체면 `exit` 때 다른 스레드가 아직 쓰는 중에 destructor가 돌 수 있고, 그것이 세 번째 증상이었다.

---

## 3. 변경 요약

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 추가 테스트 | 1 |

- `mlx_cxx_nemotron.cpp`: `Registry` 구조체(mutex, map, 카운터), register/free/lookup에 lock.
- `nemotron_registry_tests.rs`: 레이어 없는 모델로 8 스레드 × 4000회 register/free, 모든 handle이 서로 다른지 확인.

### 검증 (M3 Ultra)

- 새 stress 테스트: 수정 전 4회 중 4회가 `nemotron_free_model`이나 `nemotron_register_model`에서 크래시했고 5번째는 livelock. 수정 후 10회 중 10회 통과.
- 기본 스레드 수의 `server::batch::`: 수정 후 20회 중 20회 통과. 루트 lib 테스트의 `nemotron` 필터 통과(202개).
- fmt, clippy(`--lib --tests -D warnings`, 두 crate), contract check 통과.

---

## 4. 후속 조치

- 이 호스트의 예전 crash report에는 다른 MLX revision으로 빌드된 테스트 바이너리가 `metal_gated_delta_forward_impl`을 거쳐 `mlx::core::fast::GeneratedSourceCache::get`에서 SIGSEGV를 낸 기록이 있다. 이 수정으로 설명되지 않으며, 현재 고정된 MLX에서는 재현되지 않았다.
