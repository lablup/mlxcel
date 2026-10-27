# PR #2661: CPU 읽기 후 host-shadow flush 생략과 shadow 메모리 집계

**날짜**: 2026-10-27
**상태**: Windows gfx1151에서 완료. Linux 풀 버퍼 경로는 분석과 단위 가드로만 확인
**언어**: C++, HIP, Rust, Markdown
**위험도**: 중간

## 요약

Windows에서는 WDDM이 디바이스 메모리를 CPU 주소 공간에 매핑하지 않기 때문에, ROCm 할당기는 GPU 버퍼마다 고정(pinned) 호스트 shadow를 둡니다(#2603). PR #2661(이슈 #2643)은 #2603 리뷰에서 MEDIUM으로 남겨 둔 후속 과제 두 가지를 해결합니다.
- **읽기:** 호스트 읽기가 더 이상 shadow를 dirty로 표시하지 않으므로, 매 디코드 단계의 토큰 읽기가 다음 커널 전에 버퍼 전체를 호스트에서 디바이스로 복사하는 비용을 치르지 않습니다.
- **집계:** 고정 shadow 바이트를 집계하여 active 메모리 옆에 보고하고, Windows ROCm 메모리 추정에서 디바이스 한도가 아니라 Windows가 보는 RAM을 기준으로 예약합니다.

측정된 개선은 작습니다(Qwen3-0.6B-4bit와 Qwen3-1.7B-bf16에서 +0.2~+2.7%). 출력은 바이트 단위로 동일합니다. 남은 업로드 대부분이 실제 호스트 쓰기이기 때문입니다.

## 1. 문제 정의

`Buffer::raw_ptr()`는 접근할 때마다 `host_dirty = true`를 설정했습니다. `array::data<T>()`는 읽는 쪽과 쓰는 쪽을 구분하지 못하기 때문입니다. const 오버로드가 non-const 오버로드로 `const_cast`합니다. 그러면 `gpu_ptr()`가 다음 커널 전에 버퍼 전체를 flush하므로, logits나 토큰 버퍼를 CPU에서 읽을 때마다 업로드 비용이 들었습니다. shadow 메모리는 `hipHostMalloc`으로 할당되면서 어디에도 집계되지 않았습니다. 그런데 고정 메모리는 검증 호스트에서 Windows가 보는 약 28 GiB RAM으로 제한되고, 디바이스 메모리는 92 GiB 이상입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 읽기 경로 | 새 `rocm::host_read_ptr(const array&)`(`rocm.h`에 선언, `no_rocm.cpp`에 스텁). `data<T>()`처럼 동기화하고, 오래된 shadow를 갱신하며, clean 상태로 둡니다. 읽기만 한다는 것이 증명된 곳에서만 씁니다. 브리지의 `item_*`, `array_to_raw_bytes`, `array_evaluated_bytes`, `try_array_to_raw_bytes`, `staged_write`, `MaskedScatter`의 flatten 검사(이제 포인터와 오프셋 비교)입니다. |
| 버퍼 상태 | `RocmBuffer::host_valid`는 shadow가 디바이스와 같다는 뜻입니다. `ensure_host_shadow(buf, for_write)`는 shadow가 dirty도 valid도 아닐 때만 내려받습니다. `gpu_ptr`는 `host_valid`를 지웁니다. `malloc_async`의 캐시 적중은 두 플래그를 모두 지웁니다. 0바이트 버퍼는 shadow를 받지 않습니다. |
| 해제 경로 | `release_host_shadow`가 유일한 해제 지점입니다. `free_async`, 스칼라 풀, 아레나 래퍼 재사용(이전에는 누수), alien 버퍼와 MoE 워크스페이스 뷰의 셸 전용 해제가 모두 이를 거칩니다. |
| 카운터 | `host_shadow_bytes_`, `host_shadow_downloads_`, `host_shadow_uploads_`를 relaxed atomic으로 두고, `hipHostMalloc`이 성공한 뒤에만 움직입니다. `rocm::host_shadow_stats()`, `MLX_ROCM_HOST_SHADOW_STATS=1`(종료 시 출력), `memory::snapshot()`, `Model loaded` 줄과 로드 로그로 노출합니다. |
| 추정 | `memory_estimate_host_shadow.rs`가 Windows ROCm에서만 `batch × (vocab × 4 + ctx × 4)`를 예약하고 `GlobalMemoryStatusEx`의 가용 RAM과 비교합니다. `fits`는 두 검사를 모두 통과해야 하며, preflight 오류는 호스트 RAM 부족을 따로 밝힙니다. 다른 환경에서는 0입니다. |
| 빌드 | `mlxcel-core/build.rs`가 Windows에서 `advapi32`를 링크합니다. MLX CPU `device_info.cpp`가 레지스트리를 읽어서, 이것이 없으면 테스트 바이너리가 링크되지 않았습니다. |
| 테스트 | `tests/rocm_host_shadow.rs`(호스트에서 실행): 읽기 후 커널은 업로드를 추가하지 않음, 쓰기 후 커널은 업로드 하나를 추가하고 커널이 값을 봄, 읽기-쓰기-커널은 flush함, 오래된 valid shadow를 가진 재사용 버퍼는 디바이스 바이트를 제공함, 방금 읽은 버퍼를 커널이 제자리에서 덮어쓰면 shadow가 무효화됨, 빈 배열은 shadow를 받지 않음, 바이트 카운터가 시작값으로 돌아옴. 기존 추정기 테스트 세 개의 환경 변수 경합도 고쳤습니다. |

## 3. 기술적 선택과 그 이유

### 3.1 기본은 dirty, 증명된 읽기만 clean

이슈는 쓰는 지점에서 쓰기를 표시하는 방식을 선호했고, 내용 검사를 대안으로 두었습니다. 쓰는 쪽을 전수 조사해 보니 모든 쓰기가 MLX 코어나 CPU 백엔드에 있었습니다. CPU 연산 출력, `array::init`, gguf 리더, 호스트 fill입니다. 쓰는 지점에서 표시하려면 `mlx/array.h`를 새 코어 오버레이로 두어야 하고, 그래도 constness에 의존하므로 const를 벗겨 내는 쓰기는 조용히 놓칩니다. `gpu_ptr()`에서 내용을 검사하면 커널 인자마다 버퍼를 한 번씩 훑어야 합니다. 반대 설계는 알 수 없는 접근을 모두 안전한 dirty 경로에 두고 읽기가 증명된 곳에서만 flush를 건너뛰므로, 놓친 경우의 결과가 데이터 손상이 아니라 불필요한 복사입니다.

### 3.2 무효화 지점이 `gpu_ptr`인 이유

리뷰에서 빌드된 ROCm 백엔드 전체를 검색했습니다. 커널, hipMemcpy와 hipMemset 대상, JIT 및 커스텀 커널 인자, `Load::eval_gpu`까지 모든 디바이스 쓰기가 const 오버로드를 포함해 `gpu_ptr`에서 포인터를 얻습니다. HIP 그래프는 꺼져 있어(`use_hip_graphs()`가 false) 커널은 인코딩될 때 실행되고, `host_access`는 이전처럼 동기식 다운로드 전에 디바이스를 동기화합니다. 따라서 앞선 쓰기가 끝나지 않은 상태에서 읽기가 `host_valid`를 설정할 수 없습니다. donation과 `copy_shared_buffer`는 `RocmBuffer`를 공유하므로 플래그가 바이트와 함께 이동하며, 제자리 donation 테스트가 이를 직접 확인합니다.

### 3.3 작고 명시적인 추정 항목

실측한 상주 shadow 바이트는 로드 후 0, 종료 시 8바이트였습니다. 가중치가 `gpu_ptr`로 로드되어 shadow를 만들지 않기 때문입니다. 예약량은 시퀀스마다 f32 logits 한 행과 int32 컨텍스트 id(Qwen3 4096 토큰 기준 623 KB)입니다. Windows에서 고정 메모리를 제한하는 것이 OS가 보는 RAM이므로 그것과 비교합니다. 판단을 바꾸는 일은 드물고, 올바른 한도를 검사와 보고에 넣는 것이 목적입니다.

## 4. 검증

모두 Windows 11, Radeon 8060S gfx1151, ROCm 10.1.0에서 실행했습니다.
- **처리량**(`generate -n 256 --temp 0`, 98토큰 프롬프트, 5회, 중앙값, 기준은 `main` cbe7919f): Qwen3-0.6B-4bit prefill 81.23에서 83.42 tok/s, decode 102.58에서 104.02. Qwen3-1.7B-bf16 prefill 70.62에서 72.38, decode 31.90에서 31.95. 10쌍 모두 출력이 바이트 단위로 동일. 서버 채팅(일반, 스트리밍)과 `/v1/messages` 정상.
- **카운터**(11토큰 실행): 다운로드 913, 업로드 899. clean으로 남은 읽기 14건이 개선분이고, 나머지는 `array::init`으로 호스트에서 만든 스칼라(토큰당 약 80개)입니다.
- **테스트:** `rocm_host_shadow` 3개 모두 통과(위 사례 전부), `memory_estimate` lib 테스트 72개, fmt, mlxcel-core 및 mlxcel lib, 바이너리, 새 테스트에 대한 clippy, 오버레이 레코드, 환경 변수 문서, 라이선스 헤더.

## 5. 남은 공백

- **업로드 대부분은 그대로:** 디코드 토큰당 약 80개의 호스트 생성 스칼라와 인덱스 배열은 `array::init`을 통한 실제 쓰기입니다. 이를 디바이스에서 만들거나 묶는 것이 다음 개선 지점이며 별도 변경입니다.
- **Linux 풀 버퍼:** Linux에서도 async 풀을 켜면 스트림 순서 풀 버퍼에 shadow가 생기고 새 읽기 경로가 적용되지만, Linux 호스트에서 실행하지 않았습니다.
- **잠재 경로:** `decode_capture_replay`는 `gpu_ptr` 없이 버퍼를 쓸 수 있고(현재 호출자 없음), shadow 할당이 할당기 뮤텍스 밖에서 일어나 두 스레드가 같은 버퍼를 동시에 건드리면 이중 할당될 수 있습니다(현재 그런 경로 없음).
- **기존 Windows 테스트 빌드 공백:** `cargo test --features rocm` 전체가 Windows에서 링크되지 않고, mlxcel-core lib 테스트와 workspace clippy가 컴파일되지 않아 Rust 단위 가드는 Linux CI에서만 돕니다.
- **Windows에서의 drift 검사:** `rocm_overlay.py`가 `bash`를 실행하는데 이 호스트에서는 Git Bash가 아닌 bash로 연결됩니다. records는 `--pin`으로 통과하고, drift는 리뷰 커밋 전에 개발 단계에서만 실행되었습니다.

## 6. 학습 포인트

- **constness는 쓰기 신호가 아니다:** 계측 지점 전에 API가 constness를 지운다면, 쓰는 쪽이 아니라 증명된 읽는 쪽을 표시해야 합니다. 그래야 놓친 경우의 대가가 정확성이 아니라 성능이 됩니다.
- **크기 0 할당:** `hipHostMalloc(0)`은 성공하면서 null을 반환합니다. null을 실패로 보는 검사는 모든 빈 배열을 크래시로 만듭니다.
- **위험을 직접 테스트하라:** stale 읽기 위험은 방금 shadow를 읽은 버퍼를 커널이 덮어쓸 때만 드러납니다. donation이 실제로 일어났는지(새 shadow가 할당되지 않았는지)를 단언해야 테스트가 의미를 가집니다.
