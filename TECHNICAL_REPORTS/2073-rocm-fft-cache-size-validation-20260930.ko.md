# 기술 보고서: PR #2073 - ROCm에서 MLX_ROCM_FFT_CACHE_SIZE 값 검증

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (ROCm overlay 헤더), Rust (통합 테스트), Markdown

**위험도**: 낮음 (ROCm overlay에 헤더 전용 파싱 함수 하나와 생성자 guard 두 개, `fft.hip` 주석, ROCm 전용 새 테스트와 문서가 전부입니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #2051(에픽 #1801의 일부)은 PR #2049(#1876)의 보안 리뷰에서 발견되었고, main에 이미 있던 문제입니다. ROCm overlay는 hipFFT plan cache의 용량인 `MLX_ROCM_FFT_CACHE_SIZE`를 검사 없는 `std::stoul`로 읽었습니다. 잘못 입력한 값이 메시지와 함께 거부되지 않았습니다. `0`은 첫 plan 삽입에서 undefined behavior로 이어졌고, 숫자가 아닌 값은 모든 FFT에서 `stoul`이라는 메시지만 담은 예외를 던졌으며, `-1`과 `8abc`는 각각 `SIZE_MAX`와 8로 조용히 받아들여졌습니다. 영향을 받는 것은 GPU에서 FFT를 실행하는 오디오 경로(Kokoro TTS, Phi-4-multimodal 오디오)입니다.

이 PR은 overlay의 `lru_cache.h`에 `capacity_from_env()`를 추가합니다. 이 함수는 `gpu_watchdog_seconds()`가 `MLX_ROCM_GPU_WATCHDOG_SECS`에 이미 적용하던 `strtol` 규칙을 그대로 씁니다. 설정하지 않으면 조용히 기본값을 쓰고, 1부터 `INT_MAX`까지는 그 값을 쓰며, 그 밖의 값은 변수 이름과 기본값을 적은 한 줄을 stderr에 출력한 뒤 기본값을 씁니다. 두 cache 생성자는 이제 upstream CUDA처럼 용량이 0이면 `LRUCache requires capacity > 0.`을 던집니다. `tests/rocm_fft_cache_env.rs`는 값마다 child 프로세스를 하나씩 띄워 경고 줄 수와 GPU rfft 결과를 CPU stream과 비교합니다.

이 PR은 빌드 시스템의 결함도 드러냈고, 지금은 #2075가 추적합니다. overlay의 HIP 오브젝트는 자기 `.hip` 소스에만 의존하므로, 헤더만 고친 수정은 warm build에서 컴파일되지 않습니다.

## 1. 문제 정의

### 함수 내부 static 뒤에 숨은 검사 없는 파싱

`LRUBytesKeyCache` 생성자는 `capacity_ = std::stoul(env)`를 실행했습니다. 환경 변수를 쓰는 사용처는 `fft.hip`의 `fft_plan_cache()` 하나뿐이며, 이는 첫 device FFT에서 초기화되는 함수 내부 static입니다(#2049 이후 기본값 128, 이전에는 8). 값별 동작은 헤더를 호스트 g++ 14로 검사한 probe와 gfx1151에서 확인했습니다.

| 값 | 이전 동작 |
|---|---|
| `0`, `0x10` (0으로 읽힘) | 받아들여짐. 첫 `put()`이 `while (size() >= capacity_)`를 실행하고, `0 >= 0`이 참이므로 빈 `std::list`에 `back()`과 `pop_back()`을 호출합니다. undefined behavior입니다. probe는 `std::bad_alloc`을 던졌고, `-D_GLIBCXX_ASSERTIONS`로는 `!this->empty()` assertion으로 중단됩니다. gfx1151에서는 첫 GPU rfft가 `std::bad_alloc`으로 실패했습니다. |
| `abc`, 빈 문자열 | `what()`이 `stoul`인 `std::invalid_argument`. 변수 이름이 없습니다. |
| `99999999999999999999999` | `std::out_of_range`, 역시 `stoul`. |
| `-1` | `SIZE_MAX`로 조용히 받아들여짐. cache가 절대 evict하지 않습니다. |
| `8abc` | 8로 조용히 받아들여짐. |

static 초기화 중에 예외가 나가면 static은 초기화되지 않은 상태로 남으므로, 이후의 모든 FFT가 생성자를 다시 실행하고 다시 예외를 던졌습니다. 실패는 프로세스가 끝날 때까지 지속되었고, 메시지는 환경 변수를 가리키지 않았습니다.

### upstream CUDA는 그대로 따라 할 대상이 아님

고정된 MLX 81ba1c6a의 upstream CUDA는 `MLX_CUDA_FFT_CACHE_SIZE`를 `env::get_var`를 통해 `atoi`로 읽습니다. `0`이나 숫자가 아닌 값은 0이 되고 생성자가 `LRUCache requires capacity > 0.`을 던집니다. 적어도 undefined behavior는 아니지만 첫 FFT는 여전히 실패하고, `-1`은 여전히 `SIZE_MAX`가 됩니다. CUDA를 그대로 따르면 결함 두 개가 남습니다.

## 2. 변경 요약

- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/lru_cache.h`** (+47/-4): `mlx::core::rocm`에 `inline size_t capacity_from_env(const char* env_var, size_t default_capacity)`를 추가했습니다. `errno`를 초기화한 뒤 base 10 `std::strtol`로 읽고, 읽은 것이 없거나 `*end != '\0'`이거나 `ERANGE`이거나 `v <= 0` 또는 `v > INT_MAX`이면 거부합니다. 거부하면 `[ROCm] ignoring invalid MLX_ROCM_FFT_CACHE_SIZE="<value>" (expected a positive integer); using the default 128`을 출력하고 기본값을 반환합니다. `LRUBytesKeyCache`는 이 함수로 용량을 정하고, `LRUBytesKeyCache`와 `LRUCache(size_t)`는 모두 용량이 0이면 `std::runtime_error("LRUCache requires capacity > 0.")`을 던집니다.
- **`.../rocm/fft.hip`** (+3/-1): `fft_plan_cache()`의 주석이 규칙을 설명하고 `capacity_from_env`를 가리킵니다. 이 수정은 warm build가 `fft.hip`을 다시 컴파일하게 만드는 역할도 합니다(3절).
- **`tests/rocm_fft_cache_env.rs`** (신규, 304줄, `#![cfg(feature = "rocm")]`, `gpu_backend_kind()`가 ROCm이 아니면 건너뜀): parent 테스트가 자기 바이너리를 ignored child 테스트로 값마다 한 번씩, 한 번에 하나씩 `--test-threads=1`로 다시 실행합니다. 값은 거부 대상인 `0`, `-1`, `abc`, 빈 문자열, `8abc`, `0x10`, `99999999999999999999999`, `2147483648`과 수락 대상인 `16`, `1`, `2147483647`, 미설정입니다. child는 길이 400과 512로 GPU rfft를 두 번 실행해 CPU stream과 비교하고(허용 오차 1e-5), 완료 표시를 출력합니다. parent는 stdout과 stderr를 각각 별도 스레드로 읽고, 120초 예산을 넘긴 child는 종료시키며, 종료 코드 0, 완료 표시, 그리고 거부 값이면 변수를 언급하는 stderr 줄이 정확히 하나, 수락 값이면 0개임을 요구합니다.
- **`src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`**: 항목 25(24는 #2052용으로 예약). 값별 이전 동작, 새 규칙, CUDA와의 비교, 빌드 의존성 메모를 담았고 #1813의 upstream 후보로 표시했습니다.
- **`docs/environment-variables.md`**, **`docs/installation.md`**: 유효 범위(2147483647 이하의 양의 정수)와 그 밖의 값은 경고 후 기본값을 쓴다는 점.

커밋: `512d05b3`은 수정, 테스트, 문서이고 `0c287245`는 environment-variables 문단의 줄바꿈을 정리합니다.

## 3. 기술적 선택과 그 이유

### 새 규칙 대신 watchdog 변수의 파싱 규칙을 재사용

overlay에는 이미 검증하는 정수 변수가 하나 있었습니다. `device.cpp`가 `MLX_ROCM_GPU_WATCHDOG_SECS`를 `strtol`로 읽고, 문자열 전체가 숫자여야 하며, 잘못된 값은 경고 후 기본값을 씁니다. 이 규칙을 그대로 쓰면 두 ROCm 변수가 같은 방식으로 동작하고, 문서 한 문장으로 둘 다 설명할 수 있습니다. 같은 이유로 clamp(예를 들어 `0`을 1로 바꾸기)는 채택하지 않았습니다. 두 번째 규칙이 생기고, 사용자가 요청하지 않은 값으로 조용히 실행하게 됩니다.

`strtol`은 앞쪽 공백과 `+` 부호를 받아들입니다(` 16`과 `+16` 모두 16으로 읽히며, 호스트 probe로 확인했습니다). 이는 허용하기로 했습니다. base가 10으로 고정되어 있으므로 `0x10`은 뒤에 붙은 문자 때문에 거부됩니다.

### 상한을 INT_MAX로

상한은 `SIZE_MAX`나 `LONG_MAX`가 아니라 `std::numeric_limits<int>::max()`입니다. CUDA는 같은 역할의 변수를 `int`로 읽으므로 `INT_MAX`를 넘는 값은 어느 backend에서도 의미가 없고, `int`에 맞는 상한을 두면 Linux에서 `long`에는 들어가는 `2147483648`이 사실상 무제한 cache로 받아들여지지 않습니다. 테스트가 양쪽 경계를 고정합니다. `2147483647`은 수락, `2147483648`은 거부입니다.

### 한 번만 경고하고 static은 초기화되게

잘못된 값에서 예외를 던지지 않고 기본값을 반환하기 때문에 경고가 한 번만 출력됩니다. static 초기화가 끝나므로 다시 실행되지 않습니다. 예외를 던졌다면 이슈가 설명한 지속적 실패 패턴이 되풀이되었을 것입니다. 조용히 거부했다면 설정 실수가 숨겨졌을 것이므로, 출력 줄에는 변수 이름, 따옴표로 감싼 값(빈 값이나 공백 값도 보이게), 대신 쓴 기본값을 적습니다.

### 두 생성자 모두에 용량 0 guard 유지

`capacity_from_env()` 이후로는 환경 변수에서 용량 0이 나올 수 없으므로, `LRUBytesKeyCache`의 예외는 앞으로 누군가 기본값 0을 넘기는 경우에 대한 이중 방어입니다. `LRUCache(size_t)`의 유일한 사용처인 `device.h`의 `graph_cache_{400}`은 상수이지만 같은 guard를 두었습니다. `put()`에 같은 `while (size() >= capacity_)` 루프가 있어서 용량 0이면 같은 undefined behavior가 되기 때문입니다. 메시지는 upstream CUDA와 글자 그대로 같습니다.

### 값마다 프로세스 하나

plan cache는 변수를 한 번만 읽는 함수 내부 static입니다. 한 프로세스에서 값 12개를 시험하면 첫 값만 시험하게 됩니다. 테스트는 `tests/rocm_gpu_faults.rs`의 방식을 따라, 변수를 설정한 채 테스트 바이너리를 ignored child 테스트로 다시 실행합니다. child는 전용 환경 변수가 있을 때만 동작하므로 `--include-ignored` 실행이 호스트 환경 그대로 child를 돌리지 않습니다. child들은 GPU를 공유하므로 하나씩 실행합니다.

### 수정이 실제로 컴파일되도록 fft.hip 수정

overlay의 `CMakeLists.txt`에서 각 HIP 오브젝트는 `DEPENDS`에 `${hip_src}`만 적은 `add_custom_command`로 빌드됩니다. 컴파일러의 헤더 의존성은 추적되지 않으므로, `lru_cache.h`를 바꿔도 warm build에서는 `hip_objs/fft.o`가 그대로 남습니다. PR 작성자가 바로 이것을 겪었습니다. 헤더 수정 후 첫 테스트 실행은 여전히 이전 동작을 보였고, `fft.hip`이 바뀐 뒤에야 통과했습니다. 따라서 `fft.hip`의 주석 수정은 기존 빌드 디렉터리 위에서 이 브랜치를 빌드하는 사람에게 실제로 필요한 변경입니다. `graph_cache_`를 생성하는 `Device`는 CMake가 헤더 의존성을 추적하는 일반 C++ 소스인 `device.cpp`에 있으므로, `LRUCache(size_t)` guard는 이 우회에 의존하지 않습니다.

## 4. 검증

PR 작성자 (gfx1151, ROCm 7.15):

- `cargo test --features rocm --test rocm_fft_cache_env`: 12개 케이스 모두 통과.
- 되돌림 검사: `lru_cache.h`를 main으로 되돌리면 거부 대상 8개가 모두 실패합니다. `0`과 `0x10`은 첫 GPU rfft가 `std::bad_alloc`으로, `abc`, 빈 문자열, overflow 값은 `stoul`로 실패하고, `-1`, `8abc`, `2147483648`은 경고를 출력하지 않습니다.
- `cargo test --features rocm --test rocm_fft_plan_cache`와 `--test rocm_gpu_faults`: 통과.
- `cargo clippy --features rocm --test rocm_fft_cache_env -- -D warnings`, `cargo fmt --check`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`: 통과.
- `-D_GLIBCXX_ASSERTIONS`를 켠 호스트 g++ 14 probe로 같은 값들과 ` 16`, `+16`(둘 다 16으로 수락)을 확인했습니다.

오케스트레이터 검증 (gfx1151, origin/main `2cee9cf4` 위의 브랜치):

- `make verify-rocm`이 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke 실행(32 토큰)이 통과했습니다. 이 PR이 `fft.hip`을 수정하므로 게이트의 warm build는 수정 사항을 컴파일했습니다.
- `verify-test-rocm`은 네 target에서 실패했고, 모두 이 PR이 원인이 아닙니다. 세 개는 알려진 기준 실패인 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, 그리고 #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`입니다. 네 번째는 이전부터 간헐적으로 실패하던 `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference`로, #2072가 추적합니다.
- 새 `tests/rocm_fft_cache_env.rs`는 전체 suite 안에서 통과했습니다.

## 5. 학습 포인트

- **헤더 의존성이 없는 custom build 명령은 헤더 수정을 보이지 않게 만듭니다.** overlay는 각 `.hip` 파일을 `add_custom_command(... DEPENDS ${hip_src})`로 컴파일하므로, CMake는 HIP 오브젝트가 include하는 헤더를 전혀 모릅니다. 헤더만 고친 수정은 문제없이 빌드되지만 이전 오브젝트가 링크되고, 첫 테스트 실행은 이전 동작을 보고해서 수정이 틀린 것처럼 보입니다. 이번에도 `lru_cache.h` 수정 후 첫 테스트 실행은 실패했고, `fft.hip`이 바뀌어 다시 컴파일된 뒤에야 통과했습니다. #2075가 해결되기 전까지는(예를 들어 `DEPFILE`과 `-MD`, 또는 `IMPLICIT_DEPENDS`로) overlay 헤더를 바꾸면 HIP 오브젝트를 clean build하거나 그 헤더를 include하는 모든 `.hip` 파일을 수정해야 하며, warm build에서 "고쳐졌다"는 결과는 다시 컴파일된 오브젝트에서 나왔는지 확인해야 합니다. `lru_cache.h` 하나만 해도 `device.h`를 통해 HIP 소스 37개에 포함됩니다.
- **static 초기화에서 나간 예외는 계속 반복됩니다.** C++는 예외 후 함수 내부 static의 초기화를 다시 시도하므로, 잘못된 환경 변수 하나가 프로세스가 끝날 때까지 모든 FFT의 실패로 이어졌습니다. 검증된 파싱에서 기본값을 반환하는 것이 경고를 한 번만 출력하게 만드는 방법입니다.
- **어떤 변수가 잘못됐는지 말해야 합니다.** `std::stoul`의 `what()`은 함수 이름입니다. 새 출력은 변수 이름, 원래 값, 대신 쓴 값을 알려 주므로 사용자가 설정을 고칠 수 있습니다.
- **upstream이 아니라 형제 코드의 규칙을 따릅니다.** upstream CUDA의 `atoi` 처리는 `0`에서 실패하고 `-1`을 받아들입니다. overlay의 watchdog 변수에 이미 더 나은 규칙이 있었고, 이를 재사용해 ROCm 변수들의 동작을 일관되게 유지했습니다.
- **수정을 되돌리고 테스트가 실패하는지 확인합니다.** 되돌림 실행이 거부 대상 8개 각각을 테스트가 실제로 잡아낸다는 증거입니다.

## 6. 주의 사항과 검증하지 않은 부분

- **Metal과 CUDA**는 실행하지 않았습니다. 변경은 ROCm 빌드만 복사하는 `patches-rocm/`과 문서, `#![cfg(feature = "rocm")]` 테스트에 한정됩니다.
- **`lru_cache.h`를 include하는 다른 HIP 오브젝트**(`device.h` 경유)는 warm build에서 다시 빌드되지 않습니다. 이들은 변수를 읽지도 `graph_cache_`를 생성하지도 않으므로 동작이 바뀌는 것은 없지만, 일부만 오래된 빌드는 직접 시험하지 않은 상태입니다. #2075가 이 틈을 없앱니다.
- **되돌림 증거는 저장소에 없습니다.** 8개 케이스 되돌림 실행과 호스트 probe는 수동으로 실행했습니다.
- **gfx1151만** 실행했습니다.
- **앞쪽 공백과 `+`는 수락됩니다.** `strtol`과 watchdog 변수를 따른 의도된 선택이지만, 문서는 값을 양의 정수로만 설명합니다.

## 7. 남은 작업

- #2075: HIP 오브젝트가 include하는 헤더에 의존하게 해서, overlay 헤더만 바뀐 경우에도 warm build에서 다시 컴파일되게 합니다.
- #1813: 다른 upstream 후보와 함께 항목 25를 fork에 제안.
- #2072: 게이트에서 본 `rocm_mxfp4_quant` 간헐 실패. 이 PR과 무관합니다.

참조: #2051 (이 PR로 닫힘), #1801, #1876, #1813, #2037, #2052, #2072, #2075, PR #2049.
