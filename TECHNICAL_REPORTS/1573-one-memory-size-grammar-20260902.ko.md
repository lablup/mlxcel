# 기술 보고서: PR #1573 - chore(execution): 메모리 크기를 하나의 문법으로 파싱

**작성일**: 2026-09-02
**작성자**: mlxcel maintainers
**리뷰어**: 구현 및 단위 테스트 사이클
**상태**: 완료 (단위 테스트로 검증. 실제 바이너리 preflight 수용 실행은 머지 오케스트레이터에 인계, 4.2절 참조)
**언어**: Rust, Markdown
**위험도**: Medium (공유 파서가 `MLXCEL_WIRED_LIMIT`과 `MLXCEL_CACHE_LIMIT`도 관장하므로, 결함이 보고된 추정 경로뿐 아니라 모든 실행의 allocator cap 경로 위에 있다)

---

## 요약

`MLXCEL_MEMORY_LIMIT`을 읽는 곳이 두 군데였고 두 곳의 문법이 서로 달랐다. `src/execution/runtime.rs`의 `parse_memory_size`는 MLX allocator cap을 설정하면서 `4G`, `4GB`, `512M`, `512MB`, 그리고 바이트 값을 받았다. 반면 `src/execution/memory_estimate.rs`의 메모리 추정 preflight는 `parse_optional_memory_size_bytes`와 `parse_scaled_memory_size`라는 자기 파서를 따로 들고 있었고 `GB`와 `MB`만 받았다. 그래서 `MLXCEL_MEMORY_LIMIT=4G`는 allocator를 4 GiB로 묶으면서도 `mlxcel inspect`와 `--estimate-memory`에서는 그냥 버려졌고, preflight는 머신 전체 통합 메모리를 기준으로 가용량을 보고했다. 128 GB 머신이라면 프로세스가 실제로 할당받을 수 있는 양을 32배 부풀려 말한 셈이다. 이 PR은 `parse_memory_size`를 유일한 문법으로 만들어 preflight에 노출하고, `K`/`KB`를 추가하고, 허용되는 모든 표기를 양쪽 테스트로 고정한다. 결함의 실제 원인은 '파서가 둘'보다 좁았다. 두 문법은 어느 테스트든 실제로 써 본 표기에서는 전부 일치했고, 양쪽 어느 테스트도 써 본 적 없는 짧은 접미사에서만 갈라졌다.

---

## 1. 문제 정의

### 1.1 배경: 변수 하나, 독자 둘, 그리고 늦게 도는 쪽은 하나뿐

`MLXCEL_MEMORY_LIMIT`(이슈 #55)은 MLX allocator에 거는 soft cap이다. `initialize_runtime()`이 `resolve_memory_limit()`을 거쳐 이 값을 읽고 `mlxcel_core::memory::set_memory_limit(...)`을 호출하므로, working set이 cap을 넘어서려는 순간 MLX가 스래싱 대신 예외를 던진다.

메모리 추정 preflight(이슈 #56)도 '이 프로세스가 실제로 얼마나 쓸 수 있는가'라는 같은 질문에 답해야 하는데, 정작 allocator cap이 아직 적용되지 않은 경로에서 돈다. `memory_estimate.rs`의 `resolve_available_memory`는 이를 위해 5단계 우선순위를 문서화해 두었고, 첫 단계가 `mlxcel_core::memory::memory_limit()`이 아니라 환경 변수인 것은 의도적이다.

```
1. MLXCEL_MEMORY_LIMIT이 0이 아닌 값으로 설정된 경우  <- 추정 전용 명령을 잡아낸다
2. mlxcel_core::memory::memory_limit()이 0이 아닌 경우 <- 이미 적용된 MLX cap
3. HardwareCapabilities::unified_memory_gb << 30       <- 머신 수치
4. /proc/meminfo의 MemAvailable, 이어서 MemTotal (Linux)
5. 0
```

1단계가 존재하는 이유는 `mlxcel inspect`와 `serve --estimate-memory`가 런타임 bring-up 이전에 추정하기 때문이다. 그 시점에는 2단계가 0을 읽는다. 설계 자체는 옳았다. 결함은 1단계가 그 변수를 두 번째 파서로 읽었다는 데 있다.

### 1.2 갈라짐, 그리고 사용자가 본 것

| 입력 | 런타임 allocator cap (수정 전) | preflight 가용량 (수정 전) |
|---|---|---|
| `4GB` | 4 GiB | 4 GiB |
| `4G` | 4 GiB | **머신 전체** |
| `512MB` | 512 MiB | 512 MiB |
| `512M` | 512 MiB | **머신 전체** |
| `4096` | 4096 바이트 | 4096 바이트 |

양쪽 다 조용히 실패한다. 경고가 없고, `hw.unified_memory_gb`로 떨어지는 경로 자체가 설정 없이 도는 모든 실행이 밟는 정상 분기라서 preflight 출력이 전혀 이상해 보이지 않는다. 4 GiB로 묶어 놓고 `inspect`에 '이 모델이 들어가느냐'를 물은 사용자는 128 GB를 근거로 한 '들어간다'를 받아 들고, 로드 시점에 cap을 만났다.

### 1.3 왜 어떤 테스트도 잡지 못했나

여기가 앞으로 가져갈 대목이다. 두 문법은 `GB`, `MB`, 바이트 값에서 일치했고 오직 맨 `G`와 맨 `M` 접미사에서만 갈라졌다. 그런데 존재하던 모든 테스트가 일치하는 쪽 표기만 썼다.

- `runtime_tests.rs`는 `"64GB"`, `"128gb"`, `"512MB"`, `"1073741824"`, `"1.5GB"`, `"abc"`를 덮었다. 자기 파서가 받아들이던 맨 `G`와 맨 `M`은 어디에서도 단언되지 않았다.
- `memory_estimate.rs`는 `"512MB"`를 `estimate_total_memory`로 관통시켜 확인했고, 파서 수준에서 `"0"`, `"none"`, `"-1GB"`, `"NaNGB"`, `"1.5GB"`를 확인했다.

문서도 같은 방향으로 흘렀고 심지어 자기들끼리 어긋나 있었다. `docs/environment-variables.md`의 세 행은 같은 파서를 쓰는 세 변수를 설명하는데 그중 둘이 서로 다른 문법을 적어 두었다. `MLXCEL_WIRED_LIMIT`과 `MLXCEL_MEMORY_LIMIT`은 `bytes, NGB, NMB`로 적혀 있었고, 네 줄 아래에서 같은 함수를 읽는 `MLXCEL_CACHE_LIMIT`은 `bytes, NG/NGB, NM/NMB`로 적혀 있었다. `mlxcel --help` 블록은 앞의 두 변수에 대해 `supports GB, MB, or bytes`라고 했다. `MLXCEL_MEMORY_LIMIT` 문서를 따라 읽은 사람이라면 애초에 `4G`를 쓸 일이 없고, `MLXCEL_CACHE_LIMIT` 행을 읽었거나 파서를 직접 읽은 사람이라면 쓴다.

구조적 결함은 파서가 둘이라는 점이다. 다만 그 구조적 결함이 이렇게 오래 틀린 숫자를 만들어낼 수 있었던 것은, 어떤 테스트도 단언하지 않고 어떤 문서도 일관되게 적지 않은 표기가 받아들여지고 있었기 때문이다. 당연히 덮여 있으리라 짐작하기 쉬운 표기까지 포함해 모든 표기를 고정하는 것이 첫날에 실패했을 검사다.

### 1.4 위험 평가

| 위험 | 영향 | 발생 가능성 |
|------|------|------------|
| preflight가 머신 RAM 대 요청 cap의 비율만큼 가용량을 부풀리고, cap에 걸릴 모델에 `fits = true`를 보고 | Medium (크래시가 아니라 잘못된 진행 판단) | `4G` 형태를 쓰는 모든 실행에서 확실 |
| 크기 값 변수의 세 번째 독자가 세 번째 문법을 들고 추가됨 | Low (이제 `pub(crate)` 소유자가 하나) | 정리 이전에는 중간 정도. 이미 한 번 재현된 패턴이었다 |
| 파서 통합이 기존 배포의 `MLXCEL_WIRED_LIMIT`이나 `MLXCEL_CACHE_LIMIT`을 조용히 옮김 | 일어난다면 High (모든 실행의 allocator cap) | Low. 2.2절의 감사로 범위가 한정된다 |

---

## 2. 기술 리뷰

### 2.1 영향 범위

`parse_memory_size`는 `MLXCEL_MEMORY_LIMIT` 전용 파서가 아니다. `mlxcel_core::set_wired_limit`을 구동하고 기본값이 `gpu_max_memory_size()`인 `MLXCEL_WIRED_LIMIT`, 그리고 CUDA 버퍼 캐시 상한인 `MLXCEL_CACHE_LIMIT`(이슈 #627)의 파서이기도 하다. 문법이든 반환 타입이든 건드리면 두 바이너리의 모든 실행에서 allocator cap이 움직인다. 그래서 기존 테스트를 회귀 고정용으로 그대로 남기고, 추상적으로 추론하는 대신 표기 하나하나를 감사했다.

### 2.2 동작 감사

받아들여지던 모든 입력이 값을 유지한다. 결과가 달라지는 경우를 남김없이 적으면 다음과 같다.

| 입력 부류 | 런타임 (수정 전) | 런타임 (수정 후) | preflight (수정 전) | preflight (수정 후) |
|---|---|---|---|---|
| `4GB`, `4gb`, ` 4 GB `, `1.5GB`, `4.1GB` | 4 GiB / floor | 동일 | 동일 | 동일 |
| `4G`, `512M` | 허용 | 동일 | **무시** | **허용, 같은 값** |
| `8K`, `8KB` | 거부 | **허용** | 거부 | **허용** |
| `1024` (접미사 없음) | 1024 바이트 | 동일 | 동일 | 동일 |
| `1.5` (접미사 없음) | 거부 | 거부 | 거부 | 거부 |
| `abc`, `GB` | `None` | `None` | `None` | `None` |
| `-1GB`, `NaNGB`, `infGB` | `Some(0)` | **`None`** | `None` | `None` |
| `1e30GB` | `usize::MAX` | `u64::MAX` | `u64::MAX` | `u64::MAX` |
| `0GB` | `Some(0)` | `Some(0)` | `None` | `None` (resolver의 filter 경유) |

두 행에는 설명이 필요하다.

**`-1GB` 행이 유일한 실제 동작 변경이고, 그것도 변수 하나에만 해당한다.** Rust의 float에서 int로 가는 `as` 캐스트는 포화 연산이라 `(-1.0 * 2^30) as usize`는 랩어라운드가 아니라 `0`이었다. `resolve_memory_limit`과 `resolve_cache_limit`은 이미 0을 `None`으로 매핑하고 있었으니 최종 상태가 같다. `resolve_wired_limit`은 그렇지 않았다. `if limit > 0 { ... } else { None }`으로 떨어지므로 쓰레기 값인 `MLXCEL_WIRED_LIMIT=-1GB`가 wired limit을 조용히 *해제*했고, 정작 `MLXCEL_WIRED_LIMIT=abc`는 `gpu_max_memory_size()`로 폴백했다. 이제 둘 다 기본값으로 폴백한다. 회귀가 아니라 일관성 복구이지만, 배포 환경이 알아챌 수 있는 유일한 한 줄이다.

**`1e30GB` 행은 변경이 아니다.** 두 옛 파서 모두 이미 포화 처리하고 있었다. 런타임 쪽은 포화 캐스트를 통해 암묵적으로, preflight 쪽은 `bytes.min(u64::MAX as f64)`로 명시적으로. 새 코드는 이를 `if bytes >= u64::MAX as f64 { return Some(u64::MAX); }`로 적고 테스트로 고정해서, 읽는 사람이 언어 보장을 알고 있어야만 안심할 수 있는 상태를 없앤다. 이걸 '랩어라운드 수정'이라고 쓰면 틀린 서술이 된다.

### 2.3 타입

파서는 `u64`를 반환한다. `mlxcel_core::memory::set_memory_limit` / `set_cache_limit` FFI가 받는 타입이고 preflight가 계산에 쓰는 타입이다. wired limit 경로와 `RuntimeSetup`의 `Option<usize>` 세 필드는 여전히 `usize`를 쓰므로 좁히는 일은 경계 한 곳, `clamp_to_usize`에서만 일어난다. `usize::try_from(bytes).unwrap_or(usize::MAX)`는 mlxcel이 빌드하는 모든 64비트 타깃에서 무손실이다. 포화를 넣은 것은 가상의 32비트 빌드에서 큰 cap이 작은 값으로 잘리는 대신 clamp되게 하기 위해서다.

### 2.4 코드 품질

`parse_scaled_memory_size`는 삭제되고 `parse_optional_memory_size_bytes`는 unset 검사와 위임만 남는다. preflight는 이제 크기 산술을 하나도 소유하지 않는다. `0` / `none` / 빈 문자열 처리는 원래 자리인 각 resolver에 남겼다. 세 resolver가 이 부분에서 서로 다르기 때문이다. `resolve_wired_limit`은 추가로 `max`와 `""`를 `gpu_max_memory_size()`로 매핑하는데, 이는 문법 규칙이 아니라 resolver의 정책이다.

---

## 3. 기술적 선택과 그 이유

### 3.1 새 공유 모듈이 아니라 `runtime.rs`에 파서 하나

**상황**: 어느 쪽도 소유하지 않는 `src/execution/size_grammar.rs`를 새로 만드는 것이 자연스러운 직관이다.

**선택**: `runtime.rs`에 `pub(crate)`로 둔다.

**이유**: `runtime.rs`는 이미 세 환경 변수 상수와 세 resolver를 전부 소유한다. 파서만 빼내면 상수와 정책은 이 파일에, 문법은 저 파일에 남는다. 그러면 '여기서 `4G`가 무슨 뜻인가'라는, 정확히 이번 결함을 낳은 종류의 질문을 던질 때 봐야 할 곳이 두 군데가 된다. 새 모듈은 소유자가 셋이 될 때 값을 한다. 오늘 preflight는 공동 소유자가 아니라 소비자다. `pub(crate)`인 이유를 문서 주석에 명시해 두어서, 나중에 읽는 사람이 왜 가시성이 모듈보다 넓은지 알 수 있게 했다.

### 3.2 파서는 `u64`, 경계에서 `usize`

**상황**: 파서를 `usize`로 두고 preflight가 넓히게 할 수도 있었다.

**선택**: `u64` 반환, 호출 지점 세 곳에서 `clamp_to_usize`로 좁힌다.

**이유**: 바이트 수는 포인터 폭이 아니라 고정 폭 수량이다. FFI setter와 `memory_estimate` 모듈 전체가 이미 `u64`를 쓴다. `usize`가 등장하는 이유는 `set_wired_limit`과 `RuntimeSetup`이 그보다 먼저 있었기 때문일 뿐이다. 한 곳에서 넓히고 세 곳에서 좁히는 방향이 소비자가 실제로 쓰는 타입 안에 산술을 남기고, 손실이 일어나는 유일한 연산을 암묵적 `as`가 아니라 이름과 주석이 붙은 자리에 놓는다.

### 3.3 접미사 없는 바이트 값은 정수만

**상황**: 접미사 분기가 `f64`를 파싱하므로 대칭성만 보면 `1.5`를 1바이트로 받을 수도 있었다.

**선택**: 접미사 없는 분기는 `parse::<u64>()`를 유지하고 `1.5`를 거부한다.

**이유**: 소수점 바이트는 아무도 의도해서 쓰지 않는 값이다. 이를 받아들이면 `1.5G`를 쓰려다 낸 오타 `1.5`가 모든 할당을 실패시키는 1바이트 cap이 된다. 거부하면 `None`이 되어 멀쩡한 기본값으로 떨어진다. 옛 동작도 그대로 보존된다. 옛 분기가 `parse::<usize>()`였다.

### 3.4 `K`/`KB`를 지금 추가

**상황**: 이슈가 요구한 것은 기존 두 문법이 일치하는 것까지였다.

**선택**: 같은 변경에서 킬로바이트 접미사를 추가한다.

**이유**: 이 변경의 요지는 배워야 할 문법이 하나라는 것이다. `K`에 구멍을 남기면 '어떤 접미사가 되느냐'의 답이 여전히 '코드를 봐라'가 되고, 그건 이 변경이 끝내려는 상태 그 자체다. 회귀 가능성도 없다. `K` 접미사 값은 두 옛 파서에서 모두 `None`이었다.

### 3.5 `resolve_paged_slab_blocks`는 건드리지 않음

`memory_estimate.rs`에는 환경 변수 기반의 크기성 resolver가 하나 더 있다. 이슈 #1137이 소유하고 있고 메인테이너 판단을 기다리는 중이라, 같은 파일에 있어도 의도적으로 이 diff 밖에 둔다.

---

## 4. 검증

### 4.1 실행한 것

| 명령 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --lib execution::` | 126 passed, 0 failed |
| `cargo clippy --profile test-fast --lib --tests --features metal,accelerate -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `python3 scripts/ci/check_cross_repo_refs.py` | 통과 |

126개 중 13개가 파서와 preflight 테스트다.

- `parse_memory_size_gb`, `_mb`, `_bytes`, `_fractional_gb`, `_invalid`: 기존 테스트. `usize`를 `u64`로 바꾼 리터럴 하나를 빼면 그대로 두어 옛 표기의 회귀 고정 역할을 맡긴다.
- `parse_memory_size_accepts_every_suffix_spelling`: `4G` == `4GB` == `4gb` == `" 4 GB "`, `512M` == `512MB`, `8K` == `8KB` == 8192, 그리고 맨 `1024`.
- `parse_memory_size_fractional_is_exact_floor`: `1.5GB` = 1610612736, `4.1GB` = 4402341478, `0.5M` = 524288.
- `parse_memory_size_rejects_garbage`: `-1GB`, `NaNGB`, `infGB`, `abc`, `GB`, 맨 `1.5`가 모두 `None`. `0`은 `Some(0)`.
- `parse_memory_size_saturates_instead_of_wrapping`: `1e30GB`가 `u64::MAX`.
- `available_memory_honors_short_suffix_env_limit`: `MLXCEL_MEMORY_LIMIT=512M`을 `estimate_total_memory`로 관통시켜 `512MB` 형제 테스트와 같은 512 MiB를 단언한다. 보고된 결함의 end-to-end 테스트이고 `main`에서는 실패한다.
- `parse_optional_memory_size_accepts_the_runtime_grammar`: `4G` == `4GB` == `4gb`, `512M` == `512MB`, `8K`, 맨 `1024`, 그리고 `0GB`는 여전히 unset.

### 4.2 실행하지 않은 것과 그 이유

실제 바이너리 수용 실행에는 이 작업 단위가 의도적으로 수행하지 않은 `cargo build --release`가 필요하다. 그래서 머지 오케스트레이터에 넘기고 PR 본문에 적어 두었다.

```
MLXCEL_MEMORY_LIMIT=4G    ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
MLXCEL_MEMORY_LIMIT=4GB   ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
MLXCEL_MEMORY_LIMIT=4096M ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
                          ./target/release/mlxcel inspect -m models/mlx/qwen3-0.6b-4bit | grep Available:
```

앞의 셋은 동일한 4.00 GB 수치를, 넷째는 머신 수치를 출력해야 한다. `main`에서는 `4G`와 `4096M` 호출이 머신 수치를 출력하고, 그것이 결함이다.

`mlxcel inspect`는 모델을 위치 인자가 아니라 `-m/--model`로 받는다는 점을 적어 둔다. 이슈 본문의 `mlxcel inspect models/qwen3-0.6b-4bit`은 애초에 실행되지 않았을 것이고, 거기 적힌 체크포인트는 이 트리에서 `models/mlx/qwen3-0.6b-4bit`에 있다.

---

## 5. 학습 포인트

**환경 변수는 상수 이름이 아니라 소비자를 기준으로 grep한다.** 여기서 두 독자는 같은 문자열 리터럴 `"MLXCEL_MEMORY_LIMIT"`을 각각 선언한 상수(`runtime.rs:32`, `memory_estimate.rs:135`)를 통해 참조했다. 상수 이름으로 찾으면 파일 하나가 나오고, 변수 표기 자체로 찾으면 둘 다 나온다. 프로젝트 메모리에 이미 들어 있는 `rope_scaling` 부류와 같은 모양이다. 한 곳에서 파싱하고 다른 곳에서 소비하는 설정 키는 *소비자*를 검색해야만 간극이 드러난다.

**문법은 가장 덜 쓰이는 표기만큼만 테스트되어 있다.** 이번 갈라짐은 정확히 어떤 테스트도 단언하지 않은 표기에 자리 잡았고, 문서도 같은 모양으로 흘러 한 표의 인접한 행들이 같은 함수를 두 가지 문법으로 서술하는 지경이 되어 있었다. 파서가 N개 표기를 받으면 단언 N개는 철저함이 아니라 최저선이다. 테스트되지 않은 표기는 호출자마다 다른 뜻이 되어도 막을 것이 없고, 문서가 추적을 멈추는 것도 바로 그 표기들이다.

**정당한 기본값을 겸하는 폴백은 자기 실패를 가린다.** preflight가 `hw.unified_memory_gb`로 떨어지는 것은 설정 없는 실행에서는 올바른 동작이고, 그래서 무시된 `MLXCEL_MEMORY_LIMIT`이 완전히 정상으로 보이는 출력을 만들었다. 뒤쪽 단계가 흔한 경우인 우선순위 체인은 *앞쪽* 단계를 고정하는 테스트가 필요하다. 앞쪽 단계가 발화하지 않았을 때의 증상이 평범한 경로와 구분되지 않기 때문이다.

**2의 거듭제곱 곱셈은 이진 부동소수점에서 정확하다.** 두 파서 모두 1024^n으로 스케일하므로 `4.1GB` 같은 소수 값도 floor 이전에 잃는 것이 없고 모든 호출자에서 같은 4402341478로 떨어진다. 여기에는 고칠 정밀도 문제가 없었고, 십진 인식 파싱을 넣었다면 지금 두 경로에서 동일한 값들이 오히려 달라졌을 것이다.

**이슈 본문의 줄 번호보다 트리를 믿는다.** 이슈는 `runtime.rs:217-232`를 지목했지만 파서는 `:254-269`에 있었다. 옮겨진 `docs/environment-variables.md` 줄을 인용했고, 수용 명령의 인자 형태와 체크포인트 경로도 틀렸다. 이슈의 *분석*은 세부까지 정확했다. 좌표만 밀려 있었을 뿐이다.

---

## 6. 후속 작업

- 4.2절의 실제 바이너리 수용 실행이 남아 있고, 이 PR을 머지하는 쪽의 몫이다.
- `memory_estimate.rs`의 `resolve_paged_slab_blocks`는 자기 관례를 가진 별도의 환경 변수 기반 sizing resolver로 남아 있고 이슈 #1137이 소유한다.
- `resolve_wired_limit`의 unset 매칭은 여전히 정확한 문자열 비교(`Some("0") | Some("none") | Some("NONE")`)라서, 앞뒤 공백이 붙은 `" none "`은 limit을 해제하지 않고 기본값으로 떨어진다. preflight의 trim + 대소문자 무시 검사와 어긋나는 이 비대칭은 이번 변경 이전부터 있었고, diff를 한 가지 관심사로 유지하려고 손대지 않았다.
