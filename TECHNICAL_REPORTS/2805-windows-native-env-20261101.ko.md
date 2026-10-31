# PR #2805: Windows에서 Rust가 설정한 환경 변수가 MLX의 getenv에 닿게 하기

**작성일**: 2026-11-01 (머지 전 작성)
**상태**: 리뷰 중
**언어**: Rust, C++, Python
**위험도**: 중간 (Windows에서 효력이 없던 설정이 적용되기 시작해 서빙 동작이 달라진다. 다른 플랫폼은 변화 없음)

## 요약

Windows에서는 `std::env::set_var`가 C 런타임의 `getenv`를 갱신하지 않아, mlxcel이 MLX용으로 쓰는 기본값이 모두 효력이 없었다. 새 `mlxcel_core::native_env` 헬퍼는 `_wputenv_s`도 함께 호출하고, 프로덕션의 모든 쓰기가 이 헬퍼를 지나며, CI 가드가 직접 쓰기를 거부한다. 설정이 이제 적용되므로 RTX Spark N1X에서 Gemma 4, Laguna, 오래 도는 서버의 동작이 달라지고, 그 측정은 소유자가 따로 한다.

## 1. 문제 정의

`std::env::set_var`는 `SetEnvironmentVariableW`를 호출한다. C 런타임은 narrow와 wide 테이블을 따로 가지며 `getenv`는 narrow를 읽는다. MLX의 `env::get_var`는 `std::getenv` 위의 `atoi`이므로, Rust에서 쓴 값은 Rust만 보고 네이티브 코드는 보지 못했다. #2799가 `MLX_PTX_CACHE_DIR`에서 이를 발견해 그 변수 하나만 오버레이에서 우회했다.

헬퍼 테스트에서 `_wputenv_s` 호출을 뺀 측정: 8개 중 5개가 실패한다. `the_cuda_cache_defaults_reach_mlx`에서는 `mlx::core::env::get_var("MLX_CUDA_GRAPH_CACHE_SIZE")`가 2000이 아니라 기본값을 돌려줬다.

### 감사 결과

| 변수 | 쓰는 곳 | 네이티브 읽는 곳 | 이전 Windows에서 효력 |
|---|---|---|---|
| `MLX_USE_CUDA_GRAPHS` | `src/loading/mod.rs`, `start_server`에서 호이스트 | `device.cpp` `use_cuda_graphs` | 없음: Gemma 4에서 캡처가 켜진 채 |
| `MLX_CUDA_GRAPH_CACHE_SIZE` | `hardware.rs` | `device.cpp`에서 `lru_cache.h` | 없음: 2000이 아니라 400 |
| `MLX_CUDA_SDPA_CACHE_SIZE` | `hardware.rs` | `scaled_dot_product_attention.cpp`에서 `lru_cache.h` | 없음: 2000이 아니라 256 |
| `MLX_MAX_OPS_PER_BUFFER` | `cuda_graph_budget.rs`; `hardware.rs` (Metal 전용) | `get_graph_limits`에서 `env::max_ops_per_buffer` | 없음: 12.1의 Laguna에서 100이 아니라 20 |
| `MLX_MAX_MB_PER_BUFFER` | `cuda_graph_budget.rs` | `get_graph_limits`에서 `env::max_mb_per_buffer` | 없음: 12.1의 Laguna에서 1000이 아니라 25 |
| `MLX_PTX_CACHE_DIR` | `ensure_persistent_ptx_cache` | `ptx_cache_dir_win.h`, `GetEnvironmentVariableW` | 있음, #2799부터 |
| `MLXCEL_QMM_TILE_M` | `cuda_kernel_knobs.rs` (`mlxcel tune`, 시작 시) | `qmm_sm80.cu`, 호출마다 | 없음: 튜너가 같은 실행끼리 비교함 |
| `MLXCEL_QMV_MULTIROW_MAX_ROWS` | `cuda_kernel_knobs.rs` (시작 시) | `qmv.cu`, 한 번 읽고 고정 | 없음 |

`MLXCEL_KV_BOUNDARY_V_LAYERS`와 `MLXCEL_PREFILL_CHUNK`는 Rust만 읽으므로 원래 동작했다. 가드의 규칙을 하나로 두려고 이 둘도 헬퍼를 지나게 했다.

`MLX_ENABLE_TF32`는 테스트 전용 쓰기, 즉 `#[cfg(test)]` ctor뿐이다. `src/bin/mlx_server.rs:3019`와 `:3043`은 `#[cfg(test)] mod tests` 안의 `ScopedEnv`다. 둘 다 프로덕션이 아니다.

## 2. 기술적 결정

### 2.1 C 런타임은 하나, 측정으로 확인

헬퍼는 Rust의 `_wputenv_s`가 MLX가 읽는 런타임에 닿아야 동작한다. `dumpbin /directives`로 `utils.cpp.obj`, nvcc의 `sort.cu.obj`, `device.cpp.obj`, `mlx_cxx_bridge.lib`의 모든 멤버에서 `RuntimeLibrary=MD_DynamicRelease`와 `/DEFAULTLIB:msvcrt`를 확인했다. Rust 타깃에는 `crt-static`이 없고, 서버는 `getenv`를 `api-ms-win-crt-environment-l1-1-0.dll`에서 가져온다. 테스트는 브리지의 `std::getenv`와 `mlx.lib` 안의 `env::get_var`로 읽으므로, 런타임이 갈라지는 빌드에서는 실패한다.

### 2.2 `_wputenv_s` 다음에 `std::env`

`_wputenv_s`는 두 테이블과 OS 블록을 모두 갱신한다. 빈 값은 세 곳 모두에서 변수를 지우므로, 빈 값이 빈 문자열로 남는 `std::env`의 계약을 지키려고 `std::env`를 마지막에 호출한다. C 런타임에는 빈 값이 없어 `getenv`는 NULL을 돌려주며, 오버레이의 문서화된 규칙은 이미 빈 값을 미설정으로 본다.

### 2.3 비 ASCII 값

ACP 949에서 임시 프로브로 확인: 한글은 `getenv`로 그대로 돌아오고, `é😀`는 `_wputenv_s`가 0을 돌려준 채 `e??`로 돌아온다. `std::env`와 `GetEnvironmentVariableW`는 정확하다. 감사한 값 중 이런 문자를 담을 수 있는 것은 `MLX_PTX_CACHE_DIR`뿐이고, 이 때문에 #2799의 wide 읽기를 유지한다. 나머지는 정수다.

### 2.4 `_wputenv_s`는 긴 값에서 프로세스를 끝낸다

같은 프로브에서 32766자 값은 받아들이고, 32767자 값은 오류를 돌려주지 않고 프로세스를 끝냈다. 헬퍼는 길이를 먼저 확인해 한 줄을 출력하고 그 값은 `std::env`에 맡긴다. 프로덕션 값이 이 길이에 가까울 일은 없지만, 입력 때문에 프로세스를 끝낼 수 있는 헬퍼를 그대로 둘 이유도 없다.

### 2.5 가드는 프로덕션의 모든 쓰기를 금지

네이티브 코드가 어떤 이름을 읽는지는 쓰는 곳에서 보이지 않고, 나중에 읽는 곳이 추가되어도 쓰는 곳을 고칠 필요가 없어야 한다. 그래서 `check_native_env_writes.py`는 `native_env.rs` 밖의 `std::env` 쓰기를 허용하지 않고, `#[cfg(test)]` 항목과 `#[cfg(test)] mod`로 선언된 파일만 예외로 한다. 파일 이름이 아니라 선언으로 판단한다. 경로 필터가 없는 기존 `env var docs` 잡에서 돈다.

### 2.6 스레드

헬퍼는 `set_var`의 계약을 그대로 두고, 각 쓰는 곳의 SAFETY 주석도 유지한다. 쓰는 곳별 확인:

- `hardware.rs`, `cuda_graph_budget.rs`, `apply_env_once`, `ensure_persistent_ptx_cache`, `TurboKvCacheArgs`, `bench_engine`: 런타임과 MLX 연산 이전에 메인 스레드에서 쓴다.
- `cuda_kernel_knobs.rs` `QmmTileOp::run`: 시작 시 쓰기가 아니다. `mlxcel tune`에서 후보 사이에 값을 덮어쓰며, Windows에서는 `_wputenv_s`가 이전 문자열을 해제한다. `synchronize_default`가 먼저 스트림을 비우고 튜너는 오프라인이므로 그 문자열을 잡고 있는 네이티브 코드가 없다. 경쟁 없음.
- `loading/mod.rs`: 호이스트가 메인 스레드에서 쓰고, 이후 로드 지점의 호출은 `var_os` 검사에서 돌아간다. 워커의 로드가 쓰기에 닿는 경우는 시작 시 `config.json`을 읽지 못한 것처럼 호이스트가 아무것도 못 본 때뿐이다. 다른 스레드가 살아 있을 때의 첫 쓰기이고, `qmm_sm80.cu`는 호출마다 `getenv`를 부른다. Linux와 같은 위험이며 이 PR 이전부터 있었고, 그때쯤 MLX가 이미 스위치를 고정했을 수 있다. 첫 쓰기는 문자열을 해제하지 않는다.

## 3. 변경 요약

- `native_env.rs`: `set_native_var`, `remove_native_var`, `native_getenv`, `mlx_env_int`, `log_effective_native_env`. 브리지에 `native_env_read`와 `mlx_env_get_var_int` 추가.
- 쓰는 곳: `hardware.rs`, `cuda_graph_budget.rs`, `cuda_kernel_knobs.rs`, `ensure_persistent_ptx_cache`, `loading/mod.rs`, `turbo_args.rs`, `bench_engine.rs`.
- `RUST_LOG=mlxcel_core::native_env=debug`가 `main`, `start_server`, Gemma 4 쓰기에서 감사 대상 변수를 `getenv`가 돌려주는 값으로 출력한다.
- CI: `env var docs`에 가드와 그 테스트, `cpu-link`가 Linux에서 `native_env` 실행.
- 문서: `docs/environment-variables.md`, `ptx_cache_dir_win.h`의 `windows_env_value` 주석.

## 4. 결과

| 확인 | 위치 | 결과 |
|---|---|---|
| `native_env` 테스트, `_wputenv_s` 호출 제거 | N1X | 8개 중 5개 실패 |
| `native_env` 테스트, 이 PR | N1X | 8개 모두 통과 |
| `cuda_ptx_cache_trust_windows` (#2799) | N1X | 22개 모두 통과 |
| `cuda_ptx_cache_win_test.sh` | N1X | 통과. 심볼릭 링크 케이스는 #2799처럼 건너뜀 |
| 이 트리에 대한 가드와 그 단위 테스트 | N1X | 통과, 8개 |
| `cargo clippy -p mlxcel-core --features cuda --lib --tests` | N1X | 아래 `weights.rs`의 두 lint를 빼면 깨끗함 |
| `mlxcel generate`와 `mlxcel-server`, 설정 파일만 있는 디렉터리 | N1X | 아래 표 |
| `mlxcel tune --op qmm-tile --op qmv-multirow`, 임시 캐시 | N1X | tactic 2개 기록 |
| 그 tactic으로 `MLXCEL_AUTOTUNE=cache`, 두 바이너리 | N1X | `getenv`로 `MLXCEL_QMM_TILE_M=128`, `MLXCEL_QMV_MULTIROW_MAX_ROWS=8` |
| 운영자가 `MLX_USE_CUDA_GRAPHS=1`, `MLX_CUDA_GRAPH_CACHE_SIZE=400` 설정 | N1X | 유지됨 |

두 바이너리에서 `getenv`가 돌려준 값:

| 디렉터리 | `MLX_USE_CUDA_GRAPHS` | 그래프 캐시 | SDPA 캐시 | Ops | MB | PTX 디렉터리 |
|---|---|---|---|---|---|---|
| `llama` | 미설정 | 2000 | 2000 | 미설정 | 미설정 | `~\.cache\mlxcel\cuda-ptx\7ca40033132d` |
| `gemma4_text` | 0 | 2000 | 2000 | 미설정 | 미설정 | 동일 |
| `laguna`, 전문가 256 | 미설정 | 2000 | 2000 | 100 | 1000 | 동일 |

Laguna의 값은 이 장치가 compute capability 12.1임도 확인해 준다. 패밀리 외의 유일한 조건이다.

### 이 호스트에서 달라지는 것

| 설정 | 모델 | 이전, 이후 | GB10 측정 (`cuda-graph-budget-gb10-2026-09-12.md`) |
|---|---|---|---|
| 그래프 캡처 끔 | Gemma 4 계열 | 켜짐, 꺼짐 | `gemma-4-26b-a4b-it-4bit`: 디코드 -10.8%, 프리필 164에서 473 tok/s |
| 그래프 예산 | 전문가가 있는 Laguna | 20 ops와 25 "MB", 100과 1000 | `laguna-xs-2.1-nvfp4`: 디코드 +17.4%, 피크 +0.37 GB |
| 그래프 캐시 용량 | 전체 | 400, 2000 | 중단 임계값이 누적 미스 800에서 4000으로. 메모리는 형상이 쌓일 때만 쓴다 |
| SDPA 캐시 용량 | 전체 | 256, 2000 | 중단 임계값 512에서 4000으로. 가득 찬 캐시의 메모리는 측정된 적 없음 |

Autotune은 `MLXCEL_AUTOTUNE` 없이는 동작하지 않고, 이 호스트에는 tactic 캐시가 없었다. 운영자 값은 원래 MLX에 닿았으므로, 새 바이너리에서 이전 설정을 재현할 수 있다: `MLX_USE_CUDA_GRAPHS=1 MLX_CUDA_GRAPH_CACHE_SIZE=400 MLX_CUDA_SDPA_CACHE_SIZE=256 MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=25`.

## 5. 트레이드오프와 남은 문제

- **서빙 A/B는 여기서 돌리지 않음:** 소유자가 N1X에서 수정 전 서버와 비교한다. 위 표의 수치는 N1X가 아니라 GB10의 것이다.
- **읽기 확인은 `env::get_var` 수준이고 LRU까지는 아님:** 캐시를 400개 넘게 키운 실행은 없고, 가중치를 로드한 실행도 없다. `lru_cache.h`가 읽는 함수는 같은 함수다.
- **동시 `getenv` 아래 `_wputenv_s` 첫 쓰기의 안전성**은 UCRT의 환경 락에 기댄 판단이며, 스트레스 테스트로 확인한 것은 아니다.
- **Windows CI는 `native_env` 테스트를 돌리지 않음:** 호스티드 Windows 잡은 이 크레이트들에 대해 컴파일만 한다. 가드는 어디서나 돌고, Linux는 `cpu-link`로 테스트를 돌린다.
- **`MLX_ENABLE_TF32` 테스트 고정은 Windows에서 효력 없음:** ctor가 일반 `set_var`를 쓰므로 Windows 테스트 실행에서 MLX의 TF32 기본값이 켜진 채다. 이 PR 범위 밖이고 `tf32_pin_tests`는 돌리지 않았다.
- **`src/lib/mlxcel-core/src/weights.rs:1051`과 `:1061`:** 테스트의 `#[cfg(not(unix))]` 조기 반환 때문에 생긴 미사용 변수와 도달 불가 코드가 Windows의 테스트 타깃에서 `-D warnings`에 걸린다. 이 PR 이전부터 있었고 건드리지 않았다.

## 6. 배운 점

- 이슈는 프로덕션처럼 보이는 `mlx_server.rs`의 쓰기를 가리켰지만 테스트 모듈 안에 있었다. 감싸는 `#[cfg(test)]`를 확인하자 한 번에 정리됐다.
- 예산을 올렸다고 알리는 시작 로그는 읽는 쪽이 그 값을 봤다는 증거가 아니다. Windows에서 그 줄은 Rust의 관점에서만 참이었고, 그래서 새 디버그 줄은 `getenv`를 거쳐 읽는다.
- `_wputenv_s`는 긴 값을 잘못된 인자 경로로 보내 프로세스를 끝내므로, 문서의 errno 반환은 그 입력을 다루지 않는다.
