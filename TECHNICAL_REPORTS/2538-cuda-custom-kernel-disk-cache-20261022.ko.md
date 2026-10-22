# PR #2538: 커스텀 CUDA 커널을 디스크에 캐시해 프로세스마다 반복되던 컴파일 제거

**날짜**: 2026-10-22
**상태**: 완료 (PR 열림; #2492 종료)

## 요약

이슈 #2492는 GB10(sm_121)에서 granite-4.0-h-tiny-4bit 디코드가 v0.7.0의 약 77 tok/s에서 약 57 tok/s로 떨어졌다고 보고했고, bisect 결과는 granite의 단일 토큰 MoE를 fused 디코드 커널로 보낸 #2283이었다. 이슈는 디바이스별 라우팅 기본값을 요구했다. PR #2538은 GB10에서 이 커널이 토큰당으로는 느리지 않다는 것을 보인다. MLX는 모든 `mx.fast.cuda_kernel`을 `use_disk_cache = false`로 빌드하므로, 프로세스마다 첫 디코드 스텝 안에서 fused gate-up/down 커널 쌍을 NVRTC로 다시 컴파일했고, 이 약 1.2초가 `decode_tok_per_sec`에 포함되었다. 수정은 `jit_module.cpp` 오버레이에 있다. 커스텀 커널도 내장 커널처럼 `MLX_PTX_CACHE_DIR`에 기록되고, 캐시 파일은 rename으로 게시되며, 손상된 항목은 미스로 처리되어 다시 빌드된다. `MLXCEL_FUSED_MOE` 기본값은 모든 디바이스에서 그대로이고 디바이스별 정책은 추가하지 않았다. 캐시 적용 후 granite 디코드(`-n 256`)는 89.20 tok/s로, `MLXCEL_FUSED_MOE=0`의 81.63, main의 53.63보다 빠르다. 9개 파일, +629/-78.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp` | 커스텀 커널을 디스크 캐시에 기록; 임시 파일 작성 후 rename으로 게시, `.txt` 맵을 `.ptx`보다 먼저; 읽기 시 비었거나 짧거나 형식이 잘못된 항목 거부; 캐시 히트가 로드에 실패하면 삭제, 재컴파일, 재게시를 한 번 수행; 캐시 디렉터리를 만들 수 없으면 예외 대신 캐시 없이 진행 |
| `src/models/granitemoehybrid.rs` | `forward`가 새 `forward_routed(x, try_fused)`에 위임; 게이트 자체는 변경 없음 |
| `src/models/granitemoehybrid_tests.rs` | #2283/#2245 라우팅 테스트를 fused 강제 경로, gather 강제 경로, `forward`가 `MLXCEL_FUSED_MOE`를 따르는지 확인하는 테스트로 분리; `=0`에서도 건너뛰지 않음 |
| `tests/cuda_custom_kernel_disk_cache.rs` (신규) | 세 프로세스 캐시 테스트 (채우기, 웜 히트, 손상된 캐시) |
| `src/models/switch_layers.rs`, `src/lib/mlxcel-core/src/lib.rs` | 게이트와 캐시 키에 대한 doc 주석 |
| `docs/environment-variables.md`, `docs/installation.md` | `MLXCEL_FUSED_MOE` 계열 목록에 `granitemoehybrid` 추가; 캐시 기본 경로와 커스텀 커널 포함 여부 문서화 |
| `docs/benchmark_results/fused-moe-jit-cache-gb10-2026-10-22.md` (신규) | 측정 기록 |

## 2. 기술적 선택과 그 이유

### 2.1 회귀는 토큰당 비용이 아니라 고정 비용이었다

`decode_tok_per_sec`는 첫 샘플 토큰부터 마지막 토큰까지를 센다. 프리필은 빠지지만 첫 단일 토큰 forward는 포함되고, 커스텀 커널의 첫 실행과 그 NVRTC 컴파일이 바로 여기에 들어간다. 커널이 토큰당 느리다면 `-n`이 커질수록 차이가 벌어져야 한다. 실제로는 줄어들었다:

| 모델 | `-n` | 기본값 | `MLXCEL_FUSED_MOE=0` |
|---|---|---|---|
| granite | 2 | 0.82 tok/s (2.44 s) | 1.66 tok/s (1.20 s) |
| granite | 32 | 11.72 (2.73 s) | 21.16 (1.51 s) |
| granite | 256 | 53.03 (4.83 s) | 65.84 (3.89 s) |
| lfm2 | 1024 (970개 생성) | 117.28 | 128.87 |

디코드 시간 = 고정 비용 + 토큰 수 x 토큰당 비용으로 맞추면, granite는 fused가 토큰당 9.4 ms에 고정 2.4 s, gather가 토큰당 10.6 ms에 고정 1.2 s이고, lfm2는 fused 6.5 ms, gather 7.0 ms다. fused 커널이 토큰당 더 빠르고 약 1.2~1.3 s의 고정 비용을 더 지고 있었다. 수정된 바이너리에서 granite `-n 2`는 기본값 3.78 tok/s, `=0` 3.84 tok/s로, 추가 절편이 사라졌다.

### 2.2 디바이스별 라우팅 정책을 두지 않음

이슈의 두 번째 수용 기준(디바이스별 결정)은 해당 없음으로 표시했다. main에서의 손실은 granite나 sm_121에 국한되지 않았다. lfm2가 더 크게 손해를 봤고(`=0`이 +45.5%, granite는 +24.1%), qwen3-30b-a3b는 `=0`이 +2.3%였으며, `MLXCEL_FUSED_MOE_SGY`를 2, 4, 16으로 바꿔도 기본값 대비 -1.3%~+0.5% 안에 머무는 동안 `=0`은 25.6% 앞섰다. 수정 후에는 세 계열 모두 fused 기본값이 이긴다(쌍별 비교로 `=0`이 granite에서 8.1%, lfm2에서 6.2%, qwen3-30b에서 17.2% 느림). sm_121에서 커널을 끄면 이 이득을 버리게 된다. gather 경로도 빨라졌는데(granite 65.84에서 81.63), granite의 SSM 스텝도 커스텀 커널이라 이제 함께 캐시되기 때문이다.

### 2.3 커스텀 커널 캐시는 내장 커널 캐시만큼 안전하다

업스트림 `custom_kernel.cpp`는 `get_jit_module`에 `use_disk_cache = false`를 넘긴다. `JitModule`은 이 값과 관계없이 캐시 읽기를 시도했고 쓰기만 건너뛰었으므로, 이 PR 이전에도 커스텀 커널은 심어진 캐시 파일을 로드할 수 있었다. 모듈 이름은 `custom_kernel_<name><템플릿 인자 해시>_<전체 소스 해시>`이므로, 이름만으로 키가 정해지는 내장 커널(그래서 `ensure_persistent_ptx_cache`가 MLX 핀별 디렉터리를 쓴다)보다 더 엄격하게 키가 정해진다. 오버레이는 이 매개변수를 무시하고(`(void)use_disk_cache`) 업스트림과 시그니처를 맞추려고 남겨 둔다.

### 2.4 게시 프로토콜

fused 디코드 커널이 캐시에 들어가면 잘못된 항목 하나가 이후 모든 실행의 디코드를 깨뜨리므로, 게이트뿐 아니라 쓰기 경로도 바꿨다:

- 각 파일을 `<path>.tmp<pid>_<tick>`에 쓰고, 닫고, 상태를 확인한 뒤 rename한다. 동시에 읽는 프로세스(병렬 테스트 바이너리, CLI 실행 옆의 서버)는 부분 파일을 보지 않는다. 한 프로세스 안에서는 `get_jit_module`이 쓰기를 직렬화하므로 pid와 steady clock tick으로 이름이 유일하다.
- `.txt` 커널 맵을 `.ptx`보다 먼저 게시하고, 맵이 성공했을 때만 `.ptx`를 게시한다. `read_cached_ptx`는 `.ptx` 존재로 히트를 판단하므로, `.ptx`가 있으면 맵은 완전하다. 같은 항목을 두 프로세스가 동시에 게시해도 내용이 바이트 단위로 같으므로, 한쪽의 맵과 다른 쪽의 코드가 섞여도 문제없다.
- `read_cached_ptx`는 빈 `.ptx`, 짧은 읽기, 탭이 없거나 필드가 빈 맵 줄, 빈 맵을 미스로 처리한다. 업스트림은 이를 모두 받아들인 뒤 `cuModuleLoadDataEx`에서 실패했다.
- 히트 후 `load_module`이 실패하면 커널 목록을 비우고, `cuModuleGetFunction` 이전에 이미 로드된 모듈 핸들이 있으면 언로드하고, `.ptx`를 삭제하고, 재컴파일, 재게시 후 한 번 더 로드한다. 두 번째 실패는 원래의 오류 경로를 따른다.
- `create_directories`는 `error_code`를 받는다. 캐시를 쓸 수 없으면 프로세스는 콜드 경로로 계속 진행한다.

### 2.5 granite 라우팅을 어느 디바이스에서나 테스트 가능하게

`forward_routed(x, try_fused)`가 라우팅 결정을 매개변수로 받으므로, 환경에 `MLXCEL_FUSED_MOE=0`이 설정되어도 #2245 테스트가 건너뛰지 않는다. 각 경로는 해당 후보와 비트 단위로 일치해야 하고, 공용 헬퍼는 두 후보가 서로 다름을 확인한다(따라서 비트 일치는 곧 라우팅 검증이다). 세 번째 테스트는 실제 `forward`가 두 설정 모두에서 게이트를 따르는지 확인한다.

### 2.6 리뷰 중 변경

- `83df4487`: 게시 단계가 임시 파일을 닫기 전에 상태를 확인했고(마지막 flush 실패를 놓침), 맵이 실패해도 `.ptx`를 게시했다. 이제 닫은 후 확인하고, `.ptx`를 맵 성공에 묶고, 임시 파일 이름에 프로세스 id를 넣는다.
- `5b8c4d2c`: 읽기에서 비었거나 형식이 잘못된 항목을 거부하고, 히트 후 로드 실패 시 한 번 재컴파일한다.
- `af9d047c`: 재시도 전에 첫 모듈 핸들을 언로드한다.

## 3. 검증

PR의 테스트 계획과 측정 기록에 따른 결과이며, GB10(sm_121, 커널 7.0.0-1019-nvidia, 드라이버 580.178.04, `MLX_CUDA_ARCHITECTURES=121`, test-fast 프로필, 순서를 돌리며 교차 실행, 블록마다 null arm)에서 측정했다:

| 확인 항목 | 결과 |
|---|---|
| granite-4.0-h-tiny, `-n 256`, n=7 | 수정 기본값 89.20, 수정 `=0` 81.63, main 53.63 tok/s; null 하한 -1.2~+0.7% |
| lfm2-8b-a1b, n=5 | 109.65 / 102.25 / 70.74; null 하한 -0.3~+2.3% |
| qwen3-30b-a3b, n=5 | 56.20 / 46.56 / 45.05; null arm 미실행 |
| `cuda_custom_kernel_disk_cache` | 오버레이를 되돌리면 실패, 적용하면 통과 |
| `granitemoehybrid` lib 테스트 | 17개 통과, `MLXCEL_FUSED_MOE=0`에서도 통과 |
| `switch_layers::`, `fused_moe` | 57개, 4개 통과, CUDA, `--test-threads=1` |
| clippy `-D warnings`, `cargo fmt --check`, `check_env_var_docs.py` | 깨끗함 |

캐시 테스트는 `MLX_PTX_CACHE_DIR`를 임시 디렉터리로 설정하고 fused MoE 커널 쌍을 실행한 뒤, `moe_gateup_kernel_cu`와 `moe_down_kernel_cu`가 각각 비지 않은 `.ptx`와 탭으로 구분된 맵을 남기고 `.tmp` 잔여물이 없는지 확인한다. 이어서 자기 바이너리를 같은 디렉터리로 자식 프로세스로 다시 실행해, stderr에 JIT 안내가 없고 출력이 부모와 비트 단위로 같아야 한다. 그다음 gate-up `.ptx`를 비우고 down 커널의 맵을 탭 없는 줄로 바꾼다. 세 번째 프로세스는 재컴파일하고(안내 출력), 부모와 비트 단위로 일치하며, 두 항목을 완전한 상태로 남겨야 한다. MLX가 `MLX_PTX_CACHE_DIR`를 프로세스당 한 번, 첫 JIT 컴파일 때 읽으므로 이 테스트는 바이너리 안에 단독으로 있다.

## 4. 미검증 및 미해결 항목

- **테스트되지 않은 분기:** 비어 있지 않고 형식도 맞지만 `cuModuleLoadDataEx`가 거부하는 항목. 이런 항목을 심는 테스트는 없고, 재시도는 콜드 경로의 컴파일과 게시를 재사용한다.
- **기존 문제, 후속 작업:** 캐시 키에 compute capability나 NVRTC 버전이 없으므로, 같은 MLX 핀에서 여러 GPU 세대가 캐시 디렉터리를 공유하면 다른 아키텍처용 cubin을 로드한다(내장 커널과 커스텀 커널 모두).
- **기존 문제, 후속 작업:** `ensure_persistent_ptx_cache`를 호출하는 것은 `mlxcel`과 `mlxcel-server`뿐이다. 벤치 바이너리, 테스트 바이너리, 임베딩 사용자는 MLX의 `$TMPDIR/mlx/<version>/ptx`를 쓰며 소유자 확인 없이 신뢰한다. 다중 사용자 호스트에서는 이 디렉터리를 먼저 만든 사람이 예측 가능한 이름으로 모듈을 심을 수 있다. 공유 마운트의 `MLXCEL_CACHE_DIR`도 같은 노출을 가진다.
- **사소한 항목:** 임시 파일을 쓰고 rename하기 전에 프로세스가 종료되면 `.tmp<pid>_<tick>` 파일이 남고 이를 지우는 곳이 없다. `get_ptx_path`는 245자를 넘는 이름에 대해 디렉터리를 중첩하는데, mlxcel의 커스텀 커널 중 그 길이에 닿는 것은 없다.
- **측정하지 않음:** ROCm과 Metal(오버레이는 CUDA 전용이라 영향 없음), `mlxcel-server` 정상 상태(서버는 어느 쪽이든 수명당 한 번 컴파일하며, 재시작 후 첫 요청만 달라진다), release 프로필 절대 수치, Windows CUDA 빌드.
- **오래된 근거:** `FUSED_MOE_MAX_DFF_CUDA = 8192`를 정한 #626 측정과 이 PR 이전에 기록된 GB10 fused MoE 수치는 이 고정 비용이 디코드 구간에 들어간 상태였다. 커널 사용 시 lfm2 +12%라는 기록은 main의 `-n 256`에서 성립하지 않는다. 다시 측정하지 않았다.

## 5. 학습 포인트

- 첫 디코드 스텝 안에 들어가는 일회성 비용은 `decode_tok_per_sec`에서 토큰당 속도처럼 보인다. 디코드 A/B를 커널 탓으로 돌리기 전에 `-n`을 최소 두 값으로 실행하고 절편과 기울기를 구해야 한다. `-n`이 커질수록 줄어드는 차이는 고정 비용이다.
- bisect는 비용을 드러낸 커밋을 찾지, 그 비용을 가진 코드를 찾는 것은 아니다. #2283은 granite를 커스텀 커널로 보냈을 뿐이고, 비용은 모든 커스텀 커널에 대한 MLX의 캐시 정책에 있었다.
- 이후 실행이 읽는 디스크 캐시에는 게시 프로토콜(임시 파일, 닫기, 확인, rename, 히트를 판단하는 파일보다 데이터를 먼저)과 의심스러운 것은 모두 미스로 처리하는 읽기가 필요하다. 그렇지 않으면 잘못된 쓰기 한 번이 이후 모든 실행을 실패시킨다.
- 프로세스 전역 설정을 한 번만 읽는 경우(여기서는 `MLX_PTX_CACHE_DIR`) 프로세스 간 동작은 테스트 바이너리를 `--exact`와 환경 표지로 자식 프로세스로 다시 실행해 검증하고, 그 테스트를 바이너리 안에 단독으로 둔다.
