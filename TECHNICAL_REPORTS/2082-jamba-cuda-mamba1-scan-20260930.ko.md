# 기술 보고서: PR #2082 - CUDA에서 Jamba Mamba 스캔을 융합 커널로 처리 (그래프와 비트 단위 동일)

**작성일**: 2026-09-30

**상태**: GB10에서 구현과 검증 완료, 머지 대기.

**언어**: C++ (mlxcel-core 브리지, CUDA JIT 소스), Rust, Markdown

**위험도**: 낮음에서 중간. CUDA의 Jamba가 스텝 단위 그래프 스캔 대신 커스텀 커널을 쓰게 된다. 입력 dtype이 모두 같으면 커널 결과는 그래프 스캔과 비트 단위로 같고, 그렇지 않으면 그래프 스캔으로 돌아간다. Metal 동작은 그대로이고 CUDA의 Mamba와 Falcon-Mamba도 바뀌지 않았다.

## 요약

이슈 #1981은 GB10의 `mlxcel-server`에서 약 3.3k 토큰짜리 Jamba 채팅 요청 하나에 300초 가까이 걸린다고 보고했다. 그 수치를 낸 바이너리는 PR #2000 이전 코드였고, 당시 Jamba 프리필 스캔은 스텝마다 상태 텐서를 점점 커지는 결과에 `concatenate`했기 때문에 프롬프트 길이의 제곱에 비례해 느려졌다. GB10에서 그 부분만 되돌려 측정하면 548 토큰 프리필이 8.1초, 1059 토큰이 26.6초로, 3299 토큰으로 외삽하면 260~300초가 된다. 3299 토큰으로 서버를 실제로 돌리자 호스트의 121 GB 메모리가 바닥나 OOM으로 종료됐다.

main에서는 같은 요청이 이미 4.3초였다. 프리필 4.1초 중 약 2.3초가 남아 있던 타임스텝별 그래프 루프였는데, Mamba 레이어 26개마다 타임스텝 하나에 MLX 연산을 10개가량 실행하는 구조였다. #2005의 융합 Mamba1 스캔 커널에 Metal 포트만 있었기 때문이다. 이 PR은 CUDA 포트를 추가한다. 같은 요청의 서버 시간은 4.33초에서 1.51초로(2턴은 4.17초에서 1.53초로) 줄었고 그리디 출력은 바뀌지 않았다.

## 1. 원인 분석

| 코드 | 스캔 구현 | GB10 비용 |
|---|---|---|
| #2000 이전 (이슈의 바이너리) | 스텝 루프에 스텝마다 `[B, 1, 5120, 16]` 상태를 `concatenate` | 548 토큰 프리필 8.1초, 1059 토큰 26.6초 (각 n=3), 3299 토큰에서 OOM |
| main `929c80ab` | 스텝 단위 그래프 루프, `y` 행은 한 번에 stack | 요청당 4.3초, 프리필 중 약 2.3초가 스캔 |
| 이 PR | 레이어당 CUDA 커널 1회 | 요청당 1.5초 |

main에서 스캔이 차지하는 몫은 스캔을 `delta`, `B`, `C`를 소비하는 모양만 맞춘 연산으로 바꾼 측정용 빌드로 쟀다. 같은 세션에서 main 프리필이 3.59~3.74초일 때 이 빌드는 1.33~1.35초였다(각 n=3). 이슈가 꼽은 나머지 후보는 이 체크포인트에 해당하지 않는다. `num_experts`가 1이라 MoE 라우팅은 dense 전문가 하나로 끝나고, 서버의 512 토큰 청크 프리필도 이미 적용되고 있었다.

`jamba-v0.1-4bit`라는 이름의 체크포인트는 축소판이다. 레이어 28개(Mamba 26개, 어텐션 2개는 7번과 21번), hidden 2560, Mamba intermediate 5120, d_state 16이고 디스크 용량은 1.7 GB다.

## 2. 변경 사항

- **`mlx_cxx_kernels.cpp`**: `mamba1_selective_scan`의 `mx.fast.cuda_kernel` 포트를 추가했다. 두 변형은 반올림 방식이 달라서 `KernelPorts` 표를 따로 둔다(float32 상태의 Metal 커널은 `mamba1_scan_ports`, CUDA 커널은 `mamba1_scan_graph_exact_ports`). 런처는 실행 중인 백엔드의 포트가 어느 표에 있는지로 변형을 고르고 백엔드 종류를 직접 비교하지 않는다(`scripts/ci/check_kernel_port_dispatch.py`가 검사하는 규칙). `mamba1_scan_kernel_available()`은 이제 두 표를 읽는다(`MLXCEL_MAMBA1_SCAN_KERNEL=0`은 그대로 존중). 새 함수 `mamba1_scan_kernel_accepts(x, delta, b, c, a, d)`는 커널을 써도 되는 호출인지 판정한다. 커널이 있고, 기본 장치가 GPU이고, `N <= 32`이며, CUDA에서는 입력 6개의 부동소수점 dtype이 모두 같아야 한다. CUDA에서는 상태를 활성값 dtype으로 주고받고 Metal에서는 float32를 유지한다.
- **`jamba.rs`**: `JambaMambaMixer::ssm_step`은 `mamba1_scan_kernel_accepts`가 참이면 프리필과 디코드 모두 커널을 호출한다. 테스트 전용 thread-local 카운터가 그래프 루프를 돈 타임스텝 수를 센다.
- **`mamba.rs`**: 융합 경로는 계속 Metal 전용이다. Mamba의 그래프 경로는 `x_proj`를 타임스텝 하나씩 계산하고 커널 경로는 시퀀스 전체를 한 번에 투영하므로, CUDA에서 바꾸면 출력이 달라진다.
- **테스트**: `cuda_kernel_is_bit_identical_to_the_graph_scan`, `cuda_kernel_declines_mixed_dtype_inputs`, `jamba_mamba_prefill_takes_the_fused_scan_where_a_port_exists`를 추가했다. Metal 커널의 정확도 테스트는 Metal에서만 돌게 했다.
- **문서**: `docs/environment-variables.md`의 `MLXCEL_MAMBA1_SCAN_KERNEL` 항목과 `docs/benchmark_results/jamba-mamba1-scan-kernel-gb10-2026-09-30.md`.

## 3. 기술적 결정

### Metal 커널의 float32 상태가 아니라 그래프의 반올림을 재현

이슈는 그리디 출력이 현재 구현과 같을 것을 요구한다. Metal 커널은 상태를 float32로 유지하는데, #2005의 기록을 보면 긴 프롬프트에서 그래프 스캔과 그리디 출력이 갈렸다. CUDA 포트는 그래프가 하는 연산을 그대로 수행한다. `new_state = (delta * x) * B`, `dtA = exp(delta * A)`, `state = state * dtA + new_state`를 각각 활성값 dtype으로 반올림하고, 그다음 `y = state @ C`와 `y + D * x`를 계산한다. 비트 단위로 같게 만든 요소는 세 가지다.

첫째, 지수 함수는 `unary_ops.cuh`에 있는 MLX의 `Exp` 디바이스 functor를 쓴다. 그래프의 `exp` 연산이 실행하는 코드와 같다.

둘째, `y_t`는 그래프의 matmul을 재현한다. MLX는 `[B, D, N] @ [B, N, 1]`을 `gemv`로 보내고, `K = N < 64`이면 레인 하나가 원소 하나를 맡아 float로 곱한 뒤 32 레인 워프에서 `cg::reduce`로 더하고 T로 반올림한다. 커널도 N 이후 레인은 0을 내며 같은 방식으로 계산한다.

셋째, 곱셈과 덧셈에 `__fmul_rn`/`__fadd_rn`과 `__hmul_rn`/`__hadd_rn`을 쓴다. 그래프는 연산마다 별도 커널을 실행하므로 곱을 반올림한 다음에 더한다. NVRTC는 기본값이 `--fmad=true`라서 인라인된 `a * b + c`를 반올림이 한 번뿐인 fma 하나로 합친다. 일반 연산자를 쓴 첫 초안은 f32까지 포함해 출력의 절반가량이 그래프와 1 ulp씩 달랐고, `_rn` 형태로 바꾸자 차이가 모두 사라졌다.

### dtype이 모두 같을 때만 커널 사용

그래프는 dtype이 섞이면 연산마다 승격하므로(float32 `A_log`라면 첫 스텝 이후 상태가 float32가 된다) 커널은 입력 dtype이 모두 같을 때만 정확하다. CUDA에서 `mamba1_scan_kernel_accepts`가 이를 확인하고 아니면 거짓을 돌려주므로, 그런 모델은 그래프 스캔과 기존 출력을 유지한다. 공개된 Jamba 변환본은 모두 bf16이라 커널을 쓴다.

### 기본 장치가 CPU면 그래프 스캔 유지

커스텀 커널은 GPU 스트림에서만 실행된다. `MLXCEL_DEVICE=cpu`에서는 전에 그래프 스캔이 CPU에서 돌던 자리에서 커널 호출이 예외를 던지게 되므로, `accepts`는 기본 장치가 GPU인지도 확인한다. GB10에서 CPU 장치로 Jamba 생성을 돌려 main과 같은 출력을 확인했다.

## 4. 검증

호스트는 GB10, 커널 7.0.0-1019-nvidia, 드라이버 580.178.04, CUDA 13.0이며 `--features cuda` 릴리스 빌드를 썼다.

서버(기본 실행, 3299 토큰 채팅, `max_tokens` 64, temperature 0, 서버 실행당 프롬프트 3개, ABBA 순서, 칸마다 n=6):

| | main | 이 PR |
|---|---|---|
| 1턴 | 4.33초 (4.23~4.49) | 1.51초 (1.51~1.74) |
| 2턴 | 4.17초 (4.08~4.28) | 1.53초 (1.52~1.56) |

CLI(`mlxcel generate --profile`)의 프리필은 main이 3.59~4.16초(n=8, 두 세션), 이 PR이 2.14~2.21초(n=3)였고 디코드는 69~77 tok/s에서 76~79 tok/s가 됐다. CLI 프리필에는 커널을 NVRTC로 한 번 컴파일하는 시간이 들어 있다. 서버는 이 비용을 시작 시 워밍업에서 치른다(워밍업 1.1초에서 1.8초).

출력 동일성은 세 층위에서 확인했다. 커널과 그래프 스캔은 bf16, f16, f32, d_state 8과 16, 1·7·33 스텝, 새 상태와 이어받은 상태 모두에서 출력 행과 최종 상태가 비트 단위로 같다. 서버에서는 이 PR의 두 실행이 첫 번째 main 실행과 요청 12개 모두 문자 단위로 같았다(reasoning과 content). 두 번째 main 실행은 첫 번째 main 실행과 요청 하나(한 프롬프트의 2턴)에서 달랐으니 이 호스트에서 main 자체가 실행마다 완전히 결정적이지는 않다. 그 요청에서도 PR 실행은 첫 번째 main 실행과 같았다. CLI 그리디 출력은 548, 1059, 3299 토큰에서 main과 같다.

회귀 방지 테스트 `jamba_mamba_prefill_takes_the_fused_scan_where_a_port_exists`는 커널로는 통과하고, `MLXCEL_MAMBA1_SCAN_KERNEL=0`에서는 "prefill and decode walked 49 graph-scan timesteps"로 실패한다. 이 설정이 main이 CUDA에서 타던 경로다.

CUDA에서 `--test-threads=1`로 `models::jamba`(11개 통과, 1개 ignored), `models::mamba`, `mamba1` 패리티 테스트가 통과한다. fmt와 clippy(`-D warnings`, lib와 tests, 메인 크레이트와 mlxcel-core)도 깨끗하다.

## 5. 다루지 않은 것

- **Jamba의 프롬프트 캐시 재사용**: main과 이 PR 모두 서버 로그에 매 턴 `cached=0`이 찍혀 턴마다 전체 대화 이력을 다시 프리필한다. 이슈에서 2턴이 1턴보다 느렸던 이유다.
- **프로세스마다 NVRTC 컴파일**: MLX는 커스텀 CUDA 커널을 디스크에 캐시하지 않으므로 프로세스마다 커널을 한 번 컴파일한다(약 0.7초).
- **CUDA의 Mamba / Falcon-Mamba**: 여전히 그래프 스캔이다(2절 참고).
- **Metal**: 로직은 같다(`gpu_kernel_backend()`가 Metal인 조건은 예전 검사인 `metal::is_available()`과 같다). 다만 여기서 실행하지는 않았다. ROCm은 포트가 없다.
