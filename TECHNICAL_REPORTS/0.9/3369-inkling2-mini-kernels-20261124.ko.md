# PR #3369: Inkling 2 Mini Metal 커널

**날짜**: 2026-11-24
**상태**: Draft PR 공개 (#3359 종료). 범위를 좁힌 테스트 통과, M3 Ultra (512 GB)에서 Inkling 2 Mini 체크포인트 두 개로 검증

## 요약

이 PR은 mlx-vlm의 Inkling 2 Mini Metal 커널 세 가지를 이식한다. 융합 short convolution, attention head 준비, 그룹화된 NVFP4/MXFP8 expert matvec이다. 각 커널은 대체하는 ops 경로와 비트 단위로 같고, 게이트를 벗어나면 그 경로로 돌아간다. `MLXCEL_INKLING2_KERNELS=0`을 주면 세 커널이 모두 꺼진다. 디코드 속도는 네이티브 체크포인트에서 32.1에서 35.3 tok/s로, 4-bit 변환본에서 36.8에서 44.8 tok/s로 올라 둘 다 mlx-vlm(32.8, 38.5)보다 빠르다. 8토큰 verify forward는 60.4에서 51.8 ms로 줄었다. greedy 출력은 커널 사용 여부와 관계없이 바이트 단위로 같다. 그룹 커널이 실제로 실행되도록 Inkling 2가 prompt-lookup verify 라운드를 지원하게 했다.

## 1. 문제

#3367 이후 Inkling 2 Mini의 디코드 속도는 네이티브 32.1 tok/s, 4-bit 32.0 tok/s였고, mlx-vlm은 32.8과 38.5였다. 업스트림에는 이식에서 빠진 커널이 세 개 있었다. 토큰마다 레이어 방문이 48번 일어나고, 방문마다 short convolution 네 개가 각각 작은 연산 8개 정도로, head 정규화 네 개가 각각 7개 정도로 돌았다. 짧은 verify에서는 route마다 expert 가중치를 따로 디코딩했다.

| 위험 | 영향 | 대응 |
|---|---|---|
| 커널이 수치를 바꿈 | greedy 출력이 ops 경로 및 mlx-vlm과 달라짐 | 비트 단위 parity 테스트와 greedy 출력 일치로 확인 |
| 커널이 #3367의 trim 레이아웃을 깨뜨림 | speculative rewind가 부정확해짐 | 커널이 같은 history를 기록함 |

## 2. 설계

### 2.1 커널과 게이트

| 커널 | 게이트 | 업스트림과 다른 점 |
|---|---|---|
| Short convolution (02053f5c) | bf16 `x`, kernel 4, `B <= 4`, `L <= 8`, `C % 128 == 0`, f32 history, Metal | `W - 1 + L` 창 대신 다음 margin history(`min(W - 1 + margin, R + L)`행)를 직접 기록. padding mask 없음 |
| Head 준비 (de08fdf8) | loop 모델, gamma 없는 value norm, `[1, 1..=8, H, 128]` bf16, `d_rel == 16` | 모델이 보관하는 f32 `q_scale` / `rel_scale`을 읽음 |
| 그룹 expert matvec (8e80d434) | loop 모델, speculative verify forward, `[1, 2..=8, hidden]` bf16, top-k 64 미만, 모든 plane이 bias 없는 네이티브 NVFP4 또는 MXFP8 | 업스트림 인자 대신 `speculative_append_active()`로 게이트 |

launcher는 `src/lib/mlxcelverse/kernels/inkling2_kernels.cpp`와 `inkling2_grouped_experts.cpp`에 있다. 각 launcher는 Metal 항목만 있는 `KernelPorts` 표를 두고 `select_kernel_port`로 선택된다. bridge 선언은 `Result`를 반환하고, `mlxcel_core::inkling2_kernels`가 거부를 `None`으로 바꾼다. JIT preamble에 MLX의 fp4/fp8 디코더가 포함되지 않아서 그룹 커널에 직접 넣었다. lane 배정, dot 묶음, 합산 순서, global scale 적용 위치는 `fp_qmv_fast_impl` / `fp_qmv_impl`을 따르므로 각 route 결과가 해당 `gather_qmm` 행과 같다.

### 2.2 그룹 커널 실행 경로

일반 디코드는 다중 토큰 forward를 하지 않고, Inkling 2는 원래 verify 라운드를 거부했다. 이제 모델이 `supports_speculative_verify_rewind`를 보고하고, `rewind_decode_appends`를 `trim_state`로 구현한다. `trim_state`는 상태를 바꾸기 전에 모든 방문을 먼저 확인한다. verify forward는 convolution 행을 `max(16, L)`개 더 보관하고(`conv::history_margin`), sliding KV의 margin 64가 가장 큰 prompt-lookup 블록(65)까지 감당한다. 따라서 허용되는 크기의 거부된 블록은 모두 정확히 되감을 수 있다.

## 3. 검증

| 항목 | 결과 |
|---|---|
| Short convolution parity: f32·bf16 출력, residual 없음·bf16·f32, strided `x`·history·residual, history 0~19행 | 비트 단위 일치, 다음 history 일치 |
| Short convolution rewind: speculative 블록 5, 0·2·4·5 유지 후 다음 step | history와 출력 비트 단위 일치 |
| Head 준비 parity: `L` 1·4·8, scale 0.25·5·1, strided q·r, `L = 8`에서 NaN 주입 | 비트 단위 일치(`u16` view) |
| 그룹 expert: Mini 크기의 NVFP4(expert별 global scale 포함)와 MXFP8, expert 5가 tile 두 개, 나머지는 길이 1~3 | 토큰별·배치 `SwitchGLU`와 비트 단위 일치 |
| 게이트 | CPU, batch 2, `L = 9`, head_dim 64, `d_rel` 8, f16, loop 아닌 config, affine plane, top-k 64, verify 아닌 forward에서 거부 |
| greedy 128토큰, 두 체크포인트, arm마다 3회 | arm 간 동일 |
| `--prompt-lookup` | 두 arm 모두 실행되고 출력 동일 |

디코드(`--profile`, A = 커널 끔): 네이티브는 32.28 / 32.97 / 30.95에서 32.80 / 36.61 / 36.47 tok/s, 4-bit는 35.94 / 37.21 / 37.28에서 44.75 / 44.72 / 44.80 tok/s. 네이티브 verify forward 중앙값은 width 4가 52.0에서 45.7 ms, width 8이 60.4에서 51.8 ms. speculative scope를 빼면 그룹 커널만 빠져 46.9 / 54.5 ms가 되므로, 그룹 커널의 몫은 2.6%와 5.0%다. argmax digest는 모든 모드에서 일치한다.

GPU lock 아래 순차 실행한 selector: `--lib inkling`(142개 통과, `models::inkling2::` 포함), `--lib coverage_tests`(18개). clippy, fmt, CI 계약 스크립트도 통과했다.

## 4. 한계와 후속 과제

- short convolution과 head 준비 커널은 함께 측정했다. 그룹 커널만 verify 타이밍으로 따로 분리했다.
- 예시 프롬프트에서는 prompt-lookup의 제안 채택이 적어 일반 greedy보다 디코드가 느리다(27.4 / 31.7 tok/s). 출력은 일반 greedy와 213자까지 같고, 그 뒤로는 다중 토큰 verify forward가 다른 표현을 고른다. 이 결과는 두 arm에서 같다.
- 커널은 Metal 전용이다. CUDA나 HIP로 이식할 때는 `check_kernel_dtype_keys.py`의 `EXPECTED_IN_SCOPE`에 해당 파일을 추가해야 한다.
- Inkling 2 Small은 `loop_config`가 없어서 업스트림과 같이 loop 커널을 쓰지 않는다. Small의 short convolution 커널은 단위 테스트로만 검증했다.
