# 기술 보고서: PR #2086 - gfx1151 decode를 kernel 단위로 profile하고 #1814 port 순서를 정함

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `0d5d6db0`, origin/main `f9aefa39` 위로 rebase됨, 머지 대기 중.

**언어**: Rust (bench binary flag와 phase mark, test gate), Bash (idle-GPU guard, profile driver), Python (trace 절단, attribution, report, CI checker), Markdown, CSV/JSON (커밋된 profile 데이터)

**위험도**: 낮음 (추론 경로는 바뀌지 않습니다. bench binary에 flag 두 개와 opt-in 환경 변수가 추가되며 기본값에서는 출력이 그대로입니다. test gate 세 개가 Metal 또는 CUDA에서 모든 GPU backend로 넓어지고, 나머지는 script, 문서, 데이터입니다)

## 요약

Issue #2061 (#1814의 일부, epic #1801)은 #1814가 가설로 남겨 둔 측정을 요구했습니다. ROCm decode 시간이 kernel별로 어디에 쓰이는지, 그리고 #1814에서 분리된 다섯 개 port issue를 어떤 순서로 진행해야 하는지입니다. 이 PR은 profiling harness를 추가하고, Radeon 8060S (gfx1151)에서 checkpoint 네 개로 실행한 결과를 `docs/benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md`와 `benchmarks/rocm_profiles/gfx1151_929c80ab/`의 원자료로 공개합니다. 같은 세션에서 두 번째 미해결 질문도 정리했습니다. MLX의 ROCm `gather_mm`은 동작하므로, 그 numeric test가 이제 ROCm에서도 실행됩니다.

측정된 순서는 **#2067 > #2065 > #2064 > #2068 > #2063** (#1814 항목 7, 5, 4, 8, 3)입니다. #1814의 가설은 호출 빈도를 근거로 항목 3 (모든 모델에서 layer마다 token마다 실행되는 fused add-RMSNorm과 RoPE-append)을 맨 앞에, 항목 7 (한 모델 계열에서만 쓰이는 SSM update)을 뒤쪽에 두었습니다. Profile은 이 둘을 뒤집습니다. #2067은 granite-4.0-h-tiny decode GPU 시간의 29.8%, Nemotron-3-Nano의 20.0%와 두 모델 dispatch의 절반 이상에 닿는 반면, #2063은 두 fusion 모두 모든 backend에서 꺼진 채로 배포되기 때문에 모든 모델에서 0%입니다.

그 밖의 결과는 다음과 같습니다. Dense decode는 사실상 kernel 하나 (`qmv_wide_kernel`, Llama 3.1 8B decode GPU 시간의 94.8%)이고 약 192 GB/s로 동작합니다. Qwen3-30B-A3B에서 #2065가 차지하는 46.8%의 대부분은 이미 그 대역폭에 가까운 expert GEMV이므로, 회수 가능한 것은 약 5%와 token당 약 430개의 dispatch뿐입니다. Profiler 자체 비용은 dispatch 수에 비례하며 (Llama는 noise 수준, Qwen3는 5.3%, hybrid는 20~23%), 그래서 문서는 시간이 아니라 비율을 보고합니다.

## 1. 문제 정의

### 1.1 측정 없는 순서

2026-09-30에 #1814가 아홉 개 issue로 나뉘었습니다. 그중 다섯 개 (#2063, #2064, #2065, #2067, #2068)는 `KernelPorts` table의 `.rocm = nullptr` 칸을 채우는 HIP port이고, 모두 #2061에 의존합니다. #1814는 이들의 순서에 대해 출발점이 되는 가설만 제시했습니다. `fused_add_rms_norm`과 `fused_rope_qk_append`는 모든 모델에서 layer마다 token마다, sampler 두 개는 모든 모델에서 token마다 한 번, MoE와 SSM kernel은 해당 계열에서만 token마다, paged attention은 paged 경로에서만 실행된다는 것입니다. Issue는 순서를 정하는 것은 profile이며, 빈도 논리는 reviewer가 검증할 대상이지 전제로 삼을 것이 아니라고 명시했습니다.

이 PR 이전의 ROCm decode 데이터는 #2056 baseline뿐이었습니다. End-to-end tok/s (Llama-3.1-8B-4bit 35.41 tok/s)는 있었지만 kernel별 분해가 없어서, GEMV 대역폭 상한과의 차이가 kernel 시간인지, graph fallback 시간인지, host 시간인지 알 수 없었습니다.

### 1.2 마지막 `BACKEND_ENUMERATION_TODO` 항목

`grouped_gemm_numeric_tests.rs`는 세 test를 `!metal_is_available() && !cuda_is_available()`로 막고 있어서, ROCm에서는 GPU를 건드리기 전에 반환했습니다. 이 test는 mlxcel port가 아니라 MLX 자체의 `gather_mm`을 검사하므로 적용할 `KernelPorts` table이 없고, 이 파일은 `scripts/ci/check_kernel_port_dispatch.py` checker rule 4의 마지막 예외였습니다. Test를 돌려 보지 않고 gate를 넓히는 것은 #1806 abort로 이어진 바로 그 방식이라 배제되었습니다.

## 2. 변경 요약

Commit 다섯 개:

- **`0908c888`** `test(rocm): run MLX gather_mm numeric tests on every GPU backend`: gate 세 개가 `crate::gpu_backend_available()`를 읽고, `BACKEND_ENUMERATION_TODO`는 빈 set이 됩니다.
- **`10bee856`** `feat(bench): add a ROCm per-kernel decode profile harness`: phase mark, `--temperature` / `--top-p`, guard script, profile driver와 후처리기, unit test, `docs/benchmarks.md`.
- **`c098e02d`** `docs(rocm): publish the gfx1151 decode profile and #1814 port order`: 결과 문서와 `benchmarks/rocm_profiles/gfx1151_929c80ab/` 아래 파일 53개.
- **`523882d1`** `docs(rocm): correct per-step counts in the gfx1151 decode profile`: review 반영 (생성 token 128개 대 window 안 forward pass 127회, SSD step당 정확한 dispatch 수 47과 50, #2063 opt-in 상한을 0.83~0.89%로 수정).
- **`0d5d6db0`** `fix(rocm): harden the decode profile scripts after security review`: EXIT trap에서 `guard.log`의 로컬 경로 제거, guard의 더 안전한 PID parsing과 option 검증, INT/TERM 정리, slot을 쓰는 `Dispatch` class. 커밋된 데이터에 대한 `rocm_decode_profile.py report` 출력은 변경 전후 byte 단위로 같습니다.

### 2.1 Harness

- **Phase mark** (`src/bin/bench_decode/phase_marks.rs`). `MLXCEL_BENCH_PHASE_MARKS=1`이면 bench가 stderr에 `warmup_start`, `measured_start`, `decode_start`, `measured_end` 네 줄을 `CLOCK_MONOTONIC`과 `CLOCK_BOOTTIME` 양쪽 값으로 출력합니다 (tracer마다 쓰는 clock이 다르기 때문입니다). `measured_end`는 마지막 `synchronize_default()` 전에 읽습니다. Decode loop가 이미 마지막 token을 기다렸기 때문입니다. `decode_start`는 loop의 시작이 generator 내부에 있으므로 `measured_end`에서 generator 자신의 `decode_time_ms`를 빼서 구합니다. 설정하지 않으면 `scripts/bench_decode.sh`가 보는 bench 출력은 바뀌지 않습니다.
- **`mlxcel-bench-decode`의 `--temperature`와 `--top-p`** (기본값 0.0과 1.0이므로 이전처럼 greedy). Greedy argmax는 sampler를 dispatch하지 않기 때문에, profile에 sampler가 보이게 하려고 추가했습니다.
- **`scripts/rocm_gpu_guard.sh`**: #2056의 idle-GPU 확인을 재사용 가능한 script로 만든 것 (4절).
- **`scripts/rocm_decode_profile.sh`**: 모델마다 plain bench 실행 한 번과 `rocprofv3 --kernel-trace --hip-graph-trace --stats -f csv` 아래 실행 한 번을 모두 guard를 거쳐 수행합니다. Shape는 `bench_decode.sh` 기본값인 pp512/tg128, 같은 process 안 20-token warmup, `--ignore-eos`입니다. 전체 trace (각 20~60 MB)는 `--trace-dir`로 보존하지 않으면 임시 디렉터리에 둡니다.
- **`scripts/rocm_decode_profile.py`**: phase mark로 trace에서 측정 decode 구간을 잘라내고, dispatch에 역할을 붙이고 (`assign_roles()`), 배포 설정에서 각 port가 어떤 역할에 닿는지 판정하고 (`reach()`), `<run>_decode_kernels.csv`와 `<run>_summary.json`을 씁니다. `report` subcommand가 표를 만듭니다. `tests/test_rocm_decode_profile.py`가 합성 decode step으로 역할 규칙을 고정합니다.

### 2.2 `f9aefa39` 위로의 rebase

PR #2084가 `src/bin/bench_decode.rs`의 같은 함수들에 phase별 `[Memory]` 출력 (issue #2062)을 추가했습니다. merge-resolver가 두 동작을 모두 유지하는 방향으로 충돌을 해결했습니다. 메모리 counter는 각 phase 경계에서 출력되고, phase mark는 그 주변에 출력됩니다 (`reset_peak_memory()` 다음 `warmup_start`, `measured_start` 다음 측정 pass, 그다음 `decode_start` / `measured_end`, 그다음 `print_memory_phase("after measured pass")`).

## 3. 측정 결과

### 3.1 방법 요약

Checkpoint 네 개: `Meta-Llama-3.1-8B-Instruct-4bit` (dense, f16), `Qwen3-30B-A3B-4bit` (MoE, expert 128개 중 8개 활성), `granite-4.0-h-tiny-4bit` (Mamba2 36 + attention 4 layer, MoE), `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` (Mamba2 23, MoE 23, attention 6). 각각 greedy와 `--temperature 0.7`, `--temperature 0.7 --top-p 0.95`로 실행했습니다. ROCm 10.0.0, HIP 7.15.26333, rocprofv3 1.3.5, mlxcel `929c80ab`에 harness 수정을 더한 것, overlay `75915908`과 `LOCAL_FIXES.md` 항목 1~26입니다.

Token당은 생성된 token당 (bench tok/s의 분모)을 뜻합니다. 128개 token 중 첫 번째는 prefill에서 나오므로 window에는 forward pass 127회가 들어 있고, token당 호출 수는 step당 호출 수의 127/128로 읽힙니다 (Qwen3의 step당 expert GEMV 144회는 token당 142.9회로 보입니다). GPU 시간은 window 안 kernel 구간의 합집합이고, host gap은 window wall time에서 그것을 뺀 값입니다. 절단은 검증됩니다. 모든 실행에서 절단점에 걸친 kernel이 없고, 첫 decode dispatch 전에 device가 2~57 ms 쉬며, 절단 직전 dispatch는 항상 첫 token의 sampling입니다.

### 3.2 모델별 (greedy)

| 모델 | GPU ms/token | Host gap ms/token (profiled) | Host gap ms/token (plain wall - GPU) | Dispatch/token |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 23.34 | 1.80 | 3.63 | 492 |
| Qwen3-30B-A3B | 13.36 | 3.63 | 2.77 | 1540 |
| granite-4.0-h-tiny | 11.47 | 8.59 | 4.83 | 3197 |
| Nemotron-3-Nano | 14.28 | 8.93 | 5.05 | 2008 |

Decode GPU 시간 비율 기준 상위 kernel (token당 호출 수):

| 모델 | Kernel |
|---|---|
| Llama 3.1 8B | `qmv_wide_kernel` 94.8% (160), `kernel_sdpav_1pass` 2.1% (32), `rms_norm_kernel` 1.2% (64), `binary_vv<Add>` 0.6% (64), `copy_gg_byval` 0.5% (65), `rope_single_freqs_1d` 0.4% (64) |
| Qwen3-30B-A3B | `gather_qmv_wide_kernel` 42.1% (143), `qmv_wide_kernel` 33.1% (144), `kernel_sdpav_1pass` 4.6% (48), `block_sort_kernel` 4.3% (48, router top-k), `rms_norm_kernel` 3.9% (191), compiled SwiGLU 1.6% (48) |
| granite-4.0-h-tiny | `qmv_wide_kernel` 24.3% (168), `gather_qmv_wide_kernel` 21.9% (119), `binary_vv<Add, f32>` 6.6% (143), `binary_g<Multiply, f32>` 5.6% (178), `block_sort_kernel` 3.6% (40), `rms_norm_kernel` 3.4% (116) |
| Nemotron-3-Nano | `qmv_wide_kernel` 38.6% (116), `gather_qmv_wide_kernel` 27.9% (46), `binary_vv<Add, f32>` 4.1% (92), `binary_g<Multiply, f32>` 3.4% (114), `gemv_batched_inline<f32>` 2.2% (46), `copy_v` 2.1% (229) |

표에서 세 가지가 읽힙니다.

- **Dense decode는 `qmv`입니다.** Llama의 GEMV는 token당 약 4.2 GB를 22.1 ms에 옮기므로 192 GB/s입니다. #1814 port 목록의 어떤 항목도 이 kernel을 건드리지 않으므로, 어떤 port도 dense decode를 크게 바꾸지 못합니다.
- **Qwen3의 expert GEMV도 대역폭에 가깝습니다.** 48 layer x expert 8개 x 2048 x 768 행렬 3개, 4.5 bit이면 token당 1.02 GB를 5.63 ms에 옮기므로 181 GB/s이고, dense GEMV (0.69 GB를 4.42 ms, 157 GB/s)와 비슷합니다. 따라서 fused MoE 비율의 대부분은 fused kernel도 여전히 해야 하는 일입니다.
- **Hybrid 모델이 GPU를 놀립니다.** Token당 dispatch 3197개와 2008개, plain 실행 기준 host gap token당 4.8 ms와 5.1 ms (dispatch당 약 1.5 us와 2.5 us), 그리고 1~2 us짜리 f32 elementwise kernel이 길게 이어지는데, 이것이 Mamba2 SSD step입니다.

### 3.3 Profiling overhead와 비율로 보고하는 이유

각 greedy 실행은 같은 guard 아래에서 profiler 없이도 수행했습니다.

| 모델 | Dispatch/token | Decode tok/s, plain | Decode tok/s, profiled | Profiled가 느린 정도 |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 492 | 37.08 | 39.78 | -6.8% (더 빠름, 실행 간 편차 이내) |
| Qwen3-30B-A3B | 1540 | 61.99 | 58.86 | 5.3% |
| granite-4.0-h-tiny | 3197 | 61.35 | 49.86 | 23.0% |
| Nemotron-3-Nano | 2008 | 51.74 | 43.08 | 20.1% |

rocprofv3는 dispatch마다 host 시간을 더하고 kernel 실행 시간은 거의 바꾸지 않으므로, 비용이 dispatch 수를 따라갑니다. 순위가 host gap에 달려 있는 hybrid 모델에서 profiled wall time이 20~23% 느리므로, profiled 절대 시간은 연구 대상인 바로 그 양을 과대평가합니다. Decode GPU 시간의 비율은 이 영향을 받지 않고, host gap은 plain wall time per token에서 trace된 GPU 시간을 뺀 값 (3.2 표의 세 번째 열)을 우선합니다. Overhead 수치는 모델당 plain 한 번과 profiled 한 번이므로, Llama 값은 비용이 noise 이하라는 뜻일 뿐입니다 (#2056 baseline과 `LOCAL_FIXES.md` 항목 24는 이 모델에서 35.4~35.9 tok/s).

## 4. Idle-GPU guard

UMA 호스트에서는 다른 GPU 사용자나 compiler가 같은 메모리 bus를 다툽니다. 이 세션 동안 호스트는 다른 unit과 orchestrator gate와 공유되었습니다. `scripts/rocm_gpu_guard.sh [--idle-secs N] [--max-attempts N] [--max-wait SECS] [--log FILE] -- COMMAND`는 다음과 같이 동작합니다.

1. `/sys/class/kfd/kfd/proc`이 비어 있고 compiler process (`/proc/<pid>/comm`을 `cargo`, `rustc`, `clang*`, `hipcc`, `nvcc`, `cc1`, `cc1plus`, `ld*`, `lld`, `collect2`와 정확히 비교)가 없는 상태가 `--idle-secs` (기본 90)초 연속될 때까지 기다립니다.
2. 1 Hz monitor 아래에서 COMMAND를 실행합니다.
3. 어떤 sample에서든 COMMAND나 그 자손이 아닌 GPU process, 또는 compiler가 보이면 그 시도를 거부하고 1단계로 돌아갑니다.

모든 sample은 `--log`에 기록됩니다. 종료 상태는 첫 깨끗한 시도의 COMMAND 종료 상태이거나, 모든 시도가 경합했거나 `--max-wait`가 끝나면 75입니다. INT와 TERM은 COMMAND와 monitor를 멈춥니다. Driver는 종료 시 log에서 로컬 경로를 지웁니다.

커밋된 `benchmarks/rocm_profiles/gfx1151_929c80ab/guard.log`에 guard가 실제로 동작한 기록이 있습니다. 21:03:05의 granite greedy profiled 시도는 세 sample 동안 깨끗했고, sample 4에 두 번째 `mlxcel-bench-de` process (4초 뒤 시작된 다른 unit의 bench)가 나타났으며, sample 5가 `CONTENDED foreign_gpu=[2682448:mlxcel-bench-de]`를 기록했고, 시도는 `REJECTED (contended), exit 0; rerunning`으로 끝났습니다. Bench 자체는 0으로 종료했다는 점이 핵심입니다. 출력은 정상으로 보였고 guard가 없었다면 그대로 공개되었을 것입니다. 이후 idle streak가 몇 번 더 초기화된 뒤 21:15:18에 시도 2가 시작되어 `CLEAN, exit 0`으로 끝났습니다. 다른 실행의 시도 하나 (21:48의 Qwen3 `t0.7`)는 guard를 통과했지만, 이 unit의 CPU를 많이 쓰는 trace 분석이 함께 돌았기 때문에 (guard는 이를 감시하지 않습니다) 수동으로 폐기했고, log에 교체 사실이 남아 있습니다. 채택된 16개 실행은 19:36부터 22:14 KST까지 분포하며, 각 실행에서 모든 sample은 GPU process가 없거나 실행 자신만 있었고 compiler는 없었습니다. 재실행된 경우 `_bench.log`에는 두 시도가 모두 남고, 후처리기는 마지막 것만 읽습니다.

## 5. Attribution과 port 순서

### 5.1 Kernel 이름에서 port로

Kernel 이름은 primitive를 알려 줄 뿐, 그것을 요청한 mlxcel op를 알려 주지 않습니다. `binary_vv<Add>`는 residual join일 수도, SSD step의 일부일 수도, logit bias의 일부일 수도 있습니다. MLX는 매 decode step의 graph를 같은 순서로 평가하므로 op마다 같은 dispatch 묶음을 남기고, `assign_roles()`가 이 묶음에 역할을 붙입니다 (`sampler_tail`, `add_rms_join_post_attn`, `add_rms_join`, `rope_append`, `moe_expert_gemv`, `moe_activation`, `moe_weighted_sum`, `moe_gather_indices`, `ssm_step`, `ssm_conv`, `ssm_silu`, `ssm_gated_norm`). 규칙은 개수로 검증했습니다. Step당 Mamba2 layer마다 SSD dispatch가 granite에서 정확히 47개, Nemotron에서 50개이고 (두 모델의 `ssm_step` graph는 별개지만 평행한 코드가 만듭니다), Qwen3 layer마다 expert GEMV 3개, arange 3개, weighted-sum dispatch 4개이며, 모든 SSM과 MoE 역할의 개수가 layer 수의 정수배입니다. `forward_fused_kernel`이 `topk_indices`와 `scores`를 호출자에게서 받으므로 router top-k는 의도적으로 MoE에서 뺐습니다.

그다음 `reach()`가 `929c80ab` 소스를 기준으로, 배포 설정에서 mlxcel이 각 모델에 대해 실제로 그 port를 호출하는지를 반영합니다. 표의 각 칸은 닿는 비율이고, 괄호는 호출 여부와 무관하게 해당 port 계열이 덮는 fallback 비용입니다.

### 5.2 배포 설정에서 port별 비율

| 모델, 실행 | #2063 | #2064 | #2065 | #2067 | #2068 |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B, greedy | 0 (1.66) | 0 (0.08) | 0 | 0 | 0 |
| Llama 3.1 8B, t0.7 | 0 (1.65) | 0.37 | 0 | 0 | 0 |
| Llama 3.1 8B, t0.7 p0.95 | 0 (1.92) | 1.17 | 0 | 0 | 0 |
| Qwen3-30B-A3B, greedy | 0 (2.82) | 0 (0.17) | **46.79** | 0 | 0 |
| Qwen3-30B-A3B, t0.7 | 0 (2.77) | 0.69 | 46.52 | 0 | 0 |
| Qwen3-30B-A3B, t0.7 p0.95 | 0 (2.67) | 2.75 | 46.03 | 0 | 0 |
| granite-4.0-h-tiny, greedy | 0 | 0 (0.20) | 0 (25.42) | **29.82** | 0 |
| granite-4.0-h-tiny, t0.7 | 0 | 0.72 | 0 (22.51) | 30.75 | 0 |
| granite-4.0-h-tiny, t0.7 p0.95 | 0 | 2.51 | 0 (21.62) | 31.28 | 0 |
| Nemotron-3-Nano, greedy | 0 (0.26) | 0 (0.17) | 0 (29.22) | **20.01** | 0 |
| Nemotron-3-Nano, t0.7 | 0 (0.26) | 0.68 | 0 (27.48) | 20.28 | 0 |
| Nemotron-3-Nano, t0.7 p0.95 | 0 (0.25) | 3.81 | 0 (25.99) | 21.50 | 0 |

비율은 절감의 상한입니다. 그중 kernel이 얼마를 회수할 수 있는지가 순서를 정합니다.

1. **#2067 SSM update (항목 7).** Decode GPU 시간의 29.8% (granite)와 20.0% (Nemotron), 그리고 token당 dispatch 1679개와 1141개로, 각 모델 dispatch의 절반 이상이며 token당 5 ms host gap의 원천입니다. `ssm_kernel_available()`가 모든 single-token SSD step을 gate하므로 port는 호출됩니다. Kernel 하나의 메모리 traffic은 SSM state로, granite 기준 token당 약 113 MB, 180 GB/s에서 약 0.6 ms이고, 현재 graph는 3.42 ms (Nemotron 2.9 ms)를 씁니다. 대부분 회수 가능합니다.
2. **#2065 fused MoE decode (항목 5).** Qwen3 decode GPU 시간의 46.8%에 닿지만 (`qwen3_moe.rs:223`), 42.1%p가 181 GB/s로 도는 expert GEMV입니다. Fused kernel이 회수할 수 있는 것은 activation, weighted sum, index 생성 (4.7%, token당 0.62 ms)과 token당 dispatch 524개 중 약 430개입니다. Granite와 Nemotron은 현재 범위의 port가 닿지 않는 MoE fallback 비용 25~29%를 갖고 있습니다. Granite의 `block_sparse_moe`는 `forward_fused_kernel`이 아니라 `SwitchGLU::forward`를 부르고, Nemotron-H의 기본 분기는 `gather_qmm`입니다 (kernel 분기는 `MLXCEL_FUSED_MOE_RELU2`와 #2069가 필요합니다).
3. **#2064 sampler (항목 4).** Greedy decode에서는 0, temperature만 쓰면 0.4~0.7%, top-p를 쓰면 전체 vocabulary에 대한 `rocprim` radix sort와 scan이 매 token 돌아서 1.2~3.8%입니다. 이 수치에는 `--ignore-eos`가 더하는 logit-bias dispatch 몇 개가 포함되어 약간 높게 나옵니다.
4. **#2068 paged attention (항목 8).** 비율 없음. Bench는 한 sequence를 dense `KVCache`로 decode하며, 어떤 trace에도 paged kernel이나 paged fallback이 나타나지 않습니다.
5. **#2063 fused add-RMSNorm과 RoPE-append (항목 3).** 모든 backend에서 배포 설정으로는 0입니다. #905가 Metal에서 decode 이득이 없다고 측정한 뒤 `FUSED_ADD_RMSNORM_DEFAULT`와 `FUSED_ROPE_APPEND_DEFAULT`가 `false`이고 (`layers.rs:808`, `:820`), 이들을 부르는 것은 `llama3.rs` (그리고 profile하지 않은 `gemma.rs`, `iquestloopcoder.rs`)뿐입니다. Opt-in하면 Llama 3.1은 post-attention join 하나에만 닿아 0.83% (top-p 실행에서 0.89%)입니다. `rope_scaling`이 frequency table을 만들어 RoPE kernel을 우회하기 때문입니다 (`llama3.rs:669`).

**측정된 비율 없이 #2068이 #2063보다 앞서는 이유.** 둘 다 여기서는 0이므로, 동률은 데이터가 아니라 범위로 판정했습니다. #2068은 이 profile이 다루지 않는 실제 경로 (batched paged serving)와 그 port가 해소할 ROCm test skip 36개를 갖고 있습니다. #2063은 어느 backend의 기본 경로에도 없고, 누군가 opt-in해도 최선이 한 모델의 1% 미만입니다. 문서는 이 배치가 측정값으로 읽히지 않도록 이를 명시합니다.

**#1814 가설과의 비교.** 빈도 논리는 항목 3과 4가 모든 모델에서, 항목 7이 한 계열에서 돈다는 점에서는 맞았습니다. 그러나 profile이 바로 보여 주는 두 가지를 놓쳤습니다. 모든 곳에서 도는 kernel이라도 호출자가 꺼 둔 채 배포하면 아무 데도 닿지 않으며, token당 비용은 layer마다 47개 dispatch로 된 f32 graph (granite에서 token당 3.42 ms)와 add 하나에 norm 하나로 된 join (Llama decode의 0.83%, 약 0.19 ms) 사이에서 10배 이상 차이 납니다. #1814의 ranking comment는 새 순서 7, 5, 4, 8, 3을 이전 순서 3, 4, 5, 7, 8과 함께 기록합니다.

## 6. MLX의 ROCm `gather_mm`

각 test를 gfx1151에서 정확한 이름으로 실행했고, GPU 경로를 확인하려고 각각 rocprofv3 아래에서도 실행했습니다.

| Test | 결과 | Trace에 나온 kernel |
|---|---|---|
| `gather_mm_matches_dense_per_expert_reference` | 통과 | `gather_batched_gemm_kernel<float, false, true>` x4, `<float, false, false>` x1, Tensile `Cijk_*` GEMM x4 (hipBLASLt, 정렬된 single-row 경우) |
| `gather_mm_selects_the_indexed_expert` | 통과 | `gather_batched_gemm_kernel<float, false, false>` |
| `gather_mm_half_precision_matches_reference` | 통과 | `gather_batched_gemm_kernel<hip_bfloat16, ...>`, `gather_batched_gemm_kernel<__half, ...>` |

Overlay는 `GatherMM::eval_gpu` (`matmul.cpp`)를 직접 구현하며, 결과가 test 허용 오차 안에서 f64 host reference와 일치합니다. Negative control로 reference를 일부러 잘못된 expert로 향하게 하자 세 test 모두 값 assertion (`grouped_gemm_numeric_tests.rs:111`)에서 실패했으므로, 통과는 skip이 아니라 증거입니다. 이제 gate는 `gpu_backend_available()`를 읽고 (Metal과 CUDA에서도 그대로 실행됩니다), `BACKEND_ENUMERATION_TODO`는 비어 있으며, checker는 `0 awaiting a predicate`를 출력합니다. #1814가 맡았던 마지막 rule 4 예외가 이로써 닫힙니다.

## 7. 기술적 선택과 그 이유

- **Kernel 이름이 아니라 host clock으로 decode 구간을 자릅니다.** Warmup pass와 측정 pass는 같은 kernel을 실행하므로, kernel 이름만으로는 어떤 dispatch가 측정 decode에 속하는지 알 수 없습니다. Bench가 두 clock으로 출력하는 mark와 두 가지 청결 검사 (절단점에 걸친 kernel 없음, 첫 dispatch 전 device idle 구간)가 실행마다 절단을 검증 가능하게 만듭니다.
- **Step 안의 위치로 역할을 붙이고 개수로 고정합니다.** 역할 규칙은 각 op의 graph 순서를 기준으로 하며 layer당 정확한 dispatch 수로 검증했습니다. Kernel 이름만으로 귀속시키면 residual add와 SSD add가 같은 port로 계산되었을 것입니다.
- **배포 설정에서의 도달을 보고하고 fallback 비용은 괄호에 둡니다.** 아무도 부르지 않는 `.rocm` 칸을 채우는 port는 아무것도 아끼지 못합니다. "닿음"과 "덮음"을 나눈 덕분에 #2063이 괄호 속 큰 숫자가 아니라 0이 되고, granite의 연결되지 않은 MoE가 #2065의 이득이 아니라 후속 작업으로 드러납니다.
- **비율을 회수 가능성으로 가중합니다.** 원래 비율만으로 순위를 매기면 #2065가 1위입니다. GEMV에 대한 대역폭 계산이 그 비율 대부분이 회수 불가능함을 보여 주므로 #2067이 앞섭니다.
- **시간이 아니라 비율.** 가장 중요한 모델에서 profiler overhead가 23%에 이르므로, profiled 절대 시간은 dispatch가 많은 hybrid 쪽으로 순위를 치우치게 했을 것입니다.
- **Log를 남기는 script로서의 guard.** #2056의 guard는 절차였습니다. 종료 코드와 sample별 log가 있는 script로 만들어, 공개된 모든 실행을 감사할 수 있고 before/after 수치가 필요할 port PR들이 재사용할 수 있습니다.
- **Gate를 넓히기 전에 `gather_mm` test를 negative control과 함께 실행합니다.** #1806의 교훈을 그대로 따른 것입니다. Gate 변경은 trace와, 반드시 실패해야 하는 test 변형으로 뒷받침됩니다.

## 8. 검증

PR 본문 기준, rebase 전 gfx1151에서:

- `cargo test --release --features rocm -p mlxcel-core --lib grouped_gemm_numeric_tests -- --test-threads=1 --nocapture`: 3개 통과. 각각 rocprofv3 아래에서 정확한 이름으로 재실행. 잘못된 expert reference로 세 개 모두 실패.
- `python3 scripts/ci/check_kernel_port_dispatch.py`: `0 awaiting a predicate`. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` 통과.
- `cargo clippy -p mlxcel --features rocm --bin mlxcel-bench-decode -- -D warnings`와 `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`: 경고 없음.
- `cargo test --features rocm --test dead_doc_pointers`: 통과. `python3 -m unittest tests/test_rocm_decode_profile.py`: 17개 통과. Script 세 개에 `bash -n`.

`f9aefa39` 위로 rebase한 뒤 orchestrator 검증:

- merge-resolver가 PR #2084의 phase별 `[Memory]` 출력과의 `src/bin/bench_decode.rs` 충돌을 해결했고 두 동작을 모두 유지했습니다.
- Bench binary clippy는 경고 없음, `tests/test_rocm_decode_profile.py`는 17개 통과, bench harness Python test는 35개 통과.
- Rebase된 bench의 기능 실행에서 `[Memory]` 출력과 phase mark 네 개가 순서대로 모두 출력되었습니다.
- 전체 `make verify-rocm`은 orchestrator가 rebase된 head에서 실행했습니다.

## 9. 남은 위험과 검증하지 않은 부분

- **장치 하나, 세션 하나.** 모든 수치는 `929c80ab`, overlay `75915908`의 gfx1151 결과입니다. 다른 RDNA나 CDNA 장치, 다른 overlay, 향후 `qmv` 변경은 비율을 바꿀 수 있습니다. 커밋된 script로 profile을 재현할 수 있습니다.
- **Overhead 수치는 단일 실행입니다.** 모델당 plain 한 번과 profiled 한 번입니다. Llama 값 (profiled가 6.8% 빠름)은 noise이고, hybrid의 20~23%에는 편차 정보가 없습니다.
- **Guard의 사각지대.** Sampling이 1 Hz이므로 1초 미만의 GPU 작업은 sample 사이로 빠질 수 있고, compiler가 아닌 process의 CPU 부하는 감시하지 않습니다 (Qwen3 `t0.7` 재실행은 수동으로 잡은 것입니다).
- **비율은 speedup이 아니라 상한입니다.** ROCm에서는 이 경로들 중 어느 것도 전환할 kernel이 없으므로 A/B를 실행하지 않았습니다. 각 port PR이 자신의 before/after를 측정해야 합니다.
- **Attribution은 규칙 기반입니다.** 규칙은 layer당 개수와 합성 step test로 고정되어 있지만, graph 순서가 여기서 profile한 네 모델과 다른 모델은 별도 확인이 필요합니다. `reach()`는 `929c80ab`의 소스를 반영하므로, 이후 `FUSED_*_DEFAULT`, `block_sparse_moe`, `fused_moe_forward`가 바뀌면 도달 열이 바뀝니다.
- **#2068은 측정하지 않았습니다.** Batched paged serving은 이 harness 밖에 있으며, 그 순위는 데이터가 아니라 범위에 근거합니다.
- **여기서 검증하지 않은 것:** Metal과 CUDA (넓어진 `gather_mm` gate는 그곳에서도 그대로 실행됩니다). `--features rocm` 없이 `cargo test --test dead_doc_pointers`를 실행하면 이 호스트에서 link가 실패하며 (`kv_inplace_write.cpp`에서 `copy_gpu_inplace` 미정의), 이 변경과 무관합니다.

## 10. 학습 포인트

- **"어디서나 돈다"가 "가장 비싸다"는 뜻은 아닙니다.** 호출 빈도는 fused 경로가 켜져 있는지, 호출 한 번이 얼마나 일하는지를 무시합니다. 도달 여부를 반영한 profile 한 번이 빈도 가설이 뒤집어 놓은 순서를 바로잡았습니다.
- **비율은 회수 가능성 추정을 거쳐야 우선순위가 됩니다.** Port가 대체하는 kernel에 대한 대역폭 계산이, 여전히 해야 하는 일과 사라질 수 있는 overhead를 구분합니다.
- **Profiler를 측정하십시오.** Dispatch 수에 비례하는 tracing 비용은 바로 dispatch가 많은 모델을 왜곡하므로, profiler 없는 실행은 부가 작업이 아니라 방법의 일부입니다.
- **Guard는 무언가를 거부할 때 신뢰를 얻습니다.** 커밋된 log에는 0으로 종료한 실행을 실제로 거부한 기록이 있고, 이는 깨끗한 sample만 있는 log보다 나머지 16개 실행에 대한 더 강한 증거입니다.
- **Gate는 negative control과 함께 넓히십시오.** 통과한 test와 trace는 GPU 경로가 실행되었음을 보여 주고, 일부러 틀린 reference가 실패하는 것은 test가 구분할 수 있음을 보여 줍니다.

Refs: #2061, #1814, #1801, #2056, #2062, #2063, #2064, #2065, #2067, #2068, #2069, #2084, #1806, #905.
