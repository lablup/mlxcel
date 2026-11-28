# 기술 보고서: PR #2099 - ssm_update_kernel을 HIP으로 port하고 port table을 읽도록 함

**날짜**: 2026-10-04

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `2f17f286`, origin/main `c0b71344` 위, 머지 대기 중.

**언어**: C++ (HIP kernel source, kernel holder, support predicate, host 측 shape 검사), Rust (FFI doc comment, parity test), Markdown (환경 변수, 설치, benchmark 결과), CSV/TSV (bench 행, logit trace)

**위험도**: 중간 (ROCm에서 hybrid SSM 모델의 기본 single-token decode 경로가 바뀝니다. Metal과 CUDA가 공유하는 predicate, template argument, host 검사도 건드리며, 두 backend는 개발 호스트에서 실행할 수 없었습니다)

## 요약

Issue #2067 (#1814의 일부, epic #1801)은 PR #2086이 측정한 순서의 첫 번째 port입니다. 대상은 single-token Mamba2 SSM update이며, decode profile은 gfx1151에서 이것이 granite-4.0-h-tiny decode GPU 시간의 29.8%, Nemotron-H의 20.0%를 차지한다고 보고했습니다. 이 PR 이전에는 ROCm에서 그런 step마다 약 55개 op로 된 SSD graph (`ssm_step`)가 실행되었습니다. `ssm_ports()`에는 `.rocm` 항목이 없었고, `ssm_kernel_available()`은 table을 전혀 읽지 않았으므로 slot을 채워도 도달할 수 없었습니다.

이 PR은 `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`에서 네 가지를 합니다. CUDA kernel 본문에서 lane fold만 `__shfl_down(acc, o, 32)`로 바꾼 `SSM_HIP_SOURCE`를 추가하고 `.rocm`을 채웁니다. `ssm_kernel_available()`이 모든 플랫폼에서 `has_kernel_port(ssm_ports())`를 반환하게 하고, 새 `MLXCEL_SSM_KERNEL=0` 스위치를 두며 예전 `MLXCEL_SSM_CUDA_KERNEL=0`은 alias로 남깁니다. Nemotron-H가 `A_log`를 bf16 activation 옆에 f32로 저장하고 JIT cache가 template argument로 key를 만들기 때문에, launch의 template argument에 `A_log`, `B`, `C`의 dtype을 모든 backend에서 추가합니다. Kernel이 index할 수 없는 shape은 host에서 error로 거부합니다.

gfx1151에서 같은 binary로 `MLXCEL_SSM_KERNEL=0`을 "이전" arm으로 두고 측정한 결과, decode가 granite-4.0-h-tiny-4bit에서 60.39에서 88.43 tok/s (1.46x), NVIDIA-Nemotron-3-Nano-30B-A3B-4bit에서 51.35에서 74.37 tok/s (1.45x)로 올랐습니다 (세 번 실행의 중앙값). 모든 traced step이 512 token SSM state를 만나는 `w1ctx512` logit trace는 이제 ROCm kernel과 M5의 Metal kernel을 비교하며, decided position 불일치는 0 / 60 (granite), 0 / 71 (Nemotron-H)입니다. Kernel 유무에 따른 greedy 128 token 출력은 두 모델 모두 자연스럽지만 byte 단위로 같지는 않으며, trace는 모든 갈림이 undecided position에서 일어났음을 보여 줍니다.

이 issue 범위를 넘는 발견이 하나 있습니다. 모든 #1814 port가 갖추도록 요구되는 wave32 `#error` guard는 ROCm 10의 AMD clang 23에서 아무 효과가 없습니다. 이 컴파일러는 gfx1151, gfx942, gfx90a 어느 것에도 `__AMDGCN_WAVEFRONT_SIZE__`나 `__AMDGCN_WAVEFRONT_SIZE`를 정의하지 않습니다. 이 port는 guard에 의존하지 않지만, 이후 port에 대해서는 이 관용구가 아무것도 막아 주지 않습니다.

## 1. 문제 정의

### 1.1 묻지 않는 predicate 뒤의 빈 slot

Hybrid SSM 모델 (granite-4.0-h, falcon-h1, plamo-2, Nemotron-H)은 `granitemoehybrid.rs`, `falcon_h1.rs`, `plamo2.rs`, `nemotron_h.rs`에서 fused single-token 경로를 `seq_len == 1 && ssm_kernel_available()`로 gate하고, 그 뒤 launch를 `expect`합니다. 이 PR 이전의 predicate는 다음과 같았습니다.

```cpp
#ifdef __APPLE__
    return mlx::core::metal::is_available();
#else
    // MLXCEL_SSM_CUDA_KERNEL=0 forces the graph path
    ...
    return mlx::core::cu::is_available();
#endif
```

ROCm에서는 false가 나와 gate가 graph로 갔습니다. `.rocm`만 채웠다면 아무것도 바뀌지 않았을 것이고, predicate가 backend를 나열하는 형태로 남은 채 slot을 채우는 것은 `verify-kernel-port-dispatch`가 막으려는 바로 그 모양, 즉 predicate와 dispatch가 서로 다르게 답할 수 있는 상태입니다.

### 1.2 Profile이 말한 가치

PR #2086은 gfx1151에서 SSM update를 #1814 port 중 회수 가능한 비율이 가장 큰 항목으로 측정했습니다. granite-4.0-h-tiny decode GPU 시간의 29.8%, Nemotron-H의 20.0%, 그리고 두 모델 dispatch의 절반 이상입니다. CUDA port (#631)는 GB10에서 granite/falcon 계열 decode를 2.6~4.5x 올렸습니다.

### 1.3 Kernel 단위 parity test가 없었음

CUDA port의 parity는 CUDA에서의 end-to-end greedy 출력이었습니다. 같은 입력으로 kernel과 graph step을 비교하는 것이 없었으므로, 유한하고 그럴듯한 값을 내는 reduction 버그 (#1814의 wave32 규칙이 설명하는 실패)는 모델 출력이 서서히 어긋나는 형태로만 드러났을 것입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlx_cxx_kernels.cpp`, kernel | `SSM_HIP_SOURCE`, `MLXCEL_BRIDGE_ROCM_BACKEND`에서 `fast::hip_kernel`을 부르는 `SsmKernelHolderHip` / `get_ssm_kernel_hip()`, `ssm_ports()`의 `.rocm` getter |
| `mlx_cxx_kernels.cpp`, predicate | `ssm_kernel_available()`: kill switch, 그다음 GPU device가 아니면 false, 그다음 `has_kernel_port(ssm_ports())` |
| `mlx_cxx_kernels.cpp`, launch | 모든 backend에서 `TA`, `TB`, `TC` template argument (`A_log`, `B`, `C`의 dtype) |
| `mlx_cxx_kernels.cpp`, host | `ssm_update_kernel`에서 launch 전에 `std::invalid_argument`를 던지는 rank와 shape 검사 |
| `ssm_update_parity_tests.rs` (신규) | Test 다섯 개: 세 shape에서 f32와 bf16 parity, `A_log` dtype key, shape 거부, 두 kill switch |
| `mlx_cxx_bridge.h`, `lib.rs` | predicate와 kernel의 doc comment, test module 등록 |
| 문서 | `docs/environment-variables.md`의 `MLXCEL_SSM_KERNEL` 행과 alias 행, `docs/installation.md`의 ROCm 행, `rocm-correctness-gfx1151-2026-09-30.md`의 "Open" 항목이 kernel 대 kernel 행을 가리키도록 수정, 새 `rocm-ssm-update-kernel-gfx1151-2026-10-04.md` |
| 데이터 | `benchmarks/rocm_strixhalo-gfx1151_2026-10-04_ssm-kernel-{off,on}.csv`, `benchmarks/logit_traces/rocm_gfx1151_96cbce84/`의 trace 여섯 개와 `METADATA.txt`, `RUNS.txt`, `SHA256SUMS`, `README.md` |

Branch의 commit은 여섯 개입니다. port와 predicate (`96cbce84`), 모든 backend의 dtype key (`844bd94c`), 문서 (`3780b0b2`), CPU device 검사와 review 수정 (`bd2e8f31`), host 측 shape 거부 (`38071038`), 추가 거부 test와 test 문서 (`2f17f286`).

## 3. Port

### 3.1 CUDA 본문, HIP fold

HIP source는 CUDA port의 kernel 이름 형식, 입력 (`X`, `A_log`, `B`, `C`, `D`, `dt`, `state_in`), 출력 (`out`, `state_out`), grid, template argument를 그대로 유지합니다. (32, 8, 1) threadgroup에서 `threadIdx.y`가 같은 32개 lane이 (head, row) 한 쌍을 reduce합니다. 각 lane은 `Ds / 32`개의 state 원소를 돌며 state를 갱신하고 `state * C`를 누적합니다. 본문에서 바뀐 것은 fold뿐입니다.

```cpp
for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down(acc, o, 32);
}
```

HIP에서 `__shfl_down_sync`는 mask를 무시하는 호환용 shim이므로 native `__shfl_down`을 쓰고 width를 명시합니다. Wave 크기와 무관하게 fold가 맞는 이유가 이 width입니다. Wave32 target (gfx11, gfx12)에서는 32개 lane이 wave 전체이고, wave64 target (CDNA)에서는 width 32가 wave를 32 lane 두 구간으로 나누며 그것이 정확히 row 두 개입니다. `Dh`를 넘는 row의 early return은 wave의 32 lane이 모두 같은 `threadIdx.y`를 가지므로 wave-uniform입니다.

### 3.2 Wave32 guard와, 여기서 그것이 아무 일도 하지 않는 이유

#1814 port 요구사항은 lane 간 reduce를 하는 모든 port에 bitlinear port처럼 `__AMDGCN_WAVEFRONT_SIZE__`와 `__AMDGCN_WAVEFRONT_SIZE`에 대한 `#error` 검사 두 개를 넣도록 합니다. 이 port도 넣었습니다. 작업 중에 ROCm 10의 AMD clang 23이 gfx1151, gfx942, gfx90a 어디에도 두 macro를 정의하지 않는다는 것이 드러났습니다. 따라서 `#if defined(...)` 조건은 모든 target에서 false이고 guard는 발동할 수 없습니다. `SSM_HIP_SOURCE`의 주석과 benchmark 문서 모두 이 사실을 적고 있습니다. 위 fold가 wave64에서 맞는 것은 guard 때문이 아니라 width 논리 때문입니다. 남은 port에 대한 의미는 9절에 있습니다.

## 4. Predicate와 스위치

`ssm_kernel_available()`은 이제 다음 순서로 판단합니다.

1. `MLXCEL_SSM_KERNEL=0` 또는 `MLXCEL_SSM_CUDA_KERNEL=0`이면 false. 새 이름은 모든 backend에서 스위치가 하는 일을 그대로 말하고, 예전 이름은 기존 A/B script가 계속 동작하도록 남깁니다.
2. 기본 device가 GPU가 아니면 false. `mamba1_scan_kernel_accepts`와 같은 검사입니다. Custom kernel은 GPU stream에서만 실행되므로, `MLXCEL_DEVICE=cpu`에서는 launch가 throw하는 대신 gate가 CPU에서 `ssm_step`을 탑니다.
3. 그 외에는 `mlxcel::has_kernel_port(ssm_ports())`. Dispatch에서 `select_kernel_port`가 읽는 것과 같은 table입니다. 그래서 true라는 답은 dispatch가 거부하지 않는다는 뜻이고, Rust gate가 launch를 `expect`할 수 있는 근거가 됩니다.

다른 backend에는 두 가지 동작 변화가 생기며 문서에 둘 다 적혀 있습니다. Alias가 이제 Metal과 ROCm에도 적용되고 (예전에는 non-Apple 경로에서만 읽었습니다), Metal에서도 `MLXCEL_SSM_KERNEL=0`이 반영됩니다. 이전에는 Metal에서 kernel을 끌 방법이 없었습니다. 환경 변수가 없으면 Metal과 CUDA에서 `has_kernel_port`는 이전처럼 true입니다.

모델 파일은 바뀌지 않았습니다. Nemotron-H에서 `NemotronHMamba2Mixer::forward`의 gate는 `fused_mamba2_forward`를 선택하는데, 이것은 single-token mixer 전체 (input projection, convolution, 이 kernel, gated norm, output projection)를 C++ 호출 하나로 실행합니다. 따라서 Nemotron-H의 decode 수치는 SSM step만이 아니라 이 경로 전체를 Rust graph mixer와 비교한 것입니다.

## 5. 모든 backend의 dtype key

CUDA와 HIP의 JIT cache는 kernel 이름과 `template_args`로 module의 key를 만들지만, 생성되는 kernel signature는 각 입력의 runtime dtype을 받습니다. Launch는 `T` (activation, `X`와 `D`도 포함)와 `U` (state)만 이름에 넣었습니다. Nemotron-H는 `A_log`를 bf16 activation 옆에 f32로, granite는 bf16으로 저장하므로, 두 모델이 같은 cached module에 닿는 곳 (parity test처럼 한 process 안)에서는 먼저 compile한 쪽이 다른 쪽이 `A_log`를 읽을 pointer type을 정해 버립니다. `B`와 `C`도 `T`에 묶여 있지 않습니다.

수정은 `A_log`, `B`, `C`의 dtype으로 `TA`, `TB`, `TC`를 추가합니다. Kernel 본문은 이 이름을 쓰지 않으므로 바뀌는 것은 module 이름뿐이고 연산은 같습니다. 이것은 Metal을 포함한 모든 backend에 추가됩니다 (Metal은 이미 이름에 모든 입력 dtype을 넣으므로 중복입니다). 첫 버전 (`96cbce84`)은 Metal과 CUDA module 이름을 바꾸지 않으려고 `if (mlxcel::gpu_kernel_backend() == mlxcel::GpuKernelBackend::Rocm)` 안에서 추가했습니다. 이것은 `scripts/ci/check_kernel_port_dispatch.py`의 규칙 1이 금지하는 형태입니다. Custom kernel을 launch하는 파일은 `kernel_port.cpp`와 `gpu_backend.cpp` 밖에서 backend 종류로 분기하면 안 됩니다. `select_kernel_port`가 생기기 전 launcher들이 서로 어긋난 원인이 손으로 쓴 backend별 분기였기 때문입니다. 두 번째 commit (`844bd94c`)이 key를 모든 backend로 옮겼습니다. Trace는 `96cbce84`에서 만들었으며, ROCm에서는 두 commit의 launch, argument, cache key가 같다는 것을 `METADATA.txt`가 기록합니다.

`ssm_update_kernel_keys_on_a_log_dtype`은 같은 shape을 한 process에서 `A_log` bf16으로 먼저, f32로 다음에 실행합니다. Key가 없으면 두 번째 launch가 bf16 module을 재사용하고 test는 normalized RMS 0.80으로 실패합니다.

## 6. Host 측 shape 거부

Kernel은 `Dh`, `Ds`, `H`, `G`를 template 상수로 받아 raw buffer를 index합니다. Graph 경로라면 `reshape`이나 `repeat`에서 거부했을 checkpoint 설정이, kernel에서는 GPU memory를 잘못 읽거나 쓰지 않은 채 남깁니다. `ssm_update_kernel`은 이제 다음 경우 launch 전에 `std::invalid_argument`를 던집니다.

- `hidden_states`나 `B`가 rank 4가 아님
- `C`의 shape이 `B`와 다름
- token이 하나보다 많거나, `B`의 batch가 activation과 다름
- group이 양수가 아니거나 head가 group으로 나누어지지 않음
- state 폭이 32 미만이거나 32의 배수가 아님
- `A_log`, `D`, `dt`, `state_in`의 원소 수가 shape과 맞지 않음

Bridge는 이 throw를 Rust에 error로 돌려줍니다. 검사는 공유 launch 앞에 있으므로 모든 port에 적용되며, HIP port가 ROCm에서 이 경로를 도달 가능하게 만든 것입니다. `ssm_update_kernel_refuses_unsupported_shapes`가 이를 고정합니다.

## 7. 측정 결과

### 7.1 Decode 처리량

`scripts/bench_decode.sh`, pp512/tg128, 실행당 모델 하나, arm마다 세 번, arm 순서를 번갈아, `844bd94c`, 같은 binary.

| 모델 | Graph (`MLXCEL_SSM_KERNEL=0`) tok/s | Kernel tok/s | 속도 향상 (중앙값) |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 60.57 / 60.39 / 60.05, 중앙값 60.39 | 88.36 / 88.69 / 88.43, 중앙값 88.43 | 1.46x |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 51.65 / 50.87 / 51.35, 중앙값 51.35 | 74.87 / 72.87 / 74.37, 중앙값 74.37 | 1.45x |

실행 간 편차는 두 arm 모두 granite 1% 미만, Nemotron-H 3% 미만입니다. 모델별 마지막 행을 쓰는 `compare_bench_csv.py --before <off> --after <on>`은 1.47x와 1.45x를 보고합니다. Prefill은 이 경로에 있지 않아 비교하지 않았고, 공유 호스트에서 granite prefill은 두 arm에 걸쳐 379~608 tok/s 사이로 변했습니다.

향상 폭은 GB10의 CUDA port 2.6~4.5x보다 작습니다. 동시에 decode profile의 GPU 시간 비율만으로 예상할 수 있는 것보다는 큽니다. 29.8%나 20.0% 비율을 통째로 없애면 GPU 시간 기준 약 1.42x와 1.25x입니다. 다만 profile은 다른 commit에서, 이 모델들에서 그 자체로 20~23%의 비용이 드는 profiler 아래에서 측정되었고, Nemotron-H의 fused 경로는 SSM step이 아니라 mixer 전체를 대체합니다. 그러므로 이것은 공개된 비율과의 일관성 점검이지 속도 향상의 분해가 아닙니다. 초과분을 설명하는 측정은 이 PR에 없습니다 (약 55개 op graph의 host 측 dispatch가 가장 유력한 후보입니다).

### 7.2 Idle-GPU guard

모든 실행은 `scripts/rocm_gpu_guard.sh`를 거쳤습니다 (`/sys/class/kfd/kfd/proc`가 비어 있고 compiler가 없는 상태로 90초, 그다음 1 Hz 감시). 병렬 개발 unit이 GPU를 공유했으며, 그 작업과 겹친 시도는 모두 거부되어 다시 실행했습니다. 한 번을 제외한 모든 실행은 #2098에서 고친 guard (commit `c3eab7b2`)를 #2065 unit의 worktree에서 실행했으며, 이 PR에는 커밋되지 않았습니다. 그 수정은 guard가 명령 자신의 종료된 자식 process를 이유로 거부하던 문제를 없앱니다. 예외는 첫 실행 (granite, graph arm, r1)으로, `main`의 guard를 사용했습니다. 예전 guard의 결함은 잘못된 거부만 만들고 잘못된 수용은 만들지 않았으므로, 그 실행의 수용은 유효합니다.

### 7.3 Kernel 대 graph, 한 step

`ssm_update_parity_tests`는 출력과 새 state를 one-token SSD step의 float32 MLX-op reference와 비교합니다.

| Shape | Batch | Head 수 x head 차원 | Group | State |
|---|---|---|---|---|
| granite-4.0-h-tiny | 1 | 48 x 64 | 1 | 128 |
| Nemotron-H (dt를 `(1e-3, 0.1)`로 clip) | 2 | 64 x 64 | 8 | 128 |
| padded row | 1 | 4 x 60 | 2 | 64 |

허용 오차 (normalized RMS / max)는 issue에서 정한 f32 1e-5 / 1e-4, bf16 1.6e-2 / 7e-2입니다. 모델이 state를 f32로 들고 다니므로 state는 모든 경우에 f32 기준을 적용합니다. Gfx1151에서 통과합니다. 두 가지 변형으로 test가 구별력을 가진다는 것을 보였습니다. Lane fold를 16 대신 8에서 시작하면 normalized RMS 0.65~0.89로 실패하고, dtype key를 빼면 0.80으로 실패합니다. Test가 일찍 반환하는 것은 GPU backend가 없을 때, 기본 device가 CPU일 때, kill switch가 설정되었을 때뿐입니다. Metal, CUDA, ROCm에서 predicate가 false이면 skip이 아니라 실패입니다.

### 7.4 모델 logit: kernel 대 kernel (`w1ctx512`)

`w1ctx512` trace는 corpus token 512개를 prefill한 뒤 single-token step을 trace하므로, 모든 traced step이 state를 만나 fused step을 탑니다. 이 PR 전까지 `rocm-correctness-gfx1151-2026-09-30.md`의 해당 행은 Metal kernel과 ROCm graph를 비교했습니다. 이제는 kernel과 kernel을 비교합니다. `compare_logit_traces.py --decided 2.0`, Nemotron-H 파일은 `[NemotronH]` loader 행을 걸러 냈습니다.

| 모델 | Reference | Candidate | Top-1 불일치 | Decided 불일치 | 최대 gap | Perplexity ref / cand |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 60 | 0.125 | 7.090 / 7.216 |
| granite-4.0-h-tiny | Metal kernel | ROCm graph (`ssmkernel0`) | 2 / 128 | 0 / 60 | 0.125 | 7.090 / 7.290 |
| granite-4.0-h-tiny | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 59 | 0.000 | 7.290 / 7.216 |
| granite-4.0-h-tiny | `3c9edea0`의 ROCm graph | ROCm kernel | 1 / 128 | 0 / 60 | 0.125 | 7.234 / 7.216 |
| nemotron-3-nano-30b-a3b | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 71 | 0.250 | 7.489 / 7.442 |
| nemotron-3-nano-30b-a3b | Metal kernel | ROCm graph (`ssmkernel0`) | 4 / 128 | 0 / 71 | 0.250 | 7.489 / 7.460 |
| nemotron-3-nano-30b-a3b | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 68 | 0.125 | 7.460 / 7.442 |
| nemotron-3-nano-30b-a3b | `3c9edea0`의 ROCm graph | ROCm kernel | 4 / 128 | 0 / 68 | 0.125 | 7.489 / 7.442 |

모든 조합에서 decided position 불일치는 0이고, 모든 top-1 불일치에서 reference의 token은 candidate의 2위나 3위이며 reference gap은 0.25 이하입니다. 같은 binary의 `default`와 `ssmkernel0` trace가 서로 다르다는 점 (perplexity가 다르고 각각 token 하나가 뒤집힘)이 `default` trace가 실제로 kernel을 탔다는 증거입니다. Metal kernel 기준으로 ROCm kernel은 ROCm graph만큼 가깝습니다. Top-1 불일치가 3과 3인데, 여기서 graph는 2와 4, 예전 `3c9edea0` 비교에서는 4와 7이었습니다.

`w1` trace (prefill 없음)는 state를 만나지 않으므로 kernel 유무와 관계없이 graph를 탑니다. `rocm_gfx1151_c5fe9a16`과 비교하면 두 모델 모두 모든 position에서 일치합니다 (granite top-1 0 / 128, decided 0 / 116, Nemotron-H top-1 0 / 128, decided position 없음). 이것은 issue의 `w1` 기준을 충족하고 stateless 경로가 바뀌지 않았음을 보이지만 kernel 자체에 대해서는 아무것도 말하지 않습니다. 그래서 정확성 논거는 `w1ctx512` 행이 담당합니다.

### 7.5 Greedy 출력의 갈림

`mlxcel generate -p "The history of the Roman Empire" -n 128 -t 0 --no-chat-template`을 `MLXCEL_SSM_KERNEL=0` 유무로 실행했습니다. 두 모델의 두 arm 모두 자연스러운 글을 내지만 byte 단위로 같지는 않으며, GB10의 CUDA port와는 다릅니다. Granite는 세 번째 생성 token에서 갈리고 ("a rich and complex tapestry" 대 "a fascinating and complex subject"), Nemotron-H는 열두 token 정도 뒤에 갈립니다. 자유 생성은 첫 갈림 이후 모든 것이 다른 글에 조건화되므로, kernel이 틀렸는지 모델이 망설였는지를 구별하지 못합니다. Teacher-forced trace가 그 답입니다. Kernel과 graph는 모델마다 top-1 token 하나에서만 다르고, decided position에서는 한 번도 다르지 않습니다. 갈림은 합산 순서 변화가 near-tie를 만난 결과이지 오류가 아닙니다.

## 8. 기술적 선택과 그 이유

- **Metal이 아니라 CUDA 본문을 port.** CUDA와 HIP은 launch 형태, grid, `template_args`를 공유하고, `fast::hip_kernel`은 `cuda_kernel`처럼 hipRTC로 compile합니다. Fold를 빼고 본문을 한 줄씩 같게 두면 둘이 다를 때 볼 곳이 하나뿐입니다.
- **Guard에 기대지 않고 shuffle width를 명시.** Width 32 fold는 구조상 wave32와 wave64 모두에서 맞습니다. 요구사항이 정한 guard가 ROCm 10의 어느 target에서도 발동하지 않으므로 이것이 실제로 중요했습니다.
- **Predicate가 backend가 아니라 table을 읽게 함.** `has_kernel_port(ssm_ports())`는 `select_kernel_port`와 다르게 답할 수 없고, 앞으로 slot을 채우는 port는 predicate 수정 없이 도달 가능해집니다. `#ifdef __APPLE__` 분기와 `cu::is_available()` 호출이 없어졌고, issue의 첫 acceptance criterion이 이를 확인합니다.
- **스위치 이름을 바꾸고 alias는 유지.** `MLXCEL_SSM_KERNEL`은 이제 모든 backend에 작용하는 스위치를 그대로 설명하고, `MLXCEL_SSM_CUDA_KERNEL`을 남겨 기존 A/B script가 깨지지 않게 합니다. 대가는 CUDA라는 이름의 변수가 이제 Metal과 ROCm에서도 kernel을 끈다는 것이며, 문서에 적혀 있습니다.
- **ROCm 분기가 아니라 모든 backend에 dtype key.** Launch에서의 backend 분기가 첫 시도였고 `verify-kernel-port-dispatch`에 거부되었습니다. 모든 곳에 key를 추가하는 것은 Metal에서 무해하고, 같은 변경으로 CUDA의 같은 key 공백도 메웁니다.
- **잘못된 shape은 모든 port에 대해 host에서 거부.** Graph 경로는 이런 shape에서 시끄럽게 실패하지만 kernel은 조용히 실패합니다. `std::invalid_argument`를 던지면 backend별 조건 없이 시끄러운 실패가 돌아옵니다.
- **CPU device 검사를 predicate에 둠.** Mamba1 scan과 같은 규칙이고, `MLXCEL_DEVICE=cpu`를 동작하는 설정에서 error로 바꾸지 않고 graph 경로에 둡니다.

## 9. 후속 작업: wave64 guard 관용구는 ROCm 10 clang에서 무력함

#1814 port 요구사항은 `#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32`와 `#error`, 그리고 `__AMDGCN_WAVEFRONT_SIZE`에 대한 같은 검사를 요구합니다. 32 lane을 가정하는 port가 wave64 CDNA에서 유한하지만 틀린 결과를 내는 대신 compile에 실패하게 하려는 것입니다. ROCm 10의 AMD clang 23은 gfx1151, gfx942, gfx90a 어디에도 두 macro를 정의하지 않으므로, 두 조건이 모두 false이고 guard는 이 중 어디에서도 발동하지 않습니다.

현재 정확성에는 영향이 없습니다. `mlx_cxx_kernels.cpp`에서 lane 간 reduce를 하는 HIP port 두 개 (bitlinear와 이 port)는 모두 `__shfl_down`에 width 32를 명시하고, 이는 wave64에서도 맞습니다. 문제는 앞으로 올 port (#2065, #2064, #2068, #2063, #2069)입니다. 32 lane wave에 정확성이 실제로 의존하는 port가 CDNA에서 아무 문제 없이 compile되고, 요구사항 문구는 reviewer가 보호받고 있다고 믿게 만듭니다. 요구사항에는 ROCm 10에서 실제로 발동하는 장치가 필요하거나, 모든 lane 간 연산이 width를 명시하도록 하는 규칙으로 바꾸고 compile 시점 guard를 주장하지 않아야 합니다. 이 PR은 대체 방법을 정하지 않으며, 이 보고서를 쓰는 시점에 이에 대한 issue는 찾지 못했습니다.

## 10. 검증

PR 본문 기준, gfx1151 (Radeon 8060S, ROCm 10.0.0)에서:

- `38071038`에서 `make verify-rocm`: OK. Test binary 146개에 걸쳐 11,856 passed, 0 failed, 378 ignored. ROCm smoke OK.
- `2f17f286`에서 `cargo test --release --features rocm -p mlxcel-core --lib ssm_update_parity_tests -- --test-threads=1` (gate 이후 test와 문서만 변경): 5 passed.
- `cargo clippy -p mlxcel-core --lib --tests --features rocm -- -D warnings`: clean. `make verify-fmt verify-kernel-dtype-keys verify-kernel-port-dispatch`: pass, `EXPECTED_IN_SCOPE`와 그 개수는 그대로.
- CPU device와 shape 검사 전후로 두 모델의 greedy 출력이 동일하므로, `844bd94c`와 `96cbce84`에서 측정한 decode와 trace 수치가 head에도 적용됩니다.

Orchestrator 검증, head `2f17f286` (origin/main `c0b71344`과 최신 상태):

- `make verify-rocm`이 모든 단계를 통과: 11,856 tests passed, 0 failed, 378 ignored, smoke OK.

## 11. 남은 위험과 검증하지 않은 부분

- **Metal과 CUDA는 실행하지 않음.** 둘 다 predicate (여전히 `has_kernel_port`이고 둘 다 true, 여기에 CPU device와 kill switch 규칙), 추가된 `TA`/`TB`/`TC` template argument (module 이름만), host shape 검사의 영향을 받습니다. Parity test는 둘 다에서 실행되도록 작성되었고, 거기서 predicate가 false이면 skip이 아니라 실패합니다.
- **Alias의 범위가 넓어짐.** Metal이나 ROCm 머신에서 `MLXCEL_SSM_CUDA_KERNEL=0`을 설정해 두고 잊은 사용자는 이제 그곳에서 graph 경로를 탑니다.
- **Hybrid SSM 모델로 `MLXCEL_DEVICE=cpu`는 실행하지 않음.** 공유 test binary 안에서 기본 device를 바꾸면 다른 test가 CPU로 옮겨지므로 CPU device 규칙을 다루는 test는 없습니다.
- **falcon-h1과 plamo-2는 ROCm에서 실행하지 않음.** 같은 gate를 타지만 측정한 것은 granite와 Nemotron-H뿐입니다.
- **장치 하나, 세션 하나, 공유 GPU.** 모든 수치는 gfx1151에서 나왔고, 개발 unit 하나가 GPU를 공유했으며 겹친 시도는 다시 실행했습니다. 공유 호스트에서 prefill은 크게 변해 비교하지 않았습니다. Wave64 하드웨어가 없었으므로 CDNA에 대한 width 논리는 실행이 아니라 추론입니다.
- **측정용 guard는 이 PR에 없음.** 한 번을 제외한 모든 실행은 이 branch 밖의 #2098 `c3eab7b2`를 사용했고, 공개된 방법은 별도로 들어가는 수정을 참조합니다.

## 12. 학습 포인트

- **Predicate가 table을 읽기 전까지 port는 도달 불가능.** `ssm_kernel_available()`을 바꾸지 않고 `.rocm`만 채웠다면 아무도 부르지 않는 kernel과 녹색 build가 남았을 것입니다.
- **JIT cache key는 signature가 받는 모든 입력 dtype을 포함해야 함.** dtype이 섞인 checkpoint (f32 `A_log`, bf16 activation)가 바로 불완전한 key가 엉뚱한 module을 조용히 재사용하는 경우이고, 두 dtype을 한 process에서 돌리는 test만 이를 잡습니다.
- **Guard에 기대기 전에 발동할 수 있음을 증명.** Wave32 `#error`는 보호처럼 보였고 어디서나 compile되었는데, 이는 아무것도 검사하지 않는 guard와 같은 증상입니다.
- **Parity test를 변형으로 점검.** 8에서 시작하는 fold와 빠진 dtype key가 둘 다 큰 편차로 실패하므로, 허용 오차가 느슨하지 않고 의미가 있다는 것이 드러납니다.
- **순서를 바꾸는 변경에 greedy 글은 맞지 않는 도구.** 자유 출력은 첫 near-tie에서 갈라져 계속 갈라진 채로 남습니다. Decided position 기준을 둔 teacher-forced trace가 정밀도 변화와 버그를 구별합니다.

Refs: #2067, #1814, #1801, #2061, #2086, #631, #1862, #1870, #2059, #2065, #2098, #2064, #2068, #2063, #2069.
