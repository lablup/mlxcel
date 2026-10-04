# 기술 보고서: PR #2101 - Metal 전용이던 kernel 네 개의 HIP port

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 시험 완료. Branch head `3833998b` (측정 binary `800e23dc`), origin/main `6668031c` 위, PR open 상태로 머지 대기 중.

**언어**: C++ (`fast::hip_kernel`을 통한 HIP kernel source, port table, predicate), Rust (`apertus.rs`, `layers.rs`, `mamba.rs`, `nemotron_h.rs`의 모델 gate, FFI, parity test, 새 integration test binary), Markdown, TSV/CSV (logit trace, bench 결과)

**위험도**: 중간 (ROCm에서 Apertus, Cohere2, Mamba, Falcon-Mamba, Jamba가 기본값으로 새 GPU kernel을 실행하며, 이 계열들은 시험 호스트에 checkpoint가 없어 kernel test로만 덮여 있습니다. Mamba1 port는 state를 float32로 유지하므로 ROCm의 Mamba 계열 출력은 graph scan과 달라집니다. Nemotron-H의 기본 경로는 main과 byte 단위로 같고, opt-in인 `MLXCEL_FUSED_MOE_RELU2` 경로는 greedy text가 같지 않습니다. Metal과 CUDA는 실행하지 않았고, Metal의 predicate에는 GPU device 조건이 추가되었습니다)

## 요약

#1814 port 항목의 마지막인 Issue #2069는 Metal port만 있고 port table을 읽지 않는 gate 뒤에 있던 kernel 네 개를 다룹니다. Fused xIELU activation (Apertus), fused add3 + LayerNorm (Cohere2), Mamba1 selective scan (Mamba, Falcon-Mamba, Jamba), 그리고 Nemotron-H의 opt-in `MLXCEL_FUSED_MOE_RELU2` 분기가 쓰는 fc1 squared-ReLU MoE kernel입니다. ROCm에서는 넷 모두 MLX graph로 fallback했습니다.

이 PR은 kernel마다 HIP source, 채워진 `.rocm` table 항목, table을 읽는 predicate를 추가하고, 각 모델 gate가 그 predicate를 읽게 합니다. CUDA는 fallback을 유지합니다. `grep -rn '\.rocm = nullptr' --include=*.cpp src/`의 결과는 16에서 12로 줄었고, `mlx_cxx_kernels.cpp`에 남은 하나는 CUDA graph-exact Mamba1 table입니다.

핵심 결정은 각 port를 무엇에 맞출 것인가입니다. xIELU와 add3는 ROCm 자체의 unfused graph와 byte 단위로 같습니다 (xIELU는 f32, f16, bf16에서 16410개 원소 중 0개가 다름, add3는 열 가지 case 모두 동일). Metal kernel이 아니라 ROCm graph의 반올림 지점을 재현했기 때문입니다. Mamba1 port는 float32-state 변형입니다. CUDA의 graph-exact 변형은 ROCm에서 재현할 수 없기 때문입니다. relu2 kernel은 f32로 계산하며 dense f32 기준과의 차이가 8.1e-6 nrms 이내입니다.

`MLXCEL_FUSED_MOE_RELU2=1`을 켠 Nemotron-3-Nano의 decode는 median 75.15에서 75.24 tok/s로 움직였습니다 (+0.1%, noise 범위). 대역폭에 묶인 GEMV만 대체하는 kernel이므로 예상한 결과입니다. Issue는 relu2 greedy text가 기본 경로와 같기를 요구했지만 같지 않으며, PR은 이를 decided position 불일치 0인 teacher-forced trace를 근거로 한 이탈 사항으로 문서화합니다.

정적 review에서 `MLXCEL_DEVICE=cpu`일 때 새 경로 네 개가 모두 "Custom kernels only run on GPU"를 던진다는 점이 발견되었습니다. Predicate가 port table만 읽었기 때문입니다. 이제 predicate는 기본 device가 GPU인지도 확인하며, 같은 문제가 있던 #2065의 fused MoE predicate 두 개에도 같은 수정을 적용했습니다.

## 1. 문제 정의

### 1.1 ROCm에서 닿을 수 없던 kernel 네 개

각 kernel은 port가 있는지 묻는 대신 Metal을 지목하는 gate에 막혀 있었습니다.

- `fused_xielu`는 `metal::is_available()`이 false이면 항상 elementwise fallback을 반환했습니다.
- `residual_add3_layer_norm`은 `ffi::metal_is_available()`을 요구했습니다.
- `mamba.rs`는 scan을 `gpu_backend_kind() == GpuBackendKind::Metal`로 막았습니다. Jamba는 `mamba1_scan_kernel_accepts`를 거쳤지만, 두 Mamba1 table 모두 `.rocm` 항목이 없어 ROCm에서는 false였습니다. 그래서 Jamba, Mamba, Falcon-Mamba는 모두 step별 graph scan을 실행했습니다.
- `fused_moe_forward`의 relu2 분기는 #2065 이후 ROCm에서 `gather_qmm`으로 사양했습니다. `moe_fc1_relu2_ports()`에 `.rocm` 항목이 없었기 때문입니다 (down kernel만 port되어 있었음).

Table만 채웠다면 앞의 세 gate는 열리지 않았을 것입니다. #2065가 fused MoE pair에서 기록한 것과 같은 교훈입니다.

### 1.2 Kernel마다 다른 계약

네 kernel은 하나의 정확성 계약을 공유하지 않습니다. xIELU와 add3는 대체하는 unfused graph와 byte 단위로 같다고 문서화되어 있어서, `MLXCEL_FUSED_XIELU`와 `MLXCEL_FUSED_ADD_NORM` (둘 다 기본값 on)이 greedy decode를 흔들지 않습니다. Mamba1 scan에는 계약이 다른 두 변형이 있습니다. Metal 변형은 state를 float32로 유지해 graph scan과 다르고, CUDA 변형 (#1981)은 매 step을 graph scan과 똑같이 반올림합니다. relu2 kernel은 dense f32 기준에 맞춥니다. 그래서 port는 시험하기 전에 kernel별로 목표를 정해야 합니다.

## 2. 변경 요약

Commit 여덟 개:

- **`64331d34`** `update(rocm): port the fc1 squared-ReLU MoE kernel to HIP`
- **`fdc613a5`** `update(rocm): port the fused xIELU kernel to HIP`
- **`38fd43f1`** `update(rocm): port the float32-state Mamba1 scan kernel to HIP`
- **`fc5170ca`** `update(rocm): port the fused add3 + LayerNorm kernel to HIP`
- **`9186b075`** `docs(rocm): list the newly ported kernels in the README`
- **`eb72b359`** `fix(rocm): gate #2069 kernel ports on the GPU device`: review 수정 (4절).
- **`800e23dc`** `fix(rocm): gate the fused MoE predicates on the GPU device too`: #2065 predicate에 대한 같은 수정.
- **`3833998b`** `docs(rocm): publish the #2069 kernel port results on gfx1151`: `docs/benchmark_results/rocm-metal-only-ports-gfx1151-2026-10-05.md`, bench CSV, `benchmarks/logit_traces/rocm_gfx1151_9186b075/` 아래 trace, `docs/installation.md`와 `docs/environment-variables.md`의 행.

파일 27개, +1584 / -124. HIP source는 모두 `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`에 있습니다.

### 2.1 `moe_fc1_relu2`

`MOE_FC1_RELU2_HIP_SOURCE`는 Metal source를 바탕으로 #2065 gate-up port의 모양을 따라 작성했습니다. Row당 32-lane wavefront 하나, `__shfl_down(v, o, 32)`를 쓰는 16..1 fold, wave32 guard, template arg의 `T`입니다. 새 bridge predicate `fused_moe_relu2_kernels_available()`이 fc1과 down table을 모두 읽고, 분기는 두 table을 직접 검사하는 대신 이 predicate를 읽습니다. ROCm에서 이 분기의 block당 row 수 기본값은 #2065가 `run_fused_moe_two_kernel`에 대해 측정한 2이므로, 분기의 두 kernel이 같은 launch 모양을 씁니다. Flag는 opt-in으로 남습니다.

### 2.2 `fused_xielu`

조기 반환은 이제 `fused_xielu_kernel_available()`을 읽습니다. HIP kernel은 ROCm에서 `apertus_xielu`와의 byte 단위 일치를 목표로 합니다. ROCm의 graph는 overlay elementwise kernel의 연쇄이며, 각 kernel은 float로 넓혀 한 번 계산한 뒤 `T`로 반올림하고, `Expm1`은 device `expm1f`를 호출합니다. Port는 모든 중간값을 같은 방식으로 `T`를 거쳐 반올림하고, 같은 `expm1f`를 호출하며, graph처럼 `minimum`에서 NaN을 전파하고, `#pragma clang fp contract(off)`로 f32 경로가 `x * beta`를 마지막 덧셈에 fuse하지 못하게 합니다. ROCm에서는 원소 수를 template argument 대신 `x_shape[0]`에서 읽습니다 (4.2절). 새 test가 비교할 수 있도록 `apertus_xielu`는 `pub(crate)`가 되었습니다.

### 2.3 `mamba1_selective_scan`

`MAMBA1_SCAN_HIP_SOURCE`는 Metal의 float32-state 변형을 옮긴 것이며, `simd_sum`은 16..1 `__shfl_down` fold로 바뀌었습니다. CUDA의 graph-exact 변형은 ROCm에서 닿을 수 없습니다. Graph scan의 `state @ C`는 K = N (8 또는 16)이어서 overlay GEMV 조건 `K % 32 == 0`을 통과하지 못하고, overlay는 이를 rocBLAS로 보냅니다. Custom kernel은 rocBLAS의 reduction 순서를 복제할 수 없습니다. 새 predicate `mamba1_scan_float_state_kernel_available()`이 float32-state table을 읽고, `mamba.rs`는 이를 읽은 다음 Jamba의 gate처럼 `mamba1_scan_kernel_accepts`를 읽습니다. 그래서 32보다 넓은 state는 column을 쓰지 않은 채 남기는 대신 graph scan을 탑니다.

Kernel은 sequence 길이를 `X_shape`에서 읽습니다. 첫 실행은 fault (`0x4000000000`에서 HSA memory fault)를 냈습니다. Fork의 `hip_kernel`은 `<input>_shape`와 `_strides`를 pointer로 선언했지만 launch는 값으로 넘기기 때문입니다. 구현자는 overlay의 `custom_kernel.cpp`에서 이를 고쳤지만, 그 사이 머지된 #2100이 같은 버그를 `LOCAL_FIXES.md` 항목 30으로 고쳤습니다. Rebase 때 branch는 자신의 사본을 버리고 #2100의 수정을 씁니다. `docs/environment-variables.md`, `custom_kernel.cpp`, `LOCAL_FIXES.md`의 conflict는 main 쪽 버전으로 해결했습니다.

### 2.4 `fused_add3_layer_norm`

`residual_add3_layer_norm`은 새 `fused_add3_layer_norm_available()`을 읽고, `.expect` 메시지도 이 predicate를 가리킵니다. HIP kernel은 Metal kernel의 reduction이 아니라 ROCm의 unfused pair를 재현합니다. Residual은 overlay의 compiled `Add`처럼 덧셈마다 `T`로 반올림하고, norm은 overlay의 `layer_norm_kernel<T, 256, 4>`를 따릅니다 (row당 256 thread, 4개씩 strided group, width 32의 16..1 `__shfl_xor` fold, wavefront 합 여덟 개를 다시 fold, `1.0f / sqrtf`, bias를 memory에서 읽는 `T(w * norm + b)`). ROCm build는 `MLXCEL_BRIDGE_ROCM_BACKEND`를 통해 row당 256 thread로 launch합니다. Row는 6656 상한에 맞춘 register에 담깁니다 (thread당 float 28개).

### 2.5 Test

- 새 `fused_moe_relu2_parity_tests`는 Nemotron-3-Nano 모양, 4 bit와 8 bit에서 flag를 끈 상태와 켠 상태로 `fused_moe_forward`를 실행해 dense f32 기준 및 `gather_qmm` 분기와 비교합니다. 이를 위해 `fused_moe_parity_tests.rs`의 helper 세 개가 `pub(crate)`가 되었습니다.
- 새 `apertus_tests::fused_xielu_kernel_matches_graph_every_dtype`은 f32, f16, bf16에서 kernel을 `apertus_xielu`와 비교하고, Metal과 ROCm에서 kernel 경로를 탔는지 확인하며, ROCm에서는 다른 bit가 0개임을 확인합니다. 기존 bf16 bit-for-bit test는 이제 HIP kernel을 실행합니다.
- `residual_add3_layer_norm_matches_the_unfused_pair`는 `metal` feature에서만 compile되었습니다. 이제 모든 backend에서 build되고, Metal과 ROCm에서 kernel 경로를 확인하며, `MLXCEL_FUSED_ADD_NORM=0`이면 눈에 보이게 skip하고, f32, 너비 1025와 5, 6656 상한을 추가해 열 가지 case가 되었습니다.
- Mamba1 f32 parity test에 N = 32가 추가되었고, 새 test가 Metal과 ROCm에서 predicate가 true임을 확인해 parity test가 조용히 skip되지 않게 합니다. bf16 test도 이제 ROCm에서 실행됩니다.
- 새 integration test binary `tests/cpu_device_custom_kernel_gates.rs` (4.1절).

Tolerance는 하나도 완화하지 않았습니다.

## 3. 측정 결과

### 3.1 환경

AMD Ryzen AI MAX+ 395와 Radeon 8060S (`gfx1151`, RDNA 3.5), 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 (HIP 7.15.26333), MLX pin `81ba1c6a`, ROCm overlay `75915908`, `cargo build --release --features rocm`. Before: origin/main `6668031c`. After: branch `800e23dc`.

### 3.2 Kernel parity

| Kernel | ROCm에서의 기준 | gfx1151 결과 | Negative check |
|---|---|---|---|
| `fused_xielu` | `apertus_xielu` graph, f32 / f16 / bf16 | byte 단위 동일 (dtype마다 16410개 중 0개 다름) | 양수 분기에 `alpha_n`을 쓰면 실패 (nrms 0.30) |
| `fused_add3_layer_norm` | unfused `compiled_add3` + `fast::layer_norm` | 열 가지 case 모두 byte 단위 동일 (f16, bf16, f32; 너비 5, 1025, 4096, 6656) | `1.0f / sqrtf`를 `rsqrtf`로 바꾸면 실패 |
| `mamba1_selective_scan` (float32 state) | f32 scalar 기준 1e-5 이내; bf16은 graph scan보다 덜 정확하지 않음 | N 32까지 통과; `mamba1_scan_parity_tests` 5 passed | 8에서 시작하는 lane fold가 N = 32에서 실패 (상대 오차 0.87) |
| `moe_fc1_relu2`와 #2065 down kernel | dense f32 기준; `gather_qmm`은 자체 흔들림 이내 | 기준에서 8.1e-6 nrms / 4.2e-4 nmax; `gather_qmm`에서 5.45e-3 ~ 5.70e-3, `gather_qmm` 자체는 기준에서 5.08e-3 ~ 5.43e-3 | 8에서 시작하는 lane fold가 실패 (nrms 0.76) |

`rsqrtf`로 add3가 실패한다는 것은 test가 반올림 한 단계의 변화도 잡아낸다는 뜻이며, 계약이 tolerance가 아니라 byte 단위 일치인 이유입니다.

### 3.3 Decode 처리량

이 호스트에 checkpoint가 있는 것은 Nemotron-H뿐입니다. `scripts/bench_decode.sh` pp512/tg128, 양쪽 모두 `MLXCEL_FUSED_MOE_RELU2=1`, before와 after를 run마다 번갈아 실행했고 모든 run은 `scripts/rocm_gpu_guard.sh`를 거쳤습니다. Before에서는 flag가 `gather_qmm`으로 사양했고, after에서는 HIP fc1과 down kernel을 탑니다.

| 모델 | Before tok/s | After tok/s | 변화 (median) |
|---|---|---|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit, `MLXCEL_FUSED_MOE_RELU2=1` | 75.15 / 75.22 / 75.09 (median 75.15) | 75.24 / 75.38 / 75.13 (median 75.24) | +0.1%, noise 범위 |

이는 flag에 대한 Metal 쪽 설명과 맞습니다. Flag는 `gather_qmm`이 이미 대역폭 가까이에서 실행하는 routed fc1과 fc2 GEMV만 대체합니다. 원본 행: `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_relu2-before.csv`와 `..._relu2-after.csv`.

### 3.4 relu2 greedy 이탈

Issue는 flag를 켠 Nemotron-H의 greedy 128-token 출력이 기본 경로와 같기를 요구했습니다. 세 prompt에서 text는 약 20 token 동안 같다가 near-tie에서 갈라집니다 ("user query" 대 "user request", "checks if" 대 "checks whether"). Kernel은 fc1, relu², fc2를 f32로 유지하지만 `gather_qmm`은 중간값을 bf16으로 반올림합니다. 그래서 kernel이 f32 기준에 약 천 배 가깝습니다 (8.1e-6 대 5.1e-3 ~ 5.4e-3 nrms). 첫 token이 뒤집히면 free-running 생성은 그 뒤 전체가 다른 text를 조건으로 삼으므로, 여기서 text 일치는 쓸모 있는 척도가 아닙니다. Teacher-forced `w1ctx512` trace (`python3 scripts/compare_logit_traces.py <reference> <candidate> --decided 2.0`):

| 기준 | 후보 | Top-1 불일치 | Decided 불일치 | 최대 gap | Perplexity 기준 / 후보 |
|---|---|---|---|---|---|
| ROCm 기본 | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 1 / 128 | 0 / 66 | 0.500 | 7.410 / 7.436 |
| Metal M5 기본 (`metal_m5_d1128266`) | ROCm `MLXCEL_FUSED_MOE_RELU2=1` | 5 / 128 | 0 / 71 | 0.250 | 7.489 / 7.436 |

기본 경로는 바뀌지 않았습니다. Branch의 `w1ctx512` trace는 main `6668031c`에서 만든 trace와 모든 data 행이 byte 단위로 같습니다.

## 4. PR 안에서 고친 review 지적

### 4.1 HIGH: CPU device

Custom kernel은 GPU stream에서만 실행되며, CPU stream에서는 `eval_cpu`가 "Custom kernels only run on GPU"를 던집니다. Port table은 "이 backend에 kernel이 있는가"에 답할 뿐 "기본 device가 GPU인가"에는 답하지 않습니다. 그래서 `MLXCEL_DEVICE=cpu`인 ROCm build는 새 경로 네 개 모두에서 CPU stream 위에 custom kernel을 만들었을 것입니다. 이제 각 predicate는 `ssm_kernel_available`과 `mamba1_scan_kernel_accepts`가 이미 그랬듯이 `default_device() == Device::gpu`도 요구합니다.

이어서 orchestrator가 #2065의 `fused_moe_kernels_available()`과 `moe_down_kernel_available()`에서도 같은 문제를 찾았습니다. `MLXCEL_DEVICE=cpu`에서 SwitchGLU decode가 HIP launch로 들어갔고, bridge 호출이 이미 `Ok`를 반환한 뒤 `eval_cpu`가 던졌습니다. 두 predicate 모두 device 조건을 갖게 되었고, Nemotron-H는 CPU에서 `forward_nonfused`를 탑니다. Metal도 `metal::is_available()`과 backend 종류 검사를 통해 같은 문제가 있었으며, GPU에서의 동작은 바뀌지 않습니다.

`tests/cpu_device_custom_kernel_gates.rs`는 CPU device에서 predicate 여섯 개가 모두 false인지 확인하고, CPU에서 `fused_xielu` (scalar 공식과 비교)와 `residual_add3_layer_norm` (unfused pair와 비교)을 실행합니다. 프로세스 전역의 기본 device를 옮기므로 별도 test binary로 두었고, 공유 lib test binary는 그 변화를 보지 않습니다. Device 조건을 빼면 이 test는 실패합니다.

### 4.2 MEDIUM: activation 크기마다 hipRTC compile 한 번

xIELU port는 처음에 Metal처럼 원소 수 `n`을 template argument로 넘겼습니다. hipRTC는 template argument를 cache key로 쓰고 eviction이 없으므로, activation 크기가 다를 때마다 kernel을 하나 더 compile해 보관하게 됩니다. ROCm에서는 이제 `x_shape[0]`을 읽어 dtype마다 kernel 하나만 만듭니다. Metal의 template argument는 그대로입니다.

### 4.3 기타

- Mamba의 gate에 N <= 32 검사가 없었습니다. 이제 `mamba1_scan_kernel_accepts`를 거칩니다.
- Kernel 없이도 통과할 수 있던 test를 강화했습니다 (2.5절의 kernel 경로 확인과 눈에 보이는 skip).
- 주석은 더 이상 ROCm relu2 분기를 byte 단위 동일이라 부르지 않고, xIELU의 일치 범위를 backend별로 시험한 dtype으로 한정하며 (Metal bf16; ROCm f32, f16, bf16; Metal f32와 f16은 tolerance만), add3의 일치는 compiler flag 일치가 아니라 test로 고정된다고 설명합니다.

## 5. 기술적 선택과 그 이유

- **xIELU와 add3는 Metal kernel이 아니라 같은 backend의 graph에 맞춥니다.** Metal kernel이 Metal graph와 byte 단위로 같은 것은 Metal의 반올림 지점과 reduction 순서를 복제했기 때문입니다. ROCm graph는 이것이 다르므로 (elementwise op마다 한 번의 반올림, device `expm1f`, `__shfl_xor` fold를 쓰는 256-thread norm), Metal kernel을 충실히 옮겼다면 ROCm에서는 어느 것과도 byte 단위로 같지 않았을 것입니다. add3의 제약 (byte 단위 일치, tolerance 완화 없음)은 열 가지 case 모두에서 지켜졌습니다.
- **float32-state Mamba1 변형을 옮깁니다.** Graph의 `state @ C`가 rocBLAS에서 실행되므로 ROCm에서 graph-exact 출력은 불가능합니다. float32-state 변형은 계약이 이미 정해져 있고 (Metal의 것), 같은 f32와 bf16 test를 통과하며, Mamba, Falcon-Mamba, Jamba가 port 하나를 공유하게 합니다.
- **relu2는 f32로 두고 greedy 이탈을 받아들입니다.** `gather_qmm`의 text에 맞추려면 그 bf16 중간 반올림을 재현해야 하고, 그러면 kernel이 f32 기준에서 멀어집니다. Trace는 뒤집힌 token이 near-tie이며 ROCm 기본과 Metal 양쪽에 대해 decided 불일치가 0임을 보여 줍니다. Flag는 opt-in으로 남고 이탈은 PR 본문에 적혀 있습니다.
- **Table뿐 아니라 device로도 gate합니다.** #1801 규칙 (gate와 dispatch가 같은 table을 읽음)은 유지되며, device 조건은 table이 담을 수 없는 유일한 사실입니다.
- **ROCm에서 호출마다 달라지는 크기는 template arg가 아니라 shape에서 읽습니다.** hipRTC cache를 dtype당 항목 하나로 제한합니다.
- **relu2 분기에 #2065의 ROCm geometry를 재사용합니다.** Block당 row 수 기본값 2로 분기의 두 kernel이 down kernel에 대해 측정된 선택과 맞습니다.
- **Rebase 때 중복된 overlay 수정을 버립니다.** #2100이 같은 by-value shape 버그를 이미 고쳤으므로, 사본을 하나만 두어 `LOCAL_FIXES.md`에 서로 다른 항목 두 개가 생기지 않게 합니다.

## 6. 검증

gfx1151, `--release --features rocm`:

- 최종 head에서 `make verify-rocm`: 147 suite, 11867 passed, 0 failed, 378 ignored (코드 head `800e23dc`, 이후 커밋은 문서만 변경).
- 3.2절의 negative check를 포함한 kernel parity: xIELU는 세 dtype에서 byte 단위 동일, add3는 열 가지 case에서 byte 단위 동일, `mamba1_scan_parity_tests` 5 passed, `fused_moe_relu2_parity_tests` 범위 이내.
- `cpu_device_custom_kernel_gates` 통과, device 조건을 빼면 실패.
- `models::mamba::`와 `models::jamba::`: 22 passed, 2 ignored. Cohere2 모델 test: 13 passed, 7 ignored.
- `mlxcel-core`와 `mlxcel`에 대한 `cargo clippy -D warnings`: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: 통과, dtype-key pin은 범위 안에서 9.
- Nemotron-H 기본 `w1ctx512` trace가 main과 byte 단위로 같음.

## 7. 남은 위험과 검증하지 않은 부분

- **Metal과 CUDA는 실행하지 않았습니다.** 이 호스트에 하드웨어가 없습니다. Metal의 kernel source와 table 항목은 손대지 않았지만, Metal은 이제 넓어진 add3와 Mamba1 test, 새 relu2 test와 CPU device test를 처음으로 실행하며, predicate에 GPU device 조건이 추가되었습니다 (GPU 동작은 그대로, `MLXCEL_DEVICE=cpu`에서는 던지는 대신 fallback). CUDA에서는 fc1_relu2, xIELU, add3, float-state Mamba1 table이 null로 남고, add3 test는 unfused pair를 자기 자신과 비교합니다.
- **Apertus, Cohere2, Mamba, Falcon-Mamba, Jamba는 end-to-end로 실행하지 않았습니다.** 호스트에 checkpoint가 없어 이 계열의 decode 처리량과 모델 출력은 측정되지 않았습니다. 이들은 이제 ROCm에서 기본값으로 새 kernel을 실행합니다. Mamba 계열은 float32-state 출력이 되며, 이전 ROCm graph scan과 다릅니다.
- **xIELU와 add3의 byte 단위 일치는 ROCm compiler에 달려 있습니다.** Compiler flag 일치가 아니라 test로 고정되어 있습니다. `expm1f`나 norm kernel을 바꾸는 ROCm compiler 또는 overlay 업그레이드는 이를 깨뜨릴 수 있고, test가 그것을 알려 줄 것입니다.
- **Wave64 (CDNA)는 시험하지 않았습니다.** AMD clang 23은 `__AMDGCN_WAVEFRONT_SIZE`의 어느 철자도 정의하지 않아 wave32 `#error` guard는 작동하지 않습니다 (#2098). 그곳에서의 정확성은 명시한 shuffle width에 기댑니다. add3 port는 `__shfl_xor(..., 32)`로 fold하고 256-thread block에 32-lane group 여덟 개가 있다고 가정합니다.
- **장치 하나, 모델 하나만 측정했습니다.** 처리량은 gfx1151의 Nemotron-H에서만 나왔습니다.
- **C++ 경고** 하나가 `mlx_cxx_kernels.cpp:3346` 근처에서 식 끝에 소멸되는 temporary에 대해 나옵니다. 이 PR 이전부터 있었는지는 확인하지 않았습니다.

권장 후속 작업: 넓어진 add3와 Mamba1 test, 새 relu2와 CPU device test의 Metal 실행; relu2 near-tie 주장에 대한 Metal greedy 확인; 작동하지 않는 `__AMDGCN_WAVEFRONT_SIZE` macro에 기대지 않는 ROCm port용 wave size 검사; ROCm 호스트에 Apertus, Cohere2, Mamba, Falcon-Mamba, Jamba checkpoint 마련; `mlx_cxx_kernels.cpp:3346` 경고 분류.

## 8. 학습 포인트

- **Byte 단위 일치는 kernel이 아니라 backend의 성질입니다.** 같은 계약 ("unfused graph와 동일")이 Metal과 ROCm에서 서로 다른 code를 요구했습니다. Backend마다 graph가 반올림하는 위치가 다르기 때문입니다. 다른 backend의 kernel을 충실히 옮긴 port도 계약을 놓칠 수 있습니다.
- **Negative check는 실패할 수 있어야 합니다.** N <= 16이면 Mamba1 fold의 lane 16..31은 0을 담고 있어서, 8에서 시작하는 fold도 통과했습니다. N = 32를 추가하자 check가 실제로 작동했습니다 (상대 오차 0.87).
- **Port table은 device가 아니라 backend를 설명합니다.** `MLXCEL_DEVICE=cpu`는 GPU backend를 유지한 채 stream만 옮깁니다. Table만 읽는 predicate는 던지는 경로를 엽니다. 같은 빈틈이 머지된 predicate (#2065)와 Metal에도 있었습니다.
- **hipRTC template argument는 cache key입니다.** 호출마다 달라지는 값 (원소 수)은 shape argument에 두어야 하며, 그렇지 않으면 cache가 끝없이 커집니다.
- **Free-running greedy text는 수치 변경에 대한 약한 일치 검사입니다.** Near-tie 하나가 뒤집히면 그 뒤가 모두 바뀝니다. Decided gap 기준을 둔 teacher-forced trace가 수치 개선과 결함을 구분합니다.
- **병렬 unit이 같은 버그를 고칠 수 있습니다.** #2100과 이 branch 모두 fork의 by-value shape 선언을 고쳤고, rebase 때 main 쪽 버전으로 해결해 수정을 하나만 남겼습니다.

Refs: #2069, #1814, #2065, #2098, #2100, #1981, #2005, #2007, #1801, #1813.
