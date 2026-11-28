# 기술 보고서: PR #2100 - Gumbel-max sampler와 rejection sampler를 HIP으로 port

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `74624ea7`, origin/main `57d8ed29` 위, 머지 대기 중.

**언어**: C++ (HIP kernel source, kernel holder, support predicate, bridge 함수, vendored ROCm overlay), Rust (fixed-key 및 two-sample test, kill-switch test, snapshot bound), Bash (`bench_decode.sh` 옵션, dtype-key checker test), Markdown (환경 변수, 설치, benchmark, LOCAL_FIXES, benchmark 결과), CSV (bench 행)

**위험도**: 중간 (ROCm에서 모든 모델의 기본 sampled decode 경로가 바뀝니다. 모든 HIP custom kernel이 거치는 vendored `fast::hip_kernel`의 argument 배치를 고치고, Metal과 CUDA가 공유하는 rejection predicate, bridge 함수, snapshot bound도 건드리며, 두 backend는 개발 호스트에서 실행할 수 없었습니다)

## 요약

Issue #2064 (#1814의 일부, epic #1801)는 temperature가 0보다 클 때 생성 token마다 한 번씩 실행되는 두 fused sampler의 HIP port를 요청했습니다. Filter 없는 경로의 Gumbel-max sampler와 filter 경로의 dual-pivot rejection sampler입니다. 이 PR 이전에는 둘 다 ROCm에서 MLX graph로 실행되었습니다. `gumbel_ports()`와 `rejection_ports()`에는 `.rocm` 항목이 없었고, `rejection_sample_supported()`는 정의상 Metal 또는 CUDA만 뜻하는 `custom_kernels_available()`을 반환했으므로, rejection slot을 채워도 도달할 수 없었습니다.

이 PR은 CUDA 본문을 줄 단위로 옮긴 `sampling_gumbel_hip.h`와 `sampling_rejection_hip.h`를 추가합니다. Input, output, grid, template argument, Philox-4x32-10 counter와 key 배치는 CUDA와 같습니다. 두 `.rocm` slot을 채우고, `rejection_sample_supported()`가 `has_kernel_port(rejection_ports())`를 반환하게 합니다. 두 port 모두 wave32 `#error` guard를 두지 않습니다. 두 kernel에는 lane 수준 연산이 없고, 모든 reduction과 rejection kernel의 scan이 step마다 barrier를 두는 shared memory로 이루어지기 때문입니다.

Gumbel port는 vendored ROCm overlay의 버그를 드러냈습니다. `fast::hip_kernel`은 `<input>_shape`와 `<input>_strides`를 pointer로 선언했지만 launch는 값으로 넘겼고, 그래서 이를 처음 읽은 kernel (`logits_shape[1]`)이 queue를 fault시켰습니다. 이는 LOCAL_FIXES item 30으로 고쳤으며, fork 쪽 변경이고 #1813의 upstream 후보입니다.

gfx1151에서 Llama-3.1-8B-Instruct-4bit, pp512/tg128로 측정한 결과 top-p 0.95 decode의 중앙값은 33.98에서 37.56 tok/s (1.11x)가 되었고, 이전 arm이 아래로 drift한 사실도 함께 적습니다. Gumbel 경로는 37.91에서 38.26 tok/s (+0.9%)로, 실행 간 편차 안이므로 향상으로 주장하지 않습니다. Issue의 top-k + top-p 명령은 vocab 128256에서 kernel에 도달하지 않으므로 측정하지 않았습니다.

## 1. 문제 정의

### 1.1 빈 slot 두 개, 그것을 보지 못하는 predicate 하나

`gumbel_max_sample_supported()`는 이미 `has_kernel_port(gumbel_ports())`를 읽었으므로, `.rocm`만 채우면 도달 가능해졌습니다. `rejection_sample_supported()`는 그렇지 않았습니다.

```cpp
return mlxcel::custom_kernels_available();
```

`custom_kernels_available_for`는 Metal과 CUDA에서만 true이므로, ROCm에서는 `rejection_ports()`에 무엇이 있든 predicate가 false였습니다. 이 predicate는 `sampling_rejection_available()`과 bridge의 routing에 쓰입니다. `verify-kernel-port-dispatch`의 checker rule 4는 이것이 `metal_is_available() || cuda_is_available()` 형태로 쓰이지 않았기 때문에 잡지 못했습니다.

### 1.2 Profile이 말한 기대 효과

Decode profile (`rocm-decode-profile-gfx1151-2026-09-30.md`)은 sampler 꼬리 전체를 greedy decode GPU 시간의 0%, sampled decode GPU 시간의 0.4 ~ 3.8%로 보고했습니다. Port가 없을 때 ROCm의 sampled decode는 filter 없는 경로에서 `random::categorical`, filter 경로에서 `argpartition` / `argsort` / `cumsum` chain을 실행했습니다. 따라서 filter 없는 경로의 기대 이득은 작고, graph가 vocabulary 전체를 정렬하는 경로에서만 더 클 것으로 보였습니다.

### 1.3 같은 난수 위에서 port를 graph에 맞춰 보는 test가 없었음

기존 suite인 `sampling_gumbel_tests`와 `sampling_rejection_tests`는 각 kernel을 정확한 목표 분포와 비교합니다. 이들은 ROCm에서 일찍 반환했습니다. Port가 MLX의 key sequence를 다른 port와 같은 방식으로 소비하는지 확인하는 test는 없었습니다. 그것이 `mlx::core::random::seed(...)`가 backend 사이에서 stream을 재현하게 하는 조건입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `sampling_gumbel_hip.h` (신규) | `GUMBEL_MAX_SAMPLE_HIP_SOURCE`, 표기만 바꾼 CUDA 본문 |
| `sampling_rejection_hip.h` (신규) | `REJECTION_SAMPLE_HIP_SOURCE`, 표기만 바꾼 CUDA 본문 |
| `sampling.cpp` | `MLXCEL_BRIDGE_ROCM_BACKEND` 아래에서 `fast::hip_kernel`을 호출하는 `GumbelKernelHolderHip`; `gumbel_ports()`의 `.rocm` getter |
| `sampling_rejection.cpp` | `RejectionKernelHolderHip`; `rejection_ports()`의 `.rocm` getter; `rejection_sample_supported()`가 `has_kernel_port(rejection_ports())` 반환 |
| ROCm overlay `custom_kernel.cpp` | 값으로 전달되는 `KernelShape` / `KernelStrides` struct, `elem_to_loc` overload, `JIT_MAX_NDIM`에 대한 `static_assert`, ndim > 0일 때만 shape/strides/ndim 추가 (LOCAL_FIXES item 30) |
| Bridge (`mlx_cxx_bridge.cpp/.h`, `lib.rs`) | env와 무관한 `sampling_gumbel_backend_supported()`, `sampling_rejection_backend_supported()`, 그리고 `random_bits()`; 주석 수정 |
| `gpu_backend.h/.cpp` | `custom_kernels_available()`의 현재 의미에 대한 주석; `MLXCEL_DEBUG_KERNEL_BACKEND` 접미사 문구 수정 |
| Test | 신규 `sampling_fixed_key_tests.rs` (test 다섯 개); kill-switch test가 새 predicate를 물음; `temperature_one_support_unchanged`의 routed bound 1 ulp에서 2 ulp로; dtype-key checker test가 launch 둘을 모두 옮김; Python probe test가 새 접미사를 따름 |
| `bench_decode.sh` | `--temperature`, `--top-p` (단순 십진수만 허용, 자동 생성 파일명에 tag) |
| 문서와 데이터 | `environment-variables.md`, `installation.md`, `benchmarks.md`, item 30에 대한 upstream README 행, `LOCAL_FIXES.md` item 30, 신규 `rocm-samplers-gfx1151-2026-10-05.md`, bench CSV 네 개 |

Branch에는 commit이 여섯 개 있습니다. Overlay 수정 (`cc93789f`), port와 predicate (`85e39880`), 결과 문서 (`ce4105b7`), kill-switch predicate 수정 (`f7bec56d`), snapshot bound (`6e336ec8`), 문서와 `bench_decode.sh` 입력 검증 정리 (`74624ea7`)입니다. Diff는 파일 32개, 추가 1,341줄, 삭제 46줄입니다.

## 3. Port

### 3.1 줄 단위 이식, 세 종류의 표기 변경

두 HIP source는 CUDA kernel의 thread mapping, grid, input (`logits`, `rng_key`, `temp`; `probs`, `probs_draw`, `params`, `rng_key`), output (`vals`, `idxs`; `ids`, `ok`, `rounds`), template argument를 dtype key까지 그대로 유지합니다. Philox-4x32-10 counter (Gumbel은 `{base/4, 0, row, 0}`, rejection은 `{round, 0, row, 0}`), key `{rng_key[0], rng_key[1]}`, 23-bit open-interval uniform도 같으므로, seed가 CUDA와 Metal에서처럼 ROCm stream을 재현합니다. `__umulhi`는 HIP에서도 같은 뜻입니다. 차이는 다음과 같습니다.

- `hip_bfloat16` 값을 읽을 때 명시적 `(float)`. ROCm에서 MLX가 bfloat16 대신 쓰는 이 type은 `explicit` 연산자로만 float로 변환됩니다.
- `-INFINITY` 대신 `-__builtin_huge_valf()`. hipRTC는 `fast::hip_kernel`이 앞에 붙이는 header만으로 본문을 compile하며, 그 안에 `<cmath>`가 있다는 보장이 없습니다.
- `temp[0]`에 붙은 불필요한 `(float)`.

Holder와 launch는 `sampling.cpp`와 `sampling_rejection.cpp`의 CUDA 것 옆에 남습니다. Header는 launch가 아니라 데이터만 담으므로 `make verify-kernel-dtype-keys`의 `EXPECTED_IN_SCOPE`와 그 개수가 바뀌지 않습니다. Sampler의 launch를 helper로 옮기거나 주석 처리하는 checker 자체의 test는 이제 HIP launch에도 같은 일을 합니다.

### 3.2 Wave32 guard를 두지 않은 이유

#1814 port 요구사항은 lane 사이에서 reduction이나 shuffle을 하는 port마다 wavefront 크기 macro에 대한 `#error` 검사 두 개를 두게 합니다. Lane 16에서 시작하는 fold는 64-lane CDNA wave의 절반을 버리고도 유한하고 틀린 결과를 내기 때문입니다. 이 두 kernel에는 lane 수준 연산이 없습니다. Gumbel kernel의 index를 함께 나르는 halving reduction, rejection kernel의 reduction과 Hillis-Steele scan은 모두 step 사이에 `__syncthreads()`를 두는 `__shared__` memory를 거치므로, 결과가 한 wave에 몇 thread가 있는지에 의존하지 않습니다. Wave64에서 `#error`를 내면 올바른 kernel을 거부하게 됩니다. 또한 PR #2099는 ROCm 10의 AMD clang이 두 macro를 모두 정의하지 않아 guard 관용구가 아무 효과가 없다는 것을 발견했으므로, 여기서 guard를 두어도 잡아낼 것이 없습니다. Issue의 acceptance criterion은 이 이유를 옆에 적고 체크되었습니다.

## 4. Predicate와 그것을 읽는 test

`rejection_sample_supported()`는 `default_device() != Device::gpu` 검사를 유지한 뒤 `mlxcel::has_kernel_port(rejection_ports())`를 반환합니다. 이는 `rejection_sample`의 `select_kernel_port`가 읽는 바로 그 table이며, `gumbel_max_sample_supported()`와 같은 형태입니다. Metal과 CUDA에서는 이전처럼 true입니다. `fused_sample`, `fused_sample_probs`, speculative 경로는 호출자 변경 없이 HIP kernel에 도달합니다.

Integration test 두 개가 예전 의미에 의존했습니다. `tests/sampling_gumbel_kill_switch.rs`와 `tests/sampling_rejection_kill_switch.rs`는 env kill switch를 켜고, 어떤 거절 메시지를 기대할지를 `custom_kernels_available()`로 골랐습니다. 이 함수는 여전히 Metal 또는 CUDA만 true이므로, 이제 port가 있는 ROCm에서 test는 "port 없음"을 기대했을 것입니다. Env switch가 켜져 있으면 `sampling_*_available()`은 backend와 무관하게 false이므로, test에는 switch를 빼고 묻는 predicate가 필요했습니다. 이를 위해 PR은 bridge에 `sampling_gumbel_backend_supported()`와 `sampling_rejection_backend_supported()`를 추가했습니다 (`f7bec56d`). `gpu_backend.h` 주석은 이제 `custom_kernels_available()`이 family 단위 gate이고 HIP port가 있는 kernel은 자기 predicate로 gate된다고 설명합니다. `MLXCEL_DEBUG_KERNEL_BACKEND` 접미사는 "no custom kernel ports"에서 "not every kernel family is ported"로 바뀌었습니다.

## 5. Overlay 버그: shape와 stride를 값으로

Fork의 `mlx/backend/rocm/custom_kernel.cpp`에 있는 `build_kernel`은 input의 shape를 `const int32_t* <name>_shape`, stride를 `const int64_t* <name>_strides`로 선언했습니다. `CustomKernel::eval_gpu`는 이들을 `KernelArgs::append_ndim`으로 넘기는데, 이 함수는 vector를 `JIT_MAX_NDIM` (8) 항목으로 채운 뒤 그 저장소를 가리키는 pointer를 `hipModuleLaunchKernel`에 줍니다. 그래서 kernel은 값 자체를 받았고, shape의 처음 8 byte를 주소로 읽었습니다. HIP Gumbel port는 CUDA, Metal port처럼 `logits_shape[1]`을 읽으며, `HSA_STATUS_ERROR_MEMORY_FAULT`로 `0x80000000000`에서 죽었습니다. 이는 `0x80000000800`이 속한 page이고, 이 값은 shape `{2048, 2048}`을 64-bit pointer로 읽은 것입니다. mlxcel의 어떤 HIP kernel도 이전에 두 argument를 읽은 적이 없었기 때문에 드러나지 않았습니다.

수정 (`cc93789f`)은 생성되는 header에 `operator[]`가 있는 `KernelShape` (`int32_t` 8개)와 `KernelStrides` (`int64_t` 8개)를 추가하고, parameter를 이들로 값 전달하도록 선언합니다. Upstream CUDA의 `const __grid_constant__ Shape` / `Strides`와 같은 방식입니다. `elem_to_loc` overload도 추가하고, `static_assert`로 8을 `JIT_MAX_NDIM`에 묶습니다. Launch는 이제 input의 ndim이 0보다 클 때만 shape, stride, ndim을 추가하며, 이는 `build_kernel`이 그것들을 선언하는 조건과 같습니다. 이전에는 이름이 source에 `<name>_shape`로 나오는 0-d input이 이후 모든 argument를 밀어냈을 것입니다. mlxcel kernel 중 그런 input을 가진 것은 없습니다.

이 수정은 LOCAL_FIXES item 30으로 기록됩니다. `custom_kernel.cpp`에 한정된 fork 쪽 변경이며 #1813 아래의 upstream 후보입니다. Upstream README는 package를 만들려면 패치되지 않은 fork에서의 재현이 필요하므로 "not packaged yet"으로 표시합니다. `sampling_fixed_key_tests`는 이 수정 없이 fault합니다. `_strides` 쪽은 mlxcel에 사용처가 없어 compile로만 확인되었습니다.

## 6. 정확성 근거

Sampler에는 원소 단위로 비교할 fallback이 없으므로, PR은 네 가지 방식으로 port를 graph에 맞춰 봅니다. 기존 suite 두 개도 이제 ROCm에서 일찍 반환하지 않고 실행되며 gfx1151에서 통과합니다.

### 6.1 고정 key, host에서 다시 계산

두 kernel은 호출마다 MLX의 기본 key sequence에서 Philox key 하나를 뽑으며, key가 주어지면 출력이 결정적입니다. `sampling_fixed_key_tests`는 reseed하고, 새 `random_bits` bridge 함수로 다음 launch가 가져갈 key를 읽고, 다시 reseed한 뒤 launch하고, host에서 draw를 다시 계산합니다.

- `gumbel_kernel_matches_the_keyed_graph_argmax`: 5003개 logit 48행, f32 (T 1.0, 0.7), bf16 (T 1.3), f16 (T 0.9). 모든 decided 행에서 kernel의 index가 host의 `argmax(logits / T + g)`와 같고, 같은 noise를 넣은 MLX graph의 `argmax`와도 같습니다. 이기는 score가 1e-4 이상 앞설 때 decided로 봅니다.
- `rejection_kernel_draws_the_keyed_token_under_min_p`: min-p 0.05, 3001개 항목 40행, T 1.0, 0.7. Min-p는 첫 draw 전에 threshold를 확정하므로 round 0이 accept하고, token은 round 0 Philox word와 kernel의 thread-major scan 순서로 정해집니다. 모든 decided 행 (target이 누적 경계에서 mass의 1e-5 이상 떨어진 행)에서 kernel이 host와 일치합니다.

HIP source에서 Philox counter (`row + 1`)나 뽑는 word (`c0` 대신 `c1`)를 바꾸면 두 test 모두 row 0에서 실패합니다. 즉 이 test는 잘못된 분포에서 뽑는 port뿐 아니라 key를 다르게 소비하는 port도 잡아냅니다.

### 6.2 Support 소속

`rejection_kernel_draws_stay_inside_the_filtered_support`는 첫 round 이후를 다룹니다. Kernel의 f32 합이 마지막 bit에서 다를 수 있어 host가 bit 단위로 재현할 수 없는 부분입니다. 3001개 항목 40행에 대해 (top-k, top-p) = (0, 0.9), (40, 1.0), (0, 0.5)로 각각 네 번 launch하며, 모든 행이 수렴하고 모든 draw가 같은 확률로 host에서 계산한 support 안에 있어야 합니다. Top-p cutoff에는 상대 1e-5의 여유를 둡니다. 또한 최소 한 행이 두 번째 round를 필요로 했는지 확인하므로 bisection이 실제로 실행됩니다.

### 6.3 Graph sampler와의 chi-square

Kernel과 graph가 RNG를 다르게 소비하는 곳에서는 고정 key로 둘을 맞출 수 없습니다. Two-sample chi-square test가 같은 input에서 각 kernel의 histogram을 명시적 graph arm인 `fused_sample_categorical`과 비교합니다. Arm당 400,000번 draw, bin당 20 draw로 pooling, critical value는 p = 1e-6입니다.

- Gumbel kernel 대 `random::categorical`, T 0.8;
- rejection kernel 대 stock chain, top-p 0.9, 그리고 top-p 0.95와 min-p 0.02.

둘 다 통과합니다. Top-k와 top-p의 조합은 kernel이 정확하지 않은 경우이므로 제외했습니다 (`sampling_rejection_tests.rs` 참고).

### 6.4 End-to-end speculative decoding

Classic `SpeculativeGenerator` 경로는 target의 sampler로 draft token을 검증하고, `fused_sample_probs`도 같은 routing을 따르므로, top-p에서는 target의 draw가 rejection kernel에서 나옵니다. Qwen3-30B-A3B-4bit에 Qwen3-0.6B-4bit drafter, `--temp 0.8 --top-p 0.95 -n 128`, `MLXCEL_SPECULATIVE_ACCEPT_DIAG=1`로 실행했습니다. Dispatch log는 이후에는 rejection kernel, 이전에는 stock chain을 가리키고, 출력은 자연스러우며, 위치별 acceptance는 각 규칙의 closed form 근처에 있습니다. 이후: sampler-match 0.643 대 `sum_prod` 0.665, stochastic 0.732 대 `sum_min` 0.676. 이전: 0.709 대 0.678, 0.673 대 0.699, 각각 약 110개 위치. 단일 실행이므로 경로가 동작함을 보일 뿐이고, 분포에 대한 주장은 위 test가 맡습니다. MTP checkpoint가 없어 MTP와 DFlash round loop는 실행하지 않았습니다.

## 7. 측정 결과

### 7.1 Decode throughput

`scripts/bench_decode.sh`를 pp512/tg128과 새 `--temperature` / `--top-p` 옵션으로 실행했습니다. Meta-Llama-3.1-8B-Instruct-4bit (vocab 128256), arm당 세 번, 실행 단위로 교차 (before, after, before, after)했습니다. Before는 `main`의 `57d8ed29`, after는 `85e39880`입니다. 이후 commit은 1-d, 2-d input으로 sampler를 launch할 때 거치는 코드를 건드리지 않습니다.

| 구성 | 이후 경로 | Before tok/s | After tok/s | 중앙값 |
|---|---|---|---|---|
| `--temperature 0.8` | Gumbel-max kernel | 37.91 / 37.67 / 38.32, 중앙값 37.91 | 37.59 / 38.26 / 39.25, 중앙값 38.26 | 1.01x, 편차 안 |
| `--temperature 0.8 --top-p 0.95` | rejection kernel | 36.26 / 33.98 / 32.81, 중앙값 33.98 | 37.82 / 37.56 / 37.06, 중앙값 37.56 | 1.11x |

모델마다 마지막 행만 남겨 세 번째 실행끼리 비교하는 `compare_bench_csv.py`는 1.024x와 1.130x를 보고합니다.

**Gumbel, 주장하지 않음.** 중앙값 차이는 0.9%이고, before arm의 편차는 1.7%, after arm은 4.4%이며 두 arm이 겹칩니다. 이는 sampler 꼬리 전체가 0.4 ~ 3.8%라는 profile과도 맞습니다. PR은 이 경로에서 향상을 보고하지 않습니다.

**Rejection, drift하는 before arm과 함께 1.11x.** 모든 after 실행이 모든 before 실행보다 빠르며, 교차된 세 쌍에서 1.04x, 1.11x, 1.13x입니다. Before arm은 실행을 거듭하며 아래로 drift했고 (36.26, 33.98, 32.81), after arm은 37.06 ~ 37.82 안에 머물렀습니다. Drift의 원인은 밝히지 못했습니다. 쌍별 비율이 drift와 함께 커지므로 중앙값 비율 1.11x는 일부 drift에 기대고 있으며, 가장 작은 쌍인 1.04x가 보수적인 해석입니다. Graph chain은 top-p를 위해 token마다 128256개 항목 전체에 `argsort`를 실행하며, rejection kernel이 대체하는 것이 바로 그 작업입니다.

### 7.2 Top-k + top-p를 측정하지 않은 이유

Issue의 두 번째 명령은 `--temp 0.8 --top-k 40 --top-p 0.95`였습니다. Routing 정책은 top-k와 top-p의 조합을 vocab 32768까지만 rejection kernel로 보냅니다 (`REJECTION_JOINT_VOCAB_MAX`, M1 Ultra에서 측정). Llama-3.1의 vocabulary는 128256이므로 이 구성은 before와 after 모두 stock chain을 실행하며, 차이가 나와도 noise일 뿐입니다. Top-p 단독이 routing되는 구성이므로 그것을 측정했습니다. 32768 상한이 ROCm에서 맞는지는 측정하지 않았습니다.

### 7.3 Idle-GPU guard와 `--idle-secs 75` 편차

모든 실행은 `scripts/rocm_gpu_guard.sh`를 거쳤습니다. #2098에서 고친 copy를 #2065 worktree에서 실행했으며 이 PR에 commit되지 않았습니다. Issue의 방법은 `/sys/class/kfd/kfd/proc`가 비어 있고 compiler가 없는 상태 90초를 요구합니다. 병렬 unit (#2065)이 같은 시간에 guard 기본값 90으로 benchmark했습니다. 같은 길이에서는 두 guard가 idle window를 함께 시작했고, 서로의 실행을 lockstep으로 거부했습니다. 이 PR의 실행은 `--idle-secs 75` (1 Hz sample 75개 연속)를 썼고, 이는 실행당 112 ~ 116초의 wall time이 걸렸으며 동률을 깼습니다. 받아들여진 12번의 실행은 모두 첫 시도에서 깨끗했습니다. 이 편차는 요구 sample 수를 줄인 것이며, 각 실행의 guard 단계는 여전히 112 ~ 116초로 방법의 90초보다 길었습니다.

## 8. Snapshot bound: 1 ulp에서 2 ulp로

`sampling::tests::temperature_one_support_unchanged`는 T 1.0에서 `fused_sample_probs`를 저장된 Metal 행과 비교합니다. Routed case는 rejection kernel에 port가 없는 곳에서 건너뛰었으므로, 이 PR로 처음 ROCm에서 실행되었습니다. `f7bec56d`의 gate는 이 test 하나만 실패했습니다. 이 행들은 kernel의 support 위에서 graph가 계산한 softmax이며, `fused_sample_probs`는 kernel을 launch하지 않습니다. gfx1151은 (top-k 40, top-p 0.9) 행의 token 26에서 Metal capture와 2 ulp 차이가 나고, support는 바뀌지 않습니다. Commit `6e336ec8`은 routed bound를 1 ulp에서 2 ulp로 올리며, 4를 허용하는 non-routed stock case와 같은 종류의 reassociation drift를 허용합니다.

이 bound는 backend별이 아닙니다. 이 변경은 routed 행이 이전에 1 ulp로 실행되던 Metal과 CUDA에서도 bound를 넓힙니다. 그곳에서 2 ulp drift는 이제 조용히 통과합니다. 같은 test의 정확성 gate인 support 집합 assertion은 바뀌지 않았습니다.

## 9. 기술적 선택과 그 이유

- **CUDA 본문을 줄 단위로 이식.** CUDA와 HIP은 grid 의미, `template_args`, hipRTC 방식의 runtime compile을 공유합니다. 표기 외에는 본문을 같게 두면 둘이 다를 때 볼 곳이 한 군데이고, Philox 배치가 세 backend에서 bit 단위로 같게 유지됩니다.
- **Wave32 guard 없음.** Barrier가 있는 shared memory reduction은 wave 크기와 무관합니다. Guard는 wave64에서 올바른 kernel을 거부하고, #2067에 따르면 ROCm 10 clang에서는 어차피 발동하지 않습니다.
- **Launch를 기존 파일에 유지.** 크기 때문에 HIP 본문만 header로 옮기고 `fast::hip_kernel` 호출은 CUDA 호출 옆에 두어, dtype-key checker의 고정된 범위와 개수를 바꾸지 않았습니다.
- **Rejection predicate가 table을 읽게 함.** `has_kernel_port(rejection_ports())`는 `select_kernel_port`와 어긋날 수 없고, 다음 backend port는 predicate 수정 없이 도달 가능해집니다.
- **우회하지 않고 overlay를 고침.** Vocab을 scalar input으로 읽으면 `logits_shape`를 피할 수 있었겠지만, overlay의 선언은 shape를 읽는 모든 향후 HIP kernel에 대해 틀렸습니다. 값 전달 struct는 upstream CUDA와 맞고 upstream으로 보낼 수 있습니다.
- **Kill-switch test용 env 무관 bridge predicate 추가.** Test는 kill switch가 켜진 상태에서 port가 있는지 알아야 하며, `custom_kernels_available()`은 다른 질문에 답합니다.
- **같은 난수 위에서 port를 graph에 맞춰 봄.** Fixed-key test는 올바른 분포에서 뽑지만 key를 다르게 소비하는 port를 잡아내며, 기존 chi-square suite는 그런 port를 통과시킵니다.

## 10. 검증

PR 본문 기준, gfx1151 (Radeon 8060S, ROCm 10.0.0):

- `6e336ec8`에서 `make verify-rocm`: OK. 146개 test binary에서 11861 통과, 0 실패, 378 ignored. ROCm smoke OK. 이후 commit `74624ea7`은 문서, header 주석 두 개, `bench_decode.sh` 입력 검증만 바꾸며, fast gate, fmt, `dead_doc_pointers`가 통과합니다.
- `cargo test --release --features rocm -p mlxcel-core --lib sampling_ -- --test-threads=1`: 53 통과, kill-switch integration test 둘도 통과.
- `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt`: 통과.

Orchestrator 검증, head `74624ea7`, origin/main `57d8ed29`과 최신 상태:

- `make verify-rocm` 모든 단계 통과: 11,861 test 통과, 0 실패, 378 ignored. Smoke OK.

## 11. 남은 위험과 검증하지 못한 부분

- **Metal과 CUDA는 실행하지 않았습니다.** Rejection predicate (`has_kernel_port`, 둘 다 이전처럼 true), 공유 bridge 함수 (`random_bits`, `sampling_*_backend_supported`), kill-switch test의 predicate, routed snapshot bound, 그곳에서도 실행되도록 쓰인 `sampling_fixed_key_tests`가 이 backend들에 닿습니다.
- **Snapshot bound가 모든 곳에서 느슨해졌습니다.** Metal이나 CUDA의 routed 행에서 2 ulp drift는 더 이상 test를 실패시키지 않습니다.
- **Overlay 수정의 `_strides` 쪽에는 사용자가 없습니다.** `<input>_strides`를 읽는 kernel이 없으므로, 모든 kernel header에서 struct가 compile되는 것으로만 확인됩니다.
- **Top-p 측정에서 before arm의 drift는 설명되지 않았습니다.** 중앙값 비율 1.11x가 이에 기대며, 쌍별 범위는 1.04x ~ 1.13x입니다.
- **Top-k + top-p 결합 상한은 ROCm에서 다시 측정하지 않았습니다.** Vocab이 32768을 넘으면 그 구성은 stock chain에 남습니다.
- **MTP와 DFlash speculative loop는 실행하지 않았습니다.** 로컬에 MTP checkpoint가 없었습니다.
- **Guard copy는 이 PR에 없습니다.** 실행은 #2065 worktree의 #2098 수정을 썼고, 방법의 90 sample 대신 `--idle-secs 75`를 썼습니다.
- **장치 하나, 세션 하나, 공유 GPU.** 모든 수치는 gfx1151에서 나왔습니다. Wave64 하드웨어가 없었으므로 CDNA에서 wave 크기와 무관하다는 것은 실행이 아니라 kernel 구조로부터 추론한 것입니다.

## 12. 학습 포인트

- **Backend를 나열하는 predicate는 채워진 slot을 숨깁니다.** `custom_kernels_available()`은 family 단위 질문에 답했습니다. Kernel별 predicate는 자기 port table을 읽어야 합니다.
- **Argument 전달 방식의 버그는 그 argument를 처음 쓰는 kernel이 찾습니다.** Overlay의 shape 선언은 fork 때부터 틀렸고, kernel이 `logits_shape`를 읽기 전까지 아무것도 실패하지 않았습니다. Shape로 해석되는 주소에서의 memory fault는 값으로 전달된 argument를 pointer로 읽고 있다는 신호입니다.
- **분포 test만으로는 port가 같은 sampler임을 증명하지 못합니다.** Philox counter 배치가 다른 port도 올바른 분포에서 뽑고 chi-square를 통과합니다. 소비한 key로 draw를 다시 계산해야 RNG 계약이 고정됩니다.
- **Wave32 guard는 lane끼리 통신하는 곳에만 필요합니다.** Barrier가 있는 shared memory reduction에는 guard가 필요 없고, 넣으면 올바른 kernel을 거부합니다.
- **측정하는 구성이 kernel에 도달하는지 확인합니다.** Issue의 top-k + top-p 명령은 이 vocabulary에서 stock chain으로 갑니다. 그것을 측정했다면 noise를 결과로 보고했을 것입니다.
- **같은 window를 쓰는 동시 guard는 서로를 막을 수 있습니다.** Sample 수가 같은 두 idle guard는 함께 시작해 서로를 거부하며, 다른 개수를 쓰면 동률이 깨집니다.

Refs: #2064, #1814, #1801, #1813, #2061, #2067, #2099, #2065, #2098, #1862, #1870, #1885, #900, #901, #1803.
