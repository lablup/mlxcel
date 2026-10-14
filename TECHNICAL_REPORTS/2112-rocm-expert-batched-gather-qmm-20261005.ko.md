# 기술 보고서: PR #2112 - Expert-batched gather_qmm prefill kernel 수정과 기본 활성화

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 측정 완료. 코드 head `c587178e`, origin/main `df05f1f9` (#2107 `022067ad` 포함) 위로 rebase, 머지 대기 중.

**언어**: HIP C++ (ROCm overlay의 `quantized/qmm.hip`), Rust (신규 ROCm integration test, doc comment 하나), Markdown (결과 문서, `LOCAL_FIXES.md` item 9, upstream packaging 표)

**위험도**: 중간 (ROCm 기본 동작이 바뀝니다. Expert가 64개 이하인 MoE의 정렬된 affine bf16/f16 prefill이 이제 다시 작성한 kernel로 실행됩니다. Metal과 CUDA build는 `patches-rocm/`를 복사하지 않으므로 영향이 없습니다)

## 요약

Issue #2066 (#1814의 일부, epic #1801)은 ROCm overlay의 expert-batched `gather_qmm` kernel이 왜 bf16에서 틀린 결과를 내는지 (`LOCAL_FIXES.md` item 9, 이 때문에 opt-in으로 바뀌었음), 그리고 이 kernel이 gfx1151의 MoE prefill을 빠르게 할 수 있는지를 물었습니다. 당시 Mixtral-8x7B-4bit의 prefill은 약 26 tok/s였습니다.

결함은 산술이 아니라 index 읽기에 있었습니다. Kernel은 `lhs_indices[b]`와 `rhs_indices[b]`를 평평한 `[B]` 배열로 읽었는데, MLX는 두 index 배열을 activation의 batch shape에 맞춰 복사 없이 broadcast합니다. Broadcast된 호출에서 kernel은 두 배열의 끝을 넘어 읽었습니다. bf16만 실패한 이유는 gate가 bf16만 이 kernel로 보냈기 때문입니다. 새 test는 이를 재현했고, dequantize한 f32 reference 대비 상대 L2 오차가 1.415였습니다 (unsorted 경로는 2.3e-3).

이 PR은 두 index 배열을 batch shape와 stride로 읽게 하고, inner loop를 다시 작성하며 (index 수정만으로는 대체 대상인 per-row wide kernel보다 2.3배 느렸음), f16을 추가하고, issue의 규칙에 따라 kernel을 기본으로 켭니다. gfx1151에서 512-token prefill은 granite-4.0-h-tiny-4bit에서 535.7에서 918.6 tok/s로 (1.71배), Mixtral-8x7B-Instruct-v0.1-4bit에서 25.95에서 125.6 tok/s로 (4.84배) 올랐고 decode는 변하지 않았습니다. 48개 test case 모두 expert-batched 오차가 unsorted 경로 오차의 0.1% 이내이며, logit trace는 #1809 Metal reference 대비 decided mismatch가 0입니다.

## 1. 문제 정의

### 1.1 Kernel과 꺼져 있던 이유

`GatherQMM::eval_gpu`는 정렬되고 transpose된 affine, group size 64, 4 또는 8 bit 호출 중 `M == 1`, `B >= 64`, `E <= 64`, `B / E >= 4`인 것 (expert가 64개 이하인 MoE의 prefill)을 `gather_qmv_expert_batched_kernel`로 보냅니다. 이 kernel은 expert마다 weight를 한 번 읽어 그 expert의 모든 row에 쓰고, per-row gather kernel은 route된 row마다 weight를 다시 읽습니다. Item 9는 bf16에서 unsorted 경로 대비 1.0이 넘는 상대 오차를 기록했고 (f32와 f16은 정상), 원인 없이 kernel을 opt-in (`MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`)으로 바꿨습니다.

### 1.2 Prefill 시간이 쓰인 곳

현재 gate로는 baseline의 느린 두 checkpoint 어느 쪽에도 이 kernel이 닿지 않았기 때문에, issue는 profile을 먼저 요구했습니다. 512-token prefill을 `rocprofv3 --kernel-trace --stats`로 측정하고 bench의 phase mark 사이를 잘라낸 결과입니다.

| Model | Activation, expert 수, top-k | 호출당 row (B) | 측정된 prefill | 지배적인 kernel | 비중 |
|---|---|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1-4bit | f16, 8, 2 | 1024 | 20.3 s | `gather_qmv_warp_shared_kernel<__half, ...>` (호출당 210 ms) | 99.1% |
| gpt-oss-20b-MXFP4-Q4 | bf16, 32, 4 (mxfp4) | 2048 | 67.8 s | `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` (호출당 940 ms) | 99.8% |
| granite-4.0-h-tiny-4bit | bf16, 64, 6 | 3072 | 0.96 s | `gather_qmv_wide_kernel<hip_bfloat16, ...>` (호출당 4.9 ms) | 61.8% |

세 kernel 모두 per-row gather kernel입니다. 호스트에 있는 checkpoint 중 bf16 전용 gate가 닿는 것은 granite뿐이었습니다. Mixtral의 f16 prefill은 expert-batched kernel이 대체하는 종류의 per-row kernel에서 돌았고, issue의 step 3은 f16 추가 전에 바로 이것을 확인하라고 요구했습니다. gpt-oss는 item 10의 non-affine fallback에서 돌며, 이는 어떤 affine kernel로도 대체할 수 없습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `patches-rocm/.../quantized/qmm.hip` | `gather_index_loc`, `find_sorted_expert_run` helper, 두 expert-batched kernel의 stride 기반 index 읽기, inner loop 재작성, grid z를 expert로, f16 dispatch, 기본 활성화 |
| `tests/rocm_gather_qmm_expert_batched.rs` (신규) | Unsorted 경로와 dequantize한 f32 reference에 대한 48개 case, ROCm 전용 |
| `patches-rocm/LOCAL_FIXES.md` | Item 9를 원인, 재작성, 측정, 기본값으로 다시 작성하고 #1813 upstream 후보로 표시 |
| `docs/mlxcelverse/upstream/README.md` | Item 9를 "Held"에서 "Not packaged yet"으로 옮기고 fork package에 필요한 것 기록 |
| `docs/benchmark_results/rocm-moe-prefill-gfx1151-2026-10-05.md` (신규) | Profile, 원인, 정확성, prefill과 decode 수치, 짧은 prompt sweep, 결정 |
| `src/models/switch_layers_mxfp_tests.rs` | Doc comment: kernel이 기본으로 켜져 있고 전용 test가 있음 |

Commit은 세 개입니다. 수정과 활성화 (`fc868bdf`), test bound와 문서 표현을 다듬은 review 후속 (`87d5f6d3`), 짧은 prompt sweep (`c587178e`). origin/main 대비 diff는 6개 파일, 762줄 추가, 246줄 삭제입니다.

## 3. Item 9의 근본 원인

### 3.1 Broadcast된 index를 평평하게 읽음

Kernel은 평평하게 펼친 batch의 index `b`를 받아 `lhs_indices[b]`와 `rhs_indices[b]`를 읽었습니다. 이는 두 index 배열이 연속된 `[B]` 배열일 때만 맞습니다. MLX는 index 배열을 activation의 batch shape에 맞춰 실제로 만들지 않고 broadcast하므로, index 배열은 broadcast 축에서 stride가 0일 수 있습니다. Item 9를 발견한 호출은 `x` `[T, 1, K]`와 정렬된 `rhs_indices` `[T, 1]`이었습니다. Batch는 `[T, T]`로 broadcast되고, rhs stride는 `(1, 0)`, implicit lhs stride는 `(0, 1)`이며, `B = T * T`입니다. `b`를 `T * T - 1`까지 평평하게 읽으면 원소가 `T`개인 두 배열의 끝을 넘어갑니다.

### 3.2 bf16만 실패한 이유

f32와 f16은 이 kernel에 도달한 적이 없습니다. Gate가 `x.dtype() == bfloat16`을 요구했기 때문입니다. 예전 item 9의 "f32와 f16은 정상"은 이 kernel이 아니라 다른 kernel에 대한 관찰이었습니다.

### 3.3 Model이 영향을 받지 않은 이유

`SwitchGLU`의 정렬 경로는 `x`를 expert 순으로 gather하고 평평한 `[B]` index를 넘기므로 평평한 읽기로도 맞았습니다. 결과 문서는 예전 kernel을 granite의 `w256` trace에 강제로 켜도 Metal 대비 decided mismatch가 0이었다고 기록합니다. 결함은 broadcast index를 쓰는 직접적인 `gather_qmm` 호출, 그리고 앞으로 그런 index를 만드는 호출자에서 드러날 수 있었습니다.

### 3.4 재현 test

`tests/rocm_gather_qmm_expert_batched.rs`는 네 가지 index layout을 다룹니다. `Gathered` (`SwitchGLU`의 shape), `Shared` (activation row 하나, `x`의 stride 0), `BroadcastRows` (`x` `[T, 1, 1, K]`, `rhs` `[T, top_k]`), `BroadcastIndices` (`x` `[T, 1, K]`, `rhs` `[T, 1]`, 원래 호출). 예전 kernel에서 `BroadcastRows`는 dequantize한 f32 reference 대비 상대 L2 오차 1.415였고, unsorted 경로는 2.3e-3이었습니다. 새 kernel에서 stride 읽기만 되돌려도 test가 다시 실패하므로, test는 재작성이 아니라 수정 자체를 고정합니다.

## 4. 수정

### 4.1 Stride 기반 index 읽기

`gather_index_loc(b, batch_shape, strides, batch_ndim)`은 평평한 batch 원소 `b`를 index 배열 안의 위치로 바꿉니다. 1차원 batch이면 `b * strides[0]`, 아니면 `elem_to_loc`입니다. `qmm.hip`에서 실제로 launch되는 per-row gather kernel들이 이미 이 방식으로 index를 읽습니다. Launcher는 그 kernel들을 위해 이미 계산한 collapse된 batch shape와 lhs, rhs stride를 넘깁니다. Activation row 조회와 run 탐색 모두 이 함수를 씁니다.

`find_sorted_expert_run`은 한 expert에 route된 batch 원소의 run을 stride 기반 rhs 배열에 대한 이진 탐색 두 번 (lower bound, upper bound)으로 찾습니다. `right_sorted`가 broadcast된 batch의 평평한 순서로 정렬되어 있다는 뜻이라는 점에 기댑니다.

Launch되지 않는 `gather_qmv_idot_expert_batched_kernel`도 같은 결함을 갖고 있었으므로 같은 읽기와 run 탐색을 쓰게 했습니다. 나중에 연결되더라도 결함이 되살아나지 않습니다.

### 4.2 Index 수정만으로 부족했던 이유

Stride 읽기만 넣었을 때 kernel은 granite prefill에서 1.33 s를 썼고, 대체 대상인 wide kernel은 0.58 s였습니다 (호출당 11.1 대 4.9 ms). 2.3배 느렸습니다. 정확하지만 더 느린 kernel을 켜면 모든 대상 model에서 prefill이 좋아져야 한다는 issue의 규칙을 통과하지 못합니다. 예전 inner loop에 원인이 네 가지 있었습니다.

- **Row마다 weight를 다시 읽음.** Run의 row loop가 가장 바깥에 있어서 row마다 packed word와 group의 scale, bias를 모두 다시 읽었습니다. Expert의 weight를 row 사이에 공유한다는 kernel의 목적은 cache에 맡겨져 있었습니다.
- **Group마다 reduction.** Column의 16개 lane이 64개 원소 group마다 partial sum 두 개를 lane 사이에서 reduce했습니다.
- **4 bit에서 노는 lane.** Lane마다 한 step에 값 8개를 맡고 stride가 `16 * 8 = 128`이었지만 group은 값 64개라서, lane 0에서 7만 일을 했습니다.
- **직렬 run 탐색.** Grid z는 `min(B, E)`개의 run slot이었고, block z는 앞선 run 경계를 모두 걸어서 z번째 run을 찾았으므로 뒤쪽 block일수록 직렬 작업이 많았습니다.

### 4.3 재작성

- 각 lane은 packed 32-bit word 하나와 그 group의 scale, bias를 읽고, 다음 word를 읽기 전에 run의 `TOKENS = 4`개 row에 적용합니다. Expert의 weight를 row마다가 아니라 4개 row마다 한 번 읽습니다.
- 각 row는 word마다 `scale * qx + bias * sum(x)`를 register에 누적하고, lane 사이 reduction은 group마다가 아니라 끝에서 row당 한 번만 합니다.
- K loop는 `16 * VALS` stride로 word 단위를 돌므로 4 bit와 8 bit 모두 column의 16개 lane이 일합니다.
- Grid z는 `E`입니다. Block z는 expert z이고 이진 탐색 두 번으로 run을 찾습니다. Run이 비어 있는 block은 탐색 후 바로 반환합니다.
- Run 끝을 넘는 row는 run의 첫 row를 재사용해서 unroll된 loop에 분기가 없게 하고, 그 합은 저장하지 않습니다.
- Run 탐색이 모든 bounds 반환보다 먼저 일어나므로 block의 모든 thread가 `__syncthreads()`에 도달합니다 (예전 kernel은 barrier 전에 `row >= M || col >= N`으로 반환했습니다).
- Launcher는 block의 x 차원을 instantiate된 `THREADS_PER_COL`에 맞춰 16으로 고정합니다. 예전에는 `select_qmv_threads_per_col`에서 값을 받았습니다.
- `static_assert`로 kernel을 gate와 같은 affine 4/8 bit로 제한하고, 도달할 수 없던 일반 bit와 non-affine 분기를 뺐습니다.

같은 profile에서 다시 작성한 kernel은 granite prefill에서 0.32 s (호출당 2.7 ms), Mixtral prefill에서 4.52 s (호출당 47 ms, warp-shared kernel은 210 ms)를 씁니다.

### 4.4 f16

Kernel은 원소 type으로 template되어 있습니다. Profile 결과 (Mixtral prefill의 99.1%가 f16 per-row kernel 하나)를 근거로 gate가 `bfloat16`과 함께 `float16`도 받도록 했고, launcher는 item 12가 warp-shared kernel에서 한 것처럼 `__half`나 `hip_bfloat16`으로 dispatch합니다. Issue는 dtype별 instantiation이 `indexing.hip`의 compile을 26 s에서 348 s로 늘린 적이 있어서 (item 17) 새 instantiation을 dtype 하나로 제한했습니다. f16은 instantiation 두 개 (4 bit, 8 bit)를 더합니다. Build의 `hipcc` 명령 그대로 `qmm.hip`을 compile하면 37.5 s와 37.7 s였고 main은 38.4 s와 38.0 s였으므로 compile 시간은 변하지 않았습니다. 더 단순해진 kernel이 추가 instantiation을 상쇄합니다.

### 4.5 기본값

Issue는 규칙을 정해 두었습니다. 대상이 되는 모든 dtype이 dequantize한 f32 reference 대비 unsorted 경로 자체의 오차 안에서 일치하고, 측정한 모든 대상 model에서 512-token prefill이 좋아지며 decode는 변하지 않을 때만 기본으로 켭니다. granite와 Mixtral의 bf16, f16에서 두 조건 모두 성립하므로 (5절, 6절) `parse_warp_kernel_env("MLX_ROCM_GATHER_QMV_EXPERT_BATCHED", ...)`의 기본값은 이제 true입니다. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`으로 끌 수 있고, 이 변수는 여전히 호출마다 읽습니다.

## 5. 정확성 근거

### 5.1 48개 case test

`cargo test --release --features rocm --test rocm_gather_qmm_expert_batched -- --test-threads=1`는 kernel을 강제로 켜고 shape 3개 (expert 8개, top 2, K 4096, 마지막 column block이 부분적이도록 N 516인 Mixtral 유사 layer, 그리고 expert 64개, top 6인 granite의 gate/up과 down projection) x index layout 4개 x 4/8 bit x bf16/f16, 총 48개 case를 돌립니다. 각 case는 다음을 확인합니다.

- Dequantize한 weight로 expert별 dense f32 matmul을 한 reference 대비 unsorted와 expert-batched의 상대 L2 오차가 모두 dtype 반올림 bound (f16 2e-3, bf16 1e-2) 이하;
- expert-batched 오차가 unsorted 경로 오차의 1.25배 이하;
- 정렬 실행 두 번의 결과가 bit 단위로 동일.

측정값으로는 모든 case에서 expert-batched 오차가 unsorted 경로 오차의 0.1% 이내입니다 (f16 2.8e-4에서 2.9e-4, bf16 2.2e-3에서 2.3e-3, 최대 비율 1.0007). 1.25배는 합산 순서를 위한 여유이지 관측된 차이가 아닙니다.

### 5.2 Metal 대비 logit trace

Kernel을 켠 teacher-forced trace를 #1809 matrix가 쓰는 Metal reference와 `compare_logit_traces.py --decided 2.0`으로 비교한 결과입니다.

| Model, window | Metal reference | Top-1 불일치 | Decided mismatch |
|---|---|---|---|
| qwen3-30b-a3b, w8 | `metal_m1u_bec64748` | 17 / 640 | 0 / 319 |
| qwen3-30b-a3b, w256 | `metal_m1u_bec64748` | 23 / 512 | 0 / 248 |
| mixtral-8x7b-instruct, w8 | `metal_m1u_bec64748` | 1 / 640 | 0 / 345 |
| mixtral-8x7b-instruct, w256 | `metal_m1u_bec64748` | 2 / 512 | 0 / 245 |
| granite-4.0-h-tiny, w8 | `metal_m5_d1128266` | 20 / 640 | 0 / 260 |
| granite-4.0-h-tiny, w256 | `metal_m5_d1128266` | 23 / 512 | 0 / 208 |

여섯 행 모두 decided mismatch가 0입니다. Qwen3-30B-A3B는 expert가 128개라 이 kernel에 닿지 않으므로 대조군입니다. granite는 예전 ROCm per-row trace (`rocm_gfx1151_c5fe9a16`)와 비교해도 두 window 모두 decided mismatch가 0입니다. Trace는 commit되지 않았습니다.

### 5.3 Gate

PR 기준: `make verify-rocm` (`MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`)이 11,870 passed, 0 failed, 378 ignored로 통과했고, 새 test에 대한 `-D warnings` clippy, `make verify-rocm-overlay`, `cargo test --test dead_doc_pointers`도 통과했습니다. 구현 review와 보안 review 모두 CRITICAL, HIGH 문제를 찾지 못했습니다.

Orchestrator가 branch를 충돌 없이 origin/main 위로 rebase했고 (branch는 이제 #2107 `022067ad`를 포함하는 `df05f1f9` 위에 있음), rebase된 head에서 전체 `make verify-rocm` gate를 다시 실행했습니다.

## 6. 성능

### 6.1 512-token prefill과 decode

`bench_decode.sh`의 shape (512-token prompt, 128개 생성 token)로 `mlxcel-bench-decode`를 binary 하나에서 kernel off와 on을 번갈아 세 round 돌렸고, 모든 실행은 `scripts/rocm_gpu_guard.sh --idle-secs 45` 아래에서 했습니다.

| Model | Prefill tok/s 이전 (median) | 이후 (median) | 변화 | Decode tok/s 이전 / 이후 |
|---|---|---|---|---|
| granite-4.0-h-tiny-4bit (bf16, expert 64개) | 535.73 | 918.59 | 1.71배 | 88.76 / 88.65 |
| Mixtral-8x7B-Instruct-v0.1-4bit (f16, expert 8개) | 25.95 | 125.63 | 4.84배 | 9.40 / 10.43 |

두 model 모두 모든 이후 실행이 모든 이전 실행보다 빠릅니다. Decode는 이 kernel에 닿지 않습니다 (decode step은 `B = top_k`로 gate의 64보다 작음). granite의 decode median 차이는 0.1%입니다. Mixtral의 decode 편차 (여섯 실행 전체에서 8.68에서 10.71 tok/s)는 31 GiB 호스트에서 26 GB checkpoint를 돌릴 때의 잡음이므로 +1 tok/s는 변화로 보지 않습니다.

### 6.2 Gate 경계의 짧은 prompt

Gate의 하한 (granite는 `B / E >= 4`, Mixtral은 `B >= 64`)에서 off와 on의 prefill 시간 (ms)입니다.

| Model | T (B) | Off | On |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 44 (264) | 102.4 / 102.9 / 121.2 | 82.7 / 84.2 / 84.9 |
| granite-4.0-h-tiny-4bit | 64 (384) | 126.6 / 127.1 / 129.7 | 95.9 / 96.7 / 97.5 |
| granite-4.0-h-tiny-4bit | 160 (960) | 236.7 / 250.6 / 296.5 | 148.8 / 162.5 / 170.1 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 32 (64) | 1709.7 / 1716.2 | 616.9 / 669.3 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 64 (128) | 2845.0 / 2888.2 | 836.5 / 855.2 |

모든 크기에서 kernel이 더 빠르므로, 512 token 미만에서도 기본값이 유지되기 위해 gate의 기존 threshold를 바꿀 필요가 없습니다.

## 7. 기술적 선택과 그 이유

- **Gate가 아니라 읽기를 고침.** Issue는 결함을 감추려고 gate를 좁히는 것을 금지했습니다. Stride로 읽으면 MLX가 넘길 수 있는 모든 index layout에서 kernel이 맞게 되고, gate가 f16을 받아들인 뒤에도 마찬가지입니다.
- **Per-row kernel의 indexing을 따름.** `gather_index_loc`은 launch되는 gather kernel들과 같은 shape-stride 방식을 쓰고 launcher가 이미 만드는 parameter를 받으므로, expert-batched 경로에 별도 indexing 규칙이 생기지 않습니다.
- **조정이 아니라 재작성.** Index 수정만으로는 2.3배 느렸고, 원인 (가장 바깥의 row loop, group마다의 reduction, 노는 lane, 직렬 run 탐색)이 구조적이어서 parameter 조정으로는 없앨 수 없었습니다.
- **Expert마다 block 하나.** Grid z = E와 이진 탐색 두 번이 `min(B, E)`개 run slot과 직렬 탐색을 대체합니다. Gate를 통과하는 expert는 최대 64개이므로 빈 block의 비용은 탐색 한 번입니다.
- **f16만 추가.** Profile이 뒷받침하는 dtype 하나만 더해 issue의 instantiation 제한을 지켰습니다. f32는 기존 경로에 남습니다.
- **기본값은 issue의 규칙을 따름.** 호스트에서 대상이 되는 모든 dtype과 model에서 두 조건을 측정했고, 끄는 스위치는 호출마다 읽힙니다.
- **mxfp4는 별도 issue.** gpt-oss의 지배적인 kernel은 affine kernel로 대체할 수 없는 non-affine fallback이므로, 이 PR을 넓히지 않고 #2106으로 분리했습니다.

## 8. 진행 과정 메모

Developer agent는 kernel 재작성 도중 API 사용량 한도에 걸려 중단되었습니다. 한도가 풀린 뒤 context를 유지한 채 재개되었고, worktree에 commit되지 않은 상태에서 이어서 재작성, f16 dispatch, test, 측정, PR을 마쳤습니다. 이후 orchestrator가 branch를 origin/main 위로 rebase하고 gate를 다시 실행했습니다 (5.3절).

## 9. 남은 위험과 후속 작업

- **알려진 LOW: 범위를 벗어난 rhs row는 쓰이지 않음.** Grid z가 `E`이므로 rhs index가 `>= E`인 row는 어느 block에도 속하지 않아 출력이 쓰이지 않습니다. Wide kernel은 그 자리에 0을 씁니다. MLX는 범위를 벗어난 gather index의 동작을 정의하지 않으므로 그대로 두었지만, 그런 row의 출력은 두 경로 사이에서 다릅니다.
- **mxfp4 prefill (#2106).** gpt-oss-20b는 그대로이며 prefill의 99.8%가 `gather_qmv_kernel`에 있습니다.
- **Item 9의 fork package (#1813).** Fork는 평평한 읽기를 가진 이 kernel을 기본으로 실행합니다. Package는 아직 준비해야 합니다. 패치되지 않은 fork에서 `x` `[T, 1, K]`와 `rhs_indices` `[T, 1]`로 정렬된 `gather_qmm`을 호출하는 재현이 필요하고, fork가 정확성 수정만 따로 받을 수 있도록 stride 읽기를 재작성과 분리해야 합니다.
- **Metal과 CUDA는 실행하지 않음.** 변경은 ROCm build만 복사하는 `patches-rocm/`와 doc comment 하나에 한정됩니다.
- **범위.** Expert가 64개를 넘는 model은 이 kernel에 닿지 않습니다. Gate가 닿는 다른 MoE 계열은 측정하지 않았습니다 (호스트에 checkpoint 없음). Wave64 (CDNA)는 test하지 않았고, 16-lane reduction은 두 wave 폭 모두에서 wave 안에 머뭅니다. Word 하나당 4개 row라는 값은 조정하지 않았습니다. idot 변형은 고쳤지만 여전히 launch되지 않습니다.

## 10. 학습 포인트

- **특정 dtype에서만 나는 증상은 gate에서 올 수 있음.** "bf16에서만 틀림"은 bf16 산술을 가리키는 것처럼 보였지만, 실제로는 bf16만 이 kernel로 route되었을 뿐입니다. Kernel의 수치를 분석하기 전에 어떤 입력이 그 kernel에 닿을 수 있는지부터 확인해야 합니다.
- **Broadcast된 index 배열은 평평하지 않음.** MLX gather index를 받는 kernel은 batch shape와 stride로 읽어야 합니다. 평평한 읽기는 한 호출자가 우연히 만드는 layout에서만 맞습니다.
- **호출자가 쓰지 않는 layout도 test해야 함.** `SwitchGLU` layout은 예전 kernel에서도 통과했습니다. Broadcast layout만 결함을 드러냈고, test는 stride 읽기를 되돌리면 실패하도록 수정을 고정합니다.
- **정확성 수정이 성능 차이를 드러낼 수 있음.** Kernel이 맞게 된 뒤에야 측정할 수 있었고, 대체하려던 경로보다 느렸습니다. 기본 활성화를 가능하게 한 것은 수정이 아니라 재작성입니다.

Refs: #2066, #1814, #1801, #1813, #1809, #2106.
