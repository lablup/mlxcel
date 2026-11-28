# 기술 보고서: PR #2164 - 정렬된 mxfp4 gather_qmm prefill을 expert 단위로 batch 처리

**날짜**: 2026-10-07

**상태**: gfx1151 호스트에서 구현 및 검증 완료. Head `e87ffa7d` (origin/main `5c851fc4`과 최신 상태 일치), PR open, 머지 대기 중.

**언어**: HIP C++ (`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/quantized/qmm.hip`), Rust (`tests/rocm_gather_qmm_expert_batched.rs`), Markdown (`LOCAL_FIXES.md`, benchmark 페이지 두 개)

**위험도**: 낮음에서 중간 (ROCm overlay 안의 kernel arm 하나와 dispatch 조건 하나이며 기본으로 켜집니다. Metal이나 CUDA 파일은 건드리지 않고, `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`이면 이전 dispatch로 돌아갑니다)

## 요약

Issue #2106 (#1814, epic #1801의 일부이며 #2066에서 분리됨)은 gfx1151에서 `gpt-oss-20b-MXFP4-Q4`의 512-token prefill이 약 67.8초 걸리고, GPU 시간의 99.78%가 per-row non-affine `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>`에서 호출당 약 940 ms로 쓰인다고 보고했습니다. `GatherQMM::eval_gpu`는 routing된 4개 row마다 expert weight를 한 번만 읽는 `gather_qmv_expert_batched_kernel`에 affine weight만 보냈기 때문에, mxfp4는 routing된 row마다 expert weight를 다시 읽었습니다.

이 PR은 그 kernel에 mxfp4용 `AFFINE=false` arm (group size 32, group당 E8M0 scale 1 byte, bias 없음)을 추가하고, bf16 mxfp4가 기존 gate와 switch 아래에서 이 kernel로 가도록 합니다. Template instantiation은 하나만 늘고 `qmm.hip` compile 시간은 그대로입니다. gfx1151에서 512-token prefill 중앙값은 7.53에서 526.3 tok/s (69.9배)로 올랐고 decode는 변하지 않았으며, per-row 경로와 비교한 `logit_trace`에서 decided 위치 불일치는 88개 중 0개, 81개 중 0개였습니다. #2066이 정한 규칙에 따라 기본으로 켜지며, upstreaming 표시 없이 `LOCAL_FIXES.md` item 31로 기록됩니다.

## 1. 문제 정의

### 1.1 Prefill 시간이 쓰인 곳

gpt-oss-20b는 expert 32개에 top-4 routing이므로 512-token prefill은 정렬된 row `B = 2048`개, expert당 약 64 row를 만듭니다. `GatherQMM::eval_gpu`의 빠른 gather 경로 (expert-batched, wide, tiled, warp-shared)는 모두 `mode_ == QuantizationMode::Affine` 조건 아래에 있었습니다. 그래서 mxfp4는 `LOCAL_FIXES.md` item 10이 정확성을 위해 고친 generic dispatch, 즉 batch 원소마다 thread-block row 하나를 쓰는 `gather_qmv_kernel<T, uint8_t, 4, 32, false>`로 떨어졌습니다. Prefill의 각 MoE 호출은 routing된 row마다, 즉 약 64번씩 각 expert의 weight를 다시 읽었습니다. #2066 benchmark 페이지의 `rocprofv3` profile은 이 kernel에 72회 호출, 67.7초를 기록했습니다.

### 1.2 Issue가 요구한 것

Affine과 같은 형태의 정렬된 expert-batched mxfp4 경로를 필요한 instantiation만으로 만들고 (`qmm.hip`의 ahead-of-time compile 비용은 `LOCAL_FIXES.md` item 17에 기록됨), gpt-oss shape (`K = 2880`, `E = 32`, top 4)에서 unsorted 경로와 dequantize한 f32 reference에 비교하는 test, guard 아래에서 측정한 prefill 전후 수치, 변하지 않은 decode를 요구했습니다. 네 번째 기준 (LOCAL_FIXES 항목을 #1813의 upstreaming 후보로 표시)은 2026-10-06 fork 정책으로 대체되었습니다. 6절을 보십시오.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `qmm.hip`, decode helper | 새 `fp4_e2m1_to_float_scaled` (fp16 bit를 거치는 branch 없는 nibble decode)와 `kFp4HalfBitsScale = 16384.0f` |
| `qmm.hip`, `gather_qmv_expert_batched_kernel` | `static_assert`를 `!AFFINE && BITS == 4 && GROUP_SIZE == 32`까지 허용하도록 넓힘. Nibble decode와 누산 단계에 `if constexpr` 분기 |
| `qmm.hip`, `GatherQMM::eval_gpu` | Gate를 `eb_affine`과 `eb_mxfp4` (bf16 activation, group size 32, 4 bit)로 나눔. Bias pointer를 null로 넘기는 `<hip_bfloat16, uint8_t, 4, 32, false, 16>` launch 하나 추가 |
| `tests/rocm_gather_qmm_expert_batched.rs` | 고정된 affine 상수 대신 `Scheme` 타입, optional bias, `MXFP4_SHAPES`, 새 mxfp4 test, `K >= 2048`에서의 dispatch 확인 |
| `LOCAL_FIXES.md` | 새 item 31. Item 9와 10이 이를 가리킴 |
| `docs/benchmark_results/` | 새 `rocm-moe-prefill-mxfp4-gfx1151-2026-10-06.md`. #2066 페이지의 "Not measured" 줄이 이 페이지로 연결 |

Commit은 두 개입니다. Kernel, dispatch, test, 문서 (`826f9d61`), 그리고 dispatch 확인, SAFETY 주석, LOCAL_FIXES 문구를 더한 review 후속 (`e87ffa7d`). origin/main 대비 diff는 파일 5개, 302줄 추가, 61줄 삭제입니다.

## 3. 설계

### 3.1 Expert-batched kernel의 mxfp4 arm

Affine kernel의 inner loop는 이미 mxfp4에 맞는 형태였습니다. 각 lane이 packed 32-bit word 하나를 읽어 다음 load 전에 expert run의 4개 row (`TOKENS = 4`)에 적용하고, column의 16개 lane은 row마다 한 번 reduce합니다. 4 bit에서 word 하나는 값 8개를 담고, group size가 32이며 lane이 word 단위로 움직이므로 한 word의 nibble 8개는 모두 같은 group에 속합니다. mxfp4 arm은 이 loop 안에서 두 가지만 바꾸고 schedule은 그대로 둡니다.

**Decode.** e2m1 nibble은 sign 1 bit, exponent 2 bit, mantissa 1 bit입니다. `fp4_e2m1_to_float_scaled`는 이를 fp16 bit 패턴에 배치합니다.

```cpp
const uint16_t h =
    static_cast<uint16_t>(((nibble & 0x8u) << 12) | ((nibble & 0x7u) << 9));
return static_cast<float>(__builtin_bit_cast(_Float16, h));
```

Sign은 fp16 bit 15로, exponent bit는 fp16 exponent의 가장 낮은 두 bit (10, 11)로, mantissa bit는 fp16 mantissa의 최상위 bit (9)로 갑니다. e2m1 exponent `e`가 1에서 3이면 fp16 값은 `2^(e-15) * (1 + m/2)`이고 e2m1 값은 `2^(e-1) * (1 + m/2)`입니다. `e = 0`이면 fp16 값은 subnormal 0 또는 2^-15이고 e2m1 값은 0 또는 0.5입니다. 따라서 모든 code point가 정확히 e2m1 값의 2^-14배가 되며, branch도 lookup table도 없고, `v_cvt_f32_f16`이 이를 정확하게 넓힙니다. 같은 파일의 기존 `fp4_e2m1_to_float`는 값에 따라 switch합니다.

**Rescale.** Row마다 arm은 word의 decode된 값 8개에 대해 `qx = sum(x_j * w_j)`를 누산한 뒤 `acc[t] = fmaf(scale, qx * kFp4HalfBitsScale, acc[t])`를 계산합니다. f32에서 2^14 곱셈은 정확하므로 decode의 계수를 반올림 없이 되돌리고, `scale`은 kernel의 기존 `load_scale_value<ScaleT, GROUP_SIZE, AFFINE>`가 변환한 group의 E8M0 byte입니다. Affine arm은 `fmaf(scale, qx, fmaf(bias_val, xs, acc[t]))`를 유지하고, mxfp4 arm은 `xs`와 `bias_val`을 버리며, launch는 null bias pointer와 `has_bias = false`를 넘깁니다.

### 3.2 bf16 mxfp4만 추가한 이유와 compile 비용

`qmm.hip`은 target마다 ahead-of-time으로 compile되고, `LOCAL_FIXES.md` item 17은 instantiation 하나하나가 그 비용을 늘린다고 기록합니다. gpt-oss는 bf16 activation을 쓰고 호스트에 있는 유일한 mxfp4 MoE checkpoint이므로, PR은 instantiation을 정확히 하나, `gather_qmv_expert_batched_kernel<hip_bfloat16, uint8_t, 4, 32, false, 16>`만 추가합니다. f16 mxfp4와 mxfp8은 계속 per-row kernel을 씁니다. Build의 `hipcc` flag로 `qmm.hip`만 따로, main의 파일과 번갈아 compile했을 때 37.9초와 37.8초, main은 38.1초와 37.9초였습니다. 측정 가능한 변화는 없습니다.

`static_assert`도 같은 범위를 표현합니다. Kernel은 4 또는 8 bit affine, 또는 4 bit, group size 32인 non-affine에 대해서만 compile됩니다.

### 3.3 Dispatch와 #2066에서 이어받은 기본값 규칙

Gate의 형태는 그대로입니다 (sorted, transposed, `M == 1`, `B >= 64`, `E > 0`, `E <= 64`, `B / E >= 4`). Scheme 조건만 `eb_affine || eb_mxfp4`가 되었습니다. Decode step은 `B = top_k = 4`이므로 decode는 이 kernel에 오지 않고, gpt-oss에서 32 token보다 짧은 prefill도 오지 않습니다 (`B / E < 4`).

Kernel은 호출마다 읽히고 기본으로 켜진 `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED` 아래에 그대로 있습니다. #2066은 그 기본값의 규칙을 정했습니다. 모든 대상 dtype이 dequantize한 f32 reference 대비 unsorted 경로 자신의 오차 안에서 일치하고, 측정한 모든 대상 모델에서 512-token prefill이 좋아지며 decode 변화가 없을 때만 켭니다. 호스트에서 대상 mxfp4 dtype은 bf16 하나, 대상 mxfp4 checkpoint는 gpt-oss-20b 하나이고 두 조건이 모두 성립하므로 (4절), mxfp4도 affine과 함께 기본으로 켜집니다. 변수를 `0`으로 두면 모든 scheme에서 kernel이 꺼지며, mxfp4만 끄는 switch는 없습니다.

## 4. 검증

### 4.1 Kernel test

`mxfp4_expert_batched_gather_qmm_matches_unsorted_and_reference`는 kernel을 강제로 켜고, 정렬된 mxfp4 `gather_qmm`을 unsorted 경로 (per-row kernel), 그리고 dequantize한 weight로 expert마다 계산한 dense f32 matmul과 비교합니다. Shape는 gpt-oss-20b의 expert layer (expert 32개, top 4, `K = N = 2880`)와, 마지막 column block이 부분적인 `K = 512`, `N = 516`의 좁은 32-expert layer입니다. Shape마다 index layout 네 가지 (flat `Gathered`, `Shared`, broadcast `BroadcastRows`, `BroadcastIndices`), bf16 activation, 호출당 256 row로 실행하여 mxfp4 case는 8개입니다.

Reference 대비 상대 L2 오차 (expert-batched / unsorted):

| Shape | Gathered | Shared | BroadcastRows | BroadcastIndices |
|---|---|---|---|---|
| gpt-oss-20b experts | 1.654e-3 / 1.654e-3 | 1.654e-3 / 1.654e-3 | 1.657e-3 / 1.657e-3 | 1.657e-3 / 1.657e-3 |
| `K = 512`, `N = 516` | 1.653e-3 / 1.653e-3 | 1.617e-3 / 1.617e-3 | 1.673e-3 / 1.673e-3 | 1.660e-3 / 1.660e-3 |

두 경로는 모든 case에서 네 자리까지 일치하고, 정렬된 두 실행은 bit 단위로 같습니다. Test가 새 코드를 실제로 거친다는 것을 보이는 확인이 두 가지 더 있습니다.

- **의도적 파손.** Nibble decode에서 sign bit를 빼면 첫 case가 1.377로 실패하고 unsorted 경로는 1.654e-3에 머뭅니다. 비교가 새 arm에 닿는다는 뜻입니다.
- **Dispatch 확인** (`e87ffa7d`에서 추가). Per-row fallback도 정확하므로, gate가 mxfp4를 kernel로 보내지 않게 되어도 오차 비교만으로는 통과합니다. `K = 2880`에서 test는 kernel을 끈 상태로 정렬된 호출을 한 번 더 실행하고, 그 byte가 kernel을 켠 출력과 어딘가에서 달라야 한다고 요구합니다. 긴 reduction에서는 합산 순서가 달라 kernel이 실행되면 차이가 반드시 생깁니다. `K = 512`에서는 두 kernel이 같은 bf16 출력으로 반올림되므로 (측정됨), 좁은 shape는 이 확인을 건너뜁니다.

같은 파일의 affine case 48개는 바뀌지 않았고 통과합니다.

### 4.2 gfx1151의 prefill과 decode

`scripts/bench_decode.sh`의 shape (512-token prompt, 128 token 생성, 20-token warmup, `--ignore-eos`)로 `mlxcel-bench-decode`를 하나의 binary에서 `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`과 `=1`을 실행마다 번갈아 6 round 돌렸습니다. 모든 실행은 `scripts/rocm_gpu_guard.sh` 아래에서 첫 시도에 수락되었습니다.

| | Off (per-row) | On (expert-batched) |
|---|---|---|
| Prefill 중앙값, tok/s | 7.53 | 526.28 |
| Prefill 범위, tok/s | 7.41 ~ 7.92 | 524.78 ~ 529.38 |
| Decode 중앙값, tok/s | 8.45 | 8.49 |
| Decode 범위, tok/s | 8.34 ~ 8.65 | 8.40 ~ 8.64 |

Prefill은 69.9배 빨라져 65~69초 대신 약 0.97초가 걸리고, 모든 on 실행이 모든 off 실행보다 약 두 자릿수 빠릅니다. Decode 중앙값 차이는 0.4%이고 범위가 겹칩니다. 3 round 뒤에는 off의 decode 중앙값이 1.3% 앞섰기 때문에 3 round를 더 돌렸습니다. 대조군으로, 16-token prompt (`B = 64`, `B / E = 2`라 어느 쪽도 kernel에 오지 않아 같은 코드를 실행)를 세 번 번갈아 돌린 결과 off 8.46 / 8.50 / 8.47, on 8.55 / 8.47 / 8.48로, 호스트 자체의 편차가 약 1%였습니다. Decode는 변하지 않았습니다.

Kernel을 켠 `rocprofv3` trace에서 expert-batched kernel은 호출당 12.2 ms (144회)로, 이전 per-row kernel의 약 940 ms와 대비됩니다.

### 4.3 Logit trace

`benchmarks/logit_traces/`에 gpt-oss trace가 없으므로, 같은 binary의 per-row 경로 (`MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`)를 reference로, 기본값을 candidate로 하여 `tests/fixtures/wikitext2_excerpt.txt` 위에서 `scripts/compare_logit_traces.py --decided 2.0`으로 비교했습니다.

| Window | Top-1 불일치 | Decided 불일치 | Perplexity off / on |
|---|---|---|---|
| `w8` (512-token prefill이 kernel에 도달) | 29 / 640 | 0 / 88 | 161.96 / 159.95 |
| `w256` (256-token chunk 두 개, `B = 1024`) | 29 / 512 | 0 / 81 | 76.60 / 76.88 |

모든 불일치는 reference가 undecided인 위치 (top-two gap이 2.0 미만)에 있었고, script는 이를 동작 차이가 아닌 반올림으로 분류합니다. 이 corpus에서 gpt-oss는 decided 위치가 적습니다 (두 window에서 위치의 40%와 44%가 top-two gap 0.5 미만). Trace는 commit되지 않았습니다.

### 4.4 Gate

PR 기준: `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`와 `cargo test --test dead_doc_pointers`가 통과했습니다. Unit 자체의 첫 전체 `make verify-rocm` (`826f9d61`에 review commit의 test 변경을 더한 상태)은 clippy와 smoke test까지 모든 단계를 통과했지만, `verify-test-rocm`은 11,962개 통과, 1개 실패였습니다. 실패는 dispatch 확인의 첫 버전으로, 두 kernel이 같은 bf16 byte를 내는 `K = 512`에서도 실행되어 "달라야 한다"는 assertion이 성립할 수 없었습니다. `e87ffa7d`가 이 확인을 `K >= 2048`로 제한했고, 이후 test 파일은 2/2 통과, clippy도 깨끗했습니다.

Orchestrator 기준, head `e87ffa7d` (origin/main `5c851fc4`과 최신 상태 일치): `make verify-rocm`이 모든 단계를 통과했고, test 11,963개 통과, 0개 실패, 380개 ignored, smoke test OK였습니다.

검증하지 않은 것: Metal과 CUDA (호스트에 없음. 이 변경은 Metal이나 CUDA 파일을 건드리지 않습니다).

## 5. 기술적 선택과 그 이유

- **mxfp4용 warp-shared kernel arm 대신 expert-batched kernel 확장.** Issue는 둘 다 허용했습니다. Per-row 재읽기를 없애는 것은 expert-batched kernel이고, 그 loop는 decode와 누산만 바꾸면 되었으므로 #2066이 조정한 schedule을 그대로 재사용합니다.
- **Branch나 table 대신 fp16 bit를 거치는 decode.** e2m1의 bit 배치는 subnormal code point까지 포함해 상수 2^-14배로 fp16에 대응합니다. 덕분에 nibble당 mask 두 번, shift 두 번, 변환 한 번으로 decode가 끝나고, 보정하는 2^14 곱셈은 정확합니다.
- **값 단위가 아닌 word 단위 rescale.** 2^14 계수와 E8M0 scale은 word (한 group) 안에서 모두 상수이므로, word의 부분 내적에 한 번 적용하면 word당 row당 곱셈 하나만 추가됩니다.
- **Instantiation 하나.** f16 mxfp4나 mxfp8을 추가하면 호스트의 어떤 checkpoint도 쓰지 않는 dtype과 scheme을 위해 compile 시간을 치르게 됩니다. 이들은 정확한 per-row 경로를 유지합니다.
- **같은 switch, 같은 기본값 규칙.** mxfp4 전용 변수는 쓸 곳 없이 설정 표면만 늘립니다. #2066의 규칙은 mxfp4 arm이 충족한 근거 기준을 제공합니다.
- **출력 byte로 dispatch 확인.** Test 안에 profiler나 kernel counter가 필요 없고, 이후 gate 변경이 mxfp4를 조용히 빼면 실패합니다. 오차 비교만으로는 잡을 수 없는 경우입니다. 대가는 합산 순서에 대한 의존입니다. Reduction이 bf16 결과를 바꿀 만큼 길 때만 유효하므로 `K >= 2048`로 제한됩니다.

## 6. Fork 정책

이 변경은 fork된 MLX ROCm backend 위에 적용되는 ROCm overlay (`patches-rocm/`)에 있습니다. Maintainer의 2026-10-06 fork 정책에 따라 ROCm fork 수정은 mlxcelverse에 남으므로, `LOCAL_FIXES.md` item 31은 "kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there"로 끝나며 #1813의 upstreaming 후보로 표시되지 않습니다. Issue의 네 번째 수락 기준은 그 설명과 함께 취소선 처리되었습니다. Item 9는 이전의 upstreaming 표시를 그대로 갖고 있으며, 이 PR은 기존 항목의 상태를 바꾸지 않습니다.

## 7. 남은 위험과 후속 작업

- **이제 gpt-oss decode가 병목입니다.** Prefill이 해결된 뒤 profile에서 per-row mxfp4 `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>`가 호출당 1.39 ms, 504회 (decode step당 72회), 115 ms step 중 약 100 ms를 차지합니다. 이는 unsorted `B = 4` 경로로 #2106 범위 밖이며, 이 호스트에서 gpt-oss decode의 다음 목표입니다.
- **범위를 벗어난 정렬 index는 row를 쓰지 않은 채 남깁니다.** Block z는 정렬된 `rhs_indices`에서 binary search로 찾은 expert z의 run을 맡습니다. Index가 `E` 이상인 row는 어느 block에도 속하지 않아 출력이 저장되지 않습니다. Affine에서도 이미 그랬고 routing은 그런 index를 만들지 않지만, kernel이나 dispatch에서 이를 거부하지는 않습니다.
- **환경 변수가 사용자 문서에 없습니다.** `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED`는 `LOCAL_FIXES.md`, benchmark 페이지, 코드 주석에만 있고 사용자용 환경 변수 문서에는 없습니다.
- **범위 한계.** f16 mxfp4와 mxfp8은 kernel에 오지 않습니다. Wave64 (CDNA) 장치는 test하지 않았고, 16-lane reduction은 두 폭 모두에서 wave 안에 머뭅니다. Load당 4 row 계수는 mxfp4에 맞게 다시 조정하지 않았습니다. 다른 mxfp4 MoE checkpoint가 없어 기본값 규칙의 "모든 대상 모델"은 모델 하나입니다.
- **Dispatch 확인은 reduction 길이에 의존합니다.** Per-row kernel이 언젠가 expert-batched와 같은 순서로 합산하게 되면, dispatch가 맞아도 byte 비교가 실패합니다. Test 주석이 `K` 기준의 이유를 적어 둡니다.

## 8. 학습 포인트

- **작은 float 형식은 bit 배치로 decode할 수 있습니다.** 좁은 형식의 exponent와 mantissa가 넓은 형식의 낮은 exponent bit와 높은 mantissa bit에 들어가면, shift와 상수 rescale이 lookup을 대신하고 subnormal도 특별 처리 없이 맞게 나옵니다.
- **상수는 그것이 유지되는 가장 굵은 단위로 접습니다.** 여기서 decode 계수와 group scale은 모두 word 단위로 적용되어 값 단위 multiply-add에 끼어들지 않습니다.
- **정확한 fallback은 죽은 fast path를 가립니다.** 느린 경로도 정확하면 dispatch 동작 여부와 관계없이 정확도 test는 통과합니다. Test에는 두 경로에서 다른 관측값이 필요하고, 그 관측값은 실제로 달라지는 곳에서 확인해야 합니다 (`K = 512` 실패).
- **Arm을 번갈아 돌리고 대조군을 둡니다.** 3 round에서 한쪽으로 1.3% 기울었던 decode 중앙값은 6 round에서 사라졌고, 두 arm이 같은 코드를 실행하는 대조군이 호스트 자체의 편차를 측정했습니다.

Refs: #2106, #2066, #1814, #1813, #1801, #2161.
