# 기술 보고서: PR #2071 - ROCm에서 mxfp8 종단 간 검증, FP8 block 체크포인트 포함

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: Rust (테스트, example), C++ (ROCm overlay), Markdown

**위험도**: 낮음 (ROCm overlay의 SDPA fallback 판단에 CPU stream에만 영향을 주는 네 줄짜리 guard 하나, example 바이너리 수정 하나, 새 테스트, 문서 주석, 벤치마크 trace와 문서가 전부입니다. Metal과 CUDA의 라이브러리 코드 경로는 바뀌지 않으며, 그 환경에서는 실행하지 않았습니다)

## 요약

이슈 #1807(에픽 #1801의 Phase 2)은 mlxcel이 실제로 쓰는 경로에서 mxfp8이 ROCm에서 동작한다는 증거를 요구했습니다. mxfp8은 네이티브 mxfp8 체크포인트만의 문제가 아닙니다. 모든 vendor FP8 block 체크포인트는 로드 시점에 `src/models/fp8_block.rs`가 mxfp8로 재양자화합니다. 커널 수정은 이미 들어가 있었지만(PR #1818, `LOCAL_FIXES.md` 항목 8과 10), mxfp8 `gather_qmm`을 수치로 확인하는 커밋된 테스트가 없었고, FP8 block 체크포인트를 ROCm에서 처음부터 끝까지 돌려 본 적도 없었으며, ROCm `quantize`가 CPU와 비트 단위로 같지 않다는 점이 로드 시점에 문제가 되는지도 정해지지 않았습니다.

이 PR은 세 가지를 모두 해결합니다. `src/models/switch_layers_mxfp_tests.rs`는 mxfp8과 mxfp4의 `gather_qmm`(`SwitchLinear::forward` 경유)과 `quantized_matmul`(`UnifiedLinear::forward` 경유)을 호스트에서 f64로 디코드하고 누적한 기준값과 비교합니다. gather 케이스는 ROCm `GatherQMM::eval_gpu`에서 block-float 모드가 갈 수 있는 모든 분기에 도달하며, 항목 10을 되돌리면 두 gather 테스트가 모두 실패합니다. dense vendor FP8 block 체크포인트(`Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8`)를 ROCm과 CPU device에서 trace해 비교했고(이 호스트에는 Metal이 없어 CPU device가 대신합니다), decided position 50개 중 불일치는 0개입니다. GPU에서 양자화한 가중치와 CPU에서 양자화한 가중치는 decided position 346개 중 0개에서 불일치하므로, 로드 시점 양자화는 GPU에 그대로 둡니다.

CPU 기준값을 만드는 과정에서 결함 두 개가 드러났고, 둘 다 이 PR에서 고쳤습니다. `examples/logit_trace`는 `MLXCEL_DEVICE`를 무시해서 "CPU" trace가 조용히 GPU에서 실행되었습니다. 그리고 ROCm 빌드에서 `MLXCEL_DEVICE=cpu`를 쓰면 모든 attention 모델이 `NYI`로 중단되었습니다. overlay의 `ScaledDotProductAttention::use_fallback`이 stream device를 보지 않았기 때문입니다(새 `LOCAL_FIXES.md` 항목 23, `tests/cpu_device_sdpa.rs`가 지킵니다).

## 1. 문제 정의

### MoE 경로에 커밋된 수치 검사가 없었음

feasibility spike에서 ROCm의 mxfp8 `quantized_matmul`이 NaN을 반환했습니다. qmv dispatch가 한 바이트짜리 E8M0 scale을 activation dtype의 stride로 읽은 것이 원인이었습니다(항목 8). PR #1818은 검증 도중 gather 경로에서 다른 버그를 하나 더 찾았습니다. non-affine 모드가 `AFFINE=true`로 instantiate된 `gather_qmv_kernel`에 도달해서, fp 가중치가 affine 공식으로 dequantize되었습니다(상대 오차 1.0에서 1e34, 항목 10). PR #1818은 둘 다 임시 probe로 확인했습니다. `src/`, `tests/`, `examples/`를 찾아봐도 mxfp8 `gather_qmm`을 수치로 검사하는 커밋된 테스트는 없었고, `switch_layers.rs`와 모델 테스트의 mxfp8 케이스는 로더 거부만 확인했습니다. 이후 PR #2057이 `tests/rocm_mxfp4_quant.rs`에 mxfp4 검사를 커밋했지만, mxfp8은 아니었고 production 레이어를 거치지도 않았습니다.

fused MoE launcher에는 ROCm port가 없고(port table의 `.rocm = nullptr`), `fused_moe_enabled()`는 ROCm에 없는 custom kernel을 요구합니다. 따라서 ROCm에서 mxfp8 MoE가 실제로 실행하는 것은 `gather_qmm`을 거치는 MLX graph 경로이며, 검사가 다뤄야 할 경로도 그것입니다.

### ROCm에서 FP8 block 체크포인트를 실행한 적이 없었음

gfx1151 정확도 매트릭스(PR #1826)는 affine 4-bit만 trace했고 mxfp8 커버리지를 이 이슈로 넘겼습니다. 이슈의 수락 기준은 Metal 대비 decided position 불일치율이 #1809의 `--decided 2.0` 기준 안에 들어오는 것이었습니다.

### GPU `quantize`가 CPU와 비트 단위로 같지 않음

ROCm `quantize`는 CPU와 같은 E8M0 scale을 만들지만 tie를 다르게 반올림해서, 가중치 바이트의 약 3.3%가 한 단계씩 다르고 RMS 오차는 같습니다. 이슈는 이것이 decided position을 바꾸는지 물었습니다. 바꾼다면 로드 시점 재양자화를 CPU stream으로 옮기고, 아니라면 차이를 문서로 남기는 것이었습니다.

## 2. 변경 요약

- **`src/models/switch_layers_mxfp_tests.rs`** (신규, 540줄, `switch_layers.rs`에서 `#[cfg(test)] mod mxfp_tests`로 포함되어 crate-private인 `SwitchLinear`와 `gather_sort`에 접근합니다).
  - 호스트 기준값: packed code(mxfp8은 E4M3, mxfp4는 E2M1)와 E8M0 scale(`2^(e - 127)`)을 Rust에서 디코드하고, device가 가진 activation 값에서 f64로 누적합니다. 테스트 대상 backend와 공유하는 커널이 없습니다.
  - `mxfp_host_decode_matches_mlx_dequantize`: 호스트 디코더를 MLX `dequantize`와 비트 단위로 고정해서, 틀린 기준값이 조용히 통과할 수 없게 합니다.
  - `mxfp8_gather_qmm_matches_host_reference`, `mxfp4_gather_qmm_matches_host_reference`: gather 케이스 다섯 개(unsorted decode `B = 4`, unsorted prefill `B = 24`, sorted prefill `B = 128`, activation stride 0인 sorted shared-activation `B = 32`, multi-row `M = 4`)를 bf16, f16, f32로 실행하고, `N = 320`으로 두 번째 column block의 경계 검사까지 돌립니다.
  - `mxfp8_quantized_matmul_matches_host_reference`: dense 행 `M = 1, 4, 64`.
  - `mxfp_matmuls_match_host_reference_on_cpu_device`: MLX CPU device에서 축소 매트릭스를 약 2.5초에 실행합니다(두 모드, unsorted prefill, sorted shared-activation, multi-row gather, dense `M = 1`과 `4`, bf16과 f16, 출력 폭 64).
  - 허용 범위(상대 L2): 2e-2 (bf16), 4e-3 (f16), 2e-3 (f32). backend로 제한하지 않으며 `lock_default_device`를 잡고 실행합니다.
- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/scaled_dot_product_attention.cpp`** (+12): `use_fallback`이 shape 검사보다 먼저, GPU가 아닌 stream이면 `true`를 반환하고, 그때 `force_fused`가 켜져 있으면 `invalid_argument`를 던집니다. `LOCAL_FIXES.md` 항목 23에 기록했습니다.
- **`tests/cpu_device_sdpa.rs`** (신규, 138줄): decode 형태의 query를 포함해 CPU device에서 attention을 실행하고 호스트 softmax 기준값과 비교합니다. 수정 전 overlay에서는 중단됩니다.
- **`examples/logit_trace.rs`** (+10): `mlxcel::initialize_runtime_checked()`를 호출해 `MLXCEL_DEVICE`를 따르고, 잘못된 override에는 경고를 내며, 모든 trace에 `# device` 헤더 줄을 씁니다.
- **`src/models/fp8_block.rs`** (문서만): 재양자화를 기본 device에 두는 이유와 측정값을 설명합니다.
- **`src/lib/mlxcel-core/src/hardware.rs`** (문서만): `quant_mode_support`의 근거 목록이 mxfp8 테스트와 체크포인트 실행을 인용합니다. 표에는 이미 ROCm mxfp8이 `Native`로 되어 있었고, 표 자체는 바뀌지 않습니다.
- **`docs/benchmark_results/rocm-fp8-block-gfx1151-2026-09-30.md`** (신규)와 **`benchmarks/logit_traces/fp8_block_gfx1151/`** (신규): 결과 문서, trace 파일 8개, `METADATA.txt`, `RUNS.txt`, `SHA256SUMS`.
- **`docs/installation.md`**: ROCm 상태 표의 mxfp8 행에 FP8 block 체크포인트를 추가하고, ROCm 빌드의 CPU device 행("동작하지만 느림")을 새로 넣었습니다. **`rocm-correctness-gfx1151-2026-09-12.md`**: 낡은 mxfp8 공백 문구가 새 실행 결과를 가리키게 했습니다.

커밋: `a9a05ac5`는 테스트, SDPA guard, `logit_trace` 수정을 추가합니다. `35375780`은 벤치마크 결과와 trace를 추가합니다. `c26086b2`는 sorted shared-activation 케이스가 실제로 stride-0 schedule에 도달하게 하고 TF32를 고려해 f32 범위를 넓힙니다. `9d1f570e`는 CPU arm을 38초에서 2.5초로 줄이고 SAFETY 주석을 추가합니다.

## 3. 기술적 선택과 그 이유

### lib target 안에서 production 레이어를 거쳐 테스트

테스트는 `gather_qmm`, `quantized_matmul` FFI를 직접 부르지 않고 `SwitchLinear::forward`와 `UnifiedLinear::forward`를 호출합니다. gather 입력은 `SwitchGLU`가 쓰는 것과 같은 `gather_sort`로 만들기 때문에, sorted와 unsorted shape가 모델 코드가 실제로 만드는 것과 같습니다. crate-private 항목이 필요해서 파일은 `tests/`가 아니라 `mlxcel` lib target의 `#[path]` 모듈로 두었습니다. 알아 둘 부수 효과가 하나 있습니다. 같은 target에 있는 기존 `fp8_block` round-trip 테스트가 이미 ROCm 기본 device에서 실행되고 있으므로, 계획 4단계에는 별도의 ROCm round-trip 테스트가 필요하지 않았습니다.

### MLX에 고정한 호스트 f64 기준값

PR #2057은 MLX CPU backend와 비교했습니다. 여기서는 기준값을 Rust에서 디코드합니다. MLX CPU `fp_qmm_t`는 activation dtype으로 누적하므로 bf16에서는 측정한 backend 중 가장 부정확하고(ROCm 1.8e-3에 비해 8.2e-3), scalar라서 느립니다. 호스트 f64 기준값은 device 입력의 반올림을 제외하면 정확하고 테스트 대상 커널 어느 것과도 독립적입니다. 손으로 쓴 디코더의 위험은 커널과 같은 방식으로 틀리는 것인데, MLX `dequantize`와의 비트 단위 비교 테스트가 이를 막습니다.

### 테스트가 도달하는 분기를 나열하고 그 주장을 확인

`GatherQMM::eval_gpu`는 특수 경로(grouped WMMA prefill, expert-batched, tiled, wide, idot, warp-shared)를 모두 `mode_ == Affine` 조건으로 막으므로, block-float 모드는 항목 10의 `gather_qmv_kernel<T, uint8_t, BITS, 32, false>` launch 하나에만 도달합니다. 케이스 목록은 그 안의 모든 arm을 치도록 만들었습니다. `T` 세 가지, `BITS` 두 가지, `implicit_lhs` false와 true, 그리고 sorted-rhs schedule에서는 activation stride `K`와 0 둘 다입니다. 첫 버전의 shared-activation 케이스는 expert 8개에 slot 16개였는데, `use_sorted_rhs_schedule`은 `B / E >= 4`를 요구하므로 모듈 문서가 다룬다고 주장한 stride-0 분기는 실제로 실행되지 않았습니다. 커밋 `c26086b2`가 slot을 32개로 늘렸습니다. opt-in expert-batched 커널(항목 9)은 affine 전용이라 이 모드에서는 도달할 수 없습니다. 이것으로 "기본 경로인가 opt-in 경로인가"라는 이슈의 질문에 답이 나옵니다. 기본 경로입니다.

### 모든 backend에서 성립하는 허용 범위

테스트는 ROCm으로 제한하지 않습니다. bf16과 f16 범위는 activation dtype의 반올림에서 나왔고, 관측된 가장 부정확한 정상 backend(MLX CPU)보다 약 2.4배 위에 있습니다. f32 범위는 처음에 1e-5였지만 CUDA sm80 이상에서는 실패했을 것입니다. 그곳에서는 sorted prefill 케이스(`B >= 8 * E`)가 MLX의 grouped GEMM을 타고, 기본값이 켜짐인 `MLX_ENABLE_TF32`가 f32 activation을 10-bit mantissa로 반올림합니다. 지금은 2e-3이며, 여전히 항목 10 결함(1.0 이상)보다 세 자릿수 아래입니다. CPU arm은 범위가 특정 GPU에 맞춰진 것이 아님을 보이려고 있습니다. 전체 매트릭스는 모든 backend에서 전역 default-device lock을 38초 동안 잡고 있었기 때문에 축소 매트릭스로 줄였습니다.

### Metal이 없으므로 CPU device를 기준으로

수락 기준은 Metal을 지목합니다. 이 호스트에는 Metal이 없고, 공식 Qwen3.5 FP8 릴리스(27B는 31 GB, 35B-A3B MoE는 37.5 GB)는 30 GiB 호스트에서 CPU 기준값을 만들 여유가 없습니다. 커뮤니티 체크포인트 `ReAligned-Qwen3.5-0.8B-FP8`은 로더가 변환하는 vendor 레이아웃(`quant_method: fp8`, 128x128 block, `*.weight_scale_inv`)을 그대로 쓰므로 같은 로드 경로를 탑니다. dense 모델이라 재양자화와 dense mxfp8 `quantized_matmul`을 다루고, MoE gather 경로는 op 테스트가 다룹니다. CPU backend는 이 체크포인트를 trace 토큰당 약 2.7분에 처리하므로, CPU arm은 한 폭(`w32`, position 256개, decided 50개)만이며 병렬 프로세스 8개의 결과를 조립했습니다. CPU는 bf16으로 누적하므로 Metal보다 약한 기준입니다.

### 로드 시점 양자화는 GPU에 유지

tie 반올림의 영향은 GPU에서 두 번 trace해서 분리했습니다. 한 번은 배포 상태 그대로, 한 번은 `requantize_block_fp8_weights`가 임시로 CPU stream에서 양자화하게 했습니다(로컬 스위치이며 커밋하지 않았습니다). 두 arm 모두 GPU에서 계산하고 ROCm trace는 재실행해도 바이트 단위로 같으므로, 모든 차이는 packed 가중치에서 옵니다. `w1`, `w8`, `w256`에 걸쳐 decided position 346개 중 불일치는 0개이며, top-1 불일치 42개는 모두 기준 gap 0.375 이하에 있습니다. 양자화를 CPU로 옮기면 0.8B 로드가 1초 미만에서 약 4분으로 늘어납니다(`w1` trace가 6초에서 250초가 되었습니다). 그래서 차이를 억지로 없애지 않고 `fp8_block.rs`에 문서로 남겼고, quantize stream이 바뀌지 않았으므로 계획 4단계에 새 테스트도 필요하지 않습니다.

### upstream CUDA처럼 SDPA를 stream device로 판단

overlay의 `use_fallback`은 shape만 보고 판단했기 때문에, CPU stream에서 decode 형태의 query가 들어오면 fused primitive를 만들었고, 그 `eval_cpu`는 `NYI`를 던지며 cxx bridge를 넘으면 `std::terminate`가 됩니다. upstream CUDA의 `has_fused_kernel`은 shape보다 stream device를 먼저 확인하며, 이 수정은 그 순서를 따릅니다. CPU stream에서 `force_fused`를 요구하면 조용히 fallback하지 않고 잡을 수 있는 `invalid_argument`를 던지므로, fused 커널을 요구한 호출자는 그것을 쓸 수 없다는 사실을 알게 됩니다.

## 4. 검증

PR 작성자 (gfx1151):

- `cargo test --features rocm --lib models::switch_layers::mxfp_tests models::fp8_block -- --test-threads=1`: 기존 round trip을 포함해 16개 통과. ROCm의 최악 상대 L2 오차는 1.8e-3 (bf16), 2.2e-4 (f16), 2.9e-7 (f32). CPU device 축소 매트릭스는 8.2e-3 (bf16), 1.0e-3 (f16)이며, 이전 전체 매트릭스 실행은 7.9e-3, 1.0e-3, 1.2e-7이었습니다.
- 되돌림 검사: 항목 10을 되돌리고 overlay를 다시 빌드하면 두 gather 테스트가 첫 케이스에서 non-finite 출력으로 실패합니다. SDPA guard를 되돌리면 `cargo test --features rocm --test cpu_device_sdpa`가 `NYI`로 중단됩니다. 복원하면 둘 다 통과합니다.
- 체크포인트 `Lazarus-Ai/ReAligned-Qwen3.5-0.8B-FP8` @ `db97e6a7`: `w32`(`32 8 8 0`)에서 CPU 대 ROCm은 top-1 불일치 14개, decided 50개 중 0개, 불일치 지점의 최대 gap 0.125, perplexity 100.80 (CPU)과 100.95 (ROCm). 모든 불일치에서 CPU의 토큰은 ROCm의 두 번째 후보입니다. #1809에서 Metal과 ROCm 12쌍의 최대 gap은 1.125였습니다.
- ROCm에서 `mlxcel generate`: 텐서 132개를 565 ms에 재양자화, 56에서 62 tok/s, 일관된 텍스트.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`, `mlxcel`(lib, tests, examples)과 `mlxcel-core`(lib, tests)의 clippy `-D warnings`: 통과.

오케스트레이터 검증 (gfx1151, 브랜치를 origin/main `9484ffc2`에 rebase):

- `make verify-rocm`이 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 네 target에서 실패했습니다. 세 개는 알려진 기준 실패입니다. bf16 `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, 그리고 #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`입니다. 네 번째는 `qmm 2880x2880 M=1`의 `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference`로, 이 PR 이전부터 있던 문제이며 이 PR이 원인이 아닙니다. 이 PR 없이 main `9484ffc2`에서 10번 중 7번, `4595b06f`(#2057 머지 커밋)에서 10번 중 6번 실패하며, f32, f16, bf16 모두에서 나타납니다. 이 게이트에서 등록한 별도 이슈가 추적합니다.
- 나머지 target은 모두 통과했습니다. 이 PR의 새 mxfp 테스트와 `tests/cpu_device_sdpa.rs`도 포함됩니다.

## 5. 학습 포인트

- **기준 실행은 어디서 실행됐는지 기록해야 합니다.** `examples/logit_trace`는 `MLXCEL_DEVICE`를 읽지 않았으므로 처음의 "CPU" trace는 사실 GPU trace였고, 잘못된 이유로 ROCm과 완벽히 일치했을 것입니다. 수정 후에는 CLI처럼 device를 결정하고 trace 헤더에 기록하므로, trace 파일이 어느 device에서 나왔는지 스스로 말해 줍니다.
- **테스트의 커버리지 주장은 dispatch 조건과 대조해야 합니다.** 모듈 문서는 stride-0 sorted schedule을 다룬다고 했지만, 케이스가 schedule의 `B / E >= 4` 조건을 넘지 못했습니다. `qmm.hip`의 조건을 읽고 각 케이스의 크기를 그에 맞춰 본 것이 이를 찾아냈습니다. 같은 읽기로 항목 9의 경로가 block-float 모드에서는 도달할 수 없다는 것도 확인했고, 이슈가 열어 둔 질문이 정리되었습니다.
- **실행하지 않은 backend의 허용 범위는 그 backend의 기본값을 고려해야 합니다.** ROCm에 맞춘 f32 범위(측정값 2.9e-7)는 grouped GEMM에 TF32가 기본으로 켜진 CUDA sm80 이상에서 실패했을 것입니다. 각 backend의 반올림 모델로 범위를 정하고, 그래도 결함보다 훨씬 아래인지 확인해야 backend 중립 테스트가 호스트에 맞춰지지 않습니다.
- **수치 차이를 판단할 때는 변수 하나만 분리합니다.** 두 arm 모두 GPU에서 계산하면서 GPU 양자화 가중치와 CPU 양자화 가중치를 비교하고, trace가 결정적인지 확인했기 때문에 모든 차이를 `quantize`의 tie 반올림 탓으로 돌릴 수 있었습니다. ROCm을 CPU device와 비교했다면 CPU의 bf16 누적 효과가 섞였을 것입니다.
- **중단되는 탈출구는 없는 것보다 나쁩니다.** `MLXCEL_DEVICE=cpu`는 `gfx` 빌드가 맞지 않거나 커널이 의심스러울 때 자연스럽게 떠올리는 fallback인데, ROCm에서는 첫 attention 호출에서 중단되었습니다. 이제는 느리지만 동작하며, 설치 문서는 이를 서빙 모드가 아닌 정확도 기준과 탈출구로 설명합니다.

## 6. 주의 사항과 검증하지 않은 부분

- **Metal과 CUDA**는 실행하지 않았습니다. 영향을 받는 것은 `examples/logit_trace`(CLI가 이미 적용하던 기본 wired limit과 arch 호환성 거부도 이제 적용합니다)와 모든 backend에서 실행되는 새 테스트입니다. 그곳의 라이브러리 코드 경로는 바뀌지 않으며, SDPA 수정은 ROCm overlay에만 있습니다.
- **기준은 Metal이 아니라 CPU device입니다.** 수락 기준은 더 약한 기준(bf16 누적)에 대해 충족되었습니다. 같은 체크포인트의 Metal trace가 있어야 기준을 문구 그대로 닫을 수 있습니다.
- **CPU arm은 한 폭만 다룹니다** (`w32`, decided position 50개). `w1`, `w8`, `w256`은 양자화 비교를 위해 ROCm에서만 trace했습니다.
- **MoE FP8 체크포인트는 실행하지 않았습니다.** `Qwen/Qwen3.5-35B-A3B-FP8`과 `Qwen/Qwen3.5-27B-FP8`은 이 호스트와 CPU 기준값에 비해 너무 큽니다. mxfp8 MoE 경로는 op 수준에서만 검증되었습니다.
- **되돌림 증거는 저장소에 없습니다.** 항목 10과 SDPA의 되돌림 검사, CPU 양자화 실험 스위치는 수동으로 실행했습니다.
- **gfx1151만** 실행했고, group size는 ROCm `gather_qmm`이 받는 유일한 fp group size인 32만 다룹니다.
- **이슈 계획 5단계**(mxfp8을 `Native`로 전환)에는 코드가 필요하지 않았습니다. #1806의 capability table에 ROCm mxfp8이 이미 `Native`로 되어 있습니다. 이 PR은 그 항목을 뒷받침하는 커밋된 근거를 더합니다.

## 7. 남은 작업

- `qmm 2880x2880 M=1`에서 간헐적으로 실패하는 `rocm_mxfp4_quant::quantized_matmul_matches_dequantized_reference` (이 게이트에서 등록한 별도 이슈). 이 PR 이전부터 있었고 #2057 머지 커밋에서도 재현됩니다.
- #1809: FP8 block 체크포인트의 Metal trace가 CPU device 기준값을 대체할 수 있습니다.
- #1813: 항목 8, 10과 함께 항목 23(SDPA stream device guard)을 fork에 제안.
- #1814: ROCm fused MoE port. 그때까지 mxfp8 MoE는 범용 `gather_qmm` 커널로 실행됩니다.

참조: #1807 (이 PR로 닫힘), #1801, #1806, #1808, #1809, #1813, #1814, #2037, PR #1818, PR #1826, PR #2057.
