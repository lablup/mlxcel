# 기술 보고서: PR #2253 - ROCm decode 배치의 행을 하나의 양자화 곱으로 실행

**날짜**: 2026-10-09

**상태**: 전체 `make verify-rocm` 게이트 통과 후 머지. 헤드 `474163e3`(origin/main `7a3fcc4a` 위로 리베이스). #2156을 닫음.

**언어**: C++/HIP(ROCm 오버레이 `patches-rocm/mlx/backend/rocm/quantized/qmm.hip`), Rust(신규 `tests/rocm_qmm_batched_rows.rs`, 신규 `examples/qmm_batch_rows_probe.rs`), Python(신규 `scripts/bench_serving_metrics.py`, `scripts/bench_serving_concurrency.py`, 신규 `tests/test_bench_serving_concurrency.py`), Bash(`scripts/benchmark_paged_decode_production.sh`), Markdown(`LOCAL_FIXES.md` 항목 43, `docs/environment-variables.md`, `docs/benchmarks.md`, 신규 결과 페이지와 데이터 디렉터리)

**위험도**: 중간. 폴드는 ROCm에서 모든 배치 양자화 projection을 처리하는 커널을 바꾸고, 라우트 변경은 2~8행 bf16 GEMM을 fused WMMA 커널에서 빼냅니다. 둘 다 ROCm 오버레이 안에 한정되며, 폴드는 배치 차원이 없는 가중치와 행 연속(row contiguous) 활성값에만 적용되고 이때 출력 바이트는 어느 쪽이든 같습니다. `MLX_ROCM_WMMA_QMM=1`은 여전히 WMMA 커널을 강제합니다. 단일 스트림 decode는 노이즈 범위 안에서 변하지 않았습니다.

## 요약

gfx1151에서 동시 요청 네 개의 서버 decode는 요청당 약 6 tok/s로, 단일 스트림의 약 32 tok/s에 크게 못 미쳤습니다. 이슈는 decode와 끼어든 prefill을 분리하고, decode 구간을 프로파일링하고, 지목된 후보 세 가지를 확인하고, 국소적인 원인이면 고치라고 요구했습니다.

배치 decode 스텝 자체가 단일 스트림 스텝의 5.3배였습니다. 배치 forward는 모든 projection에 `[B, 1, K]` 활성값을 넘기는데, ROCm 오버레이의 `QuantizedMatmul::eval_gpu`는 이를 한 행짜리 곱 `B`개로 읽었습니다. 그 결과 `qmv_warp_shared_batched_kernel`이 실행되었고, 이 커널은 배치 원소마다 가중치 전체를 한 번씩 읽습니다. 모델 수준 batch-4 스텝 트레이스에서 이 실행들이 148.5 ms 중 134.8 ms를 차지했습니다.

PR #2253은 가중치에 배치 차원이 없고 활성값이 행 연속일 때 배치를 행 수에 접어 넣습니다. Metal 백엔드가 같은 호출을 읽는 방식과 같습니다. 이제 행들은 `qmv_wide_kernel`이 가중치를 한 번 읽는 동안 모두 처리됩니다. bf16 체크포인트에서는 접힌 행이 fused WMMA 커널로 갔을 텐데, 이 커널은 2~8행에서 한 행 GEMV의 7~8배가 걸리므로, `select_qmm_route`가 이제 bias가 있는 8행 이하의 4비트와 8비트 GEMM을 이 커널로 보내지 않습니다.

Meta-Llama-3.1-8B-Instruct-4bit에서 batch-4 ~1K 서버 decode 스텝은 166.7 ms에서 65.9 ms로, 요청당 decode는 6.0에서 15.3 tok/s로 바뀌었습니다. 단일 스트림 `mlxcel-bench-decode`는 37.69에서 37.84 tok/s였습니다. Qwen3-0.6B-4bit(bf16) 배치 스텝은 batch 2~8에서 1.5~1.6배 줄었습니다. ~16K에서는 요청당 속도가 다른 요청의 청크 prefill로 결정되며, 그 시간의 78%가 flash SDPA입니다. 이는 #2251로 등록했습니다. Metal 기준 비율은 Metal 호스트가 없어 측정하지 않았습니다.

## 1. 문제 정의

### 1.1 증상

PR #2103 결과(`docs/benchmark_results/rocm-paged-attention-gfx1151-2026-10-05.md`, `mlxcel-server --parallel 4 --ctx-size 131072`, 요청당 decode 토큰 128개, 3회 중앙값)는 paged-attention 두 방식 모두에서 요청당 decode를 batch 4 ~1K에서 5.6~6.1 tok/s, ~4K에서 10.2~10.5, ~16K에서 10.9~13.9로 보고했습니다. 같은 모델의 단일 스트림 `bench_decode`는 약 35~37 tok/s입니다.

그 페이지는 차이를 다른 요청이 prefill하는 동안 기다리는 시간 탓으로 돌렸지만 측정하지는 않았습니다. 이슈는 약 1000 tok/s로 1K prefill 세 번을 해도 약 21초의 decode 구간 중 약 3초에 불과하므로 그 설명이 불완전하다고 지적했습니다. 또한 문맥이 길어질수록 속도가 오르는 것은 스텝당 연산 비용으로는 나올 수 없는 모양이라고 적었습니다.

### 1.2 하네스가 스텝을 볼 수 없었음

`scripts/bench_serving_concurrency.py`는 `decode_tok_s = (completion_tokens - 1) / (total_s - ttft)`로 계산하므로, 구간에 다른 요청의 prefill 청크에 쓰인 스케줄러 시간이 모두 들어갑니다. 서버는 이미 `mlxcel_batch_decode_steps_total`, `mlxcel_batch_decode_tokens_total`, `mlxcel_batch_mixed_steps_total`, `mlxcel_batch_prefill_chunks_total`, `llamacpp:tokens_predicted_seconds_total`을 내보내고 있었지만 하네스가 읽지 않았습니다. 이슈의 1단계는 레벨별 증분을 출력해 배치 스텝 비용을 보이게 하는 것이었습니다.

### 1.3 근본 원인

새 카운터로 보니 batch-4 ~1K 레벨은 127개 decode 스텝 동안 점유율 4.00, prefill 청크 0개였습니다. 즉 느린 속도의 원인은 decode 스텝 자체였습니다. batch 1의 31.3 ms에 비해 166.7 ms였습니다.

모델 수준 프로브 `examples/profile_batched_decode.rs`는 서버가 호출하는 것과 같은 `forward_batched`를 스케줄러 없이 실행합니다. `rocprofv3 --kernel-trace` 아래에서 batch-4 스텝은 148.5 ms였고 그중 141.3 ms(95.2%)가 GPU busy였으며, 양자화 projection이 스텝당 `qmv_warp_shared_batched_kernel<f16>` 161회 실행으로 134.8 ms를 차지했습니다. SDPA는 3.1 ms, 복사는 2.4 ms였습니다.

배치 forward는 각 projection에 2차원 가중치에 대한 `[B, 1, K]` 활성값을 넘깁니다. `QuantizedMatmul::eval_gpu`는 앞쪽 `B`를 원소당 `M = 1`인 배치 차원으로 다뤘으므로, 디스패치가 배치 원소마다 가중치 전체를 읽는 배치 GEMV를 택했습니다. decode 스텝은 가중치 대역폭에 묶이므로, batch 4는 단일 스트림 스텝 네 번에 오버헤드를 더한 비용이 들었습니다. 모델 수준에서 batch-4 `forward_batched`(157.4 ms)가 순차 `forward` 네 번(111.6 ms)보다 느렸습니다.

`examples/qmm_batch_rows_probe.rs`는 projection만 떼어 측정합니다(Llama 3.1 8B 형상, f16, 4비트 group 64, 캐시를 피할 만큼 가중치 사본을 돌려 씀, 한 레이어 projection 합계의 호출당 마이크로초).

| 활성값 | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]`(단일 스트림) | 562 | 554 | 562 |
| `[B, 1, K]` 수정 전 | 1868 | 2747 | 4666 |
| `[B, 1, K]` 수정 후 | 681 | 908 | 1368 |
| `[B, K]`(변경 없음) | 676 | 901 | 1358 |

같은 행을 `[B, K]`로 넘기면 이미 하나의 곱으로 실행되었습니다. 배치 decode가 쓰는 3차원 레이아웃만 원소별 경로를 탔습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `patches-rocm/.../quantized/qmm.hip`, `QuantizedMatmul::eval_gpu` | `batch_count > 1`이고, 가중치의 배치 차원이 모두 크기 1이고, `x_batch_count == batch_count`이고, `x`가 행 연속이면 `M *= batch_count`로 바꾸고 배치 없는 곱 하나로 진행 |
| `patches-rocm/.../quantized/qmm.hip`, `select_qmm_route` | 신규 `wide_qmv_rows`(`M <= 8`, 4비트 또는 8비트, bias 있음). `MLX_ROCM_WMMA_QMM=1`이 아니면 이 형상에서 WMMA 라우트를 건너뜀 |
| `patches-rocm/LOCAL_FIXES.md` | 항목 43에 수정, 측정값, fork 정책(mlxcelverse에 유지, upstream에 제안하지 않음) 기록 |
| `tests/rocm_qmm_batched_rows.rs`(신규) | `decode_batch_rows_are_one_product`: f16과 bf16, B = 2, 4, 8에서 `[B, 1, K]`, `[B, K]`, `[1, B, K]`가 같은 바이트를 반환하고 f32 CPU 기준과 일치하는지 확인. strided `[B, 1, K]` 뷰는 배치 경로에 남고 기준과 일치하는지 확인 |
| `examples/qmm_batch_rows_probe.rs`(신규) | Llama 3.1 8B projection 형상 다섯 개를 레이아웃별로 측정하고 레이아웃 간 결과 일치 확인 |
| `scripts/bench_serving_metrics.py`(신규) | `/metrics` 파싱, 레벨별 증분, 점유율, 이슈의 원시 비율, 요청 기준 decode 스텝 ms 계산 |
| `scripts/bench_serving_concurrency.py` | `--metrics`가 경로 카운터와 함께 배치 카운터를 읽고 레벨마다 두 줄 출력 |
| `scripts/benchmark_paged_decode_production.sh` | `--metrics` 전달 |
| `tests/test_bench_serving_concurrency.py`(신규) | 미리 만든 `/metrics` 텍스트로 파싱과 증분 계산을 검사하는 테스트 12개 |
| `docs/environment-variables.md`, `docs/benchmarks.md` | `MLX_ROCM_WMMA_QMM`의 소수 행 예외, `--metrics` 출력 |
| `docs/benchmark_results/rocm-batched-decode-gfx1151-2026-10-08.md`와 데이터 디렉터리(신규) | 결과 페이지, 원시 출력, guard 로그, 트레이스, 하네스 스크립트 |

커밋은 세 개입니다. 하네스 확장(`3fd1654d`, 파일 5개, 374줄 추가, 19줄 삭제), 수정(`895592d3`, 파일 5개, 436줄 추가, 3줄 삭제), 데이터를 포함한 결과 페이지(`474163e3`, 파일 159개, 12,575줄 추가)입니다. 합계 파일 169개, 13,385줄 추가, 22줄 삭제.

## 3. 설계

### 3.1 배치를 행으로 접기

폴드는 `QuantizedMatmul::eval_gpu`에서 singleton-batch 플래그를 계산한 직후에 있습니다.

```cpp
if (batch_count > 1 && w_singleton_batch && x_batch_count == batch_count &&
    x.flags().row_contiguous) {
  M *= batch_count;
  batch_count = 1;
  x_batch_count = 1;
  x_singleton_batch = true;
}
```

각 조건은 폴드가 결과를 바꿀 수 있는 경우 하나씩을 막습니다.

- **배치 없는 가중치.** 모든 배치 원소가 같은 행렬을 곱하므로 배치는 그저 행이 더 많은 것입니다. 배치된 가중치(원소별 행렬)는 배치 경로를 유지합니다.
- **활성값 배치와 출력 배치가 같음.** 배치를 활성값이 들고 있어야 합니다. broadcast된 활성값에는 폴드가 적용되지 않습니다.
- **행 연속 활성값.** 이때 `[B, M, K]`는 `[B * M, K]`와 메모리 배치가 같습니다. `[B, 2, K]` 텐서의 한 행 건너 한 행을 취한 뷰처럼 strided한 뷰는 배치 경로를 유지하며, 테스트가 이 경우를 확인합니다.

출력은 행 연속으로 할당되므로 `[B, M, N]`과 `[B * M, N]`은 같은 바이트이고, 커널 뒤에 reshape나 복사가 필요 없습니다.

Metal 백엔드도 같은 호출을 이렇게 읽습니다. `w.ndim() == 2`이고 `x`가 행 연속이면 `M = x.size() / K`입니다. ROCm 오버레이가 여기서 갈라져 있었습니다.

폴드 덕분에 batch-4 decode 행은 `M <= 8` dense 디스패치에 도달하고, 여기서 `qmv_wide_kernel`이 가중치를 한 번 읽으며 모든 행을 처리합니다. 같은 폴드가 batch-4 ~1K prefill(`[4, 899, K]`)을 3,596행 GEMM 하나로 만들기 때문에 batch-4 TTFT도 3.37초에서 2.42초로 줄었습니다.

### 3.2 bf16 라우트도 바꿔야 했던 이유

폴드만으로는 bf16 배치 decode가 더 느려졌습니다. 폴드 전 bf16 `[B, 1, K]`는 배치 qmv로 가서 한 행 비용의 약 `B`배였습니다. 폴드 후에는 접힌 행이 `select_qmm_route`의 WMMA 분기를 만났는데, 이 분기는 128행 상한 아래에서 2행 이상의 bf16 affine GEMM을 받습니다. 2~8행에서 이 커널은 한 행 GEMV의 7~8배가 걸렸습니다.

| bf16 프로브, 형상 다섯 개 합계의 호출당 us | B = 2 | B = 4 | B = 8 |
|---|---:|---:|---:|
| `[1, 1, K]` | 550 | 519 | 567 |
| `[B, 1, K]`, 폴드만(WMMA) | 4396 | 4218 | 3752 |
| `[B, 1, K]`, 최종(qmv) | 664 | 944 | 1571 |
| `[1, B, K]`, 폴드만(WMMA, 이 변경 전과 동일) | 4501 | 3843 | 4111 |

모델 수준에서 Qwen3-0.6B-4bit batch-2 스텝은 수정 전 10.6~10.7 ms에서 폴드만 적용했을 때 22.5~24.0 ms가 되었습니다(번갈아 실행한 두 라운드, 각각 3회 중앙값).

두 번째 변경은 `wide_qmv_rows = M <= 8 && (bits == 4 || bits == 8) && bias 있음`을 추가하고, `MLX_ROCM_WMMA_QMM=1`이 아니면 이 경우 WMMA 분기를 건너뜁니다. 이는 dense 경로에서 `qmv_wide_kernel` 게이트가 받는 형상(bias가 있는 4비트 또는 8비트 affine, 8행 이하)과 정확히 같으므로, 이 행들은 f16이 이미 타는 qmv / dequantize 교차점을 따릅니다. `qmv_wide_kernel`이 다루지 않는 6비트와 bias 없는 레이아웃은 이전 라우트를 유지합니다. 이 규칙은 8행 이하의 bf16 `[1, B, K]` 입력(짧은 프롬프트, verify 스텝)에도 적용되는데, 이 입력은 이 PR 이전에도 WMMA 커널을 탔습니다. 프로브의 마지막 행은 그 경로도 마찬가지로 느렸음을 보여 줍니다.

### 3.3 확인한 후보

이슈는 후보 세 가지를 지목하고 각각에 대해 판정과 판정 근거 측정을 요구했습니다. 조사 중에 두 가지를 더 확인했습니다.

| 후보 | 판정 | 판정 근거 측정 |
|---|---|---|
| 배치 projection이 `DequantGemm`에 도달해 256 MiB LRU에 들어가지 않는 f16 가중치를 매번 다시 만듦 | batch 4에서 수정 전후 모두 배제 | batch-4 decode 트레이스에는 양자화 임베딩의 스텝당 행 조회 외에 `affine_dequantize`나 Tensile GEMM 디스패치가 없고, 모든 projection이 qmv 실행 한 번(스텝당 161회)입니다. `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=64`에서 batch-4 ~1K decode는 6.2, 6.1, 6.2 tok/s로 그대로였습니다(없을 때 6.1). batch 8에서 `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`은 기본 교차점과 임계값 9 모두에서 hits=0, misses=7728이었고, 모델 수준 스텝은 96.8 ms와 99.3 ms였으므로 거기서도 decode는 LRU에 도달하지 않습니다. miss는 prefill 패스에서 나옵니다(#2232) |
| 배치 qmv 라우트: `[B, 1, K]`를 한 행 곱 `B`개로 읽음 | 확인, 수정 | batch-4 스텝의 95%가 `qmv_warp_shared_batched_kernel`(트레이스 148.5 ms 중 134.8 ms). 프로브의 `[B, 1, K]` 대 `[B, K]` 열. 폴드만으로 서버 스텝이 166.7 ms에서 64.5 ms로 줄어듦 |
| tick당 호스트 측 스케줄러 작업 | 원인에서 배제 | 서버 수준 트레이스: 호스트 갭이 batch 1에서 스텝당 3.1 ms, batch 4 ~1K에서 6.7 ms(GPU busy 91.0%, 88.3%). 반면 스텝 차이는 5배이며 projection 커널로 설명됨 |
| decode 스텝 사이에 끼어든 prefill 청크 | ~1K에서 배제, ~16K 수치를 결정하는 요인으로 확인 | ~1K: 모든 batch-4 레벨의 127개 decode 스텝에서 prefill 청크 0개, mixed 스텝 0개, 점유율 4.00. ~16K: decode 스텝 415개 사이에 prefill 청크 28개, 점유율 1.22, `mixed_steps` 0. 수정 전, 각 요청은 다음 요청의 16K 프롬프트가 prefill되는 동안(2,048토큰 청크 7개, 각각 약 7초) 약 2 tok/s로 decode했고, 혼자 남은 마지막 요청은 16.4 tok/s였음 |
| 하네스 구간에 다른 요청의 prefill이 포함됨 | ~16K에서 확인, ~1K에서는 아님 | ~1K에서는 네 요청이 함께 첫 토큰을 받고 보조를 맞춰 decode함. ~16K에서는 처음 세 요청의 구간 대부분이 다음 요청의 prefill임 |

### 3.4 하네스의 스텝 계산

`bench_serving_metrics.py`는 점유율을 decode 토큰 수 나누기 decode 스텝 수로, decode 스텝을 `tokens_predicted_seconds` 나누기 decode 토큰 수로 보고합니다. 후자는 한 요청의 두 토큰 사이 시간입니다. 이슈가 제안한 비율인 `tokens_predicted_seconds` 나누기 decode 스텝 수는 스텝 하나를 decode 중인 요청마다 한 번씩 세므로, batch 4에서는 스텝의 네 배가 됩니다. 하네스는 둘 다 출력하되 원시 비율에 표시를 붙이고, decode 스텝이 없는 레벨에서는 0으로 나누지 않고 `None`을 반환합니다. `/metrics` 엔드포인트가 없으면 스텝 비용을 알 수 없다는 한 줄을 출력하므로, 이전 빌드에 대해서도 부하 생성기는 그대로 동작합니다.

## 4. 프로덕션 영향

4비트 또는 8비트 affine 체크포인트(f16 또는 bf16)에서 동시 요청이 둘 이상인 ROCm 서버 decode가 대상입니다. gfx1151에서 측정했습니다(Meta-Llama-3.1-8B-Instruct-4bit, f16 scale, `scripts/rocm_gpu_guard.sh` 아래 3회 중앙값).

| 경우 | 요청당 decode tok/s, 수정 전 | 수정 후 | decode 스텝 ms, 수정 전 | 수정 후 |
|---|---:|---:|---:|---:|
| batch 1, ~1K | 32.0 | 32.0 | 31.3 | 31.2 |
| batch 4, ~1K | 6.0 | 15.3 | 166.7 | 65.9 |
| batch 1, ~16K | 24.6 | 24.3 | 40.8(1회) | 41.2 |
| batch 4, ~16K | 5.6 | 7.1 | 383.1(1회) | 364.0 |

단일 스트림 `mlxcel-bench-decode`(pp512, tg128)를 하나의 guard 구간 안에서 수정 전후를 번갈아 실행했습니다. 수정 전 37.51, 37.69, 37.95 tok/s, 수정 후 37.46, 37.84, 37.91 tok/s입니다(중앙값 37.69와 37.84, +0.4%). 단일 스트림 활성값은 `[1, 1, K]`라 폴드가 건드리지 않고, bf16 라우트 변경은 2행 이상에서만 WMMA를 빼므로 단일 스트림 decode는 원래 WMMA를 타지 않았습니다.

bf16 모델 수준 배치 스텝(`profile_batched_decode`, 1,024토큰 프롬프트, 3회 중앙값, LOCAL_FIXES 항목 43의 수치): Qwen3-0.6B-4bit는 batch 2, 4, 8에서 10.7, 18.5, 34.7 ms에서 6.9, 12.0, 22.2 ms가 되었습니다. gemma-3-4b-it-4bit는 어느 빌드, 어느 배치 크기에서도 변하지 않았으므로 그 배치 스텝은 이 projection에 묶여 있지 않습니다. 더 이상 원인을 나누어 보지 않았습니다.

batch-4 ~16K는 요청당 5.6에서 7.1 tok/s로만 바뀌었습니다. 서버는 각 ~16K 프롬프트를 2,048토큰 청크로 prefill하고 청크 사이마다 decode 배치에 스텝 하나를 주므로, decode 중인 요청은 청크당 약 한 토큰씩 진행합니다. batch-4 ~16K 레벨 트레이스(172.4초, GPU busy 98.6%)에서 SDPA가 134.9초(78%)였고, 그중 129.3초가 `kernel_sdpa_flash_wmma<__half, true, 128, 64, 64>` 672회 호출(호출당 약 192 ms)이었습니다. 이는 이 이슈의 범위 밖인 prefill 처리량 문제이며 #2251로 등록했습니다.

Metal과 CUDA는 영향을 받지 않습니다. 변경은 ROCm 오버레이, ROCm 전용 테스트, Python 도구에 있습니다.

## 5. 문서

- **`docs/environment-variables.md`.** `MLX_ROCM_WMMA_QMM` 문단과 표 항목에, 설정하지 않으면 bias가 있는 8행 이하의 4비트와 8비트 GEMM이 WMMA 커널을 건너뛰고 qmv / dequantize 교차점을 따른다는 점, 그 이유(모든 행에 대해 가중치를 한 번 읽음, gfx1151에서 WMMA가 한 행 GEMV의 7~8배), 배치 decode 스텝의 `[B, 1, K]`가 그런 GEMM이라는 점, `MLX_ROCM_WMMA_QMM=1`은 여전히 커널을 강제한다는 점을 적었습니다.
- **`docs/benchmarks.md`.** 배치 서빙 래더 명령이 이제 `--metrics`를 넘기며, 레벨마다 출력되는 decode 스텝, 점유율, prefill 청크, mixed 스텝, decode 스텝 ms에 대한 설명을 붙였습니다.
- **`LOCAL_FIXES.md` 항목 43.** 폴드, 라우트 변경, 측정값, 테스트, fork 정책(mlxcelverse에 유지, upstream에 제안하지 않음)을 기록했습니다.
- **결과 페이지** `docs/benchmark_results/rocm-batched-decode-gfx1151-2026-10-08.md`: 환경, decode 스텝 표, GPU busy 대 호스트 갭 분리, 모델 수준 수정 전후 커널, 프로브 표, bf16 절, 후보 판정, ~16K 분석, 재현 명령. 데이터 디렉터리에는 세션별 guard 로그, 원시 출력, 커널 통계, 하네스 스크립트가 있습니다.

## 6. 검증

gfx1151(Radeon 8060S, ROCm 7.15)에서 모든 GPU 실행을 `scripts/rocm_gpu_guard.sh` 아래에서 했습니다. 컴파일러나 다른 GPU 프로세스가 보인 시도는 버리고 다시 실행했으며, 깨끗한 시도만 보고합니다.

- **회귀 테스트.** `cargo test --release --features rocm --test rocm_qmm_batched_rows -- --test-threads=1`은 `7a3fcc4a` 위로 리베이스하기 전과 후 모두 통과합니다. 폴드 전에는 f16 4096 x 4096 batch-4 경우가 `[B, K]`와 최대 3.9e-3 차이 났으므로, 수정이 없으면 바이트 비교가 실패합니다.
- **dequant 캐시 테스트.** `cargo test --release --features rocm --test rocm_qmm_dequant_cache -- --test-threads=1` 통과.
- **Python.** `python3 -m unittest discover -s tests -p 'test_*.py'`: 105개 통과. `pytest tests/test_bench_serving_concurrency.py`: 12개 통과.
- **린트와 스크립트 게이트.** `cargo clippy --features rocm --example qmm_batch_rows_probe --test rocm_qmm_batched_rows -- -D warnings` 경고 없음. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay verify-binary-assets`와 `cargo test --features rocm --test dead_doc_pointers` 통과.
- **측정.** 1, 3, 4절의 서버 매트릭스, 단일 스트림 decode, 프로브, 트레이스. 명령과 스크립트는 결과 페이지의 데이터 디렉터리에 있습니다.
- **전체 게이트.** `7a3fcc4a` 위의 head `474163e3`에서 `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`로 실행한 `make verify-rocm`: `[verify-rocm] OK`, cargo 테스트 스위트 164개에서 12195개 통과, 0개 실패, 403개 무시(mlxcel-core lib 스위트에서 8906개 통과, 165개 무시), ROCm 스모크는 GPU에서 32개 토큰을 생성했습니다.

검증하지 않은 것:

- **Metal 기준 비율**(이슈 3단계). gfx1151 머신에는 Metal 호스트가 없습니다. 결과 페이지에 측정하지 않았다고 기록했습니다.
- **Metal과 CUDA 빌드.** 이 호스트에서 사용할 수 없습니다. Metal이나 CUDA 코드는 건드리지 않았습니다.
- **~16K에서의 모델 수준 프로브.** `examples/profile_batched_decode.rs`는 청크로 나누지 않은 16K prefill에서 `hipLaunchKernel ... invalid configuration argument`로 중단되며, `2aa5211f` 기준 수정 전후 빌드 모두 그렇습니다. 서버는 prefill을 청크로 나누므로 영향이 없습니다. 이 측정 뒤에 머지된 #2237이 strided 복사 커널의 launch grid를 바꾸었지만, 그것이 이 중단을 고치는지는 확인하지 않았습니다.

## 7. 기술적 선택과 그 이유

- **고치기 전에 측정.** 하네스 확장을 먼저 넣었으므로 수정 전 수치부터 decode 스텝과 끼어든 prefill이 분리되어 있습니다. 그것이 없으면 ~1K와 ~16K의 차이가 크기만 다른 한 문제처럼 보였고, 있으니 원인이 다른 두 문제로 보였습니다.
- **모델 코드가 아니라 `eval_gpu`에서 폴드.** 각 모델의 배치 forward에서 `[B, 1, K]`를 `[B, K]`로 reshape하면 호출자 하나만 고치고 다음 호출자에게 같은 함정을 남깁니다. 백엔드에서 접으면 모든 호출자가 고쳐지고, Metal 백엔드가 같은 호출을 읽는 방식과 일치하며, 복사 비용도 없습니다.
- **strided 활성값은 배치 경로에 유지.** 행 연속이 아닌 뷰를 접으려면 복사나 다른 인덱싱이 필요합니다. 배치 커널은 이미 이를 올바르게 처리합니다.
- **라우트 변경을 `qmv_wide_kernel` 형상에 맞춤.** wide GEMV가 가중치 한 번 읽기로 모든 행을 처리할 수 있는 곳에서만 WMMA를 건너뛰므로, 행이 더 느린 대체 경로로 가지 않습니다. wide GEMV가 다루지 않는 6비트와 bias 없는 레이아웃은 라우트를 유지합니다.
- **오버라이드 유지.** `MLX_ROCM_WMMA_QMM=1`은 여전히 WMMA 커널을 강제하므로, 어떤 디바이스나 형상이 그쪽을 선호한다고 밝혀져도 변수 하나로 이전 동작을 쓸 수 있습니다.
- **~16K 수치는 여기서 쫓지 않음.** 트레이스는 시간이 prefill 어텐션에 있음을 보였고, 이슈는 이를 범위 밖으로 명시합니다. 커널별 근거와 함께 #2251로 등록했습니다.
- **테스트에서 바이트 동일성 확인.** `[B, 1, K]`, `[B, K]`, `[1, B, K]`가 같은 바이트를 반환하는지 단언하면 세 레이아웃이 같은 커널을 탄다는 것을 확인할 수 있습니다. 기준값에 대한 허용 오차 검사로는 알 수 없는 부분입니다.

## 8. 남은 위험과 후속 작업

- **#2251: 긴 문맥 청크 prefill을 flash SDPA가 지배함.** 최대 16K 키에 대한 2,048 쿼리 청크가 호출당 약 192 ms로, 대략 1.5 TFLOP/s 수준이며 디바이스가 아니라 커널이 속도를 제한함을 시사합니다(확정되지 않음). 이것이 고쳐질 때까지 batch-4 ~16K 요청당 decode는 약 7 tok/s에 머뭅니다.
- **모델 수준 프로브의 16K 중단.** `profile_batched_decode`는 청크로 나누지 않은 16K prefill에서 "invalid configuration argument"로 실패합니다. 이 이슈의 범위 밖이며 #2237 이후 다시 확인하지 않았습니다.
- **8보다 큰 배치 크기.** batch 9 이상에서는 접힌 행이 `qmv_wide_kernel` 범위를 벗어나 해당 행 수에 대해 기존 라우트가 고르는 경로를 탑니다(bf16은 128행 상한 아래에서 WMMA 커널). batch 2, 4, 8만 측정했고, 이슈의 매트릭스는 `--parallel 4`를 씁니다.
- **라우트 규칙이 wide 커널의 opt-out을 확인하지 않음.** `wide_qmv_rows`는 `qmv_wide_kernel`의 비트, bias, 행 수 조건을 따라가지만 `MLX_QMV_NO_WIDE`는 확인하지 않습니다. 이 변수가 설정되면 소수 행 bf16은 WMMA를 건너뛰고 tiled 또는 warp-shared GEMV로 갑니다. 이 조합은 측정하지 않았습니다.
- **낡은 테스트 주석.** `decode_batch_rows_are_one_product`는 bf16 경우가 fused WMMA 라우트를 탄다고 설명합니다. 라우트 변경 후에는 qmv를 타므로 이 테스트는 해당 크기에서 더 이상 WMMA를 실행하지 않습니다. 주석만 틀렸고 단언은 유효합니다.
- **Gemma 3 4B 배치 스텝.** 어느 빌드에서도 변하지 않았으므로 다른 무언가가 제한하고 있습니다. 조사하지 않았습니다.

## 9. 학습 포인트

- **하네스가 문제의 그 양을 보고하게 만들 것.** 클라이언트 측 tok/s는 decode 스텝과 다른 요청의 prefill을 섞었습니다. 서버 자체 카운터로 레벨별 점유율과 스텝 시간을 출력하자, 막연한 차이가 prefill이 섞이지 않은 점유율 4.00의 166.7 ms 스텝으로 드러났습니다.
- **호출자가 실제로 쓰는 레이아웃을 확인할 것.** `[B, K]`는 이미 하나의 곱으로 실행되었고, 배치 decode가 쓰는 `[B, 1, K]`만 원소별 경로를 탔습니다. 같은 행을 모든 레이아웃으로 측정하는 프로브가 한 번의 실행으로 이를 찾았습니다.
- **배치가 순차 호출보다 느리면 디스패치 문제입니다.** batch-4 `forward_batched` 157.4 ms 대 `forward` 네 번 111.6 ms라는 결과가 연산이나 스케줄링에 기반한 설명을 모두 배제했습니다.
- **행 수를 바꾸는 수정은 행을 라우트 임계값 너머로 옮길 수 있습니다.** 폴드는 f16에서는 옳았지만 bf16 행을 소수 행에서 느린 커널로 보냈습니다. 변경을 끝내기 전에 두 번째 dtype을 측정해서 batch 2에서 약 2배의 회귀를 잡았습니다.
- **기준 백엔드가 같은 호출을 어떻게 읽는지 비교할 것.** Metal 백엔드는 이미 배치를 접고 있었습니다. ROCm 오버레이의 차이가 결함이었고, Metal 코드가 수정의 정확한 조건을 알려 주었습니다.
