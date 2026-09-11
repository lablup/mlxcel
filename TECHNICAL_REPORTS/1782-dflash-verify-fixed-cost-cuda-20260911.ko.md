# GB10에서 다중 행 speculative verify의 고정 비용은 어디에 있었나, 그리고 두 가지 수정

이슈 #1782. 호스트 GB10(sm_121), MLX 핀 `81ba1c6a`, CUDA release 빌드. 전체 측정 기록: `docs/benchmark_results/dflash-verify-fixed-cost-gb10-2026-09-11.md`.

## 배경

PR #1771은 이 호스트에서 Laguna DFlash 페어링을 측정해 어떤 블록 폭도 classic decode를 이기지 못한다는 결과를 냈다. verify 블록의 디바이스 비용은 고정 77 ms에 행당 3.3 ms를 더한 꼴로 맞았고, 2행에서도 단일 토큰 스텝의 2.7배였다. PR은 이를 다중 행 forward가 CUDA 백엔드에서 "eager하게, launch-bound로" 돈다고 해석했다. 이슈는 그 해석을 이미 검증한 상태로 작성됐다. 핀된 MLX의 capture 경로에는 행 수에 따라 갈라지는 코드가 없으므로, 고정 항을 먼저 귀속시킨 뒤에 고쳐야 했다. 이슈는 그래프 쪽 후보 셋(GB10의 20 op / 25 MB 그래프 예산, `subgraph_to_key`가 `is_updatable`을 지우는 경우, 키 변동)을 들고, 코드를 건드리지 않는 두 환경변수 실험을 먼저 하라고 했다.

Qwen 3.5 페어링(`qwen3.5-4b-4bit`와 `qwen3.5-4b-dflash`)을 쓴 이유는 `main`에서 `mlxcel-server`로 바로 돌기 때문이다. Laguna 쪽은 #1771이 필요하다.

## 측정이 말한 것

모든 arm이 같은 바이너리, 설정마다 서버 하나, 고정된 코드 프롬프트로 greedy 스트리밍 200토큰, 워밍업 1회 폐기 후 n = 3, GPU 락 보유, 유휴 호스트. classic decode 58.42 tok/s(스텝당 17.1 ms). 블록 2는 25.82 tok/s, verify 라운드당 디바이스 작업 42.6 ms로 classic 스텝의 2.5배, Laguna의 2.7배와 같은 모양이다.

그래프 제어 변수는 가장 싼 실험이었고 2행에서는 효과가 없었다. `MLX_MAX_MB_PER_BUFFER=400`(GB10의 25 MB를 H100 값으로)은 블록 2를 오히려 조금 늦추고 블록 8에서 약 5%를 줬다. `MLX_MAX_OPS_PER_BUFFER=100`은 블록 2에 변화가 없고 블록 8에서 약 10%를 줬다. `MLX_USE_CUDA_GRAPHS=0`은 classic을 8% 늦추면서 블록 8은 6% 빠르게 만들었다. capture에서 얻는 게 없는 경로라면 capture 문제가 아니다. 그래프 재사용은 모든 arm에서 정상이었다(블록 2: `cudaGraphInstantiate` 260회 대 `cudaGraphExecUpdate` 71089회). #1545의 Volta 결과와 같다.

verify 라운드당 nsys 커널 표(`--cuda-graph-trace=node`, 프로파일링이 실행을 15~46% 부풀리므로 비율과 launch 수만 비교)는 고정 항을 두 곳에 두었고, 둘 다 그래프 기계가 아니었다.

| 범주 | 블록 2, 라운드당 | 블록 8 | 블록 16 |
|---|---|---|---|
| f32로 도는 드래프터: `copy_v<__half, float>` 가중치 upcast와 f32 cutlass GEMM | 38.0 ms, 243 launch | 31.2 ms, 239 | 30.0 ms, 228 |
| GDN chunked scan: 63단계 Horner 루프의 5 us 배치 matmul과 f32 subtract | 21.9 ms, 4364 | 40.3 ms, 7170 | 38.9 ms, 7266 |
| 타깃 양자화 projection | 15.7 ms (`qmv_multirow`) | 43.5 ms (`qmm_sm80`) | 43.7 ms |
| 마스크를 materialize한 타깃 attention | 0.3 ms | 1.1 ms | 2.0 ms |

1. 드래프터가 float32로 돌고 있었다. `DFlashDrafter::load`는 bf16 텐서를 조건 없이 f16으로 바꿨고(Apple Silicon 규칙), 타깃 로더는 Ampere 이후 CUDA에서 bf16을 유지한다. MLX는 bf16 활성값과 f16 가중치의 곱을 float32로 승격하는데, 드래프터가 읽는 residual stream은 타깃의 bf16 hidden state이므로 첫 matmul에서 승격이 일어나고 그 뒤는 전부 f32였다. 라운드마다 드래프터 가중치 540M개를 72번 upcast하고(grid 크기가 정체를 말해 준다: 32.8M은 `fc.weight`, 24.9M은 MLP 행렬, 10.5M은 `o_proj`), 그 뒤에 0.75~1.2 ms짜리 f32 GEMM이 따라온다. 행 수와 무관하니 고정 항의 모양이다.
2. 24개 gated-delta linear attention 레이어가 모든 verify 블록에서 64행 chunked scan을 탔다. `gated_delta_chunked`는 `T`가 얼마든 레이어마다 63단계 유한 Neumann 급수로 `(I + T)`를 역산하므로 2행과 16행이 같은 1512개 작은 matmul과 1512개 subtract를 낸다. 블록 8과 16에서는 부분 accept가 accept된 행을 같은 scan으로 다시 돌리기 때문에 라운드당 1.65배가 된다. CUDA에서는 verify 패스가 요청한 chain-parity 플래그를 ops 폴백이 무시했으므로 블록이 단일 토큰 체인과 비트 동일하지도 않았다. 변경 전 블록 16은 classic greedy 텍스트와 갈라졌다.
3. 행당 항은 attention이 아니라 양자화 projection 커널의 전환이다. 2~7행은 `qmv_multirow_kernel`로 가고(2행 verify는 가중치에 대해 classic 스텝보다 15% 더 낸다), `M = 8`부터 디스패처가 `qmm_sm80_kernel`로 바꾸며 이는 단일 행 비용의 3.5배이고 8~16행 사이에서는 평평하다. head_dim 256 레이어 여덟 개의 materialize된 마스크 attention은 라운드당 2 ms 미만이다.

## 변경

`mlxcel-core`로 로드되는 드래프터(DFlash, Muse Glimmer, Inkling MTP, Qwen 3.5 MTP)는 이제 `apply_drafter_load_dtype_policy`를 적용한다. 이는 비양자화 체크포인트에 대한 타깃 로더의 `bf16_to_f16_at_load`를 그대로 따른다. Apple Silicon과 Ampere 이전 CUDA에서는 f16, Ampere 이후 CUDA에서는 bf16, `MLXCEL_KEEP_BF16`과 `MLXCEL_CUDA_F16_NORMALIZE`는 같은 방식으로 존중한다. 순수 정책 함수는 모든 분기를 단위 테스트한다.

gated-delta ops 폴백은 마스크 없고 스칼라 게이팅인 `CHAIN_PARITY_SEQUENTIAL_MAX_T = 32`행 이하 블록에 대해 chain parity를 지킨다. fused 단일 스텝의 순차 루프를 도는데, 이는 `gated_delta_step`을 `T`번 적용한 것, 즉 classic 체인 자체의 산술이므로 블록이 `T`개 단일 토큰 스텝과 비트 동일하다. 테스트가 ops 경로에서 이를 chunked scan(근사만 맞는다)과 대비해 고정한다. 더 넓은 블록과 prefill은 scan을 유지한다. 두 변경 모두 같은 바이너리에서 끌 수 있는 스위치를 둔다(`MLXCEL_CUDA_F16_NORMALIZE=1`, `MLXCEL_GDN_CHAIN_PARITY=0`).

## 변경 후

| 설정 | n | tok/s (최소~최대) | classic 대비 | 라운드 디바이스 sync | greedy 텍스트 == classic |
|---|---|---|---|---|---|
| classic | 3 | 58.33 (57.27~59.38) | | | 예 |
| 블록 2 | 3 | 72.57 (72.21~72.99) | 1.24x (전 0.44x) | 18.6 ms (전 42.6) | 예 |
| 블록 3 | 3 | 76.42 (76.36~76.51) | 1.31x | 24.4 ms | 예 |
| 블록 4 | 3 | 76.88 (76.62~77.28) | 1.32x (전 0.53x) | 29.9 ms (전 58.5) | 예 |
| 블록 6 | 3 | 59.06 (58.84~59.19) | 1.01x | 47.4 ms | 예 |
| 블록 7 | 3 | 53.45 (53.32~53.59) | 0.92x | 53.9 ms | 예 |
| 블록 8 | 3 | 48.84 (48.48~49.32) | 0.84x (전 0.49x) | 59.2 ms (전 81.6) | 예 |
| 블록 16 | 3 | 48.63 (48.48~48.83) | 0.83x (전 0.53x) | 64.1 ms (전 78.7) | 예 (전에는 아니오) |
| 스위치 둘 다 끔, 블록 2 | 3 | 25.98 (25.90~26.12) | 0.45x | 42.2 ms | 예 |
| 드래프터 수정만, 블록 2 | 3 | 36.85 (35.97~38.13) | 0.63x | 40.9 ms | 예 |
| GDN 수정만, 블록 2 | 3 | 35.87 (35.76~36.00) | 0.61x | 25.9 ms | 예 |

2행 verify는 이제 classic 스텝의 1.09배다. 각 수정 단독으로는 36 tok/s 근처이고 둘을 합치면 72.6인데, 라운드가 호스트 draft 빌드, 호스트 verify 빌드, 디바이스 sync의 직렬 사슬이라 각 수정이 서로 다른 고리를 없애기 때문이다. greedy 텍스트는 모든 arm에서 classic과 바이트 단위로 같고, 블록 16도 chain-parity 루프 덕에 다시 일치한다.

## 고치지 않은 것과 권고

폭 곡선은 4행과 6행 사이에 절벽이 있다(4행 1.32x, 6행 1.01x, 7행 0.92x). multirow `qmv` 커널은 누산기 폭을 2, 4, 8로 디스패치하므로 5~7행은 레지스터 비용이 큰 8폭 인스턴스를 탄다. 블록 8과 16은 이기지 못한다. 남은 비용은 `M >= 8`에서의 `qmm_sm80` 항으로 여기서 건드리지 않았고, 체크포인트의 기본 폭 16은 그 전환의 반대편에 있다. 이 페어링의 측정된 설정은 `--draft-block-size 4`다. 하드웨어별 기본 폭을 정하는 것은 패밀리마다 교차점이 다른 정책 문제라(NVFP4 체크포인트의 `fp_qmv`도 같은 8행 경계를 가진다) 후속 이슈로 제안한다. `MLX_MAX_OPS_PER_BUFFER=100`이 모델 하나의 classic arm에서 +3%를 낸 것도 이 이슈가 아니라 CUDA decode 경로 전체의 문제이므로 별도 이슈로 제안한다.

Laguna 페어링은 다시 측정하지 않았다(#1771 필요). 그 드래프터도 같은 f16 변환을 거쳤으므로 첫 번째 수정이 적용되고, gated-delta 레이어가 없으므로 두 번째는 해당하지 않는다.

## 영향 범위

dtype 정책은 `mlxcel-core`로 로드되는 모든 드래프터 패밀리에 닿는다. gated-delta 변경은 Metal이 아닌 모든 백엔드에서 chain-parity 호출 지점(Qwen 3.5 verify와 rollback replay)에 닿고, parity 플래그만이 새 분기를 고르므로 `gated_delta_ops`의 다른 호출자에는 닿지 않는다. MLX 핀과 C++ 브리지는 바뀌지 않았다. 측정한 것은 Qwen 3.5 DFlash 페어링뿐이고, 나머지 패밀리는 코드 경로만 공유할 뿐 아직 숫자가 없다.

## 검증

`cargo fmt --all -- --check` 통과. 단위 테스트: `drafter.rs`의 정책 및 env 플래그 파싱 테스트, `gated_delta_tests.rs`의 비트 동일성 테스트, 기존 드래프터와 gated-delta 스위트를 이 호스트에서 test-fast 프로파일로 좁은 선택자로 실행했다(범위를 지정하지 않은 `cargo test --lib`는 이 변경과 무관한 `cudaStreamEndCapture` C++ abort로 이 호스트에서 중단되고, `make verify-test-cuda`는 워크스페이스 전체라 watchdog 아래에서 돌릴 수 없었다). `metal,accelerate` 게이트는 이 Linux/CUDA 호스트에서 실행할 수 없어 돌리지 않았다.
