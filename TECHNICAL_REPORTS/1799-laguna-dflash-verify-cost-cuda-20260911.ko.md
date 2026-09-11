# GB10에서 Laguna DFlash 검증 라운드의 고정 비용이 어디에 있는지, 그리고 수정

이슈 #1799. 호스트 GB10 (sm_121), MLX 핀 `81ba1c6a`, CUDA 릴리스 빌드, 툴체인 1.97.1, 소스 트리 `2deae307` (PR #1771의 스쿼시 머지. 이전 스윕이 측정한 PR 헤드 `dab18fdf`와 바이트 단위로 동일). 전체 측정 기록: `docs/benchmark_results/laguna-dflash-verify-cost-gb10-2026-09-11.md`.

## 배경

PR #1771은 Laguna DFlash 드래프터를 fail-closed 정확성 게이트 뒤에 출시하면서, 이 호스트에서는 어떤 블록 폭도 클래식 디코드를 이기지 못한다고 측정했다. 검증 라운드의 디바이스 시간은 고정 77 ms에 행당 약 3 ms를 더한 형태였고 2행에서도 단일 토큰 스텝의 2.5배였으며, PR은 이를 다중 행 forward가 "eager하게, launch-bound로" 실행되기 때문이라고 설명했다. 이슈는 그 표현을 가설로 낮춘 상태로 작성되었다. 핀된 MLX의 캡처 경로에는 행 수에 따라 분기하는 코드가 없고, #1782가 Qwen 3.5에서 같은 직관을 이미 반증했으며, #1782의 두 수정은 여기에 해당하지 않았다(이 드래프터는 이미 bf16이었고, Laguna에는 gated-delta 레이어가 없다). 그 스윕은 다른 세션의 빌드가 CPU를 점유한 상태에서 돌기도 했다. 그래서 과제는 유휴 호스트에서 `nsys --cuda-graph-trace=node`로 먼저 귀속하는 것이었고, 측정된 부정 결과도 유효한 종료로 인정되었다.

## 측정이 말한 것

유휴 호스트(1분 부하 0.6 미만에서 시작, 실행마다 load1 기록, `ps`에는 측정 대상 프로세스만 보임), 모든 arm이 같은 바이너리, #1782의 152토큰 원시 코드 프롬프트로 `mlxcel generate`, 폭 2에서 16을 인터리브, n = 3에 최소와 최대 보고. 유휴 호스트는 이 페어링을 구해주지 않았다. 클래식 29.28 tok/s(스텝당 34.2 ms), 블록 2는 0.52x, 블록 6이 최선으로 0.72x, 블록 16은 0.52x. 라운드당 디바이스 동기화는 최소제곱으로 71.6 ms 더하기 행당 3.79 ms였고, 라운드의 세 직렬 링크(블록 2에서 드래프터 호스트 빌드 31 ms, 검증 호스트 빌드 4 ms, 디바이스 동기화 80 ms)는 115 ms의 라운드 벽시계와 0.1 ms 안에서 합이 맞았다.

그래프 쪽 후보는 모두 경계값과 함께 부정으로 나왔다. `MLX_MAX_MB_PER_BUFFER=400`, `MLX_MAX_OPS_PER_BUFFER=100`, 둘 다 올린 것, `MLX_USE_CUDA_GRAPHS=0`은 각각 2행 디바이스 동기화를 최대 3%, 라운드를 최대 4% 움직였다. 캡처를 끄면 Qwen 3.5에서처럼 블록 8이 오히려 빨라진다. 그래프 재생은 건강하다(모든 폭에서 `cudaGraphExecUpdate` 1000회당 `cudaGraphInstantiate` 1회 미만)므로 재인스턴스화는 배제된다. 바이트 예산은 클래식 arm에서 8%의 가치가 있지만(카운터가 요소 수를 더하므로 256-expert nvfp4 스택 하나가 GB10의 25 "MB" 상한을 혼자 넘는다) 두 arm을 갈라놓지는 않는다.

프로파일은 두 항을 서로 다른 곳에 두었다. 같은 구성의 400토큰 프로파일과 200토큰 프로파일의 차이를 취하면(로드, 프리필, 정확성 프로브가 상쇄된다) 블록 8에서 라운드당 142.6 ms 중 GPU 커널 시간은 86.7 ms로 점유율 61%이고, 클래식 스텝은 96%다. 행당 항은 라우팅된 expert의 GPU 작업이다. `gather_qmm`을 거치는 `qmm_sm80`이 8행에서 39.4 ms, 16행에서 59.4 ms로 행당 2.5 ms이며, (행, expert) 쌍마다 선택된 expert를 한 번씩 읽는 하한에 가깝다. 이슈의 전제 두 가지가 여기서 무너졌다. 이 체크포인트의 attention 프로젝션, gate, 라우터, lm_head는 순수 bf16이고(1행에서는 `gemv_single`, 2행 이상에서는 cutlass bf16 GEMM) expert만 NVFP4라서 프로젝션에는 `fp_qmv`에서 `qmm_sm80`으로의 전환이 존재하지 않는다. 그리고 `SwitchGLU`는 입력을 확장해 `gather_qmm`이 `M = 1, B = 8n`을 보므로 expert는 클래식을 포함한 모든 폭에서 `qmm_sm80`을 탄다.

고정 항은 한 프리미티브의 호스트 시간이다. 같은 방식으로 차이를 취한 MLX의 NVTX 범위는 `ScaledDotProductAttention::eval_gpu`가 블록 8에서 라운드당 67.1 ms, 블록 16에서 75.8 ms인 반면 클래식 토큰당 2.8 ms임을 보여준다. 호출 수는 같은 45회(타깃 40 레이어와 드래프터 5 레이어)다. 1행 디코드 호출은 `sdpa_vector` 커널을 탄다. head_dim 128, bf16의 다중 행 마스크 호출은 cuDNN 대상이 되고, MLX는 cuDNN 실행 플랜을 query, key, value, mask의 정확한 shape과 stride를 키로 하는 LRU에 캐시한다. 검증 라운드는 KV 캐시에 행을 덧붙이므로 세 shape 클래스(타깃 full 레이어, sink가 있는 타깃 sliding 레이어, 드래프터 레이어) 각각에서 key 길이와 mask shape이 매 라운드 새롭다. 라운드당 세 번의 miss, 각각 약 22 ms의 cuDNN frontend 그래프 빌드. 행 수와 무관하고, 어떤 그래프 노브에도 보이지 않으며, 드래프터의 호스트 빌드가 폭과 무관하게 30 ms였던 것도 설명한다(다섯 레이어가 세 빌드 중 하나를 낸다).

같은 LRU는 평생 miss 카운터를 유지하고 용량의 두 배(512)를 넘으면 MLX의 치명적 `Cache thrashing` 오류를 던진다. 라운드당 miss 세 번이면 프로세스당 약 170 라운드다. 200토큰을 넘는 모든 블록 2 프로파일이 정확히 그 오류로 중단되었고(400토큰 두 번, 300토큰 한 번), 200토큰 실행과 400토큰의 블록 8, 16 실행(163, 161 라운드)은 완료되었다. 즉 이 이슈 이전에는 약 170 라운드를 넘는 Laguna DFlash 생성이 프로세스를 끝냈다.

무코드 대조군이 메커니즘을 끝까지 확인했다. `MLX_CUDA_USE_CUDNN_SDPA=0`은 모든 폭에서 디바이스 동기화 39 ms와 드래프터 호스트 빌드 21 ms를 덜어내고, 클래식 arm은 건드리지 않으며, 블록 4를 클래식 대비 1.34x, 블록 8을 1.15x에 놓는다. 다만 프리필도 cuDNN의 flash 커널에서 떼어내므로 수정안은 아니다.

## 변경

두 부분이고 둘 다 환경 변수 킬 스위치가 있으며, 라운드 루프나 Laguna 소스는 건드리지 않는다.

1. 패치된 `mlx/backend/cuda/scaled_dot_product_attention.cpp`(핀된 파일에 게이트 하나를 더한 것으로, 기존 `.cu` 패치와 같은 오버레이 방식): `supports_sdpa_cudnn`이 더 긴 key 시퀀스 위의 2에서 `MLXCEL_SDPA_FALLBACK_MAX_QUERIES`(기본 32)행 마스크 호출을 거부한다. 그런 호출은 MLX 자체의 ops fallback(CPU 백엔드와 Qwen 3.5의 head_dim-256 검증이 이미 쓰는 것과 같은 산술, sink 포함)을 타며 shape별 빌드 비용이 없다. 1행 디코드 스텝(vector 커널)과 프리필(key 길이가 query 길이와 같거나, 경계보다 행이 많다. 청크 프리필의 마지막 짧은 청크만 경로가 바뀐다)은 그대로다. `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0`이면 재빌드 없이 업스트림 디스패치로 돌아가며, 이것이 A/B의 킬 스위치다.
2. `MLX_CUDA_SDPA_CACHE_SIZE`가 CUDA 빌드에서 `hardware::apply_cuda_sdpa_cache_default`를 통해 2000으로 기본 설정된다. #818 그래프 캐시 기본값의 자매로, 같은 네 진입점에서 같은 env-wins 계약으로 적용된다. 게이트가 있으면 검증은 더 이상 플랜 캐시를 miss하지 않지만, 장수 서버에서는 프리필 프롬프트 길이의 다양성만으로도 512 miss 중단을 넘고, 그 중단은 요청 오류가 아니라 프로세스 종료다.

## 수정 후

같은 바이너리, 같은 방법, 조용한 호스트, arm당 n = 3. `ks-*`는 `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0`이다.

| 구성 | n | tok/s 평균 (최소에서 최대) | 클래식 대비 | 라운드 벽시계 ms | 라운드당 디바이스 동기화 ms |
|---|---|---|---|---|---|
| 클래식 | 3 | 28.78 (28.53에서 29.00) | | 토큰당 34.7 | |
| 블록 2 | 3 | 32.44 (32.06에서 33.13) | **1.13x** (이전 0.52x) | 54.6 (이전 115.3) | 42.0 (이전 80.4) |
| 블록 4 | 3 | 38.40 (37.81에서 39.19) | **1.33x** (이전 0.70x) | 62.8 (이전 118.6) | 48.9 (이전 84.9) |
| 블록 6 | 3 | 38.05 (37.99에서 38.10) | **1.32x** (이전 0.72x) | 71.0 (이전 126.6) | 56.2 (이전 92.5) |
| 블록 8 | 3 | 32.19 (31.68에서 32.86) | **1.12x** (이전 0.65x) | 81.8 (이전 138.3) | 66.8 (이전 103.7) |
| 블록 10 | 3 | 29.07 (28.42에서 29.49) | 1.01x | 89.4 | 74.6 |
| 블록 12 | 3 | 28.58 (27.89에서 28.95) | 0.99x | 95.9 | 80.6 |
| 블록 16 (체크포인트 기본값) | 3 | 23.91 (23.73에서 24.05) | 0.83x (이전 0.52x) | 108.6 (이전 166.9) | 93.5 (이전 131.3) |
| ks, 블록 2 | 3 | 15.42 (15.35에서 15.51) | 0.54x | 114.8 | 81.9 |
| ks, 블록 8 | 3 | 18.90 (18.69에서 19.25) | 0.66x | 139.3 | 104.9 |
| ks, 블록 16 | 3 | 15.67 (15.36에서 15.93) | 0.54x | 163.6 | 129.9 |

킬 스위치는 모든 폭에서 기준선을 2% 안에서 재현한다. 디바이스 동기화는 이제 `36.0 + 행당 3.6 ms`(이전 `71.6 + 3.79`)로, 바닥은 클래식 스텝의 2.1배에서 1.05배로 내려왔고 행당 기울기는 expert 읽기 그대로다. 블록 4와 6은 전체 범위가 클래식 범위 위에 있어 이긴다. 교차점은 10행과 12행 사이이고 체크포인트 기본값 16은 여전히 가장 나쁜 폭이다(0.83x). 수정 후 프로파일된 라운드에는 `ScaledDotProductAttention` 호스트 범위가 전혀 없고(fallback은 일반 op들이다), 블록 2는 벽시계 53.8 ms에 커널 54.7 ms로 GPU-bound이며, fallback은 cuDNN flash 커널보다 라운드당 GPU 시간을 약 10 ms 더 쓰지만 호스트 시간 69 ms를 없앤다.

같은 바이너리에서 #1782 서버 하네스로 잰 Qwen 3.5: 클래식 56.07, 블록 2 70.56(1.26x. #1782는 1.24x), 블록 4 75.58(1.35x. 1.32x), 두 폭 모두 3회 중 3회 greedy 텍스트가 클래식과 바이트 동일.

## 영향 범위: 측정한 것과 경로만 공유하는 것

측정한 것: Laguna 페어링(블록 2에서 0.52x에서 1.13x로, 블록 4에서 0.70x에서 1.33x로, 같은 바이너리의 킬 스위치 A/B)과 교차 패밀리 대조군인 Qwen 3.5(새 바이너리에서 1.26x와 1.35x, 텍스트가 클래식과 바이트 동일). Qwen 3.5의 head_dim 256은 애초에 cuDNN 대상이 아니므로 게이트가 닿을 수 없고, 그 수치는 재빌드가 다른 것을 바꾸지 않았음을 보여준다.

수치 없이 경로만 공유하는 것: Ampere 이후에서 2에서 32행의 query와 선행 컨텍스트를 가진 마스크 SDPA 호출이 `scaled_dot_product_attention.cpp`에 닿는 다른 모든 CUDA 모델. 타깃의 head_dim이 128 이하인 모든 페어링의 투기적 검증(트리 안에는 Muse Glimmer DFlash, LFM2와 LFM2.5 DSpark, Inkling MTP가 있고 여기서 다시 프로브하지 않았다. 정확성 프로브가 뒤집히면 출력이 망가지는 것이 아니라 투기가 거부된다), 재사용된 prefix-cache 접두사 위의 짧은 증분 프리필(긴 key 길이 위의 2에서 32개 새 행. 서버에서 흔한 shape이지만 여기서는 어떤 arm도 밟지 않았고, 모든 arm이 152토큰 프롬프트를 썼다), 그리고 청크 프리필의 마지막 짧은 청크. 이 호출들은 이제 cuDNN 대신 MLX fallback의 산술로 돈다. 일반 `DFlashAttention::forward`는 마스크를 넘기지 않으므로 head_dim 128 이하의 Laguna 아닌 DFlash 드래프터는 여전히 라운드마다 플랜을 하나 다시 빌드하는데, main에는 그런 페어링이 없다. Metal과 CPU는 영향이 없다. Laguna 정확성 게이트는 건드리지 않았고, 이 호스트에서의 거부(logit 동점에서 `qmm_sm80`과 `fp_qmv`의 반올림 차이)는 별도 이슈다. 캐시 크기 기본값은 모든 CUDA 프로세스에 닿는다. 캐시가 `thread_local`이라 2000은 MLX eval 스레드당이고, 빌드된 cuDNN 그래프 2000개짜리 캐시의 메모리는 측정하지 않았다.

## 고치지 않은 것

fallback의 비용은 key 길이에 따라 커진다. 라운드당 약 66 ms의 고정 호스트 시간을 `[B, heads, q_len, k_len]` 점수 행렬 위의 GPU 작업으로 바꾸는 것이라, key 350개의 블록 2에서 라운드당 약 10 ms가 더 들고, cuDNN의 고정 플랜 비용이 오히려 싸지는 지점은 블록 16 기준 key 토큰 수만 개 대다. 게이트는 `q_len`만 제한하며, 어떤 arm도 key 350개를 넘지 않았다. 청크 예산은 이제 `layers::attention`과 `causal_attention`을 거치는 우회 호출을 덮지만, fast SDPA를 직접 부르는 네 sink 경로는 이전처럼 그 예산을 우회한다(Laguna는 512-key 윈도우로 제한되고, gpt_oss의 full-attention sink 레이어는 제한이 없다).

일반 수정은 여기서 일부러 시도하지 않았다. MLX는 cuDNN 플랜 캐시를 정확한 key 길이와 mask shape으로 키잉하므로 검증 라운드는 구조적으로 캐시에 맞을 수 없다. MLX 자체의 `use_cudnn_for_decoding` 경로가 고정 크기 캐시에 대한 1행 호출에 이미 하듯이 key 길이를 패딩하거나 버킷으로 묶으면 캐시가 맞고 cuDNN의 더 빠른 flash 커널을 유지해 fallback이 쓰는 10 ms를 되찾을 수 있다. 그 캐시 키 설계가 이 이슈 밑의 결함이고 메인테이너가 별도 이슈로 올리고 있다. 약 170 라운드를 넘는 모든 블록 2 프로파일을 중단시킨 LRU의 평생 miss 카운터는 그 miss 경로가 단지 느린 것이 아니라 치명적이라는 증거이며, `MLX_CUDA_SDPA_CACHE_SIZE=2000` 기본값은 평생 예산을 키울 뿐이다.

그 밖에 남는 것은 행당 expert 항(행당 2.5 ms. 같은 expert로 라우팅되는 행들이 읽기를 공유하지 않는 한 MoE 검증에 내재한다)과 1행 gemv 대비 bf16 GEMM의 고정 12 ms다. 수정 후 폭 곡선은 블록 4에서 정점이고 체크포인트 기본값 16은 여전히 가장 나쁜 폭이며, 이는 #1797의 정책 질문이다. GB10의 기본 그래프 예산이 이 MoE 모델에서 캡처를 순손실로 만든다는 클래식 arm 발견(그래프 끄면 +12%, 두 예산 다 올리면 +16%, Qwen 3.5와 반대 부호)은 #1798을 위해 기록했고 여기서는 바꾸지 않았다.

## 검증

패치한 파일은 핀된 업스트림 파일에 게이트 하나를 더한 것이다. `git show 81ba1c6a:mlx/backend/cuda/scaled_dot_product_attention.cpp`와 `src/lib/mlx-cpp/patches/mlx/backend/cuda/scaled_dot_product_attention.cpp`의 차이는 정확히 두 hunk, 헤더 주석과 `supports_sdpa_cudnn` 안의 게이트뿐이다. 빌드 배선은 추가하지 않았다. `src/lib/mlx-cpp/CMakeLists.txt`가 이미 `patches/mlx/backend/cuda/*`를 받아온 MLX 트리에 glob으로 덮어쓰기 때문이다.

게이트가 거부하는 호출은 모두 갈 길이 있고, 그 길들은 cuDNN의 의미와 일치한다. 배열 마스크가 있거나 4에서 32행이면 `supports_sdpa_vector`도 거부하므로(배열 마스크를 받지 않고 `q_len`이 4 이상이어도 받지 않는다) `ScaledDotProductAttention::use_fallback`이 true가 되고 `mlx/fast.cpp`의 ops fallback이 돈다. 이 fallback은 배열 마스크, bool 마스크, GQA 반복, sink, `k_len - q_len` causal offset을 모두 지킨다. 2행이나 3행이고 배열 마스크가 없으면 `sdpa_vector`를 타는데, 그 causal 조건 `i <= kL - qL + q_seq_idx`는 cuDNN의 `set_causal_mask_bottom_right`와 같은 bottom-right 정렬이다. 1행 디코드 스텝은 게이트가 `q_len > 1`을 요구하므로 아예 닿지 않는다. 프런트엔드의 `force_fused=True` 경로는 cuDNN이 처리하던 자리에서 이제 예외를 던지지만, mlxcel은 그 플래그를 설정하지 않는다. backward 프리미티브는 `supports_sdpa_cudnn`을 보지 않고 자체 `use_fallback`이 `q_len % 128 == 0`을 요구하므로, 학습 shape은 게이트와 겹칠 수 없다.

측정: 모든 arm은 유휴 호스트에서 n = 3이고(시작을 load1 0.6 미만으로 게이팅했고, 런마다 load1을 기록했고, `ps`로 다른 소비자가 없는지 확인했다), 기준선과 수정 후가 같은 바이너리이며, `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0`이 바이너리 안의 킬 스위치다. 킬 스위치 arm이 모든 폭에서 수정 전 기준선을 2% 안에서 재현하므로, A/B가 재빌드가 아니라 게이트를 재는 것이 된다. 클래식 arm은 같은 세션에서 두 설정 모두 다시 측정했다. 전체 표와 nsys 귀속, 무코드 대조는 기록 문서에 있다.

게이트: `cargo fmt --all -- --check` 통과. Rust 쪽은 `hardware.rs`의 #818 테스트 옆에 `cuda_sdpa_cache_default_matches_build_feature`를 추가했고 `cargo test --profile test-fast --features cuda -p mlxcel-core hardware::`와 `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`로 검사한다. 이 호스트에서는 셀렉터 없는 `cargo test --lib`이 이 변경과 무관한 `cudaStreamEndCapture` C++ abort로 죽으므로 셀렉터를 좁힐 수밖에 없다. `metal,accelerate` 게이트는 이 Linux/CUDA 호스트에서 실행할 수 없어 돌리지 않았고, 변경이 거기 닿을 수도 없다. 패치한 파일은 CUDA 백엔드에만 컴파일되고 `cuda_sdpa_cache_default`는 `cuda` 피처가 없으면 `None`을 반환한다.
