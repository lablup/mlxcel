# PR #3049: Metal에서 점수 행렬을 통째로 만드는 prefill SDPA 제한

**날짜**: 2026-11-14
**상태**: PR 열림 (#3033 해결, #3034 대체). M3 Ultra와 M5 Max에서 검증 완료

## 요약

MLX가 어떤 SDPA 호출에 fused attention 커널을 쓰지 못하면, 그래프 fallback이 query 축 전체에 대해 `q @ k^T`를 계산하고 `[B, heads, L, L]` 점수 행렬을 버퍼 하나에 담습니다. mlxcel은 이런 호출의 query 축을 나눠 그 버퍼를 `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` 안에 두는 장치를 이미 갖고 있었지만(#672), 이 게이트가 CUDA가 아닌 빌드에서는 곧바로 반환했습니다. 그래서 Metal에서 24 head, head_dim 256 모델에 26.6K 토큰 프롬프트를 넣으면 33.9 GB 버퍼 하나를 요청했고, M4 Pro의 Metal 버퍼 한도 30.15 GB를 넘어 요청이 실패했습니다(#3033).

이 PR은 Metal에서도 게이트를 실행합니다. 호출이 점수 행렬을 만드는지는 MLX 자신(`ScaledDotProductAttention::use_fallback`)에게, 그 호출이 실제로 쓰는 mask 모드로 묻습니다. 그래서 어느 GPU 세대에서도 fused 커널을 대체하지 않습니다. Metal의 기본 예산은 CUDA의 1024 MiB가 아닌 256 MiB인데, MLX가 청크 열 개 정도를 동시에 살려 두기 때문입니다. M3 Ultra에서 28K 단일 패스 prefill의 peak footprint는 66.5 GB에서 44.4 GB(qwen3.5-4b), 114.0 GB에서 78.5 GB(qwen3.8-27b)로 줄었고, greedy 출력은 동일하며 prefill 시간 차이는 1.2% 이내입니다. MLX가 fuse하는 Gemma 4는 그대로입니다. MLX가 Qwen의 array mask 호출을 fuse하는 M5 Max에서는 28K 셀 전부에서 게이트가 한 번도 동작하지 않았고, peak와 prefill이 기준선과 같았으며, MLX가 여전히 fallback으로 보내는 호출만 청크로 나뉘었습니다.

## 1. 변경 요약

| 영역 | 변경 |
|---|---|
| `layers.rs` 청크 게이트 | 실행 시점에 정하는 `ChunkGateBackend`(Cuda, Metal, Unchunked). 예산을 먼저 확인하고 백엔드 규칙을 적용합니다. CUDA의 cuDNN 미러는 그대로이고, Metal은 mask 모드별로 MLX에 묻고, softcap은 항상 점수 행렬을 만든다고 봅니다. 청크로 나눈 호출마다 debug 로그 한 줄을 남깁니다. |
| `layers.rs` 예산 | 기본값 CUDA 1024 MiB, Metal 256 MiB. `MLXCEL_ATTENTION_CHUNK_BUDGET_MB`가 둘 다 덮어씁니다. `MLXCEL_METAL_ATTENTION_CHUNK=0`이면 Metal 게이트를 끕니다. |
| `attention` / `causal_attention` | 같은 SDPA를 호출하는 M5(Metal 4) 경로보다 게이트가 먼저 실행됩니다. |
| `window_sdpa.rs` | `sdpa_mode_is_fused`가 기존 스레드별 fused 판정 메모에 causal과 mask 없음 경우를 더합니다. |
| 테스트 | `layers_metal_chunk_tests.rs`: 모드별 판정, kill switch 파싱, head_dim 256 fp16 청크 일치성, 그리고 ignore된 연산 단위 peak memory 측정. 기존 CUDA 게이트 테스트는 백엔드를 명시해서 넘깁니다. |
| 문서와 하네스 | 환경 변수 문서, CHANGELOG, CSV 원자료를 담은 벤치마크 문서, `scripts/bench/metal_sdpa_chunk/`(CLI와 서버 재실행 스크립트). |

## 2. 기술적 선택과 그 이유

### 2.1 표를 만드는 대신 호출 모드별로 MLX에 묻기

Metal에서 MLX의 fused 지원 여부는 head dim, query 길이, `q_len == k_len` 여부, mask 모드, dtype, GPU 세대(M5 이상의 NAX)에 따라 달라집니다. 이 M3 Ultra에서 head_dim 256, `q_len == k_len >= 2048`인 causal 모드 호출은 fused지만 같은 shape의 array mask 호출은 아닙니다. M5에서는 둘 다 fused입니다. Rust 쪽 표는 MLX pin을 올릴 때마다 어긋납니다. 브리지에 이미 `sdpa_uses_fused_kernel`로 `use_fallback`이 노출되어 있으므로, 게이트는 곧 실행할 호출의 플래그 그대로 이를 묻습니다. 거절된 #3034 설계는 모든 호출에 대해 array mask 경우를 물었고, 그러면 fused causal 커널이 청크 fallback으로 바뀌었을 것입니다.

probe는 예산 검사에서 호출이 예산을 넘는다고 나온 뒤에만 실행되므로 decode와 작은 prefill 호출은 비용을 내지 않으며, 판정은 shape별로 메모됩니다. probe가 오류를 내면 "fused 아님"으로 답해 청크로 나눕니다. 청크 계산은 행 단위로 정확하므로 잘못된 "아니오"의 비용은 속도뿐이고 메모리나 출력에는 영향이 없습니다.

### 2.2 `metal` feature가 아닌 실행 시점 백엔드

macOS에서 그냥 `cargo build`하면 `metal` Cargo feature 없이 Metal 백엔드가 빌드되므로(#1988), `cfg!(feature = "metal")` 게이트는 기본 빌드에서 동작하지 않습니다. 게이트는 `hardware::gpu_backend_kind()`를 씁니다. CUDA는 컴파일 시점 검사를 그대로 두어 동작이 바이트 단위로 같습니다.

### 2.3 Metal에서 더 낮은 예산

공용 1024 MiB 예산으로 처음 측정했을 때 `[1, 24, 28000, 256]` 호출 하나의 MLX 메모리는 43.5 GB에서 26.0 GB로만 줄었습니다. MLX는 command buffer를 최대 열 개까지 앞서 실행하고(`MAX_ACTIVE_TASKS`), 청크의 점수 버퍼는 해당 command buffer가 끝나야 allocator로 돌아갑니다. 그래서 살아 있는 일시 메모리는 청크 하나가 아니라 열 개 정도입니다. 256 MiB에서는 같은 호출의 peak가 9.8 GB이고 실행 시간은 청크 없이 2.08 s, 청크로 1.41 s입니다. 비용은 기본 2048 토큰 chunked prefill에서 나타납니다. 이후 청크들도 fallback이라 청크 수가 네 배가 되어 prefill 시간이 2~3.5% 늘었고, 1024 MiB에서는 차이가 없었습니다. 청크마다 동기 평가를 하면 일시 메모리를 청크 하나로 묶을 수 있지만 forward 안에서 청크마다 host sync가 생기므로 쓰지 않았습니다.

### 2.4 M5 경로보다 앞선 게이트

M5에서 `attention`은 게이트 전에 `metal4_attention`으로 반환했습니다. 이 경로도 같은 `fast::scaled_dot_product_attention`(또는 softcap 합성 연산)을 호출하고, M5에서도 MLX가 fallback하는 호출이 있습니다(array mask를 쓰는 head_dim 512, softcap). 게이트는 fallback일 때만 동작하므로 앞으로 옮겨도 fused M5 호출에는 변화가 없습니다.

## 3. 검증

M3 Ultra, 512 GB, release 빌드. 기준선은 같은 바이너리에 `MLXCEL_METAL_ATTENTION_CHUNK=0`을 준 것으로, main의 Metal 디스패치와 같습니다. Greedy, 48 토큰.

| 경우 | 이전 | 이후 | 출력 |
|---|---|---|---|
| qwen3.5-4b 28707 토큰, 단일 패스 (CLI / 서버) | 66.46 / 66.65 GB | 44.38 / 44.59 GB | 동일 |
| qwen3.8-27b 28749 토큰, 단일 패스 (CLI / 서버) | 114.05 / 114.21 GB | 78.48 / 78.65 GB | 동일 |
| gemma-4-12b 28710 토큰 (fused, 게이트 동작 0회) | 23.57 GB, 33.65 s | 23.56 GB, 33.56 s | 동일 |
| 연산 단위 `[1, 24, 28000, 256]` array mask | 43.48 GB, 2.08 s | 9.84 GB, 1.41 s | 차이 0 |
| 4.5K 토큰, 예산 16 MiB 강제 (qwen, gemma) | | 게이트 동작 24회, 88회 | 동일 |

발견: 이 빌드에서 Qwen 3.5 계열의 full-attention 레이어는 명시적인 causal array mask를 넘기므로(`Qwen35Model::forward_internal`), 단일 패스 prefill이 CLI와 서버 경로 모두에서 보고자가 겪은 unfused 호출입니다. #3034 리뷰에 적힌 기준 수치(63.8 GB / 18.7 s, 109.7 GB / 85.5 s)가 바로 그 unfused 기준선입니다. 단위 테스트는 `layers::`, `window_sdpa`, `causal`에서 143개 통과했고, clippy와 환경 변수, 벤치마크 데이터 검사도 통과합니다.

## 4. 미검증 및 남은 항목

- **M5 이상.** M5 Max에서 검증했습니다(벤치마크 문서 "M5+ results" 섹션). 28K 셀 전부에서 게이트 동작 0회와 peak 변화 없음, 모든 셀에서 출력 동일이었고, M5가 fuse하지 않는 호출(array mask를 쓰는 head_dim 512)에서도 256 MiB가 peak가 가장 낮고 가장 빠른 예산이었습니다(16.22 GB에서 8.23 GB, 585 ms에서 412 ms). 다만 256 MiB에서 이 호출의 출력은 단일 호출과 3.05e-5(fp16 1~2 ulp) 차이가 나는데, M3 Ultra에서는 0이었고 원인은 추적하지 않았습니다.
- **연속 실행 A/B 시간.** `verify.sh`와 `server_verify.sh`는 두 arm을 쉬지 않고 이어서 실행하므로, 발열로 성능이 떨어지는 M5 Max에서는 게이트가 한 번도 동작하지 않았는데도 두 번째 arm이 최대 14%(서버 TTFT, 27B CLI prefill은 9%) 느리게 측정됐습니다. 순서를 뒤집고 매 실행 전에 쉬게 한 재측정에서는 두 arm의 차이가 2.2% 이내였습니다.
- **보고자의 장비.** 33.9 GB 단일 버퍼는 사라졌지만(청크마다 점수는 256 MiB 이내), M4 Pro에서 직접 다시 돌리지는 않았습니다.
- **남은 peak.** 남은 44~78 GB 대부분은 별도의 전체 시퀀스 일시 메모리이며(logits로 추정), 다른 이슈에서 추적합니다.
- **Chunked prefill 비용.** 기본 2048 토큰 prefill 청크에서의 2~3.5% 비용은 낮은 예산 때문입니다. 문제가 되면 그 경우 예산을 올리거나 예산 초과 폭이 작을 때 청크를 생략할 수 있습니다.
- **서버 debug 로그.** 서버의 로그 필터는 게이트의 debug 로그를 보여 주지 않으므로 서버 결과는 peak memory로 판단했습니다.

## 5. 학습 포인트

- Metal에서 지연 평가로 나눈 계산이 붙잡는 메모리는 청크 크기만이 아니라 MLX가 앞서 실행하도록 허용하는 command buffer 수로 정해집니다. 청크 하나만 살아 있다고 가정하지 말고 peak를 측정해야 합니다.
- "이 호출이 fused인가"는 정확히 그 호출에 대해 물어야 합니다. 같은 shape라도 mask 모드에 따라 답이 다릅니다.
- 판단 전에 게이트를 건너뛰는 kill switch는 같은 바이너리에서 정확한 A/B 기준선 역할도 합니다.
