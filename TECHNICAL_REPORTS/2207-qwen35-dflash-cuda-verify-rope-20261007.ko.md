# PR #2207: CUDA에서 Qwen 3.5 DFlash verify를 decode와 바이트 단위로 일치시키기

**날짜**: 2026-10-07
**상태**: CUDA(GB10)에서 구현 및 검증 완료, Metal은 구조상 변경 없음
**위험도**: 중간 (Qwen 3.5 계열의 CUDA DFlash verify 경로와 CUDA exactness probe 변경, CUDA 전체에 걸린 decline 제거)

## 요약

Qwen 3.5는 CUDA에서 DFlash를 한 번도 쓰지 못했습니다. PR #1944가 `MLXCEL_SDPA_VECTOR_LARGE_D`가 켜진 상태에서 head_dim 256/288이면 `probe_block_chain_exactness`가 `NotRun`을 반환하게 만들었기 때문입니다. 당시 `qwen3.5-4b-4bit` + `qwen3.5-4b-dflash`의 서빙 결과를 보면, width 2와 4의 greedy 출력이 classic decode와 어긋났는데 probe는 이를 잡아내지 못했습니다. Issue #2191은 그 원인이 PR #2185가 Gemma 4에서 찾아낸 MLX CUDA RoPE 커널 선택인지 검증했고, 실제로 그것이 원인이었습니다.

MLX의 CUDA RoPE는 row-contiguous인 `B=1, L=1` 입력에는 `rope_single`을, 그보다 넓은 입력에는 일반 `rope` 커널을 씁니다. 두 커널은 소스상 같은 식을 계산하지만, 드물게 bf16 원소 하나를 다르게 반올림합니다(op 단위 144개 케이스 중 1바이트). verify 블록에서 한 번에 회전시킨 K가 KV 캐시에 들어가므로, 이 차이는 이후 모든 토큰으로 이어집니다. 실제 200토큰 transcript에서 sub-op capture를 돌린 결과, 첫 차이 지점을 찾아낸 73개 행 가운데 69개에서 post-RoPE Q 또는 K가 가장 먼저 달라졌습니다. 나머지 4개 행에서는 attention 출력이 먼저 달라졌는데, 모두 K 차이가 이미 캐시에 들어간 뒤였습니다. 귀속 매트릭스(per-row RoPE off/on x `LARGE_D` 1/0, width 2와 4, n=3, logit 바이트 기준)에서는 블록 RoPE일 때 `LARGE_D` 설정과 상관없이 200행 중 67 또는 129행이 달랐고, per-row RoPE일 때는 두 설정 모두 0행이었습니다. 따라서 `LARGE_D=0`은 발산을 없애는 것이 아니라 위치만 옮깁니다. #1944의 귀속은 텍스트 sha256과 argmax 개수에 근거했기 때문에, 이 결과로 바로잡습니다.

## 설계 메모

- `Qwen3NextAttention::verify_rope_rows`는 생성 시점에 `cuda_is_available()` 값으로 한 번만 정합니다. `fast_rope` 분기에서 `B=1`이고 `L>1`인 verify 블록이면 Q와 K를 `offset + row` 위치에서 한 행씩 회전시킵니다(`fast_rope_rows_like_decode`). classic decode가 하는 호출과 똑같습니다. `B>1` 블록과 Metal은 기존처럼 블록 단위로 호출합니다. MLX의 `rope.cu`를 고치는 방안은 모든 CUDA 모델의 classic decode 수치를 바꾸게 되므로 채택하지 않았습니다.
- CUDA probe에 긴 draw를 추가했습니다. 같은 prefill에서 출발해 128개 토큰을 full-accept verify 블록과 단일 토큰 스텝 두 방식으로 진행하고, 행마다 logit 바이트를 비교합니다. 이번 변경 전의 probe는 서빙 경로가 발산하는 동안에도 프롬프트 길이 8부터 512까지 모든 길이에서 width 2와 4를 통과시켰습니다. 새 draw는 수정을 되돌리면 walk 위치 19에서 발산을 잡아내고, 수정을 적용하면 `Equal`을 반환합니다.
- 이제 CUDA probe의 모든 draw는 160토큰을 prefill합니다. probe가 거치는 KV 길이마다 새로운 CUDA graph topology가 캡처될 수 있고, 그때 생기는 캐시 miss는 MLX의 치명적인 lifetime miss 한도(#818)에 누적됩니다. `MLX_CUDA_GRAPH_CACHE_SIZE`를 이분 탐색해 보니, 이제 width 4 시작 과정은 200 엔트리 캐시 안에 들어갑니다. 8토큰 프롬프트에서 walk를 시작했을 때는 400 엔트리보다 많이 필요했습니다.
- `greedy_parity_dflash_qwen35_4b`는 CUDA에서 width 2와 4의 burst가 실제로 실행되어야 통과합니다. width 8과 16에서는 probe가 발산을 측정한 경우에만 decline을 허용합니다.

## GB10 검증 (CUDA, release, `--features cuda`)

드라이버 580.178.04, 커널 7.0.0-1019-nvidia, MLX pin `81ba1c6a`, `MLX_CUDA_ARCHITECTURES=121`.

- 서빙 `/v1/completions`, temperature 0, 환경 변수 오버라이드 없음, 200토큰 프롬프트 3개: width 2, 4, 그리고 미지정(4로 결정됨)에서 burst가 실행되었고 출력은 classic과 바이트 단위로 같았습니다. width 8과 16은 position 0의 `Diverges`로 decline했습니다.
- 전체 `speculative_parity --ignored --test-threads=1`: 7개 통과, 1개 실패. 실패한 테스트는 #2190이 다루는 기존 Gemma 4 문제 `b1_batched_baseline_probe`입니다. `greedy_parity_dflash_qwen35_4b`는 [2, 4]를 실행하고 [8, 16]을 decline했습니다.
- `long_probe_draw_sees_the_verify_rope_hazard_and_passes_with_the_fix`: 3회 실행 모두 통과했습니다.
- width 4 처리량, classic null arm을 포함해 3라운드 교차 실행: DFlash는 classic 대비 +6.7%에서 +10.7%였고, null arm의 편차 범위는 -0.6%에서 +3.7%였습니다. 모든 arm의 completion이 같았습니다. acceptance는 0.455, verify당 방출 토큰 수는 2.34였습니다.
- per-row RoPE의 verify 비용: width 2에서는 노이즈 범위 안, width 4에서는 +1.2%에서 +1.6%였습니다.
- `cargo clippy -p mlxcel --lib --tests --features cuda -- -D warnings`와 `cargo fmt --check` 모두 경고 없이 통과했습니다.

## 이 호스트에서 검증하지 못한 항목

Metal. #2158에 추가할 항목은 없습니다. per-row RoPE 플래그, 긴 draw, 160토큰 probe 프롬프트, parity 테스트의 burst 요구 조건은 모두 `cuda_is_available()`로 게이트되어 있고, 제거한 decline도 CUDA에만 적용되던 것입니다.

## 후속 과제

- 처리량은 약 1.08배입니다. 2026-09-21 기록에서 exactness를 보장하지 않는 width-4 burst는 1.18배였지만, 그때는 다른 trajectory를 따랐습니다(acceptance 0.511, 이번은 0.455). trajectory 차이와 per-row RoPE 비용이 각각 얼마나 기여하는지는 위의 verify 측정 이상으로 분리하지 않았습니다.
- width 8과 16은 이슈에서 정한 범위대로 계속 decline합니다(`M*B >= 8`에서 `qmm_sm80`).
- batched(`B>1`) DFlash verify는 블록 단위 RoPE를 그대로 씁니다. 이 경로의 exactness는 이 probe가 아니라 batched gate가 판단합니다.
