# 기술 보고서: PR #3175 - DSA 계열의 양자화 MLA latent, 측정 후 비활성 유지

**날짜**: 2026-11-20
**상태**: 완료
**위험도**: 중간 (int8 경로는 꺼진 채로 두지만, DeepSeek V3.2 로딩을 고치고 main의 DeepSeek-V3/R1 expert 라우팅을 바꾼다)

## 요약

DSA attention(`deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`)에 int8 latent 슬롯과 int8 indexer key 코덱을 추가하고 DeepSeek-V3.2-4bit로 측정했다. trace 게이트는 통과했다(모든 prefix에서 decided 불일치 0). 레이어별 indexer top-k overlap 게이트(모든 레이어에서 평균 0.99 이상)는 실패했다. 4096 토큰에서 61개 레이어 중 12개, 16384에서 6개만 통과한다. 그래서 세 타입은 `QUANTIZED_LATENT_FAMILIES` 밖에 두고, 양자화 요청은 계속 fp16으로 바뀐다. 측정을 위해 체크포인트 로딩과 라우팅을 막던 결함 세 개를 고쳤고, 그중 하나는 `DeepSeekV3Attention`에도 있었다. DeepSeek-V3와 R1(`n_group` 8)은 모든 토큰을 expert group이 없는 모델처럼 라우팅하고 있었다.

## 1. 문제

DSA 레이어는 토큰마다 두 저장소를 가진다. MLA latent와, query마다 attend할 top-k 위치를 고르는 indexer key다. 두 저장소가 함께 줄어야 양자화 latent가 메모리를 아끼고, indexer key를 양자화하면 선택되는 위치가 바뀔 수 있는데 이는 logit trace만으로는 보이지 않는다. #3163은 이식과 함께 선택 단위의 게이트를 요구했다.

## 2. 변경 내용

- `mla/latent_keys.rs`(신규): group별 scale을 쓰는 indexer key용 int8 코덱. `mla/cache.rs`가 두 plane을 latent 슬롯에 저장한다.
- `src/models/deepseek_v32.rs`, `src/models/deepseek_v32_latent.rs`: attention이 양자화 슬롯을 absorbed 경로로 읽는다. 다중 토큰 step은 한 번 dequantize한다.
- main에서 체크포인트를 돌리기 위한 수정: stacked `switch_mlp` expert 레이아웃, routed expert squeeze, rank-3 router score의 group 라우팅. stacked 로더는 expert 수가 맞지 않는 scales나 biases plane을 거부한다.
- `src/models/deepseek_v3.rs`: 같은 rank-3 group 수정. gate가 `group_mask_scores` 호출 전후로 score를 `[tokens, experts]`로 펼친다. 이전에는 rank-3 입력을 한 번 경고하고 건너뛰었다.
- `kv_arch`: DSA 크기 계산에 indexer key를 포함한다. fp16 토큰당 레이어당 1408바이트(이전 보고 1152), int8은 808.
- 도구: `examples/logit_trace`의 `MLXCEL_TRACE_KV_CACHE_MODE`와 `trace_topk` 덤프, `scripts/compare_dsa_topk.py`.
- 문서: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/environment-variables.md`, trace·top-k·decode CSV를 포함한 `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md`.

## 3. 기술적 결정

**top-k 게이트에 따라 계열을 끈 채로 둔다.** 에픽은 top-k overlap 게이트를 trace 게이트와 별도의 필수 조건으로 정했다. 선택이 바뀌어도 teacher-forced logit에서는 수백 토큰 동안 드러나지 않다가 긴 생성에서 누적될 수 있기 때문이다. 레이어 0의 overlap은 0.998이라 코덱 자체의 비용은 작고, 차이는 깊이에 따라 쌓인다. fp16에서 커널 선택(fused 대 composed)만으로도 61개 중 29개 레이어가 0.99에 못 미치므로, 이 기준이 이 계열의 자체 잡음보다 엄격할 수 있다. 기준 완화 여부는 열린 결정으로 남긴다. 코드는 allowlist에 타입을 추가하면 바로 켤 수 있다.

**V3 라우팅 수정을 별도 PR이 아니라 이 PR에 넣는다.** 리뷰어가 같은 V3.2 버그를 고치다가 V3 버그를 찾았다. 수정은 기존 호출 전후의 reshape이고, tiny-gate 테스트가 이를 고정한다(수정 전에는 {0, 1} 대신 {0, 3}을 고른다). `n_group`이 1인 체크포인트(Moonlight, Kimi-VL, Kimi K2)는 바뀌지 않는다. DeepSeek-V3와 R1의 출력은 참조 라우팅 쪽으로 바뀐다.

## 4. 측정

M3 Ultra, Metal, DeepSeek-V3.2-4bit, 기준은 fp16. 측정 빌드는 세 타입을 로컬에서만 목록에 넣었고, 커밋된 트리에는 들어 있지 않다.

Top-k overlap:

| context | 비교 | 0.99 이상 레이어 | 평균 | 최저 레이어 |
|---|---|---|---|---|
| 4096 | int8 대 fp16 | 12 / 61 | 0.9793 | 0.9662 |
| 4096 | fp16 fused 대 composed | 32 / 61 | 0.9911 | 0.9854 |
| 16384 | int8 대 fp16 | 6 / 61 | 0.9021 | 0.7438 |

Trace: 1024, 4096, 16384 뒤 모두 decided 불일치 0, 단일 시퀀스와 두 행 batched 모두. flip은 fp16 fused 대 composed 잡음 바닥과 비슷한 크기다. 이 모델의 fp16 logit 간격이 0.125라서 0이 아닌 gap은 모두 0.11 기준을 넘는다.

Decode, tok/s 중앙값 fp16 / int8: 512에서 20.76 / 19.75(-4.9%), 2048에서 17.54 / 15.44(-12.0%), 16384에서 15.68 / 13.63(-13.1%). int8 prefill은 13~20% 느리다.

## 5. 배운 점

**선택 단위 게이트는 logit trace가 놓치는 것을 잡는다.** 모든 prefix에서 trace는 통과했지만 레이어별 선택은 16384에서 overlap 0.74까지 벌어졌다. sparse attention에서 누적되는 것은 선택된 집합이므로 따로 측정해야 한다.

**한 계열을 측정하다 공유 코드의 결함이 드러날 수 있다.** 체크포인트가 main에서 로딩되지 않았고, 그 라우팅을 고치다가 dense V3 gate의 같은 버그를 찾았다. 그 버그는 한 번 찍히는 경고로만 보였다.

## 6. 후속 작업

- DeepSeek-V3.2의 fp16 기준이 4096 토큰 넘는 context에서 무너진다(4096 뒤 ppl 7.3, 8192 뒤 1403, 16384 뒤 2080). 이 PR 이전부터 있던 결함이고 아직 이슈가 없다. 16384 행은 이 무너진 기준 위에 있다.
- fp16 자체 커널 선택이 29개 레이어에서 0.99를 못 넘는 상황에서, DSA에 레이어별 0.99 overlap 기준을 유지할지 결정해야 한다.
- GLM-5는 측정하지 않았고(체크포인트가 이 호스트에 없음) V3.2의 결정을 따른다.
- 기준 커밋에서 체크포인트가 로딩되지 않아 변경 전 fp16 기준선이 없다. fp16 동일성은 tiny 모델 테스트와 리뷰에 근거한다.
- CUDA, ROCm, 이 체크포인트의 `mlxcel serve`와 prompt cache는 측정하지 않았다.
