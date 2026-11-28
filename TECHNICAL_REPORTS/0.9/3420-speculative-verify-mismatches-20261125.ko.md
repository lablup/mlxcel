# PR #3420: 남아 있던 speculative verify 불일치 두 건 추적

**날짜**: 2026-11-25
**상태**: 드래프트 PR (#3418을 닫음). M3 Ultra (512 GB)에서 `gemma-4-26b-a4b-it-4bit`, gigachat3 bf16과 8-bit, Kimi-VL, moonlight, Xing4.0, GLM-4.7-Flash로 검증했고, 좁은 범위의 lib 테스트, clippy, 계약 검사를 통과했습니다

## 요약

PR #3415의 exactness 조사는 K행 speculative verify 블록이 그 블록이 대신하는 K번의 단일 토큰 디코드와 같지 않은 곳을 두 군데 남겼습니다. 하나는 MTP 프로브가 두 양자화 커널 모두에서 거절한 Gemma 4 26B A4B MoE 체크포인트이고, 다른 하나는 엔진의 token-only verify 라운드(prompt lookup, full-model drafter)로, 이 경로를 타는 모든 모델에서 달랐습니다.

둘 다 원인을 찾았습니다. Gemma 4 26B A4B는 고쳤고, 기본 설정에서 narrow 커널로 프로브를 통과합니다. token-only 라운드는 DeepSeek V3 계열, Xing4.0, GLM-4.7-Flash 엔진 경로에 정확한 블록을 넣었지만 opt-in입니다(`MLXCEL_MLA_ROW_INVARIANT=1`). 바이트 일치를 얻는 대가로 prompt lookup 처리량이 24%에서 43% 줄고, 이 라운드의 계약은 원래 바이트 일치를 약속하지 않기 때문입니다. Arcee, K2 Horizon, OLMoE는 갈라지는 위치만 기록하고 그대로 두었습니다.

## 1. Gemma 4 26B A4B

MoE feed-forward 안에 캡처 지점을 추가하니 narrow 커널의 첫 불일치는 레이어 0의 `MoE experts`, 0번 행이었습니다. 공유 MLP와 라우터의 두 출력은 일치합니다. `Experts::forward`는 토큰 수에 따라 커널을 고릅니다. 토큰 하나면 fused GeGLU 커널, 라우팅 쌍이 64개 미만이면 컴파일된 `gather_qmm` 그래프, 그 이상이면 정렬 gather입니다.

`Experts::forward_verify_block`은 MTP verify 블록 안에서 expert를 위치 하나씩 실행합니다. K 토큰 블록은 어차피 라우팅된 expert를 K벌 읽으므로, 늘어나는 것은 커널 호출 횟수이고 가중치를 더 읽지는 않습니다.

이것만으로는 부족했습니다. expert를 고치자 불일치가 배치 블록의 attention 출력으로 옮겨 갔습니다(draw에 따라 레이어 1에서 11). 31B와 12B 기하에서는 이미 행 단위 verify attention으로 이 문제를 바로잡고 있으므로, 26B A4B 기하(헤드 16, KV 헤드 8, 글로벌 KV 헤드 2)를 Metal에서 그 집합에 넣었습니다. CUDA에서는 측정하지 않았으므로 이 기하에 대한 CUDA 정책은 그대로입니다.

| 커널 | 이전 | expert만 | expert와 행 단위 attention |
|---|---|---|---|
| wide | 다름, 레이어 0 attention | 다름 | 다름 (`qmv_wide`, 양자화된 Gemma 4 전부와 동일) |
| narrow | 다름, 레이어 0 `MoE experts` | 다름, attention 출력 | 폭 2, 4, 8과 startup, 600, 2100 토큰에서 일치 |

classic 디코드는 verify 경로에 들어가지 않습니다. 변경 후 132.65 tok/s, 변경 전 132.90 tok/s입니다(ABAB 8쌍).

26B A4B용 드래프터가 디스크에 없어서 이 타깃의 MTP 처리량은 측정하지 못했습니다.

## 2. token-only verify 라운드

DeepSeek V3 계열에서 쿼리 하나는 fused absorbed MLA 커널을 타고, 블록은 rope-folded SDPA를 탑니다. 블록이 absorbed인지 materialized인지는 `use_absorbed_mla` 비용 모델이 정합니다. bf16 체크포인트에서는 dense projection(gemv 대 gemm)과 float expert(`gather_mm`)도 다릅니다. 양자화 체크포인트에서는 MLX의 `qmv_wide`가 다릅니다.

`MLXCEL_MLA_ROW_INVARIANT=1`이면 `DeepSeekV3Model`과 `Xing40Model`이 `row_invariant` 스코프에 들어갑니다. `DeepSeekV3Attention::forward`에는 이제 `single_query_attention` 헬퍼 하나가 있고, 디코드 스텝이 이것을 쓰며, verify 블록 안에서는 각 행이 자기 디코드 스텝이 볼 latent의 앞부분만 가지고 이것을 씁니다. dense projection은 row-invariant matvec 커널을 타고, float expert는 위치별로 실행합니다. 양자화 projection은 블록 전체를 한 번에 곱합니다(`row_invariant::linear_whole_block`). 행별로 나누면 가중치를 행마다 한 번씩 읽게 되고, `qmv_wide`를 끄면 이미 row-invariant이기 때문입니다. GLM-4.7-Flash는 블록을 MTP 게이트가 이미 검사하는 행 단위 verify forward로 보냅니다.

| 체크포인트 | 변수 켬, wide | 변수 켬, narrow |
|---|---|---|
| gigachat3 bf16 | 일치 | 필요 없음 |
| gigachat3 8-bit, Kimi-VL 4-bit, moonlight 4-bit | 다름 | 일치 |
| xing4.0 8-bit과 optiq 4-bit, GLM-4.7-Flash 엔진 경로 | 다름 | 일치 |

Xing4.0은 hyper connection도 손봐야 했습니다. 공유 레이어만 스코프에 넣었을 때는 optiq의 한 칸(폭 8, 2100 토큰)이 레이어 38에서 여전히 달랐습니다. mixer는 dense `matmul`이고 `hc_expand`는 위치 축에 대한 배치 matmul인데, diff 리뷰에서 이 둘이 지목됐고, 이제 스코프 안에서는 둘 다 row-invariant입니다.

모든 칸은 폭 2, 4, 8과 프롬프트 8, 600, 2100 토큰의 조합입니다. 변수를 끄면 모든 칸이 다릅니다.

end-to-end로 narrow 커널에서 `--prompt-lookup` 텍스트를 plain 디코드와 비교하면, 변수를 켰을 때 체크포인트와 프롬프트 조합 6개가 모두 같고, 껐을 때는 6개 중 4개만 같습니다(gigachat3 8-bit과 moonlight가 2.5K 토큰 프롬프트에서 plain 디코드와 갈라졌습니다).

## 3. 왜 opt-in인가

프롬프트를 그대로 옮겨 쓰는 응답에서의 prompt lookup, ABAB 4회:

| 체크포인트 | plain 디코드 | lookup, 끔 | lookup, 켬 |
|---|---|---|---|
| gigachat3 bf16 | 114.79 | 261.70 | 148.49 |
| gigachat3 8-bit | 144.07 | 343.77 | 260.84 |
| gigachat3 8-bit, `MLXCEL_QMV_WIDE=0` | 143.91 | 320.31 | 246.96 |

plain 디코드는 느려지지 않지만(bf16은 커널 사용 시 116.58 대 114.02 tok/s, 8-bit은 149.47 대 148.22), verify 라운드는 느려집니다. 레이어마다 SDPA 한 번 대신 fused 커널을 K번 호출하고, bf16에서는 8행 matvec 곱과 위치별 expert가 더해집니다. 정확한 블록을 기본으로 켜면 계약에 없는 성질을 위해 prompt lookup 이득의 3분의 1에서 절반을 내주게 되고, 양자화 체크포인트는 기본 커널에서 여전히 바이트 일치가 아닙니다. 이슈는 "비용이 받아들일 만하면" 정확하게 만들라고 했습니다. 이 비용을 기록했고 기본값은 바꾸지 않았습니다.

디코드에는 MLX `matmul`을 그대로 쓰고 블록의 dense 곱만 행별로 계산하는 변형은 prompt lookup이 plain 디코드보다 느려서(113.6 tok/s) 버렸습니다.

## 4. 테스트

- `gemma4::moe_verify_tests`: 양자화된 expert plane, 폭 2, 4, 8, 12. verify 블록의 expert 출력이 단일 토큰 디코드와 바이트 단위로 같고, 블록 전체 곱(변경 전 동작)은 같지 않습니다.
- `deepseek_v3::verify_block_tests`: 작은 모델, 폭 2, 4, 8, 12, matvec 커널을 켠 경우와 끈 경우, 그리고 fused MLA 커널을 타는 rope 폭과 composed 형태를 타는 rope 폭. 블록이 체인과 같고, 스코프가 없으면 같지 않습니다. 두 번째 테스트는 MLA 경로 횟수를 세어 블록의 모든 행이 단일 쿼리 경로를 타는지 확인합니다.
- `a_token_only_verify_block_takes_the_row_wise_verify_forward` (GLM): append 스코프 안의 일반 forward는 모드가 켜져 있으면 행 단위 verify forward이고, 아니면 classic forward입니다.
- `linear_singleton_geometry_and_long_probe_are_limited_to_measured_backends`가 이제 26B A4B 기하를 Metal에 한정합니다.

## 5. 하지 않은 것

- Gemma 4 26B A4B의 MTP 처리량(드래프터가 디스크에 없음).
- DeepSeek V3.2, Kimi Linear, Kimi K3, LongCat은 각자의 attention 코드에 같은 MLA 분기가 있고, 바꾸지 않았습니다.
- fused MLA 커널을 다중 쿼리로 만드는 것이 정확한 블록의 8-bit 비용을 줄일 가능성이 높은 방법입니다. 여기서는 시도하지 않았고 효과도 측정하지 않았습니다.
