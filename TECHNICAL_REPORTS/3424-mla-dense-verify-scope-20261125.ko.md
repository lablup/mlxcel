# PR #3424: 남은 MLA 계열과 dense 계열의 token-only verify 정확성

**날짜**: 2026-11-25
**상태**: 드래프트 PR (#3421을 닫음). M3 Ultra (512 GB)에서 `deepseek-v2-lite-chat-4bit-mlx`, `afm-4.5b`, `k2-horizon-0.9b`, `olmoe-1b-7b-0125-4bit`, `kimi-linear-48b-a3b-instruct-4bit`, `longcat-flash-lite` 체크포인트 두 개로 검증했고, 좁은 범위의 lib 테스트, clippy, 계약 검사를 통과했습니다

## 요약

PR #3420은 `MLXCEL_MLA_ROW_INVARIANT=1`에서 token-only verify 블록(prompt lookup, full-model drafter)을 DeepSeek V3 계열, Xing4.0, GLM-4.7-Flash 엔진 경로의 단일 토큰 decode와 바이트 단위로 같게 만들었습니다. 이 PR은 같은 opt-in 아래에 DeepSeek V3.2, DeepSeek V2, `Llama3Model`(K2 Horizon), Arcee, OLMoE를 더합니다. Kimi Linear, Kimi K3, LongCat은 token-only 라운드가 모델을 거부해서 verify 블록이 만들어지지 않으므로 범위 밖으로 두고, 그 근거를 기록합니다. 기본 경로는 바뀌지 않습니다.

## 1. 블록이 생기는 계열

`prompt_lookup_unsupported_reason`은 Kimi Linear와 Kimi K3(verify rewind가 없는 모델 소유 KDA, MLA 상태)와 LongCat 두 변형(모델 소유 슬롯 하나와 cache trim이 닿지 않는 n-gram 꼬리)을 거부합니다. 로컬 체크포인트 세 개의 모든 셀에서 survey가 "the token-only verify round does not take this model"을 보고했습니다. 이슈는 Kimi Linear의 recurrent KDA 레이어가 이미 chain과 같은지 물었는데, 이 레이어에는 token-only 블록이 오지 않으므로 질문이 성립하지 않고 블록 동작은 측정하지 않았습니다. 계열마다 거부를 고정하는 테스트를 하나씩 두었습니다. DeepSeek V2는 라운드를 받고 latent cache에 같은 single-query와 블록의 분기가 있으므로 범위에 넣었습니다.

## 2. 변경

| 계열 | scope 안에서 verify 블록이 하는 일 |
|---|---|
| DeepSeek V3.2 | 각 행이 저장된 `[latent, indexer key]` 창의 자기 prefix에 대해, 자기 indexer query와 top-k로 `single_query_attention`(`forward`에서 떼어 낸 decode 단계의 DSA attention)을 실행합니다. indexer, attention, MLP, gate, LM head 곱은 row-invariant이고 float expert는 위치별로 실행합니다 |
| DeepSeek V2 | 각 행이 latent cache(`MlaLatentCache::attend`)나 head별 K/V에 혼자 append하고 attend합니다 |
| `Llama3Model`, Arcee | `FusedQKVLinear`의 분할이 `row_invariant::linear_whole_block`을 거칩니다. scope 밖에서는 레이어 자체의 forward와 같습니다. scope 안에서는 fused rope launcher와 compiled SwiGLU MLP 대신 조합된 그래프를 모든 단계에서 쓰고, 블록은 `KVCache::attend`로 한 행씩 attend합니다 |
| OLMoE | 라우팅과 expert를 위치별로 실행해 각 위치가 fused 단일 토큰 MoE 커널을 탑니다. attention은 행별입니다 |

행별 MLA 블록은 `mla::stats::as_one_call`로 레이어마다 path stat을 한 번만 기록합니다. 이전에는 폭 K 블록이 single-query 호출을 K번 기록했습니다.

## 3. 결과

엔진 경로 survey, 폭 2, 4, 8, 프롬프트 8, 600, 2100 토큰(9셀):

| 체크포인트 | 꺼짐(두 커널) | 켜짐, wide | 켜짐, narrow |
|---|---|---|---|
| deepseek-v2-lite-chat-4bit-mlx | 다름, layer 0 출력 | 다름 | 같음 |
| afm-4.5b | 다름, layer 0 출력(한 셀은 layer 1) | 같음 | 불필요 |
| k2-horizon-0.9b | 다름, layer 0 출력 | 같음 | 불필요 |
| olmoe-1b-7b-0125-4bit | 다름, layer 0 출력 | 다름 | 같음 |

양자화 체크포인트에서 wide 커널에 남는 차이는 DeepSeek V3 계열과 마찬가지로 MLX의 `qmv_wide` 때문입니다.

`origin/main` 대비 decode ABAB, 변수 해제, 4 사이클, 토큰 동일: K2 Horizon 211.50 대 211.80 tok/s, OLMoE 410.21(cold 첫 실행 제외 시 421.24) 대 421.21, V2-Lite 173.28 대 174.58.

## 4. 테스트

- `deepseek_v32::verify_tests`, `deepseek_v2::verify_tests`: tiny 모델(dense 레이어와 float MoE 레이어. V3.2는 top-k 4와 64, V2는 두 cache 배치), 폭 2, 4, 8, 12, matvec 커널 켜고 끄기. scope 없이는 블록이 다르다는 테스트와 path count 테스트를 둡니다.
- `dense_verify_block_tests`: Llama 3, Arcee, OLMoE, float와 4-bit(4-bit 절반은 `qmv_wide`를 끈 채로 실행하고 drop에서 복원).
- `deepseek_v3::verify_block_tests`: path count가 블록과 레이어당 하나이고 decode count도 확인합니다.
- Kimi Linear와 LongCat 두 변형의 `the_token_only_verify_round_refuses_this_model`, mlxcel-core의 `as_one_call_records_the_first_path_once`.

## 5. 하지 않은 것

- `deepseek-v3.2-4bit`(352 GB)은 로드하지 않았습니다. V3.2는 tiny 모델로만 검증했습니다.
- 진단은 레이어 출력만 캡처하므로 첫 차이는 레이어 단위이고 그 안의 연산은 특정하지 않았습니다. 후보 연산은 한꺼번에 바꿨습니다.
- 이 계열들에서 변수를 켰을 때의 prompt lookup 처리량은 측정하지 않았습니다.
