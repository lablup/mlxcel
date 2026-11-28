# PR #3419: Inkling-Small MTP를 모든 폭에서 정확하게 만들기, 그리고 여전히 빨라지지 않는 이유

**날짜**: 2026-11-25
**상태**: 드래프트 PR (#3280을 닫음). M3 Ultra (512 GB)에서 `inkling-small-mlx-4bit`와 네이티브 bf16 MTP 헤드로 검증했고, 좁은 범위의 lib 테스트, clippy, 계약 검사를 통과했습니다

## 요약

이슈 #3280은 Inkling-Small의 1세대 Inkling MTP에 두 가지를 요구했습니다. 하나는 verify 블록을 단일 토큰 디코드와 같게 만드는 것이고, 다른 하나는 MTP를 classic 디코드보다 빠르게 만들거나 그렇지 않다는 것을 문서로 남기는 것입니다. PR #3415는 bf16 0.6B 체크포인트에서 블록을 정확하게 만들었지만, 양자화된 Small 타깃과 그 MTP 헤드로는 측정하지 않았습니다.

Small에서는 프로브가 폭 2에서 8까지 통과했지만 기본 MTP 폭인 10에서 거절했습니다. 프로브가 보지 못하는 곳, 즉 부분 수락의 롤백에도 비정확성이 하나 더 있었습니다. 두 문제를 모두 고쳤고, 이제 드래프터를 켠 출력이 classic 디코드와 바이트 단위로 같습니다. 롤백의 replay를 없앤 덕에 폭 2는 약 1.5배 빨라졌습니다. 그래도 모든 폭에서 classic 디코드가 더 빠릅니다. 헤드의 수락률이 낮기 때문이며(라운드당 1.43에서 1.63 토큰), mlx-vlm이 측정한 값과 같습니다. MTP는 계속 opt-in입니다.

## 1. 폭 10: MLX SDPA는 쿼리가 8개를 넘으면 커널을 바꿉니다

#3415의 서베이 하네스(`survey_block_vs_chain`)를 `inkling-small-mlx-4bit`에서 실행한 결과입니다.

| 폭 | 이전 | 이후 |
|---|---|---|
| 2, 3, 4, 8 | 동일 | 동일 |
| 10 | 400116 logit 바이트 중 150040 바이트 차이. 짧은 draw에서는 layer 0의 row 3, 600 및 2100 토큰 뒤에서는 layer 5 | 동일 |
| 16 | 실행하지 않음 | 동일 |

#3415의 row-invariant 연산은 dense projection, 컨볼루션, expert를 다룹니다. expert가 이미 위치 하나씩 실행되기 때문에 양자화된 타깃에서도 유지됩니다. 폭 10에서 달라지는 것은 attention입니다. MLX의 fused SDPA는 쿼리가 8개 이하이면 per-query vector 커널을, 그보다 많으면 tiled 커널을 쓰고, 두 커널은 key 합산 순서가 다릅니다. row 0에서 2까지는 우연히 맞았고 row 3부터 달라졌습니다.

수정은 `InklingAttention::attend_row`에 있습니다. 8행보다 긴 verify 블록 안에서는 쿼리를 최대 8개씩 묶고, 각 묶음이 전체 key 집합에 attention한 뒤 출력을 이어 붙입니다. #3415의 sliding window 행 단위 경로가 적용되는 경우에는 그 경로가 먼저 실행됩니다. verify 블록 밖에서는 바뀌는 것이 없습니다.

## 2. 롤백: replay는 체인이 아니었습니다

K 토큰을 verify하고 드래프트 `a`개를 수락하면, adapter는 verify 전에 찍은 스냅샷을 복원하고 남길 `a + 1` 토큰을 한 번의 forward로 replay했습니다. 이 forward는 verify scope 밖에서 실행되므로, 2행 이상의 replay는 단일 토큰 step의 커널이 아니라 `conv1d`와 배치 expert 커널을 탔습니다. 프로브는 연속된 전체 블록만 체인과 비교하므로 이것을 보지 못했습니다. 폭 3에서 이전 prose 출력은 classic 디코드와 달라졌습니다.

이제 verify 블록은 각 short convolution의 패딩된 입력 창 `[kernel - 1 + K]`를 `InklingLayerCache::block_tails`에 기록합니다(verify 블록 안에서만 기록하고, 다른 forward는 이를 지웁니다). `keep_verified_prefix`는 각 컨볼루션 상태를 row `a + 1` 앞의 `kernel - 1`행으로 잘라내고 KV 꼬리를 `K - a - 1`만큼 trim합니다. sliding layer는 블록 뒤에 `K + W - 1`개의 key를 남기므로 trim 뒤에도 `a + W`개가 남고, 다음 step의 front trim이 체인과 같은 창을 만듭니다. row-invariant verify 블록은 체인과 같으므로, 남는 상태가 곧 체인의 상태입니다.

`forward_speculative_for_sequence`는 블록이 row-invariant scope 안에 있으면(row kernel 켜짐, 2에서 16행) 스냅샷을 찍지 않고, `verify_finalize`는 스냅샷이 없으면 이 in-place 경로를 탑니다. `MLXCEL_INKLING_ROW_INVARIANT=0`이나 더 넓은 블록은 여전히 스냅샷과 replay를 씁니다. `attention.rs`를 500행 아래로 유지하려고 `InklingShortConv`를 `inkling_layer/conv.rs`로 옮겼습니다.

## 3. 처리량

128 greedy 토큰, 프롬프트 두 개, ABAB 순서, 두 arm 모두 같은 바이너리입니다(이전 = 이전 코드 경로).

| 프롬프트 | 모드 | 이전 tok/s | 이후 tok/s | 평균 수락 길이 (이후) |
|---|---|---|---|---|
| prose | classic | | 45.18, 45.01 | |
| prose | 폭 2 | 25.57, 26.15 | 39.82, 39.53 | 1.427 |
| prose | 폭 3 | 23.30, 23.94 | 34.53, 34.61 | 1.608 |
| prose | 폭 10 | 10.35, 10.38 | 12.71, 12.73 | 1.608 |
| code | classic | | 44.41, 44.39 | |
| code | 폭 2 | 27.06, 26.98 | 41.54, 41.68 | 1.512 |
| code | 폭 3 | 24.41, 24.42 | 34.85, 34.99 | 1.628 |
| code | 폭 10 | 10.50, 10.62 | 12.71, 12.77 | 1.628 |

폭 2의 한 라운드는 22 ms인 디코드 step에 비해 약 36 ms가 들고 1.43에서 1.51 토큰을 내보내므로, classic의 0.88배(prose)와 0.94배(code)에 그칩니다. 수락되는 드래프트는 거의 모두 첫 번째 것입니다. 폭 3과 10은 드래프트를 최대 8개 더 만들면서도 라운드당 0.1에서 0.2 토큰만 더합니다.

## 4. 수락률

드래프터를 mlx-vlm 466069f5의 `speculative/drafters/inkling_mtp`와 비교했습니다. 수락된 prefix에 대한 block 0의 seed, 드래프트 step `k`에서 block `min(k, 7)` 사용, 수락 prefix를 block 0에 다시 넣기 전의 스냅샷 복원, 타깃의 pre-norm hidden, 그리고 타깃의 final norm(upstream은 분리한 드래프터에 `model.llm.norm.weight`를 복사합니다), muP 스케일, LM head, 어휘 trim이 모두 같습니다. seed 다음 드래프트를 block 1부터 시작하는(DeepSeek 방식의 깊이 인덱싱) 실험은 커밋하지 않았고, 수락 수는 128 토큰당 48 또는 49로 그대로였습니다. 낮은 수락률은 헤드의 특성이며, upstream도 라운드당 같은 1.59 토큰을 보고합니다.

## 5. 테스트

- `mtp_partial_accept_keeps_the_single_token_chain_state_on_metal`: 폭 3과 10의 모든 수락 수에 대해, verify와 finalize 뒤의 다음 두 step이 남길 토큰을 하나씩 디코드한 모델과 바이트 단위로 같아야 합니다. 이전 코드에서는 폭 3, 수락 1에서 실패합니다.
- `the_verify_block_matches_the_chain_on_metal`이 이제 폭 10과 16을 다룹니다. 이전 코드에서는 폭 10(layer 1, global)에서 실패합니다.
- `mtp_partial_accept_restores_then_replays_kv_and_conv_state_exactly`는 이제 replay 경로를 명시적으로 고정합니다(row-invariant 꺼짐).
- 무시(ignored) 상태의 실제 체크포인트 프로브 테스트에 폭 10을 추가했고, 0.6B와 4-bit Small 체크포인트 모두에서 통과합니다.

## 6. 하지 않은 것

- 어떤 폭에서도 속도 향상이 없으므로, 이슈의 "처리량 향상" 항목은 문서로 답했습니다. 남은 비용은 2행 verify(dense 연산은 한 번 읽고 expert는 두 번 읽음)와 라운드마다 두 번 도는 드래프터의 bf16 LM head입니다. 둘 다 수락률을 바꾸지는 않습니다.
- #3417의 adaptive verify width는 DFlash 라운드 루프의 정책입니다. Inkling MTP는 자체 adaptive controller가 있는 MTP 루프를 돌기 때문에 `exact_at_every_verify_width`를 연결하지 않았습니다.
- `trim_state`는 여전히 컨볼루션 상태를 비웁니다. 이 훅은 padded prefill에만 쓰이며 prompt lookup의 되감기가 아닙니다. 이 보고서를 쓸 당시 prompt lookup은 게이트에서 1세대 Inkling을 거부했고, 이후 이슈 #3422가 같은 창을 쓰는 `rewind_decode_appends`로 이를 활성화했습니다.
