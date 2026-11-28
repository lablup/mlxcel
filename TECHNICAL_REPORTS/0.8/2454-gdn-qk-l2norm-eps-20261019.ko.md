# PR #2454: normalize_qk로 GDN / KDA q/k에 기준 l2norm eps 적용

**작성일**: 2026-10-19
**상태**: Qwen3.5-4B와 Qwen3-Next-80B-A3B로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

#2361을 닫는다. gated-delta(GDN)와 Kimi Delta Attention(KDA)의 q/k 정규화는 `l2norm(x) = x / sqrt(sum(x^2) + 1e-6)`이다. mlxcel은 이를 fused `rms_norm`으로 계산하는데, `rms_norm`은 eps를 `mean(x^2)`에 더하므로 맞는 eps는 `1e-6 / head_k_dim`이다. 그런데 Qwen3-Next, Qwen 3.5, Kimi Linear는 고정값 `1e-6`을 넘겼다. 이 PR은 mlx-lm `gated_delta.normalize_qk`(ml-explore/mlx-lm#1896)를 따르는 공유 `normalize_qk`를 추가하고, 호출 지점 여섯 곳과 Kimi K3를 모두 이 함수로 보낸다.

## 문제 정의

`mean(x^2)`에 고정 `1e-6`을 더하면 실제 l2 eps는 `1e-6 * head_k_dim`이 되어, `head_dim = 128`에서 128배 커진다. 일반적인 행에서는 차이가 작다. 하지만 norm이 작은 행(값이 1e-5 수준)에서는 이전 수식이 실제 l2norm과 상대적으로 91퍼센트 어긋난다. Kimi K3는 private `qk_norm_eps`로 이미 올바른 값을 쓰고 있었으므로, 계열끼리도 upstream과도 서로 맞지 않았다.

## 기술적 결정

- **helper 하나, 구성 요소 셋.** `gated_delta.rs`에 `normalize_qk(q, k, inv_scale, eps)`, 상수 `QK_L2NORM_EPS = 1e-6`, 그리고 `normalize_qk_rms_eps`를 추가한다. `normalize_qk_rms_eps`는 mlx-lm이 Python float로 계산하는 것과 같이, f32 `inv_scale`에서 `eps * inv_scale^2`을 f64로 계산한다. `scaled_fast_rms_norm_no_weight`는 이제 `normalize_qk`만 뒷받침하므로 `pub(crate)`로 좁혔다.
- **모든 호출 지점 전환.** Qwen3-Next(한 곳), Qwen 3.5(main forward, MTP와 DFlash 검증이 거치는 speculative capture forward, 테스트 전용 `debug_prefill_no_cache`. VL과 MoE wrapper는 이를 상속), Kimi Linear KDA(한 곳, `delta_head_dim` scale), Kimi K3.
- **Kimi K3는 비트 단위로 동일하게 유지.** private `qk_norm_eps`, 저장된 `qk_eps` 필드, 오래된 "Kimi Linear is left as is" 주석을 제거했다. 제거한 경로를 그대로 재현해 공개된 `delta_head_dim` 128에서 비트 일치를 검사하는 테스트가 있다. 2의 거듭제곱이 아닌 head dim에서는 두 f32 eps 식이 1 ulp 다를 수 있으므로, 동일성은 128에 대해서만 주장한다.

## 변경 사항

- `src/models/gated_delta.rs`: `normalize_qk`, `QK_L2NORM_EPS`, `normalize_qk_rms_eps`, 가시성 축소.
- `src/models/qwen3_next.rs`, `src/models/qwen3_5.rs`, `src/models/kimi_linear.rs`, `src/models/kimi_k3.rs`: 호출 지점 전환.
- `src/models/gated_delta_normalize_qk_tests.rs`(신규), `src/models/kimi_k3_tests.rs`: 테스트.
- `CHANGELOG.md`: Unreleased / Fixed 항목.

## 검증

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`. 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1 --include-ignored`로 실행했다.

| 필터 | 결과 |
|---|---|
| normalize_qk | 4/4 |
| models::kimi_k3 | 26/26 |
| models::qwen3_next | 11/11 |
| models::qwen3_5 | 51/51 |
| models::kimi_linear | 10/10 |
| models::gated_delta | 19/21 (실패 2개는 main에도 있음) |

작은 norm 테스트는 `head_dim = 128`에서 값이 1e-5 수준인 q/k를 넣는다. `normalize_qk`는 손으로 계산한 f64 l2norm과의 상대 오차가 1.6e-7 이내이고, 같은 행에서 이전 고정 eps 경로는 0.91 어긋난다. gated_delta 실패 두 개(`gated_delta_step_output_shape_and_dtype`, `gated_delta_ops_single_token_output_shape`)는 main 빌드에서도 같은 출력 dtype assertion으로 실패한다.

`docs/architecture.md`에 대한 teacher-forced logit trace, decided gap 2. reverted arm은 이 브랜치에서 소스 파일 여섯 개를 main에서 체크아웃해 다시 빌드한 것이다.

| 체크포인트 | 폭 | Top-1 불일치 | Decided 불일치 | ppl reverted / fixed |
|---|---|---|---|---|
| Qwen3.5-4B-4bit | 1 | 2/200 | 0/79 | 19.213 / 19.199 |
| Qwen3.5-4B-4bit | 5 | 4/300 | 0/134 | 15.289 / 15.276 |
| Qwen3.5-4B-4bit | 256 | 18/1024 | 0/385 | 23.072 / 23.063 |
| Qwen3-Next-80B-A3B-4bit | 1 | 8/200 | 0/91 | 27.351 / 27.140 |
| Qwen3-Next-80B-A3B-4bit | 5 | 28/300 | 0/51 | 980.26 / 987.42 |
| Qwen3-Next-80B-A3B-4bit | 256 | 27/1024 | 0/409 | 24.109 / 24.059 |

null arm(Qwen3.5-4B에서 reverted 바이너리를 두 번 실행)은 세 폭 모두에서 바이트 단위로 같았다. 따라서 위의 변화는 eps 변경에서 온 것이다. decided 위치의 불일치는 어디에도 없고, perplexity는 6행 중 5행에서 내려갔다. greedy 128 토큰 `mlxcel generate --temp 0`은 두 체크포인트 모두 양쪽 arm에서 자연스럽다.

mlx-lm 기준 arm은 없다. 이 호스트의 oracle venv는 mlx CPU backend만 있고, `scripts/mlx_lm_logit_trace.py`는 70분 동안 Qwen3.5-4B의 forward 네 개 중 첫 번째도 끝내지 못했다. Kimi-Linear-48B는 이 호스트에 없어 trace가 없다. 그 변경은 같은 호출 교체이며 kimi_linear 단위 테스트로 검증된다.

## 후속 작업

- Qwen3-Next-80B는 512 토큰 context 뒤 폭 5에서 양쪽 arm 모두 ppl이 약 980이고 chunk 안의 모든 위치가 나쁘다(평균 NLL 6~9). 같은 context 뒤 폭 1은 27이다. speculative verify가 쓰는 형태인, cache 뒤 Qwen3-Next 다중 토큰 forward에 이미 있던 문제로 보인다. 이 PR이 만든 문제도 아니고 고치지도 않는다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 8 |
| 추가 줄 | 262 |
| 삭제 줄 | 52 |

| 커밋 | 메시지 |
|--------|---------|
| `11e9b5f2` | fix(gdn): use the reference l2norm eps for q/k via normalize_qk |
| `92421992` | docs: add changelog entry for the GDN q/k l2norm eps fix |

## 관련 항목

- ml-explore/mlx-lm#1896: upstream `normalize_qk`.
