# 기술 보고서: PR #3056 - update(models): 순수 causal hybrid prefill에 MLX causal mode 사용

**날짜**: 2026-11-15
**리뷰어**: -
**상태**: M3 Ultra, M5 Max 검증 완료
**언어**: Rust, Bash, Markdown
**위험도**: 중간

---

## 요약

Qwen 3.5, Qwen3-Next, GraniteMoeHybrid, Qwen-VL 계열의 full-attention 레이어는 모든 prefill에서 f32 causal array mask를 만들었습니다. M5 이전 하드웨어의 head_dim 256에서는 MLX가 이 mask로 SDPA를 fuse하지 못합니다. 그래서 single pass prefill은 #3049의 query 청크 fallback을 탔고, 28K 토큰에서는 3.65 GiB mask가 pass 내내 유지되었습니다(#3051).

이제 모델은 두 조건을 모두 만족할 때 causality를 MLX causal mode에 맡깁니다. causal mode가 같은 키를 가려야 하고, 호출 shape에서 MLX 자체 판정이 fuse된다고 답해야 합니다. 그 밖의 경우에는 공유 array mask를 유지합니다.

M3 Ultra에서 28K single pass의 피크는 30.13에서 19.82 GB(4B), 64.20에서 53.96 GB(27B)로 줄었고, 시간은 각각 21%, 15% 빨라졌습니다. 청크 CLI와 서버 prefill은 그대로이며 greedy 출력은 바이트 단위로 같습니다. M5 Max에서 main 빌드와 비교하면 single pass 피크는 14.53에서 8.76 GB(4B), 31.83에서 26.03 GB(27B)로 줄고 시간은 각각 21%, 23% 빨라집니다. 이후 2048 토큰 조각도 causal mode를 타므로 28K 청크 피크가 6.0%, 1.7% 내려갑니다. 여기서도 출력은 바이트 단위로 같습니다.

---

## 1. 문제 정의

### 1.1 배경

MLX의 SDPA에는 세 갈래가 있습니다. additive array mask, causal mode, mask 없음입니다. 각 갈래는 fused(메모리 선형) 커널을 쓰거나, `[B, H, L, S]` score 행렬을 만드는 fallback으로 갑니다.

#3033에서 확인한 내용은 이렇습니다. M5 이전 head_dim 256에서 array mask 갈래는 항상 fallback이고, causal mode는 `q_len == k_len`일 때 fuse됩니다. #3049는 그 fallback을 query 청크로 제한했지만, 여전히 더 느리고 mask도 필요합니다.

### 1.2 기존 문제

- hybrid와 Qwen-VL 계열은 multi-token forward마다 `create_causal_mask(seq_len, offset)`를 만들어 모든 full-attention 레이어에 넘겼습니다.
- 28K 토큰에서 이 배열은 f32로 3.65 GiB이며 single pass 내내 유지됩니다. 그리고 full-attention 레이어마다 청크 fallback을 탑니다(4B 8회, 27B 16회).

---

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core` `utils.rs` | `prefill_causal_mask(seq_len, offset, batch, PrefillAttention, activation_dtype)`와 `prefill_uses_causal_mode` 추가. 테스트용 순수 함수 `causal_mode_applies` 분리. kill switch는 `MLXCEL_PREFILL_CAUSAL_MODE`. |
| 모델 | Qwen 3.5(prefill 진입점 4개와 pipeline stage), Qwen3-Next(sequence forward와 stage), GraniteMoeHybrid, Qwen2-VL, Qwen3-VL, Qwen3-VL-MoE가 full-attention geometry와 hidden dtype을 전달. VL attention `forward`는 mask 없는 multi-token 호출을 `causal_attention`으로 보냄. |
| 벤치 하네스 | `ab_prefill.sh`에 arm별 `BASE_ENV`/`NEW_ENV`, `gated_calls` 열, `SERVER_MODELS=""` 지원 추가. |
| 문서 | 환경 변수 항목, CHANGELOG, CSV 4개(결과와 기각한 두 변형)를 담은 벤치마크 문서. |

---

## 3. 기술적 선택과 그 이유

### 3.1 causal mode가 동등한 조건

MLX causal mode에서 query `i`는 키 `0..=k_len - q_len + i`를 봅니다. `create_causal_mask(L, offset)`는 `[L, offset + L]`이므로, 이 mask로 실행된 forward는 모두 `k_len == offset + L`이었습니다. 따라서 어떤 offset에서도 두 방식은 같은 키를 허용합니다.

둘이 달라지는 경우는 모델 mask에 causality 이외의 정보가 있을 때뿐이며, 이 경우에는 array를 유지합니다.

- left padding이나 batch 행(`batch > 1`)
- 호출자가 넘긴 mask(padding, window, multimodal)
- mask가 필요 없는 single-row decode

Position id(MRoPE)는 회전만 바꾸고 참조하는 키는 바꾸지 않습니다.

### 3.2 speculative verify는 array 유지

일반 `Engine::verify`는 호출자 mask 없이 `[1, k+1]` 블록을 실행하므로 이 블록도 조건에 해당했습니다. 리뷰에서 verify 블록의 커널이 바뀌면 근소한 argmax 동률이 이전 verify 결과와 달라질 수 있다는 점이 지적되었습니다. 그래서 `DecodeLookaheadAppendScope`가 활성화되어 있지 않다는 조건을 추가했습니다.

### 3.3 선택 전에 MLX에 묻고, 아니면 공유 mask 유지

첫 버전은 동등한 곳이면 어디서나 causal mode를 썼습니다. 그런데 MLX가 causal mode에서도 fallback하는 조각이 있습니다(M5 이전, head_dim 256, `q_len != k_len`, 즉 첫 조각 이후 모든 조각). 이런 조각에서는 `causal_attention`이 자체 청크 causal fallback을 실행합니다. 이 fallback은 레이어마다 K/V를 자르고 query 청크마다 mask를 만듭니다. 그 결과 28K, 2048 청크 피크가 2.6%(4B), 4.0%(27B) 올랐고, 이는 서버 기본 경로입니다.

fallback이 array-mask 청커용 `create_causal_mask`를 직접 만드는 두 번째 버전은 더 나빴습니다. forward당 하나가 아니라 레이어마다 `[2048, 30K]` f32 mask를 하나씩 만들어 27B에서 +11.4%가 되었습니다.

최종 버전은 레이어 geometry와 hidden dtype으로 `sdpa_mode_is_fused`(MLX의 `use_fallback`, shape별 memoize)에 묻습니다. fused라고 판정한 곳에서만 causal mode를 쓰고, 그 밖에서는 공유 mask 하나를 유지합니다. 이렇게 하면 청크 피크가 base와 1 MB 이내로 같습니다.

Metal이 아닌 빌드에서는 probe가 fused 커널이 없다고 답하므로, CUDA와 CPU는 array mask를 유지하고 동작이 바뀌지 않습니다.

---

## 4. 검증

| 28K 토큰, M3 Ultra | qwen3.5-4b-4bit | qwen3.8-27b-4bit |
|---|---|---|
| CLI single pass | 30.13 → 19.82 GB, 16.89 → 13.38 s | 64.20 → 53.96 GB, 86.09 → 73.33 s |
| CLI 2048 청크 | 8.97 GB, 13.87 → 13.85 s | 25.80 GB, 74.77 → 74.74 s |
| 서버 `--parallel 1` | 9.10 GB, TTFT 13.90 → 13.89 s | 25.92 GB, TTFT 74.88 → 74.81 s |

- 두 arm 모두 같은 바이너리를 쓰며, `MLXCEL_PREFILL_CAUSAL_MODE=0`과 기본값을 하나의 lock 안에서 교차 실행했습니다.
- 반복 간 차이는 57 ms 이내입니다.
- 모든 셀과 이미지 프롬프트에서 출력이 같습니다.
- 4.6K, 2048 청크에서 gated call은 24에서 16(4B), 48에서 32(27B)로 줄었고 피크는 같습니다.

단위 테스트:

- offset 0, offset > 0, bf16 head_dim 256에서 causal mode를 array mask와 수치로 비교합니다.
- padding, batch, decode, verify, fuse되지 않는 shape 분기도 다룹니다.
- 모델 96개, scheduler 610개, prompt cache 328개 테스트가 Metal에서 통과합니다.

---

## 5. 남은 작업

- M5에서도 청크 prefill의 짧은 마지막 조각(4.6K에서 452, 494 행)은 fallback합니다. MLX가 그 shape에서 causal mode를 fuse하지 않는다고 보고하므로 main과 같이 공유 array mask를 유지합니다.
- M5 이전 하드웨어에서는 청크 prefill의 이후 조각이 여전히 MLX fallback을 탑니다. 그 경우에는 어떤 mask를 골라도 이를 피할 수 없습니다.
- 다른 hybrid 계열(`nemotron_h`, `falcon_h1`, `jamba`, `plamo2`, `lfm2`, `bailing_moe_linear`)과 tensor-parallel Qwen 3.5 runtime은 아직 array를 만듭니다. 각 레이어가 mask 없는 multi-token 호출을 `causal_attention`으로 보내는지 확인한 뒤 묶어서 후속 작업으로 다룰 수 있습니다.
