# 기술 리포트: PR #3275 - Aleph Alpha Kolibri-1 (kolibri1) 하이브리드 sliding/full MoE 디코더

**작성일**: 2026-11-23
**리뷰어**: -
**상태**: PR 열림 (closes #2357), M3 Ultra에서 vendored mlx-vlm kolibri1 모듈과 대조 검증
**언어**: Rust, Python (oracle trace 스크립트)
**위험도**: Low~Medium (새 모델 패밀리이며, 공유 AFMoE attention 블록, prefill 마스크, expert 스태킹 헬퍼를 리팩터링했으나 AFMoE 출력은 동일하고 AFMoE 테스트 32개 통과)

---

## 요약

이 PR은 Kolibri-1(`model_type: kolibri1`)을 추가합니다. 50개 레이어가 sliding-window 레이어 4개와 full-attention 레이어 1개를 번갈아 배치하고, 모든 레이어에 384 expert top-6 MoE와 공유 expert 1개를 둡니다. 구조상 attention output gate가 없는 AFMoE이므로 AFMoE의 attention 블록, 공유 MLP, 가중치 검증기, 하이브리드 prefill 마스크를 재사용합니다. `mlxcel generate`, `mlxcel-server`, `mlxcel arch`에서 동작하고, 네이티브 배치 디코드에 참여하며, prefill에서는 샘플링 행만 투영하고, 프롬프트 캐시 재사용을 위해 모델 상태 스냅샷을 제공합니다. `models/kolibri-1-mlx-4bit`(디스크 45 GB, 상주 41.9 GB, M3 Ultra에서 greedy 약 80 tok/s)에서 teacher-forced 1024 토큰 prefill은 mlx-vlm과 top-1이 1024개 중 1024개 일치하고, 1024 토큰 prefix 뒤(모든 sliding ring이 한 바퀴 돈 상태)의 디코드는 판정 가능한 388개 위치 중 1개만 불일치합니다.

---

## 1. 문제 정의

### 1.1 배경

Kolibri-1은 `nativ-community/Kolibri-1-MLX-4bit`(affine 4-bit, group size 64)로 배포됩니다. `layer_types`는 `sliding_attention` 4개와 `full_attention` 1개를 10번 반복합니다. sliding 레이어는 `RotatingKVCache(max_size = 513, keep = 0)` 안에서 RoPE로 회전하고, full 레이어는 NoPE이며 일반 `KVCache`를 씁니다. head별 Q/K RMSNorm이 RoPE 앞에서 실행되고, 레이어마다 norm 4개가 샌드위치를 이룹니다. mlxcel에는 `kolibri1` 감지 분기가 없었습니다.

### 1.2 기존 문제

- **라우터 의미가 DeepSeek-V3와 다름**: 라우터는 sigmoid에 bias를 더한 값이 아니라 raw logit에 `e_score_correction_bias`를 더한 값으로 선택합니다. 규칙이 틀리면 다른 expert 집합이 선택되는데도 문장은 매끄럽게 생성되므로, 눈으로 보는 품질 확인으로는 잡히지 않습니다.
- **스택되지 않은 expert**: MLX 변환본은 텐서 174,253개를 담고 있으며, 레이어마다 projection당 384개의 expert 플레인이 따로 있고 bias 이름은 `moe.router.expert_bias`입니다.
- **새 패밀리에 대한 CI 규칙**: #3090 이후 새 `impl LanguageModel`은 `last_row_logits`를 구현하거나 목록에 올라가야 하고, 배치 디코드에 참여하거나 `batched_decode_coverage_tests.rs`에 고정되어야 합니다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 라우터가 sigmoid + bias로 선택 | High: 매 토큰마다 잘못된 expert, 매끄럽지만 품질 저하 | DeepSeek-V3를 베낀 포팅이면 Medium. 라우터 fixture와 trace가 방어 |
| full 레이어에 RoPE 적용 | High: 긴 컨텍스트에서 조용한 품질 저하 | Medium. `--rope-on-full` trace arm이 게이트에서 실패 |
| 짧거나 잘못된 expert 스택이 `gather_qmm`에 도달 (양수 인덱스를 범위 검사하지 않음) | High: GPU 범위 밖 읽기 | Low. 로드 시 거부 |
| AFMoE 리팩터링이 AFMoE 출력을 바꿈 | High: 기존 패밀리 회귀 | Low. AFMoE 테스트 변경 없이 통과 |

---

## 2. 기술적 검토 사항

### 2.1 보안

구현 리뷰와 보안 리뷰는 `config.json`과 텐서를 신뢰할 수 없는 입력으로 취급했습니다. CRITICAL 또는 HIGH 이슈는 없었습니다.

| 이슈 | 심각도 | 상태 |
|------|--------|------|
| `num_experts`보다 플레인이 적은 스택 expert 텐서 | Medium | 로드 시 AFMoE의 `validate_stacked_experts`가 거부 |
| 길이가 틀리거나 두 attention 종류 외의 값이 든 `layer_types` | Medium | config 검증에서 거부 |
| 양자화된 라우터, 기본이 아닌 `rope_scaling`, `attention_bias` | Medium | 조용히 잘못 실행하지 않고 거부 |
| config와 shape이 다른 텐서 | Medium | sanitize 뒤, 어떤 텐서도 MLX에 닿기 전에 `validate_weights` 실행 |
| `drain_numbered_expert_planes`가 shape이 다른 플레인을 받으면 `mlx::core::stack`이 C++ 브리지 너머로 예외를 던져 프로세스가 abort됨 | Medium | 미해결. 헬퍼가 다른 패밀리와 공유되므로 공유 헬퍼 수정은 후속 작업으로 남김 |

### 2.2 성능

| 항목 | 값 |
|------|----|
| `kolibri-1-mlx-4bit` greedy 디코드, M3 Ultra | 77~83 tok/s |
| 로드 후 상주 메모리 | 41.9 GB |
| 로드 시간 | 10~12 s |

스태킹은 `sanitize_weights` 안에서 레이어 하나씩 즉시 실체화합니다. 그 레이어의 소스 플레인 384개만 살아 있는 시점이기 때문입니다. 미루면 첫 forward에서 모든 레이어를 복사하여 최대 메모리가 늘어납니다.

### 2.3 호환성 및 의존성

- **호환성 깨짐**: 없음. AFMoE attention 블록에 선택적 gate와 명시적 geometry가 추가되었고, 하이브리드 prefill 마스크가 공유 `hybrid_prefill_masks`로 옮겨졌지만 AFMoE 출력은 동일합니다.
- **새 의존성**: 없음.
- **테스트하지 않음**: MXFP4 변환본(이 호스트에 없음)과 EAGLE3 드래프터(범위 밖).

### 2.4 코드 품질

- **테스트**: `src/models/kolibri1_tests.rs`에 19개, `batched_decode_coverage_tests.rs`의 고정 항목, 리팩터링 후 재실행한 `models::afmoe`(32개).
- **구조**: `kolibri1.rs`(config, attention 레이어, 모델), `kolibri1_moe.rs`(라우터와 routed MoE), `kolibri1_weights.rs`(sanitize와 validate), `kolibri1_lm.rs`(`LanguageModel` 구현), `kolibri1_tests.rs`.
- **기술 부채(미해결 MEDIUM)**: `kolibri1_lm.rs`의 `LanguageModel` 구현이 AFMoE 것과 많이 중복됩니다. 공유 구현으로 합치면 중복이 사라지지만, AFMoE의 trait 표면을 바꾸는 것은 새 패밀리 PR의 범위 밖이었습니다.

---

## 3. 기술적 선택과 그 이유

### 3.1 독립 블록 대신 AFMoE 재사용

**배경:** Kolibri-1은 AFMoE와 attention output gate가 없다는 점, 그리고 라우터 규칙에서만 다릅니다.

| 선택지 | 장점 | 단점 |
|--------|------|------|
| AFMoE의 attention, MLP, 검증기를 복사 | AFMoE 변경 없음 | 같은 코드를 두 벌 유지해야 함 |
| **채택: AFMoE attention 블록에 선택적 gate와 명시적 geometry를 추가하고 prefill 마스크를 `hybrid_prefill_masks`로 분리** | 구현 하나, 검증기와 마스크 공유 | 이미 출시된 패밀리를 건드리므로 AFMoE 테스트 32개와 출력이 그대로여야 함 |

라우터와 캐시 배선은 공유하지 않습니다. 라우터 규칙이 다르고, 모델 소유 상태는 gemma3의 `Cache` enum(sliding 레이어는 `Rotating`, full 레이어는 `KVCache`)을 재사용합니다.

### 3.2 라우터 의미

라우터 `mlp.gate`는 양자화되지 않고 f32로 실행됩니다. logit을 계산하고, `logits + e_score_correction_bias`의 top 6를 선택한 뒤, 6개를 선택값 내림차순으로 재정렬하고, 가중치는 bias 없는 `sigmoid(logits)`에서 가져옵니다. 가중치는 `norm_topk_prob`일 때만 재정규화되며(릴리스에서는 false), routed scaling factor는 없습니다. `head_dtype`이 `float32`이면 logit을 f32로 캐스팅합니다. 라우터 fixture는 sigmoid + bias와 raw logit + bias가 서로 다른 집합을 고르는 입력을 써서, DeepSeek-V3 규칙이 실패하도록 만들었습니다.

### 3.3 full 레이어는 NoPE

full-attention 레이어는 RoPE를 적용하지 않고 sliding 레이어만 회전합니다. trace의 `--rope-on-full` arm이 둘을 구분함을 보여줍니다(4.3절).

### 3.4 로드 시 expert 스태킹과 범위 검사

`sanitize_weights`는 bias 이름을 바꾸고, 미리 스택된 `mlp.experts.{proj}.*`와 `mlp.switch_mlp.*` 내보내기를 받아들이며(멱등), 그 외에는 번호가 붙은 플레인을 공유 `drain_numbered_expert_planes`로 모아 스택합니다. 이 헬퍼는 모든 키를 이름으로 조회하므로 174k개짜리 맵을 훑지 않습니다. `num_experts`보다 플레인이 적은 스택은 이후 `validate_stacked_experts`가 거부합니다. `gather_qmm` 뒤의 gather가 양수 인덱스를 범위 검사하지 않기 때문입니다.

---

## 4. 구현 상세

### 4.1 레이어별 forward

```
h   = x + post_attn_norm(attn(input_layernorm(x)))
out = h + post_ffn_norm(moe(post_attention_layernorm(h)))
moe(z) = shared_mlp(z) + sum_{e in top6} w_e * expert_e(z)
```

sliding 레이어: head별 Q/K RMSNorm, RoPE, `RotatingKVCache(513)` 위의 attention. full 레이어: 같은 norm, RoPE 없음, 일반 `KVCache`. prefill은 `hybrid_prefill_masks`로 sliding 레이어에는 윈도 마스크를, full 레이어에는 causal 마스크를 만듭니다.

### 4.2 배선

`ModelType::Kolibri1`, `"kolibri1"` 감지 분기, `LoadedModel`, `model_metadata`, 레지스트리, 메모리 추정, tensor-parallel 모델 목록, `snapshot_sizing.rs`의 스냅샷 크기 패밀리 목록, `kv_snapshot.rs`의 `Used by:` 주석, `batched_decode_coverage_tests.rs`(`EXPECTED_NATIVE`), `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/turbo-kv-cache.md`, `docs/environment-variables.md`(`MLXCEL_FUSED_MOE` 목록과 attention-cache 스냅샷 패밀리), `CHANGELOG.md`. `MLXCEL_FUSED_MOE`가 켜져 있으면 fused 단일 토큰 디코드 MoE 커널이 routed expert에 적용됩니다.

### 4.3 실제 체크포인트 검증 (`models/kolibri-1-mlx-4bit`, M3 Ultra)

`mlxcel generate -p "Hello" -n 50 --temp 0`은 83 tok/s로 일관된 추론 텍스트를 냅니다. 792 토큰 프롬프트는 77 tok/s로 `docs/architecture.md`의 정확한 개요를 만듭니다. `mlxcel-server`의 `/v1/chat/completions`에 같은 프롬프트를 보내면 CLI와 같은 텍스트가 나옵니다.

vendored mlx-vlm 모듈 대비 teacher-forced trace(`scripts/kolibri1_oracle_trace.py`, 코퍼스 `docs/architecture.md`, `--decided 2`, mlxcel은 `MLXCEL_FUSED_MOE=0`):

| 기준 | 대상 | top-1 불일치 | 판정 가능 | Perplexity |
|---|---|---|---|---|
| mlx-vlm, 1024 토큰 forward 1회 | mlxcel prefill | 0 / 1024 | 0 / 383 | 31.24 vs 31.24, 최대 delta 0 |
| 동일 | mlx-vlm `--rope-on-full` | 601 / 1024 | 155 / 383 (40.5%), 실패 | 116.28 |
| `256 8 8 1024` 계획의 mlx-vlm chunk 4~7 | 1024 토큰 prefix 뒤 mlxcel width-one 디코드, ring 한 바퀴 돔 | 57 / 1024 | 1 / 388 (0.26%) | 18.10 vs 18.08 |
| 동일 | mlx-vlm `--rope-on-full` | 685 / 1024 | 220 / 388 (56.7%), 실패 | 155.36 |

trace의 id 없는 `PREFILL` arm은 쓰지 않았습니다. 이 패밀리도 AFMoE처럼 chunked prefill에서 빠지므로, id 없는 multi-token forward는 새 프롬프트로 시작되어 추적 chunk가 컨텍스트를 잃습니다.

---

## 5. 학습 포인트

### 5.1 매끄러운 모델이 올바른 모델은 아니다

sigmoid + bias로 expert를 선택해도 문장은 매끄럽게 나옵니다. 두 규칙이 갈라지도록 만든 fixture와 기준 모델에 대한 teacher-forced 비교만이 둘을 구분합니다. 틀린 규칙이 다른 답을 내는 fixture를 만드십시오.

### 5.2 prefill 일치가 디코드 일치를 보장하지 않는다

prefill arm은 sliding ring이 한 바퀴 돌지 않습니다. 1024 토큰 prefix 뒤의 디코드는 돌며, ring 오프셋 오류나 full 레이어의 잘못된 RoPE가 드러나는 곳입니다. 둘 다 실행하십시오.

### 5.3 음성 arm이 게이트를 보정한다

`--rope-on-full`은 판정 가능 위치 불일치가 40.5%, 56.7%로 실패하고, 올바른 포팅은 0%, 0.26%입니다. 이 차이가 통과 수치에 의미를 줍니다.

### 5.4 브리지를 넘는 네이티브 예외는 abort다

`mlx::core::stack`은 shape이 맞지 않으면 예외를 던지고, 그 예외가 C++/Rust 브리지를 넘어가므로 잘못된 체크포인트가 에러를 반환하지 않고 프로세스를 abort시킵니다. 호출이 예외를 던질 수 있다면 MLX 호출 전에 shape을 검증하십시오.

---

## 6. 추가 학습 리소스

### 핵심 용어

| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| NoPE | 위치 인코딩이 없는 레이어 | Kolibri-1 full-attention 레이어 |
| `e_score_correction_bias` | 선택에만 쓰는 expert별 bias | top-k 선택에 더하며 가중치에는 더하지 않음 |
| Teacher-forced trace | 기준 모델의 토큰을 두 모델에 넣고 위치별 logit을 비교 | 검증 방법 |

### 관련 PR/이슈

- 이슈 #2357: 이 PR이 구현하는 요구사항.
- AFMoE: 이 패밀리가 재사용하는 패밀리.
- #3090: `last_row_logits` 커버리지 규칙.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|----|
| 변경 파일 | 처음 두 커밋에서 25개, 여기에 `docs/environment-variables.md`와 리포트 2개 |
| 추가 테스트 | 19 (`models::kolibri1`) |

### 카테고리별 변경

| 카테고리 | 요약 |
|----------|------|
| 모델 | `kolibri1.rs`, `kolibri1_moe.rs`, `kolibri1_weights.rs`, `kolibri1_lm.rs`, `kolibri1_tests.rs` |
| 리팩터링 | AFMoE attention gate 옵션과 geometry, 공유 `hybrid_prefill_masks`, expert 플레인 헬퍼 사용 |
| 배선 | 감지, 로더, 레지스트리, 메타데이터, 메모리 추정, TP 목록, 스냅샷 크기 |
| 도구 | `scripts/kolibri1_oracle_trace.py` |
| 문서 | supported models, continuous batching, turbo KV cache, environment variables, CHANGELOG |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `0f0329df` | feat | add Kolibri-1 (kolibri1) hybrid sliding/full MoE |
| `5888260a` | docs | record Kolibri-1 trace validation against mlx-vlm |

---

## 8. 후속 조치

### 향후 개선

- AFMoE와 Kolibri-1의 `LanguageModel` 구현 공유 (미해결 MEDIUM).
- `drain_numbered_expert_planes`가 `mlx::core::stack` 호출 전에 플레인 shape을 검사하여, 잘못된 체크포인트가 abort 대신 에러를 반환하게 하기 (미해결 MEDIUM, 공유 헬퍼).
- MXFP4 변환본과 EAGLE3 드래프터 테스트.

---

## 부록

### A. 단위 테스트

`models::kolibri1`(19), `models::afmoe`(32), `coverage_tests`(18), 그리고 `-D warnings`로 lib과 tests에 대한 clippy.
