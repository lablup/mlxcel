# 기술 보고서: PR #1351 - feat(speculative): add the Laguna DFlash drafter

**작성일**: 2026-09-11
**작성자**: mlxcel maintainers
**리뷰어**: -
**상태**: 완료 (Linux/CUDA 호스트. `metal,accelerate` 워크스페이스 게이트는 이 호스트에서 실행할 수 없었고, block-versus-chain 정확성 probe는 이 호스트를 기본으로 거부하며, 처리량 기준은 override 아래 코드 텍스트에서 `--draft-block-size 8`일 때만 충족)
**언어**: Rust, Markdown
**위험도**: Medium (`mlxcel-core`에 새 드래프터 계열 추가, Laguna 타깃의 `SpeculativeTarget` 구현, Qwen 3.5 경로도 함께 지나가게 된 서버 DFlash 버스트 타깃 trait 일반화, 오프라인 `mlxcel generate --draft-kind dflash` 분기 신설)

---

## 요약

Poolside는 Laguna 릴리스마다 DFlash speculator를 함께 배포하지만 mlxcel의 DFlash 기계 장치는 Qwen 3.5 드래프터 형태에 고정되어 있었다. 이 PR은 `mlxcel_core::drafter::laguna_dflash`(fused QKV, per-head softplus 게이트, `aux_hidden_norms`, sliding-window 컨텍스트 어텐션)를 추가하고, Laguna 타깃에 dense 캐시와 rotating 캐시를 모두 되감는 `SpeculativeTarget`을 구현하며, `model_type: laguna` 드래프터를 `load_drafter`로 라우팅하고, 이 조합을 `mlxcel-server`와 오프라인 `mlxcel generate` 양쪽에 연결한다. 이 조합은 LFM2, Muse Glimmer arm과 같은 측정형 block-versus-chain 정확성 게이트 뒤에서 동작한다. GB10에서 Laguna XS 2.1 NVFP4와 공개 드래프터를 짝지으면 probe가 거부하므로(첫 verify 위치에서 200704 로짓 바이트 중 107246개가 다름) 기본으로는 DFlash가 꺼지고, `MLXCEL_MTP_ALLOW_INEXACT=1`을 주면 greedy 출력은 bf16 로짓 동률 위치를 제외하고 classic decode와 같고, 코드 완성에서는 라운드당 2.6~3.9개 제안을 수락하며, 처리량은 한적한 호스트에서 `--draft-block-size 8`일 때 한 번 classic을 앞섰고(1.14x), 동시 부하 아래 재실행에서는 0.82x였다. 기본 블록 16은 이 호스트에서 더 느리다.

---

## 1. 문제 정의

### 1.1 배경

DFlash 라운드 루프(`DFlashGenerator`)와 `SpeculativeTarget` trait은 타깃에 독립적이었지만, 유일한 드래프터 구현(`dflash::DFlashDraftModel`)은 분리된 `q/k/v_proj`, 게이트 없음, 컨텍스트용으로 계속 자라는 `KVCache`, Qwen 3.5 타깃 훅을 전제했다. `poolside/Laguna-XS-2.1-DFlash`는 `model_type: laguna`에 fused `self_attn.qkv_proj`, per-head `q_norm`/`k_norm`, per-head `g_proj` 게이트, 캡처한 타깃 레이어마다 하나씩의 RMSNorm, 512 sliding window, 그리고 `dflash_config { block_size 16, mask_token_id 12, num_target_layers 40, target_layer_ids [1, 13, 25, 33, 39], causal true }`를 선언한다.

### 1.2 기존 문제점

- **문제 1**: `load_drafter`가 `DrafterKind::Dflash`마다 Qwen 드래프터를 만들어 Laguna 체크포인트에서는 키 누락으로 실패했다.
- **문제 2**: #1347이 `LagunaModel::forward_with_capture`와 `LagunaCache::trim`을 준비해 두었지만 이 계열의 `SpeculativeTarget` 구현은 없었다.
- **문제 3**: 오프라인 `mlxcel generate --draft-kind dflash`는 모든 타깃에서 오류를 냈고, 서버 버스트는 `Qwen35DFlashTarget` trait에 묶여 있었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|-----|-------|-----------|
| 미묘하게 틀린 드래프터 forward도 유창한 텍스트를 만든다 (검증이 오류를 가린다) | High | Medium |
| 부분 수락 뒤 sliding `RotatingKVCache` 되감기가 이후 위치를 망가뜨린다 | High | Medium |
| 양자화 커널에서 multi-token verify와 single-token decode가 불일치한다 | Medium | High |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

드래프터 `config.json`의 값은 가중치를 읽기 전에 검증한다. 레이어 수, GQA 나누어떨어짐, window 하한, 어휘 안의 `mask_token_id`, `num_target_layers` 안에서 단조 증가하는 `target_layer_ids`, `draft_vocab_size == vocab_size`, `causal == true`가 그 대상이다. sanitizer는 가중치 키 집합을 정확히 대조하고, fused와 split 투영을 함께 가진 레이어나 불완전한 split 집합을 거부한다. `validate_target_compat`은 깊이나 어휘가 드래프터 계약과 다른 타깃을 bind 전에 거부한다.

### 2.2 성능 관점

GB10(sm_121), NVFP4 타깃, bf16 드래프터, greedy, 128 토큰, `MLXCEL_MTP_ALLOW_INEXACT=1`을 준 `mlxcel generate` 측정(정확성 probe는 이 호스트를 거부):

| 프롬프트 | Classic tok/s | DFlash 블록 16 tok/s | 평균 수락 길이 | Greedy id |
|-------|------|------|------|------|
| chat 0 (retry wrapper, `<think>` 채널) | 29.11 | 11.66 | 1.12 | 76에서 다름 (동률) |
| chat 1 (Rust LRU cache) | 26.86 | 12.24 | 1.84 | 동일 |
| chat 2 (TypeScript debounce) | 29.96 | 13.59 | 1.29 | 83, 91에서 다름 (동률) |
| code 0 (`retry_with_backoff` 본문, 템플릿 없음) | 28.70 | 24.81 | 3.27 | 101에서 다름 (동률) |
| code 1 (`lru_get` 본문) | 30.81 | 27.13 | 3.88 | 69에서 다름 (1 ulp 동률) |
| code 2 (`debounce` 본문) | 32.35 | 21.87 | 2.63 | 103에서 다름 (1 ulp 동률) |

code 0 블록 크기 스윕(classic 28.70 tok/s): 블록 4는 25.84 tok/s에 수락 2.20, 블록 6은 30.45에 3.00, 블록 8은 32.61에 3.57, 블록 12는 29.87에 3.74, 블록 16은 24.81에 3.27. 병합 후 code 0 재실행은 수락 카운터와 동률 위치를 정확히 재현했지만(블록 16에서 3.27, 블록 8에서 3.57) 블록 8의 속도 이득은 재현하지 못했다. 다른 에이전트의 cargo 빌드가 호스트에서 돌던 시점에 classic 30.00 tok/s 대 24.54 tok/s였다. 따라서 블록 8의 우위는 한적한 호스트에서 한 번 잰 값이지 GB10에서 이 조합의 안정적인 성질이 아니다. 이 호스트에서 16행 verify forward는 single-token decode 약 4회 비용(라운드당 verify 130 ms 대 classic 토큰당 35 ms)이라, 기본 블록은 라운드당 대략 5개 이상 수락해야 이득이 된다.

참조 경로를 따라 잰 드래프터의 위치별 정확도(probe b, shadow drafter): code 0은 d_0~d_5가 0.88, 1.00, 0.75, 0.62, 0.38, 0.25(8라운드 평균 수락 prefix 4.25), chat 0은 0.88, 0.50, 0.25, 0.12(평균 1.38). Poolside가 bf16 타깃으로 잰 값은 GSM8K, HumanEval, EvalPlus, Math에서 3.55~4.57이다.

서버 경로: 같은 요청을 `mlxcel-server --draft-model ... --draft-kind dflash`로 보내면 `rounds=60 proposed_tokens=835 accepted_tokens=67`로 오프라인 실행과 같은 카운터를 기록했고 11.75 tok/s였다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. Qwen 3.5 서버 버스트는 일반화된 `DFlashBurstTarget` trait을 지나가지만 동작은 그대로다(첫 hidden으로 프롬프트 마지막 행, 같은 캐시 팩토리).
- **새로운 의존성**: 없음.
- **호환성**: `MLXCEL_PRINT_TOKEN_IDS`는 `mlxcel generate`의 새 opt-in 출력이다. `MLXCEL_KEEP_BF16`과 `MLXCEL_CUDA_F16_NORMALIZE`가 Laguna 드래프터 가중치 dtype에도 적용되어 타깃과 맞춘다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: core 단위 테스트 7개(config 계약, sanitizer, 컨텍스트 window, 블록 내 인과성, window 가시성, RoPE 민감도), binary greedy-invariant 테스트 2개(수락 길이 0, 1, 2, full을 window wrap 너머까지 강제하는 oracle drafter, 무작위 가중치의 실제 드래프터), detection 테스트 1개, ignored 실제 체크포인트 probe 1개.
- **코드 복잡도**: 드래프터는 `dflash`의 형제 모듈로 `DFlashMlp`와 샘플링 헬퍼를 공유하고 라운드 루프는 손대지 않았다.
- **기술 부채**: 게이트는 MTP와 공유하므로 거부 로그 줄이 여전히 "MTP declined"라고 말한다. 버스트와 오프라인 arm이 그 뒤에 DFlash 이름의 줄을 덧붙인다.

---

## 3. 기술적 선택과 그 이유

### 3.1 이슈 본문과 어긋나는 곳은 vLLM 구현을 따른다

**컨텍스트:** 이슈 본문은 `ctx_qkv = qkv_proj(ctx_i)`와 모든 블록 행이 같은 511개 컨텍스트를 보는 고정 view를 명시했다. 병합된 참조 구현(`vllm/model_executor/models/laguna_dflash.py`, vllm-project/vllm#46853)은 투영된 컨텍스트를 각 레이어의 `input_layernorm`에 통과시킨 뒤 K/V로 투영하고, sliding window를 쿼리별 compute-time 제한으로 둔다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|-----|-----|-----|
| Option A: 이슈 본문 | 티켓과 일치 | 체크포인트가 실제로 서빙되는 방식이 아님 |
| **선택: Option B: vLLM 의미론** | 서빙 구현 및 학습 마스크의 블록 배치와 일치 | 티켓과 두 곳에서 다르며 PR에 명시 |

**선택 이유:** 공개 체크포인트의 norm 가중치가 전부 정확히 1.0이라 unit-RMS 컨텍스트 행에 대한 `input_layernorm`은 오늘은 수치상 no-op이지만, norm 가중치를 학습한 미래의 Laguna 드래프터에서는 달라진다. 쿼리별 window는 `create_causal_mask_with_window_full`이 그대로 준다.

**트레이드오프:** 학습 마스크(`speculators/models/dflash/attention.py`)는 window 하한을 bonus 위치에 고정한다. 컨텍스트가 512 토큰 미만이면 둘은 같고, 긴 window 끝단의 차이는 여기서 측정할 수 없었다.

### 3.2 ring 대신 temporal 컨텍스트 버퍼

**컨텍스트:** 드래프터는 컨텍스트 K/V만 캐시하고 proposal K/V는 forward마다 이어 붙인다.

**선택 이유:** `window - 1`로 상한을 둔 temporal 버퍼는 쿼리별 window를 단순한 additive `[block, prior + block]` 마스크로 만들고 `offset`을 RoPE용 절대 타깃 위치로 유지한다. ring이었다면 multi-row 어텐션마다 되풀어야 했다.

### 3.3 타깃의 buffered rotating 캐시

**컨텍스트:** 부분 수락 뒤 되감기는 sliding 레이어의 `RotatingKVCache`를 잘라내야 한다.

**선택 이유:** `enable_speculative_buffer(block_size)`가 verify 블록을 temporal 영역 안에 두므로 `trim`은 포인터 되감기이고 다음 append가 거부된 꼬리를 덮어쓴다. tiny 모델 greedy-invariant 테스트는 window 6에서 수락 길이 0, 1, 2, full을 강제해 wrap 이후에도 모든 trim 길이가 캐시에 닿게 한다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[변경 전]
load_drafter(Dflash) -> DFlashDrafter (Qwen 3.5 형태) -> DFlashGenerator -> SpeculativeTarget (Qwen35Model만)

[변경 후]
load_drafter(Dflash) -> model_type == laguna ? LagunaDFlashDrafter : DFlashDrafter
DFlashGenerator -> SpeculativeTarget (Qwen35Model | LagunaModel / LagunaWrapper)
mlxcel generate --draft-kind dflash -> generate_dflash::run_offline_dflash (Laguna)
mlxcel-server DFlash burst -> DFlashBurstTarget (Qwen 3.5, Qwen 3.5 VLM, Laguna)
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-core/src/drafter/laguna_dflash/attention.rs`**: fused `qkv_proj`를 q, k, v 행으로 분리. RoPE 전에 per-head `q_norm` / `k_norm`. `window - 1`보다 오래된 컨텍스트 행은 투영 전에 버리고 캐시 offset을 그만큼 전진. 컨텍스트 K/V만 `LagunaDFlashContextCache`에 넣음. `[context | block]` 위의 `create_causal_mask_with_window_full(L, prior, window)`. f32에서 per-head `softplus(g_proj(x))`.

**파일: `src/models/laguna_speculative.rs`**: `make_speculative_caches(block_size)`, `forward_speculative`(지정 레이어 뒤 캡처), `rollback_speculative_cache`(모든 캐시를 `block_size - (accepted + 1)`만큼 trim), `LagunaModel`과 `LagunaWrapper`의 `SpeculativeTarget` 구현.

**파일: `src/server/batch/speculative_burst.rs`**: `Qwen35DFlashTarget`을 `block_size`를 받는 캐시 팩토리와 `dflash_first_hidden` 훅을 가진 `DFlashBurstTarget`으로 일반화. Laguna는 캡처한 프롬프트 전체 행을 드래프터에 주고 Qwen 3.5는 마지막 행을 유지.

### 4.3 데이터 모델 변경

없음.

---

## 5. 학습 포인트

### 5.1 양자화 커널의 block-versus-chain 동률

**개념:** `T = K` verify 블록과 `K`번의 single-token decode는 같은 내적을 `M >= 2`와 `M = 1` 양자화 matmul 커널에서 다른 순서로 더한다. 타깃의 상위 2개 로짓이 bf16에서 같으면 argmax가 갈릴 수 있다.

**이 PR에서의 적용:** ignored `laguna_real_checkpoint_probe`가 실제 체크포인트에서 두 경로를 비교하고 불일치 위치마다 상위 2개 마진을 출력한다. 다섯 프롬프트에서 관측한 불일치 일곱 건 모두 chain 쪽 마진이 0.0 또는 0.125(로짓 21~34에서 bf16 1 ulp)였다. 프로덕션 게이트(`dflash_exactness_allows`)는 합성 입력에서 두 경로를 바이트 단위로 비교해 다르면 호스트를 거부하며, 이 GB10에서는 실제로 다르다.

### 5.2 거부 피드백 없이 드래프터 재기

**개념:** oracle이 라운드 루프를 참조 경로에 붙잡아 두는 동안 shadow drafter가 매 라운드 실제 드래프터의 제안을 기록하면, 앞선 거부와 무관한 위치별 정확도가 나온다.

**이 PR에서의 적용:** 이것으로 "드래프터가 `<think>` 산문에 약하다"(d_1 0.50)와 "드래프터가 고장났다"(코드에서 d_1 1.00)를 갈랐고, RoPE base, 회전 차원 수, 블록 내 마스크 변형 세 가지, 컨텍스트 norm, 캡처 레이어 off-by-one을 원인에서 제외했다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|-------|-----|-------|
| `DFlash` | block-diffusion 드래프터. masked forward 한 번으로 `block_size - 1`개 토큰을 제안 | 이번에 추가한 드래프터 계열 |
| `aux_hidden_norms` | `fc` 앞에 캡처한 타깃 레이어마다 하나씩 두는 RMSNorm | Laguna 고유 컨텍스트 경로 |
| `RotatingKVCache::enable_speculative_buffer` | sliding 레이어에서 verify 후 trim을 위한 temporal 여유 | 타깃 되감기 |

### 관련 PR/이슈

- Issue #1347: Laguna 계열 포팅 (이 드래프터가 짝을 이루는 타깃)
- vllm-project/vllm#46853: vLLM의 Laguna DFlash (참조 구현)

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|-----|---|
| 변경된 파일 수 | 23 |
| 추가된 라인 | +3293 |
| 삭제된 라인 | -83 |
| 테스트 추가 | 11 |

### 카테고리별 변경

| 카테고리 | 변경 수 | 주요 내용 |
|---------|--------|----------|
| Code Quality | 3 | 새 드래프터 모듈, 타깃 구현, 버스트 trait 일반화 |
| Performance | 1 | 블록 크기 스윕 문서화. 기본값은 체크포인트의 16 유지 |
| Documentation | 1 | `docs/supported-models.md` DFlash 행 |

### 관련 커밋

| Hash | Type | Message |
|------|------|---------|
| `154aafa1` | feat | add the Laguna DFlash drafter and target |
| `d64c73af` | merge | integrate Laguna into main's `DFlashTargetModel` design with the exactness gate |
| `76f16daf` | test | track the oracle drafter's reference position |
| `fa8f1919` | test | add a real-checkpoint DFlash probe and a RoPE sensitivity test |

---

## 8. 후속 조치

### 완료 필요

- [ ] Qwen 3.5 DFlash arm도 Laguna, LFM2, Muse Glimmer가 지금 도는 측정형 게이트를 돌릴지 결정(현재는 허용 기본값 유지).
- [ ] Apple Silicon 호스트에서 `metal,accelerate` 워크스페이스 게이트와 블록 크기 스윕 실행. 블록 8 권장은 GB10 기준이다.

### 모니터링 필요

- thinking 채널 트래픽의 요청별 수락 길이(`DFlash diagnostics` 로그, `spec_decode_*` 카운터). 이 구간에서 드래프터는 라운드당 1.1~1.8을 수락한다.

### 향후 개선 사항

- 배치(B > 1) Laguna DFlash. 긴 컨텍스트 측정에서 차이가 보이면 fixed-anchor window 옵션.

---

## 부록

### A. 테스트 결과

- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib -- drafter::laguna_dflash`: 7 passed.
- `cargo test --profile test-fast --features cuda --lib -- models::laguna_dflash_tests models::detection_tests::laguna_dflash models::laguna_tests`: 24 passed, 1 ignored (실제 체크포인트 probe).
- `cargo test -p mlxcel-core --profile test-fast --features cuda --lib -- drafter::dflash drafter::laguna_dflash drafter::tests`: 88 passed.
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`와 `-p mlxcel-core` 변형: clean. `cargo fmt --all -- --check`: clean.
- `cargo test --workspace --profile test-fast --features metal,accelerate`: 이 Linux/CUDA 호스트에서 실행 불가. 미실행.

### B. 성능 벤치마크

2.2 참고. 명령: `MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate -m models/mlx/laguna-xs-2.1-nvfp4 [--draft-model models/mlx/laguna-xs-2.1-dflash --draft-kind dflash [--draft-block-size N]] [--no-chat-template] -p ... -n 128 --temp 0`.

### C. 참고 자료

- vLLM `laguna_dflash.py`, `qwen3_dflash.py`, `v1/spec_decode/dflash.py` (`fb5138c3`).
- `vllm-project/speculators` `models/dflash/{core,attention,utils}.py` (학습 블록 배치와 마스크).
- `poolside/Laguna-XS-2.1-DFlash` 모델 카드 (bf16 타깃 기준 수락 길이).
