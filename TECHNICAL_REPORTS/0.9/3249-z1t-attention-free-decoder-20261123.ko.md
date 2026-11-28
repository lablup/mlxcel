# 기술 리포트: PR #3249 - Z1T-0 (z1t) attention-free AFT-conv 디코더

**작성일**: 2026-11-23
**리뷰어**: -
**상태**: PR 열림 (closes #2364), M3 Ultra에서 vendored mlx-vlm z1t 모듈과 대조 검증
**언어**: Rust
**위험도**: Low (새 모델 패밀리이며, 공유 코드는 레지스트리, 메모리 분류, `Used by:` 주석만 수정)

---

## 요약

이 PR은 attention이 없는 4-layer 디코더인 Extropic Z1T-0 (`model_type: z1t`)을 추가한다. 각 블록은 AFT-conv (causal depthwise conv와 causal cumulative pool)로 토큰을 섞고, 정규화는 Dynamic Tanh 하나뿐이며, 모든 내부 projection은 `tanh`로 감싼 고정 fan-in 4 sparse linear다. 모델은 `mlxcel generate`, `mlxcel-server`, `mlxcel arch`로 동작하고, 토큰당 O(1)인 decode 상태를 RWKV-7과 같이 `ModelOwnedSequenceState`에 두며, native batched decode에 참여한다. `models/z1t-0-mlx`에서 512 토큰 teacher-forced trace는 decided position 113개 중 0개에서 mlx-vlm과 어긋나고 최대 logit 차이는 0.0000이다. cumulative pool을 뺀 arm은 113개 중 62개에서 어긋나 gate를 통과하지 못한다. greedy decode 속도는 약 226 tok/s다.

---

## 1. 문제 정의

### 1.1 배경

Z1T-0은 Blaizzy/mlx-vlm#2166에서 mlx-vlm으로 포팅되었고 `mlx-community/Z1T-0-mlx` (fp32, 4.97 GB, GPT-2 BPE tokenizer)로 공개되었다. mlxcel에는 `z1t` detection arm이 없었고 구조가 비슷한 모델도 없었다. 캐시할 K/V 텐서가 없고, 시퀀스별 상태는 pool의 누적 합과 conv 입력의 마지막 `ksize - 1`개다.

### 1.2 기존 문제점

- **모델 패밀리 부재**: `detect_model_type`이 `z1t`를 인식하지 못해 체크포인트를 로드할 수 없었다.
- **sparse linear primitive 부재**: projection은 `weight [out, k]`, `bias [out]`, int32 `indices [out, k]` 테이블을 저장하고 `y[o] = sum_k W[o,k] * x[indices[o,k]] + b[o]`를 계산한다. 이를 구현한 기존 layer가 없었다.
- **새 패밀리에 대한 CI 규칙**: #3090 이후 새 `impl LanguageModel`은 `last_row_logits`를 구현하거나 목록에 올라야 하고, batched decode에 native로 참여하거나 `batched_decode_coverage_tests.rs`에 delegating 또는 per-row로 고정되어야 한다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| streaming 상태가 flat forward와 어긋남 (window나 pool의 off-by-one) | High: 모든 생성 토큰에서 품질이 조용히 떨어짐 | 수작업 포팅에서는 Medium, flat-vs-streaming 테스트와 trace로 커버 |
| 잘못된 체크포인트가 GPU gather의 범위 밖 읽기나 거대한 할당을 유발 | High: 로드나 첫 forward에서 crash 또는 OOM | Low이지만 체크포인트 config는 신뢰할 수 없는 입력 |
| 메모리 추정이 모델을 컨텍스트당 1.5 GiB KV 모델로 취급 | Medium: 잘못된 preflight 거부와 부풀려진 prompt-cache 예산 | `e36757d6` 이전에는 확정적 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

구현 리뷰와 보안 리뷰는 체크포인트의 `config.json`과 텐서를 신뢰할 수 없는 입력으로 다뤘다.

| 문제 | 심각도 | 상태 |
|------|--------|------|
| 2^32 근처의 `max_position_embeddings`가 `as i32`로 잘려 256행 `pe` 테이블이 shape 검사를 통과했고, position은 실제 값에서만 clamp되어 검사 없는 `take`가 테이블 밖을 읽음 | High | 수정 (`44e0247f`): `(max_positions + 1) * hidden`이 i32 원소 수에 들어가야 함 |
| `aft_ksize`에 상한이 없었음: 2^32는 빈 `w_conv`로 잘려 첫 forward에서 panic, 큰 값은 모든 시퀀스의 conv window를 거대하게 만듦 | High | 수정: `MAX_AFT_KSIZE = 1024`로 제한 (Z1T-0은 4) |
| sparse gather chunking이 upstream처럼 `L * out`만 측정해 batch와 fan-in을 무시했고, 큰 fan-in이면 projection마다 `8M * k` 원소를 할당 | High | 수정: `B * L * out * k`를 saturating 연산으로 측정하고 32M 원소로 제한, Z1T-0 (B = 1, k = 4)에서는 upstream chunk 크기와 같음 |
| lazy 누적 합이 긴 prefill 뒤 첫 decode step까지 각 layer의 전체 길이 `ekv`/`ek`를 메모리에 유지 | Medium | 수정: `schedule_prefill_state`로 스케줄, 출력은 bit 단위로 동일 |
| index 테이블 범위 초과 | 놓치면 High | 첫 커밋부터 로드 시 범위 검사 |
| dense 경로 weight, embedding, head의 shape 미검사 | Medium | 수정 (`5ac895ad`): `.scales` 텐서가 없을 때 `expect_dense_shape`가 `[out, in]` 검사 |
| 알 수 없는 sequence id에서 panic | Low | 미해결, RWKV-7과 같은 패턴 |

### 2.2 성능 관점

| 영역 | 이전 | 이후 | 변화 |
|------|------|------|------|
| greedy decode, `z1t-0-mlx`, M3 Ultra | 2.72 tok/s (첫 빌드) | 225.7 tok/s | 83배 |
| 같은 prompt, vendored mlx-vlm | 139.6 tok/s | 139.6 tok/s | 기준 |
| `mlxcel inspect` KV 추정 | 전체 컨텍스트 1.50 GiB (196608 B/token) | 0 (pure-SSM 분류) | 컨텍스트에 비례하는 상태 없음 |

첫 빌드는 길이 1인 시간 축에 대한 MLX strided `cumsum`에 decode step마다 layer당 93 ms를 썼다. 토큰 하나짜리 chunk는 그 자체가 inclusive cumsum이므로 `inclusive_cumsum_time`은 `len == 1`일 때 입력을 그대로 반환한다. 이 처리는 정확하며 decode 출력은 바뀌지 않는다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 다른 패밀리는 건드리지 않았다. 공유 코드 수정은 레지스트리 항목, `kv_arch::classify`의 `z1t` arm, APC opt-out 목록, `materialized_state_tail`과 `schedule_prefill_state`의 `Used by:` 주석이다.
- **새 의존성**: 없음.
- **Weight**: fp32 유지. 로더는 체크포인트에 있는 JAX 원본 키 (`embedding.*`, `blocks.N.*`, `norm.*`, `clf.*`, `pe.pe`)와 upstream이 정리한 키를 모두 받는다.

### 2.4 코드 품질 관점

- **테스트**: `src/models/z1t_tests.rs`에 13개, `hybrid_ssm_tests.rs`에 1개, `kv_arch`에 1개.
- **구조**: 패밀리를 `z1t.rs`, `z1t_config.rs`, `z1t_layers.rs`, `z1t_batch.rs`, `z1t_tests.rs`로 나눠 각 파일을 `docs/code-guidelines.md`에 따라 500줄 아래로 유지했다.
- **기술 부채**: RWKV-7에도 `e36757d6`이 Z1T에서 고친 KV 추정 문제가 그대로 있다. 범위 밖이라 그대로 두었다.

---

## 3. 기술적 선택과 그 이유

### 3.1 `KVCache`가 아닌 모델 소유 상태

**컨텍스트:** layer 상태는 `cum_eKV [B, heads, D/heads]`, `cum_eK [B, heads]`, 그리고 `eKV`와 `eK`의 마지막 `ksize - 1`행이다. 크기가 고정되어 있고 Z1T-0에서 시퀀스당 약 1 MB다.

| 대안 | 장점 | 단점 |
|------|------|------|
| 상태를 `KVCache` 텐서에 저장 | 캐시 배관 재사용 | 상태를 컨텍스트 비례로 잘못 표현하고, offset과 trim이 누적 합에 맞지 않음 |
| **선택: `ModelOwnedSequenceState<Vec<Z1tLayerState>>`** | RWKV-7, Mamba와 같은 계약이며 시퀀스별 수명 관리를 서버가 이미 처리 | 상태의 prefix caching은 별도 작업 필요 (범위 밖, APC는 opt-out) |

### 3.2 native batched decode 참여

**컨텍스트:** coverage 테스트는 모델 소유 상태를 쓰는 모든 패밀리가 native로 batch하거나 fallback을 선언하도록 요구한다.

**선택:** RWKV-7을 따른다. `z1t_batch.rs`는 `recurrent_batch` helper로 각 row의 layer 상태를 batch 축으로 쌓고, row별 position offset으로 forward를 한 번 돌린 뒤 상태를 다시 나눈다. fresh row와 초기화된 row가 섞인 경우, multi-token 호출, mask가 있는 호출은 row별 forward로 fallback한다. B=4 trace는 B=1과 512개 중 top-1 불일치 0개, 최대 logit 차이 0.0000으로 일치한다.

**트레이드오프:** step마다 row별 상태를 다시 쌓고 복사한다. 시퀀스당 약 1 MB라 작고, 다른 recurrent 패밀리도 같은 방식이다.

### 3.3 전체 원소 수 기준으로 sparse gather chunking

**컨텍스트:** gather는 reduction 전에 `[B, L, out, k]`를 만든다. `mlp.proj1` (out = 49152, k = 4)에서 2048 토큰 prefill이면 f32 원소 400M개다.

**선택:** `B * L * out * k`가 32M 원소 아래가 되도록 시퀀스 축으로 나눈다. 각 출력 row는 자신의 입력 row에만 의존하므로 시퀀스 축 분할은 정확하다. Z1T-0에서는 upstream의 8M `L * out` chunk와 같고, 더 넓은 fan-in이나 batch에서는 upstream 기준과 달리 상한이 유지된다.

### 3.4 position 테이블의 zero 확장

배포된 `pe [256, 12288]` 테이블은 수치상 0이지만 코드는 upstream을 따른다. 256 이후 position은 0행을 읽는다 (`pe_ext`에 0행 하나를 덧붙이고 position을 그 행으로 clamp). 원본 JAX `greedy_generate`는 256 토큰 window를 밀어 가지만, 이 포팅은 trace의 비교 대상인 mlx-vlm의 streaming 의미를 따른다.

---

## 4. 구현 상세

### 4.1 블록별 forward

```
x ─ DyT(norm1) ─ qkv = tanh(sparse(qkv_proj))  → Q [D], V [D], K [heads]
                  eK = exp(tanh(K)),  eKV = eK * V (head별)
                  kernel = exp(w_conv) - 1
                  num = conv(window ++ eKV) + (cum_eKV + cumsum(eKV))
                  den = conv(window ++ eK)  + (cum_eK  + cumsum(eK))
                  Y = tanh(Q) * num / (den + 1e-6)
                  out = tanh(sparse(out_proj(Y)))          ─ x += out
x ─ DyT(norm2) ─ tanh(sparse(proj2(tanh(tanh(sparse(proj1)))))) ─ x += mlp
```

conv는 upstream의 tap 순서를 따른다. `out[t] = sum_j kernel[:, j] * ext[t + j]`이고 `ext = window(ksize - 1) ++ new`이므로 tap `ksize - 1`이 현재 토큰이다. fresh 상태는 0이며 flat forward의 left padding과 같다.

### 4.2 연결

`ModelType::Z1t`, `"z1t"` detection arm, `LoadedModel`, special 및 nonstandard 로더, `model_metadata`, 레지스트리, 메모리 추정 (`kv_arch`가 Z1T를 pure SSM으로 분류), tensor-parallel 모델 목록, APC opt-out (`HYBRID_SSM_MODEL_TYPES`와 `HYBRID_SSM_ARCH_FRAGMENTS`), `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/turbo-kv-cache.md`, `CHANGELOG.md`.

`last_row_logits`는 샘플링할 row를 고르고 최종 DyT를 적용한 뒤 bias가 있는 head를 적용한다. 따라서 prefill이 모든 row를 50257 폭의 head에 통과시키지 않는다.

### 4.3 로드 시 거부 조건

`conv`가 아닌 `aft_kind`, `aft_heads`로 나누어떨어지지 않는 `hidden_size`, 0이거나 1024를 넘는 `aft_ksize`, 원소 수가 i32를 넘는 position 테이블, 범위를 벗어난 index 테이블, config와 shape가 맞지 않는 dense weight.

---

## 5. 학습 포인트

### 5.1 MLX에서 길이 1 scan은 공짜가 아니다

MLX는 scan 축 길이가 1이어도 `cumsum`을 strided scan kernel로 내린다. 토큰마다 4 layer 곱하기 2번의 scan을 도는 decode loop에서는 이 비용이 step을 지배했다. 어떤 연산이 퇴화한 shape에서 항등이면 그 shape를 검사해 연산을 건너뛴다. upstream mlx-vlm은 시간 축을 없앤 별도 `_decode_step`으로 같은 함정을 피한다.

### 5.2 인덱싱에 쓰는 타입으로 크기를 검사한다

HIGH 보안 지적 두 건은 모두 `as i32`로 좁힌 뒤 값을 검증한 데서 나왔다. 넓은 타입 (`usize` 또는 `i64`, checked 또는 saturating 연산)에서 검증하고, 상한이 성립한 뒤에만 좁힌다.

### 5.3 음성 arm으로 정확성을 입증한다

trace gate가 decided position 113개 중 0개로 통과해도, 그것만으로는 trace가 잘못된 모델을 잡아낼 수 있다는 근거가 되지 않는다. pool을 뺀 conv-only arm은 113개 중 62개로 실패하므로, gate가 올바른 포팅과 구조적으로 틀린 포팅을 구분한다는 것이 확인된다.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 관련성 |
|--------|------|--------|
| AFT (Attention Free Transformer) | q·k 없이 지수화한 key를 position에 걸쳐 가중해 토큰을 섞는 방식 | Z1T의 token mixer는 AFT-conv 변형 |
| DyT (Dynamic Tanh) | LayerNorm 대신 `weight * tanh(alpha * x) + bias` | Z1T의 유일한 정규화 |
| 고정 sparsity linear | 각 출력이 고정 index 테이블로 정한 `k`개 입력만 읽음 | 모든 내부 projection, fan-in 4 |

### 관련 PR/이슈

- Issue #2364: 이 PR이 구현하는 요구사항.
- #3090: `last_row_logits` coverage 규칙.
- #2685, #2672: 모델 소유 sequence 상태에 대한 batched decode, `z1t_batch.rs`가 따르는 패턴.
- Blaizzy/mlx-vlm#2166: 수치 기준으로 쓴 upstream 포팅.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 25 |
| 추가 줄 | +1766 |
| 삭제 줄 | -9 |
| 추가 테스트 | 15 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-----|------|
| 모델 | 파일 5개 | `z1t.rs`, `z1t_config.rs`, `z1t_layers.rs`, `z1t_batch.rs`, `z1t_tests.rs` |
| 연결 | 파일 12개 | detection, 로더, 레지스트리, metadata, 메모리 추정, TP 목록, APC opt-out |
| 보안 | 커밋 3개 | config 상한, gather 임시 메모리 상한, dense shape 검사 |
| 문서 | 파일 4개 | supported models, continuous batching, turbo KV cache, CHANGELOG |

### 관련 커밋

| Hash | 유형 | 메시지 |
|------|------|--------|
| `7b5d502` | feat | add Z1T-0 (z1t) attention-free AFT-conv decoder |
| `4ead407` | docs | document Z1T-0 support |
| `e36757d` | fix | size Z1T as having no KV cache |
| `44e0247` | fix | bound Z1T config values and sparse gather transients |
| `5ac895a` | fix | validate Z1T dense shapes and exempt Z1T from APC by arch name |

---

## 8. 후속 조치

### 향후 개선 사항

- RWKV-7에도 `e36757d6`이 Z1T에서 해결한 KV 추정 문제가 있다.
- streaming 상태의 prefix caching (`z1t`는 APC opt-out).
- dense 변형 (`linear_fan_in: null`)은 코드 경로가 받아들이고 unit test로 커버되지만, 검증할 실제 dense 체크포인트가 없다.
- 알 수 없는 sequence id에서 오류 대신 panic이 난다 (RWKV-7과 공통).

---

## 부록

### A. `models/z1t-0-mlx` 검증 (M3 Ultra)

- `mlxcel generate -p "The meaning of life is" -n 64 --temp 0`: "The meaning of life is not a thing of the past. It is not a question of meaning. ..."를 225.7 tok/s로 생성. vendored mlx-vlm 모듈도 같은 64 토큰을 생성한다.
- `mlxcel-server`, `/v1/completions` 요청 1회 (32 토큰, temperature 0): 같은 이어쓰기, `finish_reason: length`, usage 5 + 32.
- teacher-forced trace (`examples/logit_trace`, 64 토큰 chunk 8개, top-8, `compare_logit_traces.py --decided 2`):

| 기준 | 후보 | top-1 불일치 | decided 불일치 | 최대 logit 차이 |
|---|---|---|---|---|
| mlx-vlm z1t | mlxcel | 0 / 512 | 0 / 113 | 0.0000 |
| mlx-vlm z1t | mlx-vlm, conv only | 325 / 512 | 62 / 113 (실패) | n/a |
| mlxcel B=1 | mlxcel batched B=4 | 0 / 512 | 0 / 113 | 0.0000 |

perplexity: 두 full arm 모두 130.56, conv-only arm은 320.92.

### B. Unit test

`models::z1t` (13), `coverage_tests` (18), `prompt_cache::hybrid_ssm` (14), `execution::kv_arch` (57), lib과 tests에 대한 `-D warnings` clippy.
