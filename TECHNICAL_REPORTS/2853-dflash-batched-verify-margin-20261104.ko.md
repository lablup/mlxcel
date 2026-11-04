# 기술 보고서: PR #2853 - test(speculative): batched DFlash parity를 측정된 jitter margin으로 게이트

**작성일**: 2026-11-04
**상태**: 리뷰 중 (open PR; #2851 close)
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #2853은 batched DFlash verify forward가 single-row forward와 byte-identical하지 않다는 사실을 확립하고, batched parity gate를 측정된 evaluation-path jitter margin으로 다시 표현한다. M3 Ultra에서 `qwen3.5-4b-4bit`와 `qwen3.5-4b-dflash`, block 16으로 hard byte-equality gate가 row 0 token 14에서 실패했다 (batched 5388 대 B=1 15217). 하나의 burst에 동일한 프롬프트 네 개를 넣어도 `clamped_rounds=0`, ragged round 없음, 네 row 모두 같은 token에서 갈라졌으므로 accept policy와 uniform clamp가 원인이 될 수 없다. One-shot B=1 arbiter는 batched token을 1위로 두고 B=1 token을 0.25 logit 뒤에 둔다. 이는 Gemma 4 MTP verifier가 이미 `EVAL_PATH_JITTER_MARGIN`으로 게이트하는 near-tie class다. 이제 gate는 첫 mismatch를 `BATCHED_VERIFY_JITTER_MARGIN = 1.0` logit 이내에서만 허용하며, 이는 측정된 gap의 네 배다. 커밋된 margin-removal arm은 margin 0에서 관측된 mismatch 4개 중 4개를 모두 거부한다. Outcome (a)인 byte identity는 측정에 근거해 기각했다. B=1 shape을 재현하려면 row별 KV/GDN cache view에 대해 target verify forward를 row 단위로 평가해야 하며, 이는 Gemma 4 per-row architecture(#2190)이고 #2442에서 측정한 1.07x batching 이득을 소비한다. Margin은 한 호스트와 한 프롬프트 집합에서만 보정했고, runtime B>1 exactness probe는 구현하지 않았다.

---

## 1. 문제 정의

### 1.1 배경

Issue #2442 item 1은 batched DFlash round loop의 uniform clamp를 per-row accept로 교체하자는 작업이며, `greedy_parity_dflash_qwen35_batched_b4_matches_b1` gate를 전제로 한다. 이 gate는 수정되지 않은 `main`에서 실패했으므로 accept policy가 blocker가 아니었다. Issue #2851은 그 선행 실험을 요구했다. batched `[B, block]` target verify forward가 `[1, block]` forward와 byte-identical한지 확립하고, 동일하게 만들거나(outcome a) 아니면 structural-defect control을 갖춘 측정된 jitter margin으로 gate를 다시 표현하라는 것이다(outcome b).

Verify 호출은 batched round loop의 `[B, bs]` 입력에 대한 `target.verify_forward_with_capture_layers`이며, `SpeculativeTarget`을 통해 `Qwen35Model::forward_speculative`에 도달한다. 서버의 batched burst는 `run_dflash_batched_on_target`에서 같은 loop를 구동한다. 기존 runtime exactness probe인 `probe_block_chain_exactness`는 block을 chain과 B=1에서만 비교하므로 batch size에 의존하는 차이를 볼 수 없다.

### 1.2 기존 문제점

- **Byte-equality gate는 Metal에서 한 번도 통과하지 못했다.** 이 PR 이전에 B=4 gate는 96 rounds, 93 clamped, row-round당 평균 accept는 walk 2.938 대 committed 0.323이었고, row 0이 token 14에서 처음 갈라졌다 (5388 대 15217).
- **Accept policy는 그럴듯하지만 틀린 설명이었다.** Ragged accept나 stale prefix가 이후 token을 움직일 수 있으므로, gate 실패만으로는 원인을 분리할 수 없었다.
- **Divergence는 near-tie이지 분포 손상이 아니다.** One-shot B=1 arbiter의 argmax는 batched token 5388이고, B=1 token 15217은 불과 0.25 logit 뒤에 있다. 두 실행의 나머지 row는 모두 byte-identical하다.
- **기존 Qwen 3.5 exactness probe는 이 class를 다루지 못한다.** B=1에서만 평가하므로, exact하지 않은 batched width를 runtime에서 decline하겠다는 결정에 측정 근거가 없었다.
- **Hard byte gate는 batching path 전체를 비싼 재작성에 묶는다.** B=4에서 byte identity는 사각형 KV/GDN cache의 per-row view에 대해 target verify forward를 row별로 평가해야 한다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| Margin이 실제 structural parity break를 숨긴다 | 중간 (test gate의 정확성) | 낮음: 두 token이 세 번째 evaluation path의 top에서 1.0 logit 이내일 때만 허용하고, margin-removal arm이 같은 mismatch를 0에서 거부해야 한다 |
| Margin이 잘못된 호스트나 프롬프트 집합에서 보정되었다 | 중간 (다른 호스트는 fail closed로 CI를 막는다) | 중간: 보정은 M3 Ultra 한 대와 프롬프트 집합 하나를 덮는다 |
| 향후 변경이 근거 없이 margin을 올린다 | 중간 (tolerance가 관대해진다) | 낮음: 상수에 do-not-raise 지침이 있고 control이 커밋되어 있다 |
| Runtime에 B>1 exactness probe 없이 batched path가 서빙된다 | 낮음에서 중간 (검증되지 않은 runtime surface) | 확실: 이 PR은 의도적으로 probe를 추가하지 않는다 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

Runtime code, 입력 파싱, 인증, 데이터 경로 변경은 없다. 이 PR은 ignored real-model test 두 개, 공유 harness, 문서 단락 하나를 건드린다. 새로운 입력 surface도, 사용자 데이터 로깅도 없다.

### 2.2 성능 관점

변경은 test-only이며 서빙 경로를 바꾸지 않는다. 성능 쟁점은 아키텍처적이다. Outcome (b)는 batched path를 측정된 처리량 그대로 유지하고, outcome (a)는 per-row verify 평가를 추가한다. M3 Ultra에서 clamp arm은 B=1 77.70 / 77.23 / 77.83 tok/s, B=4 82.96 / 83.00 / 83.00 tok/s (512 token aggregate)로 약 1.07x였다. Byte identity에 필요한 per-row 재작성은 이 이득 전체를 소비한다. 새 harness는 mismatch가 있는 row의 첫 지점에서 one-shot B=1 forward를 한 번 더 실행하며, 이는 ignored real-model test 안에서만 일어난다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. Test 이름은 유지되고, 새 control은 추가 ignored test다.
- **새 의존성**: 없음.
- **호환성**: test는 CI hardware lane용 `#[ignore]` real-model test로 남는다. `MLXCEL_SDPA_DETERMINISTIC=1`에서도 같은 판정이다.

### 2.4 코드 품질 관점

Burst runner, top-gap arbiter, margin 규칙을 새 공유 harness 모듈 `tests/speculative_parity/qwen35_dflash_harness.rs`로 옮겨, hard gate와 identical-prompts control이 서로 다른 tolerance로 표류할 수 없게 했다. `BATCHED_VERIFY_JITTER_MARGIN` 상수는 보정 기록 전체를 정의 옆에 둔다. Margin-removal control은 주석이 아니라 test의 일부이므로, margin을 쓸모없게 만드는 미래의 완화가 조용히 통과하지 않고 실패한다.

---

## 3. 기술적 선택과 그 이유

### 3.1 Outcome (b): byte identity 대신 측정된 jitter margin

**컨텍스트:**
Gate는 `[4, block]`과 `[1, block]` target verify forward 사이의 byte identity를 요구했다. Identical-prompts control은 모든 row가 같은 block을 draft하고 모든 round가 walk를 commit해도 divergence가 남는다는 것을 보여 주었으므로, 차이는 batched target forward 내부에서 발생한다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|---|---|---|
| Option A: kernel dispatch를 고정하거나 per-row verify로 byte identity 강제 | 정확한 gate, tolerance 없음 | Per-row 평가에는 per-row KV/GDN cache view가 필요하다 (Gemma 4 per-row architecture, #2190). #2442의 1.07x batching 이득을 소비한다. 다른 shape이 dispatch를 바꾸지 않음을 증명하기 어렵다 |
| Option B: 측정된 evaluation-path jitter margin | Batching 이득 유지, 기존 Gemma 4 MTP 선례와 일치, 결정적인 위치에서는 fail closed | 구조상 near-tie를 허용한다. 보정이 호스트별이다 |
| **선택: Option B** | 측정되었고 관측 gap 대비 작으며 margin-removal control이 보호한다 | Batched forward를 exact하게 만들지 않고, 호스트별 상수를 남긴다 |

**선택 이유:**
Verify block의 leading dimension은 MLX가 선택하는 quantized-projection 및 SDPA kernel과 reduction tiling을 바꾼다. 이는 프로젝트가 prefill partitioning에 대해 이미 문서화하고, Gemma 4 MTP verifier에서 `EVAL_PATH_JITTER_MARGIN`으로 게이트하는 near-tie class다. Greedy decoding에서 모든 emitted token은 draft 품질과 무관하게 accepted prefix에 대한 target의 argmax이므로, drafter 제안은 emitted chain을 바꿀 수 없고 target forward의 evaluation path만 바꿀 수 있다. 따라서 B=1 shape 강제는 값싼 dispatch 조정이 아니라 target forward의 per-row 평가를 뜻한다.

**트레이드오프:**
Gate는 더 이상 byte-equality assertion이 아니다. 첫 mismatch가 세 번째 evaluation path(one-shot B=1)의 top에서 두 후보 token을 margin 이내에 둘 때만 통과하는 near-tie assertion이 된다. 분포를 1.0 logit보다 크게 움직이는 structural break는 여전히 실패한다.

### 3.2 Margin은 1.0이며 측정 gap의 네 배다

**컨텍스트:**
측정된 최대 divergence는 0.25 logit이다. Arbiter의 top-gap은 batched token에 대해 0, B=1 token에 대해 0.25다. Margin은 관측된 evaluation-path noise를 덮으면서 결정적인 flip을 삼키지 않아야 한다.

**선택 이유:**
`BATCHED_VERIFY_JITTER_MARGIN`은 1.0으로, 측정된 최대값의 네 배다. 이는 공짜 확대가 아니다. 규칙이 두 token을 세 번째 path에 대해 측정하고, 커밋된 margin-removal control이 margin이 실제로 작동함을 증명한다. Margin 0에서 관측된 mismatch가 모두 거부된다 (identical-prompts control에서 4개 중 4개). Batched token이 1.0 logit보다 큰 차이로 이기거나 두 후보 중 하나가 arbiter의 top에서 1.0 logit 넘게 아래에 있는 결정적 위치는 여전히 fail closed다.

**트레이드오프:**
이 상수는 이 하드웨어에서 이 class의 divergence에 보정되어 있다. 다른 quantization, kernel family, MLX 버전은 정당한 near-tie에서 1.0 logit을 넘는 첫 mismatch를 만들 수 있다. 그 경우 test는 크게 실패하고, 상수는 맹목적으로 올리는 것이 아니라 새 측정이 필요하다.

### 3.3 One-shot B=1 arbiter에 대해 두 token을 모두 측정

**컨텍스트:**
Batched token의 logit만 B=1 token의 logit과 비교하면, batched path가 우연히 가깝게 rank한 모든 쌍을 허용하게 되며, arbiter가 결정적으로 rank한 쌍까지 포함된다. One-shot forward 자체도 세 번째 evaluation path이고 incremental chain과 다를 수 있다.

**선택 이유:**
`b1_arbiter_top_gaps`는 `prompt + emitted_prefix[..index]`를 새 sequence에서 `Qwen35Model::forward_with_sequence_id`로 prefill하고, 마지막 위치 logits에서 `top - logit[want]`, `top - logit[got]`, arbiter argmax를 반환한다. 두 gap 모두 `[0, margin]` 안에 있어야 한다. 이것이 "모델이 분리하지 못하는 두 후보"와 "batched path가 모델이 훨씬 뒤에 rank한 token으로 이동한 경우"를 구분하며, 후자가 structural defect의 모습이다.

### 3.4 Identical-prompts control이 accept policy를 분리한다

**컨텍스트:**
Hard gate는 서로 다른 프롬프트 네 개를 쓰므로 ragged accept와 uniform clamp가 활성화된다. 그 실패는 target forward가 아니라 cache layout에서 비롯될 수 있다.

**선택 이유:**
Control은 `[prompts[0]; 4]`를 실행한다. 모든 row가 같은 prefix에서 같은 block을 draft하므로 모든 round가 전체 walk를 commit한다. Test는 parity 규칙을 적용하기 전에 `clamped_rounds == 0`과 `ragged == false`를 assert하므로 control이 판별력을 갖는다. Clamp나 ragged accept가 원인이라면 동일한 row는 갈라지지 않을 것이다. 실제로는 같은 token에서 같은 gap으로 갈라지므로 차이는 batched verify forward 자체에 있다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before]
tests/speculative_parity.rs
  ├── gemma4_batched.rs
  └── qwen35_dflash_batched.rs   (burst runner + byte-equality helper inline)

[After]
tests/speculative_parity.rs
  ├── gemma4_batched.rs
  ├── qwen35_dflash_harness.rs   (burst runner, arbiter, margin 규칙, 상수)
  └── qwen35_dflash_batched.rs   (hard gate + identical-prompts control)
```

`tests/speculative_parity.rs`는 test 모듈 앞에서 `#[path = "speculative_parity/qwen35_dflash_harness.rs"] mod qwen35_dflash_harness;`로 harness를 등록한다.

### 4.2 주요 코드 변경

**File: `tests/speculative_parity/qwen35_dflash_harness.rs` (신규)**

보정된 상수는 근거와 do-not-raise 지침을 함께 담는다:

```rust
pub(crate) const BATCHED_VERIFY_JITTER_MARGIN: f32 = 1.0;
```

Arbiter는 새 one-shot B=1 prefill을 실행하고 두 top-gap을 반환한다:

```rust
fn b1_arbiter_top_gaps(
    model: &mlxcel::models::Qwen35Model,
    seq_raw: u64,
    prompt: &[i32],
    emitted_prefix: &[i32],
    want: i32,
    got: i32,
) -> (f32, f32, i32)
```

Margin 규칙은 어느 한 token의 gap이 margin을 넘으면 row를 거부한다. 그래서 같은 함수가 `margin == 0.0`에서 margin-removal control이 된다:

```rust
pub(crate) fn rows_outside_margin(ties: &[NearTie], margin: f32) -> Vec<String>
```

**변경 이유:** 두 test가 같은 burst runner와 같은 arbitrated 규칙을 필요로 한다. 하나의 harness에 두어 hard gate와 control이 미묘하게 다른 tolerance를 구현하지 못하게 한다.

**File: `tests/speculative_parity/qwen35_dflash_batched.rs`**

Hard gate는 byte equality를 직접 assert하는 대신 arbiter를 모아 margin을 적용한다:

```rust
let (ties, structural) = first_mismatch_arbiters(
    model, "dflash B=4", &prompts, &batched, &reference, ARBITER_SEQ_BASE_GATE,
);
let outside = rows_outside_margin(&ties, BATCHED_VERIFY_JITTER_MARGIN);
assert!(structural.is_empty(), ...);
assert!(ragged, "no round had ragged accepts; the prompts do not exercise the clamp");
assert!(outside.is_empty(), ...);
```

새 control은 두 전제, parity 규칙, margin-removal arm 순서로 assert한다:

```rust
assert_eq!(batched.clamped_rounds, 0, ...);
assert!(!ragged, ...);
let at_zero = rows_outside_margin(&ties, 0.0);
assert!(!at_zero.is_empty(), "margin is not load-bearing ...");
```

**변경 이유:** `clamped_rounds == 0`과 `!ragged`가 control의 결론을 batched forward에 귀속시킨다. Margin-removal arm은 margin이 모든 것을 허용하는 tolerance로 조용히 변하는 것을 막는다.

**File: `docs/CONTINUOUS_BATCHING.md`**

Qwen 3.5에서 equal-length queued DFlash window가 순차 서빙과 bitwise identical하지 않다는 것, leading dimension이 kernel dispatch를 바꾼다는 것, parity gate가 four-identical-prompts control로 뒷받침되는 보정된 1.0-logit near-tie margin을 쓴다는 것(#2851)을 기록한다.

### 4.3 데이터 모델 변경

없음.

---

## 5. 학습 포인트

### 5.1 Evaluation-path jitter와 structural parity의 구분

**개념:**
Metal에서 같은 산술을 다른 leading dimension으로 구성하면 다른 quantized-projection 및 SDPA kernel을 선택하고 reduction을 다르게 tile할 수 있다. 결과는 비결정성이 아니라 작고 결정적인 logit 이동이다. 두 후보가 그 이동 폭 안에 있으면 greedy argmax가 정당하게 flip할 수 있다.

**이 PR에서의 적용:**
관측된 divergence에서 이동은 0.25 logit이고 flip은 near-tie다. Arbiter는 batched token을 1위로 둔다. Gate는 두 후보가 세 번째 evaluation path의 top에서 1.0 logit 이내에 있을 것을 요구해 jitter regime과 structural break를 구분한다. Structural defect는 분포를 훨씬 크게 움직이고 결정적 위치를 flip하며, 같은 규칙이 이를 거부한다.

**일반적인 사용 사례:**
- Quantized GPU backend에서의 batched 대 single-row parity test.
- Incremental prefill과 one-shot prefill 경로 비교.
- 배제해야 할 실패 모드가 positional 또는 cache-layout defect이고 floating-point noise가 아닌 모든 tolerance gate.

### 5.2 동일 입력 control이 policy와 실행을 분리한다

**개념:**
Batched path가 여러 이유로 갈라질 수 있을 때 동일 입력을 실행하면 policy 변수를 제거한다. 모든 row가 같은 block을 draft하고 같은 accept count를 commit하므로 ragged accept, clamping, stale prefix가 무력해진다.

**이 PR에서의 적용:**
`[prompts[0]; 4]`와 `clamped_rounds == 0`, `!ragged_round` assert가 "gate가 실패했다"를 "batched target forward가 다르다"로 바꾼다. 이것이 accept-policy 수정이 아니라 margin을 정당화하는 유일한 결론이다.

**일반적인 사용 사례:**
- Batched decode와 speculative-decoding parity.
- 공유 code path에 row별 policy state가 있는 모든 동시 pipeline.

### 5.3 Tolerance에는 제거 control이 필요하다

**개념:**
추가만 되고 필요성이 시험되지 않는 margin은 잡아야 할 defect까지 조용히 허용할 만큼 넓어질 수 있다. 제거 control은 같은 근거를 더 엄격한 설정에 적용하고 실패해야 한다고 요구한다.

**이 PR에서의 적용:**
Identical-prompts control은 관측된 mismatch를 `margin == 0`에서 다시 실행하고 최소 하나가 거부됨을 assert한다. 여기서는 4개 중 4개가 거부되므로 1.0 margin은 장식이 아니라 실제로 작동한다.

**예시 코드:**
```rust
let at_zero = rows_outside_margin(&ties, 0.0);
assert!(!at_zero.is_empty(), "margin is not load-bearing ...");
```

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|---|---|---|
| DFlash | Diffusion-style drafter를 쓰는 block-draft speculative decoding | 테스트 대상 batched verify path |
| Evaluation-path jitter | 서로 다른 kernel dispatch나 reduction tiling에서 오는 작고 결정적인 logit 이동 | 0.25-logit divergence의 원인 |
| Near-tie | Jitter margin 안에 있는 두 후보로, greedy argmax가 flip할 수 있는 상태 | Margin gate가 허용하는 것 |
| One-shot arbiter | 세 번째 evaluation path로 쓰는 새 B=1 full-prefill forward | Mismatch가 near-tie인지 결정 |
| Uniform clamp | 모든 row에 같은 accept count를 commit하는 round policy | Identical-prompts control에서 무력하며 forward를 분리한다 |
| Margin-removal control | 관측된 mismatch를 margin 0에서 다시 실행 | Margin이 실제로 작동함을 증명 |

### 관련 기술/프레임워크

- **MLX**: Apple의 array framework와 Metal kernel dispatch. Leading dimension에 따른 kernel 선택이 divergence를 만든다.
- **Gemma 4 MTP verifier**: 이 PR이 Qwen 3.5로 옮긴 `EVAL_PATH_JITTER_MARGIN`과 `check_row_parity_with_near_tie` 선례.

### 관련 PR/이슈

- Issue #2851: 이 PR이 close하는 선행 실험.
- Issue #2353: Batched DFlash clamp와 원래의 B=4 대 B=1 gate.
- Issue #2442: Batching 이득과 이 작업이 re-scope하는 per-row accept 항목.
- Issue #2190: Byte identity가 필요로 하는 Gemma 4 per-row verify architecture.
- PR #2170: Prefill partitioning과 문서화된 near-tie class.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 수 | 4 |
| 추가된 줄 | +579 |
| 삭제된 줄 | -194 |
| 추가된 test | 새 ignored real-model control 1개와 공유 harness |
| Runtime 동작 변경 | 없음 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|---|---|---|
| Tests | 3 files | 공유 harness, hard gate를 margin gate로 재표현, margin-removal arm을 갖춘 identical-prompts control |
| Documentation | 1 file | `docs/CONTINUOUS_BATCHING.md`에 non-bitwise window와 margin 기록 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `a3caa55` | test | test(speculative): gate batched DFlash parity on measured jitter |

---

## 8. 후속 조치

### 완료 필요

- [ ] 1.0을 이식 가능한 값으로 취급하기 전에 다른 backend와 checkpoint에서 margin을 보정한다 (GB10/CUDA는 2026-10-18 #2442 댓글에서 동일 프롬프트 B=2 window가 이미 갈라진 알려진 두 번째 backend다). 상수는 새 측정과 margin-removal control 재실행 없이 올리면 안 된다.
- [ ] 이 측정 결과로 #2442 item 1을 re-scope한다. Per-row accept는 그것이 서빙할 shape에 대한 exactness 질문이 정리될 때까지 차단한다.

### 검증되지 않은 surface

- Margin은 한 호스트(M3 Ultra, Metal, macOS 26.6.2)와 한 프롬프트 집합에서 보정되었다. 그 값은 다른 하드웨어, quantization, MLX 버전에 대한 근거가 아니다.
- 어느 한 token의 첫 mismatch가 1.0 logit을 넘으면 fail closed다. Evaluation-path divergence가 더 큰 호스트에서는 정당한 near-tie도 상수를 다시 측정할 때까지 gate를 실패시킨다.
- Runtime B>1 exactness probe는 구현하지 않았다. `probe_block_chain_exactness`는 여전히 block을 chain과 B=1에서만 비교하므로, 서빙되는 batched path에는 exact하지 않은 batched width를 decline할 runtime 검사가 없다. 이 PR은 test gate를 정직하게 만들 뿐 runtime gate를 추가하지 않는다.

### 모니터링 필요

- MLX, quantization, kernel dispatch 변경이 들어올 때 CI hardware lane의 `greedy_parity_dflash_qwen35_batched_b4_matches_b1`과 `identical_prompts_dflash_qwen35_b4_matches_b1`을 지켜본다.
- Margin-removal arm은 canary다. 모든 row가 byte-identical하다고 보고하기 시작하면 그 tree에서는 margin이 더 이상 필요 없다는 뜻이므로 상수를 재검토해야 한다.

### 향후 개선 사항

- Runtime이 exact하지 않은 batched width를 (test gate가 아니라) 실제로 decline해야 한다면, 하나의 `[B, block]` forward를 B개의 single-row forward와 비교하는 `probe_block_chain_exactness` 확장이나 sibling을 (model, width, batch)별 memoization과 함께 추가한다.
- Byte identity는 per-row verify 평가의 측정된 throughput budget이 있을 때만 다시 검토한다. Clamp arm이 B=1 대비 약 1.07x였으므로, 재작성이 그보다 싸야 batching 근거가 유지된다.

---

## 부록

### A. 테스트 결과

| 검사 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --test speculative_parity qwen35_dflash_batched -- --ignored --test-threads=1` | 2 passed (이전: 2 failed) |
| `MLXCEL_SDPA_DETERMINISTIC=1`로 동일 실행 | 2 passed |
| `cargo fmt --all -- --check` | 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 통과 |

### B. 측정 환경

| 항목 | 값 |
|---|---|
| Host | M3 Ultra, 80-core, 512 GB unified memory, macOS 26.6.2, Metal |
| Target | `models/qwen3.5-4b-4bit` (mlx-community/Qwen3.5-4B-4bit) |
| Drafter | `models/qwen3.5-4b-dflash` (z-lab/Qwen3.5-4B-DFlash) |
| Block size | 16 |
| 생성 token 수 | 128 |
| Sampling | Temperature 0, greedy |

### C. 근거 기록

| 관측 | 값 |
|---|---|
| 실패한 hard gate | Row 0, token 14: batched 5388 대 B=1 15217 |
| Identical-prompts control | `clamped_rounds=0`, `ragged=false`, 네 row 모두 token 14 |
| Arbiter argmax | 5388 (batched token) |
| Top-gap(batched) | 0.0 |
| Top-gap(B=1) | 0.25 |
| 나머지 row | Byte-identical |
| 선택한 margin | `BATCHED_VERIFY_JITTER_MARGIN = 1.0` (측정값의 4배) |
| Margin-removal arm | Margin 0에서 mismatch 4개 중 4개 거부 |

### D. 참고 자료

- `tests/speculative_parity/qwen35_dflash_harness.rs`
- `tests/speculative_parity/qwen35_dflash_batched.rs`
- `tests/speculative_parity.rs`
- `docs/CONTINUOUS_BATCHING.md`
- Issue #2851, issue #2442, issue #2353, issue #2190
