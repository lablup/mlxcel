# 기술 보고서: PR #2843 - perf(gemma4): prefill에서 KV-shared tail을 샘플링된 행으로 자르기

**작성일**: 2026-11-03
**상태**: 리뷰 중 (열린 PR, #2410 종료)
**언어**: Rust, Markdown
**위험도**: 낮음

---

## 요약

Gemma 4 E-series 체크포인트는 마지막 `num_kv_shared_layers`개 레이어를 앞선 레이어에서 빌려온 K/V로 실행하므로, 그 레이어들은 캐시에 아무것도 쓰지 않는다. 한 위치의 logits만 필요한 prefill도 모든 prompt 행을 이 tail에 통과시키고 그중 하나만 남기고 버리고 있었다. 이 PR은 `keep_row: Option<usize>`를 텍스트 모델 레이어 루프에 전달하고, 첫 KV-shared 레이어에서 hidden state, per-layer 입력, 두 attention family의 mask를 샘플링된 행으로 자르며, 공유 레이어의 RoPE 기준을 offset해 그 행이 절대 위치를 유지하게 한다. `mlx-community/gemma-4-e2b-it-4bit`에서 1k prompt TTFT가 2.79x, 4k prompt가 2.83x 낮아졌고, 200개 chunk에서 decided 위치 불일치 0건, trim을 끈 대조군에서도 불일치 0건을 기록했다.

---

## 1. 문제 정의

### 1.1 배경

`num_kv_shared_layers > 0`인 Gemma 4 체크포인트(E2B / E4B 계열)는 `first_kv_shared_layer_idx() = num_hidden_layers - num_kv_shared_layers`에서 디코더 스택을 나눈다. 그 index 앞의 레이어는 K/V를 캐시에 쓰고, 각 attention family의 기준 레이어는 forward 내부의 `shared_kv_store`에도 full-length slab을 게시한다. 나머지 레이어는 KV-shared다. `is_kv_shared_layer`가 참이고, 로드 시점에 `k_proj` / `v_proj` / `k_norm` 가중치가 제거되며, 자체 캐시에 쓰지 않고 빌려온 slab을 읽는다.

prefill에서 샘플링하려면 한 위치의 logits만 있으면 된다. `Gemma4Model::forward_last_with_caches_and_embeddings`는 이미 LM head projection을 그 행으로 제한했지만(issue #672), 방식은 `text_model.forward`를 full prompt 폭으로 실행한 뒤 hidden state를 자르는 것이었다. 따라서 모든 prompt 행이 KV-shared tail 전체를 통과했고, 샘플링된 행만이 반환 logits에 영향을 줄 수 있었음에도 그 비용을 지불했다. tail은 빌려온 K/V에 attention을 수행하고 캐시에 쓰지 않으므로, 나머지 행에는 아무도 읽지 않는 부작용이 없고 오직 연산만 있다.

### 1.2 기존 문제점

- **문제 1**: 한 행 prefill이 tail 비용 전체를 지불했다. #672 최적화는 LM head를 `[B, L, vocab]`에서 `[B, 1, vocab]`으로 줄였지만, transformer tail은 여전히 `L`개 행으로 실행됐다. E-series 계열에서 이는 마지막 `num_kv_shared_layers`개 레이어이며, 1k와 4k prompt에서 측정 가능한 prefill 시간 비중이다.
- **문제 2**: tail을 자를 수 있게 만드는 구조(`first_kv_shared_layer_idx`, `is_kv_shared_layer`, `shared_kv_store`)는 이미 보였지만, 한 행만 필요하다고 forward에 알릴 호출자가 없었기 때문에 결정을 내릴 지점도 없었다.
- **문제 3**: 한 행 prefill을 실행하는 trace arm이 없었다. `examples/logit_trace`는 항상 full-width forward를 채점했으므로, 같은 바이너리에서 동일한 입력으로 trimmed branch와 full-width branch를 비교할 수 없었다.
- **문제 4**: 순진한 trim은 크게 틀리지 않고 조용히 틀린다. RoPE 위치나 query-row mask를 조정하지 않고 행을 버리면 출력은 유창하지만 logits은 이동한다. issue의 원래 f16 허용 오차(`2e-2`)는 그 오차를 잡기에 너무 느슨하다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|------|------|-------------|
| 잘린 행이 잘못된 절대 위치에서 회전 | 높음 | 낮음; 공유 캐시 offset에 `p`를 더하며, offset을 깨면 parity 테스트가 실패 |
| interior 행의 재구성 mask가 잘못된 key를 허용하거나 차단 | 높음 | 낮음; 정확한 재구성이 증명되지 않으면 해당 family가 trim을 거부 |
| capture 또는 speculative 소비자가 좁아진 hidden state를 받음 | 높음 | 낮음; 모든 capture 경로가 guard 집합에 포함 |
| right-padded 배치 prefill이 잘못 trim됨 | 중간 | 낮음; 배치 입력은 제외 |
| full width 대비 작은 수치 드리프트 | 중간 | 낮음; decided 위치 불일치 0건이나, raw top-1은 200개 중 6개 뒤집힘 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

이 변경은 입력 파싱, 외부 데이터 경로, 새 의존성을 추가하지 않는다. 새 환경 변수는 모델이 읽는 boolean A/B 스위치 하나뿐이고, `keep_row`는 사용 전에 `p < l`로 경계가 확인되는 내부 index다. 호출자가 이미 `*_last_*` entry로 라우팅하지 않은 요청에서는 trim에 도달할 수 없다.

**체크리스트:**
- [x] 입력 검증: `keep_row`는 `p < l`, `b == 1`, `l > 1`일 때만 수용
- [ ] 인증/인가: 해당 없음
- [ ] 데이터 암호화: 해당 없음
- [x] 로깅(민감 정보 제외): layer, row, prompt 폭만 debug 한 줄로 기록

**발견된 문제:**

| 문제 | 심각도 | 상태 |
|------|--------|------|
| 이 변경에서 발견된 것 없음 | - | - |

### 2.2 성능 관점

절약되는 연산은 공유 tail 깊이에 비례한다. `first_kv_shared_layer_idx()` 앞의 레이어는 tail이 읽는 캐시와 공유 slab을 채우므로 여전히 full width로 실행된다. 첫 shared 레이어부터는 hidden state가 `[1, 1, hidden]`이 되고, per-layer 입력도 같은 행으로 잘리며, 각 shared 레이어는 `L`개 query 대신 빌려온 key에 대해 query 하나를 실행한다. KV 캐시는 소유한 레이어가 trim 지점 전에 정확히 한 번 갱신한다.

아래 측정은 PR의 근거다. "before" arm은 PR 이전 바이너리가 아니라, 같은 바이너리에서 `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1`을 설정해 tail을 full query 폭으로 유지한 대조군이다. arm당 3회 교차 실행, `mlxcel-bench-decode --prompt-tokens N -n 1 --warmup-tokens 1`. 이 수치에 대한 benchmark CSV는 커밋되지 않았고 PR 본문에만 있다.

**성능 영향:**

| 영역 | Before (같은 바이너리, trim 비활성) | After (trim 활성) | 개선 |
|------|-------------------------------------|-------------------|------|
| TTFT, 1024-token prompt | 211.08 / 211.36 / 211.45 ms | 75.52 / 75.92 / 75.72 ms | 2.79x |
| TTFT, 4096-token prompt | 882.08 / 876.85 / 877.65 ms | 309.30 / 311.62 / 308.98 ms | 2.83x |
| shared tail의 forward 연산 | shared 레이어당 `L`개 query 행 | shared 레이어당 1개 query 행 | tail 깊이에 비례 |
| decode, non-last-row prefill, 배치 prefill | 변화 없음 | 변화 없음 | 설계상 없음 |

### 2.3 호환성/의존성 관점

- **Breaking Changes**: `Gemma4TextModel::forward_with_speculative_sinks`에 열두 번째 인자 `keep_row: Option<usize>`가 추가된다. 저장소 내 모든 호출부는 이 PR에서 갱신되며, 두 `*_last_*` prefill entry를 제외한 모든 호출자는 `None`을 전달한다. 이는 이전 동작과 byte-identical로 문서화돼 있다. 이 public Rust 메서드를 저장소 밖에서 호출한다면 `None`을 추가해야 한다.
- **새 의존성**: 없음.
- **호환성**: trim은 체크포인트에 `num_kv_shared_layers > 0`이 있고, 입력이 단일 시퀀스이며, 행이 둘 이상이고, capture 경로가 활성화되지 않았을 때만 적용된다. 텍스트 전용과 VLM last-row prefill 모두 도달하며, decode, 배치 prefill, speculative decode, MTP verify, capture 실행은 도달하지 않는다.
- **설정**: `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1`이 trim을 비활성화한다. 운영자가 정당하게 설정할 수 있는 런타임 A/B 스위치이지 테스트 전용 knob이 아니므로 `docs/environment-variables.md`에 문서화했다.

### 2.4 코드 품질 관점

- **테스트 커버리지**: `src/models/gemma4_tests.rs`의 `mod kv_shared_tail_trim`에 새 lib 테스트 3개(4-layer KV-shared fixture 포함 476줄). `gemma4_kv_shared_tail` 필터는 3/3 통과, 전체 `gemma4` 필터는 275 passed, 40 ignored.
- **코드 복잡도**: 결정은 레이어 루프 앞의 gate 하나와 루프 안의 commit 블록 하나에 집중된다. mask 도출은 `trimmed_tail_row_masks`, `trimmed_family_row_mask`, `slice_mask_query_row`, `single_query_row_mask`로 분리되며 각각 계약이 좁다.
- **기술 부채**: gate는 제외 조건의 긴 conjunction이다. 이 장황함은 의도적이다. 각 절이 좁아진 폭을 관찰하게 될 소비자 하나에 대응한다. guard를 루프 내부에서 암묵적으로 도출하는 대안은 어떤 경로가 byte-identical로 남는지 보기 어렵게 만든다.
- **검증**: `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings`와 `cargo fmt --check` 통과.

---

## 3. 기술적 선택과 그 이유

### 3.1 forward 이후가 아니라 레이어 루프 안에서 자르기

**컨텍스트:**
호출자는 필요한 행을 이미 알고, 모델은 shared tail이 어디서 시작하는지 이미 안다. 호출자가 full forward 이후에 자를 수도 있지만, 그 시점에는 연산이 이미 소비된 뒤다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| `text_model.forward` 이후 logits 또는 hidden state 자르기 | 모델 변경 없음 | 아무것도 절약하지 못함; tail은 여전히 full width |
| **선택: forward 내부 첫 shared 레이어에서 자르기** | 목표인 tail 연산을 제거 | 레이어 루프에 상태 추가 |
| tail 레이어를 전용 single-row forward로 변경 | 깔끔한 분리 | attention, mask, RoPE 로직 중복 |

**선택 이유:**
tail은 캐시를 바꾸지 않고 좁힐 수 있는 바로 그 영역이다. 해당 레이어들이 캐시 entry를 쓰지 않기 때문이다. 첫 shared 레이어 직전에 trim을 확정하면 같은 루프, attention 코드, RoPE 처리가 두 폭 모두를 담당할 수 있다.

**트레이드오프:**
레이어 루프가 `l`과 함께 `cur_l`, `tail_row_offset`을 들고 다닌다. 로그 라인과 per-layer 입력 도출은 `cur_l`을 사용하므로, 앞으로 루프 안에 추가되는 코드는 올바른 폭 변수를 골라야 한다.

### 3.2 정확한 한 행 mask를 도출하고 아니면 trim 거부

**컨텍스트:**
잘린 query에도 유지된 key에 대한 올바른 mask가 필요하다. 텍스트 모델에는 두 attention family(`full_attention`, `sliding_attention`)와 여러 mask 형태가 있다. plain prefill mask가 있을 수도 있고, dense 경로가 mask를 적용하지 않거나 native windowed 경로가 mask를 생략할 때 `None`일 수도 있다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 항상 행의 절대 위치에서 mask 재구성 | 균일함 | window를 넘은 sliding family에서는 유지 key가 이미 clamp돼 있어 틀림 |
| 항상 기존 mask를 slice | mask가 있을 때 정확 | mask가 `None`이면 panic 또는 fallback |
| **선택: 있으면 slice, key가 full length일 때만 재구성, 그 외 마지막 행, 의심스러우면 거부** | 수용하는 모든 경우에 정확 | window를 넘은 interior 행은 tail을 trim하지 않음 |

**선택 이유:**
`trimmed_family_row_mask`는 family에 plain mask가 있으면 `Some(Some(sliced))`를 반환한다. causal 또는 windowed mask의 `p` 행은 그 query가 볼 수 있는 key를 이미 정확히 표현하기 때문이다. plain mask가 없으면 마지막 행에 대해 `Some(None)`을 반환한다. 유지된 key에 대한 한 행 causal 또는 windowed attention은 untrimmed 경로가 어차피 도달할 decode 형태이기 때문이다. interior 행에 대해서는 key가 full length일 때만(`global_live == global_offset`, sliding family는 `sliding_live + query_len <= sliding_window`) 한 행 mask를 재구성하므로 column `k`가 여전히 논리 key 위치 `k`에 대응한다. 그 밖의 조합은 `None`을 반환하고 호출자는 tail 전체를 trim하지 않는다.

**트레이드오프:**
안전하게 자를 수 있는 경우에도 trim을 하지 않을 때가 있다. 이는 의도적이다. 놓친 최적화는 지연 비용이고, 잘못된 mask는 정확성 비용이다. debug 로그가 두 결과를 구분한다.

### 3.3 공유 캐시 offset으로 절대 위치 보존

**컨텍스트:**
잘린 행은 query index `p`를 절대 위치로 유지하지만, shared 레이어는 일반 캐시 offset을 쓰지 않고 `shared_kv_store`의 기준 레이어 offset을 쓴다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| 잘린 query를 offset 0에서 회전하고 상대 위치에 의존 | offset 변경 없음 | 이 경로의 Gemma 4 RoPE는 절대적이며, 행이 첫 토큰인 것처럼 attend함 |
| 빌려온 K/V를 새 원점으로 재회전 | 개념상 단순 | 매번 공유 slab을 변경하거나 복사 |
| **선택: 공유 캐시 offset을 `ref_offset + p`로 설정** | 덧셈 하나; slab을 건드리지 않음 | 추적할 offset 변수 하나 추가 |

**선택 이유:**
shared 레이어는 `cache.set_offset(...)`에서 query 위치를 얻는다. `tail_row_offset`을 더하면 남은 query 하나가 원래 절대 위치에서 회전하고, 빌려온 key는 자기 위치를 유지한다. 테스트는 이 offset을 의도적으로 깨서 5.7e-4의 logit 이동을 관찰했고, 이는 1e-4 parity 경계를 실패시킨다.

**트레이드오프:**
`tail_row_offset`은 trim이 확정된 뒤에만 0이 아니다. 그 이전 레이어와 다른 모든 호출자는 0을 보므로 변경이 no-op이다.

### 3.4 명시적 guard 집합으로 비대상 경로를 byte-identical하게 유지

**컨텍스트:**
같은 forward가 decode, 배치 prefill, speculative verify, MTP, hidden capture, shared-K/V capture, 명시적 per-layer capture를 모두 담당한다. 좁아진 hidden state는 full prompt 폭을 기대하는 소비자를 깨뜨린다.

**선택 이유:**
`trim_row`는 다음이 모두 참일 때만 `Some`이다: 환경으로 trim이 비활성화되지 않음, `num_kv_shared_layers > 0`, `first_kv_shared_layer_idx() > 0`, `b == 1`, `l > 1`, `p < l`, speculative sink 없음, `capture_layer_ids`가 없거나 비어 있음, padding 없음, `per_row_valid_end`, `row_offsets`, `tree_positions`가 모두 `None`. commit 시점에는 추가로 레이어가 KV-shared이고, `kv_shared_layer_index`가 이미 `shared_kv_store`에 있으며, 두 family의 mask가 모두 도출됐어야 한다. 하나라도 실패하면 `cur_l == l`로 이전 동작이 유지된다.

**트레이드오프:**
조건 목록이 길고, gate와 capture 설정 사이에 "capture" 개념이 중복된다. speculative 경로의 조용한 폭 불일치에 비하면 그 중복은 싸다.

### 3.5 단일 바이너리 A/B 스위치와 trace arm

**컨텍스트:**
trim은 full-width 경로 대비 수치적으로 중립이어야 한다. 두 바이너리로 증명하면 빌드 간 차이가 끼고, 불일치가 생겼을 때 원인이 모호해진다.

**고려한 대안:**

| 대안 | 장점 | 단점 |
|------|------|------|
| PR 이전 빌드와 비교 | 예전 동작에 가장 가까움 | 빌드 둘, 환경 둘, 수치가 다를 때 모호함 |
| **선택: 같은 바이너리에서 `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1`** | 바이너리 하나, 동일 가중치와 커널; branch만 분리 | 문서로 유지해야 하는 런타임 스위치 추가 |
| 테스트 전용 feature flag | 프로덕션 스위치 없음 | trace arm이 일반 릴리스 빌드를 실행 |

**선택 이유:**
이 변수는 코드를 되돌리지 않고 새 branch를 끄므로, trimmed arm과 untrimmed arm은 tail에 공급되는 행에서만 다르다. `examples/logit_trace`에 `MLXCEL_TRACE_LAST_ROW=1`이 추가되어 각 chunk의 마지막 위치를 `forward_last_logits`로 채점하고, full-width arm이 쓰는 것과 같은 `(chunk, pos)`에 출력하므로 `scripts/compare_logit_traces.py`가 직접 비교한다. 그 arm에서 kill switch를 설정하면 대조군이 된다. `RUST_LOG=mlxcel::models::gemma4=debug` 로그가 trimmed branch 진입을 보고하므로, 도달하지 않은 실행이 드러난다.

**트레이드오프:**
실험을 위한 스위치가 프로덕션 코드에 남는다. 서빙 모드가 아니라 A/B 대조군으로 문서화돼 있다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before]
forward_last_logits* ─> forward_last_with_caches_and_embeddings
                          └─> text_model.forward (full width L)
                                └─> KV-shared tail을 포함한 모든 레이어가 L개 행 실행
                          └─> logits_at(hidden, last_pos)

[After]
forward_last_logits* ─> forward_last_with_caches_and_embeddings(Some(last_pos))
                          └─> forward_with_speculative_sinks(keep_row = Some(p))
                                ├─> 0 .. first_kv_shared_layer_idx 레이어: full width L
                                │     (캐시와 shared_kv_store 채움)
                                ├─> 첫 KV-shared 레이어: h, per-layer 입력,
                                │     global/sliding mask를 p 행으로 slice; cur_l = 1;
                                │     tail_row_offset = p
                                └─> shared tail: ref_offset + p에서 query 1행
                          └─> hidden 폭 1이면 logits_at(hidden, 0), 아니면 logits_at(hidden, last_pos)
```

### 4.2 주요 코드 변경

**파일: `src/models/gemma4.rs`**

trim 가능 여부를 정하는 gate. 레이어 루프 전에 한 번 평가한다:

```rust
let trim_row: Option<i32> = keep_row
    .filter(|_| !kv_shared_tail_trim_disabled())
    .filter(|&p| {
        self.config.num_kv_shared_layers > 0
            && self.config.first_kv_shared_layer_idx() > 0
            && b == 1
            && l > 1
            && p < l as usize
            && sinks.is_none()
            && capture_layer_ids.is_none_or(|ids| ids.is_empty())
            && !has_padding
            && per_row_valid_end.is_none()
            && row_offsets.is_none()
            && tree_positions.is_none()
    })
    .map(|p| p as i32);
```

루프가 `caches`의 mutable borrow를 들고 있으므로 캐시 geometry는 루프 전에 샘플링한다. commit은 첫 shared 레이어에서, 그 레이어의 입력이 도출되기 전에 일어난다:

```rust
if !tail_trimmed
    && i == self.config.first_kv_shared_layer_idx()
    && layer.self_attn.is_kv_shared_layer
    && layer.self_attn.kv_shared_layer_index
        .is_some_and(|ref_idx| shared_kv_store.contains_key(&ref_idx))
    && let (Some(p), Some(geometry)) = (trim_row, trim_geometry)
    && let Some((global_row, sliding_row)) = self.trimmed_tail_row_masks(...)
{
    h = slice_axis(&h, 1, p, p + 1);
    if let Some(inputs) = per_layer_inputs.as_mut() {
        *inputs = slice_axis(inputs, 1, p, p + 1);
    }
    global_mask = global_row;
    sliding_mask = sliding_row;
    cur_l = 1;
    tail_row_offset = p;
    tail_trimmed = true;
}
```

shared 레이어는 게시된 offset에 유지된 행의 위치를 더해 읽는다:

```rust
cache.set_offset(*ref_offset + tail_row_offset);
```

trim을 정확하게 만드는 부분은 mask 계약이다:

```rust
fn trimmed_family_row_mask(
    mask: Option<&MlxArray>,
    q_pos: i32,
    key_len: i32,
    full_length_keys: bool,
    p: i32,
    query_len: i32,
) -> Option<Option<UniquePtr<MlxArray>>> {
    if let Some(mask) = mask {
        return Some(Some(slice_mask_query_row(mask, p)));
    }
    if p + 1 == query_len {
        return Some(None);
    }
    if !full_length_keys || key_len <= 0 {
        return None;
    }
    Some(Some(single_query_row_mask(q_pos, key_len, None)))
}
```

`trimmed_tail_row_masks`는 실제 레이어가 있는 각 family에 대해 이를 호출한다. global family에는 `global_offset + p`, `global_live + query_len`을, sliding family에는 `sliding_offset + p`, `sliding_live + query_len`과 window 검사를 사용한다. 둘 중 하나라도 `None`이면 tail 전체의 trim이 중단된다.

**파일: `src/models/gemma4.rs` (`forward_last_with_caches_and_embeddings`)**

```rust
let hidden = self.text_model.forward_with_speculative_sinks(
    input_ids, input_embeddings, caches, mask, per_layer_inputs,
    None, None, false, None, None, None,
    Some(last_pos),
);
let row = if mlxcel_core::array_shape(&hidden)[1] == 1 { 0 } else { last_pos };
self.logits_at(&hidden, row)
```

**변경 이유:** 호출자가 더 이상 `text_model.forward`를 실행한 뒤 자르지 않는다. 한 행을 직접 요청하고, tail이 실제로 trim됐으면 row 0을, 아니면 `last_pos`를 읽는다.

**파일: `examples/logit_trace.rs`**

`MLXCEL_TRACE_LAST_ROW=1`이 한 행 arm을 선택한다. 표준 텍스트 arm 전용이며 `MLXCEL_TRACE_GEMMA4`, audio, batched rows, tensor parallel, image 입력, contiguous tracing과의 조합을 거부한다. 각 chunk는 `input_len - 1`에서 `forward_last_logits`로 채점되고 full-width arm과 같은 `(chunk, pos)`에 출력된다.

**파일: `docs/environment-variables.md`**

`MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM`을 Gemma 4 A/B 스위치로 문서화한다. 범위도 포함한다: `*_last_*` prefill entry만 trim하고, `num_kv_shared_layers > 0`일 때만 적용된다.

### 4.3 데이터 모델 변경

없음. 체크포인트 형식, config 필드, 직렬화 구조는 바뀌지 않는다.

---

## 5. 학습 포인트

### 5.1 KV-shared 레이어는 prefill tail을 자를 수 있게 한다

**개념:**
KV-shared 레이어는 앞선 레이어에서 key와 value를 빌려오고 자체 캐시 entry를 쓰지 않는다. 출력은 빌려온 slab과 입력 행의 함수지만, downstream에서 그 행들을 캐시로 읽는 곳은 없다.

**이 PR에서의 적용:**
마지막 `num_kv_shared_layers`개 레이어가 바로 prompt 행을 버려도 부작용이 없는 영역이다. 단, 유지하는 행의 RoPE와 mask가 올바르다는 조건에서다. split 앞의 레이어는 캐시와 공유 slab을 채우는 주체이므로 여전히 full width로 실행된다.

**일반적인 사용 사례:**
- cross-layer KV sharing 아키텍처에서 한 위치만 샘플링하는 prefill(`logits_to_keep = 1`).
- 메모리 제한 LM head projection. 이 PR이 #672 패턴을 head에서 transformer tail로 확장한 것이다.

### 5.2 한 query 행에 대한 정확한 mask 재구성

**개념:**
`[queries, keys]` additive mask는 key 축을 건드리지 않고 query 축에서 slice할 수 있다. `p` 행이 query `p`가 볼 수 있는 key를 이미 표현하기 때문이다. mask가 없으면 query의 논리 위치와 캐시 geometry로 행을 재구성해야 하며, 이 재구성은 유지된 key가 논리 위치에 일대일로 대응할 때만 유효하다.

**이 PR에서의 적용:**
`slice_mask_query_row`가 mask-present 경우를 처리한다. `single_query_row_mask`는 `arange(key_len) <= q_pos`로 `0 / -inf` 행을 재구성하고, sliding window가 있으면 아래 경계를 추가한다. `p + 1 == query_len` 특수 경우는 mask를 아예 반환하지 않는다. 유지된 key에 대한 한 행 causal attention은 이미 decode 형태이기 때문이다.

**일반적인 사용 사례:**
- full key history를 유지하면서 prefill의 query 축을 좁히는 모든 최적화.
- 유지된 key의 좌표계가 논리적인지 clamp된 것인지 확인해 mask 버그를 진단하는 작업.

### 5.3 민감하게 설계한 fixture로 수치 중립성 테스트

**개념:**
parity 테스트는 fixture가 실패할 수 있을 때만 의미가 있다. 여기서는 두 실패 모드가 중요하다. 가중치 엔트로피가 너무 낮으면 attention이 위치에 둔감해지고, 허용 오차가 느슨하면 작지만 체계적인 오류를 숨긴다.

**이 PR에서의 적용:**
fixture는 선형 합동 생성기 기반의 결정적 pseudo-random 가중치를 쓴다. 선형 ramp는 2-D 테스트 head 안에서 key 벡터를 거의 평행하게 만들어 깨진 RoPE가 통과할 수 있기 때문이다. parity 경계는 issue의 `2e-2` f16 허용 오차가 아니라 1e-4다. 깨진 RoPE offset은 logits을 5.7e-4, 깨진 mask 행은 9.9e-4만큼 이동한다. 두 변이를 실제로 만들어 봤고 둘 다 1e-4에서 실패한다. 올바른 trim은 f32 참조 행을 정확히 재현한다. 테스트는 trimmed hidden 폭이 1인지도 확인하므로, 조용한 fallback이 수치 비교를 무의미하게 만들 수 없다.

**일반적인 사용 사례:**
- 참조 경로 대비 정확해야 하는 최적화 검증.
- 실제 체크포인트를 로드하지 않고 모든 branch를 실행하는 최소 아키텍처 fixture 구성.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 의미 |
|--------|------|------------------|
| KV-shared 레이어 | 자체 K/V를 계산·캐시하지 않고 앞선 레이어에서 읽는 디코더 레이어 | 마지막 `num_kv_shared_layers`개 레이어가 이 PR이 자르는 영역 |
| `keep_row` | `forward_with_speculative_sinks`에 전달되는 선택적 샘플 행 index | "이 행의 logits만 필요하다"를 표현하는 유일한 새 인자 |
| RoPE 기준 offset | query를 회전할 때 쓰는 절대 위치 | `p`를 더해 잘린 행을 원래 위치에 유지 |
| Additive attention mask | attention score에 더하는 `0 / -inf` f32 mask | 한 query 행으로 slice하거나 재구성해야 함 |
| `live_len` vs `offset` | `offset`은 RoPE용으로 단조 증가하고, `live_len`은 캐시가 실제 반환하는 이전 key 수 | mask 크기는 live length로 정해야 하므로 geometry tuple이 둘 다 샘플링 |
| Decided 위치 | 참조 top-two gap이 `--decided` threshold 이상인 trace 위치 | 0/104 불일치가 parity를 판정하는 지표 |

### 관련 기술/프레임워크

- **MLX**: 런타임이 사용하는 Apple의 배열 프레임워크.
  - 공식 문서: https://ml-explore.github.io/mlx/
- **mlx-vlm**: Gemma 4 language model이 `logits_to_keep`으로 trim하는 upstream 참조 구현.
  - https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py
- **Gemma 4**: E-series 체크포인트에 cross-layer KV sharing이 있는 모델 family.

### 관련 PR/이슈

- Issue #2410: 이 PR이 종료하는 성능 이슈. `keep_row` 인자, mask와 RoPE 처리, 제외 집합, 그리고 수용에 필요한 trace와 TTFT 근거를 제안했다.
- PR #672: LM head projection을 한 위치로 제한한 이전 변경. 이 PR은 같은 아이디어를 head에서 transformer tail로 확장한다.
- PR #2842: 이 저장소의 무관한 최근 보고서이며 형식 참고로만 사용했다.
- Upstream `Blaizzy/mlx-vlm#2157`: 대응하는 upstream trim. hidden-state capture가 요청되면 full width를 유지한다.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 4 |
| 추가 라인 | +810 |
| 삭제 라인 | -25 |
| 추가 테스트 | lib 테스트 3개 |
| 커밋 | 2 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|----------|-----|------|
| 성능 | 1 | `src/models/gemma4.rs`에서 KV-shared tail을 샘플 행으로 trim |
| 테스트 | 1 | KV-shared fixture와 parity, 폭, capture guard 테스트 |
| 도구 | 1 | `examples/logit_trace.rs`의 `MLXCEL_TRACE_LAST_ROW=1` 한 행 arm |
| 문서 | 1 | `docs/environment-variables.md`의 `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM` |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `5105d8e4` | perf | perf(gemma4): trim the KV-shared tail to the sampled row in prefill |
| `9febc6a1` | docs | docs: document the Gemma 4 KV-shared tail trim kill switch |

---

## 8. 후속 조치

### 완료 필요

- [ ] 머지를 막는 항목 없음. PR은 열려 있고 #2410을 종료한다.

### 모니터링 필요

- 머지 후 KV-shared Gemma 4 체크포인트의 1k, 4k prompt TTFT를 관찰한다. 측정값은 M3 Ultra 한 대에서 나왔고 benchmark CSV로 커밋되지 않았다.
- tail을 바꾸는 향후 변경에는 수치 parity gate를 염두에 둔다. 200-chunk trace는 decided 위치 불일치 0/104, raw top-1은 6/200 뒤집힘이었고 모두 참조 top-two gap 0.13 미만이었다.
- trim이 기대되는 실행은 `RUST_LOG=mlxcel::models::gemma4=debug`에서 `trimmed the KV-shared tail to the sampled row layer=... keep_row=... prompt_rows=...`를 기록해야 한다. 이 로그가 없으면 guard 집합이 trim을 거부한 것이다.

### 향후 개선 사항

- right-padded 배치 prefill은 명시적으로 범위 밖이다. 지원하려면 행별 keep index와 padding이 shared tail에 들어가지 않는다는 확인이 필요하다.
- sliding window를 넘은 interior 행은 현재 sliding family에서 trim을 거부해 tail 전체가 untrimmed로 남는다. clamped-key 재구성으로 그 경우까지 최적화를 확장할 수 있다.
- TTFT 수치는 커밋된 산출물로 재현할 수 없다. 회귀 gate로 삼는다면 프로젝트 benchmark 문서 규칙에 따라 결과를 커밋해야 한다.
- capture 경로는 의도적으로 full width를 유지한다. 향후 소비자가 좁아진 hidden state를 받아들일 수 있다면 그 소비자에 한해 guard를 완화할 수 있다.

---

## 부록

### A. 테스트 결과

| 명령 | 결과 |
|------|------|
| `cargo test --profile test-fast -p mlxcel --lib gemma4_kv_shared_tail` | 새 테스트 3개 통과 |
| `cargo test --profile test-fast -p mlxcel --lib gemma4` | 275 passed, 40 ignored |
| `cargo clippy --profile test-fast -p mlxcel --lib --tests -- -D warnings` | 통과 |
| `cargo fmt --check` | 통과 |

세 테스트:

- `gemma4_kv_shared_tail_trim_matches_full_width_rows`: `num_kv_shared_layers = 2`, per-layer 입력 활성화, 결정적 12-token의 4-layer fixture. 세 geometry(window 안의 6-token cold prefill, window를 넘는 12-token cold prefill, 4-token warm prefix + 8-token chunk)를 각각 마지막 행과 interior 행에서 검사한다. trimmed hidden 폭이 1인지, logits이 full-width 행과 1e-4 이내로 일치하는지 확인하며, 두 family 모두 plain mask가 없는 interior 경우도 포함한다.
- `gemma4_kv_shared_tail_trim_narrows_the_hidden_state`: `keep_row = Some(5)`는 폭 1, `keep_row = None`은 폭 12를 반환한다.
- `gemma4_kv_shared_tail_trim_skipped_when_capturing_layers`: `capture_layer_ids`가 설정되면 반환 hidden state와 capture된 레이어 출력 모두 마지막 행과 interior 행에서 폭 12를 유지한다.

PR에 보고된 변이 검사: RoPE 행 offset을 깨면 logits이 5.7e-4, mask 행을 깨면 9.9e-4 이동하며 둘 다 1e-4 parity 경계를 실패한다. issue의 2e-2 f16 허용 오차로는 둘 다 잡지 못했을 것이다.

### B. 성능 벤치마크

환경: M3 Ultra 512 GB, macOS 26.6.2, base commit `afb208eb`, MLX `7ca40033132dae444d45f7fc3daae57e24c13f6c`, 모델 `mlx-community/gemma-4-e2b-it-4bit`.

Trace 비교, 200 chunks x 256 tokens, `scripts/compare_logit_traces.py --decided 2`:

| arm | top-1 불일치 | decided (gap >= 2) | 불일치 시 최대 gap |
|---|---|---|---|
| `MLXCEL_TRACE_LAST_ROW=1` vs full width | 6 / 200 | 0 / 104 | 0.125 |
| 같은 arm, trim 비활성 | 0 / 200 | 0 / 104 | 없음 |

6개 불일치는 모두 참조 top-two gap이 0.13 미만이었고 runner-up으로 떨어졌다. logit delta p50 0.109, p99 0.500으로 4-bit per-row-vs-batched 커널 반올림 계열이다. Perplexity는 88.82에서 88.42로 이동했다. `RUST_LOG=mlxcel::models::gemma4=debug`가 모든 chunk에서 trimmed branch를 확인했다: `trimmed the KV-shared tail to the sampled row layer=15 keep_row=256 prompt_rows=257`.

TTFT, `mlxcel-bench-decode --prompt-tokens N -n 1 --warmup-tokens 1`, arm당 3회 교차 실행. 대조군은 같은 바이너리에서 `MLXCEL_GEMMA4_NO_KV_SHARED_TAIL_TRIM=1`을 설정한다. PR 이전 바이너리 측정은 없었다.

| prompt | trim on (ms) | trim off (ms) | 배속 |
|---|---|---|---|
| 1024 | 75.52 / 75.92 / 75.72 | 211.08 / 211.36 / 211.45 | 2.79x |
| 4096 | 309.30 / 311.62 / 308.98 | 882.08 / 876.85 / 877.65 | 2.83x |

benchmark CSV는 커밋되지 않았다. PR은 네 파일을 변경하며 그중 benchmark 결과 파일은 없다.

### C. 참고 자료

- Issue #2410: perf: Trim the Gemma 4 KV-shared tail to the sampled row in prefill
- PR #2843: 이 보고서가 다루는 변경
- Upstream `Blaizzy/mlx-vlm#2157`과 `mlx_vlm/models/gemma4/language.py` (`logits_to_keep`)
- `docs/environment-variables.md`: 문서화된 A/B 스위치
- `scripts/compare_logit_traces.py`: trace 근거에 사용한 decided-position 비교
