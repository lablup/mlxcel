# 기술 보고서: PR #2865 - perf(speculative): commit per-row accepts in batched DFlash

**작성일**: 2026-11-06
**상태**: 검토 중 (열린 PR, #2442 종료)
**언어**: Rust, Markdown
**위험도**: 중간

---

## 요약

PR #2865는 이슈 #2442의 항목 1을 구현한다. 배치 Qwen 3.5 DFlash 라운드 루프가 라운드 전체를 가장 짧은 live row의 accept로 clamp하지 않고, 각 row의 speculative walk accept 수를 그대로 commit한다. Gemma 4의 `DivergentVerifyRows` 어텐션 레이아웃을 재사용해 각 row의 attention KV를 연속적인 valid prefix로 압축하고, stale gap을 마스킹하고, 각 row의 논리 위치에서 query와 key를 회전시키고, 각 GDN 레이어를 해당 row의 accepted prefix로 replay한다. M3 Ultra에서 `qwen3.5-4b-4bit`와 `qwen3.5-4b-dflash`, B = 4 기준으로 aggregate decode가 83.06에서 116.05 tok/s로 상승했고(1.397x), row-round당 committed accept가 0.323에서 1.036으로, 라운드 수가 96에서 67로 줄었으며, 보정된 parity gate는 per-row accepts를 기본값으로 2개 중 2개 통과했다. drafter는 여전히 공유된 직사각형 prefix를 받으므로 speculative walk 자체는 2.938에서 1.036으로 감소한다. per-row drafter context는 명시적으로 범위 밖이며, 서버 수준 동시 요청 smoke 실행은 수행되지 않았다.

---

## 1. 문제 정의

### 1.1 배경

Qwen 3.5의 배치 DFlash 라운드 루프는 모든 row가 하나의 직사각형 target cache를 공유한다. 이슈 #2353은 walk가 갈라질 때 라운드를 가장 짧은 live row의 accept로 clamp했고(`clamp_uniform_acceptance`), 따라서 모든 row가 같은 수의 verify 위치를 commit했다. 직사각형 cache에서는 이것이 올바른 처리였다. 거부된 tail을 0으로 채우는 것은 대안이 될 수 없기 때문이다. 0인 key도 `q . 0 = 0`으로 점수화되어 softmax에서 `exp(0)` 가중치를 차지하고, 이후 모든 스텝에서 해당 row의 어텐션을 희석한다.

비용은 accept 통계에 드러났다. #2442 측정에서 walk는 row-round당 평균 2.938이었지만 clamp가 commit한 값은 0.323이었고, 96개 라운드 중 93개가 clamp되었다. 그래서 배치가 B = 1 루프 대비 얻은 이득은 약 1.06x(83.06 대 78.09 tok/s)에 그쳤고, 이는 walk가 가진 잠재력보다 훨씬 낮았다.

이슈 #2442 항목 1은 per-row accepts를 요구했다. 이어서 PR #2853은 Metal 커널 디스패치 아래에서 배치 verify forward가 B = 1 forward와 바이트 단위로 동일하지 않다는 것을 확인하고, 하드 바이트 gate를 보정된 1.0-logit near-tie margin으로 대체했다. 이로써 선행 조건이 정리되었다. parity divergence의 원인은 accept 정책이 아니므로, per-row accepts는 기존 margin gate 아래에서 진행할 수 있었다.

### 1.2 기존 문제점

- **clamp는 이미 받아들인 작업을 버린다.** 매 라운드가 live row 중 최소 accept를 commit하므로, 더 멀리 walk한 row의 추가 accepted 토큰은 rollback된다.
- **거부된 tail을 0으로 채우는 것은 유효한 대안이 아니다.** stale key가 softmax에서 `exp(0)` 가중치로 계속 참여한다.
- **직사각형 cache에는 하나의 offset만 있다.** 서로 다른 수를 commit한 row들이 같은 물리적 cache 할당을 공유하면서 서로 다른 연속 prefix를 유지할 방법이 필요하다.
- **linear-attention 상태는 row별이지만 유효 길이는 그렇지 않다.** `max(accepted)`로 한 번에 배치 replay하면 짧은 row가 거부한 위치까지 진행하게 된다.
- **배치 이득이 거의 없었다.** row-round당 committed accept 0.323에서 B = 4는 B = 1 대비 약 1.06x에 머물렀다.

### 1.3 위험성

| 위험 | 영향 | 발생 가능성 |
|---|---|---|
| ragged cache 레이아웃이 이후 어텐션이나 GDN 상태를 오염시킨다 | 높음 (잘못된 토큰) | 낮음: parity gate가 per-row accepts를 기본값으로 실행되고, 압축은 단위 테스트로 검증된다 |
| 지원하지 않는 cache 모드가 per-row 경로에 도달한다 | 중간 (panic 또는 오염) | 낮음: `row_layout_supported`가 dense Fp16, 비-paged, 비-head-trimmed cache로 제한하고 압축도 같은 전제를 assert한다 |
| per-row accepts가 실제 서버 동시성과 잘 맞지 않는다 | 중간 (서빙 정확성) | 중간: 서버 수준 동시 요청 smoke 실행이 없다 |
| 호스트 부하가 throughput 비교를 왜곡한다 | 낮음 (판단 품질) | 낮음: 두 arm이 같은 경쟁 빌드 부하 아래 교대 실행되었고 clamp와 B = 1 셀이 이전 quiet-box 기록을 재현한다 |
| walk 감소를 회귀로 오해한다 | 낮음 (원인 오귀속) | 기록하지 않으면 확실: drafter context는 의도적으로 직사각형으로 유지되며, per-row drafter context는 별도 변경이다 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

네트워크, 인증, 입력 파싱, 직렬화 표면의 변경은 없다. 새로 추가된 설정 입력은 `MLXCEL_DFLASH_PER_ROW_ACCEPTS` 환경 변수뿐이며, generator 생성 시 한 번 읽고 on/off 값만 받는다. 사용자 데이터를 기록하지 않는다. `dflash_target.rs`가 남기는 accept 통계 로그는 기존 것이고 설명 문구만 갱신되었다. rollback 경로는 예상 밖 레이아웃으로 진행하지 않고 cache shape, mode, paging, head trim을 assertion으로 검증한다.

### 2.2 성능 관점

M3 Ultra, `qwen3.5-4b-4bit` target과 `qwen3.5-4b-dflash`, block 16, row당 128 토큰, temperature 0, arm당 세 번의 교대 실행으로 측정했다.

| 지표 | Clamp | Per-row | 변화 |
|---|---:|---:|---:|
| Aggregate throughput (tok/s) | 83.06 | 116.05 | 1.397x |
| B = 1 기준 (tok/s) | 78.09 | 78.09 | 1.00x |
| 라운드 수 | 96 | 67 | 30% 감소 |
| clamp된 라운드 | 93 | 0 | 제거 |
| row-round당 committed accept | 0.323 | 1.036 | 3.2x |
| row-round당 walk | 2.938 | 1.036 | 0.35x |

per-row arm은 B = 1 루프의 1.49x이고, clamp arm은 1.06x이다. Throughput은 prefill을 제외한 `DFlashBatchedRunOutput::stats`의 라운드 루프 wall clock이므로, attention 레이어마다 row별로 수행하는 `copy`와 `slice_update` 압축 비용이 측정값에 포함되어 있다. clamp 세 실행의 편차는 0.2%, per-row 세 실행의 편차는 0.36%였다.

### 2.3 호환성/의존성 관점

- **Breaking Changes**: 없음. 새 `SpeculativeTarget` 메서드는 이전 동작을 보존하는 기본 구현을 가진다.
- **New Dependencies**: 없음.
- **호환성**: per-row accepts는 target이 지원을 보고하고 cache가 head trim 없는 dense Fp16이며 paged가 아닐 때만 기본값이 된다. quantized, paged, head-trimmed cache는 자동으로 uniform clamp를 선택한다. `MLXCEL_DFLASH_PER_ROW_ACCEPTS=0`(또는 `false`, `no`, `off`)은 A/B 귀속을 위해 clamp를 복원한다.
- **내부 시그니처 변경**: `Qwen3NextAttention::forward_with_position_ids_verify`와 그 hidden-state 형제 함수에 `Option<&[i32]>` row-layout 인자가 추가되었다. 둘 다 `pub(crate)`이고 모든 호출 지점이 갱신되었다.
- **VLM**: `Qwen35VLModel`은 capability, offset, verify, rollback hook을 text model에 위임한다. attention 레이어는 MRoPE position id가 제공되면 per-row verify를 거부한다. DFlash verify 경로는 position id를 전달하지 않으므로 이는 서빙 경로가 아니라 방어적 가드이다.

### 2.4 코드 품질 관점

단위 테스트 3개가 추가되었다. `models::qwen3_5_tests`의 attention 압축 테스트 2개(row block을 자신의 valid end로 이동, 모든 row가 offset에서 끝나면 no-op)와 `mlxcel-core::utils`의 `build_divergent_row_mask` 테스트 1개(history는 참여, stale gap은 차단, 새 block은 각 row의 논리 위치에서 causal)이다. `models::qwen3_5_tests` 스위트는 이제 30개 테스트이고, DFlash 라운드 루프 모듈은 `round_loop.rs` 15개, `round_loop_batched.rs` 13개를 유지한다. mask builder는 `gemma4.rs`에서 `mlxcel-core::utils`로 이동해 Gemma 4의 약 30줄 중복을 제거했고 두 모델이 하나의 구현을 공유한다. `cargo fmt --all -- --check`와 `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`는 clean이다.

---

## 3. 기술적 선택과 그 이유

### 3.1 target capability와 인스턴스 cache 검사로 per-row accepts를 gate

**컨텍스트:**
per-row accepts는 attention cache를 row별로 압축할 수 있고 linear-attention 상태를 row별로 replay할 수 있는 target을 요구한다. 모든 `SpeculativeTarget`이 그럴 수 있는 것은 아니며, Qwen 3.5도 모든 cache 모드에서 가능한 것은 아니다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|---|---|---|
| 모든 target에 per-row 경로를 항상 적용 | 루프가 단순 | uniform rollback만 지원하는 target을 깨뜨리고 quantized/paged cache에서 panic |
| 모든 target이 새 hook을 구현하도록 강제 | 동작이 균일 | 무관한 target과 테스트 더블에 no-op 구현을 강요 |
| **선택: 기본 trait 메서드와 2단계 gate** | 기존 target은 변경 없음, 미지원 cache는 안전하게 fallback | 서빙 경로가 런타임 cache 상태에 의존 |

**선택 이유:**
`supports_per_row_accepts()`는 family 수준 답이고 기본값이 `false`이다. `row_layout_supported(caches)`는 인스턴스 수준 답이며 `KVCacheMode::Fp16`, `!is_paged_backed()`, `offset - live_len() == 0`(head trim 없음)을 검사한다. 루프는 둘 다 참이고 호출자가 비활성화하지 않았을 때만 per-row 경로를 탄다. 기본 `rollback_partial_batched_rows`와 `verify_forward_with_row_layout`은 uniform hook에 위임하며, 이는 모든 row가 offset을 공유할 때만 올바르므로 override를 잊은 target은 ragged 레이아웃을 조용히 만들지 않고 안전하게 실패한다.

**트레이드오프:**
프로덕션 동작이 런타임에 선택된 cache 모드에 의존하므로, quantized 배포는 조용히 기존 clamp를 사용하고 1.397x를 얻지 못한다. 대안은 burst를 panic 또는 거부하는 것이었지만, 이는 성능 기능을 가용성 회귀로 바꾼다.

### 3.2 두 번째 레이아웃을 만들지 않고 Gemma 4 divergent-row 레이아웃을 재사용

**컨텍스트:**
Gemma 4 MTP verifier는 이미 논리적 valid end가 다른 row들을 공유 물리 cache에 대해 처리한다. 그 `build_divergent_verify_mask`는 ragged DFlash 라운드에 필요한 세 영역, 즉 row의 valid history, stale gap, 새 block을 정확히 표현한다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|---|---|---|
| 모든 row를 `max(accepted)`로 패딩하고 거부 위치를 0으로 채움 | 압축 불필요 | 0 key가 여전히 softmax 가중치를 차지하며 #2353에서 이미 기각됨 |
| cache를 row별로 복사하고 독립 forward 실행 | 개념적으로 단순 | 배치 이득을 없애고 row마다 cache 메모리를 중복 |
| **선택: 공유 divergent-row mask와 per-row 회전** | 하나의 mask가 세 영역을 모두 표현하고 검증된 레이아웃을 재사용 | mask 생성 비용이 `O(B * l * (offset + l))` |
| DFlash 전용 mask를 새로 구현 | 모델 간 결합 없음 | 정확성을 유지해야 할 두 번째 구현 |

**선택 이유:**
`build_divergent_row_mask(l, offset, per_row_valid_end, left_padding, window)`는 `[B, 1, l, offset + l]` additive mask를 반환한다. row `r`은 valid end 아래 key를 허용하고, `[valid_end, offset)`을 차단하며, 새 key `t`를 논리 위치 `valid_end + t`로 취급하므로 query `j`는 새 key `t <= j`만 본다. `Qwen3NextAttention`은 `target_verify`, `l > 1`, 그리고 어떤 end가 공유 offset과 다를 때 `divergent_rows`를 감지하고, row 자신의 시작 위치에서 RoPE를 적용한 뒤 `attend_per_position_rows`를 실행한다. 이 함수는 query `j`를 정확한 key 폭 `offset + j + 1`과 해당 mask 슬라이스에 대해 계산한다. Gemma 4 함수는 이제 같은 core utility에 위임하므로 두 모델이 어긋날 수 없다.

**트레이드오프:**
mask는 verify forward마다 생성되고 명시적으로 materialize된다. 이 비용은 측정된 throughput에 포함되어 있고, 측정된 shape에서는 accept 이득이 이를 상회한다. 향후 fused 경로가 materialize를 피할 수 있지만 현재 이득에는 필요하지 않다.

### 3.3 drafter context는 가장 짧은 live committed prefix에서 직사각형으로 유지

**컨텍스트:**
target은 이제 ragged cache를 유지할 수 있지만, drafter의 입력 hidden 텐서와 자체 context는 여전히 하나의 직사각형 슬라이스로 만들어진다. drafter를 per-row로 만들려면 row별 drafter cache view와 row별 drafter forward가 필요하며, 이는 훨씬 큰 변경이다.

**고려한 대안:**

| 옵션 | 장점 | 단점 |
|---|---|---|
| 지금 drafter에 per-row context 제공 | walk 상승 | row별 drafter cache와 forward 배관이 필요하고 이 PR에서 측정되지 않음 |
| drafter에 가장 긴 committed prefix 제공 | 뒤처진 row에 더 많은 draft 재료 | verify block에서 잘라낸 hidden slice는 가장 짧은 accepted block 위치까지만 유효하므로 거부된 위치가 context에 섞임 |
| **선택: 가장 짧은 live committed prefix** | 캡처된 hidden slice에 대해 올바르고 기존 직사각형 계약을 재사용 | walk가 2.938에서 1.036으로 감소 |

**선택 이유:**
`drafter_ctx`는 완료되지 않은 row들의 `accepted[r] + 1` 최솟값으로 계산되고 hidden 텐서는 그 길이로 slice된다. 최솟값보다 많이 commit한 row도 target cache를 자신의 accept만큼 진행시키므로 방출 스트림은 정확하다. drafter는 다음 라운드에서 뒤처진 context로 다시 draft할 뿐이다. 이 선택은 변경을 target 쪽으로 제한하고 throughput 비교를 accept 정책에만 귀속시킨다.

**트레이드오프:**
walk가 65% 감소하므로 라운드 수 이득은 accept 수 이득이 시사하는 것보다 작다. 그래도 per-row accepts는 라운드당 방출 토큰을 결정하는 것이 walk가 아니라 committed 수이므로 분명한 이득이다. per-row drafter context는 이 PR의 결함이 아니라 후속 작업으로 기록된다.

### 3.4 attention KV를 압축하고 GDN 상태를 row별로 replay

**컨텍스트:**
ragged 라운드가 끝나면 공유 cache는 각 row의 accepted block을 공유 물리 offset에 두고 그 뒤에 거부된 위치를 담고 있다. 다음 라운드는 모든 row가 연속적인 `[0, valid_end)` prefix처럼 보여야 하며 물리 offset은 공유되어야 한다.

**선택 이유:**
`rollback_speculative_cache_rows`는 `o_post = max(valid_end_pre[r] + accepted[r] + 1)`을 계산한다. `compact_attention_rows`는 각 row의 `accepted[r] + 1` block을 `o_pre`에서 해당 row의 논리 end로 이동하고, 비워진 gap을 `o_post`까지 0으로 채우며, 호출자가 cache offset을 `o_post`로 trim한다. 겹치는 이동이 이미 덮어쓴 row를 읽지 않도록 source를 먼저 복사한 뒤 `slice_update`를 수행한다. GDN 상태는 row별로 replay된다. `replay_gdn_rows`는 각 row의 캡처된 `q/k/v/a/b`와 초기 상태를 `accepted[r] + 1`로 slice하고, chain-parity 커널을 해당 row의 길이로 실행한 뒤 per-row conv 및 recurrent 상태를 `[B, ...]` cache 텐서로 다시 쌓는다. 이로써 per-row 수치가 uniform 경로와 같은 커널을 유지한다.

**트레이드오프:**
압축은 라운드마다 attention 레이어마다 row마다 copy 한 번과 `slice_update` 한 번을 추가한다. 이는 보고된 throughput 안에서 측정되었다. uniform 경로도 fallback과 A/B 사용을 위해 그대로 유지되므로 코드는 두 arm을 함께 가진다. 이 중복은 귀속을 위해 의도된 것이다.

---

## 4. 구현 상세

### 4.1 아키텍처 변경

```
[Before, uniform clamp]
verify [B, bs] -> walk per row -> clamp_uniform_acceptance (min over live rows)
                -> rollback_partial_batched (trim bs - (a + 1) in every row)
                -> drafter hidden [B, max_a + 1]

[After, per-row accepts]
row_valid_ends = [prefill offset; B]
verify [B, bs]
  -> per-row RoPE at row_valid_ends[r]
  -> attend with build_divergent_row_mask (history, stale gap, causal block)
  -> commit accepted[r] + 1 per row
  -> compact_attention_rows: block moves to row_valid_ends[r], gap zeroed
  -> cache offset trimmed to max(new valid ends)
  -> replay_gdn_rows: per-row conv + recurrent state over accepted prefix
  -> row_valid_ends[r] += accepted[r] + 1
  -> drafter hidden [B, min_live(accepted) + 1] (rectangular, deliberately)
```

### 4.2 주요 코드 변경

**파일: `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`**

generator는 실행당 한 번 정책을 결정하고 prefill offset으로 논리 end를 초기화한다:

```rust
let per_row = self.per_row_accepts
    && target.supports_per_row_accepts()
    && target.row_layout_supported(caches);
let mut row_valid_ends: Vec<i32> = if per_row {
    let offset = T::batch_cache_offset(caches);
    vec![offset; batch_size]
} else {
    Vec::new()
};
```

verify 호출은 정책에 따라 분기하고, per-row arm에서는 clamp를 건너뛰며, rollback은 각 row의 accepted prefix를 target에 넘긴다:

```rust
if per_row {
    target.rollback_partial_batched_rows(
        caches, &verify_out, &accepted_i32, &row_valid_ends, bs as i32,
    );
    for r in 0..batch_size {
        row_valid_ends[r] += accepted_i32[r] + 1;
    }
} else {
    // uniform clamp arm: trim bs - (accepted + 1) in every row
}
```

drafter context는 가장 짧은 live row에서 직사각형으로 유지된다:

```rust
let drafter_ctx = if per_row {
    (0..batch_size)
        .filter(|&r| !finished[r])
        .map(|r| accepted_per_row[r] + 1)
        .min()
        .unwrap_or(1)
} else {
    max_accepted + 1
};
```

**변경 이유:** 라운드 루프가 정책 전환과 논리 위치 장부를 소유하고 target이 cache 레이아웃을 소유한다. 전환 지점을 한 곳에 두면 한 프로세스에서 A/B를 측정할 수 있다.

**파일: `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`**

`SpeculativeTarget`에 기본 구현이 있는 hook 네 개가 추가된다. `supports_per_row_accepts`, `row_layout_supported`, `batch_cache_offset`, `verify_forward_with_row_layout`이며, 여기에 `rollback_partial_batched_rows`가 더해진다. 기존 `rollback_partial_batched` 계약은 문서상 uniform-clamp arm으로 좁혀진다.

**변경 이유:** ragged row를 표현할 수 없는 target도 계속 컴파일되고 기존 의미를 유지한다.

**파일: `src/models/qwen3_5.rs`**

`Qwen35Model`은 capability gate, 공유 offset, ragged verify forward, ragged rollback을 구현한다. `forward_speculative_inner`는 `Option<&[i32]>`를 각 decoder 레이어로 전달한다. 비균일 rollback은 다음과 같다:

```rust
if !uniform {
    let o_post = ve_pre.iter().zip(accepted)
        .map(|(&ve, &a)| ve + a + 1).max().expect("non-empty accepted");
    for cache in caches.iter_mut() {
        if let Qwen3NextCache::Attention(kv) = cache {
            compact_attention_rows(kv, &ve_pre, accepted, o_pre, o_post);
            let roll_back = o_pre + block_size - o_post;
            if roll_back > 0 { kv.trim(roll_back); }
        }
    }
    self.replay_gdn_rows(caches, gdn_states, accepted, o_post);
    return accepted.iter().copied().max().unwrap_or(0);
}
```

**변경 이유:** attention cache가 다음 라운드 전에 row별로 유효한 연속 prefix가 되어야 하고, GDN 상태는 해당 row가 accept한 토큰만 반영해야 한다.

**파일: `src/models/qwen3_next.rs`**

`Qwen3NextAttention`에 `apply_rope_per_row`와 `attend_per_position_rows`가 추가된다. 어떤 row의 valid end라도 cache offset과 다르면 divergent 경로가 선택된다:

```rust
let divergent_rows: Option<&[i32]> = row_valid_ends.filter(|ends| {
    target_verify
        && l > 1
        && ends.len() == b as usize
        && ends.iter().any(|&end| end != offset)
});
```

각 query 위치는 B = 1 체인이 보았을 정확한 key 폭을 slice하며, divergent-row mask가 history, gap, block causality를 인코딩한다.

**변경 이유:** 공유 직사각형 attention 호출은 각 row에 다른 논리 시작점을 줄 수 없고, per-position 경로는 줄 수 있으며 Gemma 4 방식과 일치한다.

**파일: `src/lib/mlxcel-core/src/utils.rs`와 `src/models/gemma4.rs`**

`build_divergent_row_mask`가 `mlxcel-core::utils`로 이동하고, Gemma 4의 `build_divergent_verify_mask`는 한 줄 위임이 된다.

**변경 이유:** Gemma 4 MTP verifier와 Qwen 3.5 DFlash verifier가 하나의 테스트된 mask 구현을 공유한다.

**파일: `src/vision/qwen3_5_vl.rs`**

`Qwen35VLModel`이 네 개의 새 hook을 text model에 전달한다.

**변경 이유:** VLM 래퍼가 capability나 fallback을 조용히 잃지 않아야 한다.

**파일: `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/benchmark_results/dflash-per-row-accepts-m3ultra-2026-11-06.md`**

배칭 가이드와 환경 변수 표가 기본값, fallback, 측정값을 기록하고, benchmark 문서가 방법, 실행별 값, caveat, 정확성 gate를 기록한다.

### 4.3 데이터 모델 변경

없음. cache 레이아웃은 `[B, heads, capacity, dim]`으로 유지되고, 라운드마다 변하는 것은 valid-prefix 장부와 물리 offset뿐이다.

---

## 5. 학습 포인트

### 5.1 공유 직사각형 cache 안에서의 ragged acceptance

**개념:**
공유 배치 KV cache는 물리 offset 하나와 할당 하나를 가진다. ragged accept 수는 매 라운드 후 각 row의 live content를 연속 prefix로 압축하면 row별 할당을 요구하지 않는다. 불변식은 "각 row의 cache content는 연속적인 `[0, valid_end)` prefix이고, 공유 offset까지의 stale 영역은 마스킹된다"가 된다.

**이 PR에서의 적용:**
`compact_attention_rows`가 accepted block을 공유 offset에서 row의 논리 end로 이동하고 비워진 gap을 0으로 채운다. 다음 verify forward에서는 `build_divergent_row_mask`가 세 영역을 인코딩한다. 물리 offset은 새 valid end의 최댓값으로 trim되므로 할당이 raggedness에 따라 커지지 않는다.

**일반적인 사용 사례:**
- 시퀀스별 accept 수를 가진 배치 speculative decoding.
- 시퀀스가 서로 다른 길이에서 멈추고, 재개하고, rollback하는 continuous batching.
- row별 버퍼 없이 row별 논리 길이를 유지해야 하는 공유 버퍼 기법.

### 5.2 하나의 mask 레이아웃, 두 모델

**개념:**
"valid history, 차단된 gap, 논리 위치에서의 causal 새 block"을 표현하는 mask는 모델에 독립적이다. Gemma 4 구현을 core utility로 올려 두 번째 복사본이 어긋나기 전에 제거했다.

**이 PR에서의 적용:**
`build_divergent_row_mask`는 이제 Gemma 4의 `build_divergent_verify_mask`와 Qwen 3.5의 `attend_per_position_rows`가 호출한다. 이동과 함께 mask 전용 단위 테스트도 생겼다.

**일반적인 사용 사례:**
- 같은 cache 토폴로지를 가진 모델 패밀리 간 attention mask 재사용.
- 수치에 민감한 디바이스 코드를 하나의 테스트된 생성자 뒤로 통합.

### 5.3 speculative 배칭은 walk가 아니라 committed accept로 판단한다

**개념:**
scan ratio(walk)는 drafter가 제안한 양을 측정하고, committed accept 수는 target이 실제로 유지한 양을 측정한다. 배칭 변경이 accepted 토큰을 버리지 않게 되면 walk가 낮아져도 throughput은 올라갈 수 있다.

**이 PR에서의 적용:**
per-row arm의 walk는 drafter context가 여전히 가장 짧은 live prefix이기 때문에 2.938에서 1.036으로 떨어졌지만, committed accept가 0.323에서 1.036으로 오르고 라운드가 96에서 67로 줄어 aggregate throughput은 1.397x 상승했다. walk만 보고하면 회귀처럼 읽혔을 것이다.

**일반적인 사용 사례:**
- speculative decoding A/B 테스트.
- 생산자의 제안률과 소비자의 수용률이 반대 방향으로 움직일 수 있는 모든 파이프라인.

---

## 6. 추가 학습 리소스

### 핵심 키워드

| 키워드 | 설명 | 이 PR에서의 관련성 |
|---|---|---|
| DFlash | diffusion 스타일 drafter를 쓰는 block-draft speculative decoding | 여기서 변경된 배치 루프 |
| Uniform clamp | 가장 짧은 live row의 accept를 모든 row에 commit | 이전 정책이며 여전히 도달 가능 |
| Ragged accept | 한 라운드에서 row마다 다른 accept 수 | per-row accepts가 commit하는 것 |
| Valid end | row의 연속적인 유효 cache prefix 길이 | RoPE와 attention에 쓰이는 논리 위치 |
| Stale gap | 더 짧은 accept가 남긴 `[valid_end, physical_offset)` | attention에서 마스킹되고 압축에서 0으로 채워짐 |
| GDN | gated delta net linear-attention 레이어 | conv와 recurrent 상태를 row별로 replay |
| Walk | drafter가 제안하고 target이 verify한 토큰 | drafter context가 직사각형으로 유지되어 여기서는 감소 |

### 관련 기술/프레임워크

- **MLX**: array 프레임워크와 Metal 커널. `copy`와 `slice_update` 압축, per-position attention slice가 이 위에서 실행된다.
- **Gemma 4 divergent MTP verify**: 이 PR이 재사용하는 `DivergentVerifyRows` mask 레이아웃의 출처.

### 관련 PR/이슈

- 이슈 #2442: 상위 이슈. 이 PR이 항목 1을 구현하고 종료한다.
- 이슈 #2353: uniform clamp와 최초의 B = 4 대 B = 1 측정을 도입했다.
- PR #2853 / 이슈 #2851: 이 PR의 gate가 실행되는 배치 verify parity margin을 보정했다.
- 이슈 #2190: mask 레이아웃을 제공한 Gemma 4 per-row verify 아키텍처.
- 이슈 #2440: benchmark 문서가 참조하는 이전 per-row 범위 설정 작업.

---

## 7. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 15 |
| 추가 라인 | +1279 |
| 삭제 라인 | -156 |
| 테스트 추가 | 단위 테스트 3개, 무시되는 real-model throughput harness 1개 |
| 런타임 동작 변경 | 있음: cache가 압축을 지원하면 per-row accepts가 기본값 |

### 카테고리별 변경

| 카테고리 | 수 | 요약 |
|---|---|---|
| 성능/런타임 | 7 | 라운드 루프 정책 전환, target hook, Qwen 3.5 verify와 rollback, Qwen3Next per-row RoPE와 attention, VL 위임, 서버 로그 문구 |
| 공유 유틸리티 | 1 | `build_divergent_row_mask`를 `mlxcel-core::utils`로 이동, Gemma 4 위임 |
| 테스트 | 3 | 압축 단위 테스트, mask 단위 테스트, throughput 비교 harness |
| 문서 | 4 | 배칭 가이드, 환경 변수, benchmark 기록, parity harness 연결 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `ded78dce` | perf | perf(speculative): commit per-row accepts in batched DFlash |

---

## 8. 후속 조치

### 완료 필요

- [ ] per-row accepts를 켠 상태에서 실제 수집 burst 경로를 통한 서버 수준 동시 요청 smoke 테스트를 B > 1로 실행한다. 요청 완료, EOS, max-token row, 프롬프트 길이가 섞인 window를 포함해야 한다. 현재 근거는 길이가 같은 네 프롬프트에 대한 harness burst뿐이다.
- [ ] per-row drafter context를 별도 변경으로 구현하고 다시 측정한다. walk가 1.036 위로 회복될 것으로 예상되지만 이 PR의 예산에는 포함되지 않는다.

### 미검증 영역

- **서버 수준 동시 요청 smoke 실행 없음.** 측정과 parity 실행은 테스트 harness(`run_burst_with_policy`)를 사용하며 서버의 수집 burst 스케줄러를 사용하지 않는다. `src/server/batch/dflash_target.rs`의 row 수명, admission, 취소, 로깅은 기존 스위트로만 검증되고 이 측정으로는 검증되지 않는다.
- **per-row drafter context는 범위 밖.** drafter는 여전히 공유 직사각형 최소 prefix를 본다. 이것이 walk가 1.036으로 떨어진 명시된 이유이며, 측정된 1.397x가 drafter를 다룬 뒤 per-row target accepts가 낼 수 있는 하한이라는 뜻이다.
- **측정 caveat.** throughput 실행 중 다른 worktree의 Rust 빌드가 80개 코어 중 몇 개에서 컴파일하고 있었다. 두 arm은 같은 부하 아래 교대 실행되었고 clamp와 B = 1 셀이 이전 quiet-box #2442 기록(83.0 대 82.96~83.00, 78.09 대 77.70~77.83)을 재현하므로 비교가 부하로 치우치지는 않았다. 다만 절대값은 완전히 유휴한 박스에서 측정한 것이 아니다.
- **단일 호스트와 체크포인트.** M3 Ultra, macOS 26.6.2, `qwen3.5-4b-4bit`와 `qwen3.5-4b-dflash`, block 16, row당 128 토큰, temperature 0, 잘린 프롬프트 네 개다. 다른 하드웨어, quantization, block 크기, 프롬프트 분포는 측정되지 않았다.
- **fallback 경로는 여기서 재검증되지 않았다.** quantized, paged, head-trimmed cache는 변경되지 않은 uniform clamp를 탄다. 그 정확성은 상속된 것이며 per-row gate로 다시 테스트되지 않는다.
- **VLM 경로.** `Qwen35VLModel`이 hook을 위임하고 attention 레이어는 MRoPE position id가 있으면 divergent row를 거부한다. 이 PR에는 VLM DFlash parity 실행이 포함되지 않는다.

### 모니터링 필요

- per-row arm에서 `clamped_rounds`는 0을 유지해야 하고, clamp를 명시적으로 선택했거나 cache가 fallback한 경우에만 0이 아니어야 한다. dense Fp16 실행에서 0이 아니면 capability gate가 회귀한 것이다.
- 배포 후 aggregate decode throughput과 row-round당 committed accept를 관찰한다. 특히 raggedness가 가장 큰 혼합 프롬프트 길이에서 중요하다.
- MLX, quantization, 커널 디스패치가 바뀔 때마다 CI 하드웨어 레인에서 `greedy_parity_dflash_qwen35_batched_b4_matches_b1`와 `identical_prompts_dflash_qwen35_b4_matches_b1`를 실행한다.

### 향후 개선 사항

- 현재 clamp로 fallback하는 quantized 및 paged cache로 per-row accepts를 확장한다.
- mask 생성이 라운드 비용에서 측정 가능한 비중이 되면 fused mask-free attend 경로를 검토한다.
- per-row target accepts와 per-row drafter context를 함께 평가한다. 둘은 walk를 통해 상호작용하므로 쌍으로 측정해야 한다.

---

## 부록

### A. 테스트 결과

| 검사 | 결과 |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate --test speculative_parity qwen35_dflash_batched -- --ignored --test-threads=1` | per-row accepts를 기본값으로 2개 통과 |
| `throughput_dflash_qwen35_batched_clamp_vs_per_row` | 완료, arm당 세 번 교대 실행 |
| `cargo test --profile test-fast --lib models::qwen3_5_tests` | 통과, 30개 테스트(새 압축 테스트 2개 포함) |
| mlxcel-core 라운드 루프 단위 테스트 | 통과, `round_loop.rs` 15개, `round_loop_batched.rs` 13개 |
| `build_divergent_row_mask` 단위 테스트 | 통과 |
| `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` | 통과 |
| `cargo fmt --all -- --check` | 통과 |

### B. 성능 벤치마크

| 실행 | Clamp tok/s | Per-row tok/s |
|---:|---:|---:|
| 0 | 83.04 | 115.82 |
| 1 | 83.22 | 116.24 |
| 2 | 83.06 | 116.05 |
| 중앙값 | 83.06 | 116.05 |

accept와 라운드:

| Arm | 라운드 | clamp된 라운드 | row-round당 walk | row-round당 committed |
|---|---:|---:|---:|---:|
| Clamp | 96 | 93 | 2.938 | 0.323 |
| Per-row | 67 | 0 | 1.036 | 1.036 |

### C. 측정 환경

| 항목 | 값 |
|---|---|
| 호스트 | Apple M3 Ultra, 80코어, 512 GB 통합 메모리, macOS 26.6.2 |
| 백엔드 | Metal + Accelerate |
| Target | `models/qwen3.5-4b-4bit` |
| Drafter | `models/qwen3.5-4b-dflash` |
| Block 크기 | 16 |
| 생성 토큰 | row당 128 |
| 샘플링 | Temperature 0, greedy |
| 프롬프트 | 배치 DFlash parity harness의 네 프롬프트를 가장 짧은 토크나이제이션에 맞춰 절단 |
| 방법 | 폐기한 warm-up 2회 후 arm당 세 번 측정, clamp와 per-row를 교대 실행, `caffeinate -i`와 `scripts/with_indexers_paused.sh` 사용 |

### D. 참고 자료

- `src/lib/mlxcel-core/src/drafter/dflash/round_loop.rs`
- `src/lib/mlxcel-core/src/drafter/dflash/round_loop_batched.rs`
- `src/lib/mlxcel-core/src/utils.rs`
- `src/models/qwen3_5.rs`
- `src/models/qwen3_next.rs`
- `src/models/gemma4.rs`
- `src/vision/qwen3_5_vl.rs`
- `tests/speculative_parity/qwen35_dflash_harness.rs`
- `tests/speculative_parity/qwen35_dflash_throughput.rs`
- `docs/benchmark_results/dflash-per-row-accepts-m3ultra-2026-11-06.md`
- 이슈 #2442, 이슈 #2353, PR #2853, 이슈 #2851, 이슈 #2190, 이슈 #2440
